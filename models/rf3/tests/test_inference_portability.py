"""Inference must not silently inherit training-site caches or environment."""

import importlib

from atomworks.ml.transforms.cached_residue_data import LoadCachedResidueLevelData
from omegaconf import OmegaConf
from rf3.inference_engines.rf3 import RF3InferenceEngine


def test_inference_overrides_checkpoint_residue_cache(tmp_path):
    engine = RF3InferenceEngine(ckpt_path="unused.ckpt")
    cfg = OmegaConf.create(
        {
            "datasets": {
                "val": {
                    "probe": {
                        "dataset": {
                            "transform": {
                                "_target_": "rf3.data.pipelines.build_af3_transform_pipeline",
                                "is_inference": True,
                                "protein_msa_dirs": [],
                                "rna_msa_dirs": [],
                                "residue_cache_dir": str(
                                    tmp_path / "legacy-training-cache"
                                ),
                            }
                        }
                    }
                }
            }
        }
    )
    engine._construct_pipeline(cfg)
    loaders = [
        t
        for t in engine.pipeline.transforms
        if isinstance(t, LoadCachedResidueLevelData)
    ]
    assert len(loaders) == 1 and loaders[0].dir is None
    assert cfg.datasets.val.probe.dataset.transform.residue_cache_dir == str(
        tmp_path / "legacy-training-cache"
    )


def test_inference_dotenv_respects_explicit_runtime_environment(monkeypatch, tmp_path):
    import dotenv
    import rf3.inference as inference

    env = tmp_path / ".env"
    env.write_text(
        "LOCAL_MSA_DIRS=/legacy/msa\nPROTEIN_MSA_DIRS=/legacy/protein\nRF3_TEST_DEFAULT=loaded\n"
    )
    monkeypatch.setenv("LOCAL_MSA_DIRS", "/configured/msa")
    monkeypatch.setenv("PROTEIN_MSA_DIRS", "/configured/protein")
    monkeypatch.delenv("RF3_TEST_DEFAULT", raising=False)
    original = dotenv.load_dotenv

    def load(*args, **kwargs):
        return original(env, override=kwargs.get("override", False))

    monkeypatch.setattr(dotenv, "load_dotenv", load)
    importlib.reload(inference)
    import os

    assert os.environ["LOCAL_MSA_DIRS"] == "/configured/msa"
    assert os.environ["PROTEIN_MSA_DIRS"] == "/configured/protein"
    assert os.environ["RF3_TEST_DEFAULT"] == "loaded"
    monkeypatch.delenv("RF3_TEST_DEFAULT")


def test_engine_trajectory_ends_at_scored_prediction(monkeypatch, tmp_path):
    """Mock only model execution; run real engine output handling and CIF I/O."""
    from types import SimpleNamespace

    import numpy as np
    import torch
    from biotite.structure import Atom, BondList, array
    from biotite.structure.io.pdbx import CIFFile, get_structure
    from rf3.utils.inference import InferenceInput

    atoms = array(
        [
            Atom(
                [0, 0, 0],
                chain_id="A",
                res_id=1,
                res_name="GLY",
                atom_name=name,
                element=name[0],
            )
            for name in ("N", "CA", "C", "O")
        ]
    )
    atoms.bonds = BondList(4)
    atoms.set_annotation("token_id", np.zeros(4, dtype=int))
    rng = np.random.default_rng(8)
    predicted = torch.tensor(rng.normal(size=(2, 4, 3)), dtype=torch.float32)
    intermediates = [predicted * 3, predicted * 2]
    engine = RF3InferenceEngine(ckpt_path="unused.ckpt", seed=42)
    monkeypatch.setattr(engine, "initialize", lambda: None)
    engine.pipeline = lambda data: data
    engine.trainer = SimpleNamespace(
        fabric=SimpleNamespace(
            world_size=1,
            global_rank=0,
            to_device=lambda data: data,
            barrier=lambda: None,
        ),
        validation_step=lambda **kwargs: {
            "network_output": {
                "X_L": predicted,
                "X_denoised_L_traj": intermediates,
                "X_noisy_L_traj": intermediates,
            },
            "metrics_output": {},
        },
    )
    engine.run(
        InferenceInput(atoms, {}, "test"), out_dir=tmp_path, dump_trajectories=True
    )
    for sample in range(2):
        model = next((tmp_path / "test" / f"seed-42_sample-{sample}").glob("*.cif"))
        final = get_structure(CIFFile.read(model), model=1)
        for kind in ("denoised", "noisy"):
            trajectory = get_structure(
                CIFFile.read(tmp_path / "test" / f"{kind}_model_{sample}.cif")
            )
            assert trajectory.shape == (3, 4)
            np.testing.assert_array_equal(trajectory.coord[-1], final.coord)
