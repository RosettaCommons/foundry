"""Inference defaults must preserve caller configuration and serializable atoms."""

import importlib
import os

import numpy as np
import pytest
from atomworks.io.utils.io_utils import to_cif_file
from biotite.structure import AtomArray, BondList, concatenate
from biotite.structure.io import pdbx


@pytest.mark.parametrize("model", ["rfd3", "rfd3na"])
def test_generated_atoms_preserve_integer_motif_charges_in_cif(model, tmp_path):
    prepare = importlib.import_module(f"{model}.utils.inference").set_common_annotations
    generated = AtomArray(1)
    generated.bonds = BondList(1)
    generated.chain_id[:] = "A"
    generated.res_id[:] = 1
    generated.res_name[:] = "GLY"
    generated.atom_name[:] = "CA"
    generated.element[:] = "C"
    generated = prepare(generated)

    motif = concatenate([generated.copy(), generated.copy()])
    motif.res_id[:] = [2, 3]
    motif.set_annotation("charge", np.array([-1, 1], dtype=np.int8))
    motif = prepare(motif)
    atoms = concatenate([generated, motif])
    assert np.issubdtype(atoms.charge.dtype, np.integer)
    np.testing.assert_array_equal(atoms.charge, [0, -1, 1])

    path = tmp_path / "charged.cif"
    to_cif_file(atoms, path)
    restored = pdbx.get_structure(
        pdbx.CIFFile.read(path), model=1, extra_fields=["charge"]
    )
    np.testing.assert_array_equal(restored.charge, [0, -1, 1])


@pytest.mark.parametrize(
    "name",
    ["rfd3.run_inference", "rfd3na.run_inference", "foundry_cli.download_checkpoints"],
)
def test_cli_dotenv_preserves_explicit_checkpoint_configuration(
    name, monkeypatch, tmp_path
):
    import dotenv

    module = importlib.import_module(name)
    env = tmp_path / ".env"
    env.write_text("FOUNDRY_CHECKPOINT_DIRS=\nFOUNDRY_TEST_DEFAULT=loaded\n")
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    monkeypatch.setenv("FOUNDRY_CHECKPOINT_DIRS", "/configured/checkpoints")
    monkeypatch.delenv("FOUNDRY_TEST_DEFAULT", raising=False)
    original = dotenv.load_dotenv

    def load(*args, **kwargs):
        return original(env, override=kwargs.get("override", False))

    monkeypatch.setattr(dotenv, "load_dotenv", load)
    importlib.reload(module)
    assert os.environ["FOUNDRY_CHECKPOINT_DIRS"] == "/configured/checkpoints"
    assert os.environ["FOUNDRY_TEST_DEFAULT"] == "loaded"
    monkeypatch.delenv("FOUNDRY_TEST_DEFAULT")
