"""Small, checkpoint/CCD-free examples shared by batching regression tests."""

from pathlib import Path

import torch
from omegaconf import OmegaConf


def example(counts=(2, 1, 3), d=3, seed=7):
    g = torch.Generator().manual_seed(seed)
    i, l = len(counts), sum(counts)
    tok = torch.repeat_interleave(torch.arange(i), torch.tensor(counts))
    rep = torch.zeros(l, dtype=torch.bool)
    rep[torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()[:-1]])] = True
    f = {
        "atom_to_token_map": tok,
        "is_ca": rep,
        "ref_space_uid": tok.clone(),
        "ref_pos": torch.randn(l, 3, generator=g),
        "motif_pos": torch.randn(l, 3, generator=g),
        "ref_atom_name_chars": torch.randn(l, 4, 64, generator=g),
        "ref_element": torch.randn(l, 128, generator=g),
        "restype": torch.randn(i, 32, generator=g),
        "ref_motif_token_type": torch.randn(i, 3, generator=g),
        "ref_plddt": torch.rand(i, generator=g),
        "is_non_loopy": torch.zeros(i, dtype=torch.bool),
        "token_bonds": torch.eye(i),
        "unindexing_pair_mask": torch.zeros(i, i, dtype=torch.bool),
        "asym_id": torch.zeros(i, dtype=torch.long),
        "entity_id": torch.zeros(i, dtype=torch.long),
        "sym_id": torch.zeros(i, dtype=torch.long),
        "residue_index": torch.arange(i),
        "token_index": torch.arange(i),
        "is_motif_token_unindexed": torch.zeros(i, dtype=torch.bool),
        "is_motif_token_with_fully_fixed_coord": torch.zeros(i, dtype=torch.bool),
        "ref_atomwise_rasa": torch.randn(l, 3, generator=g),
        "ref_charge": torch.zeros(l),
    }
    for k in (
        "ref_mask",
        "ref_is_motif_atom_with_fixed_coord",
        "ref_is_motif_atom_unindexed",
        "has_zero_occupancy",
        "active_donor",
        "active_acceptor",
        "is_atom_level_hotspot",
        "is_motif_atom_unindexed",
        "is_motif_atom_with_fixed_coord",
        "is_motif_atom_with_fixed_seq",
        "is_virtual",
    ):
        f[k] = torch.zeros(l, dtype=torch.bool)
    return dict(
        feats=f,
        coord_atom_lvl_to_be_noised=torch.randn(d, l, 3, generator=g),
        example_id=f"example_{seed}",
        t=torch.ones(d),
        noise=torch.zeros(d, l, 3),
    )


def tiny_model(bf16=False):
    from rfd3.model.RFD3 import RFD3

    cfg = OmegaConf.load(
        Path(__file__).parents[1] / "configs/model/components/rfd3_net.yaml"
    )
    # Resolve the original config's absolute interpolations before shrinking it.
    cfg = OmegaConf.create({"model": {"net": cfg}})
    net = OmegaConf.to_container(cfg, resolve=True)["model"]["net"]
    net.pop("_target_")
    net.update(c_s=16, c_z=8, c_atom=16, c_atompair=4)
    init, dm = net["token_initializer"], net["diffusion_module"]
    init["n_pairformer_blocks"] = 1
    init["pairformer_block"]["attention_pair_bias"]["n_head"] = 2
    dm.update(c_token=24, c_t_embed=8, n_attn_keys=4, n_attn_seq_neighbours=1)
    dm["diffusion_token_encoder"]["n_pairformer_blocks"] = 1
    dm["diffusion_token_encoder"]["pairformer_block"]["attention_pair_bias"][
        "n_head"
    ] = 2
    dm["diffusion_transformer"]["n_block"] = 1
    dm["diffusion_transformer"]["diffusion_transformer_block"]["n_head"] = 2
    dm["atom_attention_encoder"]["n_blocks"] = 1
    dm["atom_attention_decoder"]["n_blocks"] = 1
    for x in (
        init["downcast"],
        dm["downcast"],
        dm["atom_attention_decoder"]["downcast"],
        dm["atom_attention_decoder"]["upcast"],
    ):
        x["cross_attention_block"].update(c_model=8, n_head=2)
    net["inference_sampler"] = dict(
        use_classifier_free_guidance=False,
        cfg_scale=1.0,
        num_timesteps=4,
        allow_realignment=False,
    )
    torch.manual_seed(123)
    model = RFD3(**net).eval()
    # CPU references use fp32 throughout; the production pairformer enables bf16
    # autocast, which is tested separately rather than hidden in primitive tolerances.
    for module in model.modules():
        if hasattr(module, "force_bfloat16"):
            module.force_bfloat16 = bf16
    return model
