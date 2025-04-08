import torch

from modelhub.model.AF3_blocks import MsaSubsampleEmbedder


def test_msa_module():
    pass


def test_msa_subsampler():
    B, N, L = 1, 100, 20
    params = {"num_sequences": 256, "msa_dim": 20, "msa_channels": 64, "S_dim": 32}
    msa_SI = torch.rand(B, N, L, 20)
    S_inputs = torch.rand(B, L, 32)
    subsampler = MsaSubsampleEmbedder(params)
    msa_SI = subsampler(msa_SI, S_inputs)
    assert msa_SI.shape == (B, N, L, 64)
    B, N, L = 1, 500, 20
    params = {"num_sequences": 256, "msa_dim": 20, "msa_channels": 64, "S_dim": 32}
    msa_SI = torch.rand(B, N, L, 20)
    S_inputs = torch.rand(B, L, 32)
    subsampler = MsaSubsampleEmbedder(params)
    msa_SI = subsampler(msa_SI, S_inputs)
    assert msa_SI.shape == (B, 256, L, 64)


def test_msa_pair_weighted_average():
    pass


def test_msa_weighting_einsum():
    B, I, S, H, c = 1, 5, 10, 8, 4

    gate_SIH = torch.randn(B, S, I, H, c)
    w_IIH = torch.randn(B, I, I, H)
    v_SIH = torch.randn(B, S, I, H, c)

    # Initialize the result tensor
    C = torch.zeros((B, S, I, H, c))

    # Perform the einsum contraction in smaller steps
    # for idx_b in range(B):
    # for idx_s in range(S):
    # for idx_i in range(I):
    # for idx_h in range(H):
    # for idx_c in range(c):
    # C[idx_b, idx_s, idx_i, idx_h, idx_c] = torch.sum(
    # v_SIH[idx_b, idx_s, :, idx_h, idx_c] * w_IIH[idx_b, :, idx_i, idx_h]
    # )
    unaggregated_weights = torch.einsum("bsihc, bijh -> bsijhc", v_SIH, w_IIH)

    weights = torch.einsum("bsihc, biih -> bsihc", v_SIH, w_IIH)
    o_SIH = gate_SIH * weights


def test_batching_pairformer():
    from modelhub.model.AF3_structure import PairformerBlock, PairformerBlock_batched

    B, I, cs, cz = 1, 100, 128, 128
    S_I = torch.randn(I, cs)
    Z_II = torch.randn(I, I, cz)
    from modelhub.tests.test_conditions import make_deterministic, seed_all

    seed_all()
    pairformer = PairformerBlock(
        c_s=cs, c_z=cz, p_drop=0.25, c=8, attention_pair_bias={"n_head": 16}
    )
    seed_all()
    pairformer_batched = PairformerBlock_batched(
        c_s=cs, c_z=cz, p_drop=0.25, c=8, attention_pair_bias={"n_head": 16}
    )
    with torch.enable_grad():
        make_deterministic()
        S_I_unbatched, Z_II_unbatched = pairformer(S_I, Z_II)
    with torch.enable_grad():
        make_deterministic()
        S_I_batched, Z_II_batched = pairformer_batched(S_I, Z_II)

    assert torch.allclose(S_I_unbatched, S_I_batched)
    assert torch.allclose(Z_II_unbatched, Z_II_batched)

    loss_unbatched = S_I_unbatched.sum()
    loss_batched = S_I_batched.sum()
    assert torch.allclose(loss_unbatched, loss_batched)

    loss_batched.backward()
    loss_unbatched.backward()

    pairformer_unbatched_dict = {}
    for name, param in pairformer.named_parameters():
        pairformer_unbatched_dict[name] = param.detach().clone()

    pairformer_batched_dict = {}
    for name, param in pairformer_batched.named_parameters():
        pairformer_batched_dict[name] = param.detach().clone()

    for name in pairformer_unbatched_dict.keys():
        print(pairformer_unbatched_dict[name].grad)
        assert torch.allclose(
            pairformer_unbatched_dict[name], pairformer_batched_dict[name]
        )
