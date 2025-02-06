import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize
from icecream import ic
from pl_bolts.callbacks.verification.batch_gradient import default_input_mapping
from pl_bolts.utils import BatchGradientVerification
from torch.nn.functional import one_hot

from rf2aa.debug import pretty_describe_dict
from rf2aa.model.AF3_structure import AtomTransformer, DiffusionModule
from rf2aa.tensor_util import assert_cmp


def test_batch_leakage():
    conf_overrides = []
    with initialize(version_base=None, config_path="../config/train"):
        conf = compose(config_name="af3_repro", overrides=conf_overrides)

    c_s = conf.model.c_s
    c_z = conf.model.c_z

    model = DiffusionModule(
        c_atom=128, c_atompair=16, c_s=c_s, c_z=c_z, **conf.model.diffusion_module
    )

    verification = BatchGradientVerification(model)

    D = 3
    I = 80
    L = 160
    C_s_inputs = conf.model.diffusion_module.diffusion_conditioning.c_s_inputs
    C_s_trunk = conf.model.c_s

    inputs = dict(
        X_noisy_L=torch.rand((D, L, 3)),
        t=torch.rand((D,)),
        f={
            "asym_id": torch.zeros(I),
            "residue_index": torch.arange(I),
            "entity_id": torch.zeros(I),
            "token_index": torch.arange(I),
            "sym_id": torch.zeros(I).long(),
            "tok_idx": torch.arange(L) // 2,
            "ref_pos": torch.rand((L, 3)),
            "ref_charge": torch.rand((L,)),
            "ref_mask": torch.arange(L) // 2,
            "ref_element": one_hot(torch.randint(127, (L,)), 128),
            "ref_atom_name_chars": torch.zeros((L, 4, 64)),
            "ref_space_uid": torch.zeros((L,)),
        },
        S_inputs_I=torch.rand((I, C_s_inputs)),
        S_trunk_I=torch.rand((I, C_s_trunk)),
        Z_trunk_II=torch.rand((I, I, c_z)),
    )

    batched_inputs = default_input_mapping(inputs)
    assert len(batched_inputs) == 2 and batched_inputs[0].shape[0] == D, (
        "default input mapping (should contain X_noisy_L and t:\n"
        + pretty_describe_dict(batched_inputs)
    )
    print(f"{verification.NORM_LAYER_CLASSES=}")

    # Assert that there are no cross-batch gradients.
    # See: https://lightning-bolts.readthedocs.io/en/latest/callbacks/monitor.html
    # for details.
    valid = verification.check(input_array=inputs, sample_idx=0)

    # Assert that the model produces the same output when run batched/unbatched.
    out_batched = model(**inputs)
    inputs_unbatched = inputs
    inputs_unbatched["X_noisy_L"] = inputs["X_noisy_L"][:1]
    inputs_unbatched["t"] = inputs["t"][:1]
    out_single = model(**inputs_unbatched)

    assert_cmp(out_single, out_batched[0:1])
    ic(out_single.shape, out_batched.shape)

    # Assert that the batch outputs are different.
    assert torch.norm(out_batched[0] - out_batched[1]) > 1
    assert valid


def plot_attention_map(attn, diag=True):
    colors = ["indigo", "yellow"]
    if diag:
        attn[np.diag_indices_from(attn)] = 2
        colors = ["indigo", "yellow", "green"]
    cmap = mcolors.ListedColormap(colors)
    plt.matshow(attn, cmap=cmap)
    plt.axis("off")  # Turn off axis
    plt.show()


def test_sequence_local_atom_attention():
    conf_overrides = []
    with initialize(version_base=None, config_path="../config/train"):
        conf = compose(config_name="af3_repro", overrides=conf_overrides)
    conf = conf.model.diffusion_module.atom_attention_encoder.atom_transformer

    # Show the model's attenion map.
    show_full = False
    if show_full:
        # Get full size attention map.
        conf.l_max = 200
        atom_transformer = AtomTransformer(c_atom=10, c_atompair=11, **conf)
        Beta_lm = atom_transformer.Beta_lm
        attn = (Beta_lm == 0).long()
        plot_attention_map(attn)

    # Show af3 supplement-style attention map.
    show_supp = False
    if show_supp:
        atom_transformer = AtomTransformer(
            c_atom=10,
            c_atompair=11,
            l_max=200,
            n_queries=32,
            n_keys=64,
            diffusion_transformer=conf.diffusion_transformer,
        )
        Beta_lm = atom_transformer.Beta_lm
        attn = (Beta_lm == 0).long()
        plot_attention_map(attn)

    atom_transformer = AtomTransformer(
        c_atom=10,
        c_atompair=11,
        l_max=10,
        n_queries=2,
        n_keys=4,
        diffusion_transformer=conf.diffusion_transformer,
    )
    L = 6
    Beta_lm = atom_transformer.Beta_lm
    Beta_lm = Beta_lm[:L, :L]

    # Show small test-case attention map.
    show_test_case = False
    if show_test_case:
        plot_attention_map((Beta_lm == 0).long(), diag=False)

    o = 0
    x = -1e10
    want_Beta_lm = torch.tensor(
        [
            [o, o, o, x, x, x],
            [o, o, o, x, x, x],
            [x, o, o, o, o, x],
            [x, o, o, o, o, x],
            [x, x, x, o, o, o],
            [x, x, x, o, o, o],
        ]
    )

    assert_cmp(Beta_lm, want_Beta_lm)
