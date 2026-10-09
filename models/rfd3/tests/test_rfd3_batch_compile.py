import os
from unittest.mock import patch

import pytest
import torch
from rfd3.model.layers.batched import initialize
from rfd3_batch_fixtures import example, tiny_model
from test_rfd3_batch_model import prepared


@pytest.mark.parametrize("backend_name", ["vanilla", "sdpa"])
@pytest.mark.parametrize("low_memory", [False, True])
def test_complete_denoiser_fullgraph_reuses_graph_with_different_masks_and_times(
    low_memory,
    backend_name,
):
    torch._dynamo.reset()
    with patch.dict(os.environ, {"RFD3_LOW_MEMORY_MODE": str(int(low_memory))}):
        model = tiny_model()
    from test_rfd3_dense_sdpa import set_backend

    set_backend(model, backend_name)
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(
        model.diffusion_module.forward_batched,
        backend=backend,
        fullgraph=True,
        dynamic=False,
    )
    with torch.no_grad():
        for exs in (
            [example(), example((1, 2), seed=8)],
            [example((1, 2), seed=10), example((2, 2, 2, 1), seed=11)],
        ):
            f, x = prepared(exs, model)
            # Fixed A capacity as well as L/I is required for graph reuse.
            if f["atom_slots"].shape[-1] != 3:
                from rfd3.inference.batching import collate_examples, prepare_attention

                batch = collate_examples(
                    exs, atom_capacity=10, token_capacity=6, slot_capacity=3
                )
                f = prepare_attention(
                    batch["f"],
                    atom_keys=4,
                    atom_neighbors=1,
                    token_keys=32,
                    token_neighbors=8,
                )
                x = batch["coords"]
            init = initialize(model.token_initializer, f)
            t = torch.rand(2, 3) + 0.5
            expected = model.diffusion_module.forward_batched(x, t, f, **init)
            actual = compiled(x, t, f, **init)
            for k in expected:
                torch.testing.assert_close(actual[k], expected[k])
    assert len(graphs) == 1


@pytest.mark.gpu
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA Inductor validation requires CUDA"
)
def test_cuda_inductor_parity():
    model = tiny_model().cuda()
    f, x = prepared([example(), example((1, 2), seed=8)], model)
    f = {k: v.cuda() for k, v in f.items()}
    x = x.cuda()
    with torch.no_grad():
        init = initialize(model.token_initializer, f)
        t = torch.ones(2, 3, device="cuda")
        expected = model.diffusion_module.forward_batched(x, t, f, **init)
        compiled = torch.compile(
            model.diffusion_module.forward_batched, fullgraph=True, dynamic=False
        )
        actual = compiled(x, t, f, **init)
        for k in ("X_L", "sequence_logits_I"):
            torch.testing.assert_close(actual[k], expected[k], atol=0.025, rtol=0.025)


@pytest.mark.skipif(
    os.environ.get("RFD3_RUN_INDUCTOR_TESTS") != "1",
    reason="Set RFD3_RUN_INDUCTOR_TESTS=1 to run the CPU C++ compiler",
)
def test_cpu_inductor_numerical_parity():
    model = tiny_model()
    with torch.no_grad():
        f, x = prepared([example(), example((1, 2), seed=8)], model)
        init = initialize(model.token_initializer, f)
        t = torch.ones(2, 3)
        expected = model.diffusion_module.forward_batched(x, t, f, **init)
        compiled = torch.compile(
            model.diffusion_module.forward_batched, fullgraph=True, dynamic=False
        )
        actual = compiled(x, t, f, **init)
        for k in ("X_L", "sequence_logits_I"):
            torch.testing.assert_close(actual[k], expected[k], atol=2e-5, rtol=2e-5)


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RFD3_RUN_MPS_COMPILE_TESTS") != "1"
    or not torch.backends.mps.is_available(),
    reason="Set RFD3_RUN_MPS_COMPILE_TESTS=1 on a Metal host",
)
@pytest.mark.parametrize("backend_name", ["vanilla", "sdpa"])
def test_mps_inductor_numerical_parity(backend_name):
    from test_rfd3_dense_sdpa import set_backend

    model = tiny_model(bf16=False).to("mps")
    set_backend(model, backend_name)
    with torch.no_grad():
        f, x = prepared([example(), example((1, 2), seed=8)], model)
        f = {k: v.to("mps") for k, v in f.items()}
        x = x.to("mps")
        init = initialize(model.token_initializer, f)
        t = torch.ones(2, 3, device="mps")
        expected = model.diffusion_module.forward_batched(x, t, f, **init)
        compiled = torch.compile(
            model.diffusion_module.forward_batched, fullgraph=True, dynamic=False
        )
        actual = compiled(x, t, f, **init)
        for k in ("X_L", "sequence_logits_I"):
            torch.testing.assert_close(actual[k], expected[k], atol=1e-4, rtol=1e-4)


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RFD3_RUN_MPS_COMPILE_TESTS") != "1"
    or not torch.backends.mps.is_available(),
    reason="Set RFD3_RUN_MPS_COMPILE_TESTS=1 on a Metal host",
)
def test_mps_inductor_fusion_respects_metal_buffer_limit():
    from types import SimpleNamespace

    from test_rfd3_batch_engine import engine

    class ManyBuffers(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.randn(64, device="mps")) for _ in range(40)]
            )

        def forward_batched(self, x):
            # One fused kernel with all 40 independent parameter buffers exceeds
            # Metal's limit; the engine must split fusion without a graph break.
            return tuple((x + w).sin() for w in self.weights)

    module = ManyBuffers().eval()
    obj = engine(SimpleNamespace(diffusion_module=module))
    obj.compile_atom_buckets = (256,)
    obj.compile_token_buckets = (256,)
    obj.compile_slot_buckets = (14,)
    with torch.no_grad():
        x = torch.randn(2, 64, device="mps")
        expected = module.forward_batched(x)
        obj._compile_diffusion_submodules()
        actual = module.forward_batched(x)
        for got, wanted in zip(actual, expected):
            torch.testing.assert_close(got, wanted, atol=1e-5, rtol=1e-5)
