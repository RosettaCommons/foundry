"""A diffusion-batch dimension can vary without changing token/atom buckets."""

import pytest
import torch
from rfd3.inference.runtime import dynamic_diffusion_batch
from rfd3.model.layers.batched import initialize
from rfd3_batch_fixtures import example, tiny_model
from test_rfd3_batch_model import prepared


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kernel_backend", ["torch", "triton"])
def test_dynamic_batch_denoiser_reuses_graph_and_preserves_outputs(kernel_backend):
    torch.set_num_threads(4)
    model = tiny_model().cuda()
    for module in model.modules():
        module.inference_kernel_backend = kernel_backend
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return torch._inductor.compile(graph, inputs)

    original = model.diffusion_module.forward_batched
    compiled = dynamic_diffusion_batch(
        torch.compile(original, backend=backend, fullgraph=True, dynamic=False)
    )
    with torch.no_grad():
        f, x = prepared([example(d=1)], model)
        f = {key: value.cuda() for key, value in f.items()}
        init = initialize(model.token_initializer, f)
        for d in (2, 4):
            coordinates = x.cuda().expand(1, d, -1, -1).contiguous()
            time = torch.tensor(1.5, device="cuda").expand(1, d)
            expected = original(coordinates, time, f, **init)
            actual = compiled(coordinates, time, f, **init)
            for key in ("X_L", "sequence_logits_I"):
                torch.testing.assert_close(
                    actual[key], expected[key], atol=3e-5, rtol=3e-5
                )
    assert len(graphs) == 1
