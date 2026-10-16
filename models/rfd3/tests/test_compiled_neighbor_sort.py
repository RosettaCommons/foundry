"""Native CUDA sort remains opaque to Inductor, with exact integer semantics."""

import pytest
import torch
from rfd3.model.layers.batched import _sort_neighbor_values, select_neighbors


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_neighbor_sort_schema_and_strides():
    indices = torch.randint(0, 17, (2, 3, 19, 2), device="cuda")[..., 0]
    torch.testing.assert_close(_sort_neighbor_values(indices), indices.sort(-1).values)
    torch.library.opcheck(
        _sort_neighbor_values, (indices,), test_utils=("test_schema", "test_faketensor")
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compiled_neighbors_preserve_padding_and_duplicates():
    x = torch.randn(1, 2, 17, 3, device="cuda")
    valid = torch.arange(17, device="cuda")[None] < 11
    sequence = torch.eye(17, device="cuda", dtype=torch.bool)[None]
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(select_neighbors, fullgraph=True, backend=backend)
    for shift in (0, 0.25):
        expected = select_neighbors(x + shift, valid, sequence, 8)
        actual = compiled(x + shift, valid, sequence, 8)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert len(graphs) == 1
    assert (
        sum(
            n.target == torch.ops.rfd3.sort_neighbor_values.default
            for n in graphs[0].graph.nodes
        )
        == 2
    )
