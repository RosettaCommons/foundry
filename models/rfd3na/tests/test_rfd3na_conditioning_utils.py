"""Unit tests for the pure helpers in ``rfd3na.transforms.conditioning_utils``.

These build training-time motif masks. The graph traversal helpers consume a bond graph
shaped like the one atomworks' ``_atom_array_to_networkx_graph`` produces: integer nodes,
each carrying a ``node_data`` attribute holding the atom name. The fixtures below mirror
that shape directly rather than round-tripping through atomworks.

- ``random_condition(p)`` — ``p == 0`` -> ``False``, ``p == 1`` -> ``True`` (``rand() < 1``
  always holds), and ``p`` outside ``[0, 1]`` asserts.
- ``get_node_idx_from_atom_name`` — returns the unique node whose ``node_data`` matches;
  raises ``ValueError`` when no node or more than one node matches.
- ``get_atom_names_within_n_bonds`` — atom names reachable within ``n_bonds`` graph edges
  of the seed (inclusive of the seed; ``n_bonds == 0`` is the seed alone).
- ``choose_furthest_from_oxygen`` — the atom name graph-furthest from the backbone ``"O"``.
- ``choose_uniformly_random_atom_name`` — a random atom name among ``occupancy > 0`` atoms,
  falling back to all atoms when none have positive occupancy.
- ``sample_island_tokens`` — a boolean mask of random contiguous islands, honouring the
  optional ``max_length`` cap on the total number of ``True`` values.

(``sample_subgraph_atoms`` is not covered here — it builds its graph via atomworks'
``_atom_array_to_networkx_graph``, which needs a bonded ``AtomArray``.)
"""

import networkx as nx
import numpy as np
import pytest
from biotite.structure import AtomArray
from rfd3na.transforms.conditioning_utils import (
    choose_furthest_from_oxygen,
    choose_uniformly_random_atom_name,
    get_atom_names_within_n_bonds,
    get_node_idx_from_atom_name,
    random_condition,
    sample_island_tokens,
)


def _linear_backbone_graph() -> nx.Graph:
    """Chain O(0)-C(1)-CA(2)-N(3)-X(4); graph distance from O is unique per node."""
    g = nx.Graph()
    for i, name in enumerate(["O", "C", "CA", "N", "X"]):
        g.add_node(i, node_data=name)
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    return g


# --- random_condition -------------------------------------------------------


def test_random_condition_zero_is_always_false():
    assert random_condition(0) is False


def test_random_condition_one_is_always_true():
    # rand() draws from [0, 1), so rand() < 1 holds for every draw. (The result is a
    # comparison, not a literal, so assert truthiness rather than identity to `True`.)
    assert random_condition(1)


def test_random_condition_rejects_out_of_range():
    with pytest.raises(AssertionError):
        random_condition(1.5)
    with pytest.raises(AssertionError):
        random_condition(-0.1)


# --- get_node_idx_from_atom_name --------------------------------------------


def test_get_node_idx_returns_matching_node():
    g = _linear_backbone_graph()
    assert get_node_idx_from_atom_name(g, "CA") == 2


def test_get_node_idx_missing_raises():
    g = _linear_backbone_graph()
    with pytest.raises(ValueError, match="No node"):
        get_node_idx_from_atom_name(g, "ZZ")


def test_get_node_idx_duplicate_raises():
    g = nx.Graph()
    g.add_node(0, node_data="CA")
    g.add_node(1, node_data="CA")
    with pytest.raises(ValueError, match="Multiple nodes"):
        get_node_idx_from_atom_name(g, "CA")


# --- get_atom_names_within_n_bonds ------------------------------------------


def test_within_one_bond_includes_seed_and_neighbours():
    g = _linear_backbone_graph()
    # CA(2) neighbours C(1) and N(3), plus itself.
    assert sorted(get_atom_names_within_n_bonds(g, "CA", 1)) == ["C", "CA", "N"]


def test_within_zero_bonds_is_seed_only():
    g = _linear_backbone_graph()
    assert get_atom_names_within_n_bonds(g, "CA", 0) == ["CA"]


# --- choose_furthest_from_oxygen --------------------------------------------


def test_choose_furthest_from_oxygen_picks_graph_furthest():
    g = _linear_backbone_graph()
    # Distances from O: C=1, CA=2, N=3, X=4 -> X is the unique furthest.
    assert choose_furthest_from_oxygen(g) == "X"


# --- choose_uniformly_random_atom_name --------------------------------------


def _occupancy_array(names: list[str], occupancy: list[float]) -> AtomArray:
    arr = AtomArray(len(names))
    arr.set_annotation("atom_name", np.array(names))
    arr.set_annotation("occupancy", np.array(occupancy, dtype=float))
    return arr


def test_choose_atom_name_only_from_positive_occupancy():
    arr = _occupancy_array(["A", "B", "C"], [1.0, 0.0, 1.0])
    np.random.seed(0)
    # B has zero occupancy and must never be chosen.
    assert {choose_uniformly_random_atom_name(arr) for _ in range(20)} <= {"A", "C"}


def test_choose_atom_name_falls_back_to_all_when_no_occupancy():
    arr = _occupancy_array(["P", "Q"], [0.0, 0.0])
    np.random.seed(0)
    assert choose_uniformly_random_atom_name(arr) in {"P", "Q"}


# --- sample_island_tokens ---------------------------------------------------


def test_sample_islands_returns_bool_mask_of_requested_length():
    np.random.seed(0)
    mask = sample_island_tokens(50)
    assert mask.shape == (50,)
    assert mask.dtype == bool


def test_sample_islands_respects_max_length_budget():
    # The trimming logic must never let the total True count exceed max_length.
    for seed in range(20):
        np.random.seed(seed)
        mask = sample_island_tokens(
            60, island_len_min=3, island_len_max=8, max_length=15
        )
        assert mask.sum() <= 15


def test_sample_islands_without_cap_stays_within_array():
    np.random.seed(0)
    mask = sample_island_tokens(40)
    assert 0 < mask.sum() <= 40
