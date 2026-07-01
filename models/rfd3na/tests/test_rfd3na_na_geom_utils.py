"""Unit tests for ``rfd3na.transforms.na_geom_utils.parse_dot_bracket``.

The parser turns a nucleic-acid dot-bracket string into 0-based base pairs and unpaired
positions. Its contract has a few non-obvious edges worth pinning:

- Each bracket family (``()``, ``[]``, ``{}``, ``<>``, and pseudoknot ``A``–``E`` / ``a``–``e``)
  has its own stack, so different families can *cross* (e.g. ``([)]``) — that is how
  pseudoknots are expressed.
- Nesting pops innermost-first, so pair indices come out inner-before-outer.
- ``.`` is an unpaired position; an unmatched *closer* and any unrecognised character are
  skipped silently (not recorded as pairs or unpaired); an unmatched *opener* leaves nothing.
"""

from rfd3na.transforms.na_geom_utils import parse_dot_bracket


def test_all_dots_are_unpaired():
    assert parse_dot_bracket("...") == ([], [0, 1, 2])


def test_single_pair():
    assert parse_dot_bracket("()") == ([(0, 1)], [])


def test_nested_pairs_pop_inner_first():
    assert parse_dot_bracket("(())") == ([(1, 2), (0, 3)], [])


def test_pair_with_unpaired_between():
    assert parse_dot_bracket("(.)") == ([(0, 2)], [1])


def test_square_brackets_are_their_own_family():
    assert parse_dot_bracket("[[]]") == ([(1, 2), (0, 3)], [])


def test_crossing_families_form_pseudoknot():
    # "(" and "[" open on separate stacks, so ")" closes "(" and "]" closes "[" — a crossing.
    assert parse_dot_bracket("([)]") == ([(0, 2), (1, 3)], [])


def test_letter_pseudoknot_brackets():
    assert parse_dot_bracket("AaAa") == ([(0, 1), (2, 3)], [])


def test_unmatched_closer_is_ignored():
    assert parse_dot_bracket(")") == ([], [])


def test_unmatched_opener_yields_nothing():
    # An opener with no closer is neither a pair nor an unpaired position.
    assert parse_dot_bracket("(") == ([], [])


def test_unknown_characters_are_skipped():
    # Non-bracket, non-dot characters are ignored entirely.
    assert parse_dot_bracket("xNy") == ([], [])


def test_mixed_string_pairs_and_unpaired():
    assert parse_dot_bracket("((..))..") == ([(1, 4), (0, 5)], [2, 3, 6, 7])


def test_empty_string():
    assert parse_dot_bracket("") == ([], [])
