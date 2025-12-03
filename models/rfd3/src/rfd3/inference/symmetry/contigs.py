import numpy as np

from foundry.utils.components import fetch_mask_from_idx


def expand_contig_to_resid_from_string(contig_string):
    """
    Expand a contig string to a list of residue indices.
    Arguments:
        contig_string: string of the form "X1-5", e.g.
    Returns:
        list of residue indices
    """
    chain = contig_string[0]
    res_range = contig_string[1:].split("-")
    res_start = int(res_range[0])
    res_end = int(res_range[1])
    return [f"{chain}{i}" for i in range(res_start, res_end + 1)]


def expand_contig_unsym_motif(unsym_motif_names):
    """
    Expand a list of unsym motif names to a list of residue indices.
    Arguments:
        unsym_motif_names: list of unsym motif names
    Returns:
        list of residue indices
    """
    expanded_contigs = [
        expand_contig_to_resid_from_string(n) for n in unsym_motif_names if "-" in n
    ]
    # now remove any unexpanded contigs
    unsym_motif_names = [n for n in unsym_motif_names if "-" not in n]
    if len(expanded_contigs) != 0:
        for c in expanded_contigs:
            unsym_motif_names.extend(c)
    return unsym_motif_names


def get_unsym_motif_mask(atom_array, unsym_motif_names):
    """
    Get a mask of the unsym motif atoms.
    Arguments:
        atom_array: atom array
        unsym_motif_names: list of unsym motif names
    Returns:
        mask of the unsym motif atoms
    """

    is_unsym_motif = np.zeros(len(atom_array), dtype=bool)
    for n in unsym_motif_names:
        is_unsym_motif = np.logical_or(is_unsym_motif, atom_array.res_name == n)
        if (
            "src_component" in atom_array.get_annotation_categories()
            and n in atom_array.src_component
        ):
            is_unsym_motif = np.logical_or(
                is_unsym_motif, atom_array.src_component == n
            )
        elif n[0].isalpha() and n[1:].isdigit():
            residue_mask = fetch_mask_from_idx(n, atom_array=atom_array)
            is_unsym_motif = np.logical_or(is_unsym_motif, residue_mask)
    return is_unsym_motif
