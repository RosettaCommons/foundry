

# TODO:
# from datahub.transforms import tip_atoms
tip_atoms = None



def motif_gp(get_mask):
    """
    Initialize can_be_gp: which atoms can be guidepost.
    """

    @wraps(get_mask)
    def out_get_mask(atom_array, *args, **kwargs):
        ret = get_mask(atom_array, *args, **kwargs)
        is_motif = ret["is_motif"]
        can_be_gp = copy.deepcopy(is_motif)

        return dict(can_be_gp=can_be_gp, **ret)

    return out_get_mask


def make_covale_compatible(get_mask):
    """
    Atoms that are involved in covalent modifications will be motif.
    """

    @wraps(get_mask)
    def out_get_mask(atom_array, *args, **kwargs):
        atom_mask = atom_array.occupancy > 0
        ret = get_mask(atom_array, atom_mask, *args, **kwargs)
        is_motif = ret.pop("is_motif")

        # atoms with covalent bonds should be motif, needs FlagAndReassignCovalentModifications transform prior to this
        atom_with_coval_bond = atom_array.covale  # (n_atoms, )
        is_motif[atom_with_coval_bond] = True
        return dict(is_motif=is_motif, **ret)

    return out_get_mask


def make_sm_compatible(get_mask):
    """
    Non-polymer atoms and atomized residues will all be taked as motif.
    Separately process non-atomized polymer atoms in other functions.
    """

    @wraps(get_mask)
    def out_get_mask(atom_array, *args, **kwargs):
        coords = torch.from_numpy(atom_array.coord)  # (n_atoms, )
        # In FlagAndReassignCovalentModificationsAllAtom, atoms in a residue that has covalent bonds with a sm will be atomized and considered as sm
        is_sm = ~atom_array.is_polymer | atom_array.atomize  # (n_atoms_atomized, )
        diffusion_mask = torch.ones(coords.shape[0]).bool()  # (n_atoms, )
        diffusion_mask_prot = get_mask(atom_array[~is_sm], *args, **kwargs).pop("is_motif")
        diffusion_mask[~is_sm] = diffusion_mask_prot.bool()
        return dict(is_motif=diffusion_mask)

    return out_get_mask


def _get_unconditional_diffusion_mask(atom_array, *args, **kwargs):
    """
    Atom_level unconditional generation of proteins, if a small molecule is present it will be given as context
    """
    is_motif = torch.zeros(atom_array.coord.shape[0]).bool()  # (n_protein_atoms, )
    return dict(is_motif=is_motif)


def _get_diffusion_mask_islands(
    atom_array,
    *args,
    island_len_min=5,
    island_len_max=30,
    n_islands_min=1,
    n_islands_max=4,
    p_island_can_be_gp=1,
    mask_upstream_atoms=False,
    seed=None,
    **kwargs,
):
    """
    Generate `motif` for atoms by choosing islands on the token level.
    All atoms within the motif token will be motif atoms.

    Args:
        atom_array: An object representing the atom information.
        *args: ignored
        island_len_min (int): Minimum length of islands to be motif. Default: 1.
        island_len_max (int): Maximum length of islands to be motif. Default: 15.
        n_islands_min (int): Minimum number of islands to be motif. Default: 1.
        n_islands_max (int): Maximum number of islands to be motif. Default: 4.
        p_island_can_be_gp (float): Probability whether the island can be guidepost.
        seed: a number used for np.random and random if it's not None.
        **kwargs: ignored

    Returns:
        dict: A dictionary containing:
            - 'is_motif' (torch.Tensor): Boolean tensor indicating motif atoms.
            - 'can_be_gp' (torch.Tensor): Boolean tensor indicating guidepost atoms.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    L = get_token_count(atom_array)
    token_starts = get_token_starts(atom_array)
    is_motif_residue = torch.zeros(L).bool()
    can_be_gp_residue = torch.zeros(L).bool()

    n_islands = np.random.randint(n_islands_min, n_islands_max)
    # print("N islands to be generated: ", n_islands)
    for _ in range(n_islands):
        mask_length = np.random.randint(island_len_min, island_len_max)
        mask_length = min(mask_length, L)
        high_start = L - mask_length
        if high_start <= 0: 
            continue
        start = random.randint(0, high_start)
        is_motif_residue[start : start + mask_length] = True
        if p_island_can_be_gp >= 1:
            can_be_gp_residue[start : start + mask_length] = True
        else:
            can_be_gp_residue[start : start + mask_length] = torch.rand(1) < p_island_can_be_gp

    # Prevents the entire thing from being motif, as this is disallowed.
    if is_motif_residue.all():
        is_motif_residue[np.random.randint(L)] = False

    is_motif = np.zeros(len(atom_array.res_id))
    for i in range(len(is_motif_residue)):
        # If true, set all the atoms in the motif as true
        if is_motif_residue[i]:
            if i == len(is_motif_residue) - 1:
                is_motif[token_starts[i] :] = 1
                break

            # Either
            if mask_upstream_atoms:
                is_motif[token_starts[i] : token_starts[i] + 4] = 1   # N, CA, C, O are motif
            else:
                is_motif[token_starts[i] : token_starts[i + 1]] = 1

    return dict(is_motif=torch.tensor(is_motif, dtype=bool))


def has_heavy_atoms_and_seq(atom_array, token_encoding):
    """
    Identify tokens that are either:
    amino acids with all heavy atoms appear and resolved;
    NAs with C1' resolved.
    """

    res_name = atom_array.res_name
    atom_name = atom_array.atom_name
    occupancy = atom_array.occupancy
    element = atom_array.atomic_number.astype(int)  # 6 for C etc.

    L = get_token_count(atom_array)
    token_starts = get_token_starts(atom_array)

    # only consider heavy atoms in amino acids, C1' in NAs
    has_heavy_token = []
    for i in range(L):
        cur_res_name = res_name[token_starts[i]]
        if i != L - 1:
            atom_names_cur_token = atom_name[token_starts[i] : token_starts[i + 1]]
            cur_occupancy = occupancy[token_starts[i] : token_starts[i + 1]]
            cur_element = element[token_starts[i] : token_starts[i + 1]]
        else:
            atom_names_cur_token = atom_name[token_starts[i] :]
            cur_occupancy = occupancy[token_starts[i] :]
            cur_element = element[token_starts[i] :]
        # for protein
        if cur_res_name in STANDARD_AA:
            true_heavy_atoms = token_encoding.token_atoms[cur_res_name][:NHEAVYPROT]
            true_heavy_atoms = [i.strip() for i in true_heavy_atoms if i != ""]
            # get heavy atom index
            if_contain_all_heavy = np.array(
                [heavy in atom_names_cur_token for heavy in true_heavy_atoms if heavy != ""]
            ).all()
            if not if_contain_all_heavy:
                has_heavy_token.append(0)
                continue
            heavy_atom_index = np.where(cur_element >= 6)[0]  # for standard aa, heavy atoms are only C, N, O, S(?)
            heavy_atom_occupancy = cur_occupancy[heavy_atom_index]

            if (heavy_atom_occupancy > 0.0).all():
                has_heavy_token.append(1)
            else:
                has_heavy_token.append(0)
        # for NAs
        elif cur_res_name in STANDARD_DNA or cur_res_name in STANDARD_RNA:
            if "C1'" in atom_names_cur_token:
                cur_atom_occupancy = cur_occupancy[np.where(atom_names_cur_token == "C1'")[0]]
                if cur_atom_occupancy > 0.0:
                    has_heavy_token.append(1)
                else:
                    has_heavy_token.append(0)
            else:
                has_heavy_token.append(0)
        else:
            has_heavy_token.append(0)

    return torch.tensor(has_heavy_token, dtype=bool)  # (num_tokens, )


def get_atom_names_within_n_bonds(cur_res_atom_array, token_encoding, source_node, n_bonds):
    """
    Get atom names within n_bonds which will constitute the motif.

    Return:
      atom_names: name of atoms that are within n_bonds.
    """
    bond_feats = tip_atoms.get_residue_bond_feats(cur_res_atom_array, token_encoding)
    bond_graph = nx.from_numpy_matrix(bond_feats.numpy())
    paths = nx.single_source_shortest_path_length(bond_graph, source=source_node, cutoff=n_bonds)
    atoms_within_n_bonds = paths.keys()
    cur_res_name = cur_res_atom_array.res_name[0]
    atom_names = [token_encoding.token_atoms[cur_res_name][i] for i in atoms_within_n_bonds]
    return atom_names


def _get_tip_mask(
    atom_array,
    *args,
    n_atomize_min=1,
    n_atomize_max=8,
    p_tip=0.8,
    bond_inclusion_p=0.5,
    unconditional=False,
    can_be_tip=None,
    token_encoding=None,
    seed=None,
    **kwargs,
):
    """
    Generate a tip atom `motif` for atoms.

    This function selects residues for atomization and creates a mask specifying which atoms
    should be included in the motif for the case of atomized motifs. Then covert it to the atom level.

    Args:
        atom_array: An object representing the atom information.
        *args: ignored
        n_atomize_min (int): Minimum number of residues to atomize. Default: 1.
        n_atomize_max (int): Maximum number of residues to atomize. Default: 8.
        p_tip (float): Probability of selecting the tip atom as the seed atom. Default: 0.8.
        bond_inclusion_p (float): Probability of including an additional bond in the motif. Default: 0.5.
            The `n_bonds` hop-distance around the seed atom that are included in the atom motif fragment
            is sampled from a geometric distribution with parameter `1-bond_inclusion_p`.
        unconditional (bool): If True, generate an unconditional mask (empty atom list for each residue). Default: False.
        can_be_tip (torch.Tensor, optional): list of residue index indicating which residue can be atomized, starting from 0.
        seed: a number used for np.random and random if it's not None.
        **kwargs: ignored

    Returns:
        dict: A dictionary containing:
            - 'is_motif' (torch.Tensor): Boolean tensor indicating motif tokens in the atom level.

    Notes:
        - If no valid residues are found for atomization, it falls back to unconditional generation.
        - The function selects between tip atom conditioning and general atom conditioning based on `p_tip`.
    """
    # assert not indep.is_sm.any()
    # old version is checking a residue:
    #   if all heavy atoms are marked as resolved
    #   if the token is either a protein residue (not including UNKNOWN) or a NA
    # If all satisfy, then is marked as is_valid_for_atomization
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    is_valid_for_atomization = has_heavy_atoms_and_seq(atom_array, token_encoding)  # (n_tokens, )
    num_token = get_token_count(atom_array)
    token_starts = get_token_starts(atom_array)

    if can_be_tip is not None:
        can_be_tip_index = np.zeros(num_token)
        can_be_tip_index[np.array(can_be_tip) - 1] = 1
        is_valid_for_atomization &= can_be_tip_index.astype(bool)
    if not is_valid_for_atomization.any():
        is_motif = torch.zeros(atom_array.coord.shape[0]).bool()  # (n_atoms, )
        is_motif[~atom_array.is_polymer] = True
        return dict(is_motif=is_motif, is_atom_motif=None)

    valid_idx = is_valid_for_atomization.nonzero()[:, 0]
    n_valid_targets = is_valid_for_atomization.sum()

    n_atomize = random.randint(n_atomize_min, min(n_atomize_max, n_valid_targets))
    atomize_i = np.random.choice(valid_idx, n_atomize, replace=False)
    is_atom_motif = {}  # dict of {res_idx: [atom_names]} where atom_names is list of atom names constituting the motif
    for i in atomize_i:
        # select the atom array for the current token
        if i == len(token_starts) - 1:
            cur_res_atom_array = atom_array[token_starts[i] :]
        else:
            cur_res_atom_array = atom_array[token_starts[i] : token_starts[i + 1]]

        if unconditional:
            atom_names = []
        else:
            if np.random.rand() < p_tip:
                # ... tip atom conditioning: choose seed atom as the furthest from
                #     the backbone oxygen
                seed_atom = tip_atoms.choose_furthest_from_oxygen(cur_res_atom_array, token_encoding)
            else:
                # ... general atom conditioning: choose random seed atom
                if i == num_token - 1:
                    cur_occupancy = atom_array.occupancy[token_starts[i] :]
                else:
                    cur_occupancy = atom_array.occupancy[token_starts[i] : token_starts[i + 1]]
                n_atoms = int(cur_occupancy.sum())
                seed_atom = np.random.choice(np.arange(n_atoms), 1)[0]

            # sample bonded fragment to show as motif from geom. distribution
            n_bonds = np.random.geometric(p=1 - bond_inclusion_p) - 1

            # get atom names within n_bonds which will constitute the motif
            atom_names = get_atom_names_within_n_bonds(cur_res_atom_array, token_encoding, seed_atom, n_bonds)
        assertpy.assert_that(atom_names).does_not_contain(None)
        is_atom_motif[i] = [name.strip() for name in atom_names]

    is_motif_all_atom = np.zeros(len(atom_array.res_id))
    for i in range(num_token):
        if i in is_atom_motif.keys():
            if i == num_token - 1:
                atom_name_cur_residue = atom_array.atom_name[token_starts[i] :]
            else:
                atom_name_cur_residue = atom_array.atom_name[token_starts[i] : token_starts[i + 1]]
            cur_indep_sele_atom = [j.strip() for j in is_atom_motif[i]]
            cur_residue_motif = np.array(
                [1 if atom_name.strip() in cur_indep_sele_atom else 0 for atom_name in atom_name_cur_residue]
            )
            if i == num_token - 1:
                is_motif_all_atom[token_starts[i] :] = cur_residue_motif
            else:
                is_motif_all_atom[token_starts[i] : token_starts[i + 1]] = cur_residue_motif
        else:
            if i == num_token - 1:
                is_motif_all_atom[token_starts[i] :] = 0
            else:
                is_motif_all_atom[token_starts[i] : token_starts[i + 1]] = 0

    # Hack this: for tip masking, don't set covalent-bonded atoms as motif
    atom_array.covale[atom_array.covale] = False
    return dict(is_motif=torch.tensor(is_motif_all_atom, dtype=bool))


def _get_mask_NA(
    atom_array,
    *args,
    **kwargs,
):
    """
    Generate "motif" for atoms that belong to nucleic acids.
    
    Args:
        atom_array (AtomArray): AtomArray object.
    
    Returns:
       dict: A dictionary containing:
            - 'is_motif' (torch.Tensor): Boolean tensor indicating motif atoms.
    """
    atom_is_standard_dna = np.isin(atom_array.res_name, STANDARD_DNA)
    atom_is_standard_rna = np.isin(atom_array.res_name, STANDARD_RNA)
    
    atom_is_standard_na = np.logical_or(
        atom_is_standard_dna,
        atom_is_standard_rna,
    )
    
    return {"is_motif": atom_is_standard_na}


def _get_debug_mask(
        atom_array,
        *args,
        **kwargs,
):

    is_motif = []
    for i in range(len(atom_array)):
        if atom_array.chain_id[i] == "A" and (int(atom_array.res_id[i]) in list(range(1,51)) or int(atom_array.res_id[i]) in list(range(61,101))):
            is_motif.append(True)
        else:
            is_motif.append(False)
    is_motif = np.array(is_motif)
    can_be_gp = np.array([True] * len(atom_array))

    return {"is_motif": is_motif, "can_be_gp": can_be_gp}

# TODO: Add tests for this mask!
def _get_ppi_mask(
        atom_array: AtomArray,
        *args,
        query_pn_unit_iids: list[str] = None,
        **kwargs,
) -> dict[str, np.ndarray]:
    """Get masks indicating what is motif and what is to be diffused for protein-protein interaction training.
    
    Args:
        atom_array (AtomArray): The atom array containing the current structural data.
        query_pn_unit_iids (list[str], optional): Enforce that one of these pn_unit_iids will be the diffused protein.
            If None, all proteins are considered for diffusion. Defaults to None.
    
    Returns:
        dict[str, np.ndarray]: A dictionary with two keys:
            - 'is_motif': A boolean array indicating which atoms are part of the motif.
            - 'can_be_gp': A boolean array indicating which atoms can be guideposts.
        
        # NOTE: In our repo, we're not yet treating is_motif and can_be_gp as distinct. If we do so in the future,
        # the value of 'can_be_gp' may need to be adjusted. For now, I will just set it to be the same as 'is_motif'.
    
    """

    # Determine which pn_units correspond to proteins
    protein_pn_unit_iids = []
    for pn_unit_iid in np.unique(atom_array.pn_unit_iid):
        pn_unit_atom_array = atom_array[atom_array.pn_unit_iid == pn_unit_iid]
        pn_unit_is_protein = np.unique(pn_unit_atom_array.is_protein)

        assert len(pn_unit_is_protein) == 1, "Each pn_unit should be either all protein or all non-protein."

        if pn_unit_is_protein[0]:
            protein_pn_unit_iids.append(pn_unit_iid)

    # Determine candidates to be masked (diffused) in training
    if query_pn_unit_iids is not None:
        candidate_masked_proteins = [iid for iid in query_pn_unit_iids if iid in protein_pn_unit_iids]
    else:
        candidate_masked_proteins = protein_pn_unit_iids
    
    if len(candidate_masked_proteins) == 0:
        raise ValueError(
            "No valid protein was found to diffuse: query_pn_unit_iids = "
            f"{query_pn_unit_iids if query_pn_unit_iids is not None else 'None (all proteins considered)'}"
        )
    
    protein_to_diffuse = random.choice(candidate_masked_proteins)

    # Create masks
    motif_mask_np = atom_array.pn_unit_iid == protein_to_diffuse
    is_motif = torch.tensor(motif_mask_np)
    can_be_gp = torch.tensor(motif_mask_np) # For now I will return this, altho it will be overwritten & re-cloned... sigh

    return {"is_motif": is_motif, "can_be_gp": can_be_gp}


def _get_complete_unconditional_mask(
    atom_array,
    *args,
    **kwargs,
):
    is_motif = np.array([False] * len(atom_array))
    can_be_gp = np.array([False] * len(atom_array))

    return {"is_motif": is_motif, "can_be_gp": can_be_gp}

get_unconditional_diffusion_mask = motif_gp(
    make_covale_compatible(make_sm_compatible(_get_unconditional_diffusion_mask))
)
get_diffusion_mask_islands = motif_gp(make_covale_compatible(make_sm_compatible(_get_diffusion_mask_islands)))
get_tip_mask = motif_gp(make_covale_compatible(_get_tip_mask))
get_na_mask = motif_gp(make_covale_compatible(_get_mask_NA))

get_debug_mask = _get_debug_mask

get_complete_unconditional_mask = _get_complete_unconditional_mask
get_ppi_mask = motif_gp(
    make_covale_compatible(make_sm_compatible(_get_ppi_mask))
)
