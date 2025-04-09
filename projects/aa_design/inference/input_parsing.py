import logging
from os import PathLike

import biotite.structure as struc
import numpy as np
from biotite.structure import Atom
from cifutils import parse
from cifutils.tools.inference import components_to_atom_array
from cifutils.utils.io_utils import to_cif_file
from datahub.datasets.parsers.base import (
    DEFAULT_CIF_PARSER_ARGS,
)
from datahub.utils.token import (
    get_token_starts,
)

from modelhub.common import exists
from modelhub.utils.ddp import RankedLogger
from projects.aa_design.inference.contigs import (
    get_design_pattern_with_constraints,
    split_contig,
)
from projects.aa_design.transforms.masks import Mask

logging.basicConfig(level=logging.INFO)
ranked_logger = RankedLogger(__name__, rank_zero_only=True)


def set_indices(array, chain, res_id_start, molecule_id):
    array.chain_id = np.full(array.shape[0], chain, dtype=array.chain_id.dtype)
    array.res_id = np.full(array.shape[0], res_id_start+array.res_id-1, dtype=array.res_id.dtype)
    # array.set_annotation('molecule_id', np.full(array.shape[0], molecule_id))
    return array

def idealized_cb_position(N, CA, C):
    '''*args: (3,)'''
    # recreate Cb given N,Ca,C
    b = CA - N
    c = C - CA
    a = np.cross(b, c, axis=-1)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + CA
    return Cb

def create_spoofed_cif_file_for_backbone(num_residues, spoofed_cif_path, ori_token=[0,0,0]):
    '''Legacy version for atom14 inference'''
    inputs = [
        {    
            "seq": "W" * num_residues,
            "chain_type": "polypeptide(l)",
            "is_polymer": True,
            "chain_id": "A",
        }
    ]
    atom_array = components_to_atom_array(inputs)
    atom_array.coord = np.nan_to_num(atom_array.coord)
    to_cif_file(atom_array, spoofed_cif_path, extra_metadata={'ori_token': str(ori_token)})
    ranked_logger.info(f"Created spoofed CIF file with {num_residues} residues at {spoofed_cif_path}")

def create_cb_atoms(array):
    # array of length 4 with N, CA, C, O
    # Returns array with CB placed ideally 
    if array.atom_name.tolist() != ['N', 'CA', 'C', 'O']:
        raise ValueError("Input array must contain exactly 4 atoms: N, CA, C, O. Got : {}".format(array.atom_names.tolist()))
    cb_atoms = array[array.atom_name == 'CA'].copy()
    cb_atoms.atom_name = np.array(['CB'], dtype=cb_atoms.atom_name.dtype)
    cb_pos = idealized_cb_position(
        array.coord[array.atom_name == 'N'].squeeze(),
        array.coord[array.atom_name == 'CA'].squeeze(),
        array.coord[array.atom_name == 'C'].squeeze()
    )
    cb_atoms.coord = cb_pos[None]
    return cb_atoms

def create_o_atoms(array):
    if array.atom_name.tolist() != ['N', 'CA', 'C']:
        raise ValueError("Input array must contain exactly 4 atoms: N, CA, C, O. Got : {}".format(array.atom_names.tolist()))

    ca_atoms = array[array.atom_name == 'CA'].copy()
    ca_atoms.atom_name = np.array(['O'], dtype=ca_atoms.atom_name.dtype)
    ca_atoms.element = np.array(['O'], dtype=ca_atoms.element.dtype)
    ca_atoms.coord = array.coord[array.atom_name == 'C'].squeeze()[None]

    return ca_atoms


def accumulate_components(
    components,
    src_atom_array,
    redesign_motif_sidechains=True,
    unindexed_components: list[str] = None,
    contig_atoms: dict = {},
):
    '''
    Subcomponents have three types, specifying either the end of a chain ("/0),
    a motif (e.g. "A20" or "A21"), or a number indicating the number of diffused residues to create.
    This function accumulates these components into a single atom array.
    
    Arguments:
        - components: list of components, where each component is either a string
        e.g. [2, A20, A21, 2, A25, 3, A30, /0, 3]
        - src_atom_array: the source atom array to fetch motifs from, or None if no input is provided. 
    
        - unindexed_components: list of components to unindex e.g. [A20, A21]

    Returns:
        - Accumulated atom array with components, and is_motif labels
    '''
    
    # 1) Construct functions for insertions
    def concatenate(a0, a):
        a0 = a if a0 is None else a0 + a
        return a0
    def fetch_motif_residue(src_chain, src_resid):
        '''
        Given source chain and resid, returns the residue if present in the source atom array

        NB: For glycines, we extend the array with a CB position so as to not leak whether
        the original residue is a glycine if sequence is masked during inference.
        '''

        assert src_atom_array is not None, "Motif provided in contigs, but no input provided. "\
            "input={} contig={}".format(input, components)

        # ... Fetch residue in the input atom array
        mask = (src_atom_array.chain_id == src_chain) & \
               (src_atom_array.res_id == src_resid)
        if not np.any(mask):
            raise ValueError(
                f"Residue {chain}{res_id} in molecule {molecule_id} not found in input atom array."
                f"Atom array: {src_atom_array}"
            )
        subarray = src_atom_array[mask]

        # Assign base properties
        subarray.set_annotation('is_motif_token', np.ones(subarray.shape[0], dtype=int))
        subarray.set_annotation('res_id', np.full(subarray.shape[0], 1))  # Reset to 1

        # Assign is motif atom and sequence
        if exists(atoms := contig_atoms.get(f"{src_chain}{src_resid}")):
            atom_mask = np.isin(subarray.atom_name, atoms)
            assert atom_mask.sum() == len(atoms), f"Not all atoms in {atoms} found in {subarray.atom_name}"
            subarray.set_annotation('is_motif_atom', atom_mask)
            subarray.set_annotation('token_has_sequence', np.ones(subarray.shape[0], dtype=int))
        elif redesign_motif_sidechains:
            n_atoms = subarray.shape[0]
            diffuse_oxygen=False        
            if n_atoms < 3:
                raise ValueError(f"Not enough data for {src_chain}{src_resid} in input atom array.")
            if n_atoms == 3:
                # Handle cases with N, CA, C only;
                subarray = subarray + create_o_atoms(subarray.copy())
                diffuse_oxygen = True  # flag oxygen for generation

            # Subset to the first 4 atoms (N, CA, C, O) only
            subarray = subarray[np.isin(subarray.atom_name, ['N', 'CA', 'C', 'O'])]
        
            # exactly N, CA, C, O but no CB. Place CB onto idealized position and conver to ALA
            # Sequence name ALA ensures the padded atoms to be diffused from the fixed backbone
            # are placed on the CB so as to not leak the identity of the residue.
            subarray = subarray + create_cb_atoms(subarray.copy())

            # Sequence name must be set to ALA such that the 
            subarray.res_name = np.full_like(subarray.res_name, 'ALA', dtype=subarray.res_name.dtype)
            subarray.set_annotation('is_motif_atom', (np.arange(subarray.shape[0], dtype=int) < (4 - int(diffuse_oxygen)) ).astype(bool))
            subarray.set_annotation('token_has_sequence', np.zeros(subarray.shape[0], dtype=int))
            
        else:
            # Provide full sequence
            subarray.set_annotation('is_motif_atom', np.ones(subarray.shape[0], dtype=int))
            subarray.set_annotation('token_has_sequence', np.ones(subarray.shape[0], dtype=int))

        # ... For now, all motif atoms also have fixed positions
        subarray.set_annotation('is_motif_atom_with_fixed_pos', subarray.is_motif_atom)

        # ... Flag parts of token for unidexing in Masking function.
        to_unindex = f'{src_chain}{src_resid}' in unindexed_components
        subarray.set_annotation('is_motif_atom_without_index', np.full(subarray.shape[0], to_unindex, dtype=int))
        if to_unindex:
            subarray = subarray[subarray.is_motif_atom.astype(bool)]
            subarray.set_annotation('is_motif_atom', np.zeros(subarray.shape[0], dtype=int))

        # ... Double check that required annotations are set within the scope of this function only
        Mask.check_has_required_annotations(subarray)
        return subarray

    def create_diffused_residues(n):
        if n <= 0:
            raise ValueError(f"Negative residue count {n} not allowed in components: {components}")

        atoms = []
        [
            atoms.extend([Atom(np.array([0.0, 0.0, 0.0], dtype=np.float32), 
                res_name='ALA', res_id=idx)
                for _ in range(5)])  
            for idx in range(1, n + 1)
        ]
        array = struc.array(atoms)
        array = Mask.set_default_annotations(array)
        array.set_annotation('element', np.array(['N', 'C', 'C', 'O', 'C']*n, dtype='<U2'))
        array.set_annotation('atom_name', np.array(['N', 'CA', 'C', 'O', 'CB']*n, dtype='<U2'))        
        Mask.check_has_required_annotations(array)
        return array

    # ... For loop accum variables
    atom_array_accum = None
    chain = "A"
    molecule_id = 0
    res_id = 1
    # 2) Insert contig information one- by one-
    for component in components:
        if component == "/0":
            # reset iterators on next chain
            chain = chr(ord(chain) + 1)
            molecule_id += 1
            res_id = 1
            continue
        
        # Create array to insert
        if str(component)[0].isalpha():  # motif (e.g. "A22")
            atom_array_insert = fetch_motif_residue(*split_contig(component))
            n=1
        else:
            n = int(component)
            if n == 0:
                continue
            atom_array_insert = create_diffused_residues(n)

        # ... Set index of insertion
        atom_array_insert = set_indices(atom_array_insert, chain, res_id, molecule_id)
        assert len(get_token_starts(atom_array_insert)) == n, \
            f"Mismatch in number of residues: expected {n}, got {len(get_token_starts(atom_array_insert))} in \n{atom_array_insert}"

        # ... Insert
        atom_array_accum = concatenate(atom_array_accum, atom_array_insert)

        # ... Increment residue ID
        res_id += n

    assert atom_array_accum is not None, "No components accumulated: {}".format(components)

    # post-process ? don't think anythings needed. @ Rafi?
    atom_array_accum.set_annotation('pn_unit_iid', atom_array_accum.chain_id)

    return atom_array_accum

def create_atom_array_from_design_specification(
    *,
    # Specification args:
    input: PathLike = None,
    length: str = '25-300',
    contig: str = None,
    contig_atoms: str = None,
    unindex: str = None,
    redesign_motif_sidechains: bool = True,
    # Other
    ligand: str = None,
    atomwise_rasa: dict = None, # TODO: eg: {'bu1':{'C12':1,'C6':1,'C7':1,'C8':1,'O4':1,'O3':1}}
    ori_token: list[float] = None,
    # Additional args:
    out_path=None,
    cif_parser_args=None,
    **_  # dump additional args
):
    '''
    Create pre-pipeline CIF file.
    
    Arguments:
        - input: path to input pdb containing coordinate data
        - contig: your typical contig string '10-10,A20-21,5-5,A25-25,5-5,A30-30,10-10'
        - unindex: string of residue indices to unindex, e.g. "A20,A21" or "A20-21" (optional)

        - length: required total length (optional)
        - ligand: name of ligand to keep from input pdb, or path to a cif file containing the ligand
        - ori_token: coordinates for origin relative to coordinates in input file.

    Returns:
        - atom_array
            is_motif_atom: indicating whether an atom is a part of a motif.
    '''
    ###########################################################################################################################
    # TODO: Flagged for cleanup.

    # 1) Load input data if provided
    if exists(input):
        atom_array_input = load_(input, cif_parser_args=cif_parser_args)['atom_array']
    elif exists(contig) or exists(length):
        atom_array_input = None
    else:
        raise ValueError("Either 'input' or 'contig' / 'length' must be provided.")
    if isinstance(length, int):
        length = f'{length}-{length}'
    if exists(length) and not exists(contig):
        # Handle cases where contigs aren't specified
        if not exists(unindex):
            if exists(contig_atoms):
                raise ValueError("Cannot specify contig atoms without contig string or unindexed residues")
            ranked_logger.warning("No input contig specified and no motif, running unconditional generation")
            contig=length
        else:
            contig = length + ',' + unindex
            length = None
    if not exists(contig_atoms):
        contig_atoms = {}
    else:
        contig_atoms = {}  # TODO: implement
        # contig_atoms = {k: v for k, v in contig_atoms.items()}

    # TODO: ... Add RASA conditioning to inputs
    # if exists(atomwise_rasa):
        # add_rasa_flags(atom_array_input, atomwise_rasa)

    # 2) Parse contigs into components
    components = get_design_pattern_with_constraints(contig, length)  # e.g. [2, A20, A21, 2, A25, 3, A30, /0, 3]
    ###########################################################################################################################

    unindexed_components = get_design_pattern_with_constraints(unindex) if exists(unindex) else []
    
    # 3) Create atom array from components
    atom_array = accumulate_components(components, 
        src_atom_array=atom_array_input, 
        redesign_motif_sidechains=redesign_motif_sidechains,
        unindexed_components=unindexed_components,
        contig_atoms=contig_atoms
    )

    # ... If ligand, post-pend it
    if exists(ligand):
        for lig in ligand.split(','):
            ligand_array = atom_array_input[atom_array_input.res_name == lig]
            # TODO: Set the is_motif_atom and is_motif_atom_without_index and token_has_sequence to be False
            # TODO: probably some chain crap here too.
            ligand_array = Mask.set_default_annotations(ligand_array, fill=True)
            atom_array = atom_array + ligand_array 

    # ... ORI token handling; detemine offset to place motif coordinates
    if exists(ori_token):
        atom_array.coord = atom_array.coord - np.array([float(x) for x in ori_token], dtype=atom_array.coord.dtype)
    else:
        # No offset
        if np.any(atom_array.is_motif_atom.astype(bool)):
            center = np.mean(atom_array.coord[atom_array.is_motif_atom.astype(bool)], axis=0)
            ranked_logger.warning("No ori_token provided for motif. Setting origin as COM of motif ({}).".format(center))
            atom_array.coord -= center
        else:
            ranked_logger.info("No ori_token and no motif provided. Setting [0,0,0] as origin.")
            atom_array.coord = np.zeros_like(atom_array.coord, dtype=atom_array.coord.dtype)
    
    # diffused atoms initialized at origin
    atom_array.coord[~atom_array.is_motif_token.astype(bool)] = 0.0

    # Ensure correct annotations before saving
    Mask.check_has_required_annotations(atom_array)

    if out_path is not None:
        to_cif_file(atom_array, out_path, extra_fields=Mask.required_annotations)
    return atom_array

def load_(
    file: PathLike,
    *,
    assembly_id: str = '1',
    cif_parser_args: dict | None = None
):
    # Default cif_parser_args to an empty dictionary if not provided
    if cif_parser_args is None:
        cif_parser_args = {}
        raise ValueError("Must specify parser args for this function. Uses non-default params for inference")

    # Convenience utilities to default to loading from and saving to cache if a cache_dir is provided, unless explicitly overridden
    if "cache_dir" in cif_parser_args and cif_parser_args["cache_dir"]:
        cif_parser_args.setdefault("load_from_cache", True)
        cif_parser_args.setdefault("save_to_cache", True)

    merged_cif_parser_args = {**DEFAULT_CIF_PARSER_ARGS, **cif_parser_args}

    # Ensure the required annotations can be loaded
    merged_cif_parser_args['extra_fields'] = list(set(
        merged_cif_parser_args.get('extra_fields', []) + Mask.required_annotations
    ))

    # Use the parse function with the merged CIF parser arguments
    result_dict = parse(
        filename=file,
        build_assembly=(assembly_id,),  # Convert list to tuple (make hashable)
        **merged_cif_parser_args,
    )

    data = {
        "atom_array": result_dict["assemblies"][assembly_id][0],  # First model
        "atom_array_stack": result_dict["assemblies"][assembly_id],  # All models
        "chain_info": result_dict["chain_info"],
        "ligand_info": result_dict["ligand_info"],
        "metadata": result_dict["metadata"],
    }

    return data
