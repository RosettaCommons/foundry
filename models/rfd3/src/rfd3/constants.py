import numpy as np

from foundry.constants import TIP_BY_RESTYPE

TIP_BY_RESTYPE

# Annot: default (diffused default)
REQUIRED_CONDITIONING_ANNOTATION_VALUES = {
    "is_motif_atom_with_fixed_seq": True,
    "is_motif_atom_with_fixed_coord": True,
    "is_motif_atom_unindexed": False,
    "is_motif_atom_unindexed_motif_breakpoint": False,
}
REQUIRED_CONDITIONING_ANNOTATIONS = list(REQUIRED_CONDITIONING_ANNOTATION_VALUES.keys())
REQUIRED_INFERENCE_ANNOTATIONS = REQUIRED_CONDITIONING_ANNOTATIONS + ["src_component"]
"""Annotations assigned to every valid atom array"""

OPTIONAL_CONDITIONING_VALUES = {
    "is_atom_level_hotspot": 0,
    "is_helix_conditioning": 0,
    "is_sheet_conditioning": 0,
    "is_loop_conditioning": 0,
    "active_donor": 0,
    "active_acceptor": 0,
    "rasa_bin": 3,
    "ref_plddt": 0,
    "is_non_loopy": 0,
    "partial_t": np.nan,
    # kept for legacy reasons
    "is_motif_token": 1,
    "is_motif_atom": 1,
}
"""Optional conditioning annotations and their default values if not provided."""

CONDITIONING_VALUES = (
    REQUIRED_CONDITIONING_ANNOTATION_VALUES | OPTIONAL_CONDITIONING_VALUES
)
"""Annotations that must be present in the AtomArray at inference time."""

INFERENCE_ANNOTATIONS = REQUIRED_INFERENCE_ANNOTATIONS + list(
    OPTIONAL_CONDITIONING_VALUES.keys()
)
"""All annotations that might be desired at inference time. Determines what AtomArray annotations will be preserved."""

SAVED_CONDITIONING_ANNOTATIONS = [
    # "is_motif_atom_with_fixed_coord",
    "is_motif_atom_with_fixed_seq",
]
"""Annotations for conditioning to save in output files"""

# fmt: off
ccd_ordering_atomchar = {
    'TRP': (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," NE1"," CE2"," CE3"," CZ2"," CZ3"," CH2"),  # trp
    'HIS': (" N  "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None),  # his
    'TYR': (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ "," OH ",  None,  None),  # tyr
    'PHE': (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ ",  None,  None,  None),  # phe
    'ASN': (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None),  # asn
    'ASP': (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," OD2",  None,  None,  None,  None,  None,  None),  # asp
    'GLN': (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None),  # gln
    'GLU': (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," OE2",  None,  None,  None,  None,  None),  # glu
    'CYS': (" N  "," CA "," C  "," O  "," CB "," SG ",  None,  None,  None,  None,  None,  None,  None,  None),  # cys
    'SER': (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None),  # ser
    'THR': (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None),  # thr
    'LEU': (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None),  # leu
    'VAL': (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2",  None,  None,  None,  None,  None,  None,  None),  # val
    'ILE': (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None),  # ile
    'MET': (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None),  # met
    'LYS': (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None),  # lys
    'ARG': (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",  None,  None,  None),  # arg
    'PRO': (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None),  # pro
    'ALA': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None),  # ala
    'GLY': (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None),  # gly
    'UNK': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None),  # unk
    'MSK': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None),  # mask
}
"""Canonical ordering of amino acid atom names in the CCD."""

symmetric_atomchar = {
    "TYR": [[" CE1", " CE2"], [" CD1", " CD2"]],
    "PHE": [[" CE1", " CE2"], [" CD1", " CD2"]],
    "ASP": [[" OD1", " OD2"]],
    "GLU": [[" OE1", " OE2"]],
    "LEU": [[" CD1", " CD2"]],
    "VAL": [[" CG1", " CG2"]],
}
"""Maps residues to their pairs of aton names corresponding to symmetric atoms."""

association_schemes = {
    'atom14': {
        #      |         Backbone atoms           |sp2-L1|sp2-R1|sp2-L2|sp2-R2|sp2-CZ|O-/S-|beta-OH|sp3-CG|sp2-CG|
        #         0       1      2      3      4     V0     V1     V2     V3      V4    V5     V6     V7     V8
        # Aromatics
        'TRP': (" N  "," CA "," C  "," O  "," CB "," CD1"," CD2"," NE1"," CE2"," CE3"," CZ2"," CZ3"," CH2"," CG "), # trp
        'HIS': (" N  "," CA "," C  "," O  "," CB "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None," CG "), # his
        'TYR': (" N  "," CA "," C  "," O  "," CB "," CD1"," CD2"," CE1"," CE2"," CZ "," OH ",  None,  None," CG "), # tyr*
        'PHE': (" N  "," CA "," C  "," O  "," CB "," CD1"," CD2"," CE1"," CE2"," CZ ",  None,  None,  None," CG "), # phe*

        # Carboxylates & amines
        'ASN': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None," ND2"," OD1",  None,  None," CG "), # asn
        'ASP': (" N  "," CA "," C  "," O  "," CB ",  None,  None," OD1"," OD2",  None,  None,  None,  None," CG "), # asp*
        'GLN': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None," NE2"," OE1",  None," CD "," CG "), # gln
        'GLU': (" N  "," CA "," C  "," O  "," CB ",  None,  None," OE2"," OE1",  None,  None,  None," CD "," CG "), # glu*

        # CB-OH and CB-SG
        'CYS': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None," SG ",  None,  None,  None), # cys
        'SER': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None," OG ",  None,  None), # ser
        'THR': (" N  "," CA "," C  "," O  "," CB "," CG2",  None,  None,  None,  None,  None," OG1",  None,  None), # thr

        # Ile/Leu/Val have a common C backbone but different placements of branching C
        'LEU': (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None), # leu*
        'VAL': (" N  "," CA "," C  "," O  "," CB "," CG1",  None,  None," CG2",  None,  None,  None,  None,  None), # val*
        'ILE': (" N  "," CA "," C  "," O  "," CB "," CG1"," CD1",  None," CG2",  None,  None,  None,  None,  None), # ile

        # MET / LYS have a common C backbone but heteroatoms inbetween
        'MET': (" N  "," CA "," C  "," O  "," CB "," CG ",  None," CE ",  None,  None," SD ",  None,  None,  None), # met
        'LYS': (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE ",  None," NZ ",  None,  None,  None,  None), # lys
        
        # Weird ones
        'ARG': (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," NH1"," CZ "," NH2",  None,  None,  None), # arg*
        'PRO': (" N  "," CA "," C  "," O  "," CB "," CG ",  None,  None,  None,  None,  None,  None," CD ",  None), # pro

        # Other
        'UNK': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # unk
        'ALA': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # ala
        'MSK': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None), # mask
        'GLY': (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None), # gly
    },

    "permute_ambiguous_only": {
        # "CYS": [6, 5,],  # SER  |  Permute *CB and SG (*CB and OG)   # CB = next virtual atom since otherwise things get messy
        # "ASP": [8, 7],  #  [6, 5],  # ASN  |  Permute CG and OD2 (CG and OD1)
        # "GLU": [9, 8],  # [7, 6],  # GLN  |  Permute CD and OE2 (CD and OE1)

        # Ambiguous, modified
        'CYS': (" N  "," CA "," C  "," O  "," CB ",  None, " SG ", None,  None,  None,  None,  None,  None,  None),  # cys
        'ASP': (" N  "," CA "," C  "," O  "," CB "," CG "," OD1", None, " OD2",  None,  None,  None,  None,  None),  # asp
        'GLU': (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1", None, " OE2",  None,  None,  None,  None),  # glu

        # Ambiguous, unmodified
        'SER': (" N  "," CA "," C  "," O  "," CB "," OG ",  None,  None,  None,  None,  None,  None,  None,  None),  # ser
        'ASN': (" N  "," CA "," C  "," O  "," CB "," CG "," OD1"," ND2",  None,  None,  None,  None,  None,  None),  # asn
        'GLN': (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," OE1"," NE2",  None,  None,  None,  None,  None),  # gln

        # Unambiguous
        'TRP': (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," NE1"," CE2"," CE3"," CZ2"," CZ3"," CH2"),  # trp
        'HIS': (" N  "," CA "," C  "," O  "," CB "," CG "," ND1"," CD2"," CE1"," NE2",  None,  None,  None,  None),  # his
        'TYR': (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ "," OH ",  None,  None),  # tyr
        'PHE': (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2"," CE1"," CE2"," CZ ",  None,  None,  None),  # phe
        'THR': (" N  "," CA "," C  "," O  "," CB "," OG1"," CG2",  None,  None,  None,  None,  None,  None,  None),  # thr
        'LEU': (" N  "," CA "," C  "," O  "," CB "," CG "," CD1"," CD2",  None,  None,  None,  None,  None,  None),  # leu
        'VAL': (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2",  None,  None,  None,  None,  None,  None,  None),  # val
        'ILE': (" N  "," CA "," C  "," O  "," CB "," CG1"," CG2"," CD1",  None,  None,  None,  None,  None,  None),  # ile
        'MET': (" N  "," CA "," C  "," O  "," CB "," CG "," SD "," CE ",  None,  None,  None,  None,  None,  None),  # met
        'LYS': (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," CE "," NZ ",  None,  None,  None,  None,  None),  # lys
        'ARG': (" N  "," CA "," C  "," O  "," CB "," CG "," CD "," NE "," CZ "," NH1"," NH2",  None,  None,  None),  # arg
        'PRO': (" N  "," CA "," C  "," O  "," CB "," CG "," CD ",  None,  None,  None,  None,  None,  None,  None),  # pro
        'ALA': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None),  # ala
        'GLY': (" N  "," CA "," C  "," O  ",  None,  None,  None,  None,  None,  None,  None,  None,  None,  None),  # gly
        'UNK': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None),  # unk
        'MSK': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None,  None,  None,  None,  None,  None),  # mask
    },

    'ccd': ccd_ordering_atomchar,
}
association_schemes['atom14-new'] = association_schemes['atom14'].copy()
association_schemes['atom14-new'] |= {
        # Optional: Break TYR oxygen from GLN / ASN groups - not implemented for rfd3 since it might be useful for people to use
        # 'TYR': (" N  "," CA "," C  "," O  "," CB "," CD1"," CD2"," CE1"," CE2"," CZ ",  None,  None," OH "," CG "), # tyr*
        # Fixed carboxylate / amide groups:
        'GLN': (" N  "," CA "," C  "," O  "," CB ",  None,  None,  None,  None," NE2"," OE1",  None," CG "," CD "), # gln
        'GLU': (" N  "," CA "," C  "," O  "," CB ",  None,  None," OE2"," OE1",  None,  None,  None," CG "," CD "), # glu*
        # Break connection with carboxylates
        'HIS': (" N  "," CA "," C  "," O  "," CB "," ND1"," CD2"," CE1",  None,  None,  None," NE2",  None," CG "), # his
}
association_schemes['dense'] = association_schemes['permute_ambiguous_only'].copy()

# fmt: on
VIRTUAL_ATOM_ELEMENT_NAME = "VX"
"""The element name annotation that will be assigned to virtual atoms"""

ATOM14_ATOM_NAMES = np.array(
    ["N", "CA", "C", "O", "CB"] + [f"V{i}" for i in range(14 - 5)]
)
"""Atom14 atom names (e.g. CA, V1)"""

ATOM14_ATOM_ELEMENTS = np.array(
    ["N", "C", "C", "O", "C"] + [VIRTUAL_ATOM_ELEMENT_NAME for i in range(14 - 5)]
)
"""Atom14 element names (e.g. C, VX)"""

ATOM14_ATOM_NAME_TO_ELEMENT = {
    name: elem for name, elem in zip(ATOM14_ATOM_NAMES, ATOM14_ATOM_ELEMENTS)
}
"""Mapping from atom14 atom names (e.g. CA, V1) to their corresponding element names (e.g. C, VX)"""

strip_list = lambda x: [(x.strip() if x is not None else None) for x in x]  # noqa

association_schemes_stripped = {
    name: {k: strip_list(v) for k, v in scheme.items()}
    for name, scheme in association_schemes.items()
}
SELECTION_PROTEIN = ["POLYPEPTIDE(D)", "POLYPEPTIDE(L)"]
SELECTION_NONPROTEIN = [
    "POLYDEOXYRIBONUCLEOTIDE",
    "POLYRIBONUCLEOTIDE",
    "PEPTIDE NUCLEIC ACID",
    "OTHER",
    "NON-POLYMER",
    "CYCLIC-PSEUDO-PEPTIDE",
    "MACROLIDE",
    "POLYDEOXYRIBONUCLEOTIDE/POLYRIBONUCLEOTIDE HYBRID",
]
