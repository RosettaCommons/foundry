from atomworks.constants import AA_LIKE_CHEM_TYPES, STANDARD_AA, UNKNOWN_AA
from atomworks.ml.encoding_definitions import TokenEncoding

# Token ordering for MPNN.
token_order = STANDARD_AA + (UNKNOWN_AA,)

# Token ordering for old versions of MPNN.
legacy_token_order = (
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
    "UNK",
)

# Atom ordering for new versions of MPNN.
atom_order = (
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
)

# Atom ordering for old versions of MPNN.
legacy_atom_order = (
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
)

# Token encoding for MPNN.
MPNN_TOKEN_ENCODING = TokenEncoding(
    token_atoms={token: atom_order for token in token_order},
    chemcomp_type_to_unknown={chem_type: "UNK" for chem_type in AA_LIKE_CHEM_TYPES},
)

# Token encoding for versions of MPNN using the legacy token order and
# new atom order.
MPNN_LEGACY_TOKEN_ENCODING = TokenEncoding(
    token_atoms={token: atom_order for token in legacy_token_order},
    chemcomp_type_to_unknown={chem_type: "UNK" for chem_type in AA_LIKE_CHEM_TYPES},
)

# Token encoding for versions of MPNN using the legacy token order and
# legacy atom order.
MPNN_LEGACY_TOKEN_LEGACY_ATOM_ENCODING = TokenEncoding(
    token_atoms={token: legacy_atom_order for token in legacy_token_order},
    chemcomp_type_to_unknown={chem_type: "UNK" for chem_type in AA_LIKE_CHEM_TYPES},
)
