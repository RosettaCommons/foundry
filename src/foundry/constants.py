# fmt: off
# ... For convenience, define BKBN, or TIP to be used as a shortcut | TIP is the largest set of fixed atom given at least 2 tip atoms
TIP_BY_RESTYPE = {
    "TRP": ["CG","CD1","CD2","NE1","CE2","CE3","CZ2","CZ3","CH2"],  # fix both rings
    "HIS": ["CG","ND1","CD2","CE1","NE2"],  # fixed ring
    "TYR": ["CZ","OH"],  # keeps ring dihedral flexible
    "PHE": ["CG","CD1","CD2","CE1","CE2","CZ"],
    "ASN": ["CB", "CG","OD1","ND2"],
    "ASP": ["CB", "CG","OD1","OD2"],
    "GLN": ["CG", "CD","OE1","NE2"],
    "GLU": ["CG", "CD","OE1","OE2"],
    "CYS": ["CB", "SG"],
    "SER": ["CB", "OG"],
    "THR": ["CB", "OG1"],
    "LEU": ["CB", "CG", "CD1", "CD2"],
    "VAL": ["CG1", "CG2"],
    "ILE": ["CB", "CG2"],
    "MET": ["SD", "CE"],
    "LYS": ["CE","NZ"],
    "ARG": ["CD","NE","CZ","NH1","NH2"],
    "PRO": None,
    "ALA": None,
    "GLY": None,
    "UNK": None,
    "MSK": None
}

# fmt: on
