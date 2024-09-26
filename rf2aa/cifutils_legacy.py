import collections

Atom = collections.namedtuple('Atom', [
    'name',
    'xyz', # Cartesian coordinates of the atom
    'occ', # occupancy
    'bfac', # B-factor
    'leaving', # boolean flag to indicate whether the atom leaves the molecule upon bond formation
    'leaving_group', # list of atoms which leave the molecule if a bond with this atom is formed
    'parent', # neighboring heavy atom this atom is bonded to
    'element', # atomic number (1..118)
    'metal', # is this atom a metal? (bool)
    'charge', # atomic charge (int)
    'hyb', # hybridization state (int)
    'nhyd', # number of hydrogens
    'hvydeg', # heavy atom degree
    'align', # atom name alignment offset in PDB atom field
    'hetero'
])

Bond = collections.namedtuple('Bond', [
    'a','b', # names of atoms forming the bond (str)
    'aromatic', # is the bond aromatic? (bool)
    'in_ring', # is the bond in a ring? (bool)
    'order', # bond order (int)
    'intra', # is the bond intra-residue? (bool)
    'length' # reference bond length from openbabel (float)
])

Residue = collections.namedtuple('Residue', [
    'name',
    'atoms',
    'bonds',
    'automorphisms',
    'chirals',
    'planars',
    'alternatives'
])

Chain = collections.namedtuple('Chain', [
    'id',
    'type',
    'sequence',
    'atoms',
    'bonds',
    'chirals',
    'planars',
    'automorphisms'
])

