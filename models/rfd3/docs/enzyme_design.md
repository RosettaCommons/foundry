# RFdiffusion3 — Enzyme design examples
RFD3 contains several knobs and dials for enzyme design. 
- input: the pdb or cif file that contains the input theozyme
- ligand: any ligand res names that are to be included (comma separated)
- unindex: which residues should have their index be inferred by the model instead of prespecified
- length: the length range of the generated protein
- select_fixed_atoms: dictionary that indicated which atoms should be fixed (can use ALL, BKBN, or TIP for all atoms in the residue, backbone atoms only and tip atoms only)
```json
{
    "M0255_1mg5_unfixed": {
        "input": "./input_pdbs/M0255_1mg5.pdb", 
        "ligand": "NAI,ACT",
        "unindex": "A108,A139,A152,A156",
        "length": "180-200",
        "select_fixed_atoms": {
            "A108": "ND2,CG",
            "A139": "OG,CB,CA",
            "A152": "OH,CZ",
            "A156": "NZ,CE,CD",
            "ACT": "OXT",
            "NAI": ""
        }
    }
}
```

