# RFdiffusion3 — Protein binder design examples
RFD3 is a highly proficient protein binder designer. The following arguments have to be specified to RFD3 to make protein binders.
- input: the PDB or CIF file of the structure you want to bind
- contig: the length range of the binder to make (indicated as a range) and which residues from the target file to consider. 
- infer_ori_strategy: how rfd3 decides to place the origin of the generated protein binder with respect to the target. We find that using the "hotspots" strategy works best
- select_hotspots: which atoms on the target should be bound (dictionary of residues on the target and atoms in those residues)
```json

{
    "insulinr": {
        "dialect": 2,
        "infer_ori_strategy": "hotspots",
        "input": "input_pdbs/4zxb_cropped.pdb",
        "contig": "40-120,/0,E6-155",
        "select_hotspots": {
            "E64": "CD2,CZ",
            "E88": "CG,CZ",
            "E96": "CD1,CZ",
            }
    },
    "pdl1": {
        "dialect": 2,
        "infer_ori_strategy": "hotspots",
        "input": "input_pdbs/5o45_cropped.pdb",
        "contig": "50-120,/0,A17-131",
        "select_hotspots": {
            "A56": "CG,OH",
            "A115": "CG,SD",
            "A123": "CD2,OH",
       }
    }
}
```
