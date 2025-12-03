# RFdiffusion3 — Nucleic acid binder design examples

### simple dsDNA binder example

The DNA chains are A and B and specified as such in the contig. RFD3 will treat these as fixed in space. the contig specifies to generate a protein chain of length between 120-130. An ori token is specified.
The length attribute shuold be the sum of all polymer lengths. in this case (120to130) + 10 + 10  = (140to150) 
```json
{
    "dsDNA_basic": { 
        "input": "./input_pdbs/1bna.pdb",
        "contig": "A1-10,/0,B15-24,/0,120-130",
        "length": "140-150",
        "ori_token": [24,20,10]
    }
}
```

### simple ssDNA binder example G-quadruplex

Similar to the previous one but done for a PDB containing one DNA strand (A)

```json
{
    "ssDNA_basic": {
        "input": "./input_pdbs/5o4d.pdb",
        "contig": "A1-23,/0,120-130",
        "length": "143-153",
        "ori_token": [-5,-10,8]
    }
}
```

### ssDNA example based on DNA sequence diffused from dsDNA pdb as input

Similar to the previous one but the input PDB has a dsDNA. one of the chains (A) is selected. However, the single stranded DNA conformation will be sampled by RFD3 because we have specified to not have any fixed DNA atoms by using `"select_fixed_atoms": {"A1-10":""}`. ori_token is not meaningful to specify when there is no fixed atom.
```json
{
    "ssDNA_diffused_from_dsDNA_pdb":{
        "input": "./input_pdbs/1bna.pdb",
        "contig": "A1-10,/0,120-130",
        "length": "130-140",
        "select_fixed_atoms": {"A1-10":""}
    }
}
```

### simple RNA binder example

Example on RNA. similar to ssDNA example.

```json
{
    "RNA_basic": {
        "input": "./input_pdbs/1q75.pdb",
        "contig": "A1-15,/0,120-130",
        "length": "135-145",
        "ori_token": [15,2,-4]
    }   
}
```

### complex example based on a protein-dsDNA input pdb with parts of protein and dna partially fixed (indexed and unindexed), with Hbond conditioning

This is a complex example which has a dsDNA specified in the contig C5-18 and D24-37. However, it also has specified indexed protein motif component A146-154 and diffusing the two flanks of the protein indexed region in the same chain. The diffused protein region has an unindexed motif specified too via `"unindex": "/0,/0,B251-B255",` (note: the chain breaks applied analogous to the contig string). Parts of the DNA has been specified to be fixed and parts of it left to be sampled by RFD3 (`select_fixed_atoms`). additionally hydrogen bond conditioning is applied to some backbone and base atoms of a few DNA bases.

```json
{
    "dsDNA_complex": {
        "input": "./input_pdbs/2r5z.pdb",
        "contig": "C5-18,/0,D24-37,/0,40-50,A146-154,80-90",
        "length": "147-167",
        "unindex": "/0,/0,B251-B255",
        "select_fixed_atoms": {
            "C9-14":"ALL",
            "D28-33":"ALL",
            "C5-8,C15-18": "",
            "D24-27,D34-37": ""
        },
        "ori_token":[25,35,20],
        "select_hbond_acceptor": {"C16":"N7,O6", "D31-32":"N7", "D28-30":"OP1,OP2,O3',O5'"},
        "select_hbond_donor": {"D31-32":"N6"}

    }
}
```
