# RFdiffusion3 — Nucleic acid binder design examples

If you would like to run the examples below, `na_binder_design.json`, located in this directory,
contains the example code. You can run it via:
```
rfd3 design out_dir=inference_outputs/na_binder/0 \
ckpt_path=/path/to/rfd3_foundry_2025_12_01.ckpt \
inputs=./na_binder_design.json
```


Or, if you have cloned the repo rather than using `pip install`:
```
python path/to/foundry/models/rfd3/src/rfd3/run_inference.py \
out_dir=inference_outputs/na_binder/0 \
ckpt_path=/path/to/rfd3_foundry_2025_12_01.ckpt \
inputs=./na_binder_design.json 
```

An example script for running these examples in batches is also provided in `run_inf_tutorial.sh`.

The input files for the different examples are already provided in `input_pdbs`, but if you would like
to see how you could download these directly from the PDB, see `get_na_input.sh`.

### 1. Simple dsDNA binder example

The DNA chains are A and B and specified as such in the contig. RFD3 will treat these as fixed in space. the contig specifies to generate a protein chain of length between 120-130. An ori token is specified.
The length attribute should be the sum of all polymer lengths. in this case (120 to 130) + 10 + 10  = (140 to 150) 
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

### 2. Simple ssDNA binder example G-quadruplex

Similar to the previous example, but done for a PDB containing one DNA strand (A):

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

### 3. ssDNA example based on DNA sequence diffused from dsDNA pdb as input

Similar to the previous example but the input PDB has a dsDNA. One of the chains (A) is selected. However, the single stranded DNA conformation will be sampled by RFD3 because we have specified to not have any fixed DNA atoms by using `"select_fixed_atoms": {"A1-10":""}`. ori_token is not meaningful to specify when there are no fixed atoms.
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

### 4. Simple RNA binder example

Example on RNA. Similar to the ssDNA example, example 2.

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

### 5. Complex example based on a protein-dsDNA input pdb with parts of protein and dna partially fixed (indexed and unindexed), with Hbond conditioning

This is a complex example which has a dsDNA specified in the contig: `C5-18` and `D24-37`. However, it also specifies an indexed protein motif component (`A146-154`) and diffuses the two flanks of the protein indexed region in the same chain. The diffused protein region has an unindexed motif specified via `"unindex": "/0,/0,B251-B255".` (*Note: the chain breaks applied are analogous to the contig string*). Parts of the DNA have been specified as fixed or to be sampled by RFD3 (`select_fixed_atoms`). Additionally hydrogen bond conditioning is applied to some backbone and base atoms of a few DNA bases.

To run this without warnings, you will need to install [hpblus](https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/) to enable hydrogen bond metrics computation. This is discussed at the end of the RFD3 README, but the instructions are reproduced here for convenience:

1. Download hbplus from here: https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/download.html (available for free)
2. Follow the installation instruction here: https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/install.html
3. Update `HBPLUS_PATH` in `foundry/.env` file with the path to your `hbplus` executable.

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
