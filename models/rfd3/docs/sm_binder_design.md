# RFdiffusion3 — Small molecule binder design examples

If you would like to run the examples below, `sm_binder_design.json`, located in this directory, contains the example code. You can run it via:
```
rfd3 design out_dir=inference_outputs/sm_binder/0 \
ckpt_path=/path/to/rfd3_foundry_2025_12_01.ckpt \
inputs=./sm_binder_design.json
```

Or, if you have cloned the repo rather than using `pip install`:
```
python path/to/foundry/models/rfd3/src/rfd3/run_inference.py \
out_dir=inference_outputs/sm_binder/0 \
ckpt_path=/path/to/rfd3_foundry_2025_12_01.ckpt \
inputs=./sm_binder_design.json 
```

An example script for running these examples in batches is also provided in `run_inf_tutorial.sh`.

The input files for the different examples are provided in `foundry/models/rfd3/docs/input_pdbs`.

## Small molecule binder examples against the ligand IAI with different RASA conditioning
RFD3 is also capable of designing proteins that bind small molecules. Here are some inputs that could be useful:
- input: a PDB or CIF file that contains the small molecule 
- ligand: the 3 letter code in the file that is the ligand to be bound
- length: how long the generated protein should be (can be a range)
- select_fixed_atoms: selecting which atoms in the ligand should be fixed to the coordinates in the PDB 
- select_exposed: selecting which atoms in the ligand should not be buried in the designed protein (exposed to solvent)
- select_buried: selecting which atoms in the ligand should be buried in the designed protein

```json
{
    "buried": {
        "input": "./input_pdbs/IAI.pdb",
        "length": "180-180",
        "ligand": "IAI",
        "select_fixed_atoms": {
            "IAI": ""
        },
        "select_buried": {
            "IAI": "C22,C23,C25,C24,C21,C20,N13,C15,C16,N14,C19,C11,N12,C18,C17,N9,O8,C4,C1,N3,C10,N5,C7,C2,C6,N27,O26,C33,C29,C32,O31,C30,N28"
        }
    },
    "partial": {
        "input": "./input_pdbs/IAI.pdb",
        "ligand": "IAI",
        "length": "180-180",
        "select_fixed_atoms": {
            "IAI": ""
        },
        "select_exposed": {
            "IAI": "C22,C23,C25,C24,C21,C20,N13,C15,C16,N14,C19,C11,N12,C18,C17"
        },
        "select_buried": {
            "IAI": "N9,O8,C4,C1,N3,C10,N5,C7,C2,C6,N27,O26,C33,C29,C32,O31,C30,N28"
        }
    }
}
```

