# Input parsing to RFdiffusion3
Contains:
<!-- - inference arguments -->
- Documentation on specification arguments

# Specification arguments (input instantiation)
### Top level structure

**UPDATE**: YAML files can now also be used if you prefer! Simply substitute the equivalent YAML for any JSON file, and all arguments/configs will work equivalently.

JSON inputs take the following top-level structure;
```yaml
{
    "name_of_design": {
        **dictionary_of_specification_args
    },
    "name_of_second_design": {
        input: PathLike = None,
        length: str = "100-300",
        contig: str = None,
        fixed_atoms: dict = None,
        unindex: str = None,
        unfix_sequence: str = None,
        redesign_motif_sidechains: bool = True,
        unfix_all=False,
        unfix_specific: null,
        ligand: str = None,
        ori_token: list[float] = None,
        atomwise_rasa: dict = None,
        add_all_na_as_motif: bool = False,
        atomwise_hbond: dict = None,
        # Additional args:
        out_path=None,
        cif_parser_args=None,
    }
}
```

### Enzyme-design Cookbook for RFdiffusion3
*Use checkpoint `/projects/ml/aa_design/models/rfd3_latest.ckpt` for enzyme design. This will be updated continuously*

The following will detail how to run RFD3 with:
- Specify fixed atoms and fixed atom contigs
- Specify RASA conditioning
- Specify a single or multiple ORI tokens
- Run inference on partially fixed ligands
- Specify partial unindexing where relative motif offsets are partially known
- Specify sequence-agnostic fixed atom motifs (e.g. carboxyllic acids)
- Specify atoms to be hydrogen bond donors or acceptors (Although dict format only 1 is useful)
  
**General arguments:** *Fixed atoms, Unindexed Residues, Ligands and Length* can all be specified rather simply as seen in this example config:

```
{
  "M0078_1al6": {
      // Inputs can be any parsable cif/pdb file. If not absolute and not found, the loader will try to find the input in the same directory as the JSON itself
      "input": "./aa_design/tests/test_data/mcsa_41/M0078_1al6.pdb",
      "ligand": "OAA,HAX", // Selection based on res_name, can also use pdb idx, e.g. X1    
      "unindex": "A244,A274,A320,A329,A375", // Supports similar syntax to a contig string
      "length": 180,  // integer or string e.g. "180-200" to randomly sample lengths
      "fixed_atoms": {
          "A244":"OG,CB",
          "A274":"ALL", // "ALL" will fix all coordinates in space
          "A320":"ND1,CG,NE2,CD2,CB,CE1",
          "A329":"TIP", // "TIP" will fix the tipatoms (see constants.py for defition by residue)
          "A375":"OD2,CG,OD1" 
      },
      "unfix_sequence": "A375"  // Contig string of components to relax sequence of. Relaxes A375 to be either GLU or ASP
      "unfix_specific": "A64,A86"  // Residues to unfix coordinates for. Use "ALL" to unfix every motif or use unfix_all
      "atomwise_hbond": { // Dictionary specifying which atoms on the fixed atoms are also hydrogen donors or acceptors
        "active_donor": {
            "A244": {
                "OG": 1
            }
        },
        "active_acceptor": {
            "A329": {
                "NH1,NH2": 1
            }
        }
    },
  },
}
```

Note you do not need to specify a `contig` for indexed components -- You can, however, mix indexed and unindexed components by supplying a contig anyway.

**ORI tokens** can be provided as either a command-line argument, or within the input (PDB or CIF) file. If you provide multiple ORI tokens in the input file, each example will be randomly instantiated with one of them. This will currently not randomize the ori tokens within the batch:
- JSON argument: `"ori_token": [0,1,2]` list of coordinates x,y,z relative to the coordinates in the input file to center on.
- Command line argument: you can use `ori_token=\[0,0,1\]` (note the backslashes to provide [ as an escaped character)
- In the input PDB file (supports multiple, will randomly select one for each batch): by adding a hetero atom line like:

    ```HETATM  106  ORI ORI D 273       0.000   0.000   0.000  1.00  0.00           X  ```
NB: if multiple of the above are provided the order will be: (i) in JSON, (ii) command line, (iii) in PDB, (iv) COM of fixed motifs (default).

You can also add inference-time jitter (jitters all batches relative to the ori token set), by specifying at the command line `inference_sampler.s_jitter_origin=1.0` to jitter the initialized COM of the diffused region by Gaussian noise of variance 1.0 Angstrom.

**Partial unindexing / sequence index-tied residues**
You can specify consecutive residues as e.g. `A11-12` (instead of `A11,A12`), this will tie the two components together in sequence (it leaks to the model that residues are together in sequence). Similarly, you can specify manually any number of residues that offsets two components, e.g. `A11,0,A12` (0 sequence offset, equivalent to just `A11-12`), or `A11,3,A12` (3-residue separation).
From our initial tests this only leads to a slight bias in the model, but newer models may show better adherence!

**Sequence agnostic motifs / functional group**
To relax the sequence constraint for a specific residue, you can specify `unfix_sequence` as a comma-separated list of components

The cases where this is expected to work are:
- Carboxylates for ASP/GLU
- Amides for ASN/GLN
- Oxygens for SER/THR

**RASA Conditioning** Three bins exist; buried (0), partially buried (1) and exposed (2). Not specifying a value simply does not provide a bin to the model.

### Debugging
- For unindexed scaffolding, you can use the option `cleanup_guideposts=False` to keep the models' outputs for the guideposts. The guideposts are saved as separate chains based on whether their relative indices were leaked to the model: e.g. for `unindex=A11-12,A22`, you should see `A11` and `A12` indexed together on one chain and `A22` on it's own chain. Indicating the model was provided with the fact that `A11` and `A12` are immediately next to one another in sequence but their distance to `A22` is unknown.
- To see the full 14 diffused virtual atoms you can use `cleanup_virtual_atoms=False`. Default is to discard them for the sake of downstream processing.
- To see the trajectories, you can use `dump_trajectories=True`. This can be useful if the outputs look strange but the config is correct, or if you want to make cool gifs of course! Trajectories do not have sequence labels and contain virtual atoms.
- To see the sequence head confidence in each sequence, you can run `spectrum b` in `pymol` as the sequence logit entropy is stored in the bfactor column of the outputs.


### Partial Diffusion
To enable partial diffusion, you can pass `partial_t` with any example. This sets the *noise level* in *angstrom* for the sampler:
- The `specification.partial_t` arg can be specified from the json or command line.
- Partial diffusion will fix/unfix ligands and nucleic acids as normal, by default it will fix non-protein components and they must be specified explicitly.
- By default, the ca-aligned `ca_rmsd_to_input` will be logged.
- Currently, partial diffusion subsets the inference schedule based on the partial_t, so `inference_sampler.num_timesteps` will affect how many steps are used but it is not equal to the number of steps used.

The following example will noise out by 15 angstroms, and constrain atoms of three residues. In this output one of the 8 diffusion outputs swapped their sequence index by one residue:
```json
    "partial_diffusion": {
        "input": "paper_examples/7v11.cif", 
        "ligand": "OQO", 
        "partial_t": 15.0,
        "unindex": "A431,A572-573",
        "fixed_atoms": {
            "A431": "TIP",
            "A572": "BKBN",
            "A573": "BKBN"
        }
    }
```

<p align="center">
  <img src=".assets/partial_diff.png" alt="Partial diffusion with RFdiffusion3" width="60%">
  <figcaption>
  Partial diffusion with tipatom constraints (`demo.json`). 
  Here, the input is in navaho, and generations are in teal. Constrained residues and atoms are in blue
  </figcaption>
</p>

The following are on my todo list and will come soon:
- Partial diffusion does not currently support differential noising, in part because the model isn't trained on this.
- Logging of TM-score of the design to the input / sequence similarity
- Explicit redesign motif sidechains


### Full example (from demo.json)
below is (a version) of the demo.json:
```json
{
    "1nzy-1": {
        "input": "M0024_1nzy.pdb", 
        "ligand": "BCA",
        "contig": "118,A137,7,A145,49",
        "unindex": "A64,A86,A90,A114",
        "ori_token": [0,0,0],
        "fixed_atoms": {
            "A64":  "O,C",
            "A86":  "CB,CA,N,C",
            "A90":  "CE1,ND1,NE2,CG,CD2",
            "A114": "N,CA",
            "A137": "NE1,CD1,CE2,CG,CD2,CZ2",
            "A145": "OD2,CG,CB,OD1",
            "BCA":  "C6B,C5B,C7B,C4B,O2B,C2B,C3B,C1B,S1P,O1B,C2P,C3P,N4P,C5P,C6P,O5P,C7P,N8P,C9P,CAP,O9P,CBP,OAP,CCP,CDP,CEP,O6A,P2A"
        },
        "length": 150,
        "atomwise_rasa": {
            "BCA": {
                "CDP,CEP,CAP,OAP,C9P,O9P,N8P,C7P,C6P,C5P,O5P,N4P,C3P,C2P,S1P,C1B,O1B,C2B,C3B,C4B,C5B,O2B,C6B,C7B": 0,
                "N1A,C2A,N3A,C4A,C5A,C6A,N6A,N7A,C8A,N9A,C1D,C2D,O2D,C3D,O3D,C4D,O4D,C5D,O5D,P1A,O1A,O2A,O3A,P2A,O4A,O5A,O6A,CBP,CCP": 2
            }
        },
        "atomwise_hbond": {
            "active_donor": {
                "A145": {
                    "OD1,OD2": 1
                },
                "A90": {
                    "ND1": 1
                }
            },
            "active_acceptor": {
                "A137": {
                    "NE1": 1
                }
            }
        },
    },
    "rsv0-1": {
        "input": "rsv0_4jhw.pdb",
        "contig": "19-19,F63-69,34-34,F196-211",
        "length": "76-76"
    }
}
```

### Multi-example files
You can also specify multiple examples within a single json/yaml if you prefer. In this case, if there are certain flags that you would like to apply to all examples in a file, you can include this in a top-level "global_args" category. For example, if using a multi-example YAML file:

```yaml
global_args:
  length: 150 # Will be applied to all examples

insulinr:
    # Design specifications here

pdl1:
    # Design specifications here

# ... More examples below if desired
```
