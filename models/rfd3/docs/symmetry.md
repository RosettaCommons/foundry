# Overview of Symmetry in RFD3

## Specifying symmetry in your input specifications
Symmetry configurations are specified within the input JSON or YAML file, nested under its own specific configuration. The symmetry specific config has the following:
```json
symmetry: {
    "id": "C3",
    "is_unsym_motif": "Y1-11,Z16-25",
    "is_symmetric_motif": true

}
```
```yaml
symmetry:
    id: "C3"
    is_unsym_motif: "Y1-11,Z16-25"
    is_symmetric_motif: true
```
- `id`                : Symmetry group ID; e.g. "C3" for a cyclic protein with 3 subunits, "D2" for a dihedral protein with 2 subunits. Note that only C and D symmetry types are supported currently.
- `is_unsym_motif`    : Comma separated string list of contig/ligand names that should NOT be symmetrized (e.g. DNA strands). If not provided, all motifs are assumed to be symmetrized. See [Designs with motifs](#designs-with-motifs) section for details.
- `is_symmetric_motif`: Boolean value whether the input motif is symmetric. Currently only symmetric input motifs are supported, therefore, `true` by default.


## Example command 
You can run the following example command:
```
./src/modelhub/inference.py inference_sampler.kind=symmetry out_dir=logs/inference_outs/sym_demo/0 ckpt_path=$cur_ckpt inputs=./projects/aa_design/tests/test_data/sym_tests.json diffusion_batch_size=1 
```
- `inference_sampler.kind`: Set `symmetry` to tern on symmetry mode.
- `diffusion_batch_size`  : `8` by default, but it is recommended to set it to `1` for symmetry due to memory limitations.
- `low_memory_mode`       : Additionally you can set this to `True` if you have memory constraints (e.g. "CUDA error: out of memory"). However, this will significantly slow the inference.


## Unconditional multimer design

As mentioned above, we currently only support C and D symmetry types.
The following provides a general overview of the types of symmetry and examples of how to run:

### Cyclic
**Defaults:**

```json
{
    "uncond_C15": {
        "length": 100,
        "symmetry": {
            "id": "C15"
        }
    }
}
```

### Dihedrals
**Defaults:**

```json
{
    "uncond_D4": { 
        "length": 100,
        "symmetry": {
            "id": "D4"
        }
    }
}
```

## Designs with motifs

As mentioned above, symmetry sampling currently only supports pre-symmetrized motifs around the origin. Therefore, `is_symmetric_motif` is set to `true` by default. 
The following are example JSON specifications for different symmetric motif scaffolding. You can also find the corresponding input PDBs in `docs/input_pdbs/symmetry_examples`. Although we only give JSON examples, you can also use YAML for everything shown below.   

The tasks that these examples describe are as follows:
- unindexed_C2_1j79, unindexed_C2_1e3v: 
 Unindexed motif scaffolding for symmetric enzyme active sites. The motifs are located within a subunit; no inter-subunit motifs.
- indexed_unsym_C2_1bfr:
 Indexed motif scaffolding for a single active site held by a symmetric enzyme. `is_unsym_motif` specifies the ligand that shouldn't be symmetrized.
- uncond_unsym_C3_6t8h:
 Unconditional generation of C3 proteins around a DNA helix. The DNA chains are the motifs. `is_unsym_motif` specifies the DNA strands that shouldn't be symmetrized.

```json
{
    "unindexed_C2_1j79": {
        "symmetry": {
            "id": "C2",
            "is_symmetric_motif": true
        },
        "input": "symmetry_examples/M0630_1j79_symmedORO.pdb",
        "ligand": "ORO,ZN",
        "unindex": "A250",
        "length": 130,
        "select_fixed_atoms": {
            "A250": "OD1,CG"
        }
    },
    "unindexed_C2_1e3v": {
        "symmetry": {
            "id": "C2",
            "is_symmetric_motif": true
        },
        "input": "symmetry_examples/M0349_1e3v.pdb",
        "ligand": "DXC",
        "unindex": "A16,A40,A100,A103",
        "length": 80,
        "select_fixed_atoms": {
            "A16": "OH,CZ,CE1,CE2",
            "A40": "OD2,CG",
            "A100": "N,CA,C,CB",
            "A103": "OD2,CG"
        }
    },
    "indexed_unsym_C2_1bfr": {
        "symmetry": {
            "id": "C2",
            "is_symmetric_motif": true,
            "is_unsym_motif": "HEM"
        },
        "input": "symmetry_examples/1bfr_C2.pdb",
        "ligand": "HEM",
        "contig": "51,M52,80",
        "length": null,
        "select_fixed_atoms": {
            "M52": "CG,SD,CE"
        }
    },
    "uncond_unsym_C3_6t8h": {
        "symmetry": {
            "id": "C3",
            "is_symmetric_motif": true,
            "is_unsym_motif": "Y1-11,Z16-25"
        },
        "input": "symmetry_examples/6t8h_C3.pdb",
        "contig": "150-150,/0,Y1-11,/0,Z16-25",
        "length": null
    }
}
```