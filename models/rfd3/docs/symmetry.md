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
- `id`                : Symmetry group ID. Supported symmetry types:
  - **Cyclic (C)**: e.g. "C3" for a cyclic protein with 3 subunits
  - **Dihedral (D)**: e.g. "D2" for a dihedral protein with 4 subunits (2×2)
  - **Tetrahedral (T)**: "T" for tetrahedral symmetry with 12 subunits
  - **Octahedral (O)**: "O" for octahedral symmetry with 24 subunits
  - **Icosahedral (I)**: "I" for icosahedral symmetry with 60 subunits
- `is_unsym_motif`    : Comma separated string list of contig/ligand names that should NOT be symmetrized (e.g. DNA strands). If not provided, all motifs are assumed to be symmetrized. See [Designs with motifs](#designs-with-motifs) section for details.
- `is_symmetric_motif`: Boolean value whether the input motif is symmetric. Currently only symmetric input motifs are supported, therefore, `true` by default.

> **⚠️ Memory Warning:** Memory requirements scale quadratically with the number of subunits. For larger complexes (especially T, O, and I symmetries), memory usage can become very high. Always use `diffusion_batch_size=1` for symmetry, and consider enabling `low_memory_mode=True` for higher-order symmetries. The memory footprint increases dramatically as both the number of subunits and the length of each subunit increase. Note: many higher order symmetires run out of memory even on large cards such as H200s, we are working on optimizations to make these networks more memory efficient. 


## Example command 
You can run the following example command:
```
./src/modelhub/inference.py inference_sampler.kind=symmetry out_dir=logs/inference_outs/sym_demo/0 ckpt_path=$cur_ckpt inputs=./projects/aa_design/tests/test_data/sym_tests.json diffusion_batch_size=1 
```
- `inference_sampler.kind`: Set `symmetry` to turn on symmetry mode.
- `diffusion_batch_size`  : **Must be set to `1` for all symmetry types** due to memory limitations. Memory scales quadratically with the number of subunits.
- `low_memory_mode`       : **Strongly recommended** for T, O, and I symmetries. Set to `True` if you encounter memory errors (e.g. "CUDA error: out of memory"). Note that this will significantly slow the inference but is often necessary for larger complexes.


## Unconditional multimer design

The following provides a general overview of the supported symmetry types and examples of how to run:

### Cyclic (C)
Cyclic symmetry with n-fold rotational symmetry around a single axis. Generates n identical subunits.

```json
{
    "uncond_C15": {
        "length": 100,
        "is_non_loopy": true,
        "symmetry": {
            "id": "C15"
        }
    }
}
```

### Dihedral (D)
Dihedral symmetry combines n-fold rotational symmetry with a 2-fold rotation perpendicular to the main axis. Generates 2n identical subunits.

```json
{
    "uncond_D4": { 
        "length": 100,
        "is_non_loopy": true,
        "symmetry": {
            "id": "D4"
        }
    }
}
```

### Tetrahedral (T)
Tetrahedral symmetry based on the symmetry of a tetrahedron. Generates 12 identical subunits.

```json
{
    "uncond_T": { 
        "length": 100,
        "symmetry": {
            "id": "T"
        }
    }
}
```

### Octahedral (O)
Octahedral symmetry based on the symmetry of a cube/octahedron. Generates 24 identical subunits.

```json
{
    "uncond_O": { 
        "length": 100,
        "symmetry": {
            "id": "O"
        }
    }
}
```

### Icosahedral (I)
Icosahedral symmetry based on the symmetry of an icosahedron/dodecahedron. Generates 60 identical subunits. This is the largest point group symmetry and is commonly found in viral capsids.

```json
{
    "uncond_I": { 
        "length": 100,
        "symmetry": {
            "id": "I"
        }
    }
}
```

> **⚠️ Memory Warning:** Memory requirements increase dramatically with the number of subunits and subunit length. The memory footprint scales approximately quadratically with the number of subunits due to pairwise interactions. For higher-order symmetries (T: 12 subunits, O: 24 subunits, I: 60 subunits), it is **essential** to:
> - Set `diffusion_batch_size=1` (required for all symmetry types)
> - Enable `low_memory_mode=True` for T, O, and I symmetries
> - Consider reducing subunit length if you encounter out-of-memory errors
> - For icosahedral symmetry (60 subunits), expect memory usage to be 25-100× higher than a single chain design

## Designs with motifs

Symmetry sampling currently only supports pre-symmetrized motifs around the origin. Therefore, `is_symmetric_motif` is set to `true` by default. 
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
    "unsym_C3_6t8h": {
        "symmetry": {
            "id": "C3",
            "is_symmetric_motif": true,
            "is_unsym_motif": "Y1-11,Z16-25"
        },
        "input": "symmetry_examples/6t8h_C3.pdb",
        "contig": "150-150,/0,Y1-11,/0,Z16-25",
        "length": null,
        "is_non_loopy": true
    }
}
```