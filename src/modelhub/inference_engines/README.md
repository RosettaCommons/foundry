# Inference with `modelhub-AF3` repository

We have reproduced AF3 and are sharing the weights with the lab to use for various tasks. 
This guide provides instructions on preparing inputs and running inference for our AF3 reproduction.

## Step 1: Prepare Inputs

> **Note:** If you already have a `CIF` or `PDB` file (e.g., from MPNN), and do not want to include MSAs, you may proceed directly to Step 2.

We enumerate two options for preparing inputs: one with a JSON API, one by creating an `AtomArray` to spoof a CIF.

### Option 1: Prepare inputs using a combination of one-letter polymer sequences, SMILES strings, CCD codes, and SDF files

Create a JSON file with each component; e.g.,

```json
[
    {
        "name": "my_example",
        "components": [
            {
                "seq": "SMNPPPPETSNPNKPKRQTNQLQYLLRVVLKTLWKHQFAWPFQQPVDAVKLNLPDYYKIIKTPMDMGTIKKRLENNYYWNAQECIQDFNTMFTNCYIYNKPGDDIVLMAEALEKLFLQKINELPTEE",
                "msa_path": "/path/to/msa", // optional
                "chain_id": "A"
            },
            {
                // We will automatically name the atoms
                // If no `chain_id` is specified, we will deterministically generate one (e.g., "B", since "A" exists above)
                "smiles": "NCCCCN1N=C(C[C@@H](C1=O)c2cccc3ncccc23)c4ccc(NC(=O)N5Cc6ccncc6C5)cc4"
            },
            {
                // We will use atom names from the CCD
                "ccd_code": "HEM"
            },
            {
                // We will automatically name the atoms (SDF files do not specify atom names)
                "path": "/path/to/sdf.sdf", 
                "override_reference_conformer": true // We will replace the RDKit-generated conformer with the ground-truth
            },
            {
                // We will use atom names from the CIF file
                "path": "/path/to/cif.cif"
            }

        ]
    }
]
```
The full API for inference via dictionaries of chemical components is specified in [CIFUtils](https://github.com/baker-laboratory/cifutils/blob/main/src/cifutils/tools/inference.py); additional contributions to support further formats (e.g., `MOL` files , as components) are welcome and relatively straight-forward to implement.

Supported input options:
-   `seq`: For proteins and nucleic acids using non-canonical one-letter codes as they appear in a CIF file.
-   `smiles`: For small molecules (ensure correctness of SMILES).
-   `ccd_code`: If your small molecule is already in the CCD.
-   `path`: If you have a `.sdf` or `.cif` file.

Coming soon: support for `mol` files as components.

### Option 2: Using a Spoofed CIF *(more complicated, more customizable)*

If you can get your inputs into an `AtomArray`, use `to_cif_file` to convert the `AtomArray` to a `CIF`. Use the pre-built inference tools in `cifutils` to convert arbitrary biological inputs (e.g., FASTA, CIFs, SMILES) into an `AtomArray`. See [cifutils tests](https://github.com/baker-laboratory/cifutils/blob/main/tests/tools/test_inference_processing.py) for examples.

#### Example Code

```python
import os
os.environ['CCD_MIRROR_PATH'] = "/projects/ml/frozen_pdb_copies/2024_12_11_ccd"
os.environ['PDB_MIRROR_PATH'] = "/projects/ml/frozen_pdb_copies/2024_12_01_pdb"

from cifutils.tools.inference import components_to_atom_array
from cifutils.utils.io_utils import to_cif_file

# Define inputs as a list of dictionaries
monomer = {
    "seq": "SMNPPPPETSNPNKPKRQTNQLQYLLRVVLKTLWKHQFAWPFQQPVDAVKLNLPDYYKIIKTPMDMGTIKKRLENNYYWNAQECIQDFNTMFTNCYIYNKPGDDIVLMAEALEKLFLQKINELPTEE",
    "chain_type": "polypeptide(l)",
    "chain_id": "A",
}

ligand_from_smiles = {
    "smiles": "NCCCCN1N=C(C[C@@H](C1=O)c2cccc3ncccc23)c4ccc(NC(=O)N5Cc6ccncc6C5)cc4",
    "chain_id": "C",
}

ligand_from_ccd = {
    "ccd_code": "7Z2",
    "chain_id": "C",
}

# Convert to AtomArrays and write to CIF files
atom_array_from_ccd = components_to_atom_array([monomer, ligand_from_ccd])
atom_array_from_smiles = components_to_atom_array([monomer, ligand_from_smiles])

to_cif_file(atom_array_from_ccd, "example_from_ccd.cif")
to_cif_file(atom_array_from_smiles, "example_from_smiles.cif")
```

## Step 2: Run `inference.py`

The apptainers that we release pre-install `modelhub`, `datahub`, and `cifutils`. Note that this abstraction also means that these apptainers are not "hackable" — if you would like to modify `modelhub`, you'll need to clone the repository, and use the development apptainer (see the main `README`).

For our inference API, we use [hydra](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/) to prepare arguments; the [documentation](https://hydra.cc/docs/advanced/override_grammar/basic/) describes the command-line override syntax that we use below. Note that Hydra syntax differes from typical CLI or `argparse` syntax in that we don't use `--arg value`, but instead `arg=value`. See below for examples.

### Using an Existing CIF or PDB File

Run the apptainer with the appropriate apptainer, checkpoint, input directory, and output directory.

Arguments to `inference.py` (which the apptainer calls behind-the-scenes):
#### Basic Arguments
- `inputs` *(required)*: Path to a file (CIF/PDB/JSON) or list of files for prediction; if given a directory, all CIF/PDB files in that directory will be predicted. To specify a list of files/directories, use Hydra's list grammar: `foo="[path_1.cif, path_2.json, path_3.pdb]"`
- `inference_engine` *(required)*: The inference configuration to use. For example, `af3`, to use the standard structure prediction model. We will introduce other configurations down-the-line, each with unique use cases.
- `ckpt_path` *(optional)*: Path to checkpoint file. Defaults to the current "best model", which is stored in a symlink in `/net/software`
- `residue_renaming_dict` *(optional)* Dictionary of residues to rename to avoid CCD clashes, given in Hydra format (e.g., `foo="{'ALA': 'L:1'}`). When parsing files, we use the given residue names to help identify any missing atoms. Thus, if a custom ligand overlaps with a ligand in the CCD, the prediction will be catastrophically wrong. To circumvent this issue, we accept a dictionary of ligands to rename. We suggest renaming all custom ligands to begin with `L:` to avoid all clashes with the CCD. WARNING: This command uses brute-force find a replace; please ensure that there are no other possible matches (e.g., atom names). Additionally, avoid `#` to mitigate possible CIF-parsing errors from PyMol. Defaults to None.
- `skip_existing` *(optional)*: Whether to skip predictions where appropriately-named output structures already exist in the `out_dir`. Defaults to False (do not skip; overwrite instead).

#### Arguments to control the model trunk and diffusion sampling
- `early_stopping_plddt_threshold` *(optional)*. The average all-atom pLDDT value estimated after a single recycle that will trigger early-exit for that prediction. Defaults to `0.5`. Using this flag can **significantly** increase structure throughput (10-20x). If we early exit:
    * There will be no output structure files (`.cif.gz`)    
    * The `.score` file will contain a field `early_stopped` that will have the value `True`; it will also contain columns indicating the value of the all-atom pLDDT after the first recycle and the threshold applied.
- `n_recycles` *(optional)*: Number of recycles within the trunk. Defaults to 10.
- `diffusion_batch_size` *(optional)*: Number of output structures in the ensemble, drawn from the same model seed and forward pass of the Pairformer. Defaults to 5.
- `num_steps` *(optional)* Number of steps for sampling of the diffusion module. The standard is 200; we see no deterioration in performance with 50 steps, but significant (>2x) speed improvements. Defaults to 200.
- `seed`  *(optional)* Model seed. Running inference multiple times with different model seeds is the best, and most expensive, way to generate output diversity. Defaults to the training seed (usually 42).

#### Arguments to control output dumping
- `out_dir` *(optional)*: Where to save predicted structures. The output files will be named the same as the input structures, or use the `name` field in the specification, if present. Defaults to the current directory (`./`).
- `dump_predictions` *(optional)*: Whether to save outputs as CIF files (vs. only the `.score` file). Defaults to True.
- `dump_trajectories` *(optional)*: Whether to dump the denoising trajectories. Defaults to False. Denoising trajectories are memory- and CPU-intensive to save to disk; we do not suggest dumping them except for a select few structures, if needed.
- `one_model_per_file` *(optional)*: Whether to save multiple structures from one diffusion batch as separate models within the same file or separate files. Defaults to False (one file with multiple models).
- `annotate_b_factor_with_plddt` *(optional)*: Whether to annotate atom-level pLDDT, overwriting the `b_factor` column in the CIF output (full name is `b_iso_or_equiv` in mmCIF files). Defaults to False. NOTE: If set to True, then `one_model_per_file` will be automatically set to True (since our CIF-saving software `biotite` does not support variable `b_factors` across models within the same file).

> *NOTE:* The CIF files are saved in a compressed format, `.cif.gz`. These compressed files can be directly loaded by PyMol or parsed by `cifutils`. If you need to inspect the uncompressed file, you can use `gunzip <PATH>`. 

> *NOTE:* The CIF output file will contain multiple **models**, one for each diffusion outputs (e.g., 5 by default). PyMol will hide secondary structure by default with multiple models; the command `dss` will display it again.

Example commands (to be run from the `inference` working directory):

### Using a JSON with multiple examples to predict
```bash
apptainer -s run --nv /net/software/containers/versions/modelhub_inference/modelhub_latest.sif inference_engine=af3 inputs='/projects/ml/modelhub/inference/examples_from_json.json'
```

### Using a PDB, specifying a covalent modification in the `CONECT` record (*example from Meg)*
See line `1672` for the manually-added bond; note as well the renaming of the ligand. Such renaming could be accomplished *a-priori* by modifying the file (as in this example), or with the `rename_residues` flag (see below).
```bash
apptainer -s run --nv /net/software/containers/versions/modelhub_inference/modelhub_latest.sif inference_engine=af3 inputs='projects/ml/modelhub/inference/example_from_pdb_with_inter_chain_bond.pdb'
```

### Using a PDB from MPNN, renaming custom ligand that overlaps with ligand names in the CCD *(example from Indrek)*
Note that in this PDB file, the ligand "HGS" is a custom ligand, whose three-letter code overlaps with a real CCD ligand. Thus, we must rename.
```bash
apptainer -s run --nv /net/software/containers/versions/modelhub_inference/modelhub_latest.sif inference_engine=af3 inputs='projects/ml/modelhub/inference/example_from_pdb_with_inter_chain_bond.pdb'
apptainer -s run --nv /net/software/containers/versions/modelhub_inference/modelhub_latest.sif inference_engine=af3 inputs='/projects/ml/modelhub/inference/example_pdb_with_clashing_ligand_name.pdb' rename_residues="{'HGS': 'L:1'}"  
```

### Using a PDB, providing some fraction of the input structure as a template
This feature uses similar syntax to contigs in RFdiffusion. You can select regions to be templated by specifying comma separated stretches of residues in your input PDB (e.g. "A1-71,B1-1"; only works for proteins right now).
```bash
apptainer -s run --nv /net/software/containers/users/ncorley/modelhub/inference_latest.sif inference_engine=af3 inputs='projects/ml/modelhub/tests/data/5vht_from_file.cif' template_selection_syntax="A1-71"
```

## Chirality
If inputs are given in a form that specifies chirality, the model will receive the corresponding features and attempt to preserve the chirality of the inputs.

Chiral formats include:
- SMILES strings (e.g., using `@`)
- CIF, PDB, or SDF files with non-zero coordinates

## Checkpoints
Multiple checkpoints are available for inference, each with their own strengths and weaknesses. We provide a default option; however, the `ckpt_path` argument is exposed to provide additional control (`ckpt_path=/net/software...`).

| Name | Path | Description |
|:----------------|:-----|:------------|
| Epoch 804 (Default) | `⁠/net/software/containers/versions/modelhub_inference/ckpts/modelhub_af3_with_confidence_ep804.ckpt` | Default checkpoint. Best protein-ligand performance. |
| Epoch 826 | `⁠/net/software/containers/versions/modelhub_inference/ckpts/modelhub_af3_with_confidence_ep826.ckpt` | Subsequent checkpoint that exhibits stronger performance on protein-DNA structures, at the cost of protein-ligand metrics. |

## Step 3: View the Predicted Structure(s)

Use the following code to view the predicted structures with `cifutils`:

```python
from cifutils.utils.visualize import view
from cifutils import parse

# View in CIFUtils (or PyMol, etc.)
out = parse("path/to/prediction.cif.gz")
atom_array = out["assemblies"]["1"][0]
view(atom_array)
```

View in PyMol like normal, or using `pymol_remote`
