# Inference with `modelhub-AF3` repository

We have reproduced AF3 and are sharing the weights with the lab to use for various tasks. 
This guide provides instructions on preparing inputs and running inference for our AF3 reproduction.

Additional variations (e.g., with chirality inputs, ligand geometry conditioning, protein backbone coordinate conditioning) are in-the-works; however, the core inference API will not change.

## Step 1: Prepare Inputs

> **Note:** If you already have a `CIF` or `PDB` file (e.g., from MPNN), and do not want to include MSAs, you may proceed directly to Step 2.

We enumerate two options for preparing inputs: one with a JSON API, one by creating an `AtomArray` to spoof a CIF.

### Option 1: Prepare inputs using a combination of one-letter polymer sequences, SMILES strings, CCD codes, and SDF files

Create a JSON file with each component; e.g.,

```json
[
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
        // We will use atom names from the SDF file
        "path": "/path/to/sdf.sdf"
    }
]
```
The full API for inference via dictionaries of chemical components is specified in [CIFUtils](https://github.com/baker-laboratory/cifutils/blob/main/src/cifutils/tools/inference.py); additional contributions to support further formats (e.g., `MOL` files and `CIF` files, as components) are welcome and relatively straight-forward to implement.

Supported input options:
-   `seq`: For proteins and nucleic acids using non-canonical one-letter codes as they appear in a CIF file.
-   `smiles`: For small molecules (ensure correctness of SMILES).
-   `ccd_code`: If your small molecule is already in the CCD.
-   `path`: If you have a `.sdf` file. Note that we will not (yet) use the coordinates from the `.sdf` file for the reference conformer (but that's in-the-works).

Coming soon: support for `cif` files and `mol` files as components.

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

## Step 2: Run `run_inference.py`

The apptainers that we release pre-install `modelhub`, `datahub`, and `cifutils`. That means in order to run inference, essentially all that is needed is `from modelhub.inference import run_inference`, `run_inference()`. For convenience, we have written a script with that functionality, and saved to `/projects/ml/modelhub/inference/run_inference.py`. Note that this also means these apptainers are not "hackable" — if you would like to modify `modelhub`, you'll need to clone the repository, and use an appptainer without `modelhub` pre-installed.

### Using an Existing CIF or PDB File

Run `run_inference.py` with the appropriate apptainer, checkpoint, input directory, and output directory.

Arguments to `run_inference.py` (and thus `inference.py`, which is called by `run_inference.py`) are:
- `inputs` (required): Path to a file (CIF/PDB/JSON) for prediction; if given a directory, all CIF/PDB files in that directory will be predicted
- `--checkpoint-path` (required): Path to checkpoint file
- `--cif_out_dir` (required): Where to save predicted structures. The output files will be named the same was as the input structures. Use `./` for current directory.
- `--n_recycles` (optional, defaults to 10): Number of recycles.
- `--diffusion_batch_size` (optional, default to 5): Number of output structures in the ensemble, drawn from the same model seed and forward pass of the Pairformer.
- `--rename_residues` (optional, default to an empty string): Dictionary of residue names to rename to avoid CCD clashes, e.g., '{"ALA": "L:1"}'. When parsing files, we use the given residue names to help identify any missing atoms. Thus, if a custom ligand overlaps with a ligand in the CCD, the prediction will be catastrophically wrong. To circumvent this issue, we accept a dictionary of ligands to rename. We suggest renaming all custom ligands to begin with `L:` to avoid all clashes with the CCD. WARNING: This command uses brute-force find a replace; please ensure that there are no other possible matches (e.g., atom names). Additionally, avoid `#` to mitigate possible CIF-parsing errors from PyMol.
- `num_steps` (optiona, default to 200)L Number of steps for sampling of the diffusino model. The default is 200. We see no deterioation in performance with 50 steps, but significant (>2x) speed improvements.

> *NOTE:* The CIF files are saved in a compressed format, `.cif.gz`. These compressed files can be directly loaded by PyMol or parsed by `cifutils`. If you need to inspect the uncompressed file, you can use `gunzip <PATH>`. 

> *NOTE:* The CIF output file will contain multiple **models**, one for each diffusion outputs (e.g., 5 by default). PyMol will hide secondary structure by default with multiple models; the command `dss` will display it again.

Example commands (to be run from the `inference` working directory):

### Using a JSON with multiple examples to predict
```bash
apptainer -s run --nv /net/software/containers/users/ncorley/modelhub/frozen_modelhub_datahub_cifutils_2025-02-06.sif python /projects/ml/modelhub/inference/run_inference.py /projects/ml/modelhub/inference/examples_from_json.json --checkpoint_path /projects/ml/modelhub/inference/weights_with_confidence_2025_01_06 --cif_out_dir ./
```

### Using a PDB, specifying a covalent modification in the `CONECT` record (*example from Meg)*
See line `1672` for the manually-added bond; note as well the renaming of the ligand. Such renaming could be accomplished *a-priori* by modifying the file (as in this example), or with the `rename_residues` flag (see below).
```bash
apptainer -s run --nv /net/software/containers/users/ncorley/modelhub/frozen_modelhub_datahub_cifutils_2025-02-06.sif python /projects/ml/modelhub/inference/run_inference.py /projects/ml/modelhub/inference/example_from_pdb_with_inter_chain_bond.pdb --checkpoint_path /projects/ml/modelhub/inference/weights_with_confidence_2025_01_06 --cif_out_dir ./
```

### Using a PDB from MPNN, renaming custom ligand that overlaps with ligand names in the CCD *(example from Indrek)*
Note that in this PDB file, the ligand "HGS" is a custom ligand, whose three-letter code overlaps with a real CCD ligand. Thus, we must rename.
```bash
apptainer -s run --nv /net/software/containers/users/ncorley/modelhub/frozen_modelhub_datahub_cifutils_2025-02-06.sif python /projects/ml/modelhub/inference/run_inference.py /projects/ml/modelhub/inference/example_pdb_with_clashing_ligand_name.pdb --checkpoint_path /projects/ml/modelhub/inference/weights_with_confidence_2025_01_06 --cif_out_dir ./ --rename_residues '{"HGS": "L:1"}'  
```

## Step 3: View the Predicted Structure(s)

Use the following code to view the predicted structures with `cifutils`:

```python
from cifutils.utils.visualize import view
from cifutils import parse

# View in CIFUtils (or PyMol, etc.)
out = parse("./predictions/json_inputs_0.cif")
atom_array = out["assemblies"]["1"][0]
view(atom_array)
```

View in PyMol like normal, or using `pymol_remote`
