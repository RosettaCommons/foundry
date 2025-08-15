# Inference with RF3

> **⚠️ Notice:** We are currently finalizing some cleanup work on the inference API. Please expect the API (including input formats and confidence outputs) to stabilize within the next 2 weeks. Thank you for your patience!

RF3 is an all-atom biomolecular structure prediction network competitive with leading open-source models. By including additional features at train-time – implicit chirality representations and atom-level geometric conditioning – we improve performance on tasks such as prediction of chiral ligands and fixed-backbone or fixed-conformer docking.

For more information, please see our preprint, [Accelerating Biomolecular Modeling with AtomWorks and RF3](https://doi.org/10.1101/2025.08.14.670328).

This guide provides instructions on preparing inputs and running inference for RF3. We will continue to update this document in the coming days and weeks, including support for arbitrary atom-level templating and more detailed examples of edge cases such as macrocyclic pepties, covalent modifications, and non-canonical amino acids.

## Step 0: Installation and Setup
### A. Installation using `uv`
```bash
git clone https://github.com/RosettaCommons/modelforge.git \
  && cd modelforge \
  && uv python install 3.12 \
  && uv venv --python 3.12 \
  && source .venv/bin/activate \
  && uv pip install -e .
```

### B. Download model weights for RF3 
```bash
wget http://files.ipd.uw.edu/pub/rf3/rf3_latest.pt
```

### C. Run a test prediction
```bash
rf3 fold tests/data/5vht_from_json.json
```

Ensure that a `.score` and a predicted `.cif.gz` are created in the specified directory, without error.

## Step 1: Prepare Inputs

RF3 accepts multiple input formats:
- `PDB/mmCIF` files (e.g., from RCSB, or from Ligand/ProteinMPNN)
- `json` file with each component specified within the RF3 input format (outlined below)
- `json` file in the AlphaFold-3 format *(coming soon)*
- Pickled `AtomArray` objects *(coming soon)*

`mmCIF` format are preferred when available, as they unambiguously specify the model inputs, with all details included (e.g., covalent bonds).

> **Note:** If you already have a `CIF` or `PDB` file, and do not want to include MSAs, you may proceed directly to Step 2.

If you do not have a pre-prepared CIF or PDB file, you may provide a combination of one-letter polymer sequences, SMILES strings, CCD codes, and SDF files through our JSON API. For example:

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
                "path": "/path/to/sdf.sdf"
            },
            {
                // We will use atom names from the CIF file
                "path": "/path/to/cif.cif"
            }

        ]
    }
]
```
The full API for inference via dictionaries of chemical components is specified in [atomworks.io](https://github.com/RosettaCommons/atomworks/blob/production/src/atomworks/io/tools/inference.py), additional contributions to support further formats (e.g., `mol` files as components) are welcome.

Supported input options:
-   `seq`: For proteins and nucleic acids using non-canonical one-letter codes as they appear in a CIF file
-   `smiles`: For small molecules (ensure correctness of SMILES, and proper indication of chirality when applicable)
-   `ccd_code`: If your small molecule is already in the CCD
-   `path`: If you have a `.sdf` or `.cif` file (including `.cif` files for small molecules)

## Step 2: Run `rf3 fold`

We can now run inference with:
```bash
rf3 fold tests/data/5vht_from_file.cif
```
or, alternatively,
```bash
rf3 fold tests/data/5vh5_from_json.json
```

For our inference API, we use [hydra](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/) to prepare arguments; the [documentation](https://hydra.cc/docs/advanced/override_grammar/basic/) describes the command-line override syntax that we use below. Note that Hydra syntax differes from typical CLI or `argparse` syntax in that we don't use `--arg value`, but instead `arg=value`. See below for examples.

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

#### Advanced structural control arguments
- `template_selection_syntax` *(optional)*: Coming soon.
- `ground_truth_conformer_selection` *(optional)*: Selection syntax for residues that should use ground truth conformers instead of generated ones. Uses AtomSelection format (e.g., "*/HEM" for all heme residues, "A1-10" for residues 1-10 in chain A, "*/ATP" for all ATP molecules). If None, no residues will use ground truth conformers. This is useful for keeping known ligand conformations while allowing the model to predict protein structure around them.

#### Arguments to control output dumping
- `out_dir` *(optional)*: Where to save predicted structures. The output files will be named the same as the input structures, or use the `name` field in the specification, if present. Defaults to the current directory (`./`).
- `dump_predictions` *(optional)*: Whether to save outputs as CIF files (vs. only the `.score` file). Defaults to True.
- `dump_trajectories` *(optional)*: Whether to dump the denoising trajectories. Defaults to False. Denoising trajectories are memory- and CPU-intensive to save to disk; we do not suggest dumping them except for a select few structures, if needed.
- `one_model_per_file` *(optional)*: Whether to save multiple structures from one diffusion batch as separate models within the same file or separate files. Defaults to False (one file with multiple models).
- `annotate_b_factor_with_plddt` *(optional)*: Whether to annotate atom-level pLDDT, overwriting the `b_factor` column in the CIF output (full name is `b_iso_or_equiv` in mmCIF files). Defaults to False. NOTE: If set to True, then `one_model_per_file` will be automatically set to True (since our CIF-saving software `biotite` does not support variable `b_factors` across models within the same file).

> *NOTE:* The CIF files are saved in a compressed format, `.cif.gz`. These compressed files can be directly loaded by PyMol or parsed by `atomworks.ml`. If you need to inspect the uncompressed file, you can use `gunzip <PATH>`. 

> *NOTE:* The CIF output file will contain multiple **models**, one for each diffusion outputs (e.g., 5 by default). PyMol will hide secondary structure by default with multiple models; the command `dss` will display it again.

Example commands:

### Using a JSON with multiple examples to predict
```bash
rf3 fold inference_engine=af3 inputs='tests/data/multiple_examples_from_json.json'
```

### Using a PDB, specifying a covalent modification in the `CONECT` record
See line `1672` for the manually-added bond; note as well the renaming of the ligand. Such renaming could be accomplished *a-priori* by modifying the file (as in this example), or with the `rename_residues` flag (see below).
```bash
rf3 fold inference_engine=af3 inputs='projects/ml/modelhub/inference/example_from_pdb_with_inter_chain_bond.pdb'
```

### Using a PDB from MPNN, renaming custom ligand that overlaps with ligand names in the CCD 
Note that in this PDB file, the ligand "HGS" is a custom ligand, whose three-letter code overlaps with a real CCD ligand. Thus, we must rename in order to avoid errors.
```bash
rf3 fold inference_engine=af3 inputs='/projects/ml/modelhub/inference/example_pdb_with_clashing_ligand_name.pdb' rename_residues="{'HGS': 'L:1'}"  
```

## Chirality
If inputs are given in a form that specifies chirality, the model will receive the corresponding features and attempt to preserve the chirality of the inputs.

Chiral formats include:
- SMILES strings (e.g., using `@`)
- CIF, PDB, or SDF files with non-zero coordinates

## Step 3: View the Predicted Structure(s)

Use the following code to view the predicted structures with `atomworks.ml`:

```python
from atomworks.ml.utils.visualize import view
from atomworks.ml import parse

# View in atomworks (or PyMol, etc.)
out = parse("path/to/prediction.cif.gz")
atom_array = out["assemblies"]["1"][0]
# (If in a notebook)
view(atom_array)
```

View in PyMol like normal, or using `pymol_remote`
