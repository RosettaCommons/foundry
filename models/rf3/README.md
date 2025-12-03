# Inference with RosettaFold3(RF3)

<div align="center">
  <img src="../../docs/_static/prot_dna.png" alt="Protein-DNA complex prediction" width="400">
</div>

> [!IMPORTANT]
> We are currently finalizing some cleanup work on the inference API. Please expect the API (including input formats and confidence outputs) to stabilize in the upcoming weeks. Thank you for your patience!

RF3 is an all-atom biomolecular structure prediction network competitive with leading open-source models. By including additional features at train-time – implicit chirality representations and atom-level geometric conditioning – we improve performance on tasks such as prediction of chiral ligands and fixed-backbone or fixed-conformer docking.

For more information, please see our preprint, [Accelerating Biomolecular Modeling with AtomWorks and RF3](https://doi.org/10.1101/2025.08.14.670328).

This guide provides instructions on preparing inputs and running inference for RF3. 

##  Installation, Setup, and a Basic Prediction
### A. Installation using `uv`
```bash
git clone https://github.com/RosettaCommons/modelforge.git \
  && cd modelforge \
  && uv python install 3.12 \
  && uv venv --python 3.12 \
  && source .venv/bin/activate \
  && uv pip install -e ".[rf3]"
```

### B. Download model weights for RF3

We offer three model checkpoints:

| Model | Training Cutoff | Description |
|-------|-----------------|-------------|
| **Latest** | 01/24 | Most recent model with bugfixes and improvements from preprint |
| **Preprint** | 01/24 | Original preprint model |
| **Benchmark** | 09/21 | For benchmarking against other models with the same date cutoff |

**Latest (recommended):**
```bash
wget http://files.ipd.uw.edu/pub/rf3/rf3_foundry_01_24_latest.ckpt
```

**Preprint:**
```bash
wget http://files.ipd.uw.edu/pub/rf3/rf3_foundry_01_24_preprint.ckpt
```

**Benchmark (09/21):**
```bash
wget http://files.ipd.uw.edu/pub/rf3/rf3_foundry_09_21_preprint.ckpt
```

> [!NOTE]
> The inference API is identical across all checkpoints.

### C. Run a test prediction
```bash
rf3 fold inputs='tests/data/5vht_from_json.json'
```

You may then specify the specific checkpoint, if desired, with:
```bash
rf3 fold inputs='tests/data/5vht_from_json.json' ckpt_path='/path/to/rf3_921.pt'
```

> [!NOTE]
> For our inference API, we use [hydra](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/) to prepare arguments; the [Hydra documentation](https://hydra.cc/docs/advanced/override_grammar/basic/) describes the command-line override syntax that we use below. Note that Hydra syntax differes from typical CLI or `argparse` syntax in that we don't use `--arg value`, but instead `arg=value`. See below for examples.

> [!TIP]
> Rather than `rf3 fold`, you may also directly use `python src/foundry/inference.py inputs='tests/data/5vht_from_json.json'` to interface with the Hydra entrypoint. This approach may yield more informative error messages in some cases.

From the above command, you should see several outputs:
- `5vht_from_json_metrics.csv` — overall confidence metrics for this example
- `5vht_from_json.score` - more granular confidence metrics for this example
- `5vht_from_json_model_0.cif.gz` - zipped model prediction for the first diffusion seed (PyMol can open `.gz` files directly)
- `5vht_from_json_model_1.cif.gz` - zipped model prediction for the second diffusion seed
- ...

For this example, the pTM in the `metrics.csv` should be `>0.8` (even without an MSA); if not, there may be something wrong with your setup.

## Common Scenarios

<details>
<summary><strong>Folding with an MSA</strong></summary>

RF3 supports `.a3m` and `.fasta` files as input MSA formats; `.a3m` is recommended. We do not at the moment support pre-paired MSAs (we will pair on-the-fly) or on-the-fly MSA computation, but both are on the roadmap. Please raise an issue if these limitations are critical for your project and we can prioritize accordingly.

📝 **Example JSON configuration** (full example found at `../../docs/releases/rf3/examples/3en2_from_json_with_msa.json`):

```json
{
    "name": "example_with_msa",
    "components": [
        {
            "seq": "AINRLQLVATLVEREV(MSE)RYTPAGVPIVNCLLSYSGQA(MSE)EAQAARQVEFSIEALGAGK(MSE)ASVLDRIAPGTVLECVGFLARKHRSSKALVFHISGLEHHHHHH",
            "msa_path": "/path/to/msa.a3m",
            "chain_id": "A"
        }
    ]
}
```

🚀 **Run the example:**

```bash
rf3 fold inputs='../../docs/releases/rf3/examples/3en2_from_json_with_msa.json'
```

---

If performing inference from a prepared `.cif` file, MSAs can also be specified directly as a category within the raw CIF data.
We will automatically extract the correct MSA paths during parsing.

📝 **Example CIF header** (full example found at `../../docs/releases/rf3/examples/3en2_from_file.cif`):
```cif
data_3EN2
#
_msa_paths_by_chain_id.A   ../../docs/releases/rf3/examples/msas/b3a35202064.a3m.gz
_msa_paths_by_chain_id.B   ../../docs/releases/rf3/examples/msas/b3a35202064.a3m.gz
# 
```

🚀 **Run the example:**

```bash
rf3 fold inputs='docs/rf3/3en2_from_file.cif
```

> [!TIP]
> Without an MSA and using default settings, the above examples will trigger "early stopping." This means that if the model determines early on that a correct prediction is unlikely, it will stop computation and only output a `metrics.csv` and `.score` file to save compute resources. You can adjust this behavior using the `early_stopping_plddt_threshold` argument (see below). In our group, we find this argument can save wasted compute on erroneous inputs.

> [!TIP]
> To ensure that a provided MSA is loaded correctly, you may use the `raise_if_missing_msa_for_protein_of_length_n` command-line argument. For example, `rf3 fold inputs='../../docs/releases/rf3/examples/3en2_from_json_with_msa.json' raise_if_missing_msa_for_protein_of_length_n=10` would raise an error if there were any proteins >=10 residues without compatible MSAs.

> [!TIP]
> For non-canonical amino acids, most MSA generation algorithms substitute `X` (unknown residue)! Ensure your MSAs adhere to this convention.

</details>

<details>
<summary><strong>Folding Many Inputs</strong></summary>

When running multiple predictions, you'll notice that the **startup cost** (importing libraries, loading models, initializing CUDA) often takes significantly longer than the actual prediction itself. This is especially problematic when running many individual `rf3 fold` commands in sequence.

💡 **Solution: Batch Processing**

Instead of running separate commands, you can process multiple inputs in a single command to amortize the startup cost. Multiple inputs can be provided in three ways:

1. **Within a single JSON file** - Define multiple examples in one configuration
2. **Via the CLI `inputs` argument** - Specify multiple files using Hydra list syntax  
3. **Via a folder of compatible inputs** - Process all CIF/PDB/JSON files in a directory

We will automatically distribute predictions across GPU's if running in a multi-GPU environment.

---

### 1️⃣ **Single JSON with Multiple Examples**

📝 **Example JSON configuration** (full example found at `../../docs/releases/rf3/examples/multiple_example_from_json.json`)

```json
[
    {
        "name": "multiple_examples_from_json(1)",
        "components": [
            {
                "seq": "MNAKEIVVHALRLLENGDARGWCDLFHPEGVLEYPYPPPGYKTRFEGRETIWAHMRLFPEYMTIRFTDVQFYETADPDLAIGEFHGDGVHTVSGGKLAADYISVLRTRDGQILLYRLFFNPLRVLEPLGLEHHHHHH",
                "chain_id": "A"
            },
            {
                "smiles": "O=C1OCC(=C1)C5C4(C(O)CC3C(CCC2CC(O)CCC23C)C4(O)CC5)C"
            }
        ]
    },
    {
        "name": "multiple_examples_from_json(2)",
        "components": [
            {
                "seq": "GSGVSLGQALLILSVAALLGTTVEEAVKRALWLKTKLGVSLEQAARTLSVAAYLGTTVEEAVKRALKLKTKLGVSLEQALLILFAAAALGTTVEEAVKRALKLKTKLGVSLEQALLILWTAVELGTTVEEAVKRALKLKTKLGVSLGQAQAILVVAAELGTTVEEAVYRALKLKTKLGVSLGQALLILEVAAKLGTTVEEAVKRALKLTTKLG",
                "chain_id": "A"
            },
            {
                "ccd_code": "MG"
            }
        ]
    }
]
```

🚀 **Run the example:**

```bash
rf3 fold inputs='../../docs/releases/rf3/examples/multiple_examples_from_json.json'
```

---

### 2️⃣ **Multiple Files via CLI Arguments**

You can specify multiple files/directories using Hydra's list syntax:

```bash
rf3 fold inputs='[../../docs/releases/rf3/examples/701r_from_file.cif, ../../docs/releases/rf3/examples/701r_from_json.json]'
```

---

### 3️⃣ **Directory Processing**

Process all CIF/PDB/JSON files in a directory at once (we will distribute across GPU's if running within a multi-GPU environment):

```bash
rf3 fold inputs='docs/rf3/examples'
```

> [!TIP]
> **Performance Tip**: For large batches, consider using `early_stopping_plddt_threshold=0.5` or lower to quickly filter out low-confidence predictions and save compute time on obviously incorrect structures.

> [!NOTE]
> All outputs will be saved in the same directory (or `out_dir` if specified), with filenames based on the `name` field in your JSON or the original input filenames.

</details>

<details>
<summary><strong>Folding with Arbitrary Biomolecules</strong></summary>
<a id="folding-with-arbitrary-biomolecules"></a>

Complex assemblies containing arbitrary biomolecules can be easily folded if prepared within a `.cif` or a `.pdb` file that adheres to RCSB conventions.

🚀 **Run an example from a prepared CIF file:**
```bash
rf3 fold inputs='../../docs/releases/rf3/examples/7o1r_from_file.cif'
```

Such files (including all bonds, covalent modifications, non-canonical amino acids, etc.) can be created either (a) directly from ProteinMPNN/LigandMPNN or other software that generates structural files; or, (b) assembled with [AtomWorks](https://github.com/RosettaCommons/atomworks) or another CIF-processing library.

For convenience, we also support a `json` API analogous to that implemented by AF3. We omit covalent modifications below as those are described in a subsequent example.

> [!TIP]
> **Performance Tip**: For small molecules, a general rule-of-thumb is that performance is best when using `CCD` codes directly, followed by `cif`/`sdf` files, and finally SMILES.

📝 **Example JSON configuration with arbitrary biomolecules** (full example found at `../../docs/releases/rf3/examples/7o1r_from_json.json`):
```json
[
    {
        "name": "7o1r_from_json",
        "components": [
            {
                "seq": "MKSLSFSLALGFGSTLVYSAPSPSSGWQAPGPNDVRAPCPMLNTLANHGFLPHDGKGITVNKTIDALGSALNIDANLSTLLFGFAATTNPQPNATFFDLDHLSRHNILEHDASLSRQDSYFGPADVFNEAVFNQTKSFWTGDIIDVQMAANARIVRLLTSNLTNPEYSLSDLGSAFSIGESAAYIGILGDKKSATVPKSWVEYLFENERLPYELGFKRPNDPFTTDDLGDLSTQIINAQHFPQSPGKVEKRGDTRCPYGYH",
                "msa_path": "../../docs/releases/rf3/examples/msas/7o1r_A.a3m.gz",
                "chain_id": "A"
            },
            {
                // A single free-floating magnesium ion
                // We will use atom names from the CCD
                // If no `chain_id` is specified, we will deterministically generate one (e.g., "B", since "A" exists above)
                // If your ligand is in the CCD, this format is preferred
                "ccd_code": "MG"
            },
            {
                // We provide the heme (HEME) via SDF from the CCD; we could have also used a CIF file
                // We will automatically name the atoms (SDF files do not specify atom names)
                "path": "../../docs/releases/rf3/examples/ligands/HEM.sdf"
            },
            {
                // We provide the imidazole ring (IMD) via SMILES
                // We will automatically name the atoms
                "smiles": "[nH]1cc[nH+]c1"
            },
        ],
    }
]
```

🚀 **Run the example:**

```bash
rf3 fold inputs='../../docs/releases/rf3/examples/7o1r_from_json.json'
```

**Supported input options:**

- **`seq`**: For proteins and nucleic acids using non-canonical one-letter codes as they appear in a CIF file. For example, `MTG(PTM)...` would suffice to include a phosphotyrosine.
- **`smiles`**: For small molecules (ensure correctness of SMILES, and proper indication of chirality when applicable)
- **`ccd_code`**: If your small molecule is already in the CCD
- **`path`**: If you have a `.sdf` or `.cif` file (including `.cif` files for small molecules)

</details>

<details>
<summary><strong>Folding with a Covalent Modification</strong></summary>

RF3 supports both standard (e.g., in the CCD) covalent modifications and custom (e.g., not in the CCD) covalent modifications.

As described in the example [Folding with Arbitrary Biomolecules](#folding-with-arbitrary-biomolecules), a well-formed CIF file includes `struct_conn` records that detail covalent modifications. RF3 can natively handle such an input.

For example, folding `7o1r`, which contains two N-glycosylations:
```bash
rf3 fold inputs='../../docs/releases/rf3/examples/7o1r_from_file.cif'
```

<p align="center">
  <img src="../../docs/_static/7o1r_covalent_modification.png" alt="7o1r Covalent Modification" width="25%"/>
</p>
<p align="center">
  <em>Figure: `7o1r` structure showing N-glycosylation covalent modification prediction with RF3 and ground truth crystal structure.</em>
</p>

Such `.cif` files complete with appropriate bonds can be composed with AtomWorks.

If you would prefer to use the JSON API, bonds can be explicitly given using PyMol-like strings of the form `chain_id/res_name/res_id/atom_name`. You will need to know the specific chain ID, residue name, residue ID, and atom name between the relevant pairs of atoms to unambiguously specify the bond.

📝 **Example JSON configuration with covalent modifcations** (full example found at `../../docs/releases/rf3/examples/7o1r_from_json.json`):
```json
[
    {
        "name": "7o1r_from_json_with_covalent_modification",
        "components": [
            {
                "seq": "MKSLSFSLALGFGSTLVYSAPSPSSGWQAPGPNDVRAPCPMLNTLANHGFLPHDGKGITVNKTIDALGSALNIDANLSTLLFGFAATTNPQPNATFFDLDHLSRHNILEHDASLSRQDSYFGPADVFNEAVFNQTKSFWTGDIIDVQMAANARIVRLLTSNLTNPEYSLSDLGSAFSIGESAAYIGILGDKKSATVPKSWVEYLFENERLPYELGFKRPNDPFTTDDLGDLSTQIINAQHFPQSPGKVEKRGDTRCPYGYH",
                "msa_path": "../../docs/releases/rf3/examples/msas/7o1r_A.a3m.gz",
                "chain_id": "A"
            },
            {
                "ccd_code": "MG"
            },
            {
                // We provide one sugar via a CIF file, with complete control over bonds and atom names (as we use the atom names from the CIF file)
                "path": "../../docs/releases/rf3/examples/ligands/NAG.cif"
            },
            {
                "path": "../../docs/releases/rf3/examples/ligands/HEM.sdf"
            },
            {
                "smiles": "[nH]1cc[nH+]c1"
            },
            {
                // We provide another sugar via the CCD, using the canonical atom names
                "ccd_code": "NAG",
                "chain_id": "F"
            }
        ],
        "bonds": [
            // We can directly target C1 on the CCD example
            ["A/ASN/133/ND2", "F/NAG/1/C1"], 
            // Things are trickier for the custom CIF example. 
            // We must determine the atom name of the atom participating in the bond, and also specify the residue name and residue index
            // `L:0`, `L:1`, etc. are guaranteed non-conflicting residue names to give custom small molecules; we will automatically name custom small molecules in this fashion if different residue names are not specified
            // If a provided residue name exists in the CCD, during parsing we will load that CCD, which will typically lead to processing errors (e.g., if atom names don't match)
            // Note as well that the provided in this example CIF is 0-indexed for the residue index, whereas in the CCD above it is 1-indexed
            ["A/ASN/161/ND2", "C/L:0/0/C1"] 
        ]
    }
]
```
</details>

<details>
<summary><strong>Creating Custom CIF Files from SDF/SMILES with AtomWorks</strong></summary>

For some more convoluted applications it is necessary to provide custom CIF files. For example, if one wants to include a bespoke covalent modification. In such situations, generally the easiest path is to use AtomWorks to create and validate the file beforehand.

For example, in the covalent modifcation example above, the custom `NAG` `.cif` file was created with:

```python
from atomworks.io.utils.io_utils import read_any, to_cif_file
from atomworks.io.tools.rdkit import atom_array_from_rdkit, sdf_to_rdkit
from atomworks.io.utils.visualize import view
import numpy as np

# Load an SDF file into an AtomArray
sdf_path = "../../docs/releases/rf3/examples/ligands/NAG.sdf"

# Load into an AtomArray
# Since SDF file files do not have atom names, we automatically generate them (e.g., C1, C2, C3, etc.)
atom_array = atom_array_from_rdkit(sdf_to_rdkit(sdf_path))

# Alternatively, you can use smiles_to_rdkit followed by atom_array_from_rdkit if you have a SMILES string rather than an SDF file
# e.g., atom_array = atom_array_from_rdkit(smiles_to_rdkit("[nH]1cc[nH+]c1"))

# Ensure it looks correct and determine the correct atom name for the covalent bond
# You can hover over an atom in PyMol (or in using the `view` function as shown below) to display the atom name
# Based on this view, we can see that the atom name for the covalent bond is C1 (the attached oxygen will leave after bond formation)
view(atom_array)

# If we want to, we can rename residues, adjust charges, add bonds, etc.
atom_array.chain_id = np.full(len(atom_array), "D")
# WARNING: Ensure that the provided residue name is not in the CCD! Otherwise, we will try and load it from the CCD and almost certainly fail!
atom_array.res_name = np.full(len(atom_array), "NAG_2") 

# ... and save to a new file
to_cif_file(atom_array, "NAG.cif")
```

Similarly, for SMILES:

```python
# For a SMILES string
smiles = "[nH]1cc[nH+]c1"  # imidazole ring
atom_array = atom_array_from_rdkit(smiles_to_rdkit(smiles))

# Customize and save
atom_array.chain_id = np.full(len(atom_array), "E")
atom_array.res_name = np.full(len(atom_array), "IMD_CUSTOM")
to_cif_file(atom_array, "imidazole_custom.cif")
```

We now know exactly which atom name corresponds to which atom, enabling us to unambiguously specify bonds as needed.

> [!TIP]
> **Leaving Groups**: If your covalent modification includes a functional group or atom that leaves upon bond formation, that group should be replaced with a `H` *a priori* or specified as a leaving atom in the CIF file using the RCSB's [leaving atom flag](https://www.iucr.org/__data/iucr/cifdic_html/2/mmcif_pdbx.dic/Ichem_comp_atom.pdbx_leaving_atom_flag.html).

</details>

<details>
<summary><strong>Folding while Templating</strong></summary> 

RF3 departs from prior structure prediction models by including explicit atomic-level, ground-truth templates during training. That is, with some probability, we show the model a portion of the **the actual ground-truth crystal structure** rather than a template identified through homology-based search like prior methods. We make this choice to maximize flexibility during inference, where we often want to fix portions of the structure while allowing other components to fold.

More specifically, we template in two ways:
- At the token-level, by providing a ground-truth distogram
- At the atom-level through the reference-conformer track, where we provide an explicit "ground truth conformer"

Different templating techniques are appropriate for different situations. Templating polymer structure is best accomplished through the token-level track (`template_selection`). Small molecule templating (but only if we want to template the entire residue) is best accomplished through the `ground_truth_conformer_selection` track, but could also be done through `template_selection` in cases where we do not want to fix the entire residue.

We work through several examples below.

For our selection strings, we use the `AtomSelection` syntax from AtomWorks.

<details>
<summary><strong>AtomSelection Query Syntax</strong></summary>

RF3 uses AtomWorks' flexible `AtomSelectionStack` query syntax for specifying structural selections. The syntax follows the pattern `CHAIN/RES_NAME/RES_ID/ATOM_NAME/TRANSFORM_ID`, where each field can be:

- **Exact value**: `A`, `ALA`, `1`, `CA`, `0`
- **Wildcard**: `*` (matches any value)
- **Range** (residues only): `5-10` (inclusive range)
- **Multiple selections**: Comma-separated for union operations

#### Selection Grammar

| Field Position | Description | Examples |
|---|---|---|
| 1. Chain ID | Chain identifier | `A`, `B`, `*` |
| 2. Residue Name | 3-letter residue code | `ALA`, `GLY`, `*` |
| 3. Residue ID | Residue number (supports ranges) | `1`, `5-10`, `*` |
| 4. Atom Name | Atom identifier | `CA`, `CB`, `*` |
| 5. Transform ID | Transformation identifier | `0`, `1`, `*` |

#### Common Selection Examples

| Selection Description | Query Syntax |
|---|---|
| All atoms in chain A | `A` |
| All alanine residues in chain A | `A/ALA` |
| Residues 5-10 in chain A | `A/*/5-10` |
| CA atoms of residues 1-5 in chain B | `B/*/1-5/CA` |
| All atoms in chains A and B | `A, B` |
| Framework regions (CDR templating) | `B/*/1-42, B/*/49-63, B/*/71-102` |
| Specific atom in specific residue | `A/ALA/15/CB` |
| All backbone atoms in chain A | `A/*/*/N, A/*/*/CA, A/*/*/C, A/*/*/O` |
</details>


#### Templating a Polymer (Protein / DNA / RNA)

It is often helpful to template one or multiple polymer chains while allowing the other chain(s) to fold unconstrained. We demonstrate with an nanobody-antigen use case below how to apply templates.

📝 **Example JSON configuration templating the antigen and the nanobody framework** (full example found at `../../docs/releases/rf3/examples/7xli_template_antigen_and_framework.json`):
```json
[
    {
        "name": "7xli_template_antigen",
        "components": [
            {
                "path": "../../docs/releases/rf3/examples/templates/7xli_chain_A.cif"
            },
            {
                "path": "../../docs/releases/rf3/examples/templates/7xli_chain_B.cif"
            }
        ],
        "template_selection": ["A", "B/*/1-42", "B/*/49-63", "B/*/71-102", "B/*/108-125"]
    }
]
```

🚀 **Run the example:**

```bash
rf3 fold inputs='../../docs/releases/rf3/examples/7xli_template_antigen_and_framework.json'
```

You may also specify templating directly via the CLI using `template_selection="[A, B/*/1-42, ...]"`.

> [!NOTE]
> **Pairwise Contact Support**: The model was trained with random pairwise contacts; our inference API could be further extended to support specification of individual (token-level) pairwise contacts in the future. Please let us know if this would be helpful for your usecase and we can prioritize appropriately.

#### Templating a Small Molecule

We find that enforcing a particular small molecule conformation has various applications within fixed-ligand protein docking, enzyme activity filtering, and other biologically relevant tasks. RF3 natively enables encouraging a particular small molecule conformations via both the ground truth reference conformer track and the template selection track. 

For the moment, the ground truth conformer track is only effective if we want to template the *entire* small molecule. Partial templating of small molecules is still possible via the `template_selection` approach. We encourage exploration of both templating techniques to find what combination(s) are most effective for a given problem. Below we provide both, which represents the strongest possible conditioning.

📝 **Example JSON configuration templating a small molecule and the corresponding protein** (full example found at `../../docs/releases/rf3/examples/1eiz_template_ligand_and_protein.json`):
```json
[
    {
        "name": "9dfn_template_ligand_and_protein",
        "components": [
            {
                "path": "../../docs/releases/rf3/examples/9dfn.cif"
            }
        ],
        "template_selection": ["A", "C", "D"],
        "ground_truth_conformer_selection": ["C", "D"]

    }
]
```

> [!NOTE]
> We template the protein above to avoid providing an MSA

🚀 **Run the example:**

```bash
rf3 fold inputs='../../docs/releases/rf3/examples/8cdz_templating_ligand.json'
```

You may also specify the ground truth conformer selection directly via the CLI, e.g., using `ground_truth_conformer_selection="[E]"`

#### Templating an Interface

RF3 was also trained to respect interface templates; that is, pairwise distances between tokens across an interface. We have yet to extend our inference API to support this use case; if it would be helpful, please raise an issue on GitHub and we can re-prioritize accordingly.

*Content coming soon...*

</details>

## Chirality

If inputs are given in a form that specifies chirality, the model will receive the corresponding features and attempt to preserve the chirality of the inputs.

**Chiral formats include:**
- SMILES strings (e.g., using `@`)
- CIF, PDB, or SDF files with non-zero coordinates

## Command-Line Arguments

<details>
<summary><strong>Basic Arguments</strong></summary>

- **`inputs`** *(required)*: Path to a file (CIF/PDB/JSON) or list of files for prediction; if given a directory, all CIF/PDB files in that directory will be predicted. To specify a list of files/directories, use Hydra's list grammar: `foo="[path_1.cif, path_2.json, path_3.pdb]"`

- **`inference_engine`** *(required)*: The inference configuration to use. For example, `rf3`, to use the standard structure prediction model. We will introduce other configurations down-the-line, each with unique use cases. Defaults to `rf3` when using the `rf3 fold` command.

- **`ckpt_path`** *(optional)*: Path to checkpoint file. Defaults to the current "best model", which is stored in a symlink in `/net/software`

- **`residue_renaming_dict`** *(optional)*: Dictionary of residues to rename to avoid CCD clashes, given in Hydra format (e.g., `foo="{'ALA': 'L:1'}`). When parsing files, we use the given residue names to help identify any missing atoms. Thus, if a custom ligand overlaps with a ligand in the CCD, the prediction will be catastrophically wrong. To circumvent this issue, we accept a dictionary of ligands to rename. We suggest renaming all custom ligands to begin with `L:` to avoid all clashes with the CCD. 
  
  > [!WARNING]
  > This command uses brute-force find and replace; please ensure that there are no other possible matches (e.g., atom names). Additionally, avoid `#` to mitigate possible CIF-parsing errors from PyMol. Defaults to None.

- **`skip_existing`** *(optional)*: Whether to skip predictions where appropriately-named output structures already exist in the `out_dir`. Defaults to False (do not skip; overwrite instead).

</details>

<details>
<summary><strong>Model Control Arguments</strong></summary>

- **`early_stopping_plddt_threshold`** *(optional)*: The average all-atom pLDDT value estimated after a single recycle that will trigger early-exit for that prediction. Defaults to `0.5`. Using this flag can **significantly** increase structure throughput (10-20x). If we early exit:
    - There will be no output structure files (`.cif.gz`)    
    - The `.score` file will contain a field `early_stopped` that will have the value `True`; it will also contain columns indicating the value of the all-atom pLDDT after the first recycle and the threshold applied.

- **`n_recycles`** *(optional)*: Number of recycles within the trunk. Defaults to 10.

- **`diffusion_batch_size`** *(optional)*: Number of output structures in the ensemble, drawn from the same model seed and forward pass of the Pairformer. Defaults to 5.

- **`num_steps`** *(optional)*: Number of steps for sampling of the diffusion module. The standard is 200; we see no deterioration in performance with 50 steps, but significant (>2x) speed improvements. Defaults to 200.

- **`seed`** *(optional)*: Model seed. Running inference multiple times with different model seeds is the best, and most expensive, way to generate output diversity. Defaults to the training seed (usually 42).

</details>

<details>
<summary><strong>Advanced Structural Control Arguments</strong></summary>

- **`ground_truth_conformer_selection`** *(optional)*: Selection syntax for residues that should use ground truth conformers instead of generated ones. Uses `AtomSelection` format; see <a>TODO: HREF TO TEMPLATING, SET ID</a> for more description of the syntax. If None, no residues will use ground truth conformers. This is useful for keeping known ligand conformations while allowing the model to predict protein structure around them. For example, if chains `C` and `D` are two difficult ligands that are not predicted correctly unconditionally, we could run `rf3 fold ... ground_truth_conformer_selection="[C,D]"` to force a particular conformation (we would also need to provide these chains through a CIF, SDF, or MOL file)

- **`template_selection`** *(optional)*: Selection syntax to provide token-level templates (for both polymers and non-polymers). Uses `AtomSelection` format. Similar to traditional homology-style templates, but also can be applied to small molecules (less rigidly adhered to than the `ground_truth_conformer_selection` approach). See <a>TODO: HREF TO TEMPLATING, SET ID</a> for more information.

- **`cyclic_chains`** *(optional)*: List of strings of chain ids that should be cyclized. If given a cif, the model will automatically detect cyclization.

</details>

<details>
<summary><strong>Output Control Arguments</strong></summary>

- **`out_dir`** *(optional)*: Where to save predicted structures. The output files will be named the same as the input structures, or use the `name` field in the specification, if present. Defaults to the current directory (`./`).

- **`dump_predictions`** *(optional)*: Whether to save outputs as CIF files (vs. only the `.score` file). Defaults to True.

- **`dump_trajectories`** *(optional)*: Whether to dump the denoising trajectories. Defaults to False. Denoising trajectories are memory- and CPU-intensive to save to disk; we do not suggest dumping them except for a select few structures, if needed.

- **`one_model_per_file`** *(optional)*: Whether to save multiple structures from one diffusion batch as separate models within the same file or separate files. Defaults to False (one file with multiple models).

- **`annotate_b_factor_with_plddt`** *(optional)*: Whether to annotate atom-level pLDDT, overwriting the `b_factor` column in the CIF output (full name is `b_iso_or_equiv` in mmCIF files). Defaults to False. 
  
  > [!NOTE]
  > If set to True, then `one_model_per_file` will be automatically set to True (since our CIF-saving software `biotite` does not support variable `b_factors` across models within the same file).

  > [!NOTE]
  > The CIF files are saved in a compressed format, `.cif.gz`. These compressed files can be directly loaded by PyMol or parsed by `atomworks.ml`. If you need to inspect the uncompressed file, you can use `gunzip <PATH>`. 
  
  > [!NOTE]
  > The CIF output file will contain multiple **models**, one for each diffusion outputs (e.g., 5 by default). PyMol will hide secondary structure by default with multiple models; the command `dss` will display it again.
</details>


## Viewing the Predicted Structure(s)

Use the following code to view the predicted structures with AtomWorks

```python
from atomworks.io.utils.visualize import view
from atomworks.io import parse

# View in atomworks (or PyMol, etc.)
out = parse("path/to/prediction.cif.gz")
atom_array = out["assemblies"]["1"][0]
# (If in a notebook)
view(atom_array)
```

**Alternative viewing options:**
- View in PyMol like normal, or using `pymol_remote`
- Use the `view_pymol()` function for direct PyMol integration

