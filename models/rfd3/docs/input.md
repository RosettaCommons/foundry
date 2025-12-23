# RFdiffusion3 — Input specification (dialect **2**)

> **TL;DR**  
> Inputs are now defined with a single `InputSpecification` class, see [`rfd3/src/rfd3/inference/input.parsing.py`](https://github.com/RosettaCommons/foundry/blob/rac_docs/models/rfd3/src/rfd3/inference/input_parsing.py) to see all possible inputs.   
> Selections like “what’s fixed?”, “what’s sequence-free?”, “which atoms are donors/acceptors?” are all expressed with the same **InputSelection** mini-language.  
> Everything is reproducibly logged back out alongside your generation – each design will create an output JSON file with all setting defined.

---

- [Quick start](#quick-start)
- [The `InputSelection` mini-language](#the-inputselection-mini-language)
- [Full schema: `InputSpecification`](#full-schema-inputspecification)
- [Common recipes (cookbook)](#common-recipes-cookbook)
- [Partial diffusion](#partial-diffusion)
- [Symmetry](#symmetry)
- [Origin (`ori_token`) and initialization](#origin-ori_token-and-initialization)
- [Validation & error messages](#validation--error-messages)
- [Metadata & logging](#metadata--logging)
- [Legacy configs (dialect=1) & migration guide](#legacy-configs-dialect1--migration-guide)
- [Multi-example files](#multi-example-files)
- [FAQ / gotchas](#faq--gotchas)

---

## InputSpecification
Here are some of the inference settings in RFdiffusion3 (RFD3):
* For the inputs that are of type `InputSelection` see section [The InputSelection mini-language](#the-inputselection-mini-language) for more details

| Field                                                          | Type              | Description                                                                             |
| -------------------------------------------------------------- | ----------------- | --------------------------------------------------------------------------------------- |
| `input`                                                        | `str`             | Path to and file name of input **PDB/CIF**. Required if you provide `contig`+`length`.  |
| `atom_array_input`                                             | `AtomArray`       | Pre-loaded `AtomArray` ([class from Biotite](https://www.biotite-python.org/latest/apidoc/biotite.structure.AtomArray.html)) (not recommended). |
| `contig`                                                       | `InputSelection`             | Indexed motif specification, e.g., `"A1-80,10,\0,B5-12"`.  More details in [next section](#contig)  |
| `unindex`                                                      | `InputSelection`   | Unindexed motif components (unknown sequence placement). Example: `A15-20,B6-10` or <!-- TO DO test out dictionary specification for this--> |
| `length`                                                       | `str?`            | Total design length constraint; `"min-max"` or int.                                    |
| `ligand`                                                       | `str?`            | Ligand(s) by resname or index.                                                         |
| `cif_parser_args`                                              | `dict?`           | Optional args to CIF loader.                                                           |
| `extra`                                                        | `dict`            | Extra metadata (e.g., logs).                                                           |
| `dialect`                                                      | `int`             | `2`=new (default), `1`=legacy.                                                         |
| `select_fixed_atoms`                                           | `InputSelection?` | Atoms with fixed coordinates.                                                          |
| `select_unfixed_sequence`                                      | `InputSelection?` | Where sequence can change.                                                             |
| `select_buried` / `select_partially_buried` / `select_exposed` | `InputSelection?` | RASA bins 0/1/2 (mutually exclusive).                                                  |
| `select_hbond_donor` / `select_hbond_acceptor`                 | `InputSelection?` | Atom-wise donor/acceptor flags.                                                        |
| `select_hotspots`                                              | `InputSelection?` | Atom-level or token-level hotspots.                                                    |
| `redesign_motif_sidechains`                                    | `bool`            | Fixed backbone, redesigned sidechains for motifs.                                      |
| `symmetry`                                                     | `SymmetryConfig?` | See `docs/symmetry.md`.                                                                |
| `ori_token`                                                    | `list[float]?`    | `[x,y,z]` origin override to control COM placement                                     |
| `infer_ori_strategy`                                           | `str?`            | `"com"` or `"hotspots"`.                                                               |
| `plddt_enhanced`                                               | `bool`            | Default `true`.                                                                        |
| `is_non_loopy`                                                 | `bool`            | Default `true`.                                                                        |
| `partial_t`                                                    | `float?`          | Noise (Å) for partial diffusion, enables partial diffusion                             |


## Quick start

### `contig`
The 'contig string' is one way to specify the portions of your final structure that come from your input PDB/CIF or are designed by RFD3. Here are a few guidelines for writing a `contig` string: 
- Different portions of the string should be comma separated
- `\0` denotes a chain break - no peptide bond is specified between the chain before/after the chain break but the break can be as large/small as makes sense for the rest of the design
- Any portions of the string that start with a letter (e.g. `A1-80`) come from the input PDB, the letter corresponds to the chain label in the input PDB/CIF file
- Any portions of the string that do **not** start with a letter are going to be designed by RFD3
- If a range is specified for a designed segment (e.g., `100–150`), the length of the designed region is sampled uniformly at random from that range, inclusive.
- The order of the `contig` string is followed in the design

> **Example** 
>
> `A1-80,10-20,A100-120,B25-50,\0,C43-56,40-60`
>
> The resulting design would have: 
>   - Residues 1-80 from chain A in the input PDB/CIF
>   - 10 to 20 designed residues that connect to residue A80
>   - Residues 100-120 from chain A in the input PDB/CIF, connected to the last residue in the designed region
>   - Residues 25-50 from chain B in the input PDB/CIF, connected to A120, even if this connection did not exist in the input PDB/CIF
>   - A chain break
>   - Residues 43-56 from chain C in the input PDB/CIF not connected to the previous chain
>   - 40-60 designed residues that connect to residue C56

### Input File Types
For more detailed information about these file types, see {doc}`intro_inference_calculations`.

#### Minimal JSON example

```json
{
  "calculation_label": {
    "input": "path/to/template.pdb",
    "contig": "A1-80",
    "length": "150-180",
    "select_fixed_atoms": true,
    "select_unfixed_sequence": "A20-35",
    "ligand": "HAX,OAA",
    "dialect": 2
  }   
}
```

#### Mininmal YAML example
```yaml
calculation_label:
  input: path/to/template.pdb
  contig: A1-80
  length: 150-180
  select_fixed_atoms: true
  select_unfixed_sequence: A20-35
  ligand: HAX,OAA
  dialect: 2
```

### Python API
```
from rfd3.inference.input_parsing import create_atom_array_from_design_specification

atom_array, metadata = create_atom_array_from_design_specification(
    input="path/to/template.pdb",
    contig="A1-80",
    length="150-180",
    select_fixed_atoms=True,
    select_unfixed_sequence="A20-35",
    dialect=2,
)
```

## The InputSelection mini-language

Fields which are specified as `InputSelection` are fields which can take either: `Bool, List, Dict`.
Dictionaries are the most expressive and can also take special :
```yaml
select_fixed_atoms:
  A1-2: BKBN
  A3: N,CA,C,O,CB  # specific atoms by atom name
  B5-7: ALL # Selects all atoms within B5,B6 and B7
  B10: TIP  # selects common tipatom for residue (constants.py)
  LIG: ''  # selects no atoms (i.e. unfixes the atoms for ligands named `LIG`)
```

[Diagram]

## Unindexing specifics

`unindex` marks motif tokens whose relative sequence placement is unknown to the model (useful for scaffolding around active sites, etc.).
Use a string to list the unindexed components and where breaks occur.
Use a dictionary if you want to fix specific atoms of those residues; atoms not fixed are not copied from the input (they will be diffused).
Breaks between unindexed components follow the contig conventions you’re used to. For example:

`"A244,A274,A320,A329,A375"`

lists multiple unindexed components; internal “breakpoints” are inferred and logged. (Offset syntax like A11-12 or A11,0,A12 still ties residues.)

## Appendix
### FAQ / gotchas
<details>
  <summary><b>Do I need select_fixed_atoms & select_unfixed_sequence every time?</b></summary>

  No. Defaults apply when input present.
  </details>

<details>
  <summary><b>Do I need select_fixed_atoms & select_unfixed_sequence every time?</b></summary>

  No. Defaults apply when input present.
  </details>

  <details>
  <summary><b>What does "ALL" vs "TIP" in unindex mean?</b></summary>

  - **`ALL`** → copy full residue
  - **`TIP`** → fix only sidechain tip atoms
  </details>

  <details>
  <summary><b>Can selections overlap?</b></summary>

  Only certain ones (fixed vs unfixed) may; RASA & donor/acceptor cannot.
  </details>

  <details>
  <summary><b>How to fix backbone but redesign sidechains?</b></summary>

  `redesign_motif_sidechains: true`
  </details>

  <details>
  <summary><b>Why "Input provided but unused"?</b></summary>

  You gave input but no contig, unindex, or partial_t.
  </details>

### Shorthand atoms for easy specification
Keyword	Expands to
BKBN	N, CA, C, O
TIP	Residue-specific “tip” atoms
ALL	All atoms of each residue


