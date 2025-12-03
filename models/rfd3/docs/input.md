# RFdiffusion3 — Input specification (dialect **2**)

> **TL;DR**  
> Inputs are now defined with a single `InputSpecification` class.  
> Selections like “what’s fixed?”, “what’s sequence-free?”, “which atoms are donors/acceptors?” are all expressed with the same **InputSelection** mini-language.  
> Everything is reproducibly logged back out alongside your generation.

---

- [What changed (high level)](#what-changed-high-level)
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

## How it works (high level)

- **Unified selections.** All per-residue/atom choices now use **InputSelection**:
  - You can pass `true`/`false`, a **contig string** (`"A1-10,B5-8"`), or a **dictionary** (`{"A1-10": "ALL", "B5": "N,CA,C,O"}`).
  - Selection fields include: `select_fixed_atoms`, `select_unfixed_sequence`, `select_buried`, `select_partially_buried`, `select_exposed`, `select_hbond_donor`, `select_hbond_acceptor`, `select_hotspots`.
- **Clearer unindexing.** For **unindexed** motifs you typically either fix `"ALL"` atoms or explicitly choose subsets such as `"TIP"`/`"BKBN"`/explicit atom lists via a **dictionary** (see examples).  
  When using `unindex`, only **the atoms you mark as fixed** are carried over from the input.
- **Reproducibility.** The exact specification and the **sampled contig** are logged back into the output JSON. We also log useful counts (atoms, residues, chains).
- **Safer parsing.** You’ll now get early, informative errors if:
  - You pass unknown keys,
  - A selection doesn’t match any atoms,
  - Indexed and unindexed motifs overlap,
  - Mutually exclusive selections overlap (e.g., two RASA bins for the same atom).
- **Backwards compatible.** Add `"dialect": 1` to keep your old configs running while you migrate. (Deprecated.)

---

## InputSpecification

| Field                                                          | Type              | Description                                                           |
| -------------------------------------------------------------- | ----------------- | --------------------------------------------------------------------- |
| `input`                                                        | `str?`            | Path to input **PDB/CIF**. Required if you provide contig+length.    |
| `atom_array_input`                                             | internal          | Pre-loaded `AtomArray` (not recommended).                             |
| `contig`                                                       | `InputSelection?` | Indexed motif specification, e.g., `"A1-80,10,\0,B5-12"`.             |
| `unindex`                                                      | `InputSelection?` | Unindexed motif components (unknown sequence placement).              |
| `length`                                                       | `str?`            | Total design length constraint; `"min-max"` or int.                   |
| `ligand`                                                       | `str?`            | Ligand(s) by resname or index.                                        |
| `cif_parser_args`                                              | `dict?`           | Optional args to CIF loader.                                          |
| `extra`                                                        | `dict`            | Extra metadata (e.g., logs).                                          |
| `dialect`                                                      | `int`             | `2`=new (default), `1`=legacy.                                        |
| `select_fixed_atoms`                                           | `InputSelection?` | Atoms with fixed coordinates.                                         |
| `select_unfixed_sequence`                                      | `InputSelection?` | Where sequence can change.                                            |
| `select_buried` / `select_partially_buried` / `select_exposed` | `InputSelection?` | RASA bins 0/1/2 (mutually exclusive).                                 |
| `select_hbond_donor` / `select_hbond_acceptor`                 | `InputSelection?` | Atom-wise donor/acceptor flags.                                       |
| `select_hotspots`                                              | `InputSelection?` | Atom-level or token-level hotspots.                                   |
| `redesign_motif_sidechains`                                    | `bool`            | Fixed backbone, redesigned sidechains for motifs.                     |
| `symmetry`                                                     | `SymmetryConfig?` | See `docs/symmetry.md`.                                               |
| `ori_token`                                                    | `list[float]?`    | `[x,y,z]` origin override to control COM placement                    |
| `infer_ori_strategy`                                           | `str?`            | `"com"` or `"hotspots"`.                                              |
| `plddt_enhanced`                                               | `bool`            | Default `true`.                                                       |
| `is_non_loopy`                                                 | `bool`            | Default `true`.                                                       |
| `partial_t`                                                    | `float?`          | Noise (Å) for partial diffusion, enables partial diffusion            |


## Quick start

### Minimal JSON example

```json
{
    "": {
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
### Mininmal YAML example
```
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

# Appendix
## FAQ / gotchas
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

## Shorthand atoms for easy specification
Keyword	Expands to
BKBN	N, CA, C, O
TIP	Residue-specific “tip” atoms
ALL	All atoms of each residue


