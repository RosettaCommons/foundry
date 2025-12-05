# RFdiffusion3 — Input Specification & Command-line arguments

---

## Contents
- [Quick start](#quick-start)
- [InputSpecification fields](#inputspecification-fields)
- [The `InputSelection` mini-language](#the-inputselection-mini-language)
- [Unindexing specifics](#unindexing-specifics)
- [Partial diffusion](#partial-diffusion)
- [Debugging recommendations](#debugging-recommendations)
- [FAQ / gotchas](#faq--gotchas)

---

## Quick start

JSON inputs take the following top-level structure;
```json
{
    "spec-1": {  // First design configuration
      "input": "<path/to/pdb>",
      "contig": "50-80,/0,A1-100",  // Diffuses length 50-80 monomer in chain A & selects indices A1 -> A100 in input pdb to have fixed coordinates and sequences  
      "select_unfixed_sequence": "A20-35", // Converts selected indices in input to have unfixed sequence (inputs become atom14).
      "ligand": "HAX,OAA",  // Selects ligands HAX and OAA based on res name in the input
    },
    "spec-2": {
      // ... args for the second (independent) configuration for design. 
    }
}
```

You can then run inference at the command line with:
```
rfd3 design out_dir=<path/to/outdir> inputs=<path/to/inputs>
```
In this document, we detail the syntax of the config structure.

## CLI arguments
Key CLI arguments (from the default config) to know include:
- `n_batches` — number of batches to generate per input key (default: 1).
- `diffusion_batch_size` — number of diffusion samples per batch (default: 8).
- `specification` — JSON overrides for the per-example InputSpecification (default: `{}`). For example, you can run `rfd3 design inputs=null specification.length=200` for a quick debug of creating a 200-length protein.
- `inference_sampler.num_timesteps` — diffusion timesteps for sampling (default: 200).
- `inference_sampler.step_scale` — scales diffusion step size; higher → less diverse, more designable (default: 1.5).
- `low_memory_mode` — memory-efficient tokenization mode; set `True` if GPU RAM is tight (default: False).

The full config of default arguments that are applied can be seen in [inference_engine/rfdiffusion3.yaml](../configs/inference_engine/rfdiffusion3.yaml)

## InputSpecification fields

Below is a table of all of the inputs that the `InputSpecification` accepts. Use these fields to describe what RFdiffusion3 should do with your inputs.


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


A few notes on the above:
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
## The InputSelection mini-language

Fields marked as `InputSelection` accept either a boolean, a contig-style string, or a dictionary. Dictionaries are the most expressive and can also use shorthand values like `ALL`, `TIP`, or `BKBN`:
```yaml
select_fixed_atoms:
  A1-2: BKBN # equivalent to 'N,CA,C,O'
  A3: N,CA,C,O,CB  # specific atoms by atom name
  B5-7: ALL # Selects all atoms within B5,B6 and B7
  B10: TIP  # selects common tip atom for residue (constants.py)
  LIG: ''  # selects no atoms (i.e. unfixes the atoms for ligands named `LIG`)
```

<p align="center">
  <img src=".assets/input_selection.png" alt="InputSelection language for foundry" width=500>
</p>

## Unindexing specifics

`unindex` marks motif tokens whose relative sequence placement is unknown to the model (useful for scaffolding around active sites, etc.).
Use a string to list the unindexed components and where breaks occur.
Use a dictionary if you want to fix specific atoms of those residues; atoms not fixed are not copied from the input (they will be diffused).
Breaks between unindexed components follow the contig conventions you’re used to. For example: `"A244,A274,A320,A329,A375"` lists multiple unindexed components; internal “breakpoints” are inferred and logged. (Offset syntax like A11-12 or A11,0,A12 still ties residues.)
You can specify consecutive residues as e.g. `A11-12` (instead of `A11,A12`), this will tie the two components together in sequence (or at least it leaks to the model that residues are together in sequence). 
Similarly, you can specify manually any number of residues that offsets two components, e.g. `A11,0,A12` (0 sequence offset, equivalent to just `A11-12`), or `A11,3,A12` (3-residue separation).
From our initial tests this only leads to a slight bias in the model, but newer models may show better adherence!

## Partial Diffusion
To enable partial diffusion, you can pass `partial_t` with any example. This sets the *noise level* in *angstroms* for the sampler:
- The `specification.partial_t` argument can be specified from JSON or the command line.
- Partial diffusion will fix/unfix ligands and nucleic acids as normal, by default it will fix non-protein components and they must be specified explicitly.
- By default, the ca-aligned `ca_rmsd_to_input` will be logged.
- Currently, partial diffusion subsets the inference schedule based on the partial_t, so `inference_sampler.num_timesteps` will affect how many steps are used but it is not equal to the number of steps used.

In the following example, RFD3 will noise out by 15 angstroms and constrain atoms of three residues. In this output one of the 8 diffusion outputs swapped its sequence index by one residue:
```json
{
    "partial_diffusion": {
        "input": "paper_examples/7v11.cif", 
        "ligand": "OQO", 
        "partial_t": 15.0,
        "unindex": "A431,A572-573",
        "select_fixed_atoms": {
            "A431": "TIP",
            "A572": "BKBN",
            "A573": "BKBN"
        }
    }
}
```
Below is an example of what the output should look like (diffusion outputs in teal, original native in navajo white):
<p align="center">
  <img src=".assets/partial_diff.png" alt="Partial diffusion" width=650>
</p>

## Debugging recommendations
- For unindexed scaffolding, you can use the option `cleanup_guideposts=False` to keep the models' outputs for the guideposts. The guideposts are saved as separate chains based on whether their relative indices were leaked to the model: e.g. for `unindex=A11-12,A22`, you should see `A11` and `A12` indexed together on one chain and `A22` on its own chain, indicating the model was provided with the fact that `A11` and `A12` are immediately next to one another in sequence but their distance to `A22` is unknown.
- To see the full 14 diffused virtual atoms you can use `cleanup_virtual_atoms=False`. Default is to discard them for the sake of downstream processing.
- To see the trajectories, you can use `dump_trajectories=True`. This can be useful if the outputs look strange but the config is correct, or if you want to make cool gifs of course! Trajectories do not have sequence labels and contain virtual atoms.

## FAQ / gotchas
<details>
  <details>
  <summary><b>Can I guide on secondary structure?</b></summary>
  Currently no - in future models we may do so, however, you can use `is_non_loopy: true` to make fewer loops. We find this produces a lot more helices and fewer loops (and less sheets).
  </details>

  <summary><b>Do I need select_fixed_atoms & select_unfixed_sequence every time?</b></summary>
  
  No. Defaults apply when input present.
  </details>

  <details>
  <summary><b>Why "Input provided but unused"?</b></summary>

  This indicates you gave an input pdb / cif (not `input: null`) but no contig, unindex, ligand or partial_t.
  </details>

  <details>
  <summary><b>What do the logged bfactors mean?</b></summary>

  The sequence head from RFD3 logs its confidence for each token in the output structure, you can run `spectrum b` in `pymol` to see it. It usually doesn't mean anything but can give you some idea if the model has gone vastly distribution if the entropy is high (uncertain assignment of sequence).
  </details>
</details>

Let us know if you have any additional questions, we'd be happy to answer them!

## Further examples of InputSelection syntax

Below is a reference for more examples of different ways you can specify inputs to select from your pdb in configs; we hope the community can find use in this flexible system for future models!
<p align="center">
  <img src=".assets/input_selection_large.png" alt="Input selection syntax" width=650>
</p>
