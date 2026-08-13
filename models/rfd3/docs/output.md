# RFdiffusion3 — Output Metrics

For each design RFD3 creates, a JSON file with metrics that can be used to evaluate the design is also produced. This document describes the shape of that JSON file and gives a complete reference for every metric it can contain, including the conditions under which each metric is (or isn't) present.

```{tip}
For more detailed information on RFdiffusion3 inputs, see {doc}`input`.
```

---

## Contents
- [Output file structure](#output-file-structure)
- [Metrics reference](#metrics-reference)
  - [Insertion & join-point metrics](#insertion--join-point-metrics)
  - [Backbone geometry, clashes & fold composition](#backbone-geometry-clashes--fold-composition)
  - [Hydrogen bonds](#hydrogen-bonds)
  - [Partial diffusion](#partial-diffusion-metrics)
- [FAQ / Gotchas](#faq--gotchas)

---

(output-file-structure)=
## Output file structure

A typical output JSON file will have the various sections:
```json
{
    "diffused_index_map": {},
    "metrics": {
        "n_chainbreaks": 0,
        "max_ca_deviation": 0.04,
        "helix_fraction": 0.31,
    },
    "specification": {},
    "inference_sampler": {},
    "ckpt_path": "...",
    "seed": null
}
```
You can see a real example of a full output JSON in the {doc}`advanced enzyme design tutorial <tutorials/advanced_enzyme_design_tutorial>`.

### Diffused Index Map
The first section of each output JSON file for your generated designs is the `diffused_index_map`. This section lists the original residue labels (chain ID and residue number) for each residue that was read in from an input structure – such as a theozyme or motif – and matches it to where this residue is now located in the final design.

### Metrics

```{note}
**Not every metric is always present.** Several groups of metrics are only computed under specific conditions (e.g., only for unindexed motifs, only when `cleanup_virtual_atoms=True`, only when a ligand is present). When the condition isn't met, the key(s) are omitted from `"metrics"` entirely — they will not appear as `null` or `0`. Each table below notes the relevant condition.
```


(metrics-reference)=
## Metrics reference

(insertion--join-point-metrics)=
### Insertion & join-point metrics

```{note}
This entire group is only computed when the design spec contains **unindexed** motif residues (i.e., you used the `unindex` input field, see {ref}`Unindexing Specifics <unindexing-specifics>`). For fully indexed or fully unconditional designs, none of these keys appear in the output at all. Computed in [`process_unindexed_outputs`](https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/src/rfd3/trainer/trainer_utils.py).
```

| Metric Name | Description |
| --- | --- |
| `insertion_rmsd` | **Insertion RMSD** (Å). Measures how accurately the unindexed motif residues were placed into the generated backbone, averaged over all unindexed motif residues. This is the root-mean-square deviation between the original motif atom positions and their matched positions in the diffused structure. **Lower is better**; values below ~0.5 Å indicate good placement. |
| `insertion.mae` | **Insertion MAE** (Å). Mean absolute error of the motif atom placement. Similar to `insertion_rmsd` but less sensitive to outliers. **Lower is better.** |
| `insertion.rmcd` | **Insertion RMCD** (Å). Root mean cubic deviation of the motif atom placement. More sensitive to large deviations than RMSD. **Lower is better.** |
| `join_point_rmsd` | **Join point RMSD** (Å), averaged over `join_point_rmsd_by_token`. **Lower is better.** ⚠️ Only covers unindexed motif residues where **no backbone atoms are held fixed** (`N`/`CA`/`C`/`O`) — i.e. the `select_fixed_atoms` fixes only side-chain atoms (e.g. `TIP`) or a single atom for that residue, so no backbone coordinates were carried over from the input and the join is measured from a `CB` or atomized atom instead. Residues with at least one fixed backbone atom are skipped for this metric, so it is *not* averaged over all motif residues and does not directly measure scaffold backbone accuracy. |
| `join_point_rmsd_by_token` | Per-residue breakdown of `join_point_rmsd`, keyed by the **input** residue label (where the residue was in your input motif, not the output design). Only contains entries for the residues described above, those with no fixed backbone atoms. |
| `n_conjoined_residues` | Number of motif residues that mapped to multiple diffused positions. Should be **0** for a clean design. |

(backbone-geometry-clashes--fold-composition)=
### Backbone geometry, clashes & fold composition

```{note}
This entire group is only computed when `cleanup_virtual_atoms=True` (the default; see {ref}`Other CLI arguments <other-useful-cli-arguments>`). If a run is launched with `cleanup_virtual_atoms=False` - e.g. while debugging virtual atoms, see {ref}`Debugging recommendations <debugging-recommendations>` - **none** of the metrics below are computed and the whole group is absent from `"metrics"`. Computed in [`get_all_backbone_metrics`](https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/src/rfd3/metrics/design_metrics.py).
```

| Metric Name | Description |
| --- | --- |
| `n_chainbreaks` | Number of backbone breaks where the CA-CA distance deviates significantly from the ideal 3.8 Å. Should be **0** for a clean design. |
| `max_ca_deviation` | Largest CA-CA distance deviation from the standard 3.8 Å bond length. **Lower is better**; very small values (< 0.1 Å) indicate a clean backbone trace. |
| `n_clashing.interresidue_clashes_w_sidechain` | Number of interresidue steric clashes considering **all** atoms (backbone + sidechain) between non-adjacent residues (distance < 1.5 Å). Should be **0** or very low. |
| `n_clashing.interresidue_clashes_w_backbone` | Number of interresidue steric clashes restricted to backbone atoms (`N`/`CA`/`C`) only. Should be **0**. |
| `n_clashing.ligand_clashes` | Number of ligand atoms that clash with the generated backbone (distance < 1.5 Å). **Only present if the design includes a ligand and has diffused backbone atoms to compare against**; omitted entirely for ligand-free designs. Should be **0**. |
| `n_clashing.ligand_min_distance` | Minimum distance (Å) between any ligand atom and the generated backbone. Values around 2.5–4.0 Å are typical and indicate the scaffold is close to the ligand without clashing. **Only present under the same condition as `n_clashing.ligand_clashes`.** |
| `helix_fraction` | Fraction of residues in alpha-helices (computed using the P-SEA algorithm). Gives a sense of fold topology. Computed with fixed motif atoms **excluded** — see note below. |
| `sheet_fraction` | Fraction of residues in beta-sheets. Fixed motif atoms excluded, same as `helix_fraction`. |
| `non_loop_fraction` | Fraction of residues in regular secondary structure (helices + sheets). Higher values generally indicate a more structured, designable protein. Fixed motif atoms excluded. |
| `loop_fraction` | Fraction of residues in loops/coils (1 minus `non_loop_fraction`). Fixed motif atoms excluded. |
| `num_ss_elements` | Number of distinct secondary structure elements (individual helices and sheets). Gives a sense of fold complexity. Fixed motif atoms excluded. |
| `radius_of_gyration` | Radius of gyration (Å). Measures how compact the structure is. Smaller values indicate a more globular, tightly packed protein. Fixed motif atoms excluded. |
| `alanine_content` | Fraction of designed residues that are alanine. The backbone-only output tends to default to alanine in many positions; very high values (> 0.4) may indicate regions where the model struggled to build meaningful secondary structure. Includes fixed motif residues by default (see note below). |
| `glycine_content` | Fraction of designed residues that are glycine. High glycine content can indicate flexible or disordered regions. Includes fixed motif residues by default. |
| `num_residues` | Total number of protein residues in the generated design (within the requested length range). Includes fixed motif residues by default. |
| `diffused_com` | Center-of-mass coordinates `[x, y, z]` of the generated (diffused) portion of the protein. |
| `fixed_com` | Center-of-mass coordinates `[x, y, z]` of the fixed motif atoms. **Only present if the design has at least one fixed-coordinate motif atom** (e.g. from `select_fixed_atoms`); omitted entirely for fully unconditional designs. |

```{note}
**Fixed-motif scoping differs between the two halves of this table.** `helix_fraction`, `sheet_fraction`, `non_loop_fraction`, `loop_fraction`, `num_ss_elements`, and `radius_of_gyration` always exclude atoms with fixed coordinates (i.e. the input motif). `alanine_content`, `glycine_content`, `num_residues`, and `diffused_com` include the whole structure (motif + diffused). This means, for a design with a sizeable fixed motif, `helix_fraction` reflects only the diffused region while `num_residues`/`alanine_content` reflect the full structure — keep this in mind when comparing designs with different motif sizes.
```

(hydrogen-bonds)=
### Hydrogen bonds

```{note}
This group is only present when the design was run with H-bond conditioning (spec uses `select_hbond_donor`/`select_hbond_acceptor`), HBPLUS is installed with `HBPLUS_PATH` set, and the H-bond selection matches real atoms. Otherwise the whole group is silently omitted (a `Could not calculate hbond metrics` warning is written to the run log). Computed in [`get_hbond_metrics`](https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/src/rfd3/metrics/hbonds_hbplus_metrics.py).
```

| Metric Name | Description |
| --- | --- |
| `num_hbonds` | Number of hydrogen bonds detected (by HBPLUS) between the motif and diffused regions. ⚠️ This counts **any** detected motif↔diffused hydrogen bond, not just bonds involving the specific atoms you requested via `select_hbond_donor`/`select_hbond_acceptor` — use `correct_donor_percent`/`correct_acceptor_percent` below to check adherence to your requested atoms specifically. |
| `correct_donor_percent` | Fraction of the requested donor atoms that actually formed a hydrogen bond in the design. **Higher is better.** |
| `correct_acceptor_percent` | Fraction of the requested acceptor atoms that actually formed a hydrogen bond in the design. **Higher is better.** |
| `donor_atom_names` | List of the donor atoms that participated in a detected H-bond, each formatted as `{atom}_{resname}_{resid}`. |
| `acceptor_atom_names` | List of the acceptor atoms that participated in a detected H-bond, each formatted as `{atom}_{resname}_{resid}`. |
| `hbond_connections` | List of the detected donor–acceptor pairs, each formatted as `{donor}-{acceptor}` (same per-atom format as above). |

(partial-diffusion-metrics)=
### Partial diffusion metrics

```{note}
Only present for partial-diffusion runs, see {ref}`Partial Diffusion <partial-diffusion>`.
```

| Metric Name | Description |
| --- | --- |
| `ca_rmsd_to_input` | CA RMSD (Å) between the generated design and the original input structure, after rigid alignment. Measures how far partial diffusion moved the backbone from the starting structure. **Only present** when `partial_t` is set and the input and output have a matching number of CA atoms. |

(faq--gotchas)=
## FAQ / Gotchas

<details>
<summary><b>Why don't I see any clash / secondary-structure / composition metrics in my output?</b></summary>

Check whether the run used `cleanup_virtual_atoms=False`. That whole group of metrics (chainbreaks, clashes, SS fractions, radius of gyration, composition, `num_residues`, `diffused_com`/`fixed_com`) requires virtual atoms to have been cleaned up first, since they need a real (non-virtual) atom count. See [Backbone geometry, clashes & fold composition](#backbone-geometry-clashes--fold-composition).
</details>

<details>
<summary><b>Why is `fixed_com` missing from my output?</b></summary>

`fixed_com` is only written when the design has at least one atom with a fixed coordinate (e.g. from a motif or `select_fixed_atoms`). Fully unconditional designs have no fixed atoms, so this key is omitted rather than set to `null`.
</details>

<details>
<summary><b>Why don't I see `insertion_rmsd` / `join_point_rmsd` for my motif?</b></summary>

These are only computed for **unindexed** motif residues (the `unindex` input field). If your motif is fully indexed via `contig`, these keys won't appear — indexed motif placement is deterministic by construction, so there's nothing to measure. See [Insertion & join-point metrics](#insertion--join-point-metrics).
</details>

<details>
<summary><b>Why does `join_point_rmsd_by_token` only list some of my unindexed residues?</b></summary>

`join_point_rmsd` only applies to unindexed residues where **none of the fixed atoms are backbone atoms**. `unindex` selects a residue by chain + residue number, but only the atoms marked fixed via `select_fixed_atoms` are actually carried over from the input — if that residue is fixed as side-chain-only (e.g. `TIP`) or a single atom, no backbone coordinates make it into the token, and it's included in this metric. Residues fixed with at least one backbone atom are excluded from this metric entirely.
</details>

Let us know if you have any additional questions, we'd be happy to answer them either in our [Slack channel](https://join.slack.com/t/proteinmodelfoundry/shared_invite/zt-3kpwru8c6-nrmTW6LNHnSE7h16GNnfLA) or in a GitHub discussion.
