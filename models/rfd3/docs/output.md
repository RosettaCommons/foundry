# RFdiffusion3 — Output Metrics

For each design RFD3 creates, a JSON file with metrics that can be used to evaluate the design is also produced. The table below is a complete reference for all metrics produced by RFD3: 

| Metric Name | Description |
| --- | --- |
| `insertion_rmsd` | **Insertion RMSD** (Å). Measures how accurately the unindexed motif residues were placed into the generated backbone, averaged over all motif residues. This is the root-mean-square deviation between the original motif atom positions and their matched positions in the diffused structure. **Lower is better**; values below ~0.5 Å indicate good placement. |
| `insertion.mae` | **Insertion MAE** (Å). Mean absolute error of the motif atom placement. Similar to `insertion_rmsd` but less sensitive to outliers. **Lower is better.** |
| `insertion.rmcd` | **Insertion RMCD** (Å). Root mean cubic deviation of the motif atom placement. More sensitive to large deviations than RMSD. **Lower is better.** |
| `join_point_rmsd` | **Join point RMSD** (Å). Measures backbone accuracy at the points where motif residues connect to the generated scaffold, averaged over all motif residues. **Lower is better**; indicates how smoothly the motif was integrated. |
| `join_point_rmsd_by_token` | Per-residue breakdown of the join point RMSD. Useful for identifying which specific residues had poor backbone joins. |
| `n_conjoined_residues` | Number of motif residues that mapped to multiple diffused positions. Should be **0** for a clean design. |
| `n_chainbreaks` | Number of backbone breaks where the CA-CA distance deviates significantly from the ideal 3.8 Å. Should be **0** for a clean design. |
| `max_ca_deviation` | Largest CA-CA distance deviation from the standard 3.8 Å bond length. **Lower is better**; very small values (< 0.1 Å) indicate a clean backbone trace. |
| `n_clashing.interresidue_clashes_w_sidechain` | Number of interresidue steric clashes involving sidechain atoms (distance < 1.5 Å between non-neighboring residues). Should be **0** or very low. |
| `n_clashing.interresidue_clashes_w_backbone` | Number of interresidue steric clashes involving only backbone atoms. Should be **0**. |
| `n_clashing.ligand_clashes` | Number of ligand atoms that clash with the generated backbone (distance < 1.5 Å). Should be **0**. |
| `n_clashing.ligand_min_distance` | Minimum distance (Å) between any ligand atom and the generated backbone. Values around 2.5–4.0 Å are typical and indicate the scaffold is close to the ligand without clashing. |
| `helix_fraction` | Fraction of residues in alpha-helices (computed using the P-SEA algorithm). Gives a sense of fold topology. |
| `sheet_fraction` | Fraction of residues in beta-sheets. |
| `non_loop_fraction` | Fraction of residues in regular secondary structure (helices + sheets). Higher values generally indicate a more structured, designable protein. |
| `loop_fraction` | Fraction of residues in loops/coils (1 minus `non_loop_fraction`). |
| `num_ss_elements` | Number of distinct secondary structure elements (individual helices and sheets). Gives a sense of fold complexity. |
| `radius_of_gyration` | Radius of gyration (Å). Measures how compact the structure is. Smaller values indicate a more globular, tightly packed protein. |
| `alanine_content` | Fraction of designed residues that are alanine. The backbone-only output tends to default to alanine in many positions; very high values (> 0.4) may indicate regions where the model struggled to build meaningful secondary structure. |
| `glycine_content` | Fraction of designed residues that are glycine. High glycine content can indicate flexible or disordered regions. |
| `num_residues` | Total number of protein residues in the generated design (within the requested length range). |
| `diffused_com` | Center-of-mass coordinates [x, y, z] of the generated (diffused) portion of the protein. |
| `fixed_com` | Center-of-mass coordinates [x, y, z] of the fixed motif atoms. |
| `num_hbonds` | Number of hydrogen bonds detected (by HBPLUS) among the conditioned donor/acceptor atoms. **Only present** when the design was run with H-bond conditioning (spec uses `select_hbond_donor`/`select_hbond_acceptor`), HBPLUS is installed with `HBPLUS_PATH` set, and the H-bond selection matches real atoms. Otherwise the whole H-bond group is silently omitted (a `Could not calculate hbond metrics` warning is written to the run log). |
| `correct_donor_percent` | Fraction of the requested donor atoms that actually formed a hydrogen bond in the design. **Higher is better.** **Only present** under the same H-bond conditions as `num_hbonds`. |
| `correct_acceptor_percent` | Fraction of the requested acceptor atoms that actually formed a hydrogen bond in the design. **Higher is better.** **Only present** under the same H-bond conditions as `num_hbonds`. |
| `donor_atom_names` | List of the donor atoms that participated in a detected H-bond, each formatted as `{atom}_{resname}_{resid}`. **Only present** under the same H-bond conditions as `num_hbonds`. |
| `acceptor_atom_names` | List of the acceptor atoms that participated in a detected H-bond, each formatted as `{atom}_{resname}_{resid}`. **Only present** under the same H-bond conditions as `num_hbonds`. |
| `hbond_connections` | List of the detected donor–acceptor pairs, each formatted as `{donor}-{acceptor}` (same per-atom format as above). **Only present** under the same H-bond conditions as `num_hbonds`. |
| `ca_rmsd_to_input` | CA RMSD (Å) between the generated design and the original input structure, after rigid alignment. Measures how far partial diffusion moved the backbone from the starting structure. **Only present** for partial-diffusion runs (`partial_t` is set) when the input and output have a matching number of CA atoms. |