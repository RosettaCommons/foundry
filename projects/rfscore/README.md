# RFScore

RFScore is an extension of our internal AlphaFold-3 clone tailored for filtering and scoring designs.

Currently, the main differences to the main model are:
- Atom Names: Unlike in AF-3, in RFScore the atom names chosen for small molecules have no effect
- Small Molecule Templates: If specified with the `template_selection_strings`, small molecules may use the ground truth coordinates for their reference conformer. This approach may be helpful when the ligand geometry is known but difficult to model without templating the small molecule directly (e.g., in the case of high-energy transition states)

## Inference
```bash
./src/modelhub/inference.py inference_engine=rfscore inputs="/projects/rfscore/assets/TS1_trp_6_conformers_0008_000_10-atomized-bb-False_4_9_MPNN.pdb" residue_renaming_dict="{TS1: L:1}" template_selection_strings="[B/L:1]" 
```