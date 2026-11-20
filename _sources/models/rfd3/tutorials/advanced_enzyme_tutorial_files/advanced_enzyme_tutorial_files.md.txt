(adv_enzyme_tutorial_files)=
# Advanced Enzyme Design Tutorial Files

This page indexes the input structures, configuration files, and example outputs that accompany the {doc}`../advanced_enzyme_design_tutorial`. All paths below are relative to this page, found in the RFD3 documentation at `foundry/models/rfd3/docs/tutorials/advanced_enzyme_tutorial_files/`.

```{note}
Your own results will not exactly match the example outputs provided here, but they should be broadly similar.
```

(adv_enzyme_tutorial_files_inputs)=
## Input Structures and Configuration

- [`1qji.cif`](./1qji.cif): The [astacin structure with a bound transition-state analog](https://www.rcsb.org/structure/1QJI) used as the source for the tutorial's [theozyme](../advanced_enzyme_design_tutorial.md#adv_enzyme_tutorial_creating_theozyme).
- [`theozyme.pdb`](./theozyme.pdb): The cropped theozyme (catalytic residues, zinc ion, and PKF ligand) used as the RFD3 input structure for every example on this page.
- [`metalloprotease_rfd3_input.json`](./metalloprotease_rfd3_input.json): The base input specification built in {ref}`Preparing the Configuration File <adv_enzyme_tutorial_preparing_config_file>`, used to generate the [basic design outputs](#adv_enzyme_tutorial_files_basic).

(adv_enzyme_tutorial_files_basic)=
## Basic Design Outputs

Outputs from [`metalloprotease_rfd3_input.json`](./metalloprotease_rfd3_input.json), generated with `dump_trajectories=True`. This is the design discussed in {ref}`Analyzing the Outputs <adv_enzyme_tutorial_analyzing_outputs>`.

Found in [`basic/`](./basic/metalloprotease_rfd3_input_test1_1_model_0.cif.gz):
- [`metalloprotease_rfd3_input_test1_1_model_0.cif.gz`](./basic/metalloprotease_rfd3_input_test1_1_model_0.cif.gz): Final designed structure.
- [`metalloprotease_rfd3_input_test1_1_model_0.json`](./basic/metalloprotease_rfd3_input_test1_1_model_0.json): Quality metrics, index mapping, and full input specification for this design.
- [`metalloprotease_rfd3_input_test1_1_denoised_model_0.cif.gz`](./basic/metalloprotease_rfd3_input_test1_1_denoised_model_0.cif.gz): Denoised trajectory.
- [`metalloprotease_rfd3_input_test1_1_noisy_model_0.cif.gz`](./basic/metalloprotease_rfd3_input_test1_1_noisy_model_0.cif.gz): Noisy trajectory.

(adv_enzyme_tutorial_files_hbond)=
## Hydrogen Bond Conditioning Outputs

Output from adding `select_hbond_donor`/`select_hbond_acceptor` to the base configuration. Discussed in {ref}`Hydrogen Bond Conditioning <adv_enzyme_tutorial_hbond_conditioning>`.

Found in [`hbond/`](./hbond/metalloprotease_rfd3_input_hbond_test1_0_model_0.cif.gz):
- [`metalloprotease_rfd3_input_hbond.md`](./hbond/metalloprotease_rfd3_input_hbond.md): Input specification used to generate this example.
- [`metalloprotease_rfd3_input_hbond_test1_0_model_0.cif.gz`](./hbond/metalloprotease_rfd3_input_hbond_test1_0_model_0.cif.gz): Final designed structure.
- [`metalloprotease_rfd3_input_hbond_test1_0_model_0.json`](./hbond/metalloprotease_rfd3_input_hbond_test1_0_model_0.json): Quality metrics, including `donor_atom_names`, `acceptor_atom_names`, and `hbond_connections`.

(adv_enzyme_tutorial_files_rasa)=
## RASA Conditioning Outputs

Output from adding `select_buried`/`select_exposed` to the base configuration. Discussed in {ref}`RASA Conditioning <adv_enzyme_tutorial_rasa_conditioning>`.

Found in [`rasa/`](./rasa/metalloprotease_rfd3_input_rasa_test1_0_model_6.cif.gz):
- [`metalloprotease_rfd3_input_rasa.json`](./rasa/metalloprotease_rfd3_input_rasa.json): Input specification used to generate this example.
- [`metalloprotease_rfd3_input_rasa_test1_0_model_6.cif.gz`](./rasa/metalloprotease_rfd3_input_rasa_test1_0_model_6.cif.gz): Final designed structure.
- [`metalloprotease_rfd3_input_rasa_test1_0_model_6.json`](./rasa/metalloprotease_rfd3_input_rasa_test1_0_model_6.json): Quality metrics for this design.

(adv_enzyme_tutorial_files_hbond_rasa)=
## Combined Hydrogen Bond and RASA Conditioning Outputs

Output from a configuration that combines both hydrogen bond conditioning and RASA conditioning.

Found in [`hbond_rasa/`](./hbond_rasa/metalloprotease_rfd3_input_hbond_rasa_test1_0_model_0.cif.gz):
- [`metalloprotease_rfd3_input_hbond_rasa.json`](./hbond_rasa/metalloprotease_rfd3_input_hbond_rasa.json): Input specification used to generate this example.
- [`metalloprotease_rfd3_input_hbond_rasa_test1_0_model_0.cif.gz`](./hbond_rasa/metalloprotease_rfd3_input_hbond_rasa_test1_0_model_0.cif.gz): Final designed structure.
- [`metalloprotease_rfd3_input_hbond_rasa_test1_0_model_0.json`](./hbond_rasa/metalloprotease_rfd3_input_hbond_rasa_test1_0_model_0.json): Quality metrics for this design.
