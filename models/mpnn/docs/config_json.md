# JSON input specification

The command line option `--config_json` takes a JSON formatted file to specify the input. 

Values have the same meaning as their command line equivalents.

Only `model_type`, `out_directory` and `structure_path` are required, all other parameters will be set as their defaults.

## Minimal example

Design 3 inputs with default settings 


The config.json file:

```json
{
	"model_type": "protein_mpnn",
	"out_directory": "design",
	"inputs": [
		{"structure_path": "file1.cif"},
		{"structure_path": "file2.cif"},
		{"structure_path": "file3.cif"}
	]
}
```

To run on the command line:

```bash
mpnn --config_json config.json
```

## Full settings

Typical defaults are given.

Top-level settings:

 	{
		"model_type": null,
		"checkpoint_path": null,
		"is_legacy_weights": null,
		"out_directory": null,
		"write_fasta": true,
		"write_structures": true,
		"inputs": [ 
			<INPUTS> 
		]
	}

`inputs` is a list of JSON objects (i.e. dictionaries) which have the parameters for each structure:

	{
		"structure_path": null,
		"name": null,
		"seed": null,
		"batch_size": 1,
		"number_of_batches": 1,
		"remove_ccds": [],
		"remove_waters": null,
		"occupancy_threshold_sidechain": 0.0,
		"occupancy_threshold_backbone": 0.0,
		"undesired_res_names": [],
		"structure_noise": 0.0,
		"decode_type": "auto_regressive",
		"causality_pattern": "auto_regressive",
		"initialize_sequence_embedding_with_ground_truth": false,
		"features_to_return": null,
		"atomize_side_chains": false,
		"fixed_residues": null,
		"designed_residues": null,
		"fixed_chains": null,
		"designed_chains": null,
		"bias": null,
		"bias_per_residue": null,
		"omit": ["UNK"],
		"omit_per_residue": null,
		"pair_bias": null,
		"pair_bias_per_residue_pair": null,
		"temperature": 0.1,
		"temperature_per_residue": null,
		"symmetry_residues": null,
		"symmetry_residues_weights": null,
		"homo_oligomer_chains": null
	}
