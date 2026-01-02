# Inference Calculation Basics
In RFdiffusion3 (RFD3), [YAML](https://yaml.org/) or [JSON](https://www.json.org/json-en.html) files are used to specify the **settings** for your inference calculations and [**configuration options**](https://hydra.cc/docs/configure_hydra/intro/) are used to provide other information about your calculation, such as the location and name of the checkpoint file you want to use. 

## Inference Settings
The inference 'settings' are how you constrain your inference calculation, such as specifying portions of the output you wish to have designed (`contig`) and specifying any symmetries that exist in your system (`symmetry`). These settings are stored in either a YAML or JSON file to be interpreted by RFdiffusion3. Runnable example sof json and yaml files can be found in `foundry/models/rfd3/docs`.

Using this type of input specification allows you to define different types of inference calculations all in the same file, and either run all of the calculation types defined in the file or specify the specific calculation you want to run via the command line. 

```{note}
For more information on many of the available options, see {doc}`input`. To see all available options, see [input_parsing.py](https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/src/rfd3/inference/input_parsing.py). 
```

## Job configurations
Once you have all of the settings you want to use to constrain your inference run in a JSON or YAML file, you can run the job using a command starting with `rfd3 design` and then including different 'configuration options'. You must include the path to the YAML/JSON file that defines your inference run(s) and the output directory: 
```bash
rfd3 design inputs=/path/to/your/yaml/or/json/file out_dir=/path/to/your/output/directory ckpt_path=/path/to/an/rfd3_checkpoint_file.pt
```

```{note}
The output directory location specified will be created if it does not exist. This setting only specifies the location the output files will be stored in, not the naming of the various output files.
```

Several options are available to you as well to control the number of designs, whether to save the trajectory files, etc. These options can be found in [`foundry/models/rfd3/configs/inference_engine/base.yaml`](https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/configs/inference_engine/base.yaml) and [`foundry/models/rfd3/configs/inference_engine/rfdiffusion3.yaml`](https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/configs/inference_engine/rfdiffusion3.yaml)

## Output Files
At the end of your inference calculation, you will be left with several output files in the directory you specified. At minimum (if you did not change any settings to include more outputs) you will be left with a JSON and a compressed CIF file (`.cif.gz`) for each design. The names of the files will be as follows: 
```bash
<name of the json or yaml file>_<settings group name>_<batch_number>_model_n.<suffix>
```
Where `n` is the design number, the numbering for the designs will start at 0. 

For an example, if I called the my JSON file `rfd3_example.json`, only ran one batch, and had a group of settings in it labeled `example_1` I would get files with names like: 
```bash
rfd3_example_example_1_0_model_0.cif.gz
rfd3_example_example_1_0_model_0.json
rfd3_example_example_1_0_model_1.cif.gz
rfd3_example_example_1_0_model_1.json
...
```
