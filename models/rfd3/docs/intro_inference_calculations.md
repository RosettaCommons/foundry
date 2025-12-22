# Understanding Inference Inputs
In RFdiffusion3 (RFD3), [YAML](https://yaml.org/) or [JSON](https://www.json.org/json-en.html) files are used to specify the **settings** for your inference calculations and [**configuration options**](https://hydra.cc/docs/configure_hydra/intro/) are used to provide other information about your calculation, such as the location and name of the checkpoint file you want to use. 

## Inference Settings
The inference 'settings' are how you constrain your inference calculation, such as specifying portions of the output you wish to have defined or taken from an input structure (`contig`) and specifying any symmetries that exist in your system (`symmetry`). These settings are stored in either YAML or JSON files to be interpreted by RFdiffusion3. We will briefly discuss these file formats here and runnable examples of these files can be found in `foundry/models/rfd3/docs`.

Using this type of input specification allows you to define different types of inference calculations all in the same file, and either run all of the calculation types defined in the file or specify the specific calculation you want to run via the command line. 

```{note}
For more information on many of the available options, see {doc}`input`. To see all available options, see [input_parsing.py](https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/src/rfd3/inference/input_parsing.py). 
```

### JSON File Format
In a JSON file different groupings of settings are denoted by curly braces (`{}`), colons (`:`) are used to specify the chosen setting for each setting type, and commas are used to separate different settings. Any strings included in the file need to be in quotes.

To start, the contents of the entire file should be surrounded by curly braces. To start a new group of settings you want to use for an inference run, give it a name (in quotes) then use a colon and an open brace to denote that you are starting a new group of settings related to that name:
```json
{
    "inference_calculation_1":{
        <Your settings go here.>
    }
}
```

If you have more than one inference calculation defined in your JSON file, make sure to add a comma between different groups: 
```json
{
    "inference_calculation_1":{
        <Your settings go here.>
    },
     "inference_calculation_2":{
        <Your settings go here.>
    }
}
```
Some settings will be one line while others will require defining a dictionary, leading to another nested set of braces: 
```json
{
    "inference_calculation_1":{
        "string_setting": "my_string_1",
        "boolean_setting": true,
        "number_setting": -27.4,
        "dictionary_setting":{
            "key1": 2,
            "key2": false,
            "key3": "my_string_2, my_string_3"
        }
    }
}
```
There is no syntax for comments in JSON files - you can't comment out a line or add notes. This results in very skimmable and easy to read, but might feel limiting for taking notes or troubleshooting.

### YAML File Format
YAML files use indentation to denote different groupings, however **tab characters are not allowed**. Comments in YAML files are denoted with `#` and can be anywhere in the line. List members are denoted using hyphens (`-`) or by enclosing the items in square brackets (`[]`) and separated by commas. For the purposes of RFD3, strings do not need to be inclosed by quotes to be parsed correctly. The inference settings in RFD3 are interpreted as key value pairs, so when you are defining your groups, make sure to use colons (`:`) to denote a relationship between the setting name and the value you want to give it. 

To start a set of inference settings, name the group and then include your settings in an indented chunk below it. For multiple inference calculations, go back one indentation level to name the new group and then continue on as before: 
```yaml
# Information about what is stored in this YAML file
inference_calculation_1: # Information about calculation 1
    <Your settings go here.>

inference_calculation_2: 
    <Your settings go here.>
```
And here's what your YAML file will look like with some settings added:
```yaml
inference_calculation_1: 
    string_setting:  my_string
    boolean_setting: true
    number_setting: -27.4
    list_setting: 
        - 0.0
        - 1.0
        - 2.0
    dictionary_setting: 
        key1: value1
        key2: value2
        key3: value3
```

## Job configurations
Once you have all of the settings you want to use to constrain your inference run in a JSON or YAML file, you can run the job using a command starting with `rfd3 design` and then including different 'configuration options'. You must include the path to the YAML/JSON file that defines your inference run(s) and the output directory in this command: 
```bash
rfd3 design inputs=/path/to/your/yaml/or/json/file out_dir=/path/to/your/output/directory ckpt_path=/path/to/an/rfd3_checkpoint_file.pt
```

```{note}
The output directory location specified will be created if it does not exist. This setting only specifies the location the output files will be stored in, not the naming of the various output files.
```

Several other options are available to you as well to control things like the number of designs, whether to save the trajectory files, etc. These options can be found in [`foundry/models/rfd3/configs/inference_engine/base.yaml`](https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/configs/inference_engine/base.yaml) and [`foundry/models/rfd3/configs/inference_engine/rfdiffusion3.yaml`](https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/configs/inference_engine/rfdiffusion3.yaml)

## Output Files
At the end of your inference calculation, you will be left with several output files in the directory you specified. At minimum (if you did not change any settings to include more outputs) you will be left with a JSON and a compressed CIF file (`.cif.gz`) for each design. The names of the files will be as follows: 
```bash
<name of the json or yaml file>_<settings group name>_0_model_n.<suffix>
```
Where `n` is the design number, the numbering for the designs will start at 0. 

For an example, if I called the my JSON file `rfd3_example.json` and had a group of settings in it labeled `example_1` I would get files with names like: 
```bash
rfd3_example_example_1_0_model_0.cif.gz
rfd3_example_example_1_0_model_0.json
rfd3_example_example_1_0_model_1.cif.gz
rfd3_example_example_1_0_model_1.json
...
```