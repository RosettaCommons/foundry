# Modelhub

- [Modelhub](#modelhub)
  - [Background](#background)
  - [Division of code between Modelhub, Datahub, and Cifutils](#division-of-code-between-modelhub-datahub-and-cifutils)
    - [Cifutils](#cifutils)
    - [Datahub](#datahub)
  - [Training, Validation, and Inference](#training-validation-and-inference)
    - [Training and Validation](#training-and-validation)
    - [Inference](#inference)
  - [Setup](#setup)
    - [Apptainers](#apptainers)
      - [Base Apptainer](#base-apptainer)
      - [Frozen Apptainer](#frozen-apptainer)
    - [Shebang](#shebang)
      - [General Use](#general-use)
      - [Debugging](#debugging)

## Background

This repository constitutes the base for deep-learning method development at the Institute for Protein Design.

It is symbiotic with two other Institute for Protein Design repositories:
- [cifutils](https://github.com/baker-laboratory/cifutils), which manages input parsing and data cleaning
- [datahub](https://github.com/baker-laboratory/datahub), which manages input featurization and holds our composable `Transform` components

Within this ontology, `modelhub` contains the *architectures*, *training* code, and *inference* endpoints.

## Division of code between Modelhub, Datahub, and Cifutils

Across our codebases, we balance the need to develop quickly with the need to write code that we can continue to maintain and that is easy to understand. We below lay out some thoughts on what code should live where.

We enforce a strict dependency flow of `modelhub` -> (depends on) `datahub` -> (depends on) `cifutils`; it would be a circular anti-pattern to thus import any `datahub` or `modelhub` functions from within `cifutils`. 

### Cifutils

[cifutils](https://github.com/baker-laboratory/cifutils) is the most static of our three codebases. Basic parsing functionality, RDKit and other molecular toolkit utilities, and `AtomArray` quality-of-life tools live in this repository. 

Examples of `cifutils` functions are:
- All functions related to **parsing structural files from source**; e.g., keeping/removing hydrogens, resolving occupancy, etc.
- Utility functions to manipulate `AtomArrays`, the core API of the `biotite` library, upon which we heavily rely
- Utility functions for common bioinformatics software, such as `RDKit`, that interface with `AtomArrays`

As a foundational library for the Institute for Protein Design, `cifutils` functions most like an open-source codebase. We must keep the code easy-to-understand and easy-to-maintain, both now and into the future. As such, `cifutils`:
- Maintains the **highest code quality standard**, requiring well-documented, easy-to-maintain code with adequate test coverage (we aim for **>85%** coverage)
- **Strictly versions** to minimize breaking changes with downstream repositories

You should write code in `cifutils` if:
- You are are writing **core** `AtomArray`-level level functionality that will be broadly useful, not only to those at the Institute for Protein Design but possibly the wider bioinformatics community (i.e., without dependencies, or even knowledge of, `datahub` or `modelhub`)
- You are willing to spend some additional time to ensure the code is **scalable, well-tested, and maintainable**

Quick-and-dirty experiments that require modifying `cifutils` can be performed by submoduling or cloning the repository and exporting a local path.

### Datahub

[datahub](https://github.com/baker-laboratory/datahub) manages data loading, preprocessing, and featurization pipelines for structure-dependent deep-learning models. We offer three core components: a `Transforms` library, a set of `Preprocessing` scripts, and `Datasets`.
- **Transforms**: A series of composable classes that take as input a dictionary containing sequence- and structure-based data (in the form of an `AtomArray`) and perform arbitrary operations, analogous to TorchVision's [approach](https://pytorch.org/vision/main/transforms.html) for computer vision 
- **Preprocessing**: Scripts and functions for common data cleaning and preparation tasks, including specialized pipelines for frequent use cases (e.g., antibodies, clash detection, cleaning PDB data, etc.). Many of these *scripts* output `parquet` files stored to disk that are sampled from at train-time, while the *functions* are called by the scripts to clean, label, or filter the data (e.g., `has_clash()`, etc.)
- **Datasets**: The base `Datasets` and `Sampler` classes used for training, imported by `modelhub`

`datahub` is less static than `cifutils`; however, it still must operate as a stand-alone library that others can continue to build around and upon, even without `modelhub`. We strive to maintain `datahub` like an open-source software project such that others in the lab can easily understand, and build upon, our base components. We focus on **maintainable** and **flexible** code - if a particular `Transform` is bespoke or non-generalizable (at least initially), then the `/projects` folder within `Modelhub` may be a more appropriate place for initial development. 

You should write code in `datahub` if:
- You are writing flexible, generic *pre-processing scripts* or *functions* that others in the lab have expressed interest in using (vs. a single-purpose pipeline or feature to test a hypothesis)
    - **Example that should live in `datahub`**: You are writing a pre-processing pipeline to label all beta barrels in the PDB. Your scripts, written in a functional manner, may be a good candidate for `datahub/scripts/preprocessing`, so long as you are willing to write them generally and include tests. Similarly, if a single function may be generalizable but the pipeline is bespoke, that single function (with a test) could still be included as a stand-alone element in `datahub`, e.g.,
    ```python
    atom_array_has_beta_barrel(atom_array: AtomArray) -> bool
    ```
    - **Example that should live in `modelhub/projects`**: You have pulled together a script that loads PDB files, includes manual annotations, and saves out to CIF. Such a script may be appropriate for the specific use case but is unlikely to generalize across other use cases. 
- You are writing `Transforms` that generalize to additional use cases beyond the current project
    - **Example that should live in `datahub`**: Any `Transform` that adds a useful annotation to an `AtomArray` (e.g., annotationg pocket residues, hydrogen bonds, SASA, etc.)
    - **Example that should live in `datahub`**: A `Transform` that pads DNA with generated B-form structure, as is done in AF-3; such a `Transform` may be applicable to both structure prediction and design, when proven effective
    - **Example that should live in `modelhub/projects`**: A `Transform` that aggregates and/or concatenates features for a bespoke model pipeline
- You are willing to spend some additional time to ensure the code is scalable, well-tested, and maintainable. Otherwise the `projects` folder of `modelhub` may be a more appropriate place in the interim

## Training, Validation, and Inference

> If you are developing at the IPD, our `shebang` executables will take care of identifying and executing with the most up-do-date apptainer. If you are not at the IPD, you will need to ensure you have the appropriate apptainer. See below for details.

NOTE: For Training, Validation, and Inference, we make heavy use of [Hydra](https://hydra.cc/) for configuration management.

Before running any of the below commands, you will need to ensure `datahub` and `cifutils` are in your `PYTHONPATH`. E.g.,
```
export PYTHONPATH="/home/<USER>/projects/datahub/src:/home/<USER>/projects/cifutils/src"
```

### Training and Validation

For Training and Validation, when you execute `train.py` or `validate.py`, you will need to provide an *experiment* Hydra config. Experiments are a Hydra best-practice pattern to enable us to maintain multiple configurations; see more in the [Hydra documentaion](https://hydra.cc/docs/patterns/configuring_experiments/)
and in the `configs/experiment` sub-directory.

For example, to test AF-3 training without confidence, run:
```
./src/modelhub/train.py experiment=quick-af3 debug=default
```

**Explanation:**
- `./src/modelhub/train.py` —  we execute our `train.py` like a bash executable, which triggers the `shebang` code to find the correct apptainer. It's equivalent to `apptainer exec --nv /path/to/apptainer python ./src/modelhub/train.py`
- `experiment=quick-af3` — we identify the experiment we want to use for training; in this case, `quick-af3`, which can be viewed at `configs/experiment/quick-af3.yaml`. This experiment is a simple test config for AF-3 that loads and runs more rapidly that the full training config
- `debug=default` - a setting letter Hydra know we are debugging; when we debug, we perform some automatic time-savings like setting a small diffusion batch size and crop size. You could remove this line if you don't want those options. You can explore more about various `debug` options in `config/debug`

For validation only, run the following:
```
./src/modelhub/validate.py experiment=quick-af3 debug=default
```

Note that since we use `hydra`, you could specify additional setup arguments using the command line. For example, by default, we `prevalidate` - running validation at the beginning of training so we develop a baseline and catch any errors (especially out-of-memory errors) before training for a full epoch. If you don't want that behavior, you could override in-line:
```
./src/modelhub/train.py experiment=quick-af3 debug=default trainer.prevalidate=false
```

You can view the flattened Hydra configuration to determine how to best override or add additional arguments by:
- Running training or validation and viewing the pretty-printed file, which looks like:
![alt text](assets/example_config.png)
- Adding `--cfg job` to your launch command, which prints the config for the application and then exits

### Inference

To support multiple models and multiple projects, we build an `InferenceEngine` for each use case. For end-users the details of the `InferenceEngine` are not necessary; the appropriate engine can be specified with with `inference_engine` argument.

For example, to run the latest AF-3 model with confidence, we can execute (if `cifutils` and `datahub` are in the `PYTHONPATH`):
```
./src/modelhub/inference.py inference_engine=af3 inputs='./tests/data/example_with_ncaa.json'
```

We can then modify the command by adding/removing arguments with Hydra to our liking; for example, to dump diffusion trajectories and only include one model per CIF file:
```
./src/modelhub/inference.py inference_engine=af3 inputs='./tests/data/example_with_ncaa.json' dump_trajectories=true one_model_per_file=true
```

More details can be found in the [inference README](src/modelhub/inference_engines/README.md)

## Setup

> If you are developing at the IPD, then our `shebang` executables will handle the Apptainer dependencies; no need to run the commands below. See the `shebang` section below.

### Apptainers
To accelerate development and better contain dependencies, we offer two apptainers:
- `base_apptainer`: Contains all of the development dependencies, pre-compiled DeepSpeed, but *NOT* `cifutils` or `datahub`. The rationale for the base apptainer is that you expose these libraries via your PYTHONPATH/PATH to allow you to develop & pull updates for these libraries without having to re-build any apptainer.
- `freeze_apptainer`: Takes the `base_apptainer` as its image, and adds versioned `cifutils`, `datahub`, and (optionally) pip-installs `modelhub` as well (useful for releasing self-contained inference code). The rationale for these apptainers is to provide designers with a stable environment to tackle design problems in.

#### Base Apptainer

To make the base apptainer, run:
```
make base_apptainer
```
from the project root. This container will **not** contain `cifutils` or `datahub`; those paths must be exported explicitly during development (e.g., the paths to their respective submodules or clones elsewhere). 

Building this apptainer pre-compiles DeepSpeed, among other actions, and is slow. You **should not** need to re-build this apptainer often; changes to `datahub` and `cifutils` can be addressed much more efficiently through the `freeze_apptainer` command specified below.

> NOTE: Since we pre-compile CUDA-specific DeepSpeed, you must run `make base_apptainer` on a GPU node

> NOTE: You will need to adjust the IPD-speciifc paths to frozen copies of the PDB and the CCD

#### Frozen Apptainer

To make a container that contains `cifutils` and `datahub`, but not `modelhub`, run:
```
make freeze_apptainer
```
This will use the `base_apptainer` pointed to by the `shebang` symlink as a base. Note that by default the versions of `cifutils` and `datahub` are fixed; update the `freeze_apptainer.spec` file to adjust the version numbers and/or add dependencies.

To make a container that contains `modelhub`, `datahub`, and `cifutils` (e.g., for production usage across the lab), run 
```
make INSTALL_PROJECT=true freeze_apptainer
```
> NOTE: Since we build from the `base_apptainer` image, which contains pre-compiled DeepSpeed, `make freeze_apptainer` does NOT need to be run from a GPU

### Shebang

#### General Use
We use `shebang` to help manage and version apptainers. Namely:
- The shebang lines (`#!/bin/bash` ...) at the top of entry point scripts like `train.py` redirect the system to to `scripts/shebang/modelhub_exec.sh`
- The script `modelhub_exec.sh` in turn identifies the correct Apptainer and executes your command
- Apptainers are symlinks in `scripts/shebang` to elsewhere on the DIGS (where they are versioned); thus, when we update apptainers, we must also update the symlink. This allows us to track which apptainers to use for a given branch of the code at any given time (provided you update the symlinks for your branch when you switch out which apptainer you run with!)

For example, to launch a dummy training run, one could type (after adding `cifutils` and `datahub` to your `PYTHONPATH`):
```
cd src/modelhub
./train.py experiment=none-00-dummy
```
> You may need to adjust the permissions on `train.py` (e.g., `chmod +x train.py`) in order to execute the file like a script.

#### Debugging
We also support VSCode-native debugging with Apptainers. To debug:
1. Update your `launch.json` to include `Python: Attach`; for example, add the configuration:
    ```
        {
            "name": "Python: Attach",
            "type": "debugpy",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 2345
            }
        }
    ```
2. Add any interactive debug breakpoints in VSCode
3. Set the `DEBUG_PORT` to `2345`, and then execute your script with `shebang` like normal. That is:
    ```
    export DEBUG_PORT=2345
    ./train.py experiment=none-00-dummy
    ```
4. When prompted in the termal, launch the VSCode debug session (shortcut: `F5`)

Happy debugging!



