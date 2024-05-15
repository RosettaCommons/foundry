RoseTTAFold All-Atom
--------------------

This repository contains the code to training and running inference on
RoseTTAFold All-Atom (RFAA), a neural network that can predict the structures
of proteins in complex with DNA, RNA, and/or small molecule ligands.

`rf2aa/` contains the model and training code.
`data/` contains code used to curate the training data from the PDB.


## Contributing to RFAA

### Set Up
```
git clone https://git.ipd.uw.edu/jue/RF2-allatom.git
cd RF2-allatom
```

If you are on digs, the S3nv.sif apptainer has all the relevant packages. To get started coding:

```
export PYTHONPATH="../RF2-allatom"
```

First, run the test suite:
```
apptainer exec --nv /software/containers/versions/SE3nv/SE3nv-20240415.sif pytest tests/
```
If all the tests pass, you have a stable version of the code.

### Running model training

We use a package called hydra to configure different training runs of the model. Config files for different training runs can be found in `rf2aa/config/train`. The base trainable version is `rf2aa/config/train/rf2aa.yaml`, to run training with this version, run:
```
/software/containers/versions/SE3nv/SE3nv-20240415.sif trainer_new.py --config-name rf2aa
```
These tests are most often run on a4000s on digs. If you have a separate installation of cifutils in your home directory, this can potentially break the tests.

If you make changes in the code, they should NOT break backwards compatibility, e.g. there should be a flag in the yaml files that would make it as if your changes were never committed. 

### Contributing to model code
Generally, we follow software engineering practices of:
1. Not duplicating functionality that is already in the code
2. Keeping functions as short as possible, and splitting complicated functions into multiple functions
3. Using object oriented programming, which means subclassing already existing classes when possible. 
4. Writing tests for our code and sending small functional PRs for review.
5. Maintaining code stability and not breaking backwards compatibility for users using the package. 

To write new blocks in RF, you can go to the rf2aa/model directory and add the new block into the simulator_blocks.py file (and be sure to add a relevant name in the blocks_factory dictionary). These names can be referenced in hydra configs: see rf2aa.yaml for an example with any keyword arguments necessary to initialize the block.

