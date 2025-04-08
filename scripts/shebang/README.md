This directory contains scripts that are not to be run directly by the user. 
They are [SHEBANG scripts](https://en.wikipedia.org/wiki/Shebang_(Unix)) that are used to run the appropriate apptainer container.

For example, the script `modelhub_exec.sh` is used to run the modelhub apptainer container with the latest apptainer image
stored locally or at the IPD.

The shebang lines (`#!/bin/bash` ...) at the top of entry point scripts like `train.py` redirect the system to here to find the correct apptainer container.