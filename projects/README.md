# Modelhub: Projects

TODO: README, describing how projects work

The RFScore code can act as an example.

To use
```
export PROJECT_PATH="/home/<USER>/modelhub/projects/<PROJECT>"
```

When we call `train.py` or `validate.py`, we will now add the `configs` directory from the project (e.g., `projects/<PROJECT>/configs`) to the Hydra search path (infront of the base configs). 

Further, we will look for any specialized Apptainer `.sif` files in `projects/<PROJECT>/scripts/shebang`, if present; otherwise, we will use the default `modelhub` apptainer.
