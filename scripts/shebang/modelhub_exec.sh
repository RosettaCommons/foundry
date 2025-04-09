#!/usr/bin/bash

###################
# You can add the path to this file as the shebang line in your python script. 
# Then by default, the python script will be executed with the python interpreter
# in the SIF_PATH container. Here, we launch the container with nvidia gpu and slurm support.
#
# Example shebang: #!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/scripts/shebang/modelhub_exec.sh" "$0" "$@"'
###################

# Let the user know this script is setting things up behind the scene
SCRIPT_PATH=$(realpath $0)
SCRIPT_DIR=$(dirname $SCRIPT_PATH)
echo '################## Start shebang info ##################'
echo "The file $SCRIPT_PATH is being run as a shebang executable.
    It will...

    1. Add the 'modelhub' and 'src/modelhub' repo directories to your PYTHONPATH.
    2. Run your python script from the right container, which contains all dependencies.
    3. Launch the container with slurm and nvidia gpu support."

# Extract the path to the Python script from the arguments
PYTHON_SCRIPT=$(realpath "$1")
shift

# Find repository root by looking for .project-root file
find_repo_root() {
    local current_dir="$1"
    while [ "$current_dir" != "/" ]; do
        if [ -f "$current_dir/.project-root" ]; then
            echo "$current_dir"
            return 0
        fi
        current_dir="$(dirname "$current_dir")"
    done
    return 1
}

echo
echo "Searching for repository root directory..."
REPO_ROOT=$(find_repo_root "$(dirname "$PYTHON_SCRIPT")")
if [ -z "$REPO_ROOT" ]; then
    echo "Error: Could not find .project-root file in any parent directory"
    exit 1
else
    echo "... found repository root at '$REPO_ROOT'"
fi

# Function to add a directory to PYTHONPATH if it's not already included
add_to_pythonpath() {
    local dir_path="$1"
    if [[ ":$PYTHONPATH:" != *":$dir_path:"* ]]; then
        export PYTHONPATH="$dir_path:$PYTHONPATH"
        echo "Added '$dir_path' to PYTHONPATH."
    else
        echo "'$dir_path' is already in PYTHONPATH."
    fi
}

# Add the src directory to PYTHONPATH if not already present
echo
echo "Checking and adding 'src' directory to PYTHONPATH..."
SRC_PATH="$REPO_ROOT/src"
add_to_pythonpath "$SRC_PATH"

# Add modelhub to PYTHONPATH if not already present
echo
echo "Checking and adding 'modelhub' directory to PYTHONPATH..."
MODELHUB_PATH="$SRC_PATH/modelhub"
add_to_pythonpath "$MODELHUB_PATH"

# Load the .env file environment variables from the repo root
echo
echo "Attempting to load environment variables from .env file:"
if [ -f "$REPO_ROOT/.env" ]; then
    echo "... loading environment variables from '$REPO_ROOT/.env'"
    export $(cat "$REPO_ROOT/.env" | grep -v '#' | xargs)
else
    echo " Warning: No .env file found at repository root ($REPO_ROOT)"
fi

# check if we are at the IPD
IPD_FILE="/software/containers/versions/rf_diffusion_aa/ipd.txt"

SIF_PATH=""

echo
echo "Fetching the appropriate apptainer image..."

if [ -z "$APPTAINER_NAME" ]; then
    if [ -n "$PROJECT_PATH" ]; then
        # Attempt to find any .sif file in the PROJECT_PATH/scripts/shebang directory
        SIF_DIR="$PROJECT_PATH/scripts/shebang"
        SIF_FILE=$(find "$SIF_DIR" -maxdepth 1 -name "*.sif" -print -quit)

        if [ -n "$SIF_FILE" ]; then
            SIF_PATH="$SIF_FILE"
        fi
    fi

    # If SIF_PATH is still empty, use the default SIF
    if [ -z "$SIF_PATH" ]; then
        SIF_NAME="modelhub.sif"
        SIF_PATH="$SCRIPT_DIR/$SIF_NAME"
    fi

    echo "... looking for a local apptainer image at '$SIF_PATH'"
    # Check if the SIF file exists
    if [ ! -f "$SIF_PATH" ]; then
        echo "... apptainer not found. To run with your own apptainer image, you can build it with 'make apptainer' and place it here: '$SIF_PATH'"
        echo "Attempting to run $PYTHON_SCRIPT with $(which python)"
    fi
else
    echo "Already running inside container $APPTAINER_NAME. Executing $PYTHON_SCRIPT with $(which python) in the existing container."
fi

# Function to print debug=mode warning
print_debug_warning() {
    echo
    echo "###############################################################################"
    echo "#                                                                             #"
    echo "#                               ⚠️  WARNING ⚠️                                  #"
    echo "#                     RUNNING WITH DEBUGPY ON PORT $DEBUG_PORT                       #"
    echo "#                      DON'T FORGET TO ATTACH A DEBUGGER                      #"
    echo "#                                                                             #"
    echo "###############################################################################"
    echo
}

if [ -n "$DEBUG_PORT" ]; then
    print_debug_warning
    python_cmd="python -m debugpy --listen $DEBUG_PORT --wait-for-client"
else
    python_cmd="python"
    echo
fi

if [ ! -z $SIF_PATH ]; then
    echo "Running $PYTHON_SCRIPT with apptainer: $SIF_PATH."
    echo '################## End shebang info ####################'
    echo
    /usr/bin/apptainer exec --nv --slurm \
        --bind "$REPO_ROOT:$REPO_ROOT" \
        --env PYTHONPATH="\$PYTHONPATH:$PYTHONPATH" \
        $SIF_PATH $python_cmd "$PYTHON_SCRIPT" "$@"
else
    echo "Running $PYTHON_SCRIPT with python: $(which python)"
    echo '################## End shebang info ####################'
    echo
    $python_cmd "$PYTHON_SCRIPT" "$@"
fi
