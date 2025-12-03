#!/usr/bin/bash

###################
# You can add the path to this file as the shebang line in your python script. 
# Then by default, the python script will be executed with the python interpreter
# in the SIF_PATH container. Here, we launch the container with nvidia gpu and slurm support.
#
# Example shebang: #!/usr/bin/env -S /bin/sh -c '"$(dirname "$0")/.ipd/shebang/rf3_exec.sh" "$0" "$@"'
###################

# Let the user know this script is setting things up behind the scene
SCRIPT_PATH=$(realpath $0)
SCRIPT_DIR=$(dirname $SCRIPT_PATH)
echo '################## Start shebang info ##################'
echo "The file $SCRIPT_PATH is being run as a shebang executable.
    It will...

    1. Add 'src/foundry', 'models/rf3/src', and 'lib/atomworks/src' to your PYTHONPATH.
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

# Add foundry, rfd3, and atomworks to PYTHONPATH
echo
echo "Adding foundry, RFD3, and atomworks to PYTHONPATH..."
FOUNDRY_PATH="$REPO_ROOT/src"
RFD3_PATH="$REPO_ROOT/models/rfd3/src"
ATOMWORKS_PATH="$REPO_ROOT/lib/atomworks/src"
add_to_pythonpath "$FOUNDRY_PATH"
add_to_pythonpath "$RFD3_PATH"
add_to_pythonpath "$ATOMWORKS_PATH"

echo
echo "Fetching the appropriate apptainer image..."

SIF_PATH="$REPO_ROOT/.ipd/apptainer/rfd3-full.sif"

echo "... looking for a local apptainer image at '$SIF_PATH'"
if [ ! -f "$SIF_PATH" ]; then
    echo "... apptainer not found. To build it, run: apptainer build .ipd/apptainer/rf3-dev.sif .ipd/apptainer/rf3-dev.def"
    echo "Attempting to run $PYTHON_SCRIPT with $(which python)"
    SIF_PATH=""
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

    # Determine GPU binding method: prefer --nvccli, fallback to --nv, then no GPU
    GPU_FLAG=""
    if command -v nvidia-container-cli &> /dev/null && [ -c /dev/nvidiactl ] 2>/dev/null; then
        GPU_FLAG="--nvccli"
        echo "Using --nvccli for GPU support"
    elif [ -c /dev/nvidiactl ] 2>/dev/null; then
        GPU_FLAG="--nv"
        echo "Using --nv for GPU support"
    else
        echo "No GPU detected, running without GPU support"
    fi

    echo '################## End shebang info ####################'
    echo
    /usr/bin/apptainer exec $GPU_FLAG --slurm \
        --bind "$REPO_ROOT:$REPO_ROOT" \
        --env PYTHONPATH="\$PYTHONPATH:$PYTHONPATH" \
        $SIF_PATH $python_cmd "$PYTHON_SCRIPT" "$@"
else
    echo "Running $PYTHON_SCRIPT with python: $(which python)"
    echo '################## End shebang info ####################'
    echo
    $python_cmd "$PYTHON_SCRIPT" "$@"
fi
