#!/bin/bash
# This script freezes CIFUtils, Datahub, and Modelhub versions within an existing apptainer.
set -e  # Exit on error

echo "Running from $PWD"

SCRIPT_PATH=$(realpath $0)
SCRIPT_DIR=$(dirname $SCRIPT_PATH)

# Check if apptainer/singularity is available
APPTAINER_BINARY=$(command -v apptainer || command -v singularity)
if [ -z "$APPTAINER_BINARY" ]; then
    echo "Error: Neither apptainer nor singularity found in PATH"
    exit 1
fi
echo "Using apptainer at: $APPTAINER_BINARY"

# This is the default apptainer that you can build from 'make apptainer'
echo "... looking for a local apptainer image at '$SCRIPT_DIR/modelhub.sif'"
SIF_PATH="$SCRIPT_DIR/modelhub.sif"
SIF_PATH=$(readlink -f "$SCRIPT_DIR/modelhub.sif" )
echo "Base SIF path to build from: $SIF_PATH"

# Generate the image name with today's date
DATE=$(date +%Y-%m-%d)
IMAGE_NAME="frozen_modelhub_${DATE}.sif"
echo "Building apptainer from image with frozen dependencies: $IMAGE_NAME"

# Check if INSTALL_PROJECT is set to true and set the image name accordingly
if ${INSTALL_PROJECT}; then
    echo "Modelhub WILL be installed in the apptainer! Ensure that this is intentional."
    IMAGE_NAME="frozen_modelhub_datahub_cifutils_${DATE}.sif"
else
    IMAGE_NAME="frozen_datahub_cifutils_${DATE}.sif"
fi

# Build Phase
echo
echo "=== Starting Build Phase ==="
echo "Running: $APPTAINER_BINARY build --notest '$IMAGE_NAME' freeze_apptainer.spec"
echo "----------------------------------------"
INSTALL_PROJECT=$INSTALL_PROJECT $APPTAINER_BINARY build \
    --nv \
    --notest \
    "$IMAGE_NAME" freeze_apptainer.spec
echo "----------------------------------------"

echo
echo "=== Build Complete ==="
echo "Container is available at: $PWD/$IMAGE_NAME" 