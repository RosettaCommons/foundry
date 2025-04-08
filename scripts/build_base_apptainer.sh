#!/bin/bash
# This script builds a datahub apptainer container.
set -e  # Exit on error

echo "Running from $PWD"

# Check if apptainer/singularity is available
APPTAINER_BINARY=$(command -v apptainer || command -v singularity)
if [ -z "$APPTAINER_BINARY" ]; then
    echo "Error: Neither apptainer nor singularity found in PATH"
    exit 1
fi
echo "Using apptainer at: $APPTAINER_BINARY"

# Generate the image name with today's date
DATE=$(date +%Y-%m-%d)
IMAGE_NAME="modelhub_${DATE}.sif"
echo "Building apptainer image: $IMAGE_NAME"

# Build Phase
echo
echo "=== Starting Build Phase ==="
echo "Running: $APPTAINER_BINARY build --notest '$IMAGE_NAME' base_apptainer.spec"
echo "----------------------------------------"
$APPTAINER_BINARY build \
    --nv \
    --notest \
    "$IMAGE_NAME" base_apptainer.spec
echo "----------------------------------------"

echo
echo "=== Build Complete ==="
echo "Container is available at: $PWD/$IMAGE_NAME" 