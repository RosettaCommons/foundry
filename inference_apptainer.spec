Bootstrap: localimage
From: ./scripts/shebang/modelhub.sif
IncludeCmd: yes
# NOTE: This apptainer was written using apptainer version `1.1.6+2-g6808b5172-ipd`

%setup
   # Create all required directories at once
   echo "Creating directory structure in container..."
   mkdir -p ${APPTAINER_ROOTFS}/opt/modelhub/{src,configs,lib}
   
   echo "Copying project files into the container..."
   
   # Copy .project-root file (if exists)
   cp -f ./.project-root ${APPTAINER_ROOTFS}/opt/modelhub/ 2>/dev/null || echo "Note: .project-root not found, skipping"

   # Copy .env file (if exists)
   cp -f ./.env ${APPTAINER_ROOTFS}/opt/modelhub/ 2>/dev/null || echo "Note: .env not found, skipping"
   
   # Copy directories with rsync
   rsync -av --info=progress2 ./src/ ${APPTAINER_ROOTFS}/opt/modelhub/src/
   rsync -av --info=progress2 ./configs/ ${APPTAINER_ROOTFS}/opt/modelhub/configs/
   rsync -av --info=progress2 ./lib/ ${APPTAINER_ROOTFS}/opt/modelhub/lib/
   
   echo "All files copied successfully."

%environment
    # Add project directories to PYTHONPATH (modelhub, datahub, cifutils)
    export PYTHONPATH=/opt/modelhub/src:/opt/modelhub/lib/datahub/src:/opt/modelhub/lib/cifutils/src:${PYTHONPATH}

%runscript
   # Run the inference.py script by default with any passed arguments
   exec python /opt/modelhub/src/modelhub/inference.py "$@"

%labels
    Author "Nate Corley <ncorley.uw@edu.com>"
    Version 1.0.0
    Description "ModelHub inference container"

%help
    This apptainer exposes inference with the Institute for Protein Design's structure prediction model.
    
    Usage:
      $ ./container.sif [args]        # Run inference.py with optional arguments
      $ apptainer exec container.sif bash  # Get a shell in the container
