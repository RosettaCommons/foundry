Bootstrap: localimage
From: ../../scripts/shebang/modelhub.sif
IncludeCmd: yes
# NOTE: This apptainer was written using apptainer version `1.1.6+2-g6808b5172-ipd`

%files
    environment.yaml /opt/environment.yaml

%post
   # Install additional packages from environment.yaml
   # This will only install packages that aren't already in the environment
   conda env update --file /opt/environment.yaml

%environment
   source /usr/etc/profile.d/conda.sh
   conda activate modelhub-apptainer

   export PATH=$PATH:/usr/local/cuda/bin
   export CUTLASS_PATH=/opt/cutlass/

   # Paths for CIFUtils
   export CCD_MIRROR_PATH=/projects/ml/frozen_pdb_copies/2024_12_11_ccd
   export PDB_MIRROR_PATH=/projects/ml/frozen_pdb_copies/2024_12_01_pdb

   # Paths for training dataset
   export AF2FB_PATH=/squash/af2_distillation_facebook

%runscript
   # NOTE: The %runscript is invoked when the container is run without specifying a different command. 
   exec python "$@"
