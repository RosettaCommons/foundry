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

%runscript
   # NOTE: The %runscript is invoked when the container is run without specifying a different command. 
   exec python "$@"
