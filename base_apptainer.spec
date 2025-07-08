Bootstrap: docker
From: nvcr.io/nvidia/pytorch:25.04-py3
IncludeCmd: yes

# NOTE: This apptainer was written using apptainer version `1.1.6+2-g6808b5172-ipd`
# To build this apptainer, use:
#     make base_apptainer

%setup
   # Create a directory in the container to bind the host's current working directory
   mkdir ${APPTAINER_ROOTFS}/modelhub_host
   # ... for mounting `/projects` with --bind
   mkdir ${APPTAINER_ROOTFS}/projects
   # ... for mounting `/databases` with --bind
   mkdir ${APPTAINER_ROOTFS}/net
   # ... for mounting `/squash` with --bind
   mkdir ${APPTAINER_ROOTFS}/squash

%files
   /etc/localtime
   /etc/hosts
   requirements.txt /opt/requirements.txt

%post
   ## GENERAL SETUP

   # Common symlinks (within container)
   ln -s /net/databases /databases
   ln -s /net/software /software
   ln -s /home /mnt/home
   ln -s /projects /mnt/projects
   ln -s /net /mnt/net

   ## PACKAGE INSTALLATION

   apt-get update
   apt-get install -y make git libaio-dev
   # Install OpenBabel (pip installation fails due to C++ build dependencies)
   apt-get install -y openbabel libopenbabel-dev python3-openbabel
   apt-get clean

   ## PYTHON DEPENDENCY INSTALLATION
   
   # Fix NGC constraints that conflict with our required packages
   # ... remove packaging constraint to allow biotite 1.3.0 installation
   sed -i '/packaging==/d' /etc/pip/constraint.txt

   # ... remove pytest constraint
   sed -i '/pytest==/d' /etc/pip/constraint.txt
   
   # Install all other Python dependencies using requirements.txt
   # (Installs into the default NGC Python environment)
   pip install -r /opt/requirements.txt
   
   # Clean up
   apt-get clean && rm -rf /var/lib/apt/lists/*

%environment
   # (Flag to increase accessible GPU memory)
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

   # (Turn off NVLink)
   export NCCL_P2P_DISABLE=1
   
%runscript
   # NOTE: The %runscript is invoked when the container is run without specifying a different command. 
   exec python "$@"

%help
   modelhub environment for running modelhub independently and for development

   To see this help message, use:
      apptainer run-help modelhub_apptainer.sif

   To build this apptainer, use:
      apptainer build --bind $PWD:/modelhub_host path/to/apptainer.sif apptainer.spec

   To run the container, use:
      apptainer exec /path/to/apptainer.sif <command>
      OR
      ./path/to/apptainer.sif <command>

   To get an interactive shell in the container, use:
      apptainer shell /path/to/apptainer.sif

%labels
    Version v1.0.0
    ApptainerVersion 1.1.6+2-g6808b5172-ipd