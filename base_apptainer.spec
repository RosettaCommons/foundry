Bootstrap: docker
From: ubuntu:24.04
IncludeCmd: yes
# NOTE: This apptainer was written using apptainer version `1.1.6+2-g6808b5172-ipd`
# To build this apptainer, use:
#     make apptainer

%setup
   # Create a directory in the container to bind the host's current working directory
   mkdir ${APPTAINER_ROOTFS}/modelhub_host
   # ... for mounting `/projects` with --bind
   mkdir ${APPTAINER_ROOTFS}/projects
   # ... for mounting `/databases` with --bind
   mkdir ${APPTAINER_ROOTFS}/net
   # ... for mounting `/squash` with --bind
   mkdir ${APPTAINER_ROOTFS}/squash

%files
   /etc/localtime
   /etc/hosts
   environment.yaml /opt/environment.yaml

%post
   # get os name
   echo "Running on OS name $(lsb_release -i | awk '{ print $3 }')"
   # get os version
   echo "... in OS version $(lsb_release -r | awk '{ print $2 }')"

   ## GENERAL SETUP
   # Switch shell to bash
   ln -sf /bin/bash /bin/sh

   # Common symlinks (within container)
   ln -s /net/databases /databases
   ln -s /net/software /software
   ln -s /home /mnt/home
   ln -s /projects /mnt/projects
   ln -s /net /mnt/net

   ## PACKAGE INSTALLATION
   apt-get update
   # Install build essentials and other required packages (needed for compiling biotite cython files)
   apt-get install -y build-essential gcc g++
   # Install make (so we can run `make format`, `make clean`, etc.)
   apt-get install -y make git wget libaio-dev
   # required X libs
   apt-get install -y libx11-6 libxau6 libxext6 libxrender1
   apt-get clean

   # Clone CUTLASS (for DeepSpeed)
   git clone https://github.com/NVIDIA/cutlass.git /opt/cutlass

   # Clone DeepSpeed (so we can pre-install the wheel)
   git clone --branch v0.16.2 https://github.com/deepspeedai/DeepSpeed.git /opt/deepspeed

   ## ENVIRONMENT CREATION & DEPENDENCY INSTALLATION
   # Download miniconda
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda.sh
   
   # Install conda
   bash /opt/miniconda.sh -b -u -p /usr

   # install everything
   # ... the environment in the container is at `/usr/envs/modelhub-apptainer`
   conda install -c conda-forge mamba
   mamba env create --file /opt/environment.yaml --name modelhub-apptainer

   # Set up conda
   conda init bash
   # Add conda environment to PATH
   export PATH=/usr/envs/modelhub-apptainer/bin:$PATH

   echo "Proceeding with DeepSpeed reinstallation."

   ## PRE-COMPILE DEEPSPEED FROM WHEEL 
   # (Overwrite deepspeed installation from the `environment.yaml`)
   pip uninstall deepspeed -y # Avoid interactive prompts

   # (Flags for building the Evoformer attention)
   export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
   export DS_BUILD_EVOFORMER_ATTN=1
   export CUTLASS_PATH=/opt/cutlass/

   # Reinstall DeepSpeed, pre-compiling the evoformer attentino kernel
   pip wheel /opt/deepspeed -w /opt/deepspeed
   pip install /opt/deepspeed/deepspeed-0.16.2+b344c04d-cp311-cp311-linux_x86_64.whl

   # Run the biotite setup command
   # (Temporary measure until we switch to released Biotite version)
   . /usr/etc/profile.d/conda.sh
   conda activate modelhub-apptainer
   python -m biotite.setup_ccd

   # clean up files to reduce size
   # ... remove conda
   mamba clean -a -y
   # ... remove other apt packages that are no longer needed
   apt-get -y purge build-essential wget
   apt-get -y autoremove
   apt-get clean
   rm /opt/miniconda.sh

%environment
   source /usr/etc/profile.d/conda.sh
   conda activate modelhub-apptainer

   export PATH=$PATH:/usr/local/cuda/bin
   export CUTLASS_PATH=/opt/cutlass/

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
