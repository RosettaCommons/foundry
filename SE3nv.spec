Bootstrap: docker
From: ubuntu
IncludeCmd: yes

%setup
rsync -a --no-g --no-o /home/dimaio/RoseTTAFold2/SE3Transformer/ $APPTAINER_ROOTFS/SE3Transformer/

%files
/etc/localtime
/etc/hosts
/etc/apt/sources.list
./Miniconda3-latest-Linux-x86_64.sh /opt/miniconda.sh

%post
## Switch shell to bash
rm /bin/sh; ln -s /bin/bash /bin/sh

# Common symlinks
ln -s /net/databases /databases
ln -s /net/software /software
ln -s /home /mnt/home
ln -s /projects /mnt/projects
ln -s /net /mnt/net

apt-get update
# required X libs
apt-get install -y libx11-6 libxau6 libxext6 libxrender1
#  git
apt-get install -y git
apt-get install -y libaio-dev
apt-get clean

# Install conda
bash /opt/miniconda.sh -b -u -p /usr

# install everything
# lock python version to 3.10.8
#      cuda version to 12.1
#      torch version to 2.2
conda install \
   -c conda-forge \
   -c nvidia/label/cuda-12.1.0 \
   -c pytorch \
   -c pyg \
   -c dglteam/label/cu121 \
   -c https://conda.rosettacommons.org \
   python==3.10.8 \
   openbabel \
   pip \
   ipython \
   ipykernel \
   numpy \
   pandas \
   seaborn \
   matplotlib \
   jupyterlab \
   pyrosetta \
   pyg \
   dgl \
   pytorch==2.2 \
   pytorch-cuda==12.1 \
   cuda

# pip extras
pip install e3nn \
   omegaconf \
   hydra-core \
   pyrsistent \
   opt_einsum \
   sympy \
   omegaconf \
   icecream \
   wandb \
   deepdiff \
   assertpy \
   pytest \
   pytest-benchmark \
   einops

# deepspeed
pip install deepspeed
#pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# SE3 transformer
pip install /SE3Transformer/

# Clean up
conda clean -a -y
apt-get -y purge build-essential git wget
apt-get -y autoremove
apt-get clean
#rm /opt/miniconda.sh

%environment
export PATH=$PATH:/usr/local/cuda/bin

%runscript

exec python "$@"

%help
SE3nv environment for running RF-diffusion, etc.
