# Installation of RFdiffusion3 on Unix Systems

## Table of Contents
- [Learning Objective](#install_tutorial_learning_objective)
- [Prerequisites](#install_tutorial_prereqs)
- [Tutorial](#install_tutorial_tutorial)
    - [Step 1: Creating a conda environment](#install_tutorial_step_1)
    - [Step 2: Installing RFdiffusion3](#install_tutorial_step_2)
    - [Step 3: Verify the installation](#install_tutorial_step_3)
- [GPU vs. CPU Execution](#install_tutorial_gpu_v_cpu)
    - [How RFdiffusion3 detects and uses your GPU](#install_tutorial_detecting_gpu)
    - [Running on latest GPU architectures (e.g. Blackwell)](#install_tutorial_blackwell)
- [Glossary](#install_tutorial_glossary)
- [Resources & References](#install_tutorial_resources_references)

(install_tutorial_learning_objective)= 
## Learning Objective
By the end of this tutorial, you will be able to install RFdiffusion3 (RFD3) on a Unix-based system using `pip` and verify that the installation was successful by running a minimal test example. After completing this tutorial, you will have a working RFdiffusion3 environment capable of running basic design tasks and ready for use in downstream protein design workflows.

(install_tutorial_prereqs)=
## Prerequisites  
RFdiffusion3 is supported on Unix-based systems (Linux or MacOS). Windows is not officially supported unless used through a Linux subsystem (e.g., [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install)). While not strictly required, using an environment manager such as [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), [pixi](https://pixi.prefix.dev/latest/), [mamba](https://github.com/mamba-org/mamba), or [uv](https://docs.astral.sh/uv/) can be helpful to manage the dependencies that RFD3 relies on. For practical protein design workloads, a machine equipped with an NVIDIA GPU with ~3 GB of memory is highly recommended, as GPU acceleration substantially reduces inference time. This requires a recent NVIDIA driver installation. RFdiffusion3 can also run on CPU-only systems, however, runtime may increase significantly depending on the design task.

List of requirements:
- Unix-based system
- Python 3.9-3.12
- Package manager (optional, but highly suggested)
- A working internet connection 
- Sufficient disk space for the [model checkpoints](#install_tutorial_checkpoint) (~3 GB)
- A downloading tool such as `wget` or `curl`
- Optional but recommended: An NVIDIA GPU with a recent driver installation and at least ~16GB of RAM

(install_tutorial_tutorial)=
## Tutorial
<details>
<summary><strong>Mini Tutorial: Requesting an Interactive GPU Node on SLURM systems</strong></summary>

If you are installing any of the models in Foundry on a high performance computing (HPC) cluster, you should install them from a compute node – specifically a GPU node is possible – not the login node. For those new to using these types of resources, we will go through how to request an interactive GPU node to use for installing RFD3 here. You can also run RFD3 by requesting an interactive GPU node. 

```{important}
This "Mini Tutorial" shows how to request an interactive GPU node for systems using [SLURM](https://slurm.schedmd.com/overview.html). This is a common workload manager that helps allocate resources and schedule jobs that various users want to run on the cluster. If your HPC system uses a different workload manager or you are using cloud computing, you will need to go to your cluster's documentation to learn how to do this. 
```

Once you are logged into your HPC system, the command to request an interactive node will have the following structure: 
```bash
salloc --gres=gpu:1 --account=<account-name> --partition=<partition-name>
```
This requests:
- 1 GPU
- access under the specified account
- resources from the specified partition

The partition and account names are going to be specific to your cluster and lab, respectively. Look through the documentation for your HPC resources or ask your lab members if you do not know what to put for these.

```{note}
Typically your interactive session will not start right away, you will need to wait until resources become available. How long you need to wait depends on your system. Once it is ready you should see the command prompt reappear in your terminal window, perhaps slightly changed to reflect that you are now on a specific node. This means the allocation was "granted."
```

You can directly run the commands to install and run tools like RFD3 from this command prompt, however it's good practice to open an interactive shell. This will be configured better for running commands on the interactive node: 
```bash
srun --pty $SHELL
```
Doing this likely won't change anything that you can see, but if you run the command `hostname` before and after you run this command, you will see different results. 

```{note}
Some systems will automatically put you on an interactive shell on the allocated compute node after running salloc. In this case running the command to create an interactive shell will do nothing – running `hostname` before and after will result in the same information printed to screen.
```

You can often add other information to your resource allocation request. For example: 
```bash
salloc --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=01:00:00 --account=<account-name> --partition=<partition-name>
```
This command includes:
- Information about the number of CPU nodes you want (CPUs are needed to launch programs, feed data to GPUs, etc. GPU nodes are almost never run in complete isolation from CPU nodes.)
- The amount of memory for your GPU node
- The time limit for your interactive node

Your cluster will have limits on the number of CPUs, memory, and time that you can request. Refer to the documentation specific for your cluster if you need these pieces of information. 

Now that you have your interactive GPU session running, continue to installing RFD3 on your system. 
<hr>
</details>

<!-- 
If you want to run RFdiffusion3 on a GPU and are working on a shared scientific cluster, a GPU is usually not available by default. In that case, you must first request one through the cluster's job scheduler (e.g., [SLURM](#SLURM)). If you are installing RFdiffusion3 on your local machine, or if you plan to run it only on CPU, you can [skip](#step-1-creating-a-conda-environment) this step.

On many [SLURM](#SLURM) clusters, the exact command for allocating a GPU depends  on the local configuration. A general pattern is:
```
salloc --gres=gpu:1 --account=<account-name> --partition=<partition-name>
```
This requests:
- 1 GPU
- access under the specified account
- resources from the specified partition

After the allocation is granted, start an interactive shell with:
```
srun --pty $SHELL
```

If you cluster requires explicit CPU cores, system memory and a time limit, a more complete request may look like:
```
salloc --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=01:00:00 --account=<account-name> --partition=<partition-name>
```

Once the session has started, verify that a GPU was allocated by running:
```
nvidia-smi
```

If the allocation was successful, this command will display information such as the GPU model, the NVIDIA driver version and the current GPU memory usage. If you see <code>command not found</code>, or if no GPU is listed, then no GPU is available in your current session.
-->

(install_tutorial_step_1)=
### Step 1: Creating a conda environment
Next, create an isolated [environment](#environment_def). Here we will specifically create a conda environment, which requires [Anaconda](https://www.anaconda.com/download) or its lightweight variants, [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) and Miniforge[https://conda-forge.org/download/]. These are often already installed on HPC clusters. 

```{note}
The rest of the tutorial assumes that you are working in a conda environment, but the commands can work with other package managers (uv, pixi, etc.) with only minor changes required. Using conda is not required for this installation or for running RFD3. 
```

Environments allow you to isolate the all the packages you need to install and any associated libraries from your system-wide installation. This prevents dependency conflicts when different tools need different dependency versions (e.g. your backbone design tool needs NumPy v.2.4.0 but your sequence generation tool needs v.2.3.5) and ensures that your global environment remains unaffected. If any issues occur during installation, the environment can simply be removed without impacting the rest of your system.

In the terminal input the following command:  
```bash
conda create -n RFD3_env python=3.12 -y
```

Here:
- `-n RFD3_env` specifies the name of the environment.
- `python=3.12` defines the Python version (a version between 3.9 and 3.12 is recommended).
- `-y` automatically confirms installation prompts

Once the creation is completed, activate the environment with:  
```bash
conda activate RFD3_env
```

You should now see `(RFD3_env)` prefixed in your terminal prompt, indicating that the environment is active. 
To verify that the correct Python interpreter is being used, run:  
```bash
which python
```
The displayed path should point to the newly created [environment](#environment_def), for example:  
```bash
/home/username/miniconda3/envs/RFD3_env/bin/python`
```

(install_tutorial_step_2)=
### Step 2: Installing RFdiffusion3
RFdiffusion3 is distributed as part of the [Foundry](https://github.com/RosettaCommons/foundry/tree/production) ([`rc-foundry`](https://pypi.org/project/rc-foundry/) on PyPI) Python package. Foundry is the RosettaCommons framework that provides a unified [command-line interface](#cli_def) for running multiple protein modeling and design deep learning models. In machine learning disciplines, this type of resource is often referred to as a ['model zoo'](#model_zoo_def). As of the last update of this tutorial, it includes [RosettaFold3 (RF3)](https://github.com/RosettaCommons/foundry/tree/production/models/rf3) for structure prediction, [MPNN](https://github.com/RosettaCommons/foundry/tree/production/models/mpnn) for inverse folding, and [RFdiffusion3](https://github.com/RosettaCommons/foundry/tree/production/models/rfd3) for generative protein design. While this tutorial focuses on RFdiffusion3, RosettaFold3 and MPNN can be installed in a similar manner.  

Install RFdiffusion3 using:  
```bash
pip install "rc-foundry[rfd3]"
```
```{important}
The quotation marks around `rc-foundry[rfd3]` are important in shells such as `zsh`, where square brackets have special meaning. Without quotes, the command may fail.
```

(install_tutorial_checkpoint)=
#### Dowloading the model checkpoint  
RFdiffusion3 requires a trained model file (a [checkpoint](#checkpoint_def)) file containing the learned neural network weights (~3 GB).  
Download the checkpoint file using:  
```bash
foundry install rfd3
```
By default, this command will download the [checkpoint](#checkpoint_def) file to `~/.foundry/checkpoints`.  


If you prefer to store the [checkpoint](#checkpoint_def) files in a custom location (for example, on a cluster with limited home directory space), you can specify a custom checkpoint directory using the `--checkpoint-dir`flag:  
```bash
foundry install rfd3 --checkpoint-dir <path/to/checkpoint_dir>`
```

This will download the [checkpoint](#checkpoint_def) file to the specified directory and register that directory via the `FOUNDRY_CHECKPOINT_DIRS` [environment variable](#env_var_def), so that RFdiffusion3 automatically searches for it there in future runs (in addition to the default `~/.foundry/checkpoints` location).

(install_tutorial_step_3)=
### Step 3: Verify the installation
To verify that RFdiffusion3 was installed correctly, we will download and run a minimal demonstration example from the official RFdiffusion3 repository. First, create a working directory and a directory for the example input files, then download the required files:
```
mkdir -p input_pdbs  
mkdir -p verify_installation
wget -P verify_installation https://raw.githubusercontent.com/RosettaCommons/foundry/refs/heads/production/models/rfd3/docs/examples/demo.json
wget -P input_pdbs https://raw.githubusercontent.com/RosettaCommons/foundry/production/models/rfd3/docs/input_pdbs/M0255_1mg5.pdb  
wget -P input_pdbs https://raw.githubusercontent.com/RosettaCommons/foundry/production/models/rfd3/docs/input_pdbs/7v11.pdb  
wget -P input_pdbs https://raw.githubusercontent.com/RosettaCommons/foundry/production/models/rfd3/docs/input_pdbs/1bna.pdb
```

```{note}
If you have trouble accessing the files via `wget`, you can also place them on your cluster via [`scp`](https://snapshooter.com/learn/linux/copy-files-scp#scp-from-local-to-remote). A zip file has been created with just these JSON and PDB files and can be found [here](https://github.com/RosettaCommons/foundry/tree/production/models/rfd3/docs/tutorials/installation_tutorial).
```

After downloading the files, enter the working directory and run the demo using:  
```bash
cd verify_installation
rfd3 design out_dir=demo_output inputs=demo.json
```
The output directory (`demo_output`) will be created automatically if it does not exist already. On a modern GPU, this example typically completes within a few minutes. On a CPU, runtime may increase substantially depending on the hardware. Some warning messages during execution are normal and can be ignored.

Expected output:  
Inside `demo_output/`, you should find structure files (.cif.gz) for each demo example and summary score files (.json). If these were generated without errors, the installation was successful. You can expect the generated structures using a molecular visualization tool such as [PyMOL](https://www.pymol.org/), or examine the score files in a text editor.

(install_tutorial_gpu_v_cpu)=
## GPU vs CPU Execution
RFdiffusion3 can run on both CPUs and GPUs, but the performance difference is very large:
- CPU only: Suitable for small tests and learning. Inference will run, but it can be very slow — often tens of minutes per example or longer depending on your CPU.
- GPU: Strongly recommended for real use, especially with multiple targets. A modern NVIDIA GPU can reduce runtime from minutes to seconds per design.

(install_tutorial_detecting_gpu)=
### How RFdiffusion3 detects and uses your GPU
To run RFdiffusion3 on a GPU no additional setup steps are required. It will automatically try to use a GPU if:
1. A compatible NVIDIA GPU is present
2. A matching CUDA toolkit is installed
3. [PyTorch](https://pytorch.org/) (an open-source machine learning library) can communicate with the GPU

You can verify your GPU availability inside your [environment](#environment_def) using:
```python
python - << 'EOF'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count(), "GPUs detected")
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
EOF
```
Make sure your environment is activated before running this check.
If `CUDA available: True` is printed and a GPU name appears, PyTorch can access your GPU.

(install_tutorial_blackwell)=
### Running on latest GPU architectures (e.g. Blackwell)
Very recent NVIDIA GPU architectures (such as Lovelace or Blackwell) may require:
- A sufficiently new NVIDIA driver
- A compatible CUDA runtime
- A recent PyTorch build that supports the architecture

In most cases, installing RFD3 as described above is sufficient. However, for very new GPUs, you may need to upgrade PyTorch to a more recent (or nightly) build that includes support for the latest CUDA versions.

Installation example:
```bash
# 1. Install rfd3 and its required dependencies
pip install "rc-foundry[rfd3]"

# 2. Upgrade to PyTorch nightly (cu128) to add Blackwell GPU support
pip install --pre torch torchvision torchaudio \
--index-url https://download.pytorch.org/whl/nightly/cu128 \
--upgrade --force-reinstall
```

Important notes:
- Replace cu128 with the CUDA version supported by your driver (check via `nvidia-smi`, if your system uses NVIDIA GPUs).
- Installing a nightly PyTorch build overrides the version originally installed with `rc-foundry`.
- After upgrading PyTorch, verify GPU detection again using the [Python check](#install_tutorial_detecting_gpu) from above.

(install_tutorial_glossary)=
## Glossary

(checkpoint_def)=
### Checkpoint
A binary file that contains the trained model weights of a neural network model. For RFdiffusion3, this file stores the learned parameters the model needs for inference. Without a checkpoint, the model cannot run.

(cli_def)=
### CLI (Command-Line Interface)
A way to interact with software by typing commands in a terminal. Foundry and RFdiffusion3 provide CLI commands like `foundry install rfd3` and `rfd3`.

(environment_def)=
### Environment
An isolated Python environment created using a package manager. It keeps project dependencies separate from your system Python to prevent version conflicts.

(env_var_def)=
### Environment Variable
A value stored in the shell session that software can read to determine paths or settings (e.g., `FOUNDRY_CHECKPOINT_DIRS`).

(model_zoo_def)=
### Model Zoo
A curated repository of model architectures that typically also includes pre-trained model weights that are available for download. [Foundry](https://github.com/RosettaCommons/foundry/tree/production) is an example of a model zoo. 

(install_tutorial_resources_references)=
## Resources & References
- RosettaCommons Foundry (GitHub Repository) https://github.com/RosettaCommons/foundry
- RFdiffusion Documentation https://github.com/RosettaCommons/foundry/tree/production/models/rfd3
- RFdiffusion3 publication https://www.science.org/doi/10.1126/science.ade2574
- PyMOL visualization software https://pymol.org/