## Agent instructions: build environments (any platform)

This repo supports two related setups:

- **Global default bio env** (recommended): `~/.venvs/molecore_foundry`
- **Repo env** (recommended when working on Foundry): `./.venv` in the repo root (direnv-enabled via `.envrc`)

### Prereqs (all platforms)

- Install `uv`
- Python: **3.12** (Foundry requires `>=3.12`)

---

## A) Global default env (portable across macOS/Linux)

Create and activate:

```bash
uv python install 3.12
uv venv ~/.venvs/molecore_foundry --python 3.12
source ~/.venvs/molecore_foundry/bin/activate
```

Install the default package set (from this repo’s env spec):

```bash
uv pip install -e /path/to/foundry/tools/envs/molecore_foundry
```

### Linux GPU: CUDA PyTorch (opt-in)

On GPU machines, override CPU torch with a CUDA wheel:

```bash
# cu124 (recommended)
uv pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# or cu121
uv pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### PyRosetta (optional)

Install installer + run it:

```bash
uv pip install pyrosetta-installer
python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta(silent=True)"
python -c "import pyrosetta; pyrosetta.init('-mute all'); print('PyRosetta OK')"
```

### PyMOL (optional)

- The env installs `pymolpy3` (wrapper). Full PyMOL is usually installed separately (system/Homebrew/conda).

Verify wrapper import:

```bash
python -c "import pymolPy3; print('pymolPy3 OK')"
```

---

## B) Repo env (Foundry dev in this repository)

From the repo root:

```bash
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[all,dev]"
```

If you use `direnv`, allow the repo once:

```bash
direnv allow
```

---

## Mirrors and Borg mount (optional but recommended)

AtomWorks can use mirrors for PDB/CCD. This repo’s `.envrc` prefers a Borg-mounted databases directory:

- Remote: `borg:/runtime/databases/foundry/`
- Local mountpoint: `~/mounts/foundry_databases/`
- Mirrors:
  - `~/mounts/foundry_databases/ccd`
  - `~/mounts/foundry_databases/pdb` (large; optional)

macOS note: `sshfs` requires macFUSE to be installed and permitted by macOS security policy.


