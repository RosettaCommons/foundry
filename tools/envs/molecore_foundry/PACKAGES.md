## Package candidates for `molecore_foundry`

This document is a **curated** list of package candidates, grouped by purpose, based on:

- **Your existing env**: `/Users/ariel/conda/envs/zipbio11_dev/bin/python -m pip list --format=freeze`
- **Foundry project requirements**: `/Users/ariel/dev/molCore/foundry/pyproject.toml`
- **Common bio “daily driver” tooling**

---

## Baseline (recommended everywhere)

- **Env plumbing**: `pip`, `setuptools`, `wheel`
- **Config/IO**: `requests`, `python-dotenv`, `pyyaml`
- **Data/notebooks**: `pandas`, `tqdm`, `ipykernel`, `jupyterlab`
- **Core bio runtime**: `atomworks`, `biotite`, `torch`

---

## Bio / structure utilities (portable)

- **Numerics**: `numpy`, `scipy`
- **Bio sequences**: `biopython`
- **Structure IO** (PDB/mmCIF): `biotite`
- **AtomWorks**: `atomworks` (IO) or `atomworks[ml]` (ML extras) — see [`RosettaCommons/atomworks`](https://github.com/RosettaCommons/atomworks)

---

## Biological database access (direct programmatic clients)

- **RCSB PDB**: `rcsbsearchapi`
- **UniProt/Ensembl/etc**: `bioservices`
- **General HTTP**: `httpx`
- **Convenience fetchers**: `gget`

Notes:
- AlphaFold DB access is typically done via HTTP; `requests/httpx` + small helper functions are usually enough.

---

## Visualization helpers (pip-friendly)

- **Plotting**: `matplotlib`, `seaborn`
- **3D in notebooks**: `py3Dmol`, `nglview`

---

## Agentic workflows + Modal (optional)

- **Agentic basics**: `pydantic`, `httpx`, `tenacity`, `rich`, `typer`, `orjson`
- **Provider SDKs**: `openai`, `anthropic`
- **Modal**: `modal` (see [`modal` on PyPI](https://pypi.org/pypi/modal))

---

## Foundry stack (ML-heavy; optional)

From Foundry’s `pyproject.toml` core deps include (not exhaustive):
- `hydra-core`, `environs`, `wandb`, `rich`, `typer`
- `torch`, `lightning`, `einops`, `opt_einsum`, `dm-tree`
- `atomworks[ml]`

RF3 adds Linux-only CUDA-oriented deps guarded by markers in Foundry:
- `cuequivariance_ops_cu12`, `cuequivariance_ops_torch_cu12`, `cuequivariance_torch` (linux only)

---

## Your previous `zipbio11_dev` env (captured)

These were present and are kept as an optional group (`zipbio11_dev`) in the env spec:

- `beautifulsoup4==4.12.3`
- `lxml==6.0.2`
- `nltk==3.9.2`
- `PyGithub==2.8.1`
- `python-dateutil==2.9.0`
- `selenium==4.39.0`
- `sendgrid==6.12.5`
- `spacy==3.8.11`
- `SQLAlchemy==2.0.45`
- `tenacity==9.0.0`
- `webdriver-manager==4.0.2`

---

## PyRosetta + PyMOL notes

- **PyRosetta** ❌: Not available for Python 3.12 (Rosetta Commons limitation). The `pyrosetta-installer` is included for when compatibility is added.
- **PyMOL** ✅: `pymolpy3` wrapper included. Install system PyMOL separately for full functionality.

