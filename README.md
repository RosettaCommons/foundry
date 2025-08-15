# Modelforge

## Installation & Usage

Follow these steps to set up **ModelForge** and run a test prediction.

---

### 1. Install the repository using `uv`

```bash
git clone https://github.com/RosettaCommons/modelforge.git \
  && cd modelforge \
  && uv python install 3.12 \
  && uv venv --python 3.12 \
  && source .venv/bin/activate \
  && uv pip install -e .
```

### 2. Download model weights
```bash
wget http://files.ipd.uw.edu/pub/rf3/rf3_latest.pt
```

### 3. run a test prediction
```bash
rf3 fold tests/data/5vht_from_json.json
```

Details on the exact formatting of the json files are available here: 