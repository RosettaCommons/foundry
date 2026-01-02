## Agent operating rules (read first)

### Repo scope
- This repository is **MolCore/foundry**. Environment/dev instructions live here:
  - `tools/envs/molecore_foundry/AGENT_SETUP.md`

### Hard rule
- **Never suggest creating or changing pull requests to the RosettaCommons repository that the user forked from.**

### Environment expectations
- Global default env is a **uv virtualenv** at `~/.venvs/molecore_foundry`.
- Repo-local env (when working in this repo) is `./.venv` and is activated via `direnv` using `.envrc`.


