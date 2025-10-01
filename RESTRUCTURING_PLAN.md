# ModelForge Repository Restructuring Plan

**Date**: October 1, 2025
**Author**: Analysis based on PyTorch Lightning patterns
**Goal**: Reorganize `src/modelhub/` following best practices while keeping `releases/` structure intact

---

## Executive Summary

This plan addresses three critical issues in the current structure:
1. **Broken imports**: RF3 uses `from metrics.base import Metric` but base classes are in `src/modelhub/`
2. **CLI entry point mismatch**: `pyproject.toml` points to non-existent `modelhub.cli:app`
3. **Package installation confusion**: Only `src/modelhub` is packaged, but RF3 code is in `releases/`
4. **Unclear organization**: Mixed base classes and implementations without clear semantic separation

**Core Principle**: Follow PyTorch Lightning's pattern - base classes live WITH their implementations, no separate `contrib/` directory.

---

## Current Structure Analysis

### What Exists Now
```
modelhub_latent/
├── pyproject.toml                      # name = "rf3" (wrong!)
├── src/
│   └── modelhub/                       # Only this is packaged
│       ├── callbacks/
│       │   ├── base.py                # BaseCallback
│       │   ├── health_logging.py      # Implementation
│       │   ├── timing_logging.py      # Implementation
│       │   └── train_logging.py       # Implementation
│       ├── metrics/
│       │   ├── base.py                # Metric + MetricManager
│       │   ├── chiral.py              # Implementation
│       │   └── rasa.py                # Implementation
│       ├── utils/
│       ├── trainers/
│       └── hydra/
├── releases/
│   └── rf3/                           # NOT packaged!
│       ├── src/rf3/
│       │   ├── callbacks/
│       │   ├── metrics/
│       │   ├── model/
│       │   └── cli.py
│       ├── configs/
│       └── tests/
└── tests/                             # Shared tests
```

### Critical Problems

#### Problem 1: Import Path Mismatch
**RF3 code uses**:
```python
from metrics.base import Metric          # Expects local resolution
from callbacks.base import BaseCallback  # Expects local resolution
```

**Reality**:
- Base classes are in `src/modelhub/metrics/base.py`
- This only works via sys.path manipulation or broken imports

**Should be**:
```python
from modelhub.metrics.metric import Metric
from modelhub.callbacks.callback import BaseCallback
```

#### Problem 2: CLI Entry Point
**Current `pyproject.toml`**:
```toml
[project.scripts]
rf3 = "modelhub.cli:app"
```

**Reality**: There is NO `src/modelhub/cli.py` file!

**Actual CLI**: Located at `releases/rf3/src/rf3/cli.py`

#### Problem 3: Package Name Confusion
**Current**: `name = "rf3"` in pyproject.toml

**Problem**:
- Repository is called "modelforge"
- Users would `pip install rf3` (gets only base framework)
- Can't do `pip install modelforge[rf3]` for selective installation

#### Problem 4: Packaging Scope
**Current**: `packages = ["src/modelhub"]`

**Problem**: RF3 code in `releases/rf3/src/rf3/` is NOT included in the package!

---

## Design Goals

### User Experience Goals
1. `pip install modelforge` → Get base framework only
2. `pip install modelforge[rf3]` → Get base + RF3
3. `pip install modelforge[all]` → Get all models (future-proof)
4. Development: `pip install -e .[rf3,dev]`

### Code Organization Goals
1. Follow PyTorch Lightning patterns (familiar to ML researchers)
2. Clear distinction between base classes and implementations
3. Proper namespacing for imports
4. Maintain `releases/` structure for development workflow

### Import Pattern Goals
```python
# Clear, unambiguous imports
from modelforge.callbacks.callback import BaseCallback
from modelforge.metrics.metric import Metric
from modelforge.utils.ddp import RankedLogger

# RF3-specific imports
from modelforge.models.rf3.model import RF3
```

---

## PyTorch Lightning Pattern Analysis

### Lightning's Structure (Reference)
```
lightning/pytorch/
├── core/                          # Primary model abstractions
│   ├── module.py                 # LightningModule (what users inherit from)
│   ├── datamodule.py             # LightningDataModule
│   └── hooks.py
├── callbacks/                     # Base + implementations together
│   ├── callback.py               # Callback base class
│   ├── model_checkpoint.py       # ModelCheckpoint implementation
│   ├── early_stopping.py         # EarlyStopping implementation
│   └── __init__.py               # Exports everything
├── loggers/                       # Base + implementations together
│   ├── logger.py                 # Logger base class
│   ├── wandb.py                  # WandbLogger implementation
│   └── __init__.py
└── utilities/                     # Pure utilities
```

### Key Insights from Lightning

1. **`core/` is for PRIMARY model abstractions** - Things users inherit from to define their model:
   - `LightningModule` - "Your model IS A LightningModule"
   - `LightningDataModule` - "Your data IS A LightningDataModule"

2. **Other directories contain base + implementations together**:
   - `callbacks/callback.py` (base) + `callbacks/model_checkpoint.py` (impl)
   - `loggers/logger.py` (base) + `loggers/wandb.py` (impl)

3. **NO `contrib/` directory** - That's a Django pattern, not Lightning!

4. **`__init__.py` exports everything** for clean imports

### When to Use `core/`

**Use `core/` IF**:
- You have a primary model class users inherit from (like `LightningModule`)
- These classes define "what you ARE building" (semantic role)

**Don't use `core/` IF**:
- All your base classes are for plugins/extensions (callbacks, metrics)
- You don't have a central model abstraction

**For ModelForge**: Since there's no primary "ModelBase" class, **we don't need `core/`**.

### Re-evaluating `inference_engines/`

**Current situation**:
- `InferenceEngine` is a minimal ABC with only 2 abstract methods (`__init__`, `eval`)
- Only RF3 uses it (no other models yet)
- It's a very thin abstraction that provides minimal value

**Question**: Do we need this abstraction at all?

**Arguments for keeping it**:
- Future-proofs for when other models are added
- Provides a consistent interface pattern

**Arguments for removing it**:
- YAGNI (You Aren't Gonna Need It) - only one implementation exists
- Adds indirection without clear benefit
- The abstraction is so minimal it doesn't enforce meaningful constraints
- Each model's inference is likely to be sufficiently different that a shared interface adds little value

**Recommendation**: **Remove the `inference_engines/` directory entirely from `src/modelhub/`**
- RF3's inference engine can live solely in `models/rf3/src/rf3/inference_engines/rf3.py`
- Remove the ABC inheritance - just make it a standalone class
- When/if a second model is added, we can evaluate whether a shared abstraction makes sense based on actual commonalities
- This follows YAGNI and keeps the code simpler

---

## Recommended Solution

### Two-Part Restructuring

#### Part 1: Fix `src/modelhub/` Structure (Simple, Lightning-style)

```
src/
└── modelhub/  (rename to modelforge?)
    ├── __init__.py                    # Export key base classes
    │
    ├── callbacks/
    │   ├── __init__.py               # Export Callback + all implementations
    │   ├── callback.py               # BaseCallback (RENAMED from base.py)
    │   ├── health_logging.py         # HealthLoggingCallback
    │   ├── timing_logging.py         # TimingLoggingCallback
    │   └── train_logging.py          # TrainLoggingCallback
    │
    ├── metrics/
    │   ├── __init__.py               # Export Metric + all implementations
    │   ├── metric.py                 # Metric, MetricManager (RENAMED from base.py)
    │   ├── chiral.py                 # ChiralMetric
    │   └── rasa.py                   # RASAMetric
    │
    ├── trainers/
    │   ├── __init__.py
    │   └── fabric.py                 # Fabric trainer wrapper
    │
    ├── utils/                         # Pure utilities (no base classes)
    │   ├── ddp.py
    │   ├── logging.py
    │   ├── weights.py
    │   ├── instantiators.py
    │   └── torch.py
    │
    └── hydra/                         # Hydra utilities
        ├── __init__.py
        └── resolvers.py
```

**Key changes**:
1. Rename `base.py` files to match their content:
   - `callbacks/base.py` → `callbacks/callback.py`
   - `metrics/base.py` → `metrics/metric.py`
2. **REMOVE** `inference_engines/` entirely (not needed with only one model)
3. Keep implementations WITH base classes (Lightning pattern)
4. No `core/` directory (not needed without primary model abstraction)
5. Proper `__init__.py` exports

#### Part 2: Integrate RF3 into Main Package (For pip install)

**Decision Point**: Choose ONE of these approaches:

##### Option A: Move RF3 into src/modelhub/ (Monorepo style)
```
src/
└── modelhub/
    ├── callbacks/, metrics/, utils/  (as above)
    └── models/                        # NEW
        └── rf3/                       # Moved from releases/rf3/src/rf3/
            ├── __init__.py
            ├── model/
            ├── data/
            ├── metrics/               # RF3-specific metrics
            ├── callbacks/             # RF3-specific callbacks
            ├── inference_engines/
            └── cli.py
```

**Pros**:
- Simple pip install: `pip install modelforge[rf3]`
- Single package, single version
- Easy code sharing between models

**Cons**:
- Changes development workflow (no separate `releases/` for development)
- RF3 code always in source tree even if not installed

##### Option B: Keep releases/ separate (Development friendly)
```
modelhub_latent/
├── src/modelhub/           # Base framework only
├── releases/
│   └── rf3/                # Stays as is for development
│       └── src/rf3/
└── pyproject.toml          # Use optional dependencies
```

**Then in `pyproject.toml`**:
```toml
[project.optional-dependencies]
rf3 = [
    "modelforge-rf3 @ file:///${PROJECT_ROOT}/releases/rf3",
    # RF3-specific deps
]
```

**Pros**:
- Keeps development workflow unchanged
- Clear separation for releases

**Cons**:
- More complex build setup
- Requires editable installs for development
- Each release needs own `pyproject.toml`

##### Option C: Hybrid - Move RF3 but keep releases/ for development (RECOMMENDED)
```
# For development: work in releases/rf3/
releases/rf3/
├── src/rf3/           # Development happens here
├── configs/           # RF3 configs here
└── tests/             # RF3 tests here

# For installation: symlink or copy during build
src/modelhub/models/rf3/  → symlink to releases/rf3/src/rf3/

# Or use build hooks to include releases/ in package
```

**Implementation**: Use hatch build hooks to include `releases/rf3/src/rf3/` as `src/modelhub/models/rf3/`

---

## Detailed Implementation Plan

### Phase 1: Rename Package (Breaking Change Decision)

**Decision needed**: Keep `modelhub` or rename to `modelforge`?

**Arguments for `modelforge`**:
- Matches repository name
- More descriptive (framework for building models)
- Enables `pip install modelforge[rf3]`

**Arguments for `modelhub`**:
- Less renaming work
- Existing code already uses it

**Recommendation**: Rename to `modelforge` for clarity and marketing.

### Phase 2: Restructure src/modelhub/ (src/modelforge/)

#### Step 2.1: Rename Base Class Files
```bash
# In src/modelhub/
git mv callbacks/base.py callbacks/callback.py
git mv metrics/base.py metrics/metric.py
# inference_engines/base.py could stay or rename to inference_engine.py
```

#### Step 2.2: Update Imports in Base Files
**In `src/modelhub/callbacks/callback.py`**:
```python
# Update any internal imports if needed
# Mainly just ensure docstrings reference correct module names
```

**In `src/modelhub/metrics/metric.py`**:
```python
# Update imports and docstrings
```

#### Step 2.3: Create Proper __init__.py Files

**`src/modelhub/callbacks/__init__.py`**:
```python
"""Callbacks for training customization.

This module provides both the base callback class and common callback implementations.
"""

from modelhub.callbacks.callback import BaseCallback
from modelhub.callbacks.health_logging import HealthLoggingCallback
from modelhub.callbacks.timing_logging import TimingLoggingCallback
from modelhub.callbacks.train_logging import TrainLoggingCallback

__all__ = [
    "BaseCallback",
    "HealthLoggingCallback",
    "TimingLoggingCallback",
    "TrainLoggingCallback",
]
```

**`src/modelhub/metrics/__init__.py`**:
```python
"""Metrics for model evaluation.

This module provides the base metric framework and common metric implementations.
"""

from modelhub.metrics.metric import Metric, MetricManager, instantiate_metric_manager
from modelhub.metrics.chiral import ChiralMetric
from modelhub.metrics.rasa import RASAMetric

__all__ = [
    "Metric",
    "MetricManager",
    "instantiate_metric_manager",
    "ChiralMetric",
    "RASAMetric",
]
```

**`src/modelhub/__init__.py`**:
```python
"""ModelForge: Open-source framework for biomolecular modeling.

This package provides the base framework for building and training biomolecular models.
"""

# Export key base classes at top level
from modelhub.callbacks.callback import BaseCallback
from modelhub.metrics.metric import Metric, MetricManager

# Version
from modelhub.version import __version__

__all__ = [
    "BaseCallback",
    "Metric",
    "MetricManager",
    "__version__",
]
```

### Phase 3: Fix RF3 Imports

#### Step 3.1: Update All RF3 Import Statements

**Find all broken imports**:
```bash
cd releases/rf3/
grep -r "from metrics.base import" src/
grep -r "from callbacks.base import" src/
grep -r "from inference_engines.base import" src/
```

**Replace with**:
```python
# OLD (broken)
from metrics.base import Metric
from callbacks.base import BaseCallback

# NEW (correct)
from modelhub.metrics.metric import Metric, MetricManager
from modelhub.callbacks.callback import BaseCallback
from modelhub.inference_engines.base import InferenceEngine
```

**Automated replacement**:
```bash
# In releases/rf3/src/
find . -name "*.py" -exec sed -i 's/from metrics\.base import/from modelhub.metrics.metric import/g' {} +
find . -name "*.py" -exec sed -i 's/from callbacks\.base import/from modelhub.callbacks.callback import/g' {} +
find . -name "*.py" -exec sed -i 's/from inference_engines\.base import/from modelhub.inference_engines.base import/g' {} +
```

#### Step 3.2: Update RF3 Utility Imports

```python
# Also update imports of shared utilities
from modelhub.utils.ddp import RankedLogger
from modelhub.utils.logging import suppress_warnings
from modelhub.utils.weights import load_checkpoint
from modelhub.utils.instantiators import instantiate_loggers, instantiate_callbacks
```

### Phase 4: Fix Build Configuration

#### Step 4.1: Update pyproject.toml

**Current**:
```toml
[project]
name = "rf3"
[project.scripts]
rf3 = "modelhub.cli:app"
[tool.hatch.build.targets.wheel]
packages = ["src/modelhub"]
```

**New**:
```toml
[project]
name = "modelforge"
description = "Open-source framework for biomolecular structure prediction and design"

# Minimal dependencies for base framework
dependencies = [
    "torch>=2.2.0,<3",
    "lightning>=2.4.0,<2.5",
    "hydra-core>=1.3.0,<1.4",
    "rootutils>=1.0.7,<1.1",
    "environs>=11.0.0,<12",
    "wandb>=0.15.10,<1",
    "rich>=13.9.4,<14",
    "jaxtyping>=0.2.17,<1",
    "beartype>=0.18.0,<1",
]

[project.optional-dependencies]
# RF3 model with its specific dependencies
rf3 = [
    "atomworks==1.0.2",
    "einops>=0.8.0,<1",
    "einx>=0.1.0,<1",
    "opt_einsum>=3.4.0,<4",
    "dm-tree>=0.1.6,<1",
    "cuequivariance_ops_cu12>=0.5.0; sys_platform == 'linux'",
    "cuequivariance_ops_torch_cu12>=0.5.0; sys_platform == 'linux'",
    "cuequivariance_torch>=0.5.0; sys_platform == 'linux'",
]

# All models
all = [
    "modelforge[rf3]",
]

# Development
dev = [
    "ruff==0.8.3",
    "pytest>=8.2.0,<9",
    # ... other dev deps
]

[project.scripts]
# Main CLI - needs to be created or use RF3 CLI directly
rf3 = "rf3.cli:app"  # Direct to RF3 for now

[tool.hatch.build.targets.wheel]
# Include both modelhub and rf3 (if using monorepo approach)
packages = ["src/modelhub"]
# OR include releases/rf3/src/rf3 via custom build hook

# For development: force-include releases
[tool.hatch.build.targets.wheel.force-include]
"releases/rf3/src/rf3" = "modelhub/models/rf3"  # If using monorepo approach
```

#### Step 4.2: Create Main CLI (If Needed)

**Option 1**: Keep `rf3` CLI pointing directly to RF3
```toml
[project.scripts]
rf3 = "rf3.cli:app"
```

**Option 2**: Create dispatcher CLI (future-proof)

**`src/modelhub/cli.py`**:
```python
"""Main ModelForge CLI."""

import typer

app = typer.Typer()

# Import RF3 CLI if available
try:
    from rf3.cli import app as rf3_app
    # Expose RF3 commands at top level
    for command_name, command in rf3_app.registered_commands:
        app.command(name=command_name)(command.callback)
except ImportError:
    pass  # RF3 not installed

if __name__ == "__main__":
    app()
```

Then:
```toml
[project.scripts]
modelforge = "modelhub.cli:app"
rf3 = "rf3.cli:app"  # Keep for backward compatibility
```

### Phase 5: Handle RF3 Package Integration

**Decision needed**: Choose approach from Phase 2, Part 2.

**Recommended: Option C (Hybrid)**

Use hatch build hooks to include RF3 from `releases/`:

**`hatch_build.py`** (in project root):
```python
"""Custom hatch build hook to include RF3 from releases/."""

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        """Include RF3 from releases/ directory."""
        if self.target_name == "wheel":
            # Add releases/rf3/src/rf3 to the wheel as modelhub/models/rf3
            build_data["force_include"] = {
                "releases/rf3/src/rf3": "modelhub/models/rf3"
            }
```

**In `pyproject.toml`**:
```toml
[tool.hatch.build.hooks.custom]
path = "hatch_build.py"
```

This way:
- Development happens in `releases/rf3/`
- Installation includes RF3 in the package
- Users can `pip install modelforge[rf3]`

### Phase 6: Update Tests

#### Step 6.1: Update Test Imports

**In `releases/rf3/tests/`**:
```python
# Update conftest.py and test files
from modelhub.metrics.metric import Metric
from modelhub.callbacks.callback import BaseCallback
```

**In root `tests/`**:
```python
# These already test shared utilities
from modelhub.utils.weights import load_checkpoint
from modelhub.metrics.metric import MetricManager
```

#### Step 6.2: Verify Test Execution

```bash
# Test shared utilities
pytest tests/

# Test RF3 (from releases/rf3/)
cd releases/rf3/
pytest tests/
```

### Phase 7: Update Documentation

#### Step 7.1: Update CLAUDE.md

Update import examples and package structure documentation.

#### Step 7.2: Update README.md

```markdown
# ModelForge

## Installation

```bash
# Base framework only
pip install modelforge

# With RF3
pip install modelforge[rf3]

# Development
pip install -e .[rf3,dev]
```

## Usage

```python
# Import base framework
from modelforge.callbacks import BaseCallback
from modelforge.metrics import Metric

# Import RF3
from modelforge.models.rf3 import RF3
```
```

---

## Migration Checklist

### Pre-Migration
- [ ] Backup current code
- [ ] Create feature branch: `git checkout -b refactor/restructure-package`
- [ ] Document current import patterns for reference

### Phase 1: Package Rename
- [ ] Decision: Keep `modelhub` or rename to `modelforge`?
- [ ] Update `pyproject.toml` name field
- [ ] Update all imports if renaming

### Phase 2: Restructure src/
- [ ] Rename `callbacks/base.py` → `callbacks/callback.py`
- [ ] Rename `metrics/base.py` → `metrics/metric.py`
- [ ] Create/update all `__init__.py` files with proper exports
- [ ] Update internal imports within src/modelhub/

### Phase 3: Fix RF3 Imports
- [ ] Find all `from metrics.base` imports in releases/rf3/
- [ ] Replace with `from modelhub.metrics.metric`
- [ ] Find all `from callbacks.base` imports
- [ ] Replace with `from modelhub.callbacks.callback`
- [ ] Update utility imports to use `modelhub.utils.*`
- [ ] Test that RF3 can import all needed modules

### Phase 4: Fix Build Config
- [ ] Update `pyproject.toml` project name
- [ ] Split dependencies: core vs rf3-specific
- [ ] Add `[project.optional-dependencies]` for rf3
- [ ] Fix `[project.scripts]` CLI entry point
- [ ] Update `[tool.hatch.build.targets.wheel]` packages
- [ ] Add build hooks if using hybrid approach

### Phase 5: RF3 Integration
- [ ] Decide on integration approach (A, B, or C)
- [ ] Implement chosen approach
- [ ] Test `pip install -e .` (base only)
- [ ] Test `pip install -e .[rf3]` (with RF3)

### Phase 6: Update Tests
- [ ] Update test imports
- [ ] Run shared tests: `pytest tests/`
- [ ] Run RF3 tests: `cd releases/rf3 && pytest tests/`
- [ ] Fix any import errors

### Phase 7: Documentation
- [ ] Update CLAUDE.md
- [ ] Update README.md
- [ ] Update any other documentation
- [ ] Add migration notes for contributors

### Phase 8: Validation
- [ ] Clean install test: `pip install -e .`
- [ ] RF3 install test: `pip install -e .[rf3]`
- [ ] Test CLI: `rf3 fold inputs=...`
- [ ] Run full test suite
- [ ] Test on clean environment

---

## Testing Strategy

### Test Scenarios

#### Scenario 1: Base Install Only
```bash
# Clean environment
python -m venv test-env
source test-env/bin/activate
pip install -e .

# Should work
python -c "from modelhub.callbacks import BaseCallback; print('OK')"
python -c "from modelhub.metrics import Metric; print('OK')"
python -c "from modelhub.utils.ddp import RankedLogger; print('OK')"

# Should fail (RF3 not installed)
python -c "from modelhub.models.rf3 import RF3"  # ImportError expected
```

#### Scenario 2: With RF3
```bash
# Clean environment
python -m venv test-env
source test-env/bin/activate
pip install -e .[rf3]

# Should work
python -c "from modelhub.models.rf3 import RF3; print('OK')"
python -c "from rf3.cli import app; print('OK')"

# Test CLI
rf3 fold inputs='releases/rf3/tests/data/5vht_from_json.json'
```

#### Scenario 3: Development Workflow
```bash
# Install in development mode with RF3
pip install -e .[rf3,dev]

# Make changes to releases/rf3/src/rf3/model/RF3.py
# Changes should be immediately available

python -c "from modelhub.models.rf3.model import RF3"  # Should see changes

# Run tests
pytest releases/rf3/tests/
```

---

## Risk Assessment

### High Risk Items
1. **Breaking all existing imports** - Requires updating many files
   - Mitigation: Use automated search/replace, test thoroughly

2. **Build configuration complexity** - Hatch build hooks can be tricky
   - Mitigation: Start with simple approach, test installation frequently

3. **Package name change** - Breaking change for any external users
   - Mitigation: Coordinate with team, announce breaking change

### Medium Risk Items
1. **CLI entry point changes** - Users may have scripts using old CLI
   - Mitigation: Keep backward-compatible entry points

2. **Test imports** - Many test files to update
   - Mitigation: Automated replacement, run tests incrementally

### Low Risk Items
1. **Documentation updates** - Time-consuming but low risk
2. **`__init__.py` exports** - Easy to fix if wrong

---

## Rollback Plan

If migration fails:
1. `git checkout main` - Revert to pre-migration state
2. Keep feature branch for later attempt
3. Document what failed for next iteration

---

## Future Considerations

### Adding New Models
Once restructured, adding a new model is straightforward:

```
releases/
├── rf3/               # Existing
└── proteinmpnn/       # New model
    ├── src/proteinmpnn/
    ├── configs/
    └── tests/
```

Then in `pyproject.toml`:
```toml
[project.optional-dependencies]
proteinmpnn = [
    "proteinmpnn-specific-deps",
]
all = [
    "modelforge[rf3,proteinmpnn]",
]
```

### Splitting Into Separate Packages (Future)
If models become too large, can later split:
- `modelforge-core` - Base framework
- `modelforge-rf3` - RF3 model
- `modelforge-proteinmpnn` - ProteinMPNN model

But this is much more complex and not recommended initially.

---

## Questions to Resolve

1. **Package name**: Keep `modelhub` or rename to `modelforge`?
   - Recommendation: Rename to `modelforge`

2. **RF3 integration**: Which approach (A, B, or C)?
   - Recommendation: Option C (hybrid with build hooks)

3. **CLI structure**: Single entry point or keep separate?
   - Recommendation: Keep `rf3` command separate for now

4. **Base file naming**: Rename `base.py` to match content?
   - Recommendation: Yes, rename to `callback.py`, `metric.py`

5. **Version strategy**: Single version or per-model versions?
   - Recommendation: Single version (simpler)

---

## Success Criteria

- [ ] `pip install modelforge` works
- [ ] `pip install modelforge[rf3]` works
- [ ] All imports are unambiguous and correct
- [ ] All tests pass
- [ ] CLI works: `rf3 fold inputs=...`
- [ ] Structure matches PyTorch Lightning patterns
- [ ] Documentation is updated
- [ ] No more `sys.path` manipulation needed

---

## Timeline Estimate

- Phase 1 (Rename decision): 1 hour
- Phase 2 (Restructure src/): 2-3 hours
- Phase 3 (Fix RF3 imports): 1-2 hours
- Phase 4 (Build config): 2-3 hours
- Phase 5 (RF3 integration): 2-4 hours
- Phase 6 (Update tests): 1-2 hours
- Phase 7 (Documentation): 1-2 hours
- Phase 8 (Validation): 2-3 hours

**Total**: 12-20 hours of focused work

---

## Conclusion

This restructuring will:
1. ✅ Fix all broken imports
2. ✅ Follow industry best practices (Lightning pattern)
3. ✅ Enable proper pip installation
4. ✅ Support selective model installation
5. ✅ Maintain development workflow in `releases/`
6. ✅ Provide clear, unambiguous imports
7. ✅ Scale to future models

The key is following PyTorch Lightning's pattern: base classes live WITH their implementations, no separate `contrib/` or overly complex directory nesting.
