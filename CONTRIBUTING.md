# Contributing to foundry

This contributor's guide is a non-exhaustive list of best practices for contributing to the foundry source code, adding a model, and/or contributing to the foundry documentation. These recommendations are a mixture of industry standards and expectations for this specific repository. 

## Table of Contents
- [Code Contributions](#code-contributions)
- [Adding a Model](#adding-a-model)
- [Documentation Contributions](#documentation-contributions)

## Code Contributions

### Code Organization
There is a strict dependency flow of Foundry -> [AtomWorks](https://github.com/RosettaCommons/atomworks). All models within Foundry use AtomWorks for manipulating and processing biomolecular structures, in both training and inference. 

Here is an overview of how this system is structured: 
- AtomWorks: I/O, preprocessing structures, data featurization
- Foundry: Model architectures, training, inference endpoints
- `models/<model>`: Released models that use the structure provided by Foundry and AtomWorks

### Installing Foundry in Editable Mode
Install both foundry and models in editable mode for development:
```bash
uv pip install -e '.[all,dev]'
```
This approach allows you to:
- Modify foundry shared utilities and see changes immediately
- Work on specific models without installing all models
- Add new models as independent packages in `models/`

### As You Code
1. Reduce cognitive overhead:
    <ol type="a">
        <li>Pick meaningful, descriptive variable names.</li>
        <li>Write docstrings and comments. All docstrings should be written using in the <a href=https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html>Google-format</a>.</li>
        <li>Follow <a href=https://peps.python.org/pep-0008/>PEP8 (Style Guide for Python Code)</a> whenever possible</li>
        <li>Follow <a href=https://peps.python.org/pep-0020/>PEP20 (The Zen of Python)</a> whenever possible.</li>
    </ol>
2. Write tests. Tests for the Foundry source code can be found in `foundry/tests`, tests for individual models will be in their respective directories. 
    ```{note}
    Running tests is not currently supported, test files may be missing. 
    ```

### As You Commit
Foundry comes with a `.pre-commit-config.yaml` that runs `make format` (via `ruff format`) before each commit, enable it once per clone: 
```bash
pip install pre-commit # if not already installed
pre-commit install
```
Once it is successfully installed, it will automatically format the repo whenever you run `git commit`, but you can apply it manually via `pre-commit run --all-files`.

Even with this, there are a few things to keep in mind to make sure your commits are easily reviewable: 
1. Keep commits as “one logical unit.” This means that each commit should be a set of related changes that accomplish one task, fix one bug, or implement one feature. Using an editor like [VS Code](https://code.visualstudio.com/docs/sourcecontrol/overview) or using [GitHub Desktop](https://docs.github.com/en/desktop) can help you stage related changes together.
1. Adhere to [semantic commit conventions](https://www.conventionalcommits.org/en/v1.0.0/).
1. Submit a draft PR so people know you are working on this & can provide advice/feedback early on.

### As you Finalize a PR
1. To make a PR merge your branch to production. The maintainers will review your PR.
1. Keep overall PR under <400 LOC (lines of code) (Rule of thumb: 500 LOC takes about 1h to review).


## Adding a Model
To be able to add new models as independent packages, make sure to install foundry and its models in 'editable' mode: 
```bash
uv pip install -e '.[all,dev]'
```
Once you have done this you can follow these steps to incorporate your model into the repository: 
1. Create a `models/<model_name>` directory with its own `pyproject.toml`
1. Add `foundry` as a dependency
1. Implement model-specific code in `models/<model-name>/src/`
1. Users can install this new model via `uv pip install -e ./models/<model_name>`

## Documentation Contributions
The external Foundry documentation is built using [Sphinx](https://www.sphinx-doc.org/en/master/#) and [GitHub Pages](https://docs.github.com/en/pages). 

To build the documentation you need to have the dependencies listed in `foundry/docs/docs_requirements.txt` installed, which can be easily done via
```bash
uv pip install -r docs/docs_requirements.txt
```

To build the documentation, navigate to the `docs` directory and run: 
```bash
make html
```
If you are new to Sphinx, please refer to the [Sphinx documentation](https://www.sphinx-doc.org/en/master/) for guidance on writing and formatting documentation. The documentation for Foundry uses MyST_parser so that documentation pages can be written in [Markdown](https://www.markdownguide.org/) or [ReStructured Text](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html).

### Organization
The `docs/source` directory contains the documentation about the Foundry source code. The `index.rst` file located here is the landing page for the documentation. 

Each model has its own `docs` folder (`foundry/models/<model_name>/docs`). The `foundry/docs/source/models` directory contains symlinks to these individual docs folders, this is necessary to allow Sphinx to see the model-specific documentation. If you are adding documentation for a new model, you will need to make a similar symlink: 
1. Make a `docs` directory in `foundry/models/<model_name>`
2. From the `foundry/docs/source/models` directory run
    ```bash
    ln -s ../../../models/<model_name>/docs <model_name>
    ```
 
Each of the model documentation directories have their own `index.md` file that organizes that model's documentation. 