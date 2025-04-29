.PHONY: init clean format

#################################################################################
# GLOBALS                                                                       #
#################################################################################

# Set the project directory
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

# Set conda/mamba binary, mamba if available, conda otherwise
CONDA_BINARY := $(shell command -v mamba || command -v conda)

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Format src directory using black
format:
	ruff format src tests scripts notebooks
	ruff check --fix src tests scripts notebooks

_github_token_error:
	@echo "==============================================================================="; \
	echo "Error: Environment variables GITHUB_USER and GITHUB_TOKEN must be set."; \
	echo ""; \
	echo "You need to set the environment variables GITHUB_USER and GITHUB_TOKEN."; \
	echo "You can create a personal access token on GitHub at:"; \
	echo "   https://github.com/settings/tokens"; \
	echo ""; \
	echo "For more info see: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic"; \
	echo ""; \
	echo "To expose these variables, you can use:"; \
	echo "export GITHUB_USER=<github-username>"; \
	echo "export GITHUB_TOKEN=<github-token>"; \
	echo ""; \
	echo "It is recommended that you set these tokens in your .bashrc or .zshrc file for future use."; \
	echo "===============================================================================";
	exit 1;

_check_conda:
	@echo "... checking if conda/mamba is installed"
	@command -v $(CONDA_BINARY) >/dev/null 2>&1 || { \
		echo "Error: Conda/mamba is not installed or not found in PATH" >&2; \
		exit 1; \
	}
	@echo "... found conda executable: $(CONDA_BINARY)"

_check_tokens:
	@echo "... checking if GITHUB_USER and GITHUB_TOKEN are set"
	@if [ -z "$(GITHUB_USER)" ] || [ -z "$(GITHUB_TOKEN)" ]; then \
		$(MAKE) _github_token_error; \
	fi
	@echo "... found GITHUB_USER ($(GITHUB_USER)) and GITHUB_TOKEN."

## Create a new conda environment and install modelhub
env:
	@echo "Creating modelhub conda environment: modelhub"

	@$(MAKE) --no-print-directory _check_tokens
	@$(MAKE) --no-print-directory _check_conda

	@$(CONDA_BINARY) env create -n modelhub --file environment.yaml
	@conda init
	@conda activate modelhub
	@pip install -e ".[dev]"
	@python -m biotite.setup_ccd


## Install modelhub locally into the current environment
install:
	# Install the conda requirements in the current activated environment
	$(CONDA_BINARY) env update --file environment.yaml
	# Install the pip requirements in the current activated environment
	@pip install -e ".[dev]"
	@python -m biotite.setup_ccd

## Build the apptainer image
# (For development)
base_apptainer:
	$(eval DATE := $(shell date +%Y-%m-%d))
	bash ./scripts/build_base_apptainer.sh

# (For inference)
# Installs modelhub, datahub, and cifutils into the apptainer image
inference_apptainer: 
	$(eval DATE := $(shell date +%Y-%m-%d))
	bash ./scripts/inference_apptainer.sh

## Run pytest and generate coverage report
test:
	pytest --cov=modelhub --cov-report=term-missing --cov-report=html --cov-report=xml tests -m "not very_slow"

## Run the contribute script to set you up for contributing to modelhub
contribute:
	@sh scripts/contribute.sh

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
