#!/bin/bash
# Replace module prefixes across .py, .ipynb, .toml, .yaml files
find . -type f \( -name "*.py" -o -name "*.ipynb" -o -name "*.toml" -o -name "*.yaml" \) \
  -exec sed -i 's/\bprojects.aa_design\b/rfd3/g' {} +


grep -R --color -nE 'aa_design' .

