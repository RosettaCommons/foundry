#!/usr/bin/env bash
# Seed determinism check, partial diffusion (noise a real structure by partial_t Angstrom
# and denoise it back; the 7v11 example from docs/examples/demo.json).
#   bash models/rfd3/scripts/verify_seed_determinism/verify_partial_diffusion.sh
# Exit code 0 = PASS (same seed -> bitwise identical; different seed -> different).
# See _common.sh for the environment variables it honours.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PARTIAL_T="${PARTIAL_T:-15.0}"
SPEC="${OUT_ROOT}/partial_diffusion_spec.json"
cat > "${SPEC}" <<EOF
{
    "partial_diffusion": {
        "input": "${FOUNDRY_ROOT}/models/rfd3/docs/input_pdbs/7v11.pdb",
        "ligand": "OQO",
        "partial_t": ${PARTIAL_T},
        "contig": "A431",
        "unindex": "A572-573",
        "select_fixed_atoms": {
            "A431": "TIP",
            "A572": "BKBN",
            "A573": "BKBN"
        },
        "allow_ligand_on_existing_chain": true
    }
}
EOF

verify_seed partial_diffusion "inputs=${SPEC}" \
  && echo "PASS: partial diffusion is seed-deterministic" \
  || { echo "FAIL: partial diffusion is not seed-deterministic"; exit 1; }
