"""Optional chemical-symmetry check; no coordinate fitting or atom omission."""

import numpy as np


def symmetry_aware_tip_rmsd(replacements):
    groups = {}
    for atom in replacements:
        if atom["tip"]:
            groups.setdefault(atom["source"], []).append(atom)
    errors = []
    swaps = []
    for source, atoms in groups.items():
        names = [atom["atom"] for atom in atoms]
        predicted = np.array([atom["coord"] for atom in atoms], dtype=float)
        target = np.array([atom["target"] for atom in atoms], dtype=float)
        if not np.isfinite([predicted, target]).all():
            raise ValueError("Nonfinite coordinates")
        squared_error = float(np.square(predicted - target).sum())
        if atoms[0]["restype"] == "ASP" and {"OD1", "OD2"} <= set(names):
            i, j = names.index("OD1"), names.index("OD2")
            exchanged = target.copy()
            exchanged[[i, j]] = target[[j, i]]
            alternative = float(np.square(predicted - exchanged).sum())
            if alternative < squared_error:
                squared_error = alternative
                swaps.append(source)
        errors.extend([squared_error / len(atoms)] * len(atoms))
    if not errors:
        raise ValueError("No tip atoms")
    return float(np.sqrt(np.mean(errors))), swaps
