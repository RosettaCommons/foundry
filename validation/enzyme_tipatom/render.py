"""Create ligand-aligned PyMOL scenes from measured, pre-insertion coordinates."""

import json
from pathlib import Path

import numpy as np
import pymol

HERE = Path(__file__).resolve().parent
ROOT = HERE / "results"
ARMS = ["native_aw2", "fixed_aw2", "padded_aw2", "padded_aw3"]
pymol.finish_launching(["pymol", "-cq"])
cmd = pymol.cmd
cmd.load(
    str(
        HERE.parents[1]
        / "models/rfd3/docs/tutorials/enzyme_tutorial_files/1euv_lig.pdb"
    ),
    "input_theozyme",
)
reference = {
    a.name: np.asarray(a.coord)
    for a in cmd.get_model("input_theozyme and hetatm and not elem H").atom
}
objects = []
load_arms = ARMS + (
    ["diagnostic_aw3_mask"] if (ROOT / "diagnostic_aw3_mask").exists() else []
)
for arm in load_arms:
    for path in sorted((ROOT / arm).glob("s*/scores.json")):
        score = json.loads(path.read_text())
        name = f"{arm}_{path.parent.name}"
        expected = name + "_target"
        cmd.load(str(path.parent / "preinsert.pdb"), name)
        cmd.load(str(path.parent / "guideposts.pdb"), expected)
        ligand = {
            a.name: np.asarray(a.coord)
            for a in cmd.get_model(expected + " and hetatm and not elem H").atom
        }
        names = sorted(set(reference) & set(ligand))
        assert len(names) >= 3
        x = np.array([ligand[n] for n in names])
        y = np.array([reference[n] for n in names])
        u, _, vt = np.linalg.svd((x - x.mean(0)).T @ (y - y.mean(0)))
        correction = np.eye(3)
        correction[-1, -1] = np.linalg.det(u @ vt)
        rotation = u @ correction @ vt
        for obj in (name, expected):
            coords = cmd.get_coords(obj)
            cmd.load_coords((coords - x.mean(0)) @ rotation + y.mean(0), obj)
        tips = [
            f"(chain {r['chain']} and resi {r['resid']} and name {r['atom']})"
            for r in score["replacements"]
            if r["tip"]
        ]
        cmd.select(name + "_tips", name + " and (" + " or ".join(tips) + ")")
        cmd.hide("everything", name + " or " + expected)
        cmd.dss(name)
        cmd.show("cartoon", name)
        cmd.show("sticks", name + "_tips or (" + name + " and hetatm)")
        cmd.show("sticks", expected + " and not hetatm")
        # Keep badly placed, unbonded tip atoms visible as well as bonded sticks.
        cmd.show("spheres", name + "_tips or (" + expected + " and not hetatm)")
        cmd.set("sphere_scale", 0.16, name + " or " + expected)
        cmd.color("cyan", name)
        cmd.color("magenta", name + "_tips")
        cmd.color("gray70", name + " and hetatm")
        cmd.color("green", expected)
        cmd.util.cnc(name + " or " + expected)
        cmd.group(name + "_group", name + " " + expected + " " + name + "_tips")
        objects.append((name, expected, score))

cmd.bg_color("white")
cmd.set("ray_opaque_background", 1)
cmd.set("antialias", 2)
cmd.set("ray_shadows", 0)
cmd.disable("all")
representatives = [
    (name, expected, score)
    for name, expected, score in objects
    if name in {arm + "_s11_d0" for arm in ARMS}
]
assert len(representatives) == 4
for slot, (name, expected, score) in enumerate(representatives, 1):
    cmd.enable(name + "_group")
    cmd.enable(name)
    cmd.enable(expected)
    cmd.set("grid_slot", slot, name)
    cmd.set("grid_slot", slot, expected)
cmd.set("grid_mode", 1)
cmd.set("grid_max", 4)
selection = " or ".join(name for name, _, _ in representatives)
cmd.orient(selection)
cmd.zoom(selection, buffer=3)
cmd.scene("representative_overviews", "store")
cmd.png(str(ROOT / "overview.png"), width=1800, height=1400, dpi=150, ray=1)
targets = " or ".join(expected for _, expected, _ in representatives)
cmd.set("cartoon_transparency", 0.75)
cmd.zoom("(" + targets + ") and not hetatm", buffer=3)
cmd.scene("representative_active_sites", "store")
cmd.png(str(ROOT / "active_sites.png"), width=1800, height=1400, dpi=150, ray=1)

cmd.disable("all")
cmd.set("grid_mode", 0)
name, expected, score = max(objects, key=lambda item: item[2]["tip_atom_rmsd"])
cmd.enable(name + "_group")
cmd.enable(name)
cmd.enable(expected)
cmd.zoom(expected + " and not hetatm", buffer=4)
cmd.scene("worst_tip_insertion", "store")
cmd.png(str(ROOT / "worst_insertion.png"), width=1200, height=1000, dpi=150, ray=1)
(ROOT / "image_manifest.json").write_text(
    json.dumps(
        {
            "panel_order": [name for name, _, _ in representatives],
            "worst": {"name": name, "tip_atom_rmsd": score["tip_atom_rmsd"]},
            "colors": {
                "scaffold": "cyan",
                "generated_tips": "magenta",
                "target_motif": "green",
                "ligand": "gray",
            },
        },
        indent=2,
    )
)
cmd.scene("representative_overviews", "recall")
cmd.save(str(ROOT / "enzyme_tipatom_comparison.pse"))
cmd.quit()
