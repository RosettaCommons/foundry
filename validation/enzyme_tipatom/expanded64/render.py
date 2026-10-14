"""Create ligand-aligned PyMOL scenes from measured, pre-insertion coordinates."""

import json
from pathlib import Path

import numpy as np
import pymol
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE / "results"
ARMS = ["native_aw2", "fixed_aw2", "padded_aw2", "padded_aw3"]
pymol.finish_launching(["pymol", "-cq"])
cmd = pymol.cmd
cmd.set("max_threads", 4)
cmd.load(
    str(
        HERE.parents[2]
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
        native_files = [
            p for p in path.parent.glob("*.json") if p.name != "scores.json"
        ]
        assert len(native_files) == 1
        score["insertion_rmsd"] = json.loads(native_files[0].read_text())["metrics"][
            "insertion_rmsd"
        ]
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

font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
font = ImageFont.truetype(font_path, 22)
small_font = ImageFont.truetype(font_path, 16)
labels = dict(
    zip(
        ARMS,
        ["Original / AW2", "Fixed / AW2", "Fixed + #397 / AW2", "Fixed + #397 / AW3"],
    )
)
manifest = {
    "metric": "native JSON metrics.insertion_rmsd",
    "panels": {},
    "contact_sheets": {},
    "colors": {
        "scaffold": "cyan",
        "generated_tips": "magenta",
        "target_motif": "green",
        "ligand": "gray",
    },
}
by_arm = {
    arm: sorted(
        [o for o in objects if o[2]["arm"] == arm],
        key=lambda o: (o[2]["insertion_rmsd"], o[2]["seed"], o[2]["sample"]),
    )
    for arm in ARMS
}
assert all(len(items) == 64 for items in by_arm.values())
# Upper middle order statistic (rank 33/64), never the best design.
representatives = [by_arm[arm][32] for arm in ARMS]
worst = [by_arm[arm][-1] for arm in ARMS]


def enable(items):
    cmd.disable("all")
    for slot, (name, target, _) in enumerate(items, 1):
        cmd.enable(name + "_group")
        cmd.enable(name)
        cmd.enable(target)
        cmd.set("grid_slot", slot, name)
        cmd.set("grid_slot", slot, target)
    cmd.set("grid_mode", int(len(items) > 1))
    cmd.set("grid_max", len(items))


def panels(items, key, closeup):
    enable(items)
    selection = " or ".join(name for name, _, _ in items)
    targets = " or ".join(target for _, target, _ in items)
    cmd.set("cartoon_transparency", 0.75 if closeup else 0)
    cmd.orient(selection)
    cmd.zoom("(" + targets + ") and not hetatm" if closeup else selection, buffer=4)
    cmd.scene(key, "store")
    path = ROOT / (key + ".png")
    cmd.png(str(path), width=1800, height=1400, dpi=150, ray=1)
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for i, (name, target, score) in enumerate(items):
        x, y = (i % 2) * 900, (i // 2) * 700
        draw.rectangle((x, y, x + 899, y + 58), fill="white")
        caption = f"{labels[score['arm']]} | seed {score['seed']} / {score['sample']}"
        draw.text((x + 15, y + 4), caption, font=font, fill="black")
        draw.text(
            (x + 15, y + 31),
            f"Insertion RMSD: {score['insertion_rmsd']:.3f} A",
            font=font,
            fill="black",
        )
    image.save(path)
    manifest["panels"][key] = [
        {"name": n, "insertion_rmsd": s["insertion_rmsd"]} for n, _, s in items
    ]


panels(representatives, "median_overviews", False)
panels(representatives, "median_active_sites", True)
panels(worst, "worst_active_sites", True)
# Show every output, ordered by seed/sample; score labels highlight failures.
for arm in ARMS:
    sheet = Image.new("RGB", (2400, 8 * 275 + 60), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (20, 15),
        labels[arm] + " | all 64 designs | JSON insertion RMSD (A)",
        font=font,
        fill="black",
    )
    items = sorted(by_arm[arm], key=lambda o: (o[2]["seed"], o[2]["sample"]))
    for i, item in enumerate(items):
        name, target, score = item
        enable([item])
        cmd.set("cartoon_transparency", 0)
        cmd.orient(name)
        cmd.zoom(name, buffer=2)
        tile = ROOT / "contact_tile.png"
        cmd.png(str(tile), width=300, height=245, dpi=100, ray=1)
        x, y = (i % 8) * 300, (i // 8) * 275 + 60
        sheet.paste(Image.open(tile).convert("RGB"), (x, y))
        draw.text(
            (x + 8, y + 248),
            f"s{score['seed']} / {score['sample']}   {score['insertion_rmsd']:.3f} A",
            font=small_font,
            fill="black" if score["insertion_rmsd"] < 1 else "red",
        )
    sheet.save(ROOT / (arm + "_all64.png"))
    manifest["contact_sheets"][arm] = [n for n, _, _ in items]
(ROOT / "contact_tile.png").unlink()
(ROOT / "image_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
cmd.scene("median_overviews", "recall")
cmd.save(str(ROOT / "enzyme64_comparison.pse"))
cmd.quit()
