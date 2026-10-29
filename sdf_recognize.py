"""sdf_recognize.py — the features a recognised tree misses, found as 3D fields (no axis assumed).

The scan becomes a signed distance field whose sign is the generalized winding number — well
defined for the open meshes scanners produce (Jacobson et al.; igl's fast winding number) — and
so does the IR spec's rebuilt solid. Two decompositions on that grid:

  residual  scan minus tree: every connected component where they disagree is one missed
            feature, signed — "added" is material the scan has and the tree lacks (a boss, a
            rib, a ledge), "removed" is material the tree claims wrongly (a recess, a hole,
            an inset). This is mesh_compare.relief() without the one-axis assumption: it sees
            end faces, cross features and oblique surfaces the Z slicer cannot.
  carve     base / added / removed by morphology alone, no tree needed: OPENING the solid by
            radius r strips every feature thinner than r standing proud (ribs, bosses);
            CLOSING by r fills every hollow narrower than r (recesses, slots, small holes).
            base + added - removed is the part again, at the scale r chooses.

Each component reports volume, bounding box, centroid, its reach (how proud or deep, from the
other body's field) and, on request, a marching-cubes mesh to look at.

    python3 sdf_recognize.py scan.stl --spec tree.json            # residual decomposition
    python3 sdf_recognize.py scan.stl --carve 1.5                 # morphological decomposition
    python3 sdf_recognize.py scan.stl --spec t.json --spin 359 --meshes out/parts/ --emit r.json

The scan and the spec are aligned the way mesh_compare.compare() aligns them: bounding-box
centres on the same XY, both standing on z = 0, the spec spun by `--spin` degrees about Z
(pass compare()'s reported spin; 0 is fine for parts recognised from this very scan).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "targets"))

MAX_CELLS = 8e6


def load(path):
    import trimesh
    return trimesh.load(str(path), force="mesh")


def grid_for(bounds, margin=2.0, max_cells=MAX_CELLS):
    """A uniform grid covering `bounds` plus `margin` mm: (origin, spacing, shape)."""
    lo, hi = np.asarray(bounds[0]) - margin, np.asarray(bounds[1]) + margin
    ext = hi - lo
    h = float((ext.prod() / max_cells) ** (1 / 3))
    shape = tuple(int(np.ceil(e / h)) + 1 for e in ext)
    return lo, h, shape


def sdf(mesh, origin, h, shape, batch=500_000):
    """The mesh's signed distance (winding-number sign) sampled on the grid, as an array."""
    import igl
    ax = [origin[i] + h * np.arange(shape[i]) for i in range(3)]
    pts = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3)
    V, F = mesh.vertices.astype(np.float64), mesh.faces.astype(np.int64)
    out = np.empty(len(pts))
    for i in range(0, len(pts), batch):
        out[i:i + batch] = igl.signed_distance(
            pts[i:i + batch], V, F, igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER)[0]
    return out.reshape(shape)


def _align(mesh, like, spin=0.0, mode="frame", dz=0.0):
    """The spec's solid posed against the scan: spun about Z, otherwise left in its own frame —
    a tree recognised FROM this scan already shares its coordinates, and centring bounding
    boxes would shift it by half of every asymmetric feature the scan has and the tree lacks
    (the very things being measured). `mode="bbox"` restores centre alignment for a spec from
    elsewhere."""
    import trimesh
    m = mesh.copy()
    if spin:
        m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(spin), [0, 0, 1]))
    if dz:
        m.apply_translation([0, 0, dz])
    if mode == "bbox":
        c, cl = m.bounds.mean(axis=0), like.bounds.mean(axis=0)
        m.apply_translation([cl[0] - c[0], cl[1] - c[1], like.bounds[0][2] - m.bounds[0][2]])
    return m


def _open(ds, r, h):
    """Morphological opening by a ball of `r` mm. The EROSION comes from the true distance
    field (half a voxel of quantisation in a mask-based erosion is enough to let a feature at
    the scale of `r` survive and regrow); the dilation back walks the grid, with 3/4 of a
    voxel of slack — in exact arithmetic the rebuilt surface lands exactly ON the original,
    so without slack every surface voxel is a coin flip and the difference is all skin."""
    from scipy import ndimage
    return ndimage.distance_transform_edt(~(ds < -r)) * h <= r + 0.75 * h


def _close(ds, r, h):
    """Closing: what the complement's opening refuses — hollows a ball of `r` cannot enter."""
    from scipy import ndimage
    return ~(ndimage.distance_transform_edt(~(ds > r)) * h <= r + 0.75 * h)


def _components(mask, reach_field, origin, h, min_volume, min_reach):
    """Connected components of `mask` as feature candidates, largest first. `reach_field` is the
    OTHER body's sdf: its most positive value inside a component is how far the feature stands
    from that body (a boss's height, a recess's depth). A component reaching less than
    `min_reach` is the skin where two tessellations of one surface disagree, not a feature."""
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        sel = lab == i
        vol = float(sel.sum()) * h ** 3
        reach = float(reach_field[sel].max())
        if vol < min_volume or reach < min_reach:
            continue
        idx = np.argwhere(sel)
        lo, hi = idx.min(axis=0), idx.max(axis=0) + 1
        out.append({"volume_mm3": round(vol, 1),
                    "bbox": [[round(float(origin[k] + h * lo[k]), 2) for k in range(3)],
                             [round(float(origin[k] + h * hi[k]), 2) for k in range(3)]],
                    "centroid": [round(float(origin[k] + h * idx[:, k].mean()), 2) for k in range(3)],
                    "reach_mm": round(reach, 2),
                    "_sel": (lab, i)})
    return sorted(out, key=lambda c: -c["volume_mm3"])


def component_mesh(comp, origin, h):
    """The component as a trimesh (marching cubes over its padded mask), for inspection."""
    import trimesh
    from skimage import measure
    lab, i = comp["_sel"]
    m = np.pad(lab == i, 1)
    verts, faces, _, _ = measure.marching_cubes(m.astype(np.float32), 0.5, spacing=(h, h, h))
    return trimesh.Trimesh(verts + (np.asarray(origin) - h), faces[:, ::-1])   # outward normals


def residual(scan, spec_or_mesh, spin=0.0, min_volume=5.0, min_reach=0.4, align="frame",
             dz=0.0, max_cells=MAX_CELLS):
    """Scan vs tree as fields. Returns {"grid", "added": [components], "removed": [...]} —
    added: the scan has material the tree lacks; removed: the tree claims material the scan
    refutes. Components smaller than `min_volume` mm³ are noise and dropped."""
    import mesh_compare as mc
    tree_mesh = spec_or_mesh if hasattr(spec_or_mesh, "vertices") else mc.spec_mesh(spec_or_mesh)
    tree_mesh = _align(tree_mesh, scan, spin, align, dz)
    origin, h, shape = grid_for(scan.bounds, max_cells=max_cells)
    ds, dt = sdf(scan, origin, h, shape), sdf(tree_mesh, origin, h, shape)
    inside_s, inside_t = ds < 0, dt < 0
    return {"grid": {"origin": [round(float(c), 3) for c in origin], "spacing": round(h, 4),
                     "shape": list(shape)},
            "added": _components(inside_s & ~inside_t, dt, origin, h, min_volume, min_reach),
            "removed": _components(inside_t & ~inside_s, ds, origin, h, min_volume, min_reach)}


def deviation(scan, spec_or_mesh, spin=0.0, dz=0.0, align="frame"):
    """One-sided scan -> CAD deviation: the distance from every scan vertex to the rebuilt
    solid's surface. This is the score volumetric overlap cannot give honestly: it uses only
    points the scanner RECORDED, so a torn cavity (whose interior winding-number fill is
    fiction) still scores by its real walls. Returns {"median_mm", "p90_mm", "p99_mm",
    "max_mm", "points"}."""
    import igl
    import mesh_compare as mc
    tree_mesh = spec_or_mesh if hasattr(spec_or_mesh, "vertices") else mc.spec_mesh(spec_or_mesh)
    tree_mesh = _align(tree_mesh, scan, spin, align, dz)
    P = scan.vertices.astype(np.float64)
    V, F = tree_mesh.vertices.astype(np.float64), tree_mesh.faces.astype(np.int64)
    d = np.abs(igl.signed_distance(P, V, F, igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER)[0])
    return {"median_mm": round(float(np.median(d)), 3),
            "p90_mm": round(float(np.percentile(d, 90)), 3),
            "p99_mm": round(float(np.percentile(d, 99)), 3),
            "max_mm": round(float(d.max()), 2), "points": len(d)}


def carve(scan, r=1.5, min_volume=5.0, min_reach=0.4, max_cells=MAX_CELLS):
    """Base / added / removed by morphology alone: whatever OPENING by `r` strips was standing
    proud of the base (ribs, bosses); whatever CLOSING fills was hollowed out of it (recesses,
    slots narrower than r). No tree needed — the scale-r decomposition a feature tree would
    redraw. An r-ball also rounds every sharp edge by about 0.29 r, so nothing shallower than
    that is distinguishable from the base's own edges: the reach gate rises to 0.35 r."""
    min_reach = max(min_reach, 0.35 * r)
    origin, h, shape = grid_for(scan.bounds, max_cells=max_cells)
    ds = sdf(scan, origin, h, shape)
    inside = ds < 0
    base = _open(ds, r, h)
    filled = _close(ds, r, h)
    return {"grid": {"origin": [round(float(c), 3) for c in origin], "spacing": round(h, 4),
                     "shape": list(shape)},
            "base_volume_mm3": round(float(base.sum()) * h ** 3, 1),
            "added": _components(inside & ~base, -ds, origin, h, min_volume, min_reach),
            "removed": _components(filled & ~inside, ds, origin, h, min_volume, min_reach)}


def _report(out, meshes_dir, origin, h):
    for kind in ("added", "removed"):
        print(f"{kind}: {len(out[kind])} component(s)")
        for k, c in enumerate(out[kind]):
            print(f"  {c['volume_mm3']:9.1f} mm3  reach {c['reach_mm']:5.2f}  "
                  f"centre {c['centroid']}  bbox {c['bbox']}")
            if meshes_dir:
                component_mesh(c, origin, h).export(str(Path(meshes_dir) / f"{kind}_{k + 1}.stl"))
        for c in out[kind]:
            c.pop("_sel", None)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Missed features as 3D fields (winding-number SDF).")
    ap.add_argument("mesh")
    ap.add_argument("--spec", help="IR JSON to difference against")
    ap.add_argument("--spin", type=float, default=0.0, help="spin the spec about Z (deg)")
    ap.add_argument("--dz", type=float, default=0.0,
                    help="lift the spec by this much first (a mesh_compare tree's z = 0 is the "
                         "scan's first level floor)")
    ap.add_argument("--align", choices=("frame", "bbox"), default="frame",
                    help="'frame': the spec shares the scan's coordinates (default); "
                         "'bbox': centre bounding boxes for a spec from elsewhere")
    ap.add_argument("--carve", type=float, metavar="R", help="morphological decomposition at radius R")
    ap.add_argument("--deviation", action="store_true",
                    help="with --spec: also print the one-sided scan -> CAD deviation")
    ap.add_argument("--min-volume", type=float, default=5.0)
    ap.add_argument("--meshes", help="write each component as an STL into this directory")
    ap.add_argument("--emit", help="write the report as JSON")
    a = ap.parse_args()
    scan = load(a.mesh)
    if a.meshes:
        Path(a.meshes).mkdir(parents=True, exist_ok=True)
    if a.carve is not None:
        out = carve(scan, a.carve, a.min_volume)
    elif a.spec:
        out = residual(scan, json.loads(Path(a.spec).read_text()), a.spin, a.min_volume,
                       align=a.align, dz=a.dz)
    else:
        ap.error("give --spec or --carve")
    g = out["grid"]
    print(f"grid {g['shape']} at {g['spacing']} mm")
    _report(out, a.meshes, g["origin"], g["spacing"])
    if a.deviation and a.spec:
        dev = deviation(scan, json.loads(Path(a.spec).read_text()), a.spin, a.dz, a.align)
        out["deviation"] = dev
        print(f"deviation scan->CAD: median {dev['median_mm']}  p90 {dev['p90_mm']}  "
              f"p99 {dev['p99_mm']}  max {dev['max_mm']} mm  ({dev['points']} points)")
    if a.emit:
        Path(a.emit).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
