"""deviation_map.py — WHERE a recognised tree disagrees with its scan, and closing that loop.

sdf_recognize.deviation() gives the score a Geomagic deviation colormap summarises: one number.
This module keeps the colormap itself, and makes it work:

  signed_map()   the SIGNED radial deviation CAD - scan over the part's skin, as a (z, theta)
                 grid. Both skins are read by mesh_compare.displacement() in ONE pinned frame
                 (the scan's axis and z range), so every cell of the scan meets the same cell of
                 the rebuilt tree. Red (+) = the tree stands proud of the scan; blue (-) = the
                 tree is low, the scan has material it lacks.
  attribute()    the map divided among the features that own it: per placed feature
                 (prism_cut / prism_pad, patterns expanded) and per base level, how much error
                 sits on it and which way it leans — a ranked work list — plus every region of
                 real disagreement NO feature owns (a missed boss, a missed recess).
  render()       the map as a PNG with the worst features outlined and named. Looking at it is
                 how this module was designed: a median deviation cannot say which feature is
                 wrong, the map says it at a glance.
  square_roll()  the spin about Z that puts the scan's mirror plane on theta = 0. A scan rolled
                 a few degrees off its design axes reads every flat floor as tilted and every
                 symmetric pair as unequal; squared first, both come out as the designer drew them.
  refine()       the map turned into corrections, iterated: each placed feature's floor moves
                 along its normal by the deviation on its floor; its outline is stretched and
                 shifted (u, v) by the deviation integrated ACROSS its edges (a strip of
                 one-sided error along an edge is the edge in the wrong place, and its area over
                 the step height is how far); each base level's outline radii follow the
                 deviation on the bare body around them. Every change is re-scored on its own
                 region after the rebuild, and a change that made its region worse is reverted —
                 refinement never trades a good feature for a better average.

What it will not do, by construction: invent a feature (unexplained regions are reported, never
modelled here), touch engravings or tapered cone steps (reported as not refined), or edit a
non-placed feature kind. The tree's own structure is the recognizer's job; this is its loop.

    python3 deviation_map.py scan.stl --spec tree.json --png dev.png --report dev.json
    python3 deviation_map.py scan.stl --spec tree.json --refine tree_refined.json --iters 3
    python3 deviation_map.py scan.stl --square              # just the roll that squares the scan
"""
import copy
import json
import math
import os
import warnings

import numpy as np

PLACED = ("prism_cut", "prism_pad")


# ---------------------------------------------------------------------------------- the map
def signed_map(scan, spec_or_mesh, dz=0.25, nth=720, spin=0.0, align="frame", scan_map=None):
    """The signed radial deviation CAD - scan on a (z, theta) grid.

    The scan's skin fixes the frame (its fitted axis and z range); the tree's skin is read in
    exactly that frame. `scan_map` reuses a displacement() of the scan already taken with the
    same dz / nth (refine() maps the scan once). The tree is posed like sdf_recognize poses it
    (`spin` about Z, `align` "frame" or "bbox").

    Returns {"axis", "z0", "dz", "nth", "zs", "th_deg", "R_scan", "R_cad", "dev", "valid",
    "cad_mesh", "stats"} — dev in mm, NaN where either skin has no data."""
    import mesh_compare as mc
    import sdf_recognize as sdf
    cad = spec_or_mesh if hasattr(spec_or_mesh, "vertices") else mc.spec_mesh(spec_or_mesh)
    cad = sdf._align(cad, scan, spin, align)
    s = scan_map or mc.displacement(scan, dz=dz, nth=nth, harmonics=0)
    nz = s["R"].shape[0]
    c = mc.displacement(cad, dz=s["dz"], nth=s["nth"], harmonics=0, axis=s["axis"],
                        z_range=(s["z0"], s["z0"] + nz * s["dz"]))
    Rc = c["R"]
    if Rc.shape[0] != nz:                       # float rounding of the pinned range: pad / trim
        Rc = np.vstack([Rc, np.full((max(0, nz - Rc.shape[0]), Rc.shape[1]), np.nan)])[:nz]
    valid = s["valid"] & ~np.isnan(Rc)
    dev = np.where(valid, Rc - s["R"], np.nan)
    return {"axis": np.asarray(s["axis"], dtype=float), "z0": s["z0"], "dz": s["dz"],
            "nth": s["nth"], "zs": s["zs"], "th_deg": (np.arange(s["nth"]) + 0.5) * 360.0 / s["nth"],
            "R_scan": s["R"], "R_cad": Rc, "dev": dev, "valid": valid, "cad_mesh": cad,
            "stats": _stats(dev[valid]), "scan_map": s}


def _stats(d):
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if not len(d):
        return {"cells": 0}
    a = np.abs(d)
    return {"cells": int(len(d)), "median_abs_mm": round(float(np.median(a)), 3),
            "p90_abs_mm": round(float(np.percentile(a, 90)), 3),
            "p99_abs_mm": round(float(np.percentile(a, 99)), 3),
            "mean_signed_mm": round(float(d.mean()), 3),
            "proud_pct": round(100.0 * float((d > 0.1).mean()), 1),
            "low_pct": round(100.0 * float((d < -0.1).mean()), 1)}


def _cell(dm, pts):
    """(row, col) of world points projected radially onto the map; -1 where off the map."""
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    ax = dm["axis"]
    row = np.floor((pts[:, 2] - dm["z0"]) / dm["dz"]).astype(int)
    col = np.floor(np.arctan2(pts[:, 1] - ax[1], pts[:, 0] - ax[0]) / (2 * np.pi) * dm["nth"]).astype(int)
    col %= dm["nth"]
    off = (row < 0) | (row >= dm["dev"].shape[0])
    row[off] = -1
    return row, col


def _lookup(dm, key, pts):
    row, col = _cell(dm, pts)
    out = np.full(len(row), np.nan)
    ok = row >= 0
    out[ok] = dm[key][row[ok], col[ok]]
    return out


# ------------------------------------------------------------------- placed-feature geometry
def _frame(f):
    o = np.asarray(f["origin"], dtype=float)
    n = np.asarray(f["normal"], dtype=float)
    n /= np.linalg.norm(n)
    x = np.asarray(f["xdir"], dtype=float)
    x -= n * (x @ n)
    x /= np.linalg.norm(x)
    return o, n, x, np.cross(n, x)


def _densify(wire, step=0.75):
    """A wire's (u, v) outline with bulge arcs expanded and long edges subdivided."""
    pts = []
    m = len(wire)
    for i in range(m):
        p, q = wire[i], wire[(i + 1) % m]
        a, b = np.array(p[:2], dtype=float), np.array(q[:2], dtype=float)
        bulge = float(p[2]) if len(p) > 2 else 0.0
        L = float(np.linalg.norm(b - a))
        k = max(1, int(math.ceil(L / step)))
        if abs(bulge) > 1e-9 and L > 1e-9:
            # DXF bulge b = tan(sweep / 4), positive = counter-clockwise from a to b. The centre
            # sits off the chord midpoint along its left normal by L (1 - b^2) / (4 b).
            sweep = 4 * math.atan(bulge)
            left = np.array([-(b - a)[1], (b - a)[0]]) / L
            c = (a + b) / 2 + left * L * (1 - bulge * bulge) / (4 * bulge)
            r = float(np.linalg.norm(a - c))
            a0 = math.atan2(a[1] - c[1], a[0] - c[0])
            k = max(k, int(math.ceil(abs(sweep) * r / step)))
            for j in range(k):
                t = a0 + sweep * j / k
                pts.append(c + r * np.array([math.cos(t), math.sin(t)]))
        else:
            for j in range(k):
                pts.append(a + (b - a) * j / k)
    return np.array(pts)


def _signed_area(uv):
    x, y = uv[:, 0], uv[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _outline(f):
    """The feature's outer wire as dense (u, v), counter-clockwise."""
    uv = _densify(f["polys"][0])
    return uv[::-1] if _signed_area(uv) < 0 else uv


def _world(f, uv):
    o, n, x, y = _frame(f)
    uv = np.atleast_2d(uv)
    return o + uv[:, :1] * x + uv[:, 1:2] * y


def _r_ref(dm, f):
    o = np.asarray(f["origin"], dtype=float)
    return max(float(np.hypot(o[0] - dm["axis"][0], o[1] - dm["axis"][1])), 1.0)


def footprint(dm, f, grow=0.0):
    """Boolean (z, theta) mask of the map cells a placed feature's outline covers, grown by
    `grow` mm. The outline is projected radially onto the skin and handled in unwrapped
    (arc-length, z) millimetres around the feature's own angle, so it never tears at 0/360."""
    import shapely
    from shapely.geometry import Polygon
    uv = _outline(f)
    w = _world(f, uv)
    ax = dm["axis"]
    o = np.asarray(f["origin"], dtype=float)
    th0 = math.atan2(o[1] - ax[1], o[0] - ax[0])
    r = _r_ref(dm, f)
    th = np.arctan2(w[:, 1] - ax[1], w[:, 0] - ax[0])
    s = ((th - th0 + np.pi) % (2 * np.pi) - np.pi) * r
    poly = Polygon(np.c_[s, w[:, 2]]).buffer(grow if grow else 0.0)
    if poly.is_empty:
        return np.zeros(dm["dev"].shape, bool)
    s0, z0, s1, z1 = poly.bounds
    zs, nth = dm["zs"], dm["nth"]
    rows = np.where((zs >= z0 - dm["dz"]) & (zs <= z1 + dm["dz"]))[0]
    th_c = np.radians(dm["th_deg"])
    ds = ((th_c - th0 + np.pi) % (2 * np.pi) - np.pi) * r
    cols = np.where((ds >= s0 - 1.0) & (ds <= s1 + 1.0))[0]
    mask = np.zeros(dm["dev"].shape, bool)
    if len(rows) and len(cols):
        S, Z = np.meshgrid(ds[cols], zs[rows])
        mask[np.ix_(rows, cols)] = shapely.contains_xy(poly, S, Z)
    return mask


# ------------------------------------------------------------------------ spec bookkeeping
def _lowered(spec):
    """The lowered placed features, each tagged with where its evidence belongs:
    (lowered feature, source name in the symbolic spec or None for engraving-derived)."""
    import ir as IR
    low = IR.lower(spec)
    masters = {f["name"] for f in spec["features"] if f["kind"] in PLACED}
    pats = {}
    for f in spec["features"]:
        if f["kind"] == "pattern":
            for i in range(1, f["count"]):
                pats[f"{f['name']}_{i}"] = f["of"]
    out = []
    for f in low["features"]:
        if f["kind"] not in PLACED:
            continue
        src = f["name"] if f["name"] in masters else pats.get(f["name"])
        out.append((f, src if src in masters else None))
    return out


def _levels(spec):
    """Base levels: every pad, with its z band — replaying lower()'s stacking rule (a sketch on
    a face's top starts at the part's running top). Returns [(pad, sketch, z0, z1)]."""
    sk, out = {}, []
    zlo = zhi = None
    for f in spec["features"]:
        if f["kind"] == "sketch":
            on = f.get("on")
            side = (on or {}).get("side", "top") if on else None
            z0 = (zhi if side == "top" else zlo) if on and zhi is not None else 0.0
            sk[f["name"]] = (f, z0)
        elif f["kind"] == "pad" and f.get("sketch") in sk:
            s, z0 = sk[f["sketch"]]
            L = float(f["length"])
            a, b = (z0 - L / 2, z0 + L / 2) if f.get("symmetric") else (z0, z0 + L)
            out.append((f, s, a, b))
            zlo = a if zlo is None else min(zlo, a)
            zhi = b if zhi is None else max(zhi, b)
    return out


# ------------------------------------------------------------------------------ attribution
def attribute(dm, spec, unexplained_mm=None, min_area_mm2=4.0):
    """The map divided among its owners.

    features: per placed feature (pattern instances and engraving facets included under their
    lowered names), per base level (its z band, minus every placed footprint) — cells, area,
    median / mean-absolute / p90 deviation, which way it leans, and error_mm3 = the integral of
    |dev| over its cells, the ranking key (a wide feature off by a little can matter more than
    a pinhole off by a lot).

    unexplained: connected regions where |dev| exceeds `unexplained_mm` (default: 3x the map's
    median |dev|, at least 0.3 mm) and NO placed feature's footprint (grown 1.5 mm) reaches —
    "added" = scan material the tree lacks, "removed" = tree material the scan refutes."""
    from scipy import ndimage as ndi
    dev, valid = dm["dev"], dm["valid"]
    zs = dm["zs"]
    cell_w = 2 * np.pi * np.nanmedian(dm["R_scan"]) / dm["nth"]
    cell_a = cell_w * dm["dz"]
    owned = np.zeros(dev.shape, bool)
    rows = []

    def record(name, kind, mask, extra=None):
        m = mask & valid
        if m.sum() < 3:
            return
        d = dev[m]
        st = _stats(d)
        rec = {"name": name, "kind": kind, "area_mm2": round(float(m.sum() * cell_a), 1),
               "error_mm3": round(float(np.abs(d).sum() * cell_a), 2),
               "median_mm": round(float(np.median(d)), 3), **st,
               "leans": "proud" if np.median(d) > 0.05 else "low" if np.median(d) < -0.05 else "even"}
        if extra:
            rec.update(extra)
        rows.append(rec)

    for f, src in _lowered(spec):
        m = footprint(dm, f, grow=1.0)
        owned |= m
        record(f["name"], f["kind"], m, {"source": src})
    for pad, s, a, b in _levels(spec):
        band = (zs >= a) & (zs < b)
        m = np.zeros(dev.shape, bool)
        m[band] = True
        record(pad["name"], "level", m & ~owned, {"z": [round(a, 2), round(b, 2)]})
    rows.sort(key=lambda r: -r["error_mm3"])

    thr = unexplained_mm or max(0.3, 3 * float(np.nanmedian(np.abs(dev[valid]))))
    reach = np.zeros(dev.shape, bool)
    for f, _ in _lowered(spec):
        reach |= footprint(dm, f, grow=1.5)
    loose = []
    for sign, kind in ((-1, "added"), (1, "removed")):
        m = valid & ~reach & (sign * np.nan_to_num(dev) > thr)
        lab, n = ndi.label(m, structure=np.ones((3, 3)))
        if n and lab[:, 0].any() and lab[:, -1].any():         # merge across the 0/360 seam
            for a_, b_ in zip(lab[:, 0], lab[:, -1]):
                if a_ and b_ and a_ != b_:
                    lab[lab == b_] = a_
        for k in np.unique(lab[lab > 0]):
            c = lab == k
            area = float(c.sum() * cell_a)
            if area < min_area_mm2:
                continue
            ri, ci = np.where(c)
            ths = dm["th_deg"][ci]
            mid = math.degrees(math.atan2(np.sin(np.radians(ths)).mean(), np.cos(np.radians(ths)).mean())) % 360
            span = float(np.ptp(((ths - mid + 180) % 360) - 180)) + 360.0 / dm["nth"]
            loose.append({"kind": kind, "z": [round(float(zs[ri.min()]), 2), round(float(zs[ri.max()]), 2)],
                          "theta_deg": [round(mid, 1), round(span, 1)], "area_mm2": round(area, 1),
                          "peak_mm": round(float(np.abs(dev[c]).max()), 3),
                          "volume_mm3": round(float(np.abs(dev[c]).sum() * cell_a), 2)})
    loose.sort(key=lambda r: -r["volume_mm3"])
    return {"summary": dm["stats"], "features": rows, "unexplained": loose,
            "unexplained_threshold_mm": round(thr, 3)}


# ----------------------------------------------------------------------------------- render
def render(dm, path, spec=None, limit=None, label=8, title=None):
    """The signed map as a PNG: z along the bottom, theta up the side, red = tree proud, blue
    = tree low, symmetric about 0. With `spec`, the `label` worst placed features are outlined
    and named, so the picture reads as a work list."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:                                    # pragma: no cover
        raise RuntimeError("render() needs matplotlib (pip install matplotlib)") from e
    dev = dm["dev"]
    lim = limit or max(0.1, round(2 * float(np.nanpercentile(np.abs(dev), 90)), 2))
    zs = dm["zs"]
    fig, ax = plt.subplots(figsize=(max(10, min(40, (zs[-1] - zs[0]) / 5)), 7))
    im = ax.imshow(dev.T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   extent=[zs[0] - dm["dz"] / 2, zs[-1] + dm["dz"] / 2, 0, 360], interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.02)
    cb.set_label("CAD - scan (mm):  red = tree proud,  blue = tree low")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("theta (deg)")
    ax.set_yticks(range(0, 361, 45))
    st = dm["stats"]
    ax.set_title(title or f"signed deviation  |dev| median {st.get('median_abs_mm')} mm, "
                          f"p90 {st.get('p90_abs_mm')} mm")
    if spec is not None:
        att = attribute(dm, spec)
        worst = [r["name"] for r in att["features"] if r["kind"] in PLACED][:label]
        for f, _ in _lowered(spec):
            if f["name"] not in worst:
                continue
            w = _world(f, _outline(f))
            th = (np.degrees(np.arctan2(w[:, 1] - dm["axis"][1], w[:, 0] - dm["axis"][0])) % 360)
            # break the outline where it wraps so a seam-crossing feature is drawn as two pieces
            jump = np.where(np.abs(np.diff(np.r_[th, th[0]])) > 180)[0]
            zz, tt = np.r_[w[:, 2], w[0, 2]], np.r_[th, th[0]]
            for seg in np.split(np.arange(len(zz)), jump + 1):
                ax.plot(zz[seg], tt[seg], "k-", lw=0.7)
            ax.text(w[:, 2].mean(), float(np.median(th)), f["name"], fontsize=7, ha="center",
                    va="center", bbox=dict(fc="white", alpha=0.6, lw=0, pad=1))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------- roll / spin
def square_roll(scan, dz=0.5, nth=720):
    """The spin about Z that squares a scan to its own mirror symmetry.

    Each row's radius residual (radius minus the row median) is compared with its reflection
    about every candidate plane theta = phi (in 0.25 degree steps when nth = 720); the plane
    with the least mean |asymmetry| wins. Spinning the scan by `spin_deg` puts that plane on
    theta = 0 (so its partner, if the part has two, lands on 90). `confidence` is how much
    better the best plane is than a typical one: near 0 the part has no mirror to square to.

    Returns {"mirror_deg", "spin_deg", "asymmetry_mm", "typical_mm", "confidence",
    "second_mirror_asymmetry_mm"}."""
    import mesh_compare as mc
    s = mc.displacement(scan, dz=dz, nth=nth, harmonics=0)
    R, valid = s["R"], s["valid"]
    res = np.where(valid, R - np.nanmedian(np.where(valid, R, np.nan), axis=1, keepdims=True), np.nan)
    keep = np.isfinite(res).mean(axis=1) > 0.9               # rows the scan covers all round
    res = res[keep]
    j = np.arange(nth)

    def asym(k):                         # reflection about phi = k * 180/nth degrees
        mirrored = res[:, (k - j - 1) % nth]
        d = np.abs(res - mirrored)
        return float(np.nanmean(d))

    scores = np.array([asym(k) for k in range(nth)])
    k = int(np.argmin(scores))
    phi = k * 180.0 / nth
    spin = -phi
    spin = (spin + 90.0) % 180.0 - 90.0                      # the smaller of the two equivalent rolls
    typ = float(np.median(scores))
    k2 = (k + nth // 2) % nth
    return {"mirror_deg": round(phi, 2), "spin_deg": round(spin, 2),
            "asymmetry_mm": round(float(scores[k]), 4), "typical_mm": round(typ, 4),
            "confidence": round(1.0 - float(scores[k]) / typ, 3) if typ > 0 else 0.0,
            "second_mirror_asymmetry_mm": round(float(scores[k2]), 4)}


def spun(mesh, deg):
    """A copy of `mesh` spun `deg` degrees about Z."""
    import trimesh
    m = mesh.copy()
    if deg:
        m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(deg), [0, 0, 1]))
    return m


# ---------------------------------------------------------------------------- the estimators
def _edge_evidence(dm, f, w=None, step=0.75):
    """Boundary samples of a placed feature: for each point p on its outline (u, v) with
    outward normal n, delta = (integral of dev across the edge, s in [-w, w]) / (the tree's own
    radius step across that band). A cut whose edge sits `e` mm inside the scan's leaves a
    strip of width e where the tree is proud by the step D: the integral is e*D, so delta = +e
    (grow). A boss `e` too wide leaves the same strip proud with a NEGATIVE step: delta = -e
    (shrink). One formula, both signs, no feature-kind special case."""
    uv = _outline(f)
    tap = abs(math.tan(math.radians(f.get("taper", 0.0))))
    w = w or min(4.0, 2.0 + 1.5 * tap)
    d = np.diff(np.vstack([uv, uv[:1]]), axis=0)
    t = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    t = (t + np.roll(t, 1, axis=0)) / 2                       # vertex tangents
    n = np.c_[t[:, 1], -t[:, 0]]
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    s = np.arange(-w, w + 1e-9, 0.25)
    P = uv[:, None, :] + s[None, :, None] * n[:, None, :]     # (points, offsets, 2)
    W = _world(f, P.reshape(-1, 2))
    dev = _lookup(dm, "dev", W).reshape(len(uv), len(s))
    rc = _lookup(dm, "R_cad", W).reshape(len(uv), len(s))
    # the floor's own error (inside) and the surrounding skin's (outside) are the floor and
    # level estimators' business: take them out, leaving only the strip the EDGE is to blame for
    ins, outs = s < -w + 0.75, s > w - 0.75
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        e_in = np.nanmedian(dev[:, ins], axis=1)[:, None]
        e_out = np.nanmedian(dev[:, outs], axis=1)[:, None]
    integ = np.nansum(dev - np.where(s[None, :] < 0, e_in, e_out), axis=1) * 0.25
    full = np.isfinite(dev).mean(axis=1) > 0.8
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # off-map samples: all-NaN rows
        step_h = np.nanmedian(rc[:, outs], axis=1) - np.nanmedian(rc[:, ins], axis=1)
    ok = full & np.isfinite(step_h) & (np.abs(step_h) > 0.15)
    delta = np.where(ok, integ / np.where(ok, step_h, 1.0), np.nan)
    return uv, n, delta


def _fit_outline(uv, n, delta, max_shift=1.5, max_grow=1.5):
    """Least-absolute fit of delta_i = n_i . (t + S (p_i - c)), S = diag(su, sv): a shift and an
    anisotropic stretch of the outline about its centroid. Returns (c, t, (su, sv)) or None."""
    ok = np.isfinite(delta)
    if ok.sum() < 8:
        return None
    p, nn, d = uv[ok], n[ok], delta[ok]
    c = uv.mean(axis=0)
    A = np.c_[nn[:, 0], nn[:, 1], nn[:, 0] * (p[:, 0] - c[0]), nn[:, 1] * (p[:, 1] - c[1])]
    half = np.maximum(np.ptp(uv, axis=0) / 2, 1.0)
    x = np.zeros(4)
    for _ in range(10):                                        # IRLS for L1: features nearby vote less
        r = d - A @ x
        wgt = 1.0 / np.maximum(np.abs(r), 0.1)
        lam = 1e-3 * np.diag([1, 1, half[0] ** 2, half[1] ** 2])
        Aw = A * wgt[:, None]
        x = np.linalg.solve(Aw.T @ A + lam, Aw.T @ d)
    t = np.clip(x[:2], -max_shift, max_shift)
    sg = np.clip(x[2:], -max_grow / half, max_grow / half)
    return c, t, sg


def _floor_offset(dm, f, w=2.0):
    """Median deviation on the feature's floor (its footprint shrunk by `w` mm), or None."""
    from scipy import ndimage as ndi
    m = footprint(dm, f)
    cell_w = 2 * np.pi * _r_ref(dm, f) / dm["nth"]
    iz, it = max(1, int(round(w / dm["dz"]))), max(1, int(round(w / cell_w)))
    inner = ndi.binary_erosion(m, structure=np.ones((2 * iz + 1, 2 * it + 1), bool)) & dm["valid"]
    if inner.sum() < 20:
        return None
    return float(np.median(dm["dev"][inner]))


def _transform_polys(polys, c, t, sg):
    out = []
    for wire in polys:
        nw = []
        for p in wire:
            u = c[0] + (1 + sg[0]) * (p[0] - c[0]) + t[0]
            v = c[1] + (1 + sg[1]) * (p[1] - c[1]) + t[1]
            nw.append([round(float(u), 4), round(float(v), 4)] + list(p[2:]))
        out.append(nw)
    return out


def _propose_placed(dm, spec, damping):
    """Per placed source feature, the corrected feature (or nothing). Pattern instances pool
    their evidence onto the master: every instance shares the master's (u, v) frame."""
    by_src = {}
    for f, src in _lowered(spec):
        if src:
            by_src.setdefault(src, []).append(f)
    props = {}
    for src, insts in by_src.items():
        master = next(g for g in spec["features"] if g["name"] == src)
        U, N, Dl, floors = [], [], [], []
        for f in insts:
            uv, n, d = _edge_evidence(dm, f)
            U.append(uv), N.append(n), Dl.append(d)
            fo = _floor_offset(dm, f)
            if fo is not None:
                floors.append(fo)
        fit = _fit_outline(np.vstack(U), np.vstack(N), np.concatenate(Dl))
        g = copy.deepcopy(master)
        change = {}
        if fit is not None:
            c, t, sg = fit
            t, sg = t * damping, sg * damping
            ext = np.ptp(np.vstack(U), axis=0) / 2
            if np.abs(t).max() > 0.05 or np.abs(sg * ext).max() > 0.05:
                g["polys"] = _transform_polys(master["polys"], c, t, sg)
                change.update(shift_uv_mm=[round(float(v), 3) for v in t],
                              grow_uv_mm=[round(float(v), 3) for v in sg * ext * 2])
        if floors:
            m = float(np.median(floors)) * damping
            if abs(m) > 0.03:
                _, nrm, _, _ = _frame(master)
                if master["kind"] == "prism_cut":   # floor = origin plane; proud floor -> go deeper
                    g["origin"] = [round(float(v), 6) for v in np.asarray(master["origin"]) - m * nrm]
                    g["depth"] = round(float(master["depth"]) + m, 4)
                else:                               # a boss's top: proud -> shorter
                    g["depth"] = round(max(0.05, float(master["depth"]) - m), 4)
                change["floor_mm"] = round(-m, 3)
        if change:
            props[src] = (g, change)
    return props


def _propose_levels(dm, spec, damping):
    """Per base level, its outline following the deviation of the bare body around it."""
    excl = np.zeros(dm["dev"].shape, bool)
    for f, _ in _lowered(spec):
        excl |= footprint(dm, f, grow=1.5)
    base = dm["valid"] & ~excl
    ax, zs, nth = dm["axis"], dm["zs"], dm["nth"]
    props, skipped = {}, []
    for pad, s, a, b in _levels(spec):
        if pad.get("taper"):
            skipped.append(pad["name"])
            continue
        rows = (zs >= a + 0.75) & (zs < b - 0.75)
        if rows.sum() < 2:
            continue
        g = copy.deepcopy(s)
        change = {}
        if g.get("polys") and not g.get("circles"):
            wire = np.array([p[:2] for p in g["polys"][0]], dtype=float)
            th = np.degrees(np.arctan2(wire[:, 1] - ax[1], wire[:, 0] - ax[0])) % 360
            half = 180.0 / max(len(wire), 1)
            dk = np.full(len(wire), np.nan)
            for i, t in enumerate(th):
                cols = np.abs(((dm["th_deg"] - t + 180) % 360) - 180) <= max(half, 360.0 / nth)
                sel = base[np.ix_(rows, cols)]
                if sel.sum() >= 5:
                    dk[i] = np.median(dm["dev"][np.ix_(rows, cols)][sel])
            if np.isfinite(dk).sum() < len(dk) // 2:
                continue
            dk = np.where(np.isfinite(dk), dk, np.nanmedian(dk))
            dk = np.median(np.c_[np.roll(dk, 1), dk, np.roll(dk, -1)], axis=1)   # despeckle
            dk *= damping
            if np.abs(dk).max() < 0.03:
                continue
            rel = wire - ax
            r = np.hypot(rel[:, 0], rel[:, 1])
            k = (r - dk) / np.maximum(r, 1e-9)
            nw = ax + rel * k[:, None]
            g["polys"] = [[[round(float(x), 4), round(float(y), 4)] + list(p[2:])
                           for (x, y), p in zip(nw, g["polys"][0])]] + g["polys"][1:]
            change["radius_mm"] = [round(float(-dk.max()), 3), round(float(-dk.min()), 3)]
        elif g.get("circles") and len(g["circles"]) == 1 and not g.get("polys"):
            sel = base[rows]
            if sel.sum() < 20:
                continue
            m = float(np.median(dm["dev"][rows][sel])) * damping
            if abs(m) < 0.03:
                continue
            cx, cy, r = g["circles"][0]
            g["circles"] = [[cx, cy, round(float(r) - m, 4)]]
            change["radius_mm"] = round(-m, 3)
        if change:
            props[s["name"]] = (g, dict(change, level=pad["name"]))
    return props, skipped


def _apply(spec, props):
    out = copy.deepcopy(spec)
    for i, f in enumerate(out["features"]):
        if f["name"] in props:
            out["features"][i] = copy.deepcopy(props[f["name"]][0])
    return out


def _region_score(dm, masks):
    m = masks & dm["valid"]
    return float(np.nanmean(np.abs(dm["dev"][m]))) if m.sum() else float("nan")


def _regions(dm, spec, names):
    """Per edited name, the map region it answers for (old and new footprints are unioned by
    the caller): placed sources -> all their instances' footprints grown 2 mm; level sketches
    -> their z band."""
    out = {}
    lows = _lowered(spec)
    levels = {s["name"]: (a, b) for pad, s, a, b in _levels(spec)}
    for name in names:
        m = np.zeros(dm["dev"].shape, bool)
        if name in levels:
            a, b = levels[name]
            m[(dm["zs"] >= a) & (dm["zs"] < b)] = True
        for f, src in lows:
            if src == name:
                m |= footprint(dm, f, grow=2.0)
        out[name] = m
    return out


def refine(spec, scan, iters=3, damping=0.7, dz=0.25, nth=720, spin=0.0, align="frame",
           levels=True, placed=True, min_gain=0.005, log=print):
    """Iterate map -> corrections -> rebuild -> keep what helped.

    Each iteration proposes, from the current tree's map: for every placed source feature a
    floor offset and an outline shift/stretch (patterns pooled onto their master), and for
    every untapered base level a per-angle radius correction — all scaled by `damping`. The
    candidate tree is rebuilt and mapped; every edit is scored on its own region (its old and
    new footprints, or its level's band) and reverted if that region got worse. Stops after
    `iters`, or when the map's median |dev| improves by less than `min_gain` mm.

    Returns (refined_spec, report). The report lists every iteration's map statistics and every
    kept / reverted edit, so what refinement did is as inspectable as what recognition did."""
    s_map = None
    import mesh_compare as mc
    s_map = mc.displacement(scan, dz=dz, nth=nth, harmonics=0)
    cur = copy.deepcopy(spec)
    dm = signed_map(scan, cur, spin=spin, align=align, scan_map=s_map)
    report = {"start": dm["stats"], "iterations": [], "not_refined": []}
    log(f"refine: start  |dev| median {dm['stats']['median_abs_mm']} p90 {dm['stats']['p90_abs_mm']} mm")
    for it in range(iters):
        props = _propose_placed(dm, cur, damping) if placed else {}
        if levels:
            lp, skipped = _propose_levels(dm, cur, damping)
            props.update(lp)
            if it == 0 and skipped:
                report["not_refined"].append({"tapered levels (cone steps)": skipped})
        if not props:
            log("refine: nothing left to correct")
            break
        cand = _apply(cur, props)
        cm = signed_map(scan, cand, spin=spin, align=align, scan_map=s_map)
        before = _regions(dm, cur, props)
        after = _regions(cm, cand, props)
        kept, reverted = {}, {}
        for name, (g, change) in props.items():
            reg = before[name] | after[name]
            b, a = _region_score(dm, reg), _region_score(cm, reg)
            rec = dict(change, before_mm=round(b, 4), after_mm=round(a, 4))
            (kept if (np.isfinite(a) and a <= b) else reverted)[name] = (g, rec)
        if reverted:
            cand = _apply(cur, kept)
            cm = signed_map(scan, cand, spin=spin, align=align, scan_map=s_map) if kept else dm
        gain = dm["stats"]["median_abs_mm"] - cm["stats"]["median_abs_mm"]
        report["iterations"].append({"iteration": it + 1, "stats": cm["stats"],
                                     "kept": {k: v[1] for k, v in kept.items()},
                                     "reverted": {k: v[1] for k, v in reverted.items()}})
        log(f"refine: iter {it + 1}  |dev| median {cm['stats']['median_abs_mm']} "
            f"p90 {cm['stats']['p90_abs_mm']} mm  kept {len(kept)}, reverted {len(reverted)}")
        if not kept:
            break
        cur, dm = cand, cm
        if gain < min_gain:
            break
    engr = sorted({f["name"] for f in spec["features"] if f["kind"] == "engrave"})
    if engr:
        report["not_refined"].append({"engravings": engr})
    report["end"] = dm["stats"]
    return cur, report


# ------------------------------------------------------------------------------------- CLI
def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items() if k not in ("cad_mesh", "scan_map")}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main():
    import argparse

    import mesh_compare as mc
    ap = argparse.ArgumentParser(description="Signed deviation map of a feature tree against its scan; "
                                             "attribute it, render it, refine the tree from it.")
    ap.add_argument("scan")
    ap.add_argument("--spec", help="IR JSON of the tree to map against the scan")
    ap.add_argument("--spin", type=float, default=0.0, help="spin the tree about Z first (degrees)")
    ap.add_argument("--align", default="frame", choices=["frame", "bbox"])
    ap.add_argument("--dz", type=float, default=0.25)
    ap.add_argument("--png", help="render the signed map here")
    ap.add_argument("--report", help="write the attribution (and any refine log) as JSON")
    ap.add_argument("--refine", metavar="OUT", help="refine the tree from the map and write it here")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--damping", type=float, default=0.7)
    ap.add_argument("--square", action="store_true", help="report the roll that squares the scan")
    a = ap.parse_args()
    scan = mc.load(a.scan)
    out = {}
    if a.square:
        sq = square_roll(scan)
        out["square"] = sq
        print(f"square: mirror plane at {sq['mirror_deg']} deg -> spin the scan {sq['spin_deg']:+} deg "
              f"(asymmetry {sq['asymmetry_mm']} mm vs typical {sq['typical_mm']} mm, "
              f"confidence {sq['confidence']})")
    if a.spec:
        with open(a.spec) as fh:
            spec = json.load(fh)
        spec = spec.get("spec", spec)
        if a.refine:
            spec, rep = refine(spec, scan, iters=a.iters, damping=a.damping, dz=a.dz,
                               spin=a.spin, align=a.align)
            out["refine"] = rep
            with open(a.refine, "w") as fh:
                json.dump(spec, fh, indent=1)
            print(f"refined tree -> {a.refine}")
        dm = signed_map(scan, spec, dz=a.dz, spin=a.spin, align=a.align)
        att = attribute(dm, spec)
        out["attribution"] = att
        st = att["summary"]
        print(f"map: |dev| median {st['median_abs_mm']} p90 {st['p90_abs_mm']} p99 {st['p99_abs_mm']} mm, "
              f"tree proud on {st['proud_pct']}% / low on {st['low_pct']}% of the skin (> 0.1 mm)")
        print("worst features (error = integral of |dev|):")
        for r in att["features"][:10]:
            print(f"  {r['name']:<24} {r['kind']:<10} error {r['error_mm3']:>8} mm3  "
                  f"median {r['median_mm']:+.3f} mm  p90|dev| {r['p90_abs_mm']} mm  ({r['leans']})")
        if att["unexplained"]:
            print(f"unexplained (|dev| > {att['unexplained_threshold_mm']} mm, no feature owns it):")
            for r in att["unexplained"][:10]:
                print(f"  {r['kind']:<8} z {r['z'][0]}..{r['z'][1]}  theta {r['theta_deg'][0]} "
                      f"({r['theta_deg'][1]} wide)  peak {r['peak_mm']} mm  {r['volume_mm3']} mm3")
        if a.png:
            render(dm, a.png, spec)
            print(f"map -> {a.png}")
    if a.report:
        with open(a.report, "w") as fh:
            json.dump(_json_safe(out), fh, indent=1)


if __name__ == "__main__":
    main()
