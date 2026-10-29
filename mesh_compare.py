"""mesh_compare.py — measure a scanned mesh, and score a feature tree against it.

A scan (STL) is real geometry at real scale, which a picture is not. It is still not a feature
tree, and its edges are soft, so this does three modest things by slicing the mesh across its Z axis:

  measure   the stepped outline (each level's diameter and height), the holes at each level
            (round, or a regular polygon with its across-flats size), and how far each rim is
            broken (chamfer / round), with the slice spacing as the resolution; --relief also
            maps the body's off-axis relief into classified patches (insets, panels, ribs);
  tree      for round stepped parts with through holes, a feature tree built from those
            measurements: one size per recognised shape, so a hexagon comes out regular; rim
            breaks that measured as a 45-degree chamfer or a quarter round become chamfer /
            fillet features, and a blank scan band between shrinking levels becomes a cone;
  compare   a volumetric overlap (IoU) between the mesh and an IR spec's solid, summed over
            slices, after centring both and searching the spin about Z.

It assumes the part's main axis is the mesh's Z axis and does not register anything else.

    python3 mesh_compare.py scan.stl                       # measure
    python3 mesh_compare.py scan.stl --spec tree.json      # ... and overlap with a tree
    python3 mesh_compare.py scan.stl --spec tree.json --scale   # scale the tree to the scan first
    python3 mesh_compare.py scan.stl --tree out.json       # a tree from the measurements, then its overlap
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load(path):
    import trimesh
    return trimesh.load(str(path), force="mesh")


def slices(mesh, zs, min_area=1.0):
    """The mesh's cross-section at each z as one shapely geometry (None where it misses). Scan
    crumbs smaller than `min_area` are dropped."""
    from shapely.ops import unary_union
    out = []
    for sec in mesh.section_multiplane([0, 0, 0], [0, 0, 1], list(zs)):
        polys = [p.buffer(0) for p in sec.polygons_full if p.area >= min_area] if sec is not None else []
        out.append(unary_union(polys) if polys else None)
    return out


def _largest(g):
    return max(g.geoms, key=lambda p: p.area) if g.geom_type == "MultiPolygon" else g


def _ring_points(ring, n=720):
    """`n` points evenly spaced along a ring: a CAD mesh's hexagon has six vertices, a scan's has
    thousands, and the measurements below must not depend on which."""
    return np.array([ring.interpolate(t, normalized=True).coords[0] for t in np.linspace(0, 1, n, endpoint=False)])


def _flats(pts, centre, n, corner_deg, flat_tol=0.3, angle_tol=1.0):
    """Fit a line to the middle two thirds of each side of an n-gon. Returns each side's distance
    from the centre, the across-flats distances (opposite pairs, even n), how far the sides are from
    evenly turned, and `regular`: whether one size and one angle describe it within the tolerances
    (mm between the largest and smallest distance; degrees off 360 / n)."""
    cx, cy = centre
    ang = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    dist, dirs = [], []
    for k in range(n):
        mid = corner_deg + (k + 0.5) * 360.0 / n
        p = pts[np.abs((ang - mid + 180) % 360 - 180) < 120.0 / n]
        q = p - p.mean(axis=0)
        nrm = np.linalg.svd(q, full_matrices=False)[2][1]
        if nrm @ (p.mean(axis=0) - (cx, cy)) < 0:
            nrm = -nrm
        dist.append(float(nrm @ (p.mean(axis=0) - (cx, cy))))
        dirs.append(math.degrees(math.atan2(nrm[1], nrm[0])))
    turn = [(dirs[(k + 1) % n] - dirs[k]) % 360 - 360.0 / n for k in range(n)]
    pairs = [dist[k] + dist[k + n // 2] for k in range(n // 2)] if n % 2 == 0 else [2 * d for d in dist]
    return {"across_flats": round(float(np.mean(pairs)), 3), "flat_pairs": [round(p, 3) for p in pairs],
            "flat_spread": round(max(pairs) - min(pairs), 3), "angle_error": round(max(abs(t) for t in turn), 2),
            "regular": bool(max(pairs) - min(pairs) <= flat_tol and max(abs(t) for t in turn) <= angle_tol)}


def _hole(ring):
    """A hole's centre, size and shape. A regular n-gon is recognised from its corners: n evenly
    spaced maxima of the radius, with min / max radius close to cos(pi / n)."""
    from shapely.geometry import Polygon
    p = Polygon(ring)
    cx, cy = p.centroid.coords[0]
    c = _ring_points(ring)
    r = np.hypot(c[:, 0] - cx, c[:, 1] - cy)
    out = {"centre": [round(cx, 2), round(cy, 2)], "diameter": round(2 * math.sqrt(p.area / math.pi), 2),
           "r_min": round(float(r.min()), 2), "r_max": round(float(r.max()), 2), "shape": "round"}
    if r.max() - r.min() > 0.06 * r.max():
        ang = np.arctan2(c[:, 1] - cy, c[:, 0] - cx)
        for n in (6, 4, 3, 8, 5):                       # a corner every 2 pi / n: fit its phase
            z = np.sum((r - r.mean()) * np.exp(1j * n * ang))
            fit = abs(z) / (np.sum(np.abs(r - r.mean())) + 1e-12)
            if fit > 0.6 and abs(r.min() / r.max() - math.cos(math.pi / n)) < 0.04:
                corner = math.degrees(np.angle(z)) / n % (360 / n)
                out.update(_flats(c, (cx, cy), n, corner), sides=n,
                           across_corners=round(2 * float(r.max()), 2), first_corner_deg=round(corner, 1))
                out["ideal_across_corners"] = round(out["across_flats"] / math.cos(math.pi / n), 2)
                out["shape"] = f"regular {n}-gon" if out["regular"] else f"{n}-gon, not regular"
                break
        else:
            out["shape"] = "irregular"
    return out


def _radius(sec):
    """Median distance of the outline from its centroid."""
    ext = _largest(sec).exterior
    c = _ring_points(ext, 360)
    cx, cy = ext.centroid.coords[0]
    return float(np.median(np.hypot(c[:, 0] - cx, c[:, 1] - cy)))


def _rim_break(mesh, z_end, inward, body, reach=2.0, step=0.05, tol=0.1):
    """How a level's outline falls away as it nears its end face at `z_end` (`inward` = +1 if the
    level lies above that face, -1 below): (height over which it is more than `tol` short of the
    body radius, radius lost at the last whole slice, shape, size). (0, 0, None, None) is a sharp
    edge at this resolution.

    The shape is read by fitting both profiles a designer would have asked for: each sample `h`
    in from the face and `d` short of the body gives one estimate of a 45-degree chamfer's size
    (h + d) and one of a quarter round's radius (h + d + sqrt(2 h d)); whichever family of
    estimates agrees on one value names the break — "chamfer" or "round", with `size` that value
    — and when neither agrees (a taper, a torn rim) both are None."""
    zs = z_end + inward * np.arange(step, reach, step)
    short = [(abs(z - z_end), body - _radius(s)) for z, s in zip(zs, slices(mesh, zs))
             if s is not None and abs(_radius(s) - body) < 3.0]
    broken = [(h, d) for h, d in short if d > tol]
    if not broken:
        return 0.0, 0.0, None, None
    hs, ds = np.array([h for h, _ in broken]), np.array([d for _, d in broken])
    shape = size = best = None
    if len(broken) >= 4:
        for name, est in (("chamfer", hs + ds), ("round", hs + ds + np.sqrt(2 * hs * ds))):
            med = float(np.median(est))
            spread = float(np.percentile(est, 90) - np.percentile(est, 10))
            if med > 0 and spread <= max(0.1, 0.2 * med) and (best is None or spread < best):
                shape, size, best = name, round(med, 2), spread
    return round(float(hs.max()), 2), round(float(ds.max()), 2), shape, size


def _holes(mesh, z_lo, z_hi):
    """Holes of one level, seen at three heights and merged by position. A scanner sees a hole's
    mouth better than its bore, so each hole reports its roundest sighting."""
    seen = []
    for sec in slices(mesh, [z_lo + 0.1 * (z_hi - z_lo), (z_lo + z_hi) / 2, z_hi - 0.1 * (z_hi - z_lo)]):
        for ring in (_largest(sec).interiors if sec is not None else []):
            h = _hole(ring)
            if h["diameter"] < 1.0:
                continue
            for group in seen:
                if math.dist(group[0]["centre"], h["centre"]) < max(2.0, 0.3 * h["diameter"]):
                    group.append(h)
                    break
            else:
                seen.append([h])
    best = [max(g, key=lambda h: ("sides" in h, h["r_min"] / h["r_max"])) for g in seen]
    return sorted(best, key=lambda h: -h["diameter"])


def _break_dict(brk):
    h, d, shape, size = brk
    return {"height": h, "radius_lost": d} | ({"shape": shape, "size": size} if shape else {})


def measure(mesh, dz=0.25, min_height=1.0):
    """Levels of the stepped outline, bottom to top, each with its holes and rim breaks (each
    break named "chamfer" or "round" with a fitted size when its profile reads as one). Slices
    are `dz` apart; rims are re-sliced every 0.05 mm."""
    z0, z1 = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
    zs = np.arange(z0 + dz / 2, z1, dz)
    rad = np.array([_radius(s) if s is not None else np.nan for s in slices(mesh, zs)])
    levels, i = [], 0
    while i < len(zs):
        j = i
        while j + 1 < len(zs) and abs(rad[j + 1] - rad[i]) < 3.0:      # same step of the outline
            j += 1
        lo, hi = float(zs[i] - dz / 2), float(zs[j] + dz / 2)
        if not np.isnan(rad[i]) and hi - lo >= min_height:             # thinner = a face being crossed
            body = float(np.median(rad[i:j + 1]))
            # an open scan can miss the level's exact midpoint; fall back to slices known to exist
            sec = next((s for zq in ((lo + hi) / 2, zs[(i + j) // 2], zs[i], zs[j])
                        for s in slices(mesh, [zq]) if s is not None), None)
            if sec is None:
                i = j + 1
                continue
            ext = _largest(sec).exterior
            ep = _ring_points(ext, 360)
            er = np.hypot(ep[:, 0] - ext.centroid.x, ep[:, 1] - ext.centroid.y)
            levels.append({"z": [round(lo, 2), round(hi, 2)], "height": round(hi - lo, 2),
                           "outer_diameter": round(2 * body, 2),
                           "centre": [round(ext.centroid.x, 2), round(ext.centroid.y, 2)],
                           "round": bool(er.max() - er.min() < 0.02 * er.max() + 0.2),
                           "bottom_break": _break_dict(_rim_break(mesh, lo, +1, body)),
                           "top_break": _break_dict(_rim_break(mesh, hi, -1, body)),
                           "holes": _holes(mesh, lo, hi)})
        i = j + 1
    return {"extents": [round(float(v), 2) for v in mesh.extents], "slice_spacing": dz, "levels": levels}


def _floor_outline(field, level, cols, zs, nth, rm, tol_mm=0.35):
    """The patch floor's measured boundary: the `level` contour of its |residual| field, as
    [[theta_deg, z], ...] simplified to `tol_mm` (measured in arc mm at the body radius `rm`).
    The columns are rolled so a seam-crossing patch is contiguous before tracing; a contour
    that degenerates (under 4 points once simplified) returns None and the caller keeps its
    rectangle."""
    from skimage import measure as skm
    from shapely.geometry import Polygon
    bins = np.unique(cols)
    gaps = np.diff(np.concatenate([bins, [bins[0] + nth]]))
    start = int(bins[(int(np.argmax(gaps)) + 1) % len(bins)])
    rolled = np.pad(np.roll(field, -start, axis=1), 1)
    best = None
    for c in skm.find_contours(rolled, level):
        if len(c) < 4 or np.any(c[0] != c[-1]):
            continue
        area = 0.5 * abs(np.dot(c[:-1, 0], c[1:, 1]) - np.dot(c[1:, 0], c[:-1, 1]))
        if best is None or area > best[0]:
            best = (area, c)
    if best is None:
        return None
    c = best[1]
    th = (c[:, 1] - 1 + start + 0.5) * 360.0 / nth          # grid col -> degrees
    z = np.interp(c[:, 0] - 1, np.arange(len(zs)), zs)      # grid row -> mm
    poly = Polygon(np.column_stack([np.radians(th) * rm, z]))  # simplify in arc mm
    poly = poly.buffer(0).simplify(tol_mm)
    if poly.is_empty or poly.geom_type != "Polygon" or len(poly.exterior.coords) < 5:
        return None
    pts = np.asarray(poly.exterior.coords)[:-1]
    return [[round(float(np.degrees(u / rm)), 2), round(float(v), 2)] for u, v in pts]


def _skeleton_path(skel, nth):
    """The skeleton's diameter as an ordered pixel path: BFS from anywhere to the farthest
    pixel, then BFS back — the longest geodesic through the line, branches ignored. The grid
    wraps in theta (columns)."""
    from collections import deque
    px = {(int(r), int(c)) for r, c in np.argwhere(skel)}

    def bfs(start):
        seen, far = {start: None}, start
        q = deque([start])
        while q:
            p = q.popleft()
            far = p
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    n = (p[0] + dr, (p[1] + dc) % nth)
                    if n in px and n not in seen:
                        seen[n] = p
                        q.append(n)
        return far, seen
    a, _ = bfs(next(iter(px)))
    b, parents = bfs(a)
    path, p = [], b
    while p is not None:
        path.append(p)
        p = parents[p]
    return path


def _detail_lines(res, zs, nth, base, tol, min_len=8.0, max_width=3.0):
    """LINE detailing in the residual — carved grooves and embossed ridges: each sign's
    well-past-noise mask, skeletonized; components that are narrow for their length (a line,
    not a panel) walked into an ordered (theta_deg, z) path with one width and depth each.
    Everything line-like but shorter than `min_len` is summarized, not pathed."""
    import scipy.ndimage as ndi
    from skimage.morphology import skeletonize
    rm = float(np.nanmedian(base))
    cw = rm * 2 * np.pi / nth
    ch = float(zs[1] - zs[0]) if len(zs) > 1 else 0.5
    out = {}
    for sign, key in ((-1, "carved"), (1, "embossed")):
        lines, short = [], {"count": 0, "length_mm": 0.0, "area_mm2": 0.0}
        lab, _ = ndi.label(sign * res > 2 * tol, structure=np.ones((3, 3), int))
        for i in range(lab.shape[0]):                  # one surface: join across theta = 0
            a, b = lab[i, 0], lab[i, -1]
            if a and b and a != b:
                lab[lab == b] = a
        for L in np.unique(lab)[1:]:
            sel = lab == L
            skel = skeletonize(sel)
            slen = float(skel.sum()) * (cw + ch) / 2
            area = float(sel.sum()) * cw * ch
            width = area / max(slen, 1e-6)
            if width > max_width or slen < 3 * width:
                continue                               # a panel or a blob, not a line
            if slen < min_len:
                short["count"] += 1
                short["length_mm"] += slen
                short["area_mm2"] += area
                continue
            path = _skeleton_path(skel, nth)
            pts = [((c + 0.5) * 360.0 / nth, float(zs[r])) for r, c in path]
            keep = pts[::max(1, len(pts) // max(int(slen / 2.0), 2))]     # ~a point every 2 mm
            if keep[-1] != pts[-1]:
                keep.append(pts[-1])
            rows = [r for r, _ in path]
            lines.append({"path": [[round(t, 2), round(z, 2)] for t, z in keep],
                          "width": round(width, 2),
                          "depth": round(float(np.median(np.abs(res[sel]))), 2),
                          "length_mm": round(slen, 1),
                          "radius": round(float(np.nanmedian(base[rows])), 2)})
        lines.sort(key=lambda e: -e["length_mm"])
        short = {k: round(v, 1) for k, v in short.items()}
        out[key] = {"lines": lines, "short": short}
    return out


def _washboard(res, zs, nth, base, dz):
    """Periodic grip texture — circumferential micro-ridges stacked along Z — read as THE
    PATTERN IT IS: per theta column, the longest band of sustained residual is autocorrelated
    along z; columns agreeing on a pitch cluster into sectors, and each sector folds all its
    repeats into one measured ridge (pitch, height, phase). One ridge plus one pattern then
    stands for the whole washboard."""
    from scipy import ndimage as ndi
    # the ridges are circumferential, so a few degrees of theta-smoothing averages them
    # IN PHASE: amplitude survives, scan noise drops by root-N, and the per-column
    # autocorrelation becomes the statistic it pretends to be
    res = ndi.uniform_filter1d(res, size=max(3, nth // 72), axis=1, mode="wrap")
    cols = {}
    for c in range(nth):
        v = res[:, c]
        act = np.convolve(np.abs(v), np.ones(9) / 9, mode="same") > 0.25
        runs, i = [], 0
        while i < len(act):
            if act[i]:
                j = i
                while j < len(act) and act[j]:
                    j += 1
                runs.append((i, j))
                i = j
            else:
                i += 1
        if not runs:
            continue
        i0, i1 = max(runs, key=lambda r: r[1] - r[0])
        if (i1 - i0) * dz < 12.0:
            continue
        x = v[i0:i1] - v[i0:i1].mean()
        denom = float(x @ x)
        if denom < 1e-9:
            continue
        best = None
        for lag in range(3, min(int(6.0 / dz) + 1, (i1 - i0) // 3)):
            a = float(x[:-lag] @ x[lag:]) / denom
            if best is None or a > best[1]:
                best = (lag, a)
        if best and best[1] > 0.3:
            cols[c] = (best[0], i0, i1)
    if not cols:
        return []
    sectors, run = [], []
    for c in list(range(nth)) + [None]:         # wrapped runs of agreeing columns
        ok = c is not None and (c in cols or (run and c - 1 in cols and (c + 1) % nth in cols))
        if ok:
            run.append(c)
            continue
        if run and len(run) >= nth // 45:       # at least 8 degrees wide
            lags = [cols[c][0] for c in run if c in cols]
            lag = sorted(lags)[len(lags) // 2]
            i0 = int(np.median([cols[c][1] for c in run if c in cols]))
            i1 = int(np.median([cols[c][2] for c in run if c in cols]))
            prof = np.nanmean(res[i0:i1, run], axis=1)
            n = (len(prof) // lag) * lag
            if n >= 2 * lag:
                fold = prof[:n].reshape(-1, lag).mean(axis=0)
                height = float(fold.max() - fold.min())
                if height >= 0.35:
                    centre = ((run[0] + (len(run) - 1) / 2) % nth + 0.5) * 360.0 / nth
                    sectors.append({
                        "theta_deg": [round(centre, 1), round(len(run) * 360.0 / nth, 1)],
                        "z": [round(float(zs[i0]), 2), round(float(zs[i1 - 1]), 2)],
                        "pitch": round(lag * dz, 2),
                        "ridge_width": round(max(float((fold > fold.max() / 2).sum()) * dz, dz), 2),
                        "height": round(height, 2),
                        "crest": round(float(fold.max()), 2),
                        "phase_z": round(float(zs[i0 + int(np.argmax(fold))]), 2),
                        "radius": round(float(np.nanmedian(base[i0:i1])), 2)})
        run = []
    return sectors


def relief(mesh, dz=0.5, nth=360, tol=0.3, min_area=20.0):
    """What the outline carries AROUND the axis that the stepped levels cannot say: the radius
    binned every 360/`nth` degrees at every `dz` of height, minus each slice's own median (the
    axisymmetric baseline), segmented where it leaves ±`tol` mm into patches of at least
    `min_area` mm². Each patch is classified by sign and angular coverage — a "recess" or "boss"
    (an inset switch well, a grip panel, a moulded pad) — with its z span, centre angle and
    angular span, depth (the deep decile, so a soft rim does not dilute a flat floor), mean body
    radius there, arc width, area, and `fill` (how much of its bounding box it covers: 1.0 is a
    clean rectangle). A patch also carries the traced `outline` of its floor (the 0.6 x depth
    contour, as [theta_deg, z] points) so tree() can cut the measured shape, not a rectangle. A full "ring_rib" / "ring_groove" is axisymmetric and so invisible to the
    residual; those are read off the baseline instead, as a bump or dip in base(z) that returns
    to the surrounding radius (which a step shoulder does not).
    Angles are about the mean slice centroid (`axis_xy`); `noise` is the median |residual|."""
    import scipy.ndimage as ndi
    z0, z1 = float(mesh.bounds[0][2]), float(mesh.bounds[1][2])
    zs = np.arange(z0 + dz, z1 - dz / 2, dz)
    secs = slices(mesh, zs)
    def _core_centre(s):
        # the CORE circle's centre, not the outline's: a ring centroid is length-weighted, and
        # one-sided texture (a sector of ridges) drags it by more than a millimetre, which then
        # reads as a one-cycle wave across the whole residual field. Trim the proud points and
        # circle-fit what remains, twice.
        pts = np.asarray(_largest(s).exterior.segmentize(0.3).coords)
        c = pts.mean(axis=0)
        for _ in range(2):
            r = np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1])
            keep = pts[r <= np.median(r) + 0.5]
            if len(keep) < 8:
                break
            A = np.c_[keep[:, 0], keep[:, 1], np.ones(len(keep))]
            x, *_ = np.linalg.lstsq(A, (keep ** 2).sum(axis=1), rcond=None)
            c = np.array([x[0] / 2, x[1] / 2])
        return c
    ax = np.median([_core_centre(s) for s in secs if s is not None], axis=0)
    R = np.full((len(zs), nth), np.nan)
    for i, s in enumerate(secs):
        if s is None:
            continue
        ext = _largest(s).exterior
        pts = np.asarray(ext.segmentize(max(ext.length / (2 * nth), 0.05)).coords)
        th = np.arctan2(pts[:, 1] - ax[1], pts[:, 0] - ax[0])
        # floor, never truncate: astype(int) rounds toward zero, which makes the bin at 180
        # degrees a measure-zero sliver that almost never receives a point — and an always-empty
        # column silently split every patch, label and line that crossed the 180 meridian
        b = np.floor(th / (2 * np.pi) * nth).astype(int) % nth
        r = np.hypot(pts[:, 0] - ax[0], pts[:, 1] - ax[1])
        sums, cnt = np.zeros(nth), np.zeros(nth)
        np.add.at(sums, b, r)
        np.add.at(cnt, b, 1)
        R[i, cnt > 0] = sums[cnt > 0] / cnt[cnt > 0]
    rows_ok = ~np.isnan(R).all(axis=1)
    base = np.full(len(zs), np.nan)
    base[rows_ok] = np.nanmedian(R[rows_ok], axis=1)
    res = R - base[:, None]
    valid = ~np.isnan(res)
    r0 = np.where(valid, res, 0.0)
    patches = []

    # a full ring rib / groove is axisymmetric, so it hides in the baseline itself, not in the
    # residual: a bump (or dip) in base(z) that RETURNS to the surrounding radius — a step does not
    idx = np.arange(len(zs))
    filled = np.interp(idx, idx[rows_ok], base[rows_ok]) if rows_ok.any() else base
    smooth = ndi.median_filter(filled, size=max(3, int(7 / dz) | 1), mode="nearest")
    dev = filled - smooth
    run = None
    for i in list(np.flatnonzero(np.abs(dev) > tol)) + [None]:
        if run and (i is None or i != run[-1] + 1 or np.sign(dev[i]) != np.sign(dev[run[0]])):
            i0, i1 = run[0], run[-1]
            before, after = filled[max(i0 - 1, 0)], filled[min(i1 + 1, len(zs) - 1)]
            rm = (before + after) / 2
            area = 2 * np.pi * rm * (i1 - i0 + 1) * dz
            if abs(before - after) <= 2 * tol and area >= min_area:     # it returns: not a shoulder
                d = float(np.percentile(np.abs(filled[i0:i1 + 1] - rm), 90))
                patches.append({"kind": "ring_rib" if dev[i0] > 0 else "ring_groove",
                                "z": [round(float(zs[i0] - dz / 2), 2), round(float(zs[i1] + dz / 2), 2)],
                                "theta_deg": [0.0, 360.0], "depth": round(d, 2),
                                "mean_radius": round(float(rm), 2),
                                "arc_mm": round(2 * np.pi * float(rm), 1),
                                "area_mm2": round(float(area), 1), "fill": 1.0})
            run = None
        if i is not None:
            run = (run + [i]) if run else [i]
    for kind, mask in (("recess", r0 < -tol), ("boss", r0 > tol)):
        lab, _ = ndi.label(mask, structure=np.ones((3, 3), int))
        for i in range(len(zs)):                       # one surface: join patches across theta = 0
            a, b2 = lab[i, 0], lab[i, -1]
            if a and b2 and a != b2:
                lab[lab == b2] = a
        def _ang(cols_):
            bins_ = np.unique(cols_)
            gaps_ = np.diff(np.concatenate([bins_, [bins_[0] + nth]]))
            span_ = nth - int(gaps_.max()) + 1
            start_ = bins_[(int(np.argmax(gaps_)) + 1) % len(bins_)]
            centre_ = ((start_ + (span_ - 1) / 2) % nth + 0.5) * 360.0 / nth
            return len(bins_), span_, centre_

        for L in np.unique(lab)[1:]:
            cells = np.argwhere(lab == L)
            rows, cols = cells[:, 0], cells[:, 1]
            rm = float(np.nanmean(base[np.unique(rows)]))
            area = len(cells) * rm * (2 * np.pi / nth) * dz
            if area < min_area:
                continue
            vals = res[lab == L]
            depth = float(abs(np.percentile(vals, 10 if kind == "recess" else 90)))
            nbins, span, centre = _ang(cols)
            fill = len(cells) / (span * (rows.max() - rows.min() + 1))
            deep = np.abs(vals) >= 0.6 * depth                 # the floor, past the walls' ramp
            _, fspan, fcentre = _ang(cols[deep])
            frows = rows[deep]
            outline = _floor_outline(np.where(lab == L, np.abs(r0), 0.0), 0.6 * depth,
                                     cols, zs, nth, rm)
            patches.append({"kind": ("ring_groove" if kind == "recess" else "ring_rib")
                            if nbins >= 0.97 * nth else kind,
                            "z": [round(float(zs[rows.min()] - dz / 2), 2),
                                  round(float(zs[rows.max()] + dz / 2), 2)],
                            "theta_deg": [round(float(centre), 1), round(span * 360.0 / nth, 1)],
                            "depth": round(depth, 2), "mean_radius": round(rm, 2),
                            "arc_mm": round(rm * span * 2 * np.pi / nth, 1),
                            "area_mm2": round(area, 1), "fill": round(float(fill), 2),
                            "floor": {"z": [round(float(zs[frows.min()] - dz / 2), 2),
                                            round(float(zs[frows.max()] + dz / 2), 2)],
                                      "theta_deg": [round(float(fcentre), 1),
                                                    round(fspan * 360.0 / nth, 1)]},
                            **({"outline": outline} if outline else {})})
    patches.sort(key=lambda p: -p["area_mm2"])
    return {"axis_xy": [round(float(ax[0]), 2), round(float(ax[1]), 2)], "dz": dz, "tol": tol,
            "noise": round(float(np.nanmedian(np.abs(res[valid]))), 3),
            # the axisymmetric CORE the texture stands on: tree() re-bases textured levels on
            # it, because a slice's outer ring reads the rib ENVELOPE, not the body
            "base_z": [round(float(z), 2) for z in zs[rows_ok]],
            "base_r": [round(float(r), 3) for r in base[rows_ok]],
            "detail": _detail_lines(r0, zs, nth, base, tol),
            "washboard": _washboard(r0, zs, nth, base, dz),
            "patches": patches}


def _snap(v, grid=0.5, within=0.3):
    """A scanned size as a designer would have typed it: the nearest `grid` multiple when the scan
    is within `within` of it, else the measurement to 0.01."""
    g = round(v / grid) * grid
    return round(g, 6) if abs(g - v) <= within else round(v, 2)


def _panel_outline_union(bands, ang, rm, zmid, fuse=2.0, tol_mm=0.35):
    """The panel floor (or a boss's plateau) as the scan measured it, UNWRAPPED: each band's
    traced outline laid out in (arc mm about the panel's centre angle, z), the texture gaps
    between bands closed morphologically (buffer out and back by `fuse`), simplified. One
    shapely Polygon in that cylinder space — or None (any band without an outline, or a
    degenerate union), which keeps the rectangle."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    polys = []
    for b in bands:
        o = b.get("outline")
        if not o or len(o) < 3:
            return None
        p = Polygon([(math.radians((th - ang + 180) % 360 - 180) * rm, z - zmid)
                     for th, z in o]).buffer(0)
        if p.is_empty:
            return None
        polys.append(p)
    u = unary_union(polys).buffer(fuse).buffer(-fuse).simplify(tol_mm)
    if u.geom_type == "MultiPolygon":
        u = max(u.geoms, key=lambda q: q.area)
    if u.is_empty or u.geom_type != "Polygon" or len(u.exterior.coords) < 5:
        return None
    # moulded corners: close then open by a radius the feature can afford, so the traced
    # outline keeps its shape but loses the die-cut corners a simplified polygon carries
    b = u.bounds
    rc = min(0.6, 0.2 * min(b[2] - b[0], b[3] - b[1]))
    if rc > 0.15:
        rounded = u.buffer(rc).buffer(-2 * rc).buffer(rc)
        if rounded.geom_type == "Polygon" and not rounded.is_empty \
                and rounded.area > 0.6 * u.area:
            u = rounded.simplify(tol_mm / 2)
    return u


def _chord_poly(upoly, rm, rf, mirror):
    """A cylinder-space polygon mapped onto its facet's tangent plane as chords at radius `rf`.
    `mirror` flips u for a plane whose xdir runs against theta (a boss's inward-facing frame).
    Micro-edges are dropped: clipping a strip near an outline vertex leaves slivers of a tenth
    of a millimetre, and the strict kernels refuse tools that carry them."""
    rf = max(rf, 1.0)
    pts = [((-1 if mirror else 1) * rf * math.sin(s / rm), v)
           for s, v in upoly.exterior.coords[:-1]]
    if sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1])) < 0:
        pts.reverse()                                        # the profile loop stays CCW
    kept = []
    for p in pts:
        if not kept or math.dist(p, kept[-1]) >= 0.25:
            kept.append(p)
    if len(kept) > 2 and math.dist(kept[0], kept[-1]) < 0.25:
        kept.pop()
    return [(round(x, 3), round(y, 3), 0) for x, y in (kept if len(kept) >= 3 else pts)]


def _facets(upoly, ang, rm, width_deg=30.0, overlap=1.0, reverse=False):
    """A wide cylinder-space polygon split into chord-sized facets: (centre angle, polygon
    recentred on it) per piece, each narrow enough for a flat tangent panel to stay honest.
    Adjacent strips overlap well past flushness so their booleans cross transversally."""
    from shapely import affinity
    from shapely.geometry import box
    s0, v0, s1, v1 = upoly.bounds
    W = rm * math.radians(width_deg)
    n = max(1, int(math.ceil((s1 - s0) / W)))
    out = []
    for j in range(n):
        clip = upoly.intersection(box(s0 + j * W - overlap, v0 - 1.0,
                                      s0 + (j + 1) * W + overlap, v1 + 1.0))
        for piece in (clip.geoms if clip.geom_type == "MultiPolygon" else [clip]):
            if piece.geom_type != "Polygon" or piece.area < 2.0 or len(piece.exterior.coords) < 4:
                continue
            smid = (piece.bounds[0] + piece.bounds[2]) / 2
            # neighbouring facets carry near-coincident walls through the overlap strips (same
            # source outline, and v maps straight to world Z), and a kernel merges anything
            # within its tolerance into the flush boolean it refuses. Alternating facets
            # breathe 0.25 mm in and out, so every such wall pair ends up a real half-millimetre apart
            # — honest geometry to the kernel, invisible against scan noise.
            b = piece.bounds
            eps = 0.25 if j % 2 == 0 else -0.25
            piece = affinity.scale(piece,
                                   xfact=1.0 + eps / max((b[2] - b[0]) / 2, 1.0),
                                   yfact=1.0 + eps / max((b[3] - b[1]) / 2, 1.0), origin="center")
            out.append((ang + math.degrees(smid / rm),
                        affinity.translate(piece, xoff=-smid)))
    # strip order matters to SolidCraft's checker, and CUTS and PADS mirror each other (found
    # empirically, one wrap at a time): a cut wrap must end on its first strip, a pad wrap must
    # begin with it — so cuts run reversed and pads forward
    return out[::-1] if reverse else out


def _core_levels(levels, rel, notes):
    """Levels re-based on the axisymmetric CORE (relief's baseline) where texture fattens them:
    a slice's outer ring through a ribbed grip reads the rib ENVELOPE, so the level pad built
    from it swallows most of every boss that should stand proud and buries the real surface
    well inside the solid. Where the envelope exceeds the baseline meaningfully, the level's
    diameter becomes the core's; the ribs then stand on it as the pads they are."""
    if not rel or not rel.get("base_z"):
        return levels
    bz, br = np.asarray(rel["base_z"]), np.asarray(rel["base_r"])
    tol = rel.get("tol", 0.3)
    out = []
    for lv in levels:
        z0, z1 = lv["z"]
        pad = 0.2 * (z1 - z0)
        m = (bz >= z0 + pad) & (bz <= z1 - pad) & ~np.isnan(br)
        lv = dict(lv)
        if m.sum() >= 3:
            core = float(np.median(br[m]))
            drop = lv["outer_diameter"] / 2 - core
            # the baseline median is steady (it averages whole slices), so even a fraction of
            # the residual tolerance is signal; untextured levels measure within ~0.1 of it
            if drop > max(tol / 2, 0.15):
                notes.append(f"level z {z0}..{z1}: textured — based on the core radius "
                             f"{round(core, 2)} (the outer envelope {lv['outer_diameter'] / 2} "
                             f"is carried by its ribs)")
                lv["outer_diameter"] = round(2 * core, 2)
        out.append(lv)
    return out


def tree(measured, snap=True):
    """A feature tree from measure()'s output, for parts this module can describe: round steps
    stacked along Z, with through holes that are round or regular polygons. Returns (spec, notes).

    Each recognised shape becomes the feature a designer would use, with one size: a step is a
    circle and a pad; a regular hexagon is an ngon with one across-flats value (so it is regular
    by construction, whatever the scan's six sides measure); equally spaced holes at one radius
    become one bolt circle with one diameter; a rim break whose profile measured as a 45-degree
    chamfer or a quarter round becomes that chamfer() / fillet(); a blank band in the scan between
    two levels with falling diameter is bridged by a tapered pad (a cone); and when `measured`
    carries a "relief" entry (see relief()), each rectangular-enough recess patch becomes a
    radial prism_cut — a flat-floored inset — while every other patch is a note. With `snap`,
    sizes within 0.3 mm of a 0.5 mm step take that value. Anything it cannot describe raises
    ValueError or is listed in the notes."""
    import ir as IR
    sn = _snap if snap else (lambda v, *a, **k: round(v, 3))
    notes = []
    levels = _core_levels(measured["levels"], measured.get("relief"), notes)
    bumpy = [lv for lv in levels if not lv["round"]]
    if not levels or (bumpy and not measured.get("relief")):
        raise ValueError("only parts made of round steps along Z are handled "
                         "(measure the relief to let textured steps through)")
    if bumpy:
        notes.append(f"{len(bumpy)} level(s) read as round although the scan is not axisymmetric "
                     f"there; the deviation is in the relief patches")
    ax = np.mean([lv["centre"] for lv in levels], axis=0)                # the part's axis
    diam = [sn(lv["outer_diameter"]) for lv in levels]
    bounds = []                            # between levels i and i+1: ("mid", z, z) or ("cone", lo, hi)
    for a, b, da, db in zip(levels, levels[1:], diam, diam[1:]):
        gap = b["z"][0] - a["z"][1]
        if gap >= 1.0 and db <= da - 1.0:
            bounds.append(("cone", a["z"][1], b["z"][0]))
        else:
            if gap >= 1.0:
                notes.append(f"scan blank z {a['z'][1]}..{b['z'][0]} with dia {da} -> {db}: "
                             f"left as a square shoulder midway")
            m = (a["z"][1] + b["z"][0]) / 2
            bounds.append(("mid", m, m))
    starts = [levels[0]["z"][0]] + [bd[2] for bd in bounds]
    ends = [bd[1] for bd in bounds] + [levels[-1]["z"][1]]
    feats, breaks, prev, coned = [], [], None, set()
    for i, lv in enumerate(levels):
        d, h = diam[i], sn(ends[i] - starts[i])
        feats.append(IR.sketch(f"step{i + 1}_outline", circles=[(0, 0, d / 2)],
                               on={"face_of": prev, "side": "top"} if prev else None))
        prev = f"step{i + 1}"
        feats.append(IR.pad(prev, f"step{i + 1}_outline", h))
        if i < len(bounds) and bounds[i][0] == "cone":
            run, dn = round(bounds[i][2] - bounds[i][1], 2), diam[i + 1]
            ang = round(math.degrees(math.atan2((d - dn) / 2, run)), 1)
            feats.append(IR.sketch(f"cone{i + 1}_outline", circles=[(0, 0, d / 2)],
                                   on={"face_of": prev, "side": "top"}))
            prev = f"cone{i + 1}"
            feats.append(IR.pad(prev, f"cone{i + 1}_outline", run, taper=ang))
            notes.append(f"{prev}: scan blank z {bounds[i][1]}..{bounds[i][2]} bridged by a cone, "
                         f"dia {d} -> {dn} ({ang} deg)")
            coned |= {(i, "top_break"), (i + 1, "bottom_break")}
        for end, side in (("bottom_break", "bottom"), ("top_break", "top")):
            br = lv[end]
            if br["height"] < 0.5 or br["radius_lost"] < 0.5 \
                    or (i == 0 and end == "bottom_break") or (i, end) in coned:
                continue
            seen = f"{end.replace('_', ' ')} of {br['height']} x {br['radius_lost']} mm"
            nd = diam[i + 1] if end == "top_break" and i + 1 < len(levels) else \
                diam[i - 1] if end == "bottom_break" and i > 0 else None
            if nd is not None and nd >= d - 0.5:               # the rim edge is swallowed by the neighbour
                notes.append(f"step{i + 1}: {seen} seen but not modelled")
                continue
            size = sn(br["size"]) if br.get("size") else None
            fits = size and size > 0 and (nd is None or size <= (d - nd) / 2 + 0.3)
            if br.get("shape") == "round" and fits:
                breaks.append(IR.fillet(f"step{i + 1}_{side}_round", size,
                                        {"edges_of": f"step{i + 1}", "side": side}))
                notes.append(f"step{i + 1}: {seen} modelled as a {size} mm round")
            elif br.get("shape") == "chamfer" and fits:
                breaks.append(IR.chamfer(f"step{i + 1}_{side}_chamfer", size,
                                         {"edges_of": f"step{i + 1}", "side": side}))
                notes.append(f"step{i + 1}: {seen} modelled as a {size} mm chamfer")
            else:
                notes.append(f"step{i + 1}: {seen} seen but not modelled")

    holes = []                                   # merged across levels: [hole, set of level indices]
    for i, lv in enumerate(levels):
        for h in lv["holes"]:
            for rec in holes:
                if math.dist(rec[0]["centre"], h["centre"]) < 1.0:
                    rec[1].add(i)
                    if ("sides" in h, h["r_min"] / h["r_max"]) > ("sides" in rec[0], rec[0]["r_min"] / rec[0]["r_max"]):
                        rec[0] = h
                    break
            else:
                holes.append([h, {i}])
    usable = []
    for h, seen in holes:
        top = max(seen)
        off = math.dist(h["centre"], ax)
        clear = top + 1 == len(levels) or off - h["r_max"] >= levels[top + 1]["outer_diameter"] / 2 - 0.5
        if min(seen) != 0 or seen != set(range(top + 1)) or not clear:
            notes.append(f"hole at {h['centre']} is not a through hole; left out")
        else:
            usable.append((h, off))

    rounds = sorted((h for h, off in usable if "sides" not in h and off > 0.5),
                    key=lambda h: math.atan2(h["centre"][1] - ax[1], h["centre"][0] - ax[0]))
    n = len(rounds)
    if n >= 3:
        rad = [math.dist(h["centre"], ax) for h in rounds]
        ang = [math.degrees(math.atan2(h["centre"][1] - ax[1], h["centre"][0] - ax[0])) for h in rounds]
        phase = [(a - k * 360.0 / n + 180) % 360 - 180 for k, a in enumerate(ang)]
        if max(rad) - min(rad) < 1.0 and max(phase) - min(phase) < 3.0:
            good = [h["diameter"] for h in rounds if h["shape"] == "round"] or [h["diameter"] for h in rounds]
            d, R, p0 = sn(float(np.median(good))), sn(2 * float(np.mean(rad))) / 2, float(np.mean(phase))
            feats.append(IR.sketch("bolt_circle", circles=[
                (round(R * math.cos(math.radians(p0 + k * 360.0 / n)), 3),
                 round(R * math.sin(math.radians(p0 + k * 360.0 / n)), 3), d / 2) for k in range(n)]))
            feats.append(IR.pocket("bolt_holes", "bolt_circle", through=True))
            notes.append(f"{n} holes read as one pattern: dia {d} on a {2 * R} bolt circle, equally spaced")
            for h in rounds:
                if h["shape"] != "round" or abs(h["diameter"] - d) > 0.5:
                    notes.append(f"hole at {h['centre']} scanned as {h['shape']}, dia {h['diameter']}; "
                                 f"given the pattern's {d}")
            rounds = []
    for k, h in enumerate(rounds):
        c = (h["centre"][0] - ax[0], h["centre"][1] - ax[1])
        feats.append(IR.sketch(f"hole{k + 1}_profile", circles=[(round(c[0], 2), round(c[1], 2), sn(h["diameter"]) / 2)]))
        feats.append(IR.pocket(f"hole{k + 1}", f"hole{k + 1}_profile", through=True))
    for k, (h, off) in enumerate((h, off) for h, off in usable if "sides" in h or off <= 0.5):
        c = (0.0, 0.0) if off < 0.5 else (round(h["centre"][0] - ax[0], 2), round(h["centre"][1] - ax[1], 2))
        name = "bore" if off < 0.5 else f"socket{k + 1}"
        if "sides" in h and h["regular"]:
            feats.append(IR.sketch(f"{name}_profile", ngons=[(c[0], c[1], sn(h["across_flats"]), h["sides"],
                                                            round(h["first_corner_deg"] * 2) / 2)]))
            notes.append(f"{name}: scan's flats measure {h['flat_pairs']} (spread {h['flat_spread']} mm, angles "
                         f"within {h['angle_error']} deg); modelled as one regular {h['sides']}-gon, "
                         f"{sn(h['across_flats'])} across flats")
        elif "sides" in h:
            notes.append(f"{name}: a {h['sides']}-gon but not regular (flats {h['flat_pairs']}); left out")
            continue
        else:
            feats.append(IR.sketch(f"{name}_profile", circles=[(c[0], c[1], sn(h["diameter"]) / 2)]))
        feats.append(IR.pocket(name, f"{name}_profile", through=True))

    pats = (measured.get("relief") or {}).get("patches", [])
    wash = (measured.get("relief") or {}).get("washboard") or []
    if wash:
        # a washboard sector PREEMPTS its own ridges: each ridge also segments as a little
        # panel, and modelling both would stack the texture twice (and the panels would then
        # shadow the pattern that explains them)
        def _is_wash_ridge(p):
            if p["kind"] not in ("recess", "boss"):
                return False
            for wb in wash:
                dth = abs((p["theta_deg"][0] - wb["theta_deg"][0] + 180) % 360 - 180)
                if dth <= wb["theta_deg"][1] / 2 + 5.0 \
                        and p["z"][0] >= wb["z"][0] - wb["pitch"] \
                        and p["z"][1] <= wb["z"][1] + wb["pitch"] \
                        and p["z"][1] - p["z"][0] <= 1.6 * wb["pitch"] \
                        and p["depth"] <= wb["height"] + 0.3:
                    return True
            return False
        eaten = sum(1 for p in pats if _is_wash_ridge(p))
        if eaten:
            notes.append(f"{eaten} relief patch(es) are washboard ridges, carried by its pattern")
            pats = [p for p in pats if not _is_wash_ridge(p)]
    for p in pats:
        if p["kind"] not in ("recess", "boss"):
            notes.append(f"{p['kind'].replace('_', ' ')} z {p['z'][0]}..{p['z'][1]} at "
                         f"{p['theta_deg'][0]} deg ({p['theta_deg'][1]} deg wide), {p['depth']} mm: "
                         f"seen but not modelled")
    covered = []                 # (ang, span, z0, z1) of every panel that became a feature
    for pkind, sgn, fname in (("recess", -1, "inset"), ("boss", 1, "boss")):
        _emit_relief_panels(pats, pkind, sgn, fname, measured, levels, sn, breaks, notes, covered)
    gaps = [a["z"][1] for a, b in zip(levels, levels[1:]) if b["z"][0] - a["z"][1] > 1.0]
    torn_above = gaps[-1] if gaps else None
    z0f = levels[0]["z"][0]
    det = (measured.get("relief") or {}).get("detail")
    if det:
        # line detailing — carved contour lines, embossed ridges — lands as named, dimensioned
        # ENGRAVE features: reference by default (ir.engrave), so the tree carries the design
        # intent without turning every rebuild into hundreds of micro-booleans
        k = 0
        for side, key in (("carve", "carved"), ("emboss", "embossed")):
            kept = 0
            for e in det.get(key, {}).get("lines", []):
                zmid = (e["path"][0][1] + e["path"][-1][1]) / 2
                # the same honesty gates the panels carry: nothing from above the scan's last
                # torn band, and nothing from the first couple of millimetres, where an end
                # face read edge-on writes fictional lines into the residual
                if (torn_above is not None and zmid > torn_above) or zmid < z0f + 2.0:
                    continue
                # a "line" thinner than the scan can resolve is an edge of something else, and
                # so is a ridge with width under a millimetre and no vertical substance at all
                # (one slice tall): a seam or a shoulder rim echoing in the residual
                zspan = max(z for _, z in e["path"]) - min(z for _, z in e["path"])
                if e["width"] < 0.3 or (e["width"] < 1.0 and zspan < 1.0):
                    continue
                # a line inside a modelled panel is either the panel's own wall read back out
                # of the residual (it hugs the traced outline, at about the panel's depth —
                # cutting it would double the edge and osculate the floor) or a REAL groove
                # crossing the interior deeper than the floor, which stays
                from shapely.geometry import Point
                near = deep = 0
                tmax = 0.0
                for t_, z_ in e["path"]:
                    for a, zm, prm, pt_, poly_ in covered:
                        p = Point(math.radians((t_ - a + 180) % 360 - 180) * prm, z_ - zm)
                        if poly_.covers(p):
                            tmax = max(tmax, pt_)
                            if poly_.exterior.distance(p) < 1.2:
                                near += 1
                            else:
                                deep += 1
                            break
                contained = near + deep
                if contained >= 0.5 * len(e["path"]) \
                        and (near >= 0.7 * contained or e["depth"] < tmax + 0.35):
                    continue
                k += 1
                kept += 1
                breaks.append(IR.engrave(f"line{k}", [(t, z - z0f) for t, z in e["path"]],
                                         e["width"], e["depth"], side=side, radius=e["radius"],
                                         build=True))
            if kept:
                tot = sum(e["length_mm"] for e in det[key]["lines"])
                notes.append(f"{kept} {key} line(s), {round(tot)} mm of path in all, modelled as "
                             f"engravings (ir.engrave, built as chord-facet prisms)")
            sh = det.get(key, {}).get("short", {})
            if sh.get("count"):
                notes.append(f"{key} texture: {sh['count']} segments under 8 mm "
                             f"({sh['length_mm']} mm, {sh['area_mm2']} mm2) — a texture field, "
                             f"not modelled")
    k = 0
    for wb in (measured.get("relief") or {}).get("washboard") or []:
        # the grip washboard: one measured ridge repeated along Z — the pattern it is
        zmid = (wb["z"][0] + wb["z"][1]) / 2
        if (torn_above is not None and zmid > torn_above) or wb["z"][0] < z0f + 2.0:
            continue
        count = int((wb["z"][1] - wb["phase_z"]) // wb["pitch"]) + 1
        if count < 4:
            continue
        half = wb["theta_deg"][1] / 2 - 1.0
        angs = np.arange(wb["theta_deg"][0] - half, wb["theta_deg"][0] + half + 2.0, 4.0)
        pts = [(a % 360, z) for a in angs for z in (wb["phase_z"], zmid, wb["z"][1])]
        from shapely.geometry import Point
        inside = sum(1 for t_, z_ in pts
                     if any(poly_.covers(Point(math.radians((t_ - a + 180) % 360 - 180) * prm,
                                               z_ - zm))
                            for a, zm, prm, pt_, poly_ in covered))
        if inside > 0.5 * len(pts):
            notes.append(f"washboard at {wb['theta_deg'][0]} deg mostly inside modelled panels; "
                         f"left to them")
            continue
        k += 1
        path = [(a % 360, wb["phase_z"] - z0f) for a in angs]
        crest = max(wb.get("crest", round(wb["height"] / 2, 2)), 0.2)
        breaks.append(IR.engrave(f"washboard{k}", path, width=wb["ridge_width"],
                                 depth=crest, side="emboss",
                                 radius=wb["radius"], build=True))
        breaks.append(IR.pattern(f"washboard{k}_stack", f"washboard{k}", count,
                                 z_step=wb["pitch"]))
        notes.append(f"washboard{k}: grip ridges at {wb['theta_deg'][0]} deg "
                     f"({wb['theta_deg'][1]} deg wide), one {wb['ridge_width']} mm ridge "
                     f"standing {crest} mm proud, repeated "
                     f"{count} x every {wb['pitch']} mm (z {wb['z'][0]}..{wb['z'][1]})")
    breaks = _coalesce_patterns(breaks, notes)
    return IR.part("part", *feats, *breaks), notes


def _coalesce_patterns(feats, notes):
    """Repeated relief features recognised as the PATTERNS a designer drew: features of one kind
    whose size, depth and radius agree, standing at evenly spaced angles (a polar ring — count 2
    only when opposite) or evenly spaced heights on one angle (a stack), collapse to their first
    instance plus one ir.pattern. The master's traced shape stands for every instance: a moulded
    pattern IS even and identical, and the per-instance differences are scan noise."""
    import ir as IR

    def _p(f):
        o = f["origin"]
        pts = f["polys"][0]
        us, vs = [p[0] for p in pts], [p[1] for p in pts]
        return {"f": f, "kind": f["kind"], "r": math.hypot(o[0], o[1]),
                "th": math.degrees(math.atan2(o[1], o[0])) % 360, "z": o[2],
                "depth": f["depth"], "taper": f.get("taper", 0.0),
                "w": max(us) - min(us), "h": max(vs) - min(vs)}

    def _alike(a, b, same_z):
        # neither taper nor depth is compared tightly: both are the noisiest measurements made
        # here (a shadowed instance scans shallow), and a pattern unifies its members on the
        # median of each instead
        return (a["kind"] == b["kind"]
                and abs(a["depth"] - b["depth"]) <= max(0.8, 0.45 * max(a["depth"], b["depth"]))
                and abs(a["r"] - b["r"]) <= 0.8
                and abs(a["w"] - b["w"]) <= 0.35 * max(a["w"], b["w"])
                and abs(a["h"] - b["h"]) <= 0.35 * max(a["h"], b["h"])
                and (abs(a["z"] - b["z"]) <= 1.0 if same_z
                     else abs((a["th"] - b["th"] + 180) % 360 - 180) <= 7.0))

    def _unify(master, group):
        ts = sorted(d["taper"] for d in group)
        t = round(ts[len(ts) // 2], 1)
        if t:
            master["f"]["taper"] = t
        else:
            master["f"].pop("taper", None)
        ds = sorted(d["depth"] for d in group)
        master["f"]["depth"] = round(ds[len(ds) // 2], 2)

    cand = [_p(f) for f in feats                      # facets (name_1, name_2 ...) stay out:
            if f["kind"] in ("prism_cut", "prism_pad") and "_" not in f["name"]]
    used, plans = set(), {}                       # master name -> (pattern feature, member names)
    for c in cand:                                # polar rings first: same z, even angles
        if c["f"]["name"] in used:
            continue
        ring = [c] + [d for d in cand if d is not c and d["f"]["name"] not in used
                      and _alike(c, d, same_z=True)]
        if len(ring) >= 2:
            ring.sort(key=lambda d: d["th"])
            diffs = [(ring[(i + 1) % len(ring)]["th"] - ring[i]["th"]) % 360
                     for i in range(len(ring))]
            step = 360.0 / len(ring)
            even = all(abs(d - step) <= 7.0 for d in diffs)
            if even and (len(ring) >= 3 or abs(step - 180.0) <= 7.0):
                names = [d["f"]["name"] for d in ring]
                used.update(names)
                _unify(c, ring)
                plans[c["f"]["name"]] = (IR.pattern(f"{c['f']['name']}_ring", c["f"]["name"],
                                                    len(ring), angle_step=round(step, 2)), names)
                notes.append(f"pattern: {' / '.join(names)} are one {len(ring)} x polar ring "
                             f"every {round(step, 1)} deg; drawn once as {c['f']['name']}")
    for c in cand:                                # stacks: same angle, even z steps
        if c["f"]["name"] in used:
            continue
        col = sorted([c] + [d for d in cand if d is not c and d["f"]["name"] not in used
                            and _alike(c, d, same_z=False)], key=lambda d: d["z"])
        if len(col) >= 3:
            dz = [b["z"] - a["z"] for a, b in zip(col, col[1:])]
            step = sorted(dz)[len(dz) // 2]
            if all(abs(d - step) <= 0.75 for d in dz) and step > 0.5:
                base = col[0]
                names = [d["f"]["name"] for d in col]
                used.update(names)
                _unify(base, col)
                plans[base["f"]["name"]] = (IR.pattern(f"{base['f']['name']}_stack",
                                                       base["f"]["name"], len(col),
                                                       z_step=round(step, 2)), names)
                notes.append(f"pattern: {' / '.join(names)} are one {len(col)} x stack every "
                             f"{round(step, 2)} mm; drawn once as {base['f']['name']}")
    if not plans:
        return feats
    masters = {m: p for m, (p, names) in plans.items()}
    members = {n for _, names in plans.values() for n in names} - set(masters)
    out = []
    for f in feats:
        if f.get("name") in members:
            continue
        out.append(f)
        if f.get("name") in masters:
            out.append(masters[f["name"]])
    return out


def _emit_relief_panels(pats, pkind, sgn, fname, measured, levels, sn, breaks, notes, covered=None):
    """The relief patches of one sign as placed features: recesses (sgn -1) become drafted
    prism_cuts into the body, bosses (sgn +1) become drafted prism_pads standing proud of it —
    the same construction mirrored through the surface. Recess bands within 15 degrees and 4 mm
    merge into one panel (one moulded panel scans as bands between the ribs that cross it);
    boss patches stay separate, because neighbouring ribs really are separate features."""
    import ir as IR
    # where measure() had to bridge a blank (no closed slices), the scanner was failing —
    # torn chrome, deep gloss. Relief data above the LAST such band mixes outer and inner
    # walls of torn slices into fictional recesses and bosses, so none of it is modelled.
    gaps = [a["z"][1] for a, b in zip(levels, levels[1:]) if b["z"][0] - a["z"][1] > 1.0]
    torn_above = gaps[-1] if gaps else None
    panels = []
    for p in sorted((p for p in pats if p["kind"] == pkind), key=lambda p: p["z"][0]):
        for g in (panels if pkind == "recess" else []):
            if abs((g["theta"] - p["theta_deg"][0] + 180) % 360 - 180) <= 15 \
                    and abs(g["rm"] - p["mean_radius"]) <= 1.5 and p["z"][0] - g["z"][1] <= 4.0:
                g["bands"].append(p)
                g["z"][1] = max(g["z"][1], p["z"][1])
                g["span"] = max(g["span"], p["theta_deg"][1])
                g["fz"][1] = max(g["fz"][1], p["floor"]["z"][1])
                g["fspan"] = max(g["fspan"], p["floor"]["theta_deg"][1])
                break
        else:
            panels.append({"bands": [p], "z": list(p["z"]), "theta": p["theta_deg"][0],
                           "span": p["theta_deg"][1], "rm": p["mean_radius"],
                           "fz": list(p["floor"]["z"]), "fspan": p["floor"]["theta_deg"][1]})
    for k, g in enumerate(panels):
        bands = g["bands"]
        atot = sum(b["area_mm2"] for b in bands)
        ang = math.degrees(math.atan2(
            sum(b["area_mm2"] * math.sin(math.radians(b["theta_deg"][0])) for b in bands),
            sum(b["area_mm2"] * math.cos(math.radians(b["theta_deg"][0])) for b in bands))) % 360
        rm = sum(b["mean_radius"] * b["area_mm2"] for b in bands) / atot
        t_raw = float(np.median([b["depth"] for b in bands]))
        t = sn(t_raw)
        fill = max(b["fill"] for b in bands)
        where = f"z {g['z'][0]}..{g['z'][1]} at {round(ang, 1)} deg ({g['span']} deg wide)" \
            + (f", {len(bands)} scan bands" if len(bands) > 1 else "")
        upoly = _panel_outline_union(bands, ang, rm, (g["z"][0] + g["z"][1]) / 2,
                                     fuse=2.0 if pkind == "recess" else 0.75)
        # a recess's flat floor is the mould insert itself, so a chord panel holds to 120 deg;
        # a boss's plateau FOLLOWS the body's curve, and a flat pad stands proud of it at the
        # edges by chord sag — honest only while narrow. A wider TRACED feature still lands,
        # as a fan of chord facets, each in its own tangent frame, following the wrap.
        if torn_above is not None and (g["z"][0] + g["z"][1]) / 2 > torn_above:
            notes.append(f"{pkind} {where}: above the scan's last torn band (z > {torn_above}), "
                         f"where slice data is unreliable; left out")
            continue
        single_max = 120 if sgn < 0 else 60
        if t < 0.3 or g["span"] > 355 or (upoly is None and (g["span"] > single_max or fill < 0.6)):
            notes.append(f"{pkind} {where}, {t} mm {'deep' if sgn < 0 else 'proud'} (fill {fill}): "
                         f"seen but too shallow, too wrapped, or too shapeless to model")
            continue
        if t_raw > 1.5 * (g["fz"][1] - g["fz"][0]):        # deeper than tall: a dome or shoulder
            notes.append(f"{pkind} {where}, {t} mm {'deep' if sgn < 0 else 'proud'} but only "
                         f"{round(g['fz'][1] - g['fz'][0], 2)} mm tall: a curved end face seen "
                         f"edge-on, not a moulded feature; left out")
            continue
        # the walls' draft, from the two contours the patch measured: its outline sits where the
        # residual crosses -tol (near the mouth), its floor where it passes 0.6 x depth — the run
        # between them, over that residual drop, is the wall's slope. The chords use each
        # contour's own radius, and a resolution slop (half an angular bin + a quarter slice) is
        # taken off the run so a vertical wall does not read as drafted.
        rel = measured["relief"]
        w = 2 * (rm + sgn * rel.get("tol", 0.3)) * math.sin(math.radians(g["span"]) / 2)
        h = g["z"][1] - g["z"][0]
        wf = min(2 * (rm + sgn * 0.8 * t_raw) * math.sin(math.radians(g["fspan"]) / 2), w)
        hf = min(g["fz"][1] - g["fz"][0], h)
        if wf < 1.5 or hf < 1.0:
            wf, hf = w, h
        drop = max(0.6 * t_raw - rel.get("tol", 0.3), 0.2)
        slop = rm * math.pi / 360 / 2 + rel.get("dz", 0.5) / 4
        ramp = max(((w - wf) / 2 + (h - hf) / 2) / 2 - slop, 0.0)
        taper = -round(min(math.degrees(math.atan2(ramp, drop)), 70.0), 1)
        wf, hf = sn(wf), sn(hf)
        rc = round(min(2.5, wf / 4, hf / 4), 2)                           # moulded corners
        B = -0.41421356       # tan(90/4), negative: on a CCW loop the convex corner bows RIGHT
        # of travel, and _geom's positive bulge bows left (found by SolidCraft's taper check —
        # the other kernels silently built the concave corners)
        rect = [(-wf / 2 + rc, -hf / 2, 0), (wf / 2 - rc, -hf / 2, B), (wf / 2, -hf / 2 + rc, 0),
                (wf / 2, hf / 2 - rc, B), (wf / 2 - rc, hf / 2, 0), (-wf / 2 + rc, hf / 2, B),
                (-wf / 2, hf / 2 - rc, 0), (-wf / 2, -hf / 2 + rc, B)]
        if upoly is None:
            pieces = [(ang, None)]
        elif g["span"] > single_max:
            # facet side walls must mate exactly where the strips meet, so the facets of a
            # wrap carry no draft: a taper would swell every strip into its neighbours
            pieces, taper = _facets(upoly, ang, rm, reverse=sgn < 0), 0.0
        else:
            taper = max(taper, -45.0)      # a traced contour is concave: keep its offset sane
            pieces = [(ang, upoly)]
        if sgn > 0:
            # a moulded rib releases on single-digit draft; the ramp estimate on a band
            # thinner than its wall run reads as a 40-degree wedge whose base swallows the
            # neighbouring ribs (and whose unions the kernels then refuse)
            taper = max(taper, -8.0)
        if not pieces:
            continue
        rf = rm + sgn * 0.8 * t_raw
        make = IR.prism_cut if pkind == "recess" else IR.prism_pad
        zc = (g["z"][0] + g["z"][1]) / 2 - levels[0]["z"][0]
        # a panel that spans body steps sits at its AVERAGE radius: a cut must still punch out
        # through the fattest step it crosses, and a pad must still root inside the thinnest,
        # or the boolean meets the surface face-on-face and the kernel refuses it
        rr = [lv["outer_diameter"] / 2 for lv in levels
              if lv["z"][1] > g["z"][0] and lv["z"][0] < g["z"][1]]
        over = 0.0 if not rr else (max(max(rr) - rm, 0.0) if sgn < 0 else max(rm - min(rr), 0.0))
        for j, (ang_j, pc) in enumerate(pieces):
            poly = rect if pc is None else _chord_poly(pc, rm, rf, sgn > 0)
            phi = math.radians(ang_j)
            half_w = max(max(abs(p[0]) for p in poly), w / 2 if len(pieces) == 1 else 0.0)
            sag = rm - math.sqrt(max(rm * rm - min(half_w, rm) ** 2, 0.0))
            # a facet's flat face straddles the curve it stands for: sunk by half its own chord
            # sag, it errs evenly in and out instead of all to one side. A single moulded floor
            # really is flat (the mould insert), so a lone recess panel keeps its plane.
            uf = min(max(abs(p[0]) for p in poly), rm)
            rp = rm + sgn * t
            straddle = 0.0 if (sgn < 0 and len(pieces) == 1) \
                else (rp - math.sqrt(max(rp * rp - uf * uf, 0.0))) / 2
            # the frame faces the body: a recess cuts outward from under the floor, a boss pads
            # inward from above its plateau — either way the extrusion crosses the surface and
            # the profile grows (negative taper) toward it. xdir keeps v = normal x xdir = +Z.
            r0 = rp - straddle
            breaks.append(make(f"{fname}{k + 1}" + (f"_{j + 1}" if len(pieces) > 1 else ""),
                               origin=[r0 * math.cos(phi), r0 * math.sin(phi), zc],
                               normal=[-sgn * math.cos(phi), -sgn * math.sin(phi), 0.0],
                               xdir=[sgn * math.sin(phi), -sgn * math.cos(phi), 0.0],
                               depth=round(t + sag + 1.0 + straddle + over, 2),
                               polys=[poly], taper=taper))
        if covered is not None:
            from shapely.geometry import box as _box
            covered.append((ang, (g["z"][0] + g["z"][1]) / 2, rm, t,
                            upoly if upoly is not None else _box(-w / 2, -h / 2, w / 2, h / 2)))
        shape_note = (f"{wf} x {hf} mm floor, corners r {rc}" if upoly is None
                      else f"{'floor' if sgn < 0 else 'plateau'} outline traced from the scan"
                      + (f", wrapped as {len(pieces)} chord facets" if len(pieces) > 1 else ""))
        notes.append(f"{fname}{k + 1}: {pkind} {where} modelled as a moulded "
                     f"{'cut' if sgn < 0 else 'pad'}: {shape_note}, {t} mm "
                     f"{'deep' if sgn < 0 else 'proud'}, walls drafted {abs(taper)} deg"
                     + (" (texture crossing the panel is cut away)" if len(bands) > 1 else ""))


def spec_mesh(spec):
    """An IR spec's solid as a trimesh (build123d reference build)."""
    import trimesh

    import b3d_emit
    part, _ = b3d_emit.emit(spec)
    vs, ts = part.tessellate(0.02, 0.05)
    return trimesh.Trimesh(np.array([[v.X, v.Y, v.Z] for v in vs]), np.array(ts), process=True)


def compare(spec, mesh, n=80, scale=False):
    """Volumetric overlap of the spec's solid with the mesh. Both are centred on their XY bounding
    boxes and stood on z = 0; the spin about Z is searched. A height inside the mesh's own extent
    where it has no closed cross-section (an open scan band) is missing data, not empty space:
    those slices are left out of the overlap, and each body's volume is its mean measured slice
    area times its own height. With `scale` the spec is first scaled to the mesh's largest extent.
    Returns {"iou_pct", "spin_deg", "scale", "dvol_pct"}."""
    from shapely import affinity
    k = 1.0
    if scale:
        import image_recognize
        k = float(max(mesh.extents)) / float(max(spec_mesh(spec).extents))
        spec = image_recognize.scale_spec(spec, k)
    a, b = mesh, spec_mesh(spec)
    top = max(a.extents[2], b.extents[2])
    zs = (np.arange(n) + 0.5) * top / n

    def cut(m):
        c = (m.bounds[0] + m.bounds[1]) / 2
        return [affinity.translate(s, -c[0], -c[1]).simplify(0.02) if s is not None else None
                for s in slices(m, zs + m.bounds[0][2])]

    sa, sb = cut(a), cut(b)

    def iou(deg, idx):
        inter = union = 0.0
        for i in idx:
            p, q = sa[i], sb[i]
            if p is None and zs[i] < a.extents[2]:        # the mesh is open here: no data
                continue
            if p is None or q is None:
                union += (p or q).area if (p or q) is not None else 0.0
                continue
            q = affinity.rotate(q, deg, origin=(0, 0))
            inter += p.intersection(q).area
            union += p.union(q).area
        return inter / union if union else 0.0

    few = range(0, n, max(1, n // 10))
    best = max(np.arange(0, 360, 2.0), key=lambda d: iou(d, few))
    best = max(np.arange(best - 2, best + 2.01, 0.25), key=lambda d: iou(d, few))
    va = float(np.mean([s.area for s in sa if s is not None])) * float(a.extents[2])
    vb = float(np.mean([s.area for s in sb if s is not None])) * float(b.extents[2])
    return {"iou_pct": round(100 * iou(best, range(n)), 2), "spin_deg": round(float(best) % 360, 2),
            "scale": round(k, 4), "dvol_pct": round(100 * (vb - va) / va, 2)}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Measure a scanned mesh; score a feature tree against it.")
    ap.add_argument("mesh")
    ap.add_argument("--spec", help="IR JSON (or an image_recognize result) to compare")
    ap.add_argument("--scale", action="store_true", help="scale the spec to the mesh's largest extent")
    ap.add_argument("--tree", metavar="OUT", help="build a feature tree from the measurements and write it here")
    ap.add_argument("--relief", action="store_true",
                    help="also map the off-axis relief (insets, panels, ribs) and model what fits")
    ap.add_argument("--emit", help="write the measurements (and comparison) as JSON")
    a = ap.parse_args()
    mesh = load(a.mesh)
    out = {"mesh": os.path.basename(a.mesh), "measured": measure(mesh)}
    m = out["measured"]
    if a.relief:
        m["relief"] = relief(mesh)
        print(f"relief: noise {m['relief']['noise']} mm, {len(m['relief']['patches'])} patch(es)")
        for p in m["relief"]["patches"]:
            print(f"  {p['kind']}  z {p['z'][0]}..{p['z'][1]}  at {p['theta_deg'][0]} deg "
                  f"({p['theta_deg'][1]} deg / {p['arc_mm']} mm wide)  depth {p['depth']}  "
                  f"area {p['area_mm2']} mm2  fill {p['fill']}")
    print(f"{out['mesh']}: extents {m['extents']} mm, sliced every {m['slice_spacing']} mm")
    for lv in m["levels"]:
        bb, tb = lv["bottom_break"], lv["top_break"]

        def _b(b):
            return f"{b['height']} x {b['radius_lost']}" + \
                (f" ({b['shape']} {b['size']})" if b.get("shape") else "")
        print(f"  z {lv['z'][0]:.1f}..{lv['z'][1]:.1f}  dia {lv['outer_diameter']}  height {lv['height']}"
              f"  rim break (height x radius lost): bottom {_b(bb)}, top {_b(tb)}")
        for h in lv["holes"]:
            what = (f"{h['shape']}, flats {h['flat_pairs']}, {h['across_corners']} across corners "
                    f"(sharp would be {h['ideal_across_corners']}), first corner at {h['first_corner_deg']} deg" if "sides" in h
                    else f"{h['shape']}, dia {h['diameter']} (radius {h['r_min']}..{h['r_max']})")
            print(f"    hole at {h['centre']}: {what}")
    spec = None
    if a.tree:
        spec, out["notes"] = tree(m)
        spec["name"] = os.path.splitext(out["mesh"])[0].lower()
        with open(a.tree, "w") as fh:
            json.dump(spec, fh, indent=1)
        print(f"tree: {len(spec['features'])} features -> {a.tree}")
        for note in out["notes"]:
            print(f"  note: {note}")
    elif a.spec:
        with open(a.spec) as fh:
            spec = json.load(fh)
    if spec:
        out["compare"] = compare(spec.get("spec", spec), mesh, scale=a.scale)
        c = out["compare"]
        print(f"tree vs mesh: overlap {c['iou_pct']}%  volume {c['dvol_pct']:+}%  "
              f"(tree x{c['scale']}, spun {c['spin_deg']} deg)")
    if a.emit:
        with open(a.emit, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
