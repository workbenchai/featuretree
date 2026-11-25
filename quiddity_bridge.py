"""Bridge quiddity's recognition records into featuretree IR features.

quiddity (github.com/pzfreo/quiddity, Apache-2.0) is a B-rep analysis-feature recognizer:
it names machining features on faces, with measured frames and profiles, and marks each
record's evidence as verified or unproved. featuretree builds editable trees and verifies
them by reconstruction. The bridge lets quiddity act as a CANDIDATE PROPOSER inside
featuretree's method competition: its records become prism_cut features, the emitter
rebuilds them, and featuretree's own gates accept or refuse the result. A wrong proposal
costs nothing — it fails verification like any other candidate.

quiddity runs as an external executable (QUIDDITY_EXE env var, or `quiddity` on PATH);
nothing here links against it, and a missing executable degrades to zero proposals.

Profile conventions agree by construction: quiddity emits boundary wires as (u, v, bulge)
in a frame {origin, run, u, v} with v = run × u; featuretree's prism_cut takes (u, v[,
bulge]) wires in the plane through `origin` with +X = `xdir`, +Y = normal × xdir. Setting
origin/normal/xdir from the record's frame maps coordinates through unchanged.

Open profiles (edge-open recesses: steps, open pockets) are closed by extending the first
wall segment backwards and the last wall segment forwards by `margin`, then closing
straight — the flaps push the closing edge outside the stock, so the cut clears the open
sides without a tangent boolean, and material beyond the solid is no-op to subtract.
"""
import json
import os
import shutil
import subprocess

import numpy as np

import ir as IR

DEF_EXE = os.environ.get('QUIDDITY_EXE') or shutil.which('quiddity')

# families bridged so far; each entry is (family, through) — through recesses get a depth
# overshoot on both ends so the cut punches clear of tangent faces
FAMILIES = (('section_recesses', False), ('section_passages', True))


def recognise(step_path, exe=None, timeout=60):
    """Run quiddity on a STEP file; return its recognition document, or None without it."""
    exe = exe or DEF_EXE
    if not exe or not os.path.exists(str(exe)):
        return None
    try:
        out = subprocess.run([exe, str(step_path)], capture_output=True, timeout=timeout)
        if out.returncode != 0:
            return None
        return json.loads(out.stdout)
    except Exception:
        return None


def _close_profile(boundary, closure, margin):
    """Boundary wire [(u, v, bulge), ...] -> closed wire. Open wires get end flaps (see
    module docstring); closed ones pass through."""
    pts = [(float(p['point'][0]), float(p['point'][1]), float(p.get('bulge') or 0.0))
           for p in boundary]
    if closure != 'open' or len(pts) < 2:
        return pts
    a0, a1 = np.array(pts[0][:2]), np.array(pts[1][:2])
    b1, b0 = np.array(pts[-1][:2]), np.array(pts[-2][:2])
    d0 = a1 - a0
    dn = b1 - b0
    n0 = np.linalg.norm(d0)
    nn = np.linalg.norm(dn)
    if n0 < 1e-9 or nn < 1e-9:
        return pts
    # the straight close runs flap-tip to flap-tip, so the flaps must be long enough that
    # it clears the whole open footprint (a short flap turns a corner-open rectangle into
    # a triangle); twice the wire's bbox diagonal always is, and the excess lies outside
    # the solid, where a cut is a no-op
    xy = np.array([p[:2] for p in pts])
    flap = max(margin, 2.0 * float(np.linalg.norm(xy.max(0) - xy.min(0))))
    start_flap = a0 - d0 / n0 * flap
    end_flap = b1 + dn / nn * flap
    # walk: start flap, walls, end flap, straight close back to the start flap
    return ([(float(start_flap[0]), float(start_flap[1]), 0.0)] + pts +
            [(float(end_flap[0]), float(end_flap[1]), 0.0)])


def proposals(doc, name_prefix='q', margin=None, verified_only=True):
    """quiddity recognition document -> list of featuretree prism_cut features.

    Only records whose evidence quiddity itself marks `verified` are proposed (its honesty
    flag, kept); `margin` defaults to 5% of the body diagonal, from the record body_key."""
    if not doc:
        return []
    by_id = {r['id']: r for r in (doc.get('document') or {}).get('records', [])}
    feats = []
    for fam, through in FAMILIES:
        for i, rec in enumerate(doc.get(fam) or []):
            src = rec.get('source') or f'{fam}/{i}'
            meta = by_id.get(f'{fam}/{i}') or by_id.get(src) or {}
            if verified_only and meta.get('evidence_status') not in (None, 'verified'):
                continue
            g = rec.get('geometry') or {}
            fr, prof = g.get('frame'), g.get('profile')
            ri = g.get('run_interval')
            if not (fr and prof and ri):
                continue
            origin = np.array(fr['origin'], float)
            run = np.array(fr['run'], float)
            u = np.array(fr['u'], float)
            nrm = np.linalg.norm(run)
            if nrm < 1e-9:
                continue
            run = run / nrm
            lo, hi = float(ri[0]), float(ri[1])
            m = margin
            if m is None:
                bk = (rec.get('parameters') or meta.get('parameters') or {}).get('body_key')
                bk = bk or rec.get('body_key')
                m = 0.05 * float(np.linalg.norm(np.array(bk[3:6], float))) if bk else 2.0
            over = m if through else 0.0           # punch past tangent caps on through cuts
            wire = _close_profile(prof.get('boundary') or [], prof.get('closure'), m)
            if len(wire) < 3:
                continue
            feats.append(IR.prism_cut(
                f'{name_prefix}_{fam}_{i}',
                origin + (lo - over) * run, run, u,
                (hi - lo) + 2 * over, polys=[wire]))
    return feats


def into_frame(feats, point, direction):
    """Map proposals from FILE coordinates into the recognizer's build frame.

    quiddity reports in file coordinates; a recovered spec's features live in the build
    frame (extrude axis rotated onto +Z, base at z = 0). `point`/`direction` are the
    forward maps that step_recognize.input_frame returns (build -> file); this applies
    their exact inverse, so a bridged cut lands on the same material the original holds."""
    R = np.array([direction((1, 0, 0)), direction((0, 1, 0)), direction((0, 0, 1))]).T
    t = np.array(point((0, 0, 0)))
    out = []
    for f in feats:
        f = dict(f)
        f['origin'] = [round(float(c), 6) for c in R.T @ (np.array(f['origin']) - t)]
        for key in ('normal', 'xdir'):
            f[key] = [round(float(c), 6) for c in R.T @ np.array(f[key])]
        out.append(f)
    return out


if __name__ == '__main__':
    import sys
    doc = recognise(sys.argv[1])
    for f in proposals(doc):
        print(json.dumps(f))
