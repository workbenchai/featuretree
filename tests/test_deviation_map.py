"""deviation_map on scans built from known solids, so every planted error is the answer.

The "scan" is a tree built with real booleans (build123d, via mesh_compare.spec_mesh); the tree
under test is the same design with a known mistake. The map must point at the mistake, with
the right sign; refine() must walk the mistake back; square_roll() must find a planted roll."""
import math

import numpy as np
import pytest

import deviation_map as DM
import ir as IR
import mesh_compare as mc


def _ring(r, n=144):
    return [(r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n)) for k in range(n)]


def _drum(r=15.0, cut=None, extra=()):
    """A 40 mm drum (a 144-gon base level, like the recognizer's harmonic levels) with an
    optional wall pocket: cut = (u_centre, v_centre, half_u, half_v, floor_radius)."""
    feats = [IR.sketch("body_outline", polys=[_ring(r)]), IR.pad("body", "body_outline", 40)]
    if cut:
        uc, vc, hu, hv, rf = cut
        feats.append(IR.prism_cut("inset", origin=(rf, 0, 20), normal=(1, 0, 0), xdir=(0, 1, 0),
                                  depth=r + 3 - rf,
                                  polys=[[(uc - hu, vc - hv), (uc + hu, vc - hv),
                                          (uc + hu, vc + hv), (uc - hu, vc + hv)]]))
    return IR.part("drum", *feats, *extra)


TRUTH_CUT = (0.0, 0.0, 5.0, 4.0, 13.5)          # a 10 x 8 mm pocket, floor 1.5 mm deep


@pytest.fixture(scope="module")
def scan():
    return mc.spec_mesh(_drum(cut=TRUTH_CUT))


def _inset(spec):
    return next(f for f in spec["features"] if f["name"] == "inset")


def test_map_sign_convention(scan):
    """A tree 0.3 mm too fat reads +0.3 (proud, red) over the bare body."""
    dm = DM.signed_map(scan, _drum(r=15.3, cut=TRUTH_CUT), dz=0.5)
    att = DM.attribute(dm, _drum(r=15.3, cut=TRUTH_CUT))
    body = next(r for r in att["features"] if r["name"] == "body")
    assert body["median_mm"] == pytest.approx(0.3, abs=0.05)
    assert body["leans"] == "proud"


def test_missing_pocket_is_unexplained_removed(scan):
    """The tree lacks the scan's pocket: tree material the scan refutes, owned by no placed
    feature — reported where it is, with its depth."""
    att = DM.attribute(DM.signed_map(scan, _drum(), dz=0.5), _drum())
    loose = att["unexplained"]
    assert loose and loose[0]["kind"] == "removed"
    assert loose[0]["z"][0] == pytest.approx(16, abs=1.0) and loose[0]["z"][1] == pytest.approx(24, abs=1.0)
    th = loose[0]["theta_deg"][0]
    assert min(th, 360 - th) < 3                               # centred on +X
    assert loose[0]["peak_mm"] == pytest.approx(1.5, abs=0.3)


def test_misplaced_pocket_is_attributed_to_it(scan):
    wrong = _drum(cut=(1.5, 0.0, 5.0, 4.0, 13.5))
    att = DM.attribute(DM.signed_map(scan, wrong, dz=0.5), wrong)
    assert att["features"][0]["name"] == "inset"                # the worst offender, by error volume


def test_footprint_covers_the_outline(scan):
    dm = DM.signed_map(scan, _drum(cut=TRUTH_CUT), dz=0.5)
    m = DM.footprint(dm, _inset(_drum(cut=TRUTH_CUT)))
    rows, cols = np.where(m)
    assert dm["zs"][rows].min() == pytest.approx(16, abs=0.6) and dm["zs"][rows].max() == pytest.approx(24, abs=0.6)
    th = (dm["th_deg"][cols] + 180) % 360 - 180
    half = math.degrees(math.atan2(5.0, 13.5))                 # +-5 mm at the floor, seen radially
    assert th.min() == pytest.approx(-half, abs=1.0) and th.max() == pytest.approx(half, abs=1.0)


def test_bulge_outline_densifies_to_the_arc():
    """A half-disc wire given by two points and a bulge of 1 (a 180-degree arc)."""
    uv = DM._densify([(-5, 0, 1.0), (5, 0, 0.0)])
    r = np.hypot(uv[:, 0], uv[:, 1])
    arc = uv[:, 1] < -1e-6                                     # ccw from (-5,0) to (5,0) dips below
    assert arc.sum() > 10 and np.allclose(r[arc], 5.0, atol=1e-6)


def test_refine_walks_a_wrong_pocket_back(scan):
    """Shifted 1.2 mm along u, 1 mm too narrow in v, floor 0.6 mm too shallow — and a body
    0.25 mm too fat. Refinement must bring every one of them back, and say what it did."""
    wrong = _drum(r=15.25, cut=(1.2, 0.0, 5.0, 3.5, 14.1))
    out, rep = DM.refine(wrong, scan, iters=4, dz=0.5, log=lambda *a: None)
    assert rep["end"]["median_abs_mm"] < rep["start"]["median_abs_mm"] * 0.5
    f = _inset(out)
    u = [p[0] for p in f["polys"][0]]
    v = [p[1] for p in f["polys"][0]]
    assert (min(u) + max(u)) / 2 == pytest.approx(0.0, abs=0.35)     # shift undone
    assert max(v) - min(v) == pytest.approx(8.0, abs=0.6)            # width restored
    assert f["origin"][0] == pytest.approx(13.5, abs=0.2)            # floor deepened
    body = next(g for g in out["features"] if g["name"] == "body_outline")
    rr = [math.hypot(x, y) for x, y, *_ in body["polys"][0]]
    assert np.median(rr) == pytest.approx(15.0, abs=0.1)
    kept = {k for it in rep["iterations"] for k in it["kept"]}
    assert {"inset", "body_outline"} <= kept


def test_refine_reverts_what_does_not_help(scan):
    """An exact tree has nothing to fix: refinement must leave it alone, not hunt noise."""
    exact = _drum(cut=TRUTH_CUT)
    out, rep = DM.refine(exact, scan, iters=2, dz=0.5, log=lambda *a: None)
    assert rep["end"]["median_abs_mm"] <= rep["start"]["median_abs_mm"] + 1e-3
    f0, f1 = _inset(exact), _inset(out)
    assert np.allclose(np.array(f0["polys"][0])[:, :2], np.array(f1["polys"][0])[:, :2], atol=0.15)


def test_square_roll_finds_a_planted_roll():
    """Two pockets mirrored about the plane at 12 degrees: spinning by -12 squares the part."""
    def cut(name, ang):
        c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        return IR.prism_cut(name, origin=(13.5 * c, 13.5 * s, 20), normal=(c, s, 0), xdir=(-s, c, 0),
                            depth=5, polys=[[(-5, -6), (5, -6), (5, 6), (-5, 6)]])
    rolled = mc.spec_mesh(_drum(extra=[cut("a", 12 + 60), cut("b", 12 - 60)]))
    sq = DM.square_roll(rolled)
    assert sq["spin_deg"] == pytest.approx(-12.0, abs=0.75) or sq["spin_deg"] == pytest.approx(78.0, abs=0.75)
    assert sq["confidence"] > 0.3


def test_render_writes_a_png(scan, tmp_path):
    pytest.importorskip("matplotlib")
    spec = _drum(cut=(1.5, 0.0, 5.0, 4.0, 13.5))
    p = DM.render(DM.signed_map(scan, spec, dz=0.5), str(tmp_path / "m.png"), spec)
    assert (tmp_path / "m.png").stat().st_size > 5000 and p.endswith(".png")
