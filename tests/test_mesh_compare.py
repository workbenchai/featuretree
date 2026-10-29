"""mesh_compare on meshes exported from known trees (so the right answer is known)."""
import pytest

import ir as IR
import mesh_compare as MC
from test_edge_breaks import _bushing


@pytest.fixture(scope="module")
def chamfered():
    return _bushing(IR.chamfer("rim", 2, {"edges_of": "boss", "side": "top"}))


def test_measure_finds_levels_hex_holes_and_the_chamfer(chamfered):
    m = MC.measure(MC.spec_mesh(chamfered))
    flange, boss = m["levels"]
    assert flange["outer_diameter"] == pytest.approx(100, abs=0.3) and flange["height"] == pytest.approx(30, abs=0.5)
    assert boss["outer_diameter"] == pytest.approx(70, abs=0.3) and boss["height"] == pytest.approx(90, abs=0.5)
    hexes = [h for h in flange["holes"] if h.get("sides") == 6]
    assert len(hexes) == 1 and hexes[0]["across_flats"] == pytest.approx(40, abs=0.2)
    assert hexes[0]["across_corners"] == pytest.approx(hexes[0]["ideal_across_corners"], abs=0.2)
    rounds = [h for h in flange["holes"] if h["shape"] == "round"]
    assert len(rounds) == 4 and all(h["diameter"] == pytest.approx(10, abs=0.2) for h in rounds)
    assert boss["top_break"]["height"] == pytest.approx(2, abs=0.15)          # the 2 mm chamfer
    assert boss["top_break"]["radius_lost"] == pytest.approx(2, abs=0.15)
    assert flange["top_break"] == {"height": 0.0, "radius_lost": 0.0}          # a sharp rim


def test_compare_scores_the_same_tree_high_and_a_wrong_one_lower(chamfered):
    mesh = MC.spec_mesh(chamfered)
    same = MC.compare(chamfered, mesh)
    assert same["iou_pct"] > 99.5 and abs(same["dvol_pct"]) < 0.2
    wrong = _bushing()
    wrong["features"][4]["ngons"] = [[0, 0, 50, 6, 0]]             # a 50 mm hex instead of 40
    assert MC.compare(wrong, mesh)["iou_pct"] < same["iou_pct"] - 2


def test_compare_finds_spin_and_scale(chamfered):
    import image_recognize as IM
    mesh = MC.spec_mesh(chamfered)
    mesh.apply_transform([[0.5, -0.8660254, 0, 0], [0.8660254, 0.5, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])  # 60 deg
    mesh.apply_translation([7, -3, 11])
    r = MC.compare(IM.scale_spec(chamfered, 0.5), mesh, scale=True)
    assert r["scale"] == pytest.approx(2, rel=0.01)
    assert r["iou_pct"] > 99


def test_tree_from_a_mesh_gives_one_size_per_shape(chamfered):
    mesh = MC.spec_mesh(chamfered)
    spec, notes = MC.tree(MC.measure(mesh))
    f = {x["name"]: x for x in spec["features"]}
    assert f["bore_profile"]["ngons"][0][2:4] == [40.0, 6]             # a hexagon, by one number
    assert f["step1_outline"]["circles"] == [[0, 0, 50.0]] and f["step1"]["length"] == 30.0
    assert f["step2_outline"]["circles"] == [[0, 0, 35.0]] and f["step2"]["length"] == 90.0
    holes = f["bolt_circle"]["circles"]
    assert len(holes) == 4 and {h[2] for h in holes} == {5.0}
    assert all((h[0] ** 2 + h[1] ** 2) ** 0.5 == pytest.approx(42, abs=0.01) for h in holes)
    assert any("one pattern" in n for n in notes)
    assert any("top break" in n for n in notes)                       # the chamfer is reported, not modelled
    assert not IR.validate(spec)
    assert MC.compare(spec, mesh)["iou_pct"] > 99.3


def test_tree_refuses_to_call_a_lopsided_hexagon_regular():
    spec = _bushing()
    hexagon = IR.ngon_poly(0, 0, 40, 6, 0)
    hexagon[1][1] += 1.5                                                # push one corner out
    spec["features"][4] = IR.sketch("hex_profile", polys=[hexagon])
    made, notes = MC.tree(MC.measure(MC.spec_mesh(spec)))
    assert any("not regular" in n for n in notes)
    assert not any(f.get("ngons") for f in made["features"])


@pytest.fixture(scope="module")
def filleted():
    return _bushing(IR.fillet("rim", 2, {"edges_of": "boss", "side": "top"}))


def test_rim_break_tells_a_round_from_a_chamfer(chamfered, filleted):
    cham = MC.measure(MC.spec_mesh(chamfered))["levels"][1]["top_break"]
    assert cham["shape"] == "chamfer" and cham["size"] == pytest.approx(2, abs=0.1)
    rnd = MC.measure(MC.spec_mesh(filleted))["levels"][1]["top_break"]
    assert rnd["shape"] == "round" and rnd["size"] == pytest.approx(2, abs=0.1)


def test_tree_models_the_chamfer_and_the_fillet(chamfered, filleted):
    for made_from, kind, key in ((chamfered, "chamfer", "distance"), (filleted, "fillet", "radius")):
        mesh = MC.spec_mesh(made_from)
        spec, notes = MC.tree(MC.measure(mesh))
        rims = [f for f in spec["features"] if f["kind"] == kind]
        assert len(rims) == 1 and rims[0][key] == 2.0
        assert rims[0]["select"] == {"edges_of": "step2", "side": "top"}
        assert any("top break" in n and "modelled as" in n for n in notes)
        assert not IR.validate(spec)
        assert MC.compare(spec, mesh)["iou_pct"] > 99.5


def test_tree_bridges_a_scan_blank_with_a_cone():
    spec = IR.part(
        "coned",
        IR.sketch("big", circles=[(0, 0, 30)]),
        IR.pad("base", "big", 20),
        IR.sketch("cone_sk", circles=[(0, 0, 30)], on={"face_of": "base", "side": "top"}),
        IR.pad("cone", "cone_sk", 10, taper=45.0),                      # dia 60 -> 40 over 10
        IR.sketch("small", circles=[(0, 0, 20)], on={"face_of": "cone", "side": "top"}),
        IR.pad("tip", "small", 20),
    )
    full = MC.spec_mesh(spec)
    holed = full.copy()
    c = holed.triangles_center[:, 2]
    holed.update_faces(~((c > 20.5) & (c < 29.5)))                      # the cone band goes unscanned
    m = MC.measure(holed)
    assert [lv["outer_diameter"] for lv in m["levels"]] == [pytest.approx(60, abs=0.5),
                                                            pytest.approx(40, abs=1.0)]
    made, notes = MC.tree(m)
    cones = [f for f in made["features"] if f["kind"] == "pad" and f.get("taper")]
    assert len(cones) == 1 and 35 <= cones[0]["taper"] <= 55
    assert any("bridged by a cone" in n for n in notes)
    assert not IR.validate(made)
    assert MC.compare(made, full)["iou_pct"] > 98.5


@pytest.fixture(scope="module")
def switched():
    return IR.part(                                                     # a drum with one inset well
        "drum",
        IR.sketch("body_outline", circles=[(0, 0, 20)]),
        IR.pad("body", "body_outline", 40),
        IR.prism_cut("switch_well", origin=[18.5, 0, 20], normal=[1, 0, 0], xdir=[0, 1, 0],
                     depth=5, polys=[[(-6, -5), (6, -5), (6, 5), (-6, 5)]]),
    )


def test_relief_finds_the_inset(switched):
    rel = MC.relief(MC.spec_mesh(switched))
    rec = [p for p in rel["patches"] if p["kind"] == "recess"]
    assert len(rec) == 1
    p = rec[0]
    assert abs((p["theta_deg"][0] + 180) % 360 - 180) < 5               # centred on 0 deg
    assert p["depth"] == pytest.approx(1.5, abs=0.3)
    assert p["z"][0] == pytest.approx(15, abs=1) and p["z"][1] == pytest.approx(25, abs=1)
    assert p["fill"] > 0.6


def test_relief_reports_a_ring_rib():
    import trimesh
    body = trimesh.creation.cylinder(radius=20, height=40, sections=128)
    rib = trimesh.creation.cylinder(radius=21, height=2, sections=128)
    drum = trimesh.util.concatenate([body.apply_translation([0, 0, 20]),
                                     rib.apply_translation([0, 0, 20])])
    rings = [p for p in MC.relief(drum)["patches"] if p["kind"] == "ring_rib"]
    assert len(rings) == 1
    assert rings[0]["depth"] == pytest.approx(1.0, abs=0.2)
    assert rings[0]["z"][0] == pytest.approx(19, abs=1) and rings[0]["z"][1] == pytest.approx(21, abs=1)


def test_tree_cuts_the_inset_it_found(switched):
    mesh = MC.spec_mesh(switched)
    m = MC.measure(mesh)
    m["relief"] = MC.relief(mesh)
    made, notes = MC.tree(m)
    cuts = [f for f in made["features"] if f["kind"] == "prism_cut"]
    assert len(cuts) == 1
    assert any("modelled as a" in n and "cut" in n for n in notes)
    assert not IR.validate(made)
    assert MC.compare(made, mesh)["iou_pct"] > 99


def test_a_moulded_boss_comes_back_as_a_prism_pad():
    spec = IR.part(
        "ribbed",
        IR.sketch("body_outline", circles=[(0, 0, 20)]),
        IR.pad("body", "body_outline", 40),
        IR.prism_pad("pad1", origin=[21.5, 0, 20], normal=[-1, 0, 0], xdir=[0, -1, 0],
                     depth=4, polys=[[(-5, -4, 0), (5, -4, 0), (5, 4, 0), (-5, 4, 0)]],
                     taper=-30.0),                       # stands ~1.5 proud, walls drafted
    )
    mesh = MC.spec_mesh(spec)
    m = MC.measure(mesh)
    m["relief"] = MC.relief(mesh)
    made, notes = MC.tree(m)
    pads = [f for f in made["features"] if f["kind"] == "prism_pad"]
    assert len(pads) == 1
    assert pads[0]["depth"] > 1.0                        # tall enough to cross the surface
    assert any("modelled as a moulded pad" in n for n in notes)
    assert not IR.validate(made)
    assert MC.compare(made, mesh)["iou_pct"] > 99


def test_carved_line_detailing_comes_back_as_a_reference_engraving():
    import math
    wave = [(t, 20 + 6 * math.sin(math.radians(t))) for t in range(0, 361, 10)]
    src = IR.part(
        "lined",
        IR.sketch("s", circles=[(0, 0, 15)]),
        IR.pad("p", "s", 40),
        IR.engrave("groove", wave, width=1.6, depth=1.0, radius=15.0, build=True),
    )
    mesh = MC.spec_mesh(src)
    bare = math.pi * 225 * 40
    assert mesh.volume < bare - 50                     # build=True really carved the line
    m = MC.measure(mesh)
    m["relief"] = MC.relief(mesh)
    made, notes = MC.tree(m)
    eng = [f for f in made["features"] if f["kind"] == "engrave"]
    assert eng and eng[0]["side"] == "carve" and not eng[0]["build"]
    assert 1.0 < eng[0]["width"] < 2.8 and 0.5 < eng[0]["depth"] < 1.4
    assert any("reference engraving" in n for n in notes)
    assert not IR.validate(made)
    # reference by default: the recovered tree rebuilds the bare drum, volumes stay comparable
    assert MC.spec_mesh(made).volume == pytest.approx(bare, rel=0.02)


def test_a_ring_of_bosses_comes_back_as_one_pattern():
    spec = IR.part(
        "ringed",
        IR.sketch("body_outline", circles=[(0, 0, 20)]),
        IR.pad("body", "body_outline", 40),
        IR.prism_pad("pad1", origin=[21.5, 0, 20], normal=[-1, 0, 0], xdir=[0, -1, 0],
                     depth=4, polys=[[(-5, -4, 0), (5, -4, 0), (5, 4, 0), (-5, 4, 0)]]),
        IR.pattern("pad_ring", "pad1", count=4, angle_step=90.0),
    )
    mesh = MC.spec_mesh(spec)
    m = MC.measure(mesh)
    m["relief"] = MC.relief(mesh)
    made, notes = MC.tree(m)
    pads = [f for f in made["features"] if f["kind"] == "prism_pad"]
    rings = [f for f in made["features"] if f["kind"] == "pattern"]
    assert len(pads) == 1 and len(rings) == 1          # four scans of one design decision
    assert rings[0]["count"] == 4 and rings[0]["angle_step"] == pytest.approx(90.0, abs=2.0)
    assert any("polar ring" in n for n in notes)
    assert not IR.validate(made)
    assert MC.compare(made, mesh)["iou_pct"] > 99


def test_a_moulded_well_comes_back_drafted_and_rounded():
    B = -0.41421356                            # quarter-round corners, convex on a CCW loop
    well = [(-4 + 2, -3, 0), (4 - 2, -3, B), (4, -3 + 2, 0), (4, 3 - 2, B),
            (4 - 2, 3, 0), (-4 + 2, 3, B), (-4, 3 - 2, 0), (-4, -3 + 2, B)]
    spec = IR.part(
        "moulded",
        IR.sketch("body_outline", circles=[(0, 0, 20)]),
        IR.pad("body", "body_outline", 40),
        IR.prism_cut("well", origin=[18, 0, 20], normal=[1, 0, 0], xdir=[0, 1, 0],
                     depth=4, polys=[well], taper=-40.0),               # 2 deep, walls drafted 40 deg
    )
    mesh = MC.spec_mesh(spec)
    m = MC.measure(mesh)
    m["relief"] = MC.relief(mesh)
    made, notes = MC.tree(m)
    cuts = [f for f in made["features"] if f["kind"] == "prism_cut"]
    assert len(cuts) == 1
    assert -55 <= cuts[0]["taper"] <= -20                               # the draft is measured
    poly = cuts[0]["polys"][0]
    # the shape comes back either as the traced floor outline (the corner rounding carried by
    # its many chords) or as the rounded-rect fallback (the rounding carried by arc bulges)
    traced = len(poly) > 8 and not any(abs(b) > 0.3 for seg in poly for b in seg[2:])
    if traced:
        us, vs = [p[0] for p in poly], [p[1] for p in poly]
        assert 6.0 < max(us) - min(us) < 9.5 and 4.0 < max(vs) - min(vs) < 7.5   # the 8 x 6 floor
    else:
        assert any(abs(b) > 0.3 for seg in poly for b in seg[2:])       # corners are arcs
    assert any("moulded cut" in n and "drafted" in n for n in notes)
    assert not IR.validate(made)
    assert MC.compare(made, mesh)["iou_pct"] > 99


def test_measure_survives_an_open_slice_at_a_level_middle():
    import numpy as np
    import trimesh
    r, h, nth, nz = 20.0, 20.0, 96, 201                                 # rings every 0.1 mm
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    ring = np.stack([r * np.cos(th), r * np.sin(th)], axis=1)
    V = np.array([[x, y, z] for z in np.linspace(0, h, nz) for x, y in ring])
    F = []
    for k in range(nz - 1):
        a, b = k * nth, (k + 1) * nth
        for t in range(nth):
            u = (t + 1) % nth
            F += [[a + t, a + u, b + t], [a + u, b + u, b + t]]
    tube = trimesh.Trimesh(V, np.array(F), process=False)
    z = tube.vertices[tube.faces][:, :, 2]
    tube.update_faces(~np.any(np.abs(z - 10.0) < 1e-6, axis=1))         # tear out the ring at z = 10
    m = MC.measure(tube)                                                # the level's midpoint slice
    assert len(m["levels"]) == 1                                        # ... is exactly z = 10
    assert m["levels"][0]["outer_diameter"] == pytest.approx(40, abs=0.3)
