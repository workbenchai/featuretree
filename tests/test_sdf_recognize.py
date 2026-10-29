"""sdf_recognize on meshes built from known solids (so each planted feature is the answer).

Ground truth is FUSED with real booleans (build123d), never concatenated: overlapping closed
meshes leave internal faces, which a scanner never records and which corrupt the distance
magnitudes sdf() builds everything on."""
import numpy as np
import pytest
import trimesh

import ir as IR
import sdf_recognize as SDF

CELLS = 2e6                                     # coarse grids keep these tests quick


def _drum(*extra):
    return IR.part("drum",
                   IR.sketch("s", circles=[(0, 0, 15)]),
                   IR.pad("p", "s", 40), *extra)


def _b3d_mesh(shape):
    vs, ts = shape.tessellate(0.02, 0.05)
    return trimesh.Trimesh(np.array([[v.X, v.Y, v.Z] for v in vs]), np.array(ts), process=True)


def _drum_solid():
    from build123d import Cylinder, Pos
    return Pos(0, 0, 20) * Cylinder(15, 40)


@pytest.fixture(scope="module")
def scanned():
    """A drum with a pocket cut into the wall (the tree misses it) and a box boss fused on
    (the tree misses that too): the ground truth for residual()."""
    from build123d import Box, Pos
    truth = _drum_solid() + Pos(-15, 0, 20) * Box(6, 6, 6) \
        - Pos(15.5, 0, 20) * Box(5, 10, 8)      # the pocket: x 13..18 cut into the wall
    return _b3d_mesh(truth)


def test_residual_finds_the_boss_and_the_pocket(scanned):
    out = SDF.residual(scanned, _drum(), max_cells=CELLS)
    assert len(out["added"]) == 1 and len(out["removed"]) == 1
    boss = out["added"][0]
    assert 90 < boss["volume_mm3"] < 125                    # the box's half outside the drum
    assert boss["centroid"][0] == pytest.approx(-16.4, abs=0.5)
    assert boss["reach_mm"] == pytest.approx(3.0, abs=0.3)  # how proud of the tree it stands
    pocket = out["removed"][0]
    assert 110 < pocket["volume_mm3"] < 170
    assert pocket["centroid"][2] == pytest.approx(20, abs=0.5)
    assert pocket["reach_mm"] == pytest.approx(2.0, abs=0.3)  # its depth into the wall


def test_residual_survives_an_open_scan(scanned):
    torn = scanned.copy()
    c = torn.triangles_center
    torn.update_faces(~((np.abs(c[:, 2] - 32) < 1.5) & (c[:, 0] > 5)))   # a hole in the shell
    out = SDF.residual(torn, _drum(), max_cells=CELLS)
    assert len(out["added"]) == 1 and len(out["removed"]) == 1           # same answer as closed


def test_carve_strips_the_rib_and_fills_the_slot():
    from build123d import Box, Cylinder, Pos
    solid = _drum_solid() + Pos(0, 0, 8) * Cylinder(16.5, 2) \
        - Pos(15.5, 0, 20) * Box(5, 10, 2)      # a ring rib, and a 2 mm slot in the wall
    scan = _b3d_mesh(solid)
    out = SDF.carve(scan, r=1.5, max_cells=CELLS)
    assert len(out["added"]) == 1 and len(out["removed"]) == 1
    ring = out["added"][0]
    # the annulus, less the stub at its base an r-ball inside the drum can still reach
    full = np.pi * (16.5 ** 2 - 15 ** 2) * 2
    assert 0.5 * full < ring["volume_mm3"] < 1.1 * full
    assert ring["centroid"][2] == pytest.approx(8, abs=0.5)
    slot = out["removed"][0]                                 # 2 mm wide: the closing ball can't enter
    assert slot["centroid"][2] == pytest.approx(20, abs=0.5)
    assert slot["volume_mm3"] > 15


def test_deviation_scores_only_what_the_scanner_saw(scanned):
    dev = SDF.deviation(scanned, _drum())
    assert dev["max_mm"] > 2.5              # the boss stands ~3 mm proud of the bare drum
    assert dev["median_mm"] < 0.1           # while most of the shell IS the drum
    perfect = SDF.deviation(scanned, scanned)
    assert perfect["p99_mm"] < 1e-6         # a body scores zero against itself


def test_component_meshes_are_watertight(scanned):
    out = SDF.residual(scanned, _drum(), max_cells=CELLS)
    g = out["grid"]
    m = SDF.component_mesh(out["added"][0], g["origin"], g["spacing"])
    assert m.is_watertight and m.volume == pytest.approx(out["added"][0]["volume_mm3"], rel=0.25)
