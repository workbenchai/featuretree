"""The SolidCraft backend: the command stream it writes, and (with the CLI installed) that
SolidCraft builds the same geometry build123d does."""
import json

import pytest

import ir as IR
import solidcraft_emit as SC
from test_edge_breaks import _bushing


def _named(cmds, name):
    return next(c for c in cmds if c["params"].get("name") == name)


def test_bushing_becomes_native_commands():
    cmds = SC.emit(_bushing(IR.chamfer("rim", 2, {"edges_of": "boss", "side": "top"})))
    assert cmds[0] == {"command": "document.units", "params": {"units": "mm"}}
    flange = _named(cmds, "flange")
    assert flange["params"]["operation"] == "new" and flange["params"]["body_name"] == "bushing"
    assert _named(cmds, "boss")["params"]["operation"] == "join"
    boss_sk = _named(cmds, "boss_outline")
    assert boss_sk["params"]["plane"]["origin"][2] == pytest.approx(30, abs=0.01)   # the flange top
    hexp = _named(cmds, "hex")
    assert hexp["params"]["operation"] == "cut" and hexp["params"]["through_all"] is True
    rim = _named(cmds, "rim")
    assert rim["params"]["distance"] == "2.0 mm"
    assert rim["params"]["edges"] and all(abs(p[2] - 120) < 1e-6 for p in rim["params"]["edges"])


def test_taper_flips_sign_and_xz_revolve_gets_its_plane():
    spec = IR.part("t",
                   IR.sketch("s", circles=[(0, 0, 20)]),
                   IR.pad("p", "s", 10, taper=5.0),
                   IR.sketch("rp", plane="XZ", polys=[[(25, 0), (30, 0), (30, 4), (25, 4)]]),
                   IR.revolve("ring", "rp"))
    cmds = SC.emit(spec)
    assert _named(cmds, "p")["params"]["taper"] == "-5.0 deg"       # IR shrinks, SolidCraft grows
    rp = _named(cmds, "rp")
    assert rp["params"]["plane"]["x_dir"] == [1, 0, 0] and rp["params"]["plane"]["y_dir"] == [0, 0, 1]
    ring = _named(cmds, "ring")
    assert ring["params"]["axis"] == "Z" and ring["params"]["angle"] == "360.0 deg"


def test_prism_cut_plane_and_circle_query_refusal():
    spec = IR.part("t",
                   IR.sketch("s", circles=[(0, 0, 20)]),
                   IR.pad("p", "s", 40),
                   IR.prism_cut("well", origin=[18, 0, 20], normal=[1, 0, 0], xdir=[0, 1, 0],
                                depth=4, polys=[[(-4, -3), (4, -3), (4, 3), (-4, 3)]], taper=-40.0))
    cmds = SC.emit(spec)
    sk = _named(cmds, "well_sk")
    assert sk["params"]["plane"] == {"origin": [18, 0, 20], "x_dir": [0, 1, 0], "y_dir": [0, 0, 1]}
    assert _named(cmds, "well")["params"]["taper"] == "40.0 deg"
    bad = IR.part("t2", IR.sketch("s", circles=[(0, 0, 20)]), IR.pad("p", "s", 10),
                  IR.fillet("rim", 1, {"circles": "top_outer"}))
    with pytest.raises(NotImplementedError):
        SC.emit(bad)


def _has_cli():
    try:
        SC.cli_path()
        return True
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _has_cli(), reason="needs solidcraft-cli")
def test_solidcraft_builds_what_build123d_builds(tmp_path):
    import b3d_emit
    import trimesh
    spec = _bushing(IR.chamfer("rim", 2, {"edges_of": "boss", "side": "top"}))
    script = tmp_path / "bushing.sc.json"
    script.write_text(json.dumps({"commands": SC.emit(spec)}))
    stl = SC.run_script(script, tmp_path / "bushing.stl")
    v = float(trimesh.load(str(stl)).volume)
    ref = b3d_emit.emit(spec)[1]["volume"]
    assert abs(v - ref) / ref < 0.005
