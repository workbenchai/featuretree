"""Regressions from recognising a moulded screwdriver barrel (2026-11-04)."""
import json

import pytest


def test_faces_of_a_missed_section_is_empty():
    # build123d's intersect returns None when a slice misses the solid (on or past an end face);
    # step_recognize read .faces() off it and crashed
    from step_recognize import _faces
    assert _faces(None) == []


def test_run_script_retry_reads_a_bare_command_list(tmp_path, monkeypatch):
    # emit() writes a bare list; the drop-and-retry path indexed it as {"commands": [...]}
    import solidcraft_emit as sc
    script = [{"command": "sketch.create", "params": {"name": "a_sk"}},
              {"command": "sketch.finish", "params": {}},
              {"command": "solid.extrude", "params": {"sketch": "a_sk", "name": "a"}}]
    p = tmp_path / "s.json"
    p.write_text(json.dumps(script))
    calls = []

    class R:
        def __init__(self, rc, err=""):
            self.returncode, self.stderr, self.stdout = rc, err, ""

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return R(1, "error: step 3 (solid.extrude): refused") if len(calls) == 1 else R(0)

    monkeypatch.setattr(sc, "cli_path", lambda: "solidcraft-cli")
    monkeypatch.setattr(sc.subprocess, "run", fake_run)
    out, dropped = sc.run_script(str(p), str(tmp_path / "o.step"))
    assert dropped == ["a"]
    assert json.loads(p.read_text()) == []          # still a list, with the refused feature gone


def test_tree_core_option():
    from mesh_compare import tree, _patches_on_envelope
    with pytest.raises(ValueError):
        tree({"levels": []}, core="hull")
    lv = [{"z": [0, 10], "outer_diameter": 35.0}]
    pats = [{"kind": "recess", "z": [2, 4], "mean_radius": 16.8, "depth": 1.0},
            {"kind": "boss", "z": [2, 4], "mean_radius": 16.8, "depth": 0.5}]
    rec, boss = _patches_on_envelope(pats, lv)
    assert rec["depth"] == pytest.approx(1.7) and rec["mean_radius"] == 17.5   # floor stays put
    assert boss["depth"] == pytest.approx(-0.2)                                # under the envelope
