"""solidcraft_emit.py — emit a featuretree IR as a native SolidCraft command script.

SolidCraft (https://github.com/tangency-labs/solidcraft) builds parts from a JSON list of the
same commands its UI runs, replayed onto an editable parametric timeline — so, as with the
FreeCAD and Onshape backends, the IR becomes NAMED, NATIVE features (sketch.create /
solid.extrude / solid.fillet ...), not a dead solid. Geometric references are world-coordinate
points the engine re-resolves against live geometry (fillet / chamfer edges straight from
ir.lower()'s picks), so no kernel ids are stored.

    python3 solidcraft_emit.py part.ir.json out.sc.json            # write the command script
    python3 solidcraft_emit.py part.ir.json out.sc.json --run x.step   # + run solidcraft-cli
    python3 solidcraft_emit.py part.ir.json out.sc.json --verify   # + volume gate vs build123d

The CLI is found via $SOLIDCRAFT_CLI, the macOS app bundle, or PATH.

Conventions pinned by tests/test_solidcraft_emit.py (live, when the CLI is installed):
  * every sketch plane is explicit — {origin, x_dir, y_dir}, normal = x_dir × y_dir, and
    extrude direction "positive" runs along that normal;
  * SolidCraft's `taper` is positive OUTWARD (the far end grows along the extrusion); the IR's
    positive SHRINKS (build123d's convention), so the sign flips;
  * a through pocket is `through_all` with direction "symmetric" (both ways, as b3d_emit);
  * all coordinates are millimetres (the script sets document.units mm first).

Coverage: sketches (circles / rects / polygons with bulge arcs) on XY, XZ or a top/bottom face;
pad (blind, symmetric, taper), pocket (through / blind, taper), revolve about Z, fillet and
chamfer from edges_of picks, prism_cut on any plane, polar_pocket. Not yet: fillet / chamfer by
{"circles"} query — that selector is resolved against live geometry, which a one-way script
cannot ask for (raises NotImplementedError).
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "targets"))
import _geom as G  # noqa: E402
import ir as IR  # noqa: E402

APP_CLI = "/Applications/SolidCraft.app/Contents/MacOS/solidcraft-cli"

# A pad that exactly MEETS the part (a cylinder on a cone's top rim, a cone flush on a
# cylinder) can give the union engine nothing to merge ("boolean Union: no result") or a result
# its plausibility guard refuses — just as OpenCASCADE does for b3d_emit, which sinks the tool
# into the part. Here the sketch plane itself is sunk 1 um into the face it sits on and the
# extrusion lengthened to match, so the far end stays exact.
SINK = 1e-3
# A TAPERED pad flush on a step (a recognized cone) osculates the wall below along its base
# circle and the step above along its top circle, and SolidCraft's union refuses tangent seams
# at any sink depth ("no result"). Its base circle is widened by this sliver instead, so both
# seams cross transversally; the silhouette error (20 um) is far below scan resolution.
BITE = 0.02


def cli_path():
    p = os.environ.get("SOLIDCRAFT_CLI")
    if p and Path(p).exists():
        return p
    if Path(APP_CLI).exists():
        return APP_CLI
    w = shutil.which("solidcraft-cli")
    if w:
        return w
    raise FileNotFoundError("solidcraft-cli not found: set SOLIDCRAFT_CLI or install SolidCraft")


def _mm(v):
    return f"{round(float(v), 6)} mm"


def _deg(v):
    return f"{round(float(v), 6)} deg"


def _r(v):
    return [round(float(c), 9) for c in v]


class Emitter:
    def __init__(self, part_name):
        self.part = part_name
        self.cmds = [{"command": "document.units", "params": {"units": "mm"}}]
        self.sk = {}                           # sketch name -> {"side", "z0"}
        self.zlo = self.zhi = None             # the part's extent so far, as ir.lower tracks it
        self.has_solid = False
        self.bitten = set()                    # sketches of tapered face-attached pads (see BITE)

    def add(self, command, **params):
        self.cmds.append({"command": command, "params": params})

    def op(self):
        op = "join" if self.has_solid else "new"
        self.has_solid = True
        return op

    # -- sketches --------------------------------------------------------------------------------
    def sketch_cmds(self, name, fr, loops):
        o, x, y, _ = fr
        self.add("sketch.create", plane={"origin": _r(o), "x_dir": _r(x), "y_dir": _r(y)}, name=name)
        for lp in loops:
            segs = lp["segments"]
            if segs[0][0] == "circle":
                self.add("sketch.circle.center", center=_r(segs[0][1]), diameter=round(2 * segs[0][2], 6))
                continue
            run = []                           # consecutive straight segments as one polyline
            for s in segs:
                if s[0] == "line":
                    if not run:
                        run.append(s[1])
                    run.append(s[2])
                else:
                    if len(run) >= 2:
                        self.add("sketch.line", points=[_r(p) for p in run])
                    run = []
                    self.add("sketch.arc.three_point", start=_r(s[1]), through=_r(s[2]), end=_r(s[3]))
            if len(run) >= 2:
                self.add("sketch.line", points=[_r(p) for p in run])
        self.add("sketch.finish")

    def f_sketch(self, f):
        on, side, z0 = f.get("on"), None, 0.0
        if on:
            side = on.get("side", "top")
            z0 = (self.zhi if side == "top" else self.zlo) or 0.0
            fr = G.frame((0, 0, z0 - SINK if side == "top" else z0 + SINK), (0, 0, 1), (1, 0, 0))
        elif f.get("plane", "XY") == "XZ":
            fr = G.XZ
        else:
            fr = G.XY
        self.sk[f["name"]] = {"side": side, "z0": z0}
        circles = [(c[0], c[1], c[2] + BITE) for c in f.get("circles", [])] \
            if f["name"] in self.bitten else f.get("circles", [])
        loops = G.sketch_loops(circles, f.get("rects", []), f.get("polys", []))
        self.sketch_cmds(f["name"], fr, loops)

    # -- solids ----------------------------------------------------------------------------------
    def f_pad(self, f):
        s, L = self.sk[f["sketch"]], float(f["length"])
        sunk = L + SINK if s["side"] else L
        kw = {"sketch": f["sketch"], "distance": _mm(sunk), "operation": self.op(), "name": f["name"]}
        if kw["operation"] == "new":
            kw["body_name"] = self.part
        if f.get("symmetric"):
            kw["direction"] = "symmetric"
        else:
            kw["direction"] = "negative" if s["side"] == "bottom" else "positive"
        if f.get("taper"):
            kw["taper"] = _deg(-f["taper"])    # IR positive shrinks; SolidCraft positive grows
        self.add("solid.extrude", **kw)
        z0 = s["z0"]
        a, b = (z0 - L / 2, z0 + L / 2) if f.get("symmetric") else \
            ((z0 - L, z0) if s["side"] == "bottom" else (z0, z0 + L))
        self.zlo = a if self.zlo is None else min(self.zlo, a)
        self.zhi = b if self.zhi is None else max(self.zhi, b)

    def f_pocket(self, f):
        s = self.sk[f["sketch"]]
        kw = {"sketch": f["sketch"], "operation": "cut", "name": f["name"]}
        if f["through"]:
            kw["through_all"], kw["direction"] = True, "symmetric"
        else:
            # `length` in from the face the sketch starts on, as every other backend reads it
            down = s["z0"] >= (self.zhi if self.zhi is not None else 0.0) - 1e-6
            depth = float(f["length"]) + (SINK if s["side"] else 0.0)
            kw["distance"], kw["direction"] = _mm(depth), "negative" if down else "positive"
            if f.get("taper"):
                kw["taper"] = _deg(-f["taper"])
        self.add("solid.extrude", **kw)

    def _edges(self, f):
        if f.get("picks") is None:
            raise NotImplementedError(f"{f['kind']} '{f['name']}': a {{\"circles\"}} query needs "
                                      f"live geometry; not in the SolidCraft backend yet")
        if not f["picks"]:
            raise RuntimeError(f"{f['kind']} '{f['name']}' selected no edges")
        return [_r(p) for p in f["picks"]]

    def f_fillet(self, f):
        self.add("solid.fillet", edges=self._edges(f), radius=_mm(f["radius"]), name=f["name"])

    def f_chamfer(self, f):
        self.add("solid.chamfer", edges=self._edges(f), distance=_mm(f["distance"]), name=f["name"])

    def f_revolve(self, f):
        kw = {"sketch": f["sketch"], "axis": "Z", "angle": _deg(f.get("angle", 360.0)),
              "operation": self.op(), "name": f["name"]}
        if kw["operation"] == "new":
            kw["body_name"] = self.part
        self.add("solid.revolve", **kw)

    def _placed_cut(self, name, fr, loops, depth, taper=0.0):
        self.sketch_cmds(name + "_sk", fr, loops)
        kw = {"sketch": name + "_sk", "distance": _mm(depth), "direction": "positive",
              "operation": "cut", "name": name}
        if taper:
            kw["taper"] = _deg(-taper)
        self.add("solid.extrude", **kw)

    def f_prism_cut(self, f):
        fr = G.frame(f["origin"], f["normal"], f["xdir"])
        self._placed_cut(f["name"], fr, G.sketch_loops(polys=f["polys"]), f["depth"], f.get("taper", 0.0))

    def f_polar_pocket(self, f):
        for i, (fr, loops, depth) in enumerate(G.polar_cuts(f)):
            self._placed_cut(f"{f['name']}_{i + 1}", fr, loops, depth)

    def build(self, spec):
        lowered = IR.lower(spec)["features"]
        attached = {f["name"] for f in lowered if f["kind"] == "sketch" and f.get("on")}
        self.bitten = {f["sketch"] for f in lowered
                       if f["kind"] == "pad" and f.get("taper") and f["sketch"] in attached}
        for f in lowered:
            getattr(self, "f_" + f["kind"])(f)
        return self.cmds


def emit(spec):
    """The IR as a SolidCraft command list (write {"commands": ...} and run it)."""
    return Emitter(spec["name"]).build(spec)


def run_script(script_path, out_path):
    """Run the script with solidcraft-cli, exporting `out_path` (format from its extension)."""
    r = subprocess.run([cli_path(), "run", str(script_path), "--out", str(out_path), "--quiet"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"solidcraft-cli failed:\n{r.stdout}\n{r.stderr}")
    return out_path


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ir")
    ap.add_argument("out", help="command script to write (JSON)")
    ap.add_argument("--run", metavar="FILE", help="run the CLI and export FILE (.step/.stl/.3mf)")
    ap.add_argument("--verify", action="store_true",
                    help="run the CLI, export STL, and gate the volume against build123d")
    a = ap.parse_args()
    spec = json.loads(Path(a.ir).read_text())
    cmds = emit(spec)
    Path(a.out).write_text(json.dumps({"commands": cmds}, indent=1))
    print(f"emitted {a.out}  ({len(cmds)} commands; run with: solidcraft-cli run {a.out} --out part.step)")
    if a.run:
        run_script(a.out, a.run)
        print(f"built {a.run}")
    if a.verify:
        import b3d_emit
        import trimesh
        stl = Path(a.out).with_suffix(".sc.stl")
        run_script(a.out, stl)
        v = float(trimesh.load(str(stl)).volume)
        ref = b3d_emit.emit(spec)[1]["volume"]
        ok = G.volume_check(v, ref, rel=5e-3)
        print(f"solidcraft volume {v:.1f} mm^3  build123d {ref:.1f} mm^3  "
              f"{'VERIFIED' if ok else 'DIFFERS'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
