# Backlog

Items here graduate to GitHub issues once Issues are enabled on this repo
(Settings → General → Features → Issues — needs an org admin).

## Expand feature trees to full assemblies and products, not just single parts

**Motivation.** featuretree today recovers an editable feature tree for ONE part. Real
products are assemblies: the AmazonBasics screwdriver scan is physically a plastic handle +
ratchet collar, a chromed bit holder with a ¼″ hex socket, and a retention magnet — and the
pipeline currently fuses all of it into a single solid, with the material seam surfacing only
as the "torn band" honesty gate. The goal: recover an entire product as an ASSEMBLY of parts,
each with its own named, editable feature tree, plus the structure that relates them.

**What already exists to build on.**
- `assembly_ir.py` — occurrences + per-part trees + typed mates/couplings/poses
  (cadgen-compatible), loop closures, and a pose solver (`assembly_kin.py`)
- `cadgen_ingest.py` — STEP + sidecar → assembly IR with a verified editable tree per part
- `mjcf_emit.py` / `threemf_emit.py` — assembly consumers (MuJoCo; 3MF with embedded IR)
- Scan recognition (`mesh_compare.py`, `sdf_recognize.py`) — single-part today; its
  torn-band / material-change detection is in effect a part-boundary detector already

**Proposed scope.**
1. *Scan → multi-part segmentation*: split a scanned product into part candidates (material
   seams / torn bands, rigid-motion evidence, interior cavities like the socket + magnet,
   symmetry breaks), each recognized in its own frame with its own tree.
2. *Seam modelling*: parts meeting at a seam get honest mating geometry (the bit holder's
   stem into the handle), flagged as inferred where unscanned.
3. *Mates from function*: axis-coincident parts get revolute/cylindrical mates (the ratchet
   shaft spins in the handle); fastened as the default.
4. *Assembly emission*: a native FreeCAD assembly (links + joints) and an Onshape assembly
   from the assembly IR, per-part trees staying editable inside it (the README already lists
   this as "next step, not yet built" for cadgen ingest).
5. *Product round-trip*: edits to any part re-verify the assembly (interference check via the
   existing Boolean-free IoU machinery).

**Acceptance sketch.** The screwdriver scan recovers as a 3-part assembly (handle, bit
holder, magnet) with per-part VERIFIED trees, a revolute mate on the shaft axis, and opens in
FreeCAD as an assembly whose parts are individually editable.
