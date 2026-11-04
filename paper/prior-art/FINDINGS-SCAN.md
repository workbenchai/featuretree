# Prior-art scan — scan recognition: displacement map, deviation metric, detailing-as-features

**Date:** 2026-11-03 · **Scope:** the scan-side additions of Oct–Nov 2026 (`displacement()`,
`deviation()`, relief/engraving/washboard/pattern recovery in `mesh_compare.py` /
`sdf_recognize.py`), i.e. the scan → editable-tree pipeline — NOT the STEP → tree pipeline,
which [FINDINGS.md](FINDINGS.md) already covers. · **Method:** targeted literature web scan,
one afternoon. No OpenAlex citation-graph traversal yet, and **no patent search** — both of
which the 2026-09-19 scan had. Treat verdicts here as provisional until those are done.

**Not legal advice.**

---

## 1. Executive verdict

**Do not claim novelty for the deviation metric, the harmonic baseline, the base-plus-height
decomposition, or the cylindrical unwrap — each is anticipated, two of them by decades.**

**What appears to be genuine white space is one combination:** automatically turning a noisy
consumer-product scan's *surface detailing* into named, parametric, EDITABLE CAD features —
an engraved line as a path with one width and one depth, a ridge field as one ridge times a
pattern (count, step), a ring of pads as one boss times 4 @ 90° — with measured
VERIFIED/PARTIAL-style verdicts and refusal gates instead of silent best-effort. Nothing
found does that end-to-end. This is the same shape of claim FINDINGS.md §3.1 defends for the
STEP side: not a method claim, an **availability-and-honesty** claim, now extended to scans.

Same recommendation as before: do not file; publish, if at all, as a tools/experience
contribution that cites its ancestors; ship.

## 2. What is definitively closed

| Claim | Killed by |
|---|---|
| One-sided scan→CAD point distance as the acceptance metric, with colormap | Standard inspection practice in every shipping tool: Geomagic Control X "3D Compare" / Accuracy Analyzer (already in FINDINGS.md), CloudCompare cloud-to-mesh distance, Leica Cyclone 3DR Compare/Inspect. Our `deviation()` is this metric, nothing more. |
| Per-cross-section harmonic decomposition of radial deviation; first harmonic = eccentricity | Classical roundness metrology: **ISO 12181** (undulations per revolution, Fourier roundness filtering, cut-offs in UPR), ISO 4291:1985. The k = 1 term absorbing the axis offset is the textbook eccentricity fact. |
| Decomposing a surface into a base surface plus a robust height function to separate relief from body | **Zatzarinni, Tal & Shamir, "Relief Analysis and Extraction," ACM TOG / SIGGRAPH Asia 2009** — the same decomposition, with robustness emphasized, 17 years ago. |
| Unwrapping a revolved scan into a 2D domain to work on it | Shipping product: **Geomagic Design X "Unroll/Reroll" (2020)** plus its Revolution / Extraction wizards; routine in inspection imaging (unrolled sewer-pipe imagery for defect detection, arXiv:1912.05222, and kin). |
| Mesh/point-cloud → editable sketch-extrude construction sequence, with geometric verification feedback | The current wave: **Point2Cyl** (CVPR 2022), **CAD-SIGNet** (2024), **Point2Skh** (CAD 2024), **Point2CAD** (CVPR 2024, B-rep not history), **CADFit** (2026 — per-operation IoU feedback over extrude/revolve/fillet/chamfer sequences). |
| Repeated-feature ("pattern") recovery as design intent | CAD-driven pattern recognition in reverse-engineered models (GRAPP; linear repetitions, explicitly design-intent framed); bolt-circle recovery is routine AFR and appears again in the 2026 design-intent-aware reverse-parametrization framework (arXiv:2611.00777 — B-rep input, verification sweeps, bolt circles). |
| Segmenting relief — including PERIODIC relief — from a scanned mesh for reverse engineering | The Cardiff line: **Liu, Martin, Langbein & Rosin — "Segmenting Reliefs on Triangle Meshes" (SPM 2006), "Segmenting Periodic Reliefs on Triangle Meshes" (Mathematics of Surfaces 2007), "Background Surface Estimation for Reverse Engineering of Reliefs" (Int. J. CAD/CAM 2007)**. Periodic relief segmentation is our washboard, nineteen years early. |

## 3. Closest lineage, and the deltas

The ancestors any write-up must cite, and what they do not do:

- **Cardiff relief papers (2006–07).** Segment reliefs, including periodic ones, from scanned
  meshes, explicitly for reverse engineering, with background-surface estimation. **Deltas:**
  user-assisted (seed points, user-drawn contours, snakes) where ours is automatic; output is
  a relief geometry patch to re-apply, not parametric features (no path + width + depth, no
  pattern count × step, no named editable operations); no per-part verdict or refusal gates;
  single-relief focus rather than a whole-part tree.
- **Zatzarinni/Tal/Shamir 2009.** The base + height decomposition, robustly. **Deltas:** the
  height field is the end product; nothing is parameterized, patterned, or emitted as CAD.
- **Design X Unroll/Reroll + wizards.** The unwrap and revolved-part workflow, in product.
  **Deltas:** interactive wizard workflow; no automatic relief/detail segmentation on the
  unwrap, no parametric detailing features, no measured verdict; and (per FINDINGS.md §3.1)
  a proprietary seat, not a callable library.
- **The neural mesh→sequence line (Point2Cyl … CADFit).** Editable sketch-extrude solids with
  verification. **Deltas:** trained on clean CAD-like inputs; recover the SOLID's history,
  not the surface detailing; no engraving / texture / washboard vocabulary; no honesty gates
  for torn or unresolvable scan regions.

## 4. What survives

> **No published system or shipping product found recovers a scanned moulded product's surface
> detailing — engraved lines, embossed ridges, periodic ridge fields, repeated pads — as
> named, parametric, editable CAD features (a path with one width and one depth; one ridge ×
> pattern(count, step); one boss × polar count), automatically, with measured verdicts and
> explicit refusal where the scan cannot support a claim (torn bands, end-face artifacts,
> phase-incoherent texture). featuretree does, in an MIT-licensed callable library.**

Load-bearing qualifiers, as in FINDINGS.md: *detailing as parametric features* (Cardiff
segments, we parameterize), *automatic* (their methods are seeded by a user), *measured
verdict + refusal* (the honesty argument, extended from STEP to scans — e.g. the
phase-coherence test that refused to invent a knurl pattern), *callable library* (every
near-neighbor is a GUI seat or a paper without code for this task).

Secondary, weaker deltas (worth a sentence each, not a claim): the harmonic baseline as a
full z-stacked SURFACE fitted by L1 (roundness metrology is per-section; composing sections
into a feature-rejecting baseline surface for segmentation is an engineering extension with
an obviousness flavour); the split-invariant washboard measurement (peak-to-valley + top
position, immune to where the baseline bisects a 50%-duty texture).

## 5. Caveats and next steps

- **No patent search was done.** The 2026-09-19 scan found the method side of STEP-recovery
  patented-and-expired; the scan side (Design X's inspection and wizard patents, 3D Systems /
  Oqton portfolio) is unexamined. Required before any commercial positioning.
- **No OpenAlex traversal.** Seed it with the Cardiff trio + Zatzarinni and walk citing works;
  that is where a closer anticipation would hide (2010s relief-reverse-engineering follow-ups,
  Chinese-language CAD/CAM literature).
- Read the Cardiff full texts (only abstracts/snippets were used here) before writing any
  "they do not parameterize" sentence in public.
- The 2026 arXiv items (CADFit 2605.01171, design-intent framework 2611.00777) are fresh and
  moving; re-check at publication time.

## Sources

- Zatzarinni, Tal, Shamir — Relief Analysis and Extraction: <https://dl.acm.org/doi/10.1145/1618452.1618482>
- Liu, Martin, Langbein, Rosin — Segmenting Reliefs: <https://users.cs.cf.ac.uk/Paul.Rosin/resources/papers/relief-SPM.pdf>
- Liu et al. — Segmenting Periodic Reliefs: <https://orca.cardiff.ac.uk/id/eprint/5215/1/PeriodicReliefs.pdf>
- ISO 12181-1/-2 (roundness, UPR): <https://standards.iteh.ai/catalog/standards/iso/f8ff9de2-b5cd-401b-a2bf-8e7059ca4a3d/iso-12181-1-2011>
- Geomagic Design X 2024.2 release notes (Unroll/Reroll, wizards): <https://oqton.com/posts/whats-new-geomagic-designx/>
- Geomagic Control X 3D Compare: <https://au.3dsystems.com/events/geomagic-controlx-3dcompare-colorbar-annotations>
- CloudCompare C2M workflow: <https://cloudcompare.org/forum/viewtopic.php?p=9016>
- Leica Cyclone 3DR Compare/Inspect: <https://cyclone3dr.leica-geosystems.com/help/2024.0/CompareInspectCloudC.html>
- Point2Cyl: <https://arxiv.org/pdf/2112.09329> · CAD-SIGNet: <https://arxiv.org/pdf/2402.17678>
- Point2Skh: <https://www.sciencedirect.com/science/article/pii/S0010448524001659>
- Point2CAD (CVPR 2024): <https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Point2CAD_Reverse_Engineering_CAD_Models_from_3D_Point_Clouds_CVPR_2024_paper.pdf>
- CADFit (2026): <https://arxiv.org/abs/2605.01171>
- Design-intent-aware reverse parametrization (2026): <https://arxiv.org/html/2611.00777>
- CAD-driven pattern recognition (GRAPP): <https://www.lirmm.fr/~subsol/WWW/GRAPP.0219.1.pdf>
- Unrolled pipe inspection imagery: <https://arxiv.org/pdf/1912.05222>
