"""Aggregate MFInstSeg eval JSONL(s) into the quiddity-comparable summary."""
import glob
import json
import sys

import numpy as np

recs = {}
for path in sys.argv[1:] or glob.glob("mfeval_*.jsonl"):
    for ln in open(path):
        try:
            r = json.loads(ln)
            recs[r["id"]] = r
        except Exception:
            pass
rs = list(recs.values())
n = len(rs)
by = lambda s: [r for r in rs if r.get("status") == s]
ok, crash, tmo, miss = by("ok"), by("crash"), by("timeout"), by("missing")
ver = [r for r in ok if r.get("verified")]
print(f"models scored: {n}")
print(f"  ok: {len(ok)}  ({len(ver)} VERIFIED = {100*len(ver)/max(n,1):.2f}% of all, "
      f"{100*len(ver)/max(len(ok),1):.2f}% of ok)")
print(f"  crash: {len(crash)}  timeout: {len(tmo)}  missing: {len(miss)}")
if ok:
    t = np.array([r["seconds"] for r in ok])
    print(f"  per-model wall: median {np.median(t):.2f}s  p95 {np.percentile(t,95):.2f}s  "
          f"max {t.max():.1f}s")
    d = np.array([abs(r["dvol_pct"]) for r in ver if r.get("dvol_pct") is not None])
    if len(d):
        print(f"  VERIFIED |dvol%|: median {np.median(d):.4f}  p95 {np.percentile(d,95):.4f}")
    f = np.array([r.get("features") or 0 for r in ok])
    print(f"  features/model: median {np.median(f):.0f}  p95 {np.percentile(f,95):.0f}")
from collections import Counter
reasons = Counter((r.get("reason") or "").split(":")[0][:60] for r in ok if not r.get("verified"))
print("top not-verified reasons:")
for k, v in reasons.most_common(8):
    print(f"  {v:6d}  {k or '(none)'}")
errs = Counter((r.get("error") or "")[:60] for r in crash)
for k, v in errs.most_common(5):
    print(f"  crash {v:5d}  {k}")
