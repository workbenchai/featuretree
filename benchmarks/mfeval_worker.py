"""One MFInstSeg eval worker: recognise ids[start::stride], append JSONL records, resume-safe.

Each model gets a hard SIGALRM timeout so a hung kernel boolean is a 'timeout' record, never a
dead sweep. Crashes are records too — quiddity's rule, adopted: nothing is silently dropped.
Usage: mfeval_worker.py IDS_FILE STEPS_DIR OUT_JSONL START STRIDE TIMEOUT_S
"""
import glob
import json
import os
import signal
import sys
import time

FEATURETREE = os.environ.get('FEATURETREE', '/Users/dnewcome/sandbox/dnewcome/featuretree')
sys.path.insert(0, FEATURETREE)

ids_file, steps_dir, out_path, start, stride, timeout = (
    sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))

import step_recognize  # noqa: E402  (imported once per worker)

ids = [ln.strip() for ln in open(ids_file) if ln.strip()]
done = set()
if os.path.exists(out_path):
    for ln in open(out_path):
        try:
            done.add(json.loads(ln)["id"])
        except Exception:
            pass

def _find(mid):
    for ext in (".step", ".stp", ".STEP"):
        p = os.path.join(steps_dir, mid + ext)
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join(steps_dir, "**", mid + ".st*p"), recursive=True)
    return hits[0] if hits else None

def _alarm(sig, frame):
    raise TimeoutError()

signal.signal(signal.SIGALRM, _alarm)
out = open(out_path, "a", buffering=1)

for i in range(start, len(ids), stride):
    mid = ids[i]
    if mid in done:
        continue
    rec = {"id": mid}
    path = _find(mid)
    if path is None:
        rec.update(status="missing")
        out.write(json.dumps(rec) + "\n")
        continue
    t0 = time.time()
    signal.alarm(timeout)
    try:
        spec, report = step_recognize.recognize(path)
        rec.update(status="ok",
                   verified=bool(report.get("verified")),
                   reason=(report.get("reason") or "")[:160],
                   dvol_pct=report.get("dvol_pct"),
                   features=len(spec["features"]) if spec else 0,
                   recovered=report.get("recovered"),
                   warnings=len(report.get("warnings") or []))
    except TimeoutError:
        rec.update(status="timeout")
    except Exception as e:
        rec.update(status="crash", error=f"{type(e).__name__}: {e}"[:200])
    finally:
        signal.alarm(0)
    rec["seconds"] = round(time.time() - t0, 3)
    out.write(json.dumps(rec) + "\n")
print(f"worker {start}/{stride} done")
