"""Run relinked catalog titles in the sandbox and print their first loader stop. Usage: run_stops.py <seconds> <titleid>..."""
import importlib.util
import re
import sys

spec = importlib.util.spec_from_file_location("audit_catalog", "/home/slegl/PycharmProjects/anyps5/scripts/audit_catalog.py")
ac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ac)

seconds = int(sys.argv[1])
pattern = re.compile(r"FAIL|runtime_error|terminate called|what\(\)|Segmentation|Aborted|unresolved|absent|missing|not implemented|NotImplemented|Unknown NID|import", re.I)


def run_one(tid):
    run = ac.RUNS / tid
    extracted = ac.HOMEBREW / tid / "extracted"
    title_dir = next((p.parent for p in extracted.rglob("eboot.bin")), None) if extracted.is_dir() else None
    if title_dir is None or not (run / "app.elf").exists():
        print(f"== {tid}: missing extracted title or run dir", flush=True)
        return
    ac.link_resources(title_dir, run)
    r = ac.run_title(run, seconds)
    log = (run / "run.log").read_text(errors="replace").splitlines() if (run / "run.log").exists() else []
    hits = [l for l in log if pattern.search(l)][:4]
    print(f"== {tid} survived={r['survived']} exit={r['exit_code']} windows={r['windows']} log_lines={r['log_lines']}", flush=True)
    for l in hits:
        print(f"   hit: {l[:220]}", flush=True)
    for l in log[-2:]:
        print(f"   tail: {l[:220]}", flush=True)


for tid in [t for arg in sys.argv[2:] for t in arg.split()]:
    try:
        run_one(tid)
    except Exception as e:  # noqa: BLE001 - report and continue with the next title
        print(f"== {tid}: error {type(e).__name__}: {e}", flush=True)
