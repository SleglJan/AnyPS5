"""Audit every title of the PS5 homebrew catalog against the current AnyPS5 build, on Linux.

Phase 1 (default): download each release ZIP (sha256 from the catalog record), extract, unwrap the SELF
containers, relink with --registry, run tools/import_audit.py. Nothing is executed.
Phase 2 (--run SECONDS): for titles whose audit found no absent import and no missing library, run the
relinked executable in the bubblewrap sandbox for SECONDS and record whether it survives.

Usage: python3 -I scripts/audit_catalog.py [--run SECONDS] [--only PPSA...]
Writes notes/catalog-audit.json and notes/catalog-audit.md.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG = ROOT / "homebrew" / "catalog"
HOMEBREW = ROOT / "homebrew"
RUNS = ROOT / "runs" / "catalog"
NOTES = ROOT / "notes"
BUILD = Path(os.environ.get("ANYPS5_BUILD", ROOT / "build"))
RELINKER = BUILD / "core" / "relinker" / "relinker"
LIBS = BUILD / "core" / "libs" / "libs"
IMPORT_AUDIT = ROOT / "AnyPS5" / "tools" / "import_audit.py"
UNFSELF = ROOT / "scripts" / "unfself.py"
SANDBOX = ROOT / "scripts" / "run-sandboxed.sh"
NAMES = HOMEBREW / "nid-db" / "aerolib.csv"
MODULE_DIRS = ("sce_module", "sce_modules", "prx")
SELF_MAGICS = (b"\x4f\x15\x3d\x1d", b"\x54\x14\xf5\xee")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(record, dest):
    if dest.exists() and sha256(dest) == record["sha256"]:
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(record["artifact_url"], timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    if sha256(dest) != record["sha256"]:
        dest.unlink()
        raise RuntimeError("sha256 mismatch")
    return "downloaded"


def extract(zip_path, dest):
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = info.filename
            if name.startswith("/") or ".." in Path(name).parts:
                raise RuntimeError(f"unsafe zip entry {name}")
            z.extract(info, dest)
    eboots = list(dest.rglob("eboot.bin"))
    if len(eboots) != 1:
        raise RuntimeError(f"{len(eboots)} eboot.bin in zip")
    return eboots[0].parent


def unwrap(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as f:
        magic = f.read(4)
    if magic in SELF_MAGICS:
        subprocess.run([sys.executable, "-I", str(UNFSELF), str(src), str(dst)], check=True,
                       capture_output=True, text=True)
        return "self"
    shutil.copyfile(src, dst)
    return "elf"


def prepare(title_dir, unw):
    if unw.exists():
        shutil.rmtree(unw)
    unw.mkdir(parents=True)
    kinds = {"eboot": unwrap(title_dir / "eboot.bin", unw / "input.elf")}
    modules = []
    for d in MODULE_DIRS:
        src = title_dir / d
        if src.is_dir():
            for m in sorted(src.iterdir()):
                if m.is_file():
                    unwrap(m, unw / d / m.name)
                    modules.append(f"{d}/{m.name}")
    kinds["modules"] = modules
    return kinds


def relink(unw, run):
    if run.exists():
        shutil.rmtree(run)
    (run / "app0").mkdir(parents=True)
    (run / "download0").mkdir()
    os.symlink(os.path.relpath(LIBS, run), run / "libs")
    p = subprocess.run([str(RELINKER), "--registry", str(unw / "input.elf"), str(run / "app.elf")],
                       capture_output=True, text=True)
    guest = re.findall(r"Guest module: (.+)", p.stdout)
    return p.returncode, (p.stdout + p.stderr).strip().splitlines()[-1] if p.returncode else "", guest


def audit(run, unw):
    cmd = [sys.executable, "-s", str(IMPORT_AUDIT), str(run / "app.registry.json"), "--libs", str(LIBS),
           "--json", str(run / "audit.json")]
    if NAMES.exists():
        cmd += ["--names", str(NAMES)]
    for d in MODULE_DIRS:
        if (unw / d).is_dir():
            cmd += ["--modules", str(unw / d)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if not (run / "audit.json").exists():
        return {"error": (p.stdout + p.stderr).strip().splitlines()[-1:]}
    a = json.load(open(run / "audit.json"))
    by = a.get("unique_by_class", {})
    absent = [(i["library"], i.get("name") or i["nid"]) for i in a["imports"] if i["class"] == "absent"]
    stubs = [(i["library"], i.get("name") or i["nid"]) for i in a["imports"] if i["class"] == "stub"]
    return {"exit": p.returncode, "implemented": by.get("implemented", 0), "stub": by.get("stub", 0),
            "absent": by.get("absent", 0), "module": by.get("module", 0),
            "absent_list": absent, "stub_list": stubs,
            "missing_libraries": a.get("missing_libraries", []), "library_mismatch": a.get("library_mismatch", [])}


def link_resources(title_dir, run):
    for entry in title_dir.iterdir():
        if entry.name in ("eboot.bin",) + MODULE_DIRS:
            continue
        target = run / "app0" / entry.name
        if not target.exists():
            os.symlink(entry.resolve(), target)


def run_title(run, seconds):
    p = subprocess.run(["bash", str(SANDBOX), str(run), str(seconds)], capture_output=True, text=True, timeout=seconds + 60)
    out = p.stdout
    log = (run / "run.log").read_text(errors="replace") if (run / "run.log").exists() else ""
    windows = re.findall(r"window 0x[0-9a-f]+: name='([^']*)'", out)
    fail = next((l for l in log.splitlines() if re.search(r"FAIL|runtime_error|terminate called|what\(\)|Segmentation|Aborted", l)), "")
    m = re.search(r"exited early with code (\d+)", out)
    return {"survived": "still running" in out, "exit_code": int(m.group(1)) if m else None,
            "windows": windows, "first_failure": fail[:300], "log_lines": len(log.splitlines()),
            "screenshot": (run / "screenshot.png").exists()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=0, metavar="SECONDS")
    ap.add_argument("--only", nargs="*", default=[])
    args = ap.parse_args()
    results = {}
    prev = json.load(open(NOTES / "catalog-audit.json")) if (NOTES / "catalog-audit.json").exists() else {}
    for rec_path in sorted(CATALOG.glob("*.json")):
        rec = json.load(open(rec_path))
        tid = rec["titleid"]
        if args.only and tid not in args.only:
            results[tid] = prev.get(tid, {})
            continue
        r = {"name": rec.get("name"), "kind": rec.get("kind"), "version": rec.get("version"),
             "source": rec.get("source_repo"), "license": rec.get("license")}
        results[tid] = r
        print(f"== {tid} {rec.get('name')}", flush=True)
        if not rec.get("artifact_url") or not rec.get("sha256"):
            r["error"] = "no release artifact in the catalog record"
            continue
        try:
            zip_path = HOMEBREW / tid / f"{tid}.zip"
            r["download"] = download(rec, zip_path)
            r["zip_mb"] = round(zip_path.stat().st_size / 1e6, 1)
            title_dir = extract(zip_path, HOMEBREW / tid / "extracted")
            unw = HOMEBREW / tid / "unwrapped"
            r["containers"] = prepare(title_dir, unw)
            run = RUNS / tid
            code, err, guest = relink(unw, run)
            r["relink"] = "ok" if code == 0 else f"exit {code}: {err}"
            r["guest_modules"] = [Path(g).name for g in guest]
            if code != 0:
                continue
            r["audit"] = audit(run, unw)
            if args.run and r["audit"].get("absent") == 0 and not r["audit"].get("missing_libraries"):
                link_resources(title_dir, run)
                r["run"] = run_title(run, args.run)
                print(f"   run: {'alive' if r['run']['survived'] else 'exit ' + str(r['run']['exit_code'])} {r['run']['windows']} {r['run']['first_failure'][:100]}", flush=True)
        except Exception as e:  # noqa: BLE001 - record and continue with the next title
            r["error"] = f"{type(e).__name__}: {e}"
            print(f"   error: {r['error']}", flush=True)
    NOTES.mkdir(exist_ok=True)
    json.dump(results, open(NOTES / "catalog-audit.json", "w"), indent=1)
    write_markdown(results)


def write_markdown(results):
    lines = ["# Catalog audit on Linux", "",
             "| Title | Kind | Relink | Implemented | Stub | Absent | Missing libs | Run |", "|---|---|---|---:|---:|---:|---|---|"]
    for tid, r in sorted(results.items(), key=lambda kv: (kv[1].get("kind") or "", kv[0])):
        a = r.get("audit", {})
        run = r.get("run")
        run_s = "" if not run else ("alive, " + (run["windows"][0] if run["windows"] else "no window")) if run["survived"] else f"exit {run['exit_code']}: {run['first_failure'][:60]}"
        rel = r.get("relink", r.get("error", ""))
        lines.append(f"| {tid} {r.get('name','')} | {r.get('kind','')} | {rel[:60]} | {a.get('implemented','')} | {a.get('stub','')} | {a.get('absent','')} | {', '.join(a.get('missing_libraries', []))[:60]} | {run_s} |")
    lines += ["", "## Absent imports per title", ""]
    for tid, r in sorted(results.items()):
        a = r.get("audit", {})
        if a.get("absent_list"):
            by = {}
            for lib, name in a["absent_list"]:
                by.setdefault(lib, []).append(name)
            lines.append(f"**{tid} {r.get('name')}** ({a['absent']})")
            for lib, names in sorted(by.items()):
                lines.append(f"- `{lib}`: {', '.join(sorted(names))}")
            lines.append("")
    (NOTES / "catalog-audit.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
