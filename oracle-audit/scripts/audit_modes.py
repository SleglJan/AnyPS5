"""Replay the execution tests' ALU kernels on the hardware oracle under several float modes and on the Vulkan hosts.

Usage: python3 -s scripts/oracle/audit_modes.py manifest.json out-dir [--only Name ...] [--hosts NVIDIA,Ryzen]

Per replayable kernel: extract the words and rows from the test source, run the whole kernel through the recompiler on each
host (KernelReplay harness), run the ALU words on the oracle in each mode, and classify every result cell.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TREE = ROOT / "AnyPS5"
HARNESS = ROOT / "build" / "tests" / "agc_driver_kernel_replay_tests"
ORACLE = TREE / "tools" / "hw-oracle" / "hw_oracle.py"
TESTS = TREE / "core" / "libs" / "prx" / "libSceAgcDriver" / "tests" / "execution"

MODES = {
    "title": dict(ieee=0, denorm32=0, denorm16=3, dx10=1, round32=0, round16=0, fp16ovfl=0),
    "old": dict(ieee=1, denorm32=3, denorm16=3, dx10=1, round32=0, round16=0, fp16ovfl=0),
    "ieee0_keep32": dict(ieee=0, denorm32=3, denorm16=3, dx10=1, round32=0, round16=0, fp16ovfl=0),
    "ieee1_flush32": dict(ieee=1, denorm32=0, denorm16=3, dx10=1, round32=0, round16=0, fp16ovfl=0),
    "title_flush16": dict(ieee=0, denorm32=0, denorm16=0, dx10=1, round32=0, round16=0, fp16ovfl=0),
    "title_noclamp": dict(ieee=0, denorm32=0, denorm16=3, dx10=0, round32=0, round16=0, fp16ovfl=0),
    "title_fp16ovfl": dict(ieee=0, denorm32=0, denorm16=3, dx10=1, round32=0, round16=0, fp16ovfl=1),
}

HEX_ROW = re.compile(r"^[0-9a-f]{8}( [0-9a-f]{8})*$")


def words_of(text):
    return [int(w, 16) for w in re.findall(r"0x([0-9a-fA-F]{8})u?", text)]


def constant(src, name, fallback):
    m = re.search(r"constexpr std::uint32_t " + name + r" = (\d+);", src)
    return int(m.group(1)) if m else fallback


def parse_rows(src, entry):
    m = re.search(r"Rows\s*\{\{(.*?)\}\};", src, re.S)
    if m:
        return [words_of(line) for line in m.group(1).split("\n") if "0x" in line]
    m = re.search(r"(?:constexpr\s+)?(?:const\s+)?std::uint32_t\s+Rows\s*\[\s*\d*\s*\]\s*\[\s*(\d+)\s*\]\s*=\s*\{(.*?)\};", src, re.S)
    if m:
        width = int(m.group(1))
        groups = re.findall(r"\{([^{}]*)\}", m.group(2))
        rows = [words_of(g) for g in groups]
        if rows and all(len(r) == width for r in rows):
            return rows
        raise RuntimeError(f"{entry['file']}: Rows[][{width}] parse gave widths {sorted(set(len(r) for r in rows))}")
    m = re.search(r"std::array<\s*Row\s*,\s*\d+>\s*Rows\s*\{\{(.*?)\}\};", src, re.S)
    if m:
        groups = re.findall(r"\{([^{}]*)\}", m.group(1))
        return [words_of(g) for g in groups if "0x" in g]
    raise RuntimeError(f"{entry['file']}: no Rows table ({entry.get('rows_source')})")


def extract(entry, kernel):
    src = (TESTS / entry["file"]).read_text()
    m = re.search(r"std::array<std::uint32_t,\s*\d+>\s*" + re.escape(kernel["name"]) + r"\s*\{(.*?)\};", src, re.S)
    if not m:
        raise RuntimeError(f"{entry['file']}: kernel {kernel['name']} not found")
    code = words_of(m.group(1))
    inputs = constant(src, "Inputs", 8)
    results = constant(src, "Results", 8)
    used = min(kernel.get("outputs") or results, results, 16)
    threads = constant(src, "Threads", 32)
    rows = parse_rows(src, entry)
    if kernel.get("alu_word_range"):
        start, end = kernel["alu_word_range"]
    else:
        start = next(i for i, w in enumerate(code) if (w >> 16) == 0xbf8c) + 1
        end = next(i for i, w in enumerate(code) if (w & 0xfff00000) == 0xe0700000)
    return code, code[start:end], rows, inputs, results, used, threads


def host_run(gpu, code, rows, inputs, results, threads, work, wave64=False):
    (work / "code.txt").write_text("\n".join(f"0x{w:08x}" for w in code) + "\n")
    (work / "rows-host.txt").write_text("".join(" ".join(f"0x{v:08x}" for v in (r + [0] * inputs)[:inputs]) + "\n" for r in rows))
    env = dict(os.environ, REPLAY_CODE=str(work / "code.txt"), REPLAY_ROWS=str(work / "rows-host.txt"), REPLAY_INPUTS=str(inputs),
               REPLAY_RESULTS=str(results), REPLAY_THREADS=str(max(threads, len(rows))), REPLAY_WAVE="64" if wave64 else "32")
    if gpu:
        env["ANYPS5_GPU"] = gpu
    p = subprocess.run([str(HARNESS)], env=env, capture_output=True, text=True, timeout=300)
    out = [[int(x, 16) for x in l.split()] for l in p.stdout.splitlines() if HEX_ROW.match(l.strip())]
    if p.returncode != 0 or len(out) != len(rows):
        return None, (p.stderr.strip().splitlines() or p.stdout.strip().splitlines() or ["no output"])[-1][:200]
    return out, ""


def oracle_run(mode, alu, rows, results, work, wave64=False):
    (work / "body.s").write_text("".join(f"  .long 0x{w:08x}\n" for w in alu))
    (work / "rows-oracle.txt").write_text("".join(" ".join(f"0x{v:08x}" for v in (r + [0] * 4)[:4]) + "\n" for r in rows))
    m = MODES[mode]
    cmd = [sys.executable, "-s", str(ORACLE), "--ieee", str(m["ieee"]), "--denorm32", str(m["denorm32"]), "--denorm16", str(m["denorm16"]),
           "--dx10-clamp", str(m["dx10"]), "--round32", str(m["round32"]), "--round16", str(m["round16"]), "--fp16-overflow", str(m["fp16ovfl"]),
           "--outs", str(min(results, 16)), str(work / "body.s"), str(work / "rows-oracle.txt")] + (["--wave64"] if wave64 else [])
    p = subprocess.run(cmd, cwd=TREE, capture_output=True, text=True, timeout=600)
    out = [[int(x, 16) for x in l.split()] for l in p.stdout.splitlines() if HEX_ROW.match(l.strip())]
    if p.returncode != 0 or len(out) != len(rows):
        return None, (p.stderr.strip().splitlines() or ["no output"])[-1][:200]
    return out, ""


def is_nan(v):
    return (v & 0x7fffffff) > 0x7f800000


def same(x, y):
    return x == y


def nan_equal(x, y):
    return x == y or (is_nan(x) and is_nan(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("out")
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--hosts", default="NVIDIA,Ryzen")
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--classes", default="yes")
    args = ap.parse_args()
    manifest = json.load(open(args.manifest))
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    hosts = [h for h in args.hosts.split(",") if h]
    modes = [m for m in args.modes.split(",") if m]
    report = []
    for entry in manifest:
        name = Path(entry["file"]).stem
        if args.only and name not in args.only:
            continue
        for kernel in entry.get("kernels", []):
            if kernel.get("replayable") not in args.classes.split(","):
                continue
            item = {"file": entry["file"], "kernel": kernel["name"], "replayable": kernel["replayable"], "areas": entry.get("areas", [])}
            work = out / name / kernel["name"]
            work.mkdir(parents=True, exist_ok=True)
            try:
                code, alu, rows, inputs, results, used, threads = extract(entry, kernel)
            except Exception as error:
                item["error"] = f"extract: {error}"
                report.append(item)
                print(f"{name}/{kernel['name']}: {item['error']}", flush=True)
                continue
            item.update(rows=len(rows), inputs=inputs, results=results, used=used, alu_words=len(alu))
            if inputs > 4 or results > 16:
                item["note"] = f"oracle rows carry 4 inputs and 16 outputs; this kernel uses {inputs}/{results}"
            host_out = {}
            for gpu in hosts:
                values, error = host_run(gpu, code, rows, inputs, results, threads, work, bool(kernel.get("wave64_variant")))
                host_out[gpu] = values
                if error:
                    item.setdefault("host_errors", {})[gpu] = error
            oracle_out = {}
            for mode in modes:
                values, error = oracle_run(mode, alu, rows, used, work, bool(kernel.get("wave64_variant")))
                oracle_out[mode] = values
                if error:
                    item.setdefault("oracle_errors", {})[mode] = error
            cells = []
            columns = used
            for r in range(len(rows)):
                for c in range(columns):
                    cell = {"row": r, "col": c, "inputs": [f"{v:08x}" for v in rows[r]]}
                    o = {m: f"{oracle_out[m][r][c]:08x}" for m in modes if oracle_out.get(m)}
                    h = {g: f"{host_out[g][r][c]:08x}" for g in hosts if host_out.get(g)}
                    cell["oracle"] = o
                    cell["hosts"] = h
                    flags = []
                    ovals = [int(v, 16) for v in o.values()]
                    if ovals and any(not nan_equal(v, ovals[0]) for v in ovals):
                        flags.append("mode-dependent")
                    elif ovals and any(not same(v, ovals[0]) for v in ovals):
                        flags.append("mode-dependent-nan-payload")
                    hvals = [int(v, 16) for v in h.values()]
                    if hvals and any(not nan_equal(v, hvals[0]) for v in hvals):
                        flags.append("hosts-differ")
                    if ovals and hvals and "title" in o:
                        t = int(o["title"], 16)
                        if any(not nan_equal(v, t) for v in hvals):
                            flags.append("recompiler-differs-from-hardware-title-mode")
                        elif any(not same(v, t) for v in hvals):
                            flags.append("nan-payload-only")
                    if any(v == 0xeeeeeeee for v in ovals):
                        flags.append("oracle-unwritten")
                    cell["flags"] = flags
                    if flags:
                        cells.append(cell)
            item["flagged_cells"] = cells
            item["cells"] = len(rows) * columns
            report.append(item)
            summary = {}
            for cell in cells:
                for f in cell["flags"]:
                    summary[f] = summary.get(f, 0) + 1
            print(f"{name}/{kernel['name']}: {len(rows)} rows x {columns} cols; flagged {summary or 'none'}" + (f"; errors host={item.get('host_errors')} oracle={item.get('oracle_errors')}" if item.get('host_errors') or item.get('oracle_errors') else ""), flush=True)
    json.dump(report, open(out / "audit.json", "w"), indent=1)
    print(f"wrote {out / 'audit.json'} ({len(report)} kernels)")


if __name__ == "__main__":
    main()
