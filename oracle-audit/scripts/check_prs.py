"""Cross-check the recompiler of a branch against the oracle in the modes #1866 (DX10_CLAMP) and #1867 (IEEE quieting) add.

Usage: python3 -s scripts/oracle/check_prs.py out-dir [file.cpp ...]
Runs every kernel of the DX10- and IEEE-sensitive tests of both audit passes through build/tests/agc_driver_kernel_replay_tests
(REPLAY_MODE = float_mode,dx10_clamp,ieee_mode,fp16_ovfl; empty = no mode) on both hosts in five modes, and compares each
cell bit for bit with the oracle in the same mode. Oracle values: pass 1 kernels are re-measured unless the saved run's words
and rows are identical; pass 2 kernels use the saved adapter values (notes/oracle-audit-run2/values.json).
"""
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_modes as am  # noqa: E402

ROOT = am.ROOT
RUN1 = ROOT / "notes" / "oracle-audit-run1"
RUN2 = ROOT / "notes" / "oracle-audit-run2"
HOST_MODES = {"none": "", "title": "0xc0,1,0,0", "title_noclamp": "0xc0,0,0,0", "ieee1_flush32": "0xc0,1,1,0", "old": "0xf0,1,1,0"}
ORACLE_OF = {"none": "title", "title": "title", "title_noclamp": "title_noclamp", "ieee1_flush32": "ieee1_flush32", "old": "old"}
GPUS = ["NVIDIA", "Ryzen"]
PASS1 = ["DivisionOutputModifiers", "DivisionResultModifiers", "F32OutputModifier", "F32ResultClamp", "F64Transcendental", "Float16Modifiers",
         "Float16TernaryClamp", "SdwaLdexpF16", "Vop2ModifierForms", "Vop3CubeClamp", "DivFixupF16", "F32MinMaxNan", "F64Arithmetic",
         "F64Conversions", "F64Division", "F64DivisionModifiers", "F64NanConversions", "F64Rounding", "SdwaFloatSelectors", "Vop3PackLdexp"]
PASS2 = ["Vop1FloatUnary", "Division", "Float16Misc", "Float16Rounding", "Float16ResultModifiers", "MixPrecision"]


def host_run(gpu, mode, code, rows, inputs, results, threads, work, wave64=False):
    (work / "code.txt").write_text("\n".join(f"0x{w:08x}" for w in code) + "\n")
    (work / "rows-host.txt").write_text("".join(" ".join(f"0x{v:08x}" for v in (r + [0] * inputs)[:inputs]) + "\n" for r in rows))
    env = dict(os.environ, REPLAY_CODE=str(work / "code.txt"), REPLAY_ROWS=str(work / "rows-host.txt"), REPLAY_INPUTS=str(inputs),
               REPLAY_RESULTS=str(results), REPLAY_THREADS=str(max(threads, len(rows))), REPLAY_WAVE="64" if wave64 else "32",
               REPLAY_MODE=HOST_MODES[mode], ANYPS5_GPU=gpu)
    p = subprocess.run([str(am.HARNESS)], env=env, capture_output=True, text=True, timeout=300)
    out = [[int(x, 16) for x in l.split()] for l in p.stdout.splitlines() if am.HEX_ROW.match(l.strip())]
    if p.returncode != 0 or len(out) != len(rows):
        return None, (p.stderr.strip().splitlines() or p.stdout.strip().splitlines() or ["no output"])[-1][:300]
    return out, ""


def saved_oracle(test, kernel, alu, rows, used):
    d = RUN1 / test / kernel
    if not d.is_dir():
        return None
    body = [int(w, 16) for w in re.findall(r"\.long 0x([0-9a-f]{8})", (d / "body.s").read_text())]
    saved_rows = [[int(v, 16) for v in l.split()] for l in (d / "rows-oracle.txt").read_text().splitlines() if l.strip()]
    if body != list(alu) or saved_rows != [(r + [0] * 4)[:4] for r in rows]:
        return None
    out = {}
    for mode in set(ORACLE_OF.values()):
        f = d / f"oracle-{mode}.txt"
        if not f.exists():
            return None
        out[mode] = [[int(x, 16) for x in l.split()] for l in f.read_text().splitlines() if am.HEX_ROW.match(l.strip())]
        if len(out[mode]) != len(rows):
            return None
    return out


def adapter_oracle(test, kernel, rows, inputs):
    values = json.load(open(RUN2 / "values.json"))
    cells = {}
    for name, a in values.items():
        if a["file"] != test + ".cpp" or a["kernel"] != kernel:
            continue
        cols = [int(c) for c in a["columns"]]
        index = {tuple((r + [0] * inputs)[:inputs][:4]): i for i, r in enumerate(rows)}
        for ri, inp in enumerate(a["inputs"]):
            key = tuple(int(v, 16) for v in inp[:4])
            row = index.get(key, ri if ri < len(rows) and key == tuple((rows[ri] + [0] * 4)[:4]) else None)
            if row is None:
                continue
            for mode in set(ORACLE_OF.values()):
                for ci, col in enumerate(cols):
                    cells.setdefault(mode, {})[(row, col)] = int(a["oracle"][mode][ri][ci], 16)
    return cells


STRIDES = {"Division.cpp": (4, 4, 32)}


def extract_with_adapter_rows(entry, kernel):
    src = (am.TESTS / entry["file"]).read_text()
    m = re.search(r"std::array<std::uint32_t,\s*\d+>\s*" + re.escape(kernel["name"]) + r"\s*\{(.*?)\};", src, re.S)
    if not m:
        raise RuntimeError(f"{entry['file']}: kernel {kernel['name']} not found")
    code = am.words_of(m.group(1))
    inputs, results, threads = STRIDES.get(entry["file"], (am.constant(src, "Inputs", 8), am.constant(src, "Results", 8), am.constant(src, "Threads", 32)))
    values = json.load(open(RUN2 / "values.json"))
    rows = []
    seen = set()
    for a in values.values():
        if a["file"] != entry["file"] or a["kernel"] != kernel["name"]:
            continue
        for inp in a["inputs"]:
            key = tuple(int(v, 16) for v in inp[:4])
            if key not in seen:
                seen.add(key)
                rows.append(list(key))
    if not rows:
        raise RuntimeError(f"{entry['file']}: no adapter rows for {kernel['name']}")
    if inputs != 4:
        raise RuntimeError(f"{entry['file']}: adapter rows have 4 dwords, the test's input stride is {inputs}")
    start, end = kernel["alu_word_range"]
    used = min(kernel.get("outputs") or results, results, 16)
    return code, code[start:end], rows, inputs, results, used, threads


def nan32(v):
    return (v & 0x7fffffff) > 0x7f800000


def nan16(v):
    return (v & 0x7fff) > 0x7c00


def classify(test, host, oracle, col, host_row, oracle_row):
    if host == oracle:
        return "equal"
    if test.startswith("F64"):
        hi = col | 1
        lo = hi - 1
        if hi < len(host_row) and hi < len(oracle_row):
            h = host_row[lo] | (host_row[hi] << 32)
            o = oracle_row[lo] | (oracle_row[hi] << 32)
            if (h & 0x7fffffffffffffff) > 0x7ff0000000000000 and (o & 0x7fffffffffffffff) > 0x7ff0000000000000:
                return "nan-payload"
    if nan32(host) and nan32(oracle):
        return "nan-payload"
    if (host >> 16) == (oracle >> 16) and nan16(host) and nan16(oracle):
        return "nan-payload"
    if (host & 0xffff) == (oracle & 0xffff) and nan16(host >> 16) and nan16(oracle >> 16):
        return "nan-payload"
    if nan16(host) and nan16(oracle) and nan16(host >> 16) and nan16(oracle >> 16):
        return "nan-payload"
    return "differs"


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    only = set(a.removesuffix(".cpp") for a in sys.argv[2:])
    manifest = {e["file"]: e for e in json.load(open(ROOT / "notes" / "execution-manifest.json"))}
    report = []
    for test in PASS1 + PASS2:
        if only and test not in only:
            continue
        entry = manifest[test + ".cpp"]
        for kernel in entry["kernels"]:
            name = kernel["name"]
            tag = f"{test}.{name}"
            work = out / tag
            work.mkdir(exist_ok=True)
            try:
                code, alu, rows, inputs, results, used, threads = am.extract(entry, kernel)
            except Exception as first:
                try:
                    code, alu, rows, inputs, results, used, threads = extract_with_adapter_rows(entry, kernel)
                except Exception as error:
                    report.append({"kernel": tag, "skipped": f"extract: {first}; {error}"})
                    print(tag, "skipped:", first, ";", error, flush=True)
                    continue
            if test in PASS2:
                cells = adapter_oracle(test, name, rows, inputs)
                if not cells:
                    report.append({"kernel": tag, "skipped": "no saved adapter values"})
                    print(tag, "skipped: no saved adapter values", flush=True)
                    continue
                oracle = {mode: cells[mode] for mode in cells}
                source = "pass 2 adapters"
            else:
                saved = saved_oracle(test, name, alu, rows, used)
                source = "pass 1 saved"
                if saved is None:
                    saved = {}
                    source = "re-measured"
                    for mode in set(ORACLE_OF.values()):
                        got, err = am.oracle_run(mode, alu, rows, results, work)
                        if got is None:
                            report.append({"kernel": tag, "skipped": f"oracle {mode}: {err}"})
                            print(tag, "skipped: oracle", mode, err, flush=True)
                            break
                        saved[mode] = got
                    else:
                        pass
                    if len(saved) != len(set(ORACLE_OF.values())):
                        continue
                oracle = {mode: {(r, c): saved[mode][r][c] for r in range(len(rows)) for c in range(used)} for mode in saved}
            result = {"kernel": tag, "oracle": source, "rows": len(rows), "used": used, "modes": {}}

            def run(gpu_mode):
                gpu, mode = gpu_mode
                return gpu, mode, host_run(gpu, mode, code, rows, inputs, results, threads, work / f"{gpu}-{mode}" if (work / f"{gpu}-{mode}").mkdir(exist_ok=True) is None else None)

            with ThreadPoolExecutor(max_workers=2) as pool:
                runs = list(pool.map(run, [(g, m) for g in GPUS for m in HOST_MODES]))
            for gpu, mode, (got, err) in runs:
                key = f"{gpu}/{mode}"
                if got is None:
                    result["modes"][key] = {"error": err}
                    continue
                omode = ORACLE_OF[mode]
                counts = {"equal": 0, "nan-payload": 0, "differs": 0}
                examples = []
                for (r, c), o in oracle[omode].items():
                    h = got[r][c]
                    k = classify(test, h, o, c, got[r], [oracle[omode].get((r, cc), 0) for cc in range(results)])
                    counts[k] += 1
                    if k == "differs" and len(examples) < 6:
                        examples.append({"row": r, "col": c, "inputs": [f"{v:08x}" for v in rows[r][:inputs]], "host": f"{h:08x}", "oracle": f"{o:08x}"})
                result["modes"][key] = {**counts, "examples": examples}
            dump = {"rows": rows, "inputs": inputs, "results": results, "used": used,
                    "oracle": {mode: {f"{r},{c}": v for (r, c), v in cells.items()} for mode, cells in oracle.items()},
                    "hosts": {f"{gpu}/{mode}": got for gpu, mode, (got, err) in runs if got is not None}}
            json.dump(dump, open(work / "values.json", "w"))
            report.append(result)
            line = " ".join(f"{k}={v.get('differs', 'ERR')}/{v.get('nan-payload', '')}" for k, v in result["modes"].items())
            print(f"{tag:<36} {source:<16} cells/mode={len(next(iter(oracle.values())))} {line}", flush=True)
    json.dump(report, open(out / "report.json", "w"), indent=1)
    print("wrote", out / "report.json")


if __name__ == "__main__":
    main()
