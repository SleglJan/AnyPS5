"""Second pass of the oracle audit: replay every ready oracle adapter on the hardware oracle in the seven modes of
audit_modes.MODES, and (where the adapter's rows are the test's own table layout) the whole kernel on the Vulkan hosts.

Usage: python3 -s scripts/oracle/audit_adapters.py [out-dir] [--only SUBSTR ...] [--hosts NVIDIA,Ryzen] [--no-hosts]

out-dir defaults to notes/oracle-audit-run2. Writes audit.json (one entry per adapter file, run-1 shape plus adapter
fields), values.json (every cell's value in every mode and on every host), skipped.json and work/<adapter>/ (the
exact body.s, rows and raw outputs of every run). Prints one log line per adapter.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import audit_modes as am  # noqa: E402
import srctables as st  # noqa: E402

ROOT = HERE.parents[1]
ADAPTERS = ROOT / "notes" / "oracle-adapters"
TESTS = am.TESTS
MODES = list(am.MODES)
FULL = 0xffffffff

# Whole-kernel host replay for kernels whose adapter rows are the test's own table layout. Each entry gives what the
# test's Run() does: input stride, result stride, threads (workgroup size), wave size, LDS (the test's own value) and
# how Input is filled. Rows always start at lane 0, so oracle row r == host lane r == the test's lane / vector r.
HOST_LAYOUT = {
    ("Float16ResultModifiers.cpp", "Code"): dict(inputs=4, results=16, threads=32, wave=32, fill=("table", "Rows", [0, 1, 2, 3], 32)),
    ("FmaRounding.cpp", "FmaCode"): dict(inputs=8, results=16, threads=64, wave=32, fill=("table", "Rows", [0, 1, 2, 3, 4], 64)),
    ("ScalarProgramFlow.cpp", "Wave32Code"): dict(inputs=4, results=16, threads=32, wave=32, fill=("table", "Rows", [0, 1, 2, 3], 32)),
    ("ScalarProgramFlow.cpp", "Wave64Code"): dict(inputs=4, results=16, threads=64, wave=64, fill=("table", "Rows", [0, 1, 2, 3], 64)),
    ("ScalarRelativeIndexing.cpp", "Code"): dict(inputs=4, results=16, threads=32, wave=32, fill=("zero", 32)),
    ("ScalarSopkMisc.cpp", "Code"): dict(inputs=4, results=16, threads=32, wave=32, fill=("zero", 32)),
    ("ScalarSop1.cpp", "Code"): dict(inputs=4, results=176, threads=32, wave=32, fill=("table", "Rows", [0, 1, 2, 3], 32)),
    ("ScalarSop2.cpp", "Code"): dict(inputs=4, results=64, threads=32, wave=32, fill=("table", "Rows", [0, 1, 2, 3], 32)),
    ("Vop1FloatUnary.cpp", "Code"): dict(inputs=4, results=64, threads=32, wave=32, fill=("table", "Rows", [0, 1, 2, 3], 32)),
    ("Vop1Vop2Integer.cpp", "Code"): dict(inputs=4, results=64, threads=32, wave=32, fill=("table", "Rows", [0, 1, 2, 3], 32)),
    ("Vop3IntegerAlu.cpp", "Code"): dict(inputs=4, results=64, threads=32, wave=32, fill=("table", "Rows", [0, 1, 2, 3], 32)),
    ("LdsAddtid.cpp", "Wave32Code"): dict(inputs=4, results=64, threads=32, wave=32, lds=32 * 32, fill=("table", "Rows", [0, 1, 2, 3], 32)),
    ("LdsIntegerAtomics.cpp", "Wave32Code"): dict(inputs=4, results=64, threads=32, wave=32, lds=32 * 32, fill=("table", "Rows", [0, 1, 2, 3], 32)),
    ("LdsReadWrite.cpp", "Wave32Code"): dict(inputs=4, results=64, threads=32, wave=32, lds=32 * 32, fill=("table", "Rows", [0, 1, 2, 3], 32)),
    ("Float16Misc.cpp", "MiscCode"): dict(inputs=4, results=16, threads=256, wave=32, fill=("table", "Vectors", [0, 1, 2], 256)),
    ("Float16Rounding.cpp", "RoundingCode"): dict(inputs=4, results=16, threads=256, wave=32, fill=("table", "Vectors", [0, 1, 2], 256)),
    ("Division.cpp", "HelperCode"): dict(inputs=4, results=4, threads=256, wave=32, fill=("table", "HelperVectors", [0, 1, 2, 3], 256)),
    ("Division.cpp", "DivisionCode"): dict(inputs=4, results=4, threads=256, wave=32, fill=("table", "DivisionVectors", [0, 1], 256)),
}
HOST_NOTES = {
    ("Float16Misc.cpp", "MiscCode"): "test dispatches 32 vectors at a time; host replay runs all 256 in one 256-thread dispatch (lane-independent ALU)",
    ("Float16Rounding.cpp", "RoundingCode"): "test dispatches 64 vectors at a time; host replay runs all 256 in one dispatch",
    ("Division.cpp", "HelperCode"): "test dispatches 32 vectors at a time; host replay runs all 256 in one dispatch",
    ("Division.cpp", "DivisionCode"): "test dispatches 32 vectors at a time; host replay runs all 256 in one dispatch",
    ("ScalarRelativeIndexing.cpp", "Code"): "Fill() is empty: Input is all zero, as in the test",
    ("ScalarSopkMisc.cpp", "Code"): "Fill() is empty: Input is all zero, as in the test",
}

# Comparison rules that the adapters state only in their notes (not as column_masks). They are ANDed into the mask
# used for flags and comparisons; values.json keeps the raw values and records the effective mask.
NOTE_MASKS = {
    ("Vop3Aliases.cpp", "AliasCode"): ({"0": 0xffff, "1": 0xffff, "2": 0xffff}, "adapter note: Check() asserts cols 0..2 by the f16 low half only"),
    ("Vop3Integer.cpp", "IntegerCode"): ({c: 0xffff for c in ("3", "4", "5", "11", "12")}, "adapter note: Check() masks results 3, 4, 5, 11, 12 to the low 16 bits"),
    ("ShaderClock.cpp", "Code"): ({str(c): 0 for c in range(8)}, "adapter note: cols 0..7 are clock values, not comparable value-for-value; Check()'s relations are evaluated instead (clock_relations)"),
}


def clock_relations(out):
    """ShaderClock Check(): cols 0..7 equal to lane (tid & ~7)'s, memtime/memrealtime end >= start, end pairs non-zero."""
    bad = []
    for tid, o in enumerate(out):
        first = out[tid & ~7]
        if any(o[i] != first[i] for i in range(8)):
            bad.append(f"lane {tid}: clock not uniform")
        pair = lambda i: o[i] | (o[i + 1] << 32)
        if pair(4) < pair(0) or pair(6) < pair(2):
            bad.append(f"lane {tid}: clock went back")
        if pair(4) == 0 or pair(6) == 0:
            bad.append(f"lane {tid}: clock read zero")
    return "ok" if not bad else "FAIL: " + "; ".join(bad[:4])


def src_of(file):
    return (TESTS / file).read_text()


def load_adapters(only):
    out = []
    for p in sorted(ADAPTERS.glob("*.json")):
        if only and not any(s in p.name for s in only):
            continue
        out.append((p.name, json.loads(p.read_text())))
    return out


def kernel_words(a):
    if a.get("words_file"):
        return [int(x, 16) for x in (ADAPTERS / a["words_file"]).read_text().split()]
    return st.kernel_words(src_of(a["file"]), a["kernel"])


def build_rows(a):
    spec = a["rows"]
    n = a.get("row_count")
    if spec == "first4":
        src = src_of(a["file"])
        try:
            rows = [(r + [0] * 4)[:4] for r in st.table(src, "Rows")]
            origin = "Rows table, words 0..3"
        except KeyError:
            rows = [[0, 0, 0, 0]] * n
            origin = "no Rows table in the test (Fill() empty): zero rows"
        if n and len(rows) > n:
            rows = rows[:n]
        return rows, origin
    if isinstance(spec, dict) and "expr" in spec:
        rows = [eval(spec["expr"], {"__builtins__": {}}, {"tid": tid}) for tid in range(spec["count"])]
        return [list(r) for r in rows], "expr"
    if isinstance(spec, dict) and "table" in spec:
        if spec.get("file"):
            rows = [[int(x, 16) for x in l.split()] for l in (ADAPTERS / spec["file"]).read_text().splitlines() if l.strip()]
            origin = f"{spec['table']} via {spec['file']}"
        else:
            t = st.table(src_of(a["file"]), spec["table"])
            fields = spec.get("words", [0, 1, 2, 3])
            rows = [[0 if f is None else r[f] for f in fields] for r in t]
            origin = f"{spec['table']} fields {fields}"
        return rows[:spec.get("count", len(rows))], origin
    raise ValueError(f"unsupported rows form {spec!r}")


def body_text(a, words):
    s, e = a["words"]
    return ("".join(f"  {l}\n" for l in a.get("pre", [])) + "".join(f"  .long 0x{w:08x}\n" for w in words[s:e])
            + "".join(f"  {l}\n" for l in a.get("post", [])))


def oracle_run(mode, body, rows, work, wave64):
    (work / "body.s").write_text(body)
    (work / "rows-oracle.txt").write_text("".join(" ".join(f"0x{v:08x}" for v in r) + "\n" for r in rows))
    m = am.MODES[mode]
    cmd = [sys.executable, "-s", str(am.ORACLE), "--ieee", str(m["ieee"]), "--denorm32", str(m["denorm32"]), "--denorm16", str(m["denorm16"]),
           "--dx10-clamp", str(m["dx10"]), "--round32", str(m["round32"]), "--round16", str(m["round16"]), "--fp16-overflow", str(m["fp16ovfl"]),
           "--outs", "16", str(work / "body.s"), str(work / "rows-oracle.txt")] + (["--wave64"] if wave64 else [])
    p = subprocess.run(cmd, cwd=am.TREE, capture_output=True, text=True, timeout=600)
    (work / f"oracle-{mode}.txt").write_text(p.stdout + ("\n# stderr\n" + p.stderr if p.stderr.strip() else ""))
    out = [[int(x, 16) for x in l.split()] for l in p.stdout.splitlines() if am.HEX_ROW.match(l.strip())]
    if p.returncode != 0 or len(out) != len(rows):
        return None, (p.stderr.strip().splitlines() or ["no output"])[-1][:300]
    return out, ""


def host_rows(file, kernel):
    lay = HOST_LAYOUT[(file, kernel)]
    fill = lay["fill"]
    if fill[0] == "zero":
        rows = [[0] * lay["inputs"] for _ in range(fill[1])]
    else:
        _, name, fields, count = fill
        t = st.table(src_of(file), name)[:count]
        rows = [([r[f] for f in fields] + [0] * lay["inputs"])[:lay["inputs"]] for r in t]
    return rows


def host_run(gpu, file, kernel, code, work):
    lay = HOST_LAYOUT[(file, kernel)]
    rows = host_rows(file, kernel)
    work.mkdir(parents=True, exist_ok=True)
    (work / "code.txt").write_text("\n".join(f"0x{w:08x}" for w in code) + "\n")
    (work / "rows-host.txt").write_text("".join(" ".join(f"0x{v:08x}" for v in r) + "\n" for r in rows))
    env = dict(os.environ, REPLAY_CODE=str(work / "code.txt"), REPLAY_ROWS=str(work / "rows-host.txt"), REPLAY_INPUTS=str(lay["inputs"]),
               REPLAY_RESULTS=str(lay["results"]), REPLAY_THREADS=str(max(lay["threads"], len(rows))), REPLAY_WAVE=str(lay["wave"]),
               REPLAY_LDS=str(lay.get("lds", 0)), ANYPS5_GPU=gpu)
    p = subprocess.run([str(am.HARNESS)], env=env, capture_output=True, text=True, timeout=300)
    (work / f"host-{gpu}.txt").write_text(p.stdout + ("\n# stderr\n" + p.stderr if p.stderr.strip() else ""))
    out = [[int(x, 16) for x in l.split()] for l in p.stdout.splitlines() if am.HEX_ROW.match(l.strip())]
    if p.returncode != 0 or len(out) != len(rows):
        return None, (p.stderr.strip().splitlines() or p.stdout.strip().splitlines() or [f"exit {p.returncode}, no output"])[-1][:300]
    return out, ""


def is_quiet_bit_only(a, b):
    """a != b and the difference is exactly the NaN quiet bit of an f32, an f16 half, or an f64 high word."""
    d = a ^ b
    if d == 1 << 22:
        return (a & 0x7f800000) == 0x7f800000 and (b & 0x7f800000) == 0x7f800000
    if d == 1 << 9:
        return (a & 0x7c00) == 0x7c00 and (b & 0x7c00) == 0x7c00
    if d == 1 << 25:
        return (a & 0x7c000000) == 0x7c000000 and (b & 0x7c000000) == 0x7c000000
    if d == 1 << 19:
        return (a & 0x7ff00000) == 0x7ff00000 and (b & 0x7ff00000) == 0x7ff00000
    return False


def adapter_variant(name, a):
    stem = name[:-5]
    prefix = f"{Path(a['file']).stem}.{a['kernel']}"
    return stem[len(prefix) + 1:] if stem.startswith(prefix + ".") else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=str(ROOT / "notes" / "oracle-audit-run2"))
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--hosts", default="NVIDIA,Ryzen")
    ap.add_argument("--no-hosts", action="store_true")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    hosts = [] if args.no_hosts else [h for h in args.hosts.split(",") if h]
    adapters = load_adapters(args.only)
    report, values, skipped = [], {}, []
    host_cache = {}
    started = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"# audit_adapters started {started}; {len(adapters)} adapter files; modes {','.join(MODES)}; hosts {','.join(hosts) or 'none'}", flush=True)
    for name, a in adapters:
        variant = adapter_variant(name, a)
        if a.get("skip"):
            skipped.append(dict(adapter=name, file=a["file"], kernel=a["kernel"], variant=variant, reason=a.get("reason", "")))
            print(f"SKIP {name}: {a.get('reason', '')[:160]}", flush=True)
            continue
        item = dict(file=a["file"], kernel=a["kernel"], adapter=name, variant=variant, status=a.get("status"),
                    wave64=bool(a.get("wave64")), lds=a.get("lds", 0), words=a["words"], words_file=a.get("words_file"))
        work = out / "work" / name[:-5]
        work.mkdir(parents=True, exist_ok=True)
        try:
            words = kernel_words(a)
            rows, origin = build_rows(a)
        except Exception as error:
            item["error"] = f"extract: {error}"
            report.append(item)
            print(f"ERROR {name}: {item['error']}", flush=True)
            continue
        if a.get("row_count") and len(rows) != a["row_count"]:
            item["row_note"] = f"rows {len(rows)} != row_count {a['row_count']}: all {len(rows)} rows run in one dispatch (one workgroup)"
        columns = sorted(((c, i) for c, i in a["columns"].items()), key=lambda x: int(x[0]))
        masks = {c: int(a.get("column_masks", {}).get(c, hex(FULL)), 16) for c, _ in columns}
        if (a["file"], a["kernel"]) in NOTE_MASKS:
            extra, why = NOTE_MASKS[(a["file"], a["kernel"])]
            for c in masks:
                if c in extra:
                    masks[c] &= extra[c]
            item["note_masks"] = dict(masks={c: f"0x{m:08x}" for c, m in extra.items()}, reason=why)
        item.update(rows=len(rows), rows_origin=origin, columns={c: i for c, i in columns},
                    column_masks={c: f"0x{m:08x}" for c, m in masks.items() if m != FULL})
        body = body_text(a, words)
        oracle_out = {}
        for mode in MODES:
            vals, err = oracle_run(mode, body, rows, work, bool(a.get("wave64")))
            oracle_out[mode] = vals
            if err:
                item.setdefault("oracle_errors", {})[mode] = err
        if (a["file"], a["kernel"]) == ("ShaderClock.cpp", "Code"):
            item["clock_relations"] = {mode: clock_relations(v) for mode, v in oracle_out.items() if v}
        key = (a["file"], a["kernel"])
        host_out = {}
        if key in HOST_LAYOUT and (a["rows"] == "first4" or (isinstance(a["rows"], dict) and "table" in a["rows"])):
            if hosts:
                if key not in host_cache:
                    res = {}
                    code = kernel_words(a)
                    for gpu in hosts:
                        res[gpu] = host_run(gpu, a["file"], a["kernel"], code, out / "work" / "host" / f"{Path(a['file']).stem}.{a['kernel']}")
                    host_cache[key] = res
                for gpu, (vals, err) in host_cache[key].items():
                    host_out[gpu] = vals
                    if err:
                        item.setdefault("host_errors", {})[gpu] = err
                lay = HOST_LAYOUT[key]
                item["host_run"] = (f"whole kernel ({len(words)} words), {lay['threads']} threads, wave{lay['wave']}, inputs {lay['inputs']}, "
                                    f"results {lay['results']}, lds {lay.get('lds', 0)}" + (f"; {HOST_NOTES[key]}" if key in HOST_NOTES else ""))
            else:
                item["host_run"] = "skipped: --no-hosts"
        else:
            if isinstance(a["rows"], dict) and "expr" in a["rows"]:
                why = "rows are computed (expr), not the test's table"
            elif key in (("ScalarSopkCompare.cpp", "CompareCode"), ("ScalarSopkWaitcnt.cpp", "WaitCode")):
                why = "the test passes its values in user-data SGPRs s8+, which the harness does not set"
            else:
                why = "adapter layout differs from the test's"
            item["host_run"] = "skipped: " + why
        cells = []
        vrec = dict(file=a["file"], kernel=a["kernel"], columns=[c for c, _ in columns], masks=[f"0x{masks[c]:08x}" for c, _ in columns],
                    inputs=[[f"{v:08x}" for v in r] for r in rows], oracle={}, hosts={})
        for mode in MODES:
            if oracle_out.get(mode):
                vrec["oracle"][mode] = [[f"{oracle_out[mode][r][i]:08x}" for _, i in columns] for r in range(len(rows))]
        for gpu, vals in host_out.items():
            if vals:
                vrec["hosts"][gpu] = [[f"{vals[r][int(c)]:08x}" if r < len(vals) else None for c, _ in columns] for r in range(len(rows))]
        values[name] = vrec
        summary = {}
        for r in range(len(rows)):
            for c, i in columns:
                m = masks[c]
                o = {md: oracle_out[md][r][i] & m for md in MODES if oracle_out.get(md)}
                h = {g: host_out[g][r][int(c)] & m for g in host_out if host_out.get(g) and r < len(host_out[g])}
                flags = []
                t = o.get("title")
                if t is not None:
                    diff = [v for md, v in o.items() if v != t]
                    if diff:
                        flags.append("mode-dependent")
                        if all(is_quiet_bit_only(v, t) for v in diff):
                            flags.append("mode-dependent-quiet-bit-only")
                hv = list(h.values())
                if hv and any(v != hv[0] for v in hv):
                    flags.append("hosts-differ")
                if hv and t is not None and any(v != t for v in hv):
                    flags.append("recompiler-differs-from-hardware-title-mode")
                    if all(v == t or is_quiet_bit_only(v, t) for v in hv):
                        flags.append("recompiler-differs-quiet-bit-only")
                if flags:
                    cells.append(dict(row=r, col=int(c), out=i, mask=f"0x{m:08x}", inputs=[f"{v:08x}" for v in rows[r]],
                                      oracle={md: f"{v:08x}" for md, v in o.items()}, hosts={g: f"{v:08x}" for g, v in h.items()}, flags=flags))
                    for f in flags:
                        summary[f] = summary.get(f, 0) + 1
        zero_cols = [c for c, i in columns if masks[c] and oracle_out.get("title") and all(oracle_out["title"][r][i] & masks[c] == 0 for r in range(len(rows)))]
        if zero_cols:
            item["all_zero_columns_title"] = [int(c) for c in zero_cols]
        item["cells"] = len(rows) * len(columns)
        item["flagged_cells"] = cells
        report.append(item)
        err = ""
        if item.get("oracle_errors") or item.get("host_errors"):
            err = f"; errors oracle={item.get('oracle_errors')} host={item.get('host_errors')}"
        print(f"{name}: {len(rows)} rows x {len(columns)} cols; hosts {'ran' if host_out else item['host_run']}; flagged {summary or 'none'}"
              + (f"; all-zero title cols {item['all_zero_columns_title']}" if zero_cols else "") + err, flush=True)
    json.dump(report, open(out / "audit.json", "w"), indent=1)
    json.dump(values, open(out / "values.json", "w"))
    json.dump(skipped, open(out / "skipped.json", "w"), indent=1)
    print(f"# wrote {out / 'audit.json'} ({len(report)} adapters run, {len(skipped)} skipped, {sum(x.get('cells', 0) for x in report)} cells)", flush=True)


if __name__ == "__main__":
    main()
