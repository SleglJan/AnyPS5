"""Run the transcendental sweep bodies through the recompiler on both hosts and compare with the oracle in ULPs.

Usage: python3 -s scripts/oracle/transcendental_hosts.py notes/oracle-transcendentals [f16|f32]
Per op and host: exact matches, class mismatches (NaN/inf/finite/sign-of-zero), max and mean ULP distance over the
finite results, with the worst examples. The host kernel is the oracle body wrapped as the execution tests do, run in
batches of 1,024 lanes by build/tests/agc_driver_kernel_replay_tests.
"""
import importlib.util
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("hosts", HERE / "lds64_hosts.py")
hosts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hosts)
spec2 = importlib.util.spec_from_file_location("sweep", HERE / "transcendental_sweep.py")
sweep = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(sweep)

MODES32 = {"flushed": "0xc0,1,0,0", "kept": "0xf0,1,0,0"}


def ordered(bits, width):
    sign = 1 << (width - 1)
    return (sign - (bits & ~sign)) if bits & sign else bits | sign


def classify(bits, width):
    expo = (bits >> (width - (5 if width == 16 else 8) - 1)) & ((1 << (5 if width == 16 else 8)) - 1)
    mant = bits & ((1 << (10 if width == 16 else 23)) - 1)
    maxe = (1 << (5 if width == 16 else 8)) - 1
    if expo == maxe:
        return "nan" if mant else ("-inf" if bits >> (width - 1) else "+inf")
    if expo == 0 and mant == 0:
        return "-0" if bits >> (width - 1) else "+0"
    return "finite"


def compare(name, host_rows, oracle_rows, inputs, width, column, out, gpu):
    mask = 0xffff if width == 16 else 0xffffffff
    exact = cls_mism = 0
    ulps = []
    worst = []
    cls_examples = []
    for x, h, o in zip(inputs, host_rows, oracle_rows):
        hv, ov = h[column] & mask, o[column] & mask
        if hv == ov:
            exact += 1
            ulps.append(0)
            continue
        ch, co = classify(hv, width), classify(ov, width)
        if ch != co and not (ch == "finite" and co == "finite"):
            if {ch, co} <= {"+0", "-0"} or {ch, co} <= {"nan"}:
                ulps.append(0)
                continue
            cls_mism += 1
            if len(cls_examples) < 4:
                cls_examples.append(f"in={x:0{width // 4}x} host={hv:0{width // 4}x} hw={ov:0{width // 4}x}")
            continue
        d = abs(ordered(hv, width) - ordered(ov, width))
        ulps.append(d)
        worst.append((d, x, hv, ov))
    worst.sort(reverse=True)
    n = len(ulps)
    return {"inputs": len(inputs), "exact": exact, "class_mismatches": cls_mism, "max_ulp": max(ulps) if ulps else 0,
            "mean_ulp": round(sum(ulps) / n, 4) if n else 0, "within_1_ulp": sum(1 for u in ulps if u <= 1), "within_2_ulp": sum(1 for u in ulps if u <= 2),
            "worst": [f"in={x:0{width // 4}x} host={hv:0{width // 4}x} hw={ov:0{width // 4}x} ulp={d}" for d, x, hv, ov in worst[:4]], "class_examples": cls_examples}


def host_sweep(gpu, body, inputs, mode, work):
    words = hosts.assemble(hosts.host_kernel(body, 4, False), False)
    results = []
    for start in range(0, len(inputs), 1024):
        batch = inputs[start:start + 1024]
        rows = [[v, 0, 0, 0] for v in batch]
        got, err = hosts.host_run(gpu, words, rows, 4, len(rows), mode, False, work / f"{gpu}-{start}")
        if got is None:
            raise RuntimeError(f"{gpu} batch {start}: {err}")
        results.extend(got)
    return results


def load_oracle(path):
    return [[int(x, 16) for x in l.split()] for l in Path(path).read_text().splitlines() if l.strip()]


def main():
    out = Path(sys.argv[1]).resolve()
    which = set(sys.argv[2:]) or {"f16", "f32"}
    report = {}
    for gpu in ("Ryzen", "NVIDIA"):
        if "f16" in which:
            inputs = list(range(65536))
            oracle = load_oracle(out / "f16-title.txt")
            body = "".join(f"{op} v{10 + i}, v4\n" for i, op in enumerate(sweep.F16_OPS))
            got = host_sweep(gpu, body, inputs, "0xc0,1,0,0", out / "hosts" / "f16")
            for i, op in enumerate(sweep.F16_OPS):
                key = f"{gpu}/f16/{op}"
                report[key] = compare(key, got, oracle, inputs, 16, i, out, gpu)
                r = report[key]
                print(f"{key:<24} exact {r['exact']:5d}/{r['inputs']} class-mism {r['class_mismatches']:4d} max {r['max_ulp']:5d} mean {r['mean_ulp']:7.3f} <=1ulp {r['within_1_ulp']:5d}  worst {r['worst'][:1]} cls {r['class_examples'][:1]}", flush=True)
        if "f32" in which:
            inputs = json.load(open(out / "f32-inputs.json"))
            body = "".join(f"{op} v{10 + i}, v4\n" for i, op in enumerate(sweep.F32_OPS))
            for mode, flags in MODES32.items():
                oracle = load_oracle(out / f"f32-{mode}.txt")
                got = host_sweep(gpu, body, inputs, flags, out / "hosts" / f"f32-{mode}")
                for i, op in enumerate(sweep.F32_OPS):
                    key = f"{gpu}/f32-{mode}/{op}"
                    report[key] = compare(key, got, oracle, inputs, 32, i, out, gpu)
                    r = report[key]
                    print(f"{key:<30} exact {r['exact']:5d}/{r['inputs']} class-mism {r['class_mismatches']:4d} max {r['max_ulp']:7d} mean {r['mean_ulp']:7.3f} <=1ulp {r['within_1_ulp']:5d}  worst {r['worst'][:1]} cls {r['class_examples'][:1]}", flush=True)
    json.dump(report, open(out / "hosts-report.json", "w"), indent=1)
    print("wrote", out / "hosts-report.json")


if __name__ == "__main__":
    main()
