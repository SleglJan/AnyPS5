"""Offline checks for oracle adapters (no GPU work).

Usage: python3 -I check_adapters.py [adapter.json ...]   (default: every *.json next to this script)

Per adapter: load the kernel words (C++ array in the test, or generated/<File>.<Kernel>.words.txt), disassemble
words[start:end] with llvm-mc (gfx1030, gfx1013 fallback), flag instructions the oracle cannot replay faithfully,
assemble template.s with pre + .long words + post for gfx1036 (wave32/64 as the adapter says), check the column map,
and evaluate a rows expression for every lane.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from disasm import dis, words as array_words  # noqa: E402

TREE = HERE.parents[1] / "AnyPS5"
TEMPLATE = TREE / "tools" / "hw-oracle" / "template.s"

MEMORY = re.compile(r"^(buffer_|tbuffer_|global_|flat_|scratch_|image_|s_load|s_buffer_load|s_store|s_buffer_store|s_scratch|s_atomic|s_buffer_atomic|ds_gws|ds_ordered|s_dcache|s_atc|exp )")
CONTROL = re.compile(r"^(s_endpgm|s_setpc|s_swappc|s_getpc|s_call|s_sethalt|s_sleep|s_trap|s_rfe|s_cbranch_cdbg|s_ttrace|s_setkill|s_icache|s_memtime|s_memrealtime|s_getreg|s_get_waveid|s_version|s_setreg|s_round_mode|s_denorm_mode|s_sendmsg)")
BRANCH = re.compile(r"^(s_branch|s_cbranch_\w+)\s+(-?\d+)")
RESERVED_DEST = re.compile(r"^\S+\s+(v0|v1|v2|v\[[0-2]:\d+\]|s4|s5|s6|s7|s\[[4-7]:\d+\])(,|\s|$)")


def kernel_words(adapter):
    gen = adapter.get("words_file")
    if gen:
        return [int(x, 16) for x in (HERE / gen).read_text().split()]
    return array_words(adapter["file"], adapter["kernel"])


def assemble(body, wave64):
    text = TEMPLATE.read_text()
    for k, v in (("@TARGET@", "gfx1036"), ("@WAVE32@", "0" if wave64 else "1"), ("@WAVESIZE@", "64" if wave64 else "32"),
                 ("@DENORM@", "0"), ("@DENORM16@", "3"), ("@IEEE@", "0"), ("@DX10_CLAMP@", "1"), ("@ROUND32@", "0"),
                 ("@ROUND16@", "0"), ("@FP16_OVERFLOW@", "0"), ("@BODY@", body)):
        text = text.replace(k, v)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "k.s").write_text(text)
        p = subprocess.run(["clang", "-x", "assembler", "-target", "amdgcn-amd-amdhsa", "-mcpu=gfx1036", "-c", str(Path(tmp) / "k.s"),
                            "-o", str(Path(tmp) / "k.o")], capture_output=True, text=True)
    return p.returncode == 0, p.stderr.strip()


def check(path):
    a = json.loads(Path(path).read_text())
    issues, info = [], []
    if a.get("skip"):
        return a, ["skip: " + str(a.get("reason", ""))], []
    ws = kernel_words(a)
    start, end = a["words"]
    body_words = ws[start:end]
    listing = dis(body_words)
    for i, n, text in listing:
        idx = start + i
        mnem = text.split()[0]
        if text.startswith("<invalid"):
            issues.append(f"word {idx}: undecodable {text}")
        if "[gfx1013 only]" in text:
            issues.append(f"word {idx}: gfx1013-only encoding on a gfx103x oracle: {text}")
        if MEMORY.match(text):
            issues.append(f"word {idx}: memory/cache instruction in body: {text}")
        if mnem.startswith("ds_") and not a.get("lds"):
            issues.append(f"word {idx}: LDS instruction but adapter lds=0: {text}")
        if CONTROL.match(text):
            info.append(f"word {idx}: control/state instruction: {text}")
        m = BRANCH.match(text)
        if m:
            target = idx + n + int(m.group(2)) if int(m.group(2)) < 32768 else idx + n + int(m.group(2)) - 65536
            if not (start <= target <= end):
                issues.append(f"word {idx}: branch to word {target} outside [{start},{end}]")
            else:
                info.append(f"word {idx}: branch to word {target} stays inside")
        if RESERVED_DEST.match(text):
            issues.append(f"word {idx}: writes a template-reserved register: {text}")
        for r in re.findall(r"\bs\[?(\d+)", text):
            if int(r) >= 48:
                issues.append(f"word {idx}: SGPR s{r} beyond the template's 48 allocated SGPRs: {text}")
        for r in re.findall(r"\bv\[?(\d+)", text):
            if int(r) >= 128:
                issues.append(f"word {idx}: VGPR v{r} beyond the template's 128 allocated VGPRs: {text}")
        if "exec" in text.split(",")[0]:
            info.append(f"word {idx}: writes EXEC: {text}")
    body = "".join(f"  {l}\n" for l in a.get("pre", [])) + "".join(f"  .long 0x{w:08x}\n" for w in body_words) + "".join(f"  {l}\n" for l in a.get("post", []))
    ok, err = assemble(body, bool(a.get("wave64")))
    if not ok:
        issues.append("assembly failed: " + err[:400])
    cols = a.get("columns", {})
    outs = list(cols.values())
    if len(set(outs)) != len(outs) or any(not (0 <= o <= 15) for o in outs):
        issues.append(f"bad column map {cols}")
    rows = a.get("rows")
    if isinstance(rows, dict) and "expr" in rows:
        n = rows.get("count", 32)
        try:
            vals = [eval(rows["expr"], {"__builtins__": {}}, {"tid": tid}) for tid in range(n)]
            if any(len(v) != 4 or any(not (0 <= x < 2**32) for x in v) for v in vals):
                issues.append("rows expression must give 4 u32 per lane")
        except Exception as e:
            issues.append(f"rows expression: {e}")
    if a.get("lds", 0) > 4096:
        issues.append(f"lds {a['lds']} > 4096")
    accepted = a.get("accepted_flags", {})
    kept = []
    for x in issues:
        m = re.match(r"word (\d+):", x)
        if m and m.group(1) in accepted:
            info.append(f"accepted: {x} ({accepted[m.group(1)]})")
        else:
            kept.append(x)
    return a, kept, info


def main():
    paths = sys.argv[1:] or sorted(str(p) for p in HERE.glob("*.json"))
    bad = 0
    for p in paths:
        a, issues, info = check(p)
        name = Path(p).name
        status = "SKIP" if a.get("skip") else ("ok" if not issues else "ISSUES")
        bad += status == "ISSUES"
        print(f"{status:6} {name}")
        for x in issues:
            print("    !", x)
        if len(paths) == 1:
            for x in info:
                print("    .", x)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
