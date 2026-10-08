"""dis.py File.cpp KernelName [cpu] -> word-indexed disassembly"""
import re, subprocess, sys
from pathlib import Path
T = Path("/home/slegl/PycharmProjects/anyps5/AnyPS5/core/libs/prx/libSceAgcDriver/tests/execution")
def words(f, k):
    src = (T / f).read_text()
    m = re.search(r"std::array<std::uint32_t,\s*\d+>\s*" + re.escape(k) + r"\s*\{(.*?)\};", src, re.S)
    if not m:
        m = re.search(r"\b" + re.escape(k) + r"\b[^=;{]*=?\s*\{(.*?)\};", src, re.S)
    return [int(w, 16) for w in re.findall(r"0x([0-9a-fA-F]{8})u?", m.group(1))]
def dis1(ws, cpu):
    b = ",".join(f"0x{x:02x}" for w in ws for x in w.to_bytes(4, "little"))
    p = subprocess.run(["llvm-mc", "--disassemble", "-triple=amdgcn-amd-amdhsa", f"-mcpu={cpu}", "-show-encoding"], input=f"[{b}]", capture_output=True, text=True)
    res = []
    for l in p.stdout.splitlines():
        enc = re.search(r"encoding: \[(.*?)\]", l)
        if enc: res.append((len(enc.group(1).split(",")) // 4, l.split(";")[0].strip()))
    return res, ("invalid" in p.stderr)
def at(ws, i, cpu):
    for n in (1, 2, 3):
        if i + n > len(ws): break
        r, bad = dis1(ws[i:i+n], cpu)
        if len(r) == 1 and r[0][0] == n and not bad:
            return n, r[0][1]
    return None
def dis(ws, cpu="gfx1030"):
    out = []; i = 0
    while i < len(ws):
        a = at(ws, i, cpu)
        tag = ""
        if not a and cpu != "gfx1013":
            a = at(ws, i, "gfx1013"); tag = "   [gfx1013 only]"
        if a:
            out.append((i, a[0], a[1] + tag)); i += a[0]
        else:
            out.append((i, 1, f"<invalid {ws[i]:08x}>")); i += 1
    return out
if __name__ == "__main__":
    f, k = sys.argv[1], sys.argv[2]
    cpu = sys.argv[3] if len(sys.argv) > 3 else "gfx1030"
    ws = words(f, k)
    for i, n, t in dis(ws, cpu):
        print(f"{i:4d} {' '.join(f'{w:08x}' for w in ws[i:i+n]):<28} {t}")
