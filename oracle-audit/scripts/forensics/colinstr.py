"""Which instruction produces an adapter's output column.

python3 -s colinstr.py Adapter.json [col ...]  -> for each result column, the last instruction in pre + words + post that
writes v(10+index), following v_mov_b32 copies back to their source. Disassembly: notes/oracle-adapters/disasm.py
(llvm-mc gfx1030 per word position, gfx1013 fallback marked "[gfx1013 only]").
"""
import functools
import json
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(os.environ.get("ANYPS5_AUDIT_ROOT") or pathlib.Path(__file__).resolve().parents[3])
ADAPTERS = ROOT / "notes" / "oracle-adapters"
sys.path.insert(0, str(ADAPTERS))
sys.path.insert(0, str(ROOT / "scripts" / "oracle"))
from disasm import dis  # noqa: E402
import srctables as st  # noqa: E402

TESTS = ROOT / "AnyPS5" / "core" / "libs" / "prx" / "libSceAgcDriver" / "tests" / "execution"
DEST = re.compile(r"^(\S+)\s+(v\[(\d+):(\d+)\]|v(\d+))\b")
MOV = re.compile(r"^v_mov_b32(?:_e32|_e64)?\s+v(\d+),\s*v(\d+)\s*$")


@functools.lru_cache(maxsize=None)
def listing(adapter_name):
    a = json.loads((ADAPTERS / adapter_name).read_text())
    if a.get("words_file"):
        ws = [int(x, 16) for x in (ADAPTERS / a["words_file"]).read_text().split()]
    else:
        ws = st.kernel_words((TESTS / a["file"]).read_text(), a["kernel"])
    s, e = a["words"]
    out = [("pre", l.strip()) for l in a.get("pre", [])]
    out += [(f"w{s + i}", t) for i, n, t in dis(ws[s:e])]
    post, chunk = [], []
    for l in a.get("post", []) + [None]:
        if l is not None and l.strip().startswith(".long"):
            chunk.append(int(l.split()[1], 0))
            continue
        if chunk:
            post += [("post.long", t) for i, n, t in dis(chunk)]
            chunk = []
        if l is not None:
            post.append(("post", l.strip()))
    return a, out + post


def writes(text, reg):
    m = DEST.match(text)
    if not m or m.group(1).startswith(("v_cmp", "v_cmpx", "s_", "buffer_", "global_", "ds_write", "ds_store")):
        return False
    if m.group(5) is not None:
        return int(m.group(5)) == reg
    return int(m.group(3)) <= reg <= int(m.group(4))


def producer(adapter_name, col):
    a, lst = listing(adapter_name)
    idx = a["columns"].get(str(col))
    if idx is None:
        return "column not mapped"
    reg, end, chain = 10 + idx, len(lst), []
    for _ in range(6):
        for k in range(end - 1, -1, -1):
            where, text = lst[k]
            if writes(text, reg):
                m = MOV.match(text)
                if m and int(m.group(1)) == reg:
                    chain.append(f"{where} {text}")
                    reg, end = int(m.group(2)), k
                    break
                return " <- ".join(chain + [f"{where} {text}"])
        else:
            return " <- ".join(chain + [f"v{reg} not written in the body (template input/zero)"])
    return " <- ".join(chain)


if __name__ == "__main__":
    name = sys.argv[1]
    a, _ = listing(name)
    cols = sys.argv[2:] or sorted(a["columns"], key=int)
    for c in cols:
        print(f"c{c}: {producer(name, int(c))}")
