import re
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

def words_of(text):
    return [int(w, 16) for w in re.findall(r"0x([0-9a-fA-F]{8})u?", text)]

code_match = re.search(r"std::array<std::uint32_t,\s*\d+>\s*(\w+Code)\s*\{(.*?)\};", src, re.S)
if not code_match:
    sys.exit("no Code array found")
code = words_of(code_match.group(2))
inputs = int(re.search(r"constexpr std::uint32_t Inputs = (\d+);", src).group(1))
results = int(re.search(r"constexpr std::uint32_t Results = (\d+);", src).group(1))
threads = int(re.search(r"constexpr std::uint32_t Threads = (\d+);", src).group(1))
rows_match = re.search(r"Rows\s*\{\{(.*?)\}\};", src, re.S)
rows = [words_of(line) for line in rows_match.group(1).split("\n") if "0x" in line]

first_waitcnt = next(i for i, w in enumerate(code) if (w >> 16) == 0xbf8c)
first_store = next(i for i, w in enumerate(code) if (w & 0xfff00000) == 0xe0700000)
alu = code[first_waitcnt + 1:first_store]

(out / "code.txt").write_text("\n".join(f"0x{w:08x}" for w in code) + "\n")
(out / "rows-host.txt").write_text("".join(" ".join(f"0x{v:08x}" for v in (r + [0] * inputs)[:inputs]) + "\n" for r in rows))
(out / "body.s").write_text("".join(f"  .long 0x{w:08x}\n" for w in alu))
(out / "rows-oracle.txt").write_text("".join(" ".join(f"0x{v:08x}" for v in (r + [0] * 4)[:4]) + "\n" for r in rows))
print(f"code {len(code)} words, alu words [{first_waitcnt + 1}, {first_store}) = {len(alu)}, inputs {inputs}, results {results}, threads {threads}, rows {len(rows)} x {len(rows[0])} values")
print("alu:", " ".join(f"{w:08x}" for w in alu))
