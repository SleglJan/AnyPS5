"""Match catalog next stops and absent imports against open upstream PRs.

Usage: python3 -I scripts/match_stops_prs.py <open-prs.json> <run-stops.log> <catalog-audit.json> <aerolib.csv>
Prints (1) per next stop: open PRs whose title names the function (strong) and PRs whose body only mentions it (weak);
(2) per open PR whose title implements import functions (feat/fix on a lib*): which named functions are a next stop,
an absent import in the catalog, or neither.
"""
import json
import re
import sys
from collections import OrderedDict

prs = json.load(open(sys.argv[1]))
log = open(sys.argv[2]).read()
audit = json.load(open(sys.argv[3]))
names = {}
for line in open(sys.argv[4]):
    parts = line.split()
    if len(parts) >= 2:
        names.setdefault(parts[0], parts[1])

stops = OrderedDict()
for block in re.split(r"^== ", log, flags=re.M)[1:]:
    tid = block.split()[0]
    m = re.search(r"undefined symbol: (\S+)", block)
    lib = re.search(r"error while loading shared libraries: (\S+?):", block)
    if m:
        nid = m.group(1).rstrip("'\"")
        stop = names.get(nid, nid)
    elif lib:
        stop = f"missing library {lib.group(1)}"
    else:
        stop = "(no loader stop)"
    stops.setdefault(stop, []).append(tid)

absent = {}
for tid, r in audit.items():
    for entry in (r.get("audit") or {}).get("absent_list") or []:
        name = entry[1] if isinstance(entry, (list, tuple)) and len(entry) > 1 else str(entry)
        absent.setdefault(name, set()).add(tid)

known = set(names.values())


def word(s, text):
    return re.search(r"(?<![A-Za-z0-9_])" + re.escape(s) + r"(?![A-Za-z0-9_])", text) is not None

print("### Next stops -> open PRs")
for stop, tids in sorted(stops.items(), key=lambda kv: (-len(kv[1]), kv[0])):
    fn = stop.replace("missing library ", "").replace(".prx", "")
    strong = [p for p in prs if word(fn, p["title"])]
    weak = [p for p in prs if p not in strong and word(fn, p["body"] or "")]
    print(f"- `{stop}` ({len(tids)}: {', '.join(tids)}): title {', '.join('#%d' % p['number'] for p in strong) or '-'}; body only {', '.join('#%d' % p['number'] for p in weak) or '-'}")

print("\n### Open PRs implementing imports -> catalog relevance")
rows = []
nohit = []
for p in prs:
    t = p["title"]
    if not re.match(r"^(feat|fix)\((lib|kernel|libkernel|libc|net|posix|video|audio|pad|user|np|save|ajm|ngs|rtc|ime|system)", t, re.I):
        continue
    m = re.search(r"\b(implement|implements|export|exports|add|adds|emulate|emulates)\b(.*)$", t, re.I)
    if not m:
        continue
    toks = set(re.findall(r"\b(sce[A-Z][A-Za-z0-9_]+|_{1,2}[a-z][A-Za-z0-9_]+|[a-z][a-z0-9_]{2,}(?=\b))", m.group(2)))
    toks = {x for x in toks if x in known}
    hits_stop = sorted(x for x in toks if x in stops)
    hits_abs = sorted(x for x in toks if x in absent and x not in stops)
    if hits_stop or hits_abs:
        rows.append((p["number"], t, hits_stop, hits_abs))
    elif toks:
        nohit.append((p["number"], t, sorted(toks)))
rows.sort(key=lambda r: (-len(r[2]), -len(r[3]), r[0]))
for n, t, hs, ha in rows:
    print(f"- #{n} {t[:80]}: next stop {', '.join('`%s` (%d)' % (h, len(stops[h])) for h in hs) or '-'}; absent import {', '.join('`%s` (%d)' % (h, len(absent[h])) for h in ha) or '-'}")
print("\n### Open import PRs whose named functions no catalog title imports")
for n, t, toks in sorted(nohit):
    print(f"- #{n} {t[:80]}: {', '.join('`%s`' % x for x in toks)}")
print(f"\n{len(rows)} import PRs with a catalog hit, {len(nohit)} without; {len(stops)} stops; {len(absent)} absent import names")
