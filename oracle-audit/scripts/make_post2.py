"""Build the next-stop table and the open-PR relevance lists as Markdown.

Usage: python3 -I scripts/make_post2.py <run-stops.log> <catalog-audit.json> <open-prs.json> <aerolib.csv> [previous-run-stops.log]
"""
import json
import re
import sys
from collections import OrderedDict

log = open(sys.argv[1]).read()
audit = json.load(open(sys.argv[2]))
prs = json.load(open(sys.argv[3]))
names = {}
for line in open(sys.argv[4]):
    parts = line.split()
    if len(parts) >= 2:
        names.setdefault(parts[0], parts[1])
prev_log = open(sys.argv[5]).read() if len(sys.argv) > 5 else ""
known = set(names.values())


def parse_stops(text):
    stops = OrderedDict()
    for block in re.split(r"^== ", text, flags=re.M)[1:]:
        tid = block.split()[0]
        m = re.search(r"undefined symbol: (\S+)", block)
        lib = re.search(r"error while loading shared libraries: (\S+?):", block)
        if m:
            stop = names.get(m.group(1).rstrip("'\""), m.group(1))
        elif lib:
            stop = f"missing library `{lib.group(1)}`"
        elif "error " in block.split("\n")[0]:
            stop = "(runner error)"
        else:
            stop = "(no loader stop)"
        stops.setdefault(stop, []).append(tid)
    return stops


stops = parse_stops(log)
prev = parse_stops(prev_log) if prev_log else {}
prev_by_title = {t: s for s, ts in prev.items() for t in ts}
library = {}
absent = {}
for tid, r in audit.items():
    for entry in (r.get("audit") or {}).get("absent_list") or []:
        if isinstance(entry, (list, tuple)) and len(entry) > 1:
            library.setdefault(entry[1], entry[0])
            absent.setdefault(entry[1], set()).add(tid)


def word(s, text):
    return re.search(r"(?<![A-Za-z0-9_])" + re.escape(s) + r"(?![A-Za-z0-9_])", text) is not None


out = []
out.append("| Next stop on `main` | Library | Titles | Open PRs naming it in the title | Mentions in a body |")
out.append("|---|---|---:|---|---|")
for stop, tids in sorted(stops.items(), key=lambda kv: (kv[0].startswith("("), -len(kv[1]), kv[0])):
    if stop.startswith("("):
        continue
    fn = stop.replace("missing library `", "").rstrip("`").replace(".prx", "")
    strong = [p["number"] for p in prs if word(fn, p["title"]) and re.match(r"^(feat|fix)\((?!ci|relinker|tools|shader|build|win|recompiler|hw-oracle|docs)", p["title"])]
    weak = [p["number"] for p in prs if p["number"] not in strong and word(fn, p["body"] or "")]
    lib = library.get(fn, "")
    cell = f"`{stop}`" if not stop.startswith("missing") else stop
    out.append(f"| {cell} | {('`%s`' % lib) if lib else ''} | {len(tids)}: {', '.join(tids)} | {', '.join('#%d' % n for n in strong) or 'none'} | {', '.join('#%d' % n for n in weak[:6]) or ''}{' …' if len(weak) > 6 else ''} |")
table = "\n".join(out)

moved = []
for stop, tids in stops.items():
    for t in tids:
        if t in prev_by_title and prev_by_title[t] != stop:
            moved.append((t, prev_by_title[t], stop))
new_titles = [t for ts in stops.values() for t in ts if prev and t not in prev_by_title]

rows = []
nohit = []
for p in prs:
    t = p["title"]
    if not re.match(r"^(feat|fix)\(", t):
        continue
    m = re.search(r"\b(implement|implements|export|exports|add|adds|emulate|emulates)\b(.*)$", t, re.I)
    if not m:
        continue
    toks = {x for x in re.findall(r"\b(sce[A-Z][A-Za-z0-9_]+|_{1,2}[a-z][A-Za-z0-9_]+|[a-z][a-z0-9_]{2,})\b", m.group(2)) if x in known}
    hs = sorted(x for x in toks if x in stops)
    ha = sorted(x for x in toks if x in absent and x not in stops)
    if hs or ha:
        rows.append((p["number"], t, hs, ha))
    elif toks:
        nohit.append((p["number"], t, sorted(toks)))
rows.sort(key=lambda r: (-len(r[2]), -max([len(absent[x]) for x in r[3]] or [0]), r[0]))

print("TABLE\n" + table)
print("\nMOVED (title, before, now):")
for t, b, n in moved:
    print(f"  {t}: {b} -> {n}")
print("\nNEW TITLES:", ", ".join(new_titles))
print("\nSTOP COUNTS:", {k: len(v) for k, v in stops.items() if k.startswith("(")})
print("\nPRS WITH A CATALOG HIT:")
for n, t, hs, ha in rows:
    print(f"- #{n} {t[:90]}: " + "; ".join(filter(None, [("next stop for " + ", ".join('%d (`%s`)' % (len(stops[h]), h) for h in hs) + " titles") if hs else "", ("absent import in " + ", ".join('%d (`%s`)' % (len(absent[h]), h) for h in ha) + " titles") if ha else ""])))
print("\nPRS WITHOUT A CATALOG HIT:")
for n, t, toks in sorted(nohit):
    print(f"- #{n} {t[:90]}: {', '.join('`%s`' % x for x in toks)}")
print(f"\nSUMMARY: {len(stops)} distinct stops; {len(rows)} import PRs with a hit, {len(nohit)} without; {len(absent)} absent import names over {len(audit)} records")
