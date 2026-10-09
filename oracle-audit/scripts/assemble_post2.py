"""Assemble notes/next-stops-post-2.md from the matcher output and the hand-checked facts of 2026-10-09."""
import json
import re
import sys

raw = open(sys.argv[1]).read(); audit = json.load(open(sys.argv[2])); run_at = sys.argv[3]; prs_at = sys.argv[4]; out_path = sys.argv[5]
absent = {}
for tid, r in audit.items():
    for lib, name in (r.get("audit") or {}).get("absent_list") or []:
        absent.setdefault(name, set()).add(tid)

intro = (f"Where each relinked catalog title stops on `main` f8c2f072, refreshing the table of 2026-10-08 above (`main` 4b6ab6bb): relinked and run {run_at} UTC, open PRs as of {prs_at} UTC. "
         "Same method: the catalog's records as of 2026-10-09 (52, up from 44: 8 new titles and 20 records with a new version, one of them a first release), every release ZIP checked against the record's sha256, `eboot.bin` and bundled modules unwrapped from their SELF containers, `main`'s relinker with `--registry` and its built `.prx` libraries, then each title run for 8 s in a bubblewrap sandbox (filesystem read-only, no network) on Linux (CachyOS, kernel 7.2.9, GCC 16.2.1). "
         "42 titles relink, including the 7 new ones with a usable release and XashPS5 (PPSA19111 v1.3, refused in the previous version); 6 are refused with `Forbidden syscall instruction` (PPSA01153, PPSA99004, PPSA99008, PPSA99011, PPSA99203 and the new PPSA99360); 4 records are unusable (PPSA77711 ships a `.ffpfsc` image, PPSA99204 and PPSA99206 return 404, PPSA99764 v2.7 ships two `eboot.bin`, so Porpoise is not in this run). "
         "39 die in the host loader on their first unresolved import, 2 on a library with no file, and DOOM runs to its exit; nothing is alive after 8 s. As @johnmaia found on the 9th, the loader binds every import at start-up, so the stop is the first absent import in loader order and the PR column names the first blocker of several, not the last.")

table_lines = raw.split("TABLE\n")[1].split("\n\nMOVED")[0].splitlines()
notes = {"`sigemptyset`": "#1641 (\"the sigset functions\")", "`sceSystemServiceLaunchWebBrowser`": "#2065 (exports it, the title does not say so)", "`dup`": "none (#1855 closed)", "`dup2`": "none", "`system`": "#1859"}
rows = ["| Next stop on `main` | Library | Titles | Open PR |", "|---|---|---:|---|"]
for l in table_lines[2:]:
    stop, lib, titles, strong, weak = [c.strip() for c in l.strip("|").split("|")]
    rows.append(f"| {stop} | {lib} | {titles} | {notes.get(stop, strong)} |")
stops_table = "\n".join(rows)

moved = ("Since the 8th: #1838 (merged 08:02 UTC) resolved `sceNetResolverStartNtoaMultipleRecordsEx` for its 4 titles, which now stop at `sceVideoOutGetResolutionStatus` (PPSA99177, PPSA99505, PPSA99515) and `getdents` (PPSA99997); "
         "#1711 (merged 09:55 UTC) resolved `socketpair` for its 2, which now stop at `getifaddrs` (PPSA99810) and `dup2` (PPSA99995). "
         "PPSA99039 (EVO-PLAYER) is v0.11.0 now, the fifth title @hikazey reported for #1838's function, and stops at `getdents` on Linux. "
         "`__inet_addr` keeps 2 titles: PPSA99169 and the new PPSA19111, since PPSA99764 could not be relinked. Every other title of the previous table stops where it did.")
new_titles = ("The 9 titles not in the previous table (7 new records, plus XashPS5 whose v1.3 is no longer refused and Cloud Play whose record now has a release): SwanStationPS5 PPSA98510 stops at `sceKernelGetFsSandboxRandomWord`; VITA5 PPSA39410 and Cloud Play PPSA99600 at `sceVideoOutGetResolutionStatus`; XashPS5 PPSA19111 at `__inet_addr`; RommPS PPSA76677 at `sceSystemServiceLaunchWebBrowser`; VLC PPSA85300 at `waitpid`; XPSemu PPSA97358 at `dup`; PokeMMO Prospero PPSA98001 at `sigemptyset`; Surround Sound Studio PPSA99051 at `sceAudioOutSysGetHdmiMonitorInfo`.")
doom = "PPSA99666 (DOOM) has no absent import and exits with code 0 after 6 log lines (32 on the 8th; the 27 `sceKernelOpen` calls with a garbage path are gone), `doom.log` still empty, the last line unchanged (#1497, #1529 open)."

hits = raw.split("PRS WITH A CATALOG HIT:\n")[1].split("\n\nPRS WITHOUT")[0].splitlines()
prrows = []
for l in hits:
    m = re.match(r"- #(\d+) (.*): ((?:next stop for|absent import in).*)$", l)
    if not m: continue
    n, title, rest = m.groups()
    st = re.search(r"next stop for (.*?) titles", rest); ab = re.search(r"absent import in (.*?) titles", rest)
    st = st.group(1) if st else ""; ab = ab.group(1) if ab else ""
    if n == "1482": ab = ab.replace(", 1 (`wait`)", "")
    prrows.append((int(n), title, st, ab))
prrows.append((1641, "feat(libkernel): implement the sigset functions", "1 (`sigemptyset`)", ", ".join(f"{len(absent.get(x, ()))} (`{x}`)" for x in ("sigfillset", "sigdelset", "sigaddset", "sigismember"))))
prrows.append((2065, "feat(libSceSystemService): report the web browser as unavailable", "1 (`sceSystemServiceLaunchWebBrowser`)", f"{len(absent.get('sceSystemServiceLaunchWebBrowser', ()))} (`sceSystemServiceLaunchWebBrowser`)"))
def nums(s): return [int(x) for x in re.findall(r"(\d+) \(", s)]
prrows.sort(key=lambda r: (-sum(nums(r[2])), -max(nums(r[3]) or [0]), r[0]))
pt = ["| Open PR | Next stop for (titles) | Absent import in (titles) |", "|---|---|---|"]
small = []
for n, title, st, ab in prrows:
    if not st and max(nums(ab) or [0]) < 3:
        small.append(f"#{n}"); continue
    if st:
        st = ", ".join(f"{m.group(1)} of {len(absent.get(m.group(2), ()))} importing `{m.group(2)}`" for m in re.finditer(r"(\d+) \(`([^`]+)`\)", st))
    pt.append(f"| #{n} {title[:72]} | {st} | {ab} |")
pr_intro = ("**Open import PRs against the same 42 titles.** 285 distinct absent imports remain over them. A PR is listed when its title names a function after \"implement\", \"export\" or \"add\" (#1641 and #2065 added by hand, their titles do not), and the count is the number of relinked catalog titles that import it; the homebrew catalog only, commercial titles are not visible from here.")
pr_table = "\n".join(pt) + f"\n\nBelow 3 titles: {', '.join(small)}."
nohit = ("24 open PRs name functions, clearly identifiable in their titles, that no relinked catalog title imports: #570, #571, #926, #950, #959, #1060, #1188, #1693, #1713, #1798, #1849, #2038, #2039, #2040, #2118, #2129, #2153, #2158, #2195, #2205, #2217, #2235, #2252, #2359. Other PR titles name functions the parser cannot tell from English words and are not counted.")
data = "Run log, audit, the open-PR list and the scripts: https://github.com/SleglJan/AnyPS5/tree/audit/oracle-2026-10-08/oracle-audit/next-stops-2026-10-09."
post = "\n\n".join([intro, stops_table, moved, new_titles, doom, pr_intro, pr_table, nohit, data, "AI-assisted: yes (Claude Code)."])
open(out_path, "w").write(post + "\n")
print("assembled:", len(post), "chars;", len(pt) - 2, "PR rows;", len(small), "small")
