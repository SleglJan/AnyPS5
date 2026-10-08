import json,sys,collections
import os as _os, pathlib as _pl
ROOT = _os.environ.get("ANYPS5_AUDIT_ROOT") or str(_pl.Path(__file__).resolve().parents[3])
r=json.load(open(ROOT + '/notes/oracle-audit-run1/audit.json'))
M=["title","old","ieee0_keep32","ieee1_flush32","title_flush16","title_noclamp","title_fp16ovfl"]
ab=dict(title='T',old='O',ieee0_keep32='K',ieee1_flush32='F',title_flush16='h',title_noclamp='c',title_fp16ovfl='v')
name=sys.argv[1]; flag=sys.argv[2] if len(sys.argv)>2 else None
for k in r:
    if k['file']!=name+'.cpp': continue
    for c in k['flagged_cells']:
        if flag and not any(f.startswith(flag) for f in c["flags"]): continue
        o=c['oracle']; h=c['hosts']
        print(f"r{c['row']:2d} c{c['col']:2d} in={' '.join(c['inputs'])} | "+' '.join(f"{ab[m]}={o[m]}" for m in M)+f" | NV={h.get('NVIDIA')} RA={h.get('Ryzen')} | {','.join(x[:12] for x in c['flags'])}")
