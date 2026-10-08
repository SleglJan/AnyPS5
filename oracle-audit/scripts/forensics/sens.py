import sys,collections,json
import os as _os, pathlib as _pl
ROOT = _os.environ.get("ANYPS5_AUDIT_ROOT") or str(_pl.Path(__file__).resolve().parents[3])
sys.path.insert(0,sys.argv[1])
from colmap import colmap
A=json.load(open(ROOT + '/notes/oracle-audit-run1/audit.json'))
F={'ieee1_flush32':'IEEE','ieee0_keep32':'D32','title_flush16':'D16','title_noclamp':'DX10','title_fp16ovfl':'OVFL'}
Q={0x00080000,0x00400000,0x0200,0x02000000,0x02000200}
tot=collections.Counter(); rows=[]
for k in A:
    cells=[c for c in k.get('flagged_cells',[]) if 'mode-dependent' in c['flags'] or 'mode-dependent-nan-payload' in c['flags']]
    if not cells: continue
    name=k['file'][:-4]; cm=colmap(name,k['kernel'])
    per=collections.defaultdict(lambda: collections.Counter())
    n=collections.Counter(); nq=collections.Counter()
    for c in cells:
        o={m:int(v,16) for m,v in c['oracle'].items()}; t=o['title']
        md='mode-dependent' in c['flags']
        n[c['col']]+=md; nq[c['col']]+= (not md)
        sig=[]
        for m,f in F.items():
            if o[m]!=t: sig.append(f+('(q)' if (o[m]^t) in Q else ''))
        if o['old']!=t and o['ieee1_flush32']==t and o['ieee0_keep32']==t: sig.append('IEEE&D32')
        if o['old']==t and (o['ieee1_flush32']!=t or o['ieee0_keep32']!=t): sig.append('old-cancels')
        per[c['col']][' '.join(sig)]+=1
    for col in sorted(per):
        rows.append((name,col,cm.get(col,'?'),n[col],nq[col],dict(per[col])))
        tot['md']+=n[col]
for r in rows: print(f"{r[0]:24s} c{r[1]:2d} md={r[3]:3d} nanpay={r[4]:2d} | {r[2][:70]:70s} | "+'; '.join(f'{v}x {k}' for k,v in sorted(r[5].items(),key=lambda x:-x[1])))
print(tot)
