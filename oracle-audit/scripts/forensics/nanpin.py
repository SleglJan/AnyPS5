import sys
sys.path.insert(0,sys.argv[1])
from pinned import tables,A,M
for k in A:
    name=k['file'][:-4]
    if name not in sys.argv[2:]: continue
    exp,chk,src=tables(name)
    for c in k['flagged_cells']:
        r,col=c['row'],c['col']
        if col>=len(exp[r]): continue
        p=exp[r][col]; t=int(c['oracle']['title'],16)
        if p!=t:
            print(f"{name:20s} r{r:2d} c{col:2d} in={' '.join(c['inputs'])} pinned={p:08x} title={t:08x} xor={p^t:08x} NV={c['hosts']['NVIDIA']} RA={c['hosts']['Ryzen']}")
