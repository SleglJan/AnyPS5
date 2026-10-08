import sys,collections
sys.path.insert(0,sys.argv[1])
from pinned import tables,A,M
for k in A:
    if 'flagged_cells' not in k: continue
    name=k['file'][:-4]
    try: exp,chk,src=tables(name)
    except Exception as e: exp=None
    if not exp: print(name,k['kernel'],'no Expected table'); continue
    if len(exp)!=k['rows']: print(name,k['kernel'],'Expected rows',len(exp),'!= rows',k['rows']); continue
    cnt=collections.Counter()
    for c in k['flagged_cells']:
        r,col=c['row'],c['col']
        if col>=len(exp[r]): cnt['col-not-in-Expected']+=1; continue
        p=exp[r][col]; asserted = chk is None or (chk[r]>>col)&1
        t=int(c['oracle']['title'],16)
        if p!=t:
            match='+'.join(m for m in M if int(c['oracle'][m],16)==p) or 'none'
            cnt[('ASSERTED' if asserted else 'unasserted', 'pin!=title', 'exact-match:'+match)]+=1
    if cnt: print(name,k['kernel'],dict(cnt))
