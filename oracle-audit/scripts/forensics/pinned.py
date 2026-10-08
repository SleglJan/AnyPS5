import json,re,sys,collections
import os as _os, pathlib as _pl
ROOT = _os.environ.get("ANYPS5_AUDIT_ROOT") or str(_pl.Path(__file__).resolve().parents[3])
T=ROOT + '/AnyPS5/core/libs/prx/libSceAgcDriver/tests/execution/'
A=json.load(open(ROOT + '/notes/oracle-audit-run1/audit.json'))
M=["title","old","ieee0_keep32","ieee1_flush32","title_flush16","title_noclamp","title_fp16ovfl"]
def tables(name):
    src=open(T+name+'.cpp').read()
    m=re.search(r"Expected\[(\d+)\]\[(\d+)\]\s*=\s*\{(.*?)\n\};",src,re.S)
    exp=None
    if m:
        rows=re.findall(r"\{([^{}]*)\}",m.group(3))
        exp=[[int(x,16) for x in re.findall(r"0x([0-9a-fA-F]+)u?",r)] for r in rows]
    m=re.search(r"Checked\[\d+\]\s*=\s*\{(.*?)\};",src,re.S)
    chk=[int(x,16) for x in re.findall(r"0x([0-9a-fA-F]+)u?",m.group(1))] if m else None
    return exp,chk,src
if __name__=='__main__':
    for name in sys.argv[1:]:
        exp,chk,src=tables(name)
        k=next(x for x in A if x['file']==name+'.cpp')
        print('##',name,'Expected cols',len(exp[0]) if exp else None,'Checked' if chk else '')
        cnt=collections.Counter()
        for c in k['flagged_cells']:
            r,col=c['row'],c['col']
            p=exp[r][col] if exp and col<len(exp[r]) else None
            asserted = p is not None and (chk is None or (chk[r]>>col)&1)
            h=c['hosts']; nv=int(h['NVIDIA'],16); ra=int(h['Ryzen'],16); t=int(c['oracle']['title'],16)
            if 'recompiler-differs-from-hardware-title-mode' in c['flags'] or 'hosts-differ' in c['flags']:
                print(f"r{r:2d} c{col:2d} pinned={p:08x} " if p is not None else f"r{r:2d} c{col:2d} pinned=----     ", 'asserted' if asserted else 'NOT-ASSERTED', f"NV={nv:08x} RA={ra:08x} T={t:08x}", 'pin==T' if p==t else ('pin==NV' if p==nv else ''), 'pin==RA' if p==ra else '')
            if p is not None:
                cnt[('pin==title' if p==t else 'pin!=title', 'asserted' if asserted else 'unasserted')]+=1
        print(cnt)
