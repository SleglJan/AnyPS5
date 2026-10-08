import json,re,sys
sys.path.insert(0,sys.argv[1] if len(sys.argv)>1 else '.')
from dis2 import disasm,man,am
def colmap(name,kernel=None):
    e=next(x for x in man if x['file']==name+'.cpp')
    k=next(x for x in e['kernels'] if kernel is None or x['name']==kernel)
    code,alu,*_=am.extract(e,k)
    cpu='gfx1013' if k.get('decoder','').endswith('gfx1013') else 'gfx1030'
    m={}
    for i,n,t in disasm(alu,cpu):
        body=t.split(';')[0].strip()
        mm=re.match(r"(\S+)\s+(v\[(\d+):(\d+)\]|v(\d+))",body)
        if not mm: continue
        if mm.group(3): regs=range(int(mm.group(3)),int(mm.group(4))+1)
        else: regs=[int(mm.group(5))]
        for j,r in enumerate(regs):
            if 10<=r<26 and not body.startswith('v_mov_b32'):
                m[r-10]=body+(f'  [dword {j}]' if len(regs)>1 else '')
    return m
if __name__=='__main__':
    for n in sys.argv[2:]:
        print('==',n)
        for c,t in sorted(colmap(n).items()): print(f'  c{c:2d} {t}')
