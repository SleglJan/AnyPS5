import json,re,subprocess,sys
import os as _os, pathlib as _pl
ROOT = _os.environ.get("ANYPS5_AUDIT_ROOT") or str(_pl.Path(__file__).resolve().parents[3])
sys.path.insert(0,ROOT + '/scripts/oracle')
import audit_modes as am
man=json.load(open(ROOT + '/notes/execution-manifest.json'))
def dis(words,cpu):
    bs=[]
    for w in words:
        bs+= [f"0x{(w>>(8*i))&0xff:02x}" for i in range(4)]
    p=subprocess.run(['llvm-mc','--disassemble','-triple=amdgcn-amd-amdhsa','-mcpu='+cpu,'-show-encoding'],input='['+','.join(bs)+']',capture_output=True,text=True)
    return p.stdout+p.stderr
for name in sys.argv[1:]:
    e=next(x for x in man if x['file']==name+'.cpp')
    k=e['kernels'][0]
    code,alu,rows,*_=am.extract(e,k)
    print('=====',name,'alu words',len(alu))
    for cpu in ('gfx1030','gfx1013'):
        print('---',cpu); print(dis(alu,cpu))
