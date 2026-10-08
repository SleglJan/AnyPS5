import json,re,subprocess,sys
import os as _os, pathlib as _pl
ROOT = _os.environ.get("ANYPS5_AUDIT_ROOT") or str(_pl.Path(__file__).resolve().parents[3])
sys.path.insert(0,ROOT + '/scripts/oracle')
import audit_modes as am
man=json.load(open(ROOT + '/notes/execution-manifest.json'))
def mc(words,cpu='gfx1030'):
    bs=[]
    for w in words: bs+=[f"0x{(w>>(8*i))&0xff:02x}" for i in range(4)]
    p=subprocess.run(['llvm-mc','--disassemble','-triple=amdgcn-amd-amdhsa','-mcpu='+cpu],input='['+','.join(bs)+']',capture_output=True,text=True)
    if 'warning' in p.stderr or 'error' in p.stderr: return None
    lines=[l.strip() for l in p.stdout.splitlines() if l.strip() and not l.strip().startswith('.text')]
    return lines
SEL=['BYTE_0','BYTE_1','BYTE_2','BYTE_3','WORD_0','WORD_1','DWORD','sel7']
DU=['UNUSED_PAD','UNUSED_SEXT','UNUSED_PRESERVE','dst_unused3']
OM=['','mul:2','mul:4','div:2']
def sdwa(w0,w1):
    op=(w0>>25)&0x3f; vs1=(w0>>9)&0xff; vd=(w0>>17)&0xff
    base=mc([(w0&~0x1ff)|0x104])
    name=base[0].split()[0].replace('_e32','') if base else f'vop2op{op:#x}'
    s0=w1&0xff
    def src(reg,neg,ab,sext,isv=True):
        t=f'v{reg}'
        if ab: t=f'|{t}|'
        if neg: t='-'+t
        if sext: t=f'sext({t})'
        return t
    a=src(s0,(w1>>20)&1,(w1>>21)&1,(w1>>19)&1)
    b=src(vs1,(w1>>28)&1,(w1>>29)&1,(w1>>27)&1)
    mods=[]
    if (w1>>13)&1: mods.append('clamp')
    if OM[(w1>>14)&3]: mods.append(OM[(w1>>14)&3])
    return f"{name}_sdwa v{vd}, {a}, {b} {' '.join(mods)} dst_sel:{SEL[(w1>>8)&7]} dst_unused:{DU[(w1>>11)&3]} src0_sel:{SEL[(w1>>16)&7]} src1_sel:{SEL[(w1>>24)&7]}  [manual decode: llvm-mc rejects]"
def disasm(alu,cpu='gfx1030'):
    out=[];i=0
    while i<len(alu):
        for n in (1,2,3):
            r=mc(alu[i:i+n],cpu)
            if r and len(r)==1:
                out.append((i,n,r[0])); i+=n; break
        else:
            if alu[i]&0x1ff==0xf9 and (alu[i]>>31)==0:
                out.append((i,2,sdwa(alu[i],alu[i+1]))); i+=2
            else:
                out.append((i,1,f'<undecodable {alu[i]:#010x}>')); i+=1
    return out
if __name__=='__main__':
    for name in sys.argv[1:]:
        e=next(x for x in man if x['file']==name+'.cpp')
        code,alu,*_=am.extract(e,e['kernels'][0])
        print('=====',name)
        for i,n,t in disasm(alu):
            print(f"  w{i:2d} {' '.join(f'{w:08x}' for w in alu[i:i+n]):26s} {t}")
