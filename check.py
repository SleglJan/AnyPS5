import sys, struct
D = sys.argv[1]
def omod(bits, m):
    mag, sign = bits & 0x7fffffff, bits & 0x80000000
    if m == 3: res = sign if mag < 0x01000000 else bits - 0x00800000
    else:
        step = 0x00800000 if m == 1 else 0x01000000
        res = (bits + step) if mag < 0x7f800000 - step else (sign | 0x7f800000)
    if mag < 0x00800000: res = 0
    if mag > 0x7f7fffff: res = bits
    return res & 0xffffffff
def f(u): return struct.unpack('<f', struct.pack('<I', u))[0]
def clamp(bits):
    if (bits & 0x7fffffff) > 0x7f800000: return 0
    v = f(bits)
    if v <= 0: return 0
    if v >= 1: return 0x3f800000
    return bits
def load(p): return [[int(t, 16) for t in l.split()] for l in open(p) if l.strip()]
print("mode        applied  base  ambiguous  neither | clamp cols ok/total | neg cols applied/base/neither")
for ieee in (0, 1):
    for d32 in range(4):
        for d16 in (0, 3):
            tag = f"i{ieee}-d{d32}-h{d16}"
            A, B = load(f"{D}/out-a-{tag}.txt"), load(f"{D}/out-b-{tag}.txt")
            app = base = amb = nei = 0
            neither_ex = []
            for r in A:
                for op in range(4):
                    b = r[op * 4]
                    for k, m in ((1, 1), (2, 2), (3, 3)):
                        hw, t = r[op * 4 + k], omod(b, m)
                        if hw == t and hw == b: amb += 1
                        elif hw == t: app += 1
                        elif hw == b: base += 1
                        else:
                            nei += 1
                            if len(neither_ex) < 3: neither_ex.append((op, m, hex(b), hex(hw), hex(t)))
            rule = ieee == 0
            cok = ctot = 0
            for ra, rb in zip(A, B):
                sc, ma = ra[4], ra[12]
                for hw, val, m in ((rb[2], sc, 0), (rb[3], sc, 2), (rb[4], sc, 3), (rb[5], ma, 1)):
                    exp = clamp(omod(val, m) if (m and rule) else val)
                    ctot += 1; cok += hw == exp
            napp = nbase = nnei = 0
            for rb in B:
                for bcol, mcol in ((0, 1), (8, 9), (10, 11), (12, 13)):
                    b, hw, t = rb[bcol], rb[mcol], omod(rb[bcol], 3)
                    if hw == t and hw != b: napp += 1
                    elif hw == b and hw != t: nbase += 1
                    elif hw != b and hw != t: nnei += 1
            print(f"{tag:10} {app:8} {base:5} {amb:10} {nei:8} | {cok}/{ctot} | {napp}/{nbase}/{nnei}", neither_ex)
