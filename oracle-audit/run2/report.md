# Second pass (adapters): report of the run agent, 2026-10-09

Setup: gfx1036 (7950X3D iGPU), hsa-rocr 7.2.4-1.1, clang/llvm-mc 23.1.1; hosts RTX 3080 (driver 615.71.09) and RADV (mesa 26.2.4).
Sources: work/oracle-audit b89ad4fd (= main 4b6ab6bb + KernelReplay.cpp); upstream/main 94b1eecb differs only in F32MinMaxNan.cpp and three new tests.
Mode letters: T titles, O old, F IEEE 1, K f32 kept, h f16/f64 flushed, c DX10_CLAMP 0, v FP16_OVFL 1. All comparisons bit-exact under each column's mask.

## Coverage
221 adapter files run (43 single, 168 parts, 4 cases, 6 wave64), 11 skipped; 62 kernels with a run, 6 with none (DsWrxchg2B64; LdsAddtid,
LdsIntegerAtomics, LdsReadWrite wave64; LdsAtomics64; WaveIdInWorkgroup). 1,547 oracle runs, 36 host runs. 69,504 cells; 23,296 with a pin
(19 kernels, 23,083 asserted); 22,944 also on both hosts (18 kernels). Hosts skipped for the 42 computed-input kernels and for ScalarSopkCompare /
ScalarSopkWaitcnt (user-data SGPRs). Repeatability: 486,528 values rerun, only ShaderClock's masked clock columns changed.

## Asserted cells where the pin differs from the hardware in the titles' mode: 163 (all quiet bit; reproduced by F and O only)
| Test:kernel | Cells | Column: instruction | Hosts | Test's own check on the hardware value |
|---|---:|---|---|---|
| Division:HelperCode | 10 + 10 | c2 v_div_fmas_f32, c3 v_div_fixup_f32 | both = pin | rejects |
| Division:DivisionCode | 1 | c0 v_div_fixup_f32 | both = pin | rejects |
| Float16Misc:MiscCode | 4 + 4 | c0 v_ldexp_f16, c1 v_frexp_mant_f16 | both = pin | accepts (NaN-tolerant) |
| Float16Misc:MiscCode | 4 | c9 v_mul_legacy_f32 | NVIDIA 7fffffff, RADV = hardware | accepts |
| Float16Rounding:RoundingCode | 6 + 6 | c1 v_add_f16, c2 v_mul_f16 | NVIDIA abcd7fff, RADV = pin | accepts |
| Vop1FloatUnary:Code | 70 | f32 c0-9, c12-29: ceil floor trunc rndne fract frexp_mant sqrt rsq rcp rcp_iflag log exp sin cos | both = pin | rejects |
| Vop1FloatUnary:Code | 48 | f16 c30-53: the same 12 ops | both = pin | rejects |
Typical: Vop1FloatUnary f32 input 7f800001, hardware 7f800001, pin 7fc00001. The other 13 table kernels: 0 of 17,736 asserted cells differ.

## Unasserted cells where the pin differs: 82 (reproduced by K and O only)
Float16Misc c9 v_mul_legacy_f32, 72 cells with an f32 denormal (hardware 0, pin 839a0691; hosts disagree with each other);
Vop1FloatUnary row 0, 10 masked cells (ceil floor fract frexp_exp frexp_mant sin; frexp_exp hardware 0, pin and hosts ffffff82).

## Host vs hardware (18 kernels): 358 of 22,944 cells (358 NVIDIA, 211 RADV, 174 between hosts, 147 quiet bit only)
None outside what the test's own check accepts. Sources: quiet bit (host = pin = F value: 21 Division, 118 Vop1FloatUnary, 8 Float16Misc,
12 Float16Rounding on RADV only); NVIDIA canonical NaN (7fffffff, abcd7fff, 7fff); low-bit differences in tolerance-compared columns
(Vop1FloatUnary rsq/rcp/log/exp/sin/cos on NVIDIA, sin/cos on RADV; Float16ResultModifiers c3 on NVIDIA).

## Mode sensitivity: 1,224 of 69,504 cells (163 quiet bit only = the asserted table)
| Field | Cells | Quiet only | Instructions |
|---|---:|---:|---|
| IEEE_MODE | 351 | 163 | sNaN quieting; Vop3Aliases c3 v_cvt_f32_f16 mul:4, c5 v_cvt_f32_ubyte0 mul:2, c6 v_cvt_f32_ubyte1 div:2 (188) |
| f32 denormals | 324 | 0 | the same 188 Vop3Aliases cells; Float16Misc c9 (72); LdsAtomics float columns (54); Vop1FloatUnary row 0 (10) |
| f16/f64 denormals | 478 | 0 | Float16Misc norm/pknorm/frexp/ldexp; Float16ResultModifiers omod results; Float16Rounding add/cvt/mul; VopcCompare (4), VopcCompareF64 (50) |
| DX10_CLAMP | 21 | 0 | Float16ResultModifiers clamped f16 (v_sqrt_f16 clamp mul:2 = 16) |
| FP16_OVFL | 247 | 0 | v_ldexp_f16, v_cvt_f16_f32, v_mul/add_f16, f16 mul:2, Vop1FloatUnary exp/log/rcp/rsq f16, v_cvt_f16_u16 |
O differs from T in 487 cells. 51 of 62 kernels have no mode-dependent cell.

## Anomalies and caveats
No errors. All-zero columns (ScalarGetreg c0-1, ScalarSop1 c38, SdwaDestination c7) match inputs or pins. Masks applied from adapter notes:
Vop3Aliases c0-2, Vop3Integer c3-5/c11/c12 low 16 bits; ShaderClock c0-7 masked (its relations hold in all modes). Not measurable on gfx1036:
FmaRounding c4-7, c9; Float16Misc c10. Conditional asserts counted as asserted: FmaRounding c0-3, Division c2. 43 kernels compute their
expectation in code (compared across modes only; Vop3Aliases and LdsAtomics are the mode-dependent ones). Instruction column from llvm-mc gfx1030.
Inferred: adapter equivalence (supported by 18 host-run kernels agreeing in every integer, scalar and LDS cell), gfx1036 ~ gfx1013, Python ports of
the checks.

## Reproduce
python3 -s scripts/oracle/audit_adapters.py notes/oracle-audit-run2  (~5 min)
python3 -s scripts/oracle/forensics/allpinned2.py notes/oracle-audit-run2 --json notes/oracle-audit-run2/pinned-vs-hardware.json > notes/oracle-audit-run2/pinned-vs-hardware.txt
python3 -s scripts/oracle/forensics/sens2.py notes/oracle-audit-run2 > notes/oracle-audit-run2/mode-sensitivity.txt
python3 -s scripts/oracle/forensics/summary2.py notes/oracle-audit-run2 > notes/oracle-audit-run2/summary.txt
