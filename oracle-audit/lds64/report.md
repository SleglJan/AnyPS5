# 64-bit LDS atomics: hardware measurements and recompiler comparison (2026-10-09)

Oracle: Ryzen 7950X3D iGPU (gfx1036, RDNA2), `tools/hw-oracle` under hsa-rocr 7.2.4. Hosts: recompiler of `main`
d70b8998 through the local replay harness (`REPLAY_MODE`, `REPLAY_LDS` 4096), RTX 3080 (615.71.09) and RADV (Mesa
26.2.4). Scripts: `scripts/oracle/lds64_matrix.py` (functional matrices), `lds64_fit.py` (model fit),
`lds64_contention.py` (contention, mixed widths, EXEC, offsets, range), `lds64_hosts.py` (hosts vs oracle). Data:
`matrix.json`, `contention.json`, `hosts-report.json`, per-run `*-body.s`, `*-rows.txt`, `*.txt` in this directory.
Modes: IEEE 0/1 × the type's denormal field kept (3) or flushed (0); f64 uses the f16/f64 field (FLOAT_MODE bits 7:6),
f32 the f32 field (bits 5:4). The titles' mode 0xc0 keeps f64 and flushes f32 denormals.

## Hardware rules (gfx1036)

- **`ds_min/max_f64`, `ds_min/max_rtn_f64`** (14 × 14 operand pairs: ±0, ±1, ±inf, ±min denormal, qNaN ±, qNaN payload,
  sNaN ±, sNaN payload, plus 60 zero-pair padding lanes; every memory and returned result of the four forms, 1,536 values per mode): both operands are first flushed to a signed zero when the denormal field
  is 0; then a signalling-NaN data operand is stored quieted; else a signalling-NaN old value is stored quieted; else a
  quiet-NaN old value is replaced by the data; else a quiet-NaN data operand loses; else the ordered compare with
  −0 below +0 picks the winner. The stored value is the flushed one, also when the old value "wins". The returned value
  is always the old value as it was in memory. `IEEE_MODE` has no effect. 0 mismatches in all four modes against this
  rule, which is exactly `AtomicFloatBits::minMax` in SpirvMemoryInstructions.cpp (the buffer float atomics' rule)
  plus the flush.
- **`ds_min/max_f32`** and `_rtn`: the same rule with the f32 field: 0 mismatches in all four modes (1,024 results each over the 256 lanes).
- **`ds_cmpst_f64/f32`** and `_rtn`: equal when both flushed operands have the same bits, or both are zeros; NaN is
  never equal; when equal `data1` is stored, otherwise memory is left unchanged (not flushed). Returned value = old.
  0 mismatches (768 f64 results per mode, 512 f32 results per mode, over the 256 lanes).
- **29 integer forms** (add, sub, rsub, inc, dec, min/max i64/u64, and, or, xor, mskor, cmpst, wrxchg, with `_rtn`):
  0 mismatches on 64 edge pairs each (carries across the dword, inc/dec wrap, operand roles). Returned value = old.
- **EXEC**: inactive lanes neither update memory nor receive a value (256 lanes, wave32 and wave64).
- **Immediate offset** `offset:8`: applied (256 lanes, both wave sizes).
- **Out of range** (address 4096 + slot, and 0xfff8, LDS size 4096): returns 0, memory unchanged, both wave sizes.
- **Misaligned** 64-bit atomic (address % 8 == 4): the wave faults and the oracle process dies (as TechnicalDebt 273 records for gfx1035); not modelled by the recompiler, which returns a value.
- **Contention**, 256 lanes on one slot (8 wave32 or 4 wave64 waves of one workgroup): `ds_add_rtn_u64` of 0xffffffff
  sums to 0xffffffff00 exactly and the returned values form one serial order; `ds_max_rtn_u64` exact;
  `ds_inc_rtn_u64` serialisable.
- **Mixed widths**: 128 lanes `ds_add_u64` 0xffffffff and 128 lanes `ds_add_u32` 1 on the high dword of the same slot
  give exactly 0xffffffff80 (= 128 · 0xffffffff + 128 · 2^32) in both wave sizes: LDS serialises 32-bit and 64-bit
  atomics on the same address.

## Recompiler (`main` d70b8998) against the hardware

Units: dwords over the 256 lanes (a 64-bit result is two dwords). Before-fix numbers from the run of 2026-10-09 ~16:30 local against `libSceAgcDriver.prx` built from `main` d70b8998 (the feat/libkernel-mlock checkout, same recompiler sources); `hosts-report.json` now holds the after-fix run, in which every matrix cell matches.

| Case | RTX 3080 | RADV | Cause |
|---|---|---|---|
| f64 min/max, kept modes | 472 / 3,072 cells | 472 | `SharedFloatMinMax64` has no signalling-NaN rule (the f32 buffer path `AtomicFloatBits::minMax` has it) |
| f64 min/max, flushed modes | 584 / 3,072 | 584 | the same plus no flush |
| f64 cmpst, kept / flushed | 0 / 40 | 0 / 40 | no flush |
| f32 min/max/cmpst, kept modes | 336 / 2,304 | 336 | `SharedFloatMinMax` and `EmitSharedAtomicCmpstF32` use float compares (`OpIsNan`, `OpFOrdLessThan`, `OpFOrdEqual`): no sNaN quieting (296 cells), and both drivers flush f32 denormals inside those compares even with the field kept (40) |
| f32, flushed modes | 412 / 2,304 | 412 | as above; the driver's flush covers the compare but the stored old value is not rewritten flushed |
| 29 integer forms | 0 / 7,424 | 0 | |
| EXEC, offsets | 0 | 0 | |
| contention add/max | exact, serial | exact, serial | |
| mixed widths, 20 runs each (`mixed-repeats.json`) | wave32: increments lost in 18 of 20 runs (16 to 80); wave64: 0 in 17, 16 in 3 | wave32: 64 lost in 19, 96 in 1; wave64: 64 in 16, 32 in 2, 96 in 2 | the lock orders 64-bit atomics against each other only; 32-bit atomics bypass it (TechnicalDebt); every loss is a multiple of 16, the low dword is always 0xffffff80; the oracle is exact in 6 of 6 runs per wave size |
| out of range 64-bit | returns 0 | returns 0 | a first comparison used a harness LDS of 4,096 dwords (16 KiB) by mistake; with 1,024 dwords (the oracle's 4 KiB) both hosts return 0 and leave memory unchanged, like the hardware |

`LdsAtomics64.cpp`'s CPU model has the same signalling-NaN gap as the emitter, and its lanes of kind 2 include the
signalling NaN 0x7ff0000000000001, so on the console the test would assert the wrong value for those lanes.

## Proposed deliverables

1. PR: shared f32 and f64 float atomics follow the measured rule through `AtomicFloatBits::minMax`/`equal`
   (integer ops, sNaN quieting, no driver dependence), with the mode's denormal flush of both operands and the flushed
   write-back (translation knows `floatMode`; the emitter needs the flag), hardware-pinned tables in the titles' mode for f32 and f64 min/max/cmpst and the out-of-range
   case, `LdsAtomics64.cpp`'s model corrected, TechnicalDebt updated to "measured on gfx1036".
2. PR, coordinated first: native 64-bit LDS atomics where `shaderSharedInt64Atomics` and
   `workgroupMemoryExplicitLayout` are available (both hosts here), lock kept as fallback; fixes the mixed-width loss.
