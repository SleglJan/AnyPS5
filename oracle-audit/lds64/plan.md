# 64-bit LDS atomics: measurement plan (2026-10-09)

## Why

TechnicalDebt (main d70b8998, line 338): the 35 64-bit DS atomics (`ds_*_u64`, `ds_*_b64`, `ds_*_f64`, their `_rtn`
forms, `ds_wrxchg_rtn_b64`) read and write their two dwords under a workgroup lock stored after the guest LDS, because
LDS is emitted as a 32-bit `Workgroup` array. They are atomic only against each other: concurrent 32-bit accesses to the
same dwords are not ordered. `ds_min/max/cmpst_f64` follow the NaN and signed-zero rules measured for the f32 forms;
"the f64 forms were not measured on hardware". History: #573 (CnRJay, merged 2026-10-05) added them with a CPU
reference test (`agc_driver_lds_atomics64`, expectations computed in code, not captured); #826 reported a 5 s GPU
timeout on RDNA 3.5 from the lock loop; #833 (LeandroLP23) made the loop elect one lane per iteration. The author
named the alternative in #573: `VK_KHR_workgroup_memory_explicit_layout` plus `shaderSharedInt64Atomics`. Both GPUs
on this machine (RTX 3080, RADV gfx1036) report both features.

Nothing open upstream touches this (live `gh pr list`, 2026-10-09 14:xxZ: #1776 LDS beyond the device limit and #1190
FLAT apertures are the only LDS PRs; no issue mentions the f64 forms or mixed-width ordering).

## What the hardware can tell us

Oracle: gfx1036 (Ryzen 7950X3D iGPU, RDNA2), `tools/hw-oracle/hw_oracle.py`; 4 KiB LDS, 4 input dwords per lane in
v4..v7, extra bytes via `--extra` after the padded rows, outputs v10..v25, `--wave64`, explicit float-mode flags.
All 29 integer forms and the 6 f64 forms assemble for gfx1030 (llvm-mc); `ds_add_f64` does not exist on RDNA2.

A. **Functional matrix, isolated lanes.** Each lane owns LDS slot `lane * 8`: `ds_write_b64` the initial value a,
   the op with data b (and c for `cmpst`/`mskor`, from `--extra`), then `ds_read_b64` of the slot → v10:11, the
   returned value → v12:13.
   - Integer forms: edge pairs (0, 1, max, min, sign bit, a == b, a = b ± 1, values whose add/sub/inc/dec carries
     across the dword boundary), both directions. Pins carries, `inc`/`dec` wrap, `mskor` operand roles, `cmpst`
     operand roles (`DS[A] == data0 ? data1 : DS[A]`), `wrxchg`.
   - `ds_min/max_f64` and `_rtn`: the 14-value operand set from the min/max matrix (±0, ±1, ±inf, ±min denormal,
     qNaN ±, qNaN payload, sNaN ±, sNaN payload), 14 × 14, in four modes (IEEE 0/1 × f64 denormals kept/flushed).
     Pins: NaN rules (which operand wins, quieting), signed zeros (−0 vs +0 order), denormal flushing by the DS unit.
   - `ds_cmpst_f64` and `_rtn`: 14 × 14 pairs with a distinct marker as data1. Pins the equality: ±0 equal, NaN
     never equal, sNaN, denormals under flush.
B. **Addressing.** Immediate offsets (`offset0:offset1`), an address at or past 4 KiB (expected: returns 0, writes
   nothing, as measured for the 32-bit forms in TD 273). A misaligned address (`addr % 8 == 4`) is reported to fault
   the wave on gfx1035; tried once, last, and recorded either way.
C. **Contention and mixed widths.** 256 lanes (8 waves in one workgroup) on one slot: `ds_add_rtn_u64` with values
   that carry across the dword (0xffffffff each), `ds_inc/dec_rtn_u64` at the wrap, `ds_max_rtn_u64`; the sum must be
   exact and the returned values one serial order. Then mixed: half the lanes `ds_add_u32` on the low dword, half
   `ds_add_u64`, both with 0x80000000, so every other add carries. Hardware serialises all LDS atomics per address,
   so the sum should be exact; the lock emulation does not order the 32-bit adds against the locked 64-bit ones, so
   on the hosts carries can be lost. Both wave sizes.
D. **EXEC masking.** Clear EXEC for half the lanes around the op and restore it: inactive lanes must not update.

E. **Hosts.** The same bodies as raw-word kernels through `KernelReplay` (`REPLAY_LDS`, `REPLAY_THREADS` 256) on the
   RTX 3080 and RADV, compared cell by cell with the oracle as in `scripts/oracle/check_prs.py`.

## Deliverables

1. **Measurement data** under `notes/oracle-lds64/` with the driver `scripts/oracle/lds64_matrix.py` and a fit
   script that checks the test's CPU model (`MinMax`, `FloatEqual`, carries, wrap) against every cell.
2. **PR, tests and TechnicalDebt:** hardware-pinned tables for the f64 forms, the carries and the contention and
   mixed-width cases (new execution test or additions to `LdsAtomics64.cpp`), the TD line changed from "not measured"
   to measured on gfx1036 with the rules as found; any divergence fixed in the same PR if it is small.
3. **PR, optional, after upstream agrees:** native 64-bit LDS atomics where the device has
   `shaderSharedInt64Atomics` and `workgroupMemoryExplicitLayout` (the LDS block declared aliased as u32 and u64),
   lock kept as the fallback. Removes the lock's hang class and the mixed-width gap, and the per-op elect loop.
   Coordination first: an issue with the measured mixed-width result if the hosts diverge, otherwise a note on #573's
   TD line in the PR.

Every public step (issue, PR, comment) needs Jan's go, a live overlap check and the reviewer pass.

## Status (2026-10-09 afternoon)

Phase 1 measured and fitted (0 mismatches against the rule below), hosts compared: see `notes/oracle-lds64/report.md`.
Hardware rule for the float forms = `AtomicFloatBits::minMax`/`equal` + flush of both operands by the mode's field;
integer forms, EXEC, offsets, contention exact; out of range returns 0; mixed 32/64-bit widths serialise. Recompiler
gaps: no sNaN rule on the shared paths (f32 and f64), driver-dependent float compares on f32, no flush, lost 32-bit increments beside locked 64-bit ones on RADV. Next: Jan's go on the PRs.

## Later, separate: the title path

When the dump of a title Jan owns arrives: the game list lives in the upstream repository's Discussions tab (Jan,
2026-10-09); next-stop method as for the homebrew, KytyPS5 as a reference for library semantics beside shadPS4.
