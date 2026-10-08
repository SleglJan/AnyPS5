# Oracle adapters for the execution-test kernels

Adapters that make a `tools/hw-oracle` run equivalent to an execution test's kernel, for the 59 kernels the
manifest (`notes/execution-manifest.json`) classifies `"replayable": "partial"`, plus the "yes" kernels whose inputs
are computed in code (ReadLane, IntegerDot, ScalarBitCount, Vop3CarryOut, Vop3Lerp, Vop3Sad) and the three
Division.cpp kernels. FmaLegacy.FmaLegacyCode was checked and needs none.

Nothing here was run on a GPU. The repository was not modified. Oracle target assumed: gfx1036 (the 7950X3D iGPU).

## Counts

| | kernels | adapter files |
|---|---|---|
| partial kernels, **ready** (one adapter, or one per wave size) | 34 | 39 |
| partial kernels, **split** into N parts (>16 columns, or per-chunk/per-input-group) | 16 | 168 |
| partial kernels, **some test cases ready, others skip** (BranchPastEndpgm ×2, LoopEndingExits) | 3 | 9 (4 ready: BranchPastEndpgm .case0/.case3, LoopEndingExits .case4.wave32/.wave64; 5 skip records) |
| partial kernels, **skip** | 6 | 6 |
| yes kernels with computed inputs (6, incl. ReadLane) + Division (3), ready | 9 | 10 |
| FmaLegacy (no adapter needed) | 1 | 0 |
| **total** | 68 kernels with adapter files | **232 JSON files: 221 ready/split, 11 skip** |

`python3 -I check_adapters.py` passes all 232 files: each one is "ok" or SKIP. Run it in parallel, e.g. `ls *.json | xargs -P 32 -n1 python3 -I check_adapters.py`. Sequentially it takes about 15 min because `disasm.py` starts llvm-mc once for every word position it probes.

## Schema

One JSON object per adapter, file name `<FileStem>.<Kernel>[.<variant>].json`.

| field | type | meaning |
|---|---|---|
| `file`, `kernel` | str | test source (`Foo.cpp`) and kernel array name |
| `manifest_class` | str | `partial` or `yes` |
| `status` | str | `ready`, `split N/M`, or `skip` |
| `skip` | bool | true if this adapter must not be run; then `reason` (str) says why. Several skip files still contain a complete pre/post for later use (DsWrxchg2B64). |
| `words` | [int, int] | `[start, end)` word indices into the kernel, end exclusive, same convention as the manifest's `alu_word_range`. Only these words go into the body, as `.long`, unmodified. |
| `words_file` | str, optional | `generated/<File>.<Kernel>.words.txt` (one `0x` word per line). Use it **instead of** the C++ array when present. It is needed for constexpr-generated code (ScalarSopkCompare, ScalarSopkWaitcnt, SdwaMoveSelectors), for `std::to_array` arrays with named constants (DsAppendConsume, DsWrxchg2, DsWrxchg2B64), and for arrays that contain a named `Literal` (IntegerDot, Vop3CarryOut, Vop3Lerp, Vop3Sad; see "Driver bug" below). |
| `pre` | [str] | assembly lines (or `.long`) placed before the words: sentinel inits, user-data SGPR constants, M0, prologue-computed address registers, input re-mapping (`v_mov_b32 v14, v6`) |
| `post` | [str] | lines after the words: moves into v10..v25, `v2` recomputation, EXEC restore, and for SdwaMoveSelectors further unmodified kernel words (see `segments`) |
| `columns` | {str: int} | **result column** (the test's Output column `c`, as a decimal string) → **oracle output index** 0..15 (= v(10+index)). A split part lists only its slice, e.g. `{"16": 0, ..., "31": 15}`. Columns the test never asserts are omitted. |
| `column_masks` | {str: str}, optional | result column → hex mask. Compare `(oracle & mask) == (host & mask)` (VopcCompare, VopcCompareF64: each part only produces some of the bits). |
| `rows` | str or object | how to build the four input dwords per lane (below) |
| `row_count` | int | number of oracle rows (lanes) to run. It must match the test's thread count whenever lanes or waves interact (cross-lane, LDS, readlane). |
| `wave64` | bool | pass `--wave64` |
| `lds` | int | LDS bytes the replay touches (≤ 4096) |
| `segments` | [object], optional | SdwaMoveSelectors only: each `{column, words, replaces_store_at, captured_into, combination}` records which kernel words post replays and which store it replaces |
| `accepted_flags` | {str: str}, optional | word index → reason. Checker flags at that word were reviewed and are accepted: never-taken branches out of the body, intended writes to v1/v2. |
| `notes` | [str] | what the comparison must know |
| `verified` | str | what was checked by disassembly / Run()/Check() reading vs inferred |

`rows` forms:
- `"first4"`: the test's Rows table, inputs 0..3, exactly as `audit_modes.py` builds rows today.
- `"unsupported: needs N inputs"`: wait for `--extra` support.
- `{"expr": "<python>", "count": N}`: a Python expression over the lane index `tid` that gives the four u32 values. It is evaluated with no builtins, `eval(expr, {"__builtins__": {}}, {"tid": tid})`. It is copied from the test's Run()/FillInput() with explicit `& 0xffffffff`. Constant tables are inlined as lambda arguments, e.g. `(lambda S, tid: [...])([...], tid)`. Some carry `"source"`, saying where the formula comes from. Group C and E checked their expressions against a g++ build of the test's FillInput() for every lane.
- `{"table": "<C array>", "layout": "...", "count": N, ["words": [field idx or null ×4]], ["file": "generated/...rows.txt"]}`: rows from a C array of structs. `words` gives which struct field goes into each dword (null = 0). `file`, when present, holds the extracted rows (Division). `Rows` tables with `count` smaller than the table use its first `count` rows.

How multi-file adapters relate to one kernel:
- `.partN.json` (`status: "split N/M"`) are column slices of one kernel. Usually they share `words` and differ in `post`/`columns`. Exceptions: ScalarSop1, ScalarSopkCompare and ScalarSopkWaitcnt have one word range per store-free chunk, FmaRounding splits by word range to avoid gfx1013-only words, and VopcCompare/VopcCompareF64 share words but use different rows and pre per input group. The union of the parts' `columns` is the kernel's comparable set.
- `.caseN.json`: one adapter per test case when the test picks the code path with a user-data SGPR (`s8` = `Cases[N]`'s selector, set in `pre`). Example: `BranchPastEndpgm.LoopBodyCode.case3.json` replays words [4,8) with `s_mov_b32 s8, 0`, so it covers Cases[3] only. Cases 4 and 5 of the same kernel have their own skip files (`.case4.json`, `.case5.json`). For LoopEndingExits, Cases[4] is ready (`.case4.wave32.json`, `.case4.wave64.json`), and `LoopEndingExits.Code.json` is the skip record for Cases[0..3].
- `.wave64.json`: the test also runs the same words at wave64 (ScalarStaticCall ×5, Vop3CarryOut).

## Kernel table

Legend: **ready**, **split N**, **skip**. "cols" = asserted columns covered.

| Kernel | Adapter(s) | Status | Notes |
|---|---|---|---|
| BranchPastEndpgm.TailBlockCode | .case0 / .case1, .case2 | case 0 ready; cases 1–2 skip | cases 1–2 reach their result through blocks interleaved with store/`s_endpgm` |
| BranchPastEndpgm.LoopBodyCode | .case3 / .case4, .case5 | case 3 ready; 4–5 skip | the loop body sits after a store/`s_endpgm` |
| CacheControl.CacheControlCode | .json | ready | words [17,20); pre `s_incperflevel 1`; no data effect from the hints (inferred) |
| DsAppendConsume.Code | .json + words_file | ready | M0/EXEC set and restored in the body; LDS ≤159 B; 12 cols |
| DsWrxchg2.Code | .json + words_file | ready | pre `v30=lane*4`; LDS ≤1023 B; 16 cols |
| DsWrxchg2B64.Code | .json | skip | needs 8 inputs in v2..v9 (v2 is reserved); pre/post already written for when `--extra` exists |
| Float16Misc.MiscCode | .json | ready (col 10 excluded) | word 36 is `v_mac_legacy_f32` on gfx1013 but `v_fmac_legacy_f32` (fused) on gfx103x |
| Float16ResultModifiers.Code | .json | ready | the template's zeroed v10..v25 already matches the prologue |
| Float16Rounding.RoundingCode | .json | ready | sentinels v10–v12; Vectors table |
| FmaRounding.FmaCode | .part1–3 | split 3 | words 21, 24, 25, 27, 32 are gfx1013-only (`v_mad_f32`, `v_mac_f32`, `v_madak`, `v_madmk`): **cols 4–7, 9 not measurable on gfx1036**; cols 0–3, 8 covered |
| LdsAddtid.Wave32Code | .part1–3 | split 3 | 36 cols; the test allocates the same 4096 B, so the out-of-range probes stay comparable |
| LdsAddtid.Wave64Code | .json | skip | needs 8 KiB LDS; offsets are immediates, so they can't be rebased |
| LdsAtomics.Code | .part1–2 | split 2 | pre `v30=lane*4`; LDS ≤2175 B; float cols not asserted for denormal lanes |
| LdsAtomics64.Code | .json | skip | immediates up to 8968 B; 6 inputs; uses v2 as base |
| LdsIntegerAtomics.Wave32Code | .part1–3 | split 3 | exactly 32 rows (contended atomics) |
| LdsIntegerAtomics.Wave64Code | .json | skip | needs 8 KiB LDS |
| LdsReadWrite.Wave32Code | .part1–4 | split 4 | 58 asserted cols; v68/v69 not asserted |
| LdsReadWrite.Wave64Code | .json | skip | needs 8 KiB LDS |
| LoopEndingExits.Code | .case4.wave32, .case4.wave64 / .json (skip) | Cases[4] ready; Cases[0..3] skip | with s8=5 the exits at words 6 and 10 are never taken (traced) |
| MixPrecision.MixCode | .json | ready | pre v16/v17 sentinels; computed rows |
| ReadFirstLane.ReadFirstLaneWave32Code | .json | ready | body overwrites v2; post recomputes it; dummy rows |
| ReadFirstLane.ReadFirstLaneWave64Code | .json | ready | same as wave32 |
| RelativeIndexing.Code | .json | ready | M0 set in the body; col 10 deliberately out of range |
| ScalarGetreg.Code | .json | ready | result = the oracle's own MODE round bits; it matches the expected 0 only when round32 = round16 = 0 |
| ScalarProgramEnd.Code | .json | ready (ALU only) | words [15,21); excludes `s_atc_probe*`, `s_dcache_discard*` (gfx1013-only), `s_round_mode`; only col 0 |
| ScalarProgramEnd.SetkillCode | .json | ready (ALU only) | `s_setkill` is not replayed; only col 0 |
| ScalarProgramEndSaved.Code | .json | ready (ALU only) | as ScalarProgramEnd.Code |
| ScalarProgramFlow.Wave32Code | .json | ready | cols 0–13; col 12 assumes no debugger (cdbg branches) |
| ScalarProgramFlow.Wave64Code | .json | ready | same |
| ScalarRelativeIndexing.Code | .json | ready | col 10 = out-of-range movreld behaviour (a hardware question) |
| ScalarSethalt.Code | .json | ready (ALU only) | words [5,7); col 1 not mapped |
| ScalarSop1.Code | .part1–11 | split 11 | one store-free chunk per part; per-lane readlane loops over exactly 32 lanes; parts 6–10 exercise exec_hi under wave32 |
| ScalarSop2.Code | .part1–4 | split 4 | cols 14, 30, 60, 61, 62 never written; the manifest's "unset v26.." is writelane read-modify-write |
| ScalarSopkCompare.CompareCode | .part1–60 + words_file | split 60 | one block per column; pre s8..s17 = Values[] (userData[8..17]) |
| ScalarSopkMisc.Code | .json | ready | `s_version` is an annotation (manifest false positive); cols 0–3 |
| ScalarSopkWaitcnt.WaitCode | .part1–24 + words_file | split 24 | pre s8 = 0x12345678; col 2c+1 via post `v_mov_b32 v11, s8` |
| ScalarStaticCall.SwappcForward | .json, .wave64 | ready | every PC-relative target computed by hand; all stay inside the body |
| ScalarStaticCall.CallForward | .json, .wave64 | ready | same |
| ScalarStaticCall.CallBackward | .json, .wave64 | ready | same; "reads s8,s9" in the manifest is a false positive |
| ScalarStaticCall.CallAfterEnd | .json, .wave64 | ready (ALU only) | only the callee words [8,15); the call/return itself is not exercised |
| ScalarStaticCall.MixedNested | .json, .wave64 | ready | all targets inside [4,21) |
| SdwaDestination.SdwaCode | .json | ready | sentinels 0xabcd1234; results 4–7 go to v15–v18 (indices 5–8) |
| SdwaMoveSelectors.Wave32Code | .part1–19 + words_file | split 19 | 294 combinations; post replays each 3-word combination (unmodified) and captures v6 instead of storing it |
| SdwaMoveSelectors.Wave64Code | .part1–19 + words_file | split 19 | same |
| ShaderClock.Code | .json | ready | cols 0–7 are clock values: compare only the relations Check() asserts; col 8 exact |
| Vop1Control.Vop1ControlCode | .json | ready | post `v_mov_b32 v10, v4` |
| Vop1FloatUnary.Code | .part1–4 | split 4 | 54 cols; per-lane Checked masks and tolerances |
| Vop1Vop2Integer.Code | .part1–3 | split 3 | 41 cols |
| Vop3Aliases.AliasCode | .json | ready | only the f16 low half of cols 0–2 is compared |
| Vop3CompareModifiers.ModifierCode | .json | ready | pre v14←v6, v15←v7 (the body never reads v6/v7) |
| Vop3Integer.IntegerCode | .json | ready | cols 3, 4, 5, 11, 12 compared on the low 16 bits |
| Vop3IntegerAlu.Code | .part1–4 | split 4 | 51 cols |
| VopcCompare.CompareCode | .part1–3 | split 3 | 8 inputs, handled by per-input-group rows/pre + `column_masks` |
| VopcCompareF64.CompareCode | .part1–2 | split 2 | same approach; 544 rows |
| Wave32Subgroup.WaveCode | .json | ready | body overwrites v2; post restores it; 64 rows = 2 waves |
| WaveIdInWorkgroup.WaveIdCode | .json | skip | `s_get_waveid_in_workgroup` has no gfx103x encoding; also writes s4 |
| WaveSleep.WaveSleepCode | .json | ready | pre `s_wakeup` |
| WaveUniform.WaveCode | .json | ready | `s_buffer_load` replaced by `s_mov s8,3; s_mov s9,5`; **wave64** (the manifest says false) |
| WriteLane.WriteLaneCode | .json | ready | prologue (exec=1 + writelane) in pre |
| ReadLane.ReadLaneCode (yes) | .json | ready | rows `0xa0000000+tid*0x01010101`, `69+tid`; exactly 32 rows |
| IntegerDot.DotCode (yes) | .json + words_file | ready | cols 0–9 |
| ScalarBitCount.BitCountCode (yes) | .json | ready | 32 rows, wave32; loop branch stays inside the body |
| Vop3CarryOut.CarryOutCode (yes) | .json, .wave64 + words_file | ready | 16 cols |
| Vop3Lerp.LerpCode (yes) | .json + words_file | ready | cols 0–7 |
| Vop3Sad.SadCode (yes) | .json + words_file | ready | cols 0–9 |
| FmaLegacy.FmaLegacyCode (yes) | — | needs none | `first4`; 20 rows parse with `parse_rows` |
| Division.HelperCode (yes) | .json | ready | rows file from HelperVectors |
| Division.DivisionCode (yes) | .json | ready | rows file from DivisionVectors |
| Division.FmaCode (yes) | .json | ready | single fused-FMA probe |

### Division.cpp
Each kernel is its own 32-lane shader. The test feeds a table 32 vectors per dispatch (8 dispatches). Lanes are independent, so one oracle run over all 256 rows is equivalent. The manifest's "512 rows" is the two tables added together.
- **HelperCode ← HelperVectors[256]**: lane i gets `{s0, s1, s2, vcc}` in v4..v7. `scale`, `scaleVcc`, `fmas` and `fixup` are expected outputs. Output column c = `Output[lane*4+c]`. Cols 0/1 are asserted only when s0==s1 or s0==s2 (210/256 vectors). Col 2 is asserted only when the host fuses FMA. Col 3 is always asserted. Rows: `generated/Division.HelperCode.rows.txt`.
- **DivisionCode ← DivisionVectors[256]**: lane i gets `[numerator, denominator, 0, 0]`; the quotient is expected. It runs only on hosts that fuse FMA. Rows: `generated/Division.DivisionCode.rows.txt`.
- **FmaCode**: reads neither table. Row 0 = `[0xbff2fb6d, 0xbcaac14c, 0xbd22126e, 0]`; only Output[0] matters (0x2f8e8ae0 means the host fuses FMA).

No Division kernel reads more than 4 dwords per lane.

## Findings that affect the existing audit

1. **Driver bug: named constants in kernel arrays are dropped.** `audit_modes.words_of` (and the first version of `disasm.py`) only collects hex literals. Kernels whose arrays contain a named constant therefore lose that word, and every later word shifts down one place. The affected yes kernels are IntegerDot, Vop3CarryOut, Vop3Lerp and Vop3Sad; the "no" kernels TypedHeaps, ViewPastLastMip ×2 are affected too. For these four, the oracle run in `oracle-audit-run1` replayed a **wrong body**: it lost the literal and pulled in part of a store. Their oracle results there are invalid. The adapters use `words_file`.
2. **Same encoding, different instruction.** VOP2 opcode 6 is `v_mac_legacy_f32` on gfx1013 but `v_fmac_legacy_f32` on gfx103x. A gfx1013-vs-gfx1030 cross-decode of every adapter's words found only this one case (Float16Misc word 36). Encodings that exist only on gfx1013 (`v_mad_f32`/`v_mac_f32`/`v_madak`/`v_madmk`, `s_dcache_discard*`, `s_dcache_wb`, `s_get_waveid_in_workgroup`) are kept out of every ready range.
3. LLVM's gfx1013 target has no `v_dot*` (IntegerDot words decode only for gfx1030/1036). Whether the PS5's gfx1013 hardware has them is not something the decoder can tell.
4. Manifest errors: WaveUniform is wave64 (the manifest says `wave64_variant: false`). ScalarSopkMisc's `s_version` and ScalarStaticCall.CallBackward's "reads s8,s9" are false positives.
5. To consume these adapters, `audit_modes.py` needs: `words_file`; pre/post around the `.long` words; `columns` remapping instead of comparing column c with v(10+c); `column_masks`; the rows forms above; `row_count`; and per-case/part iteration.

## What was verified and what was inferred

**Verified offline:**
- Every adapter's `words` range was disassembled with llvm-mc (gfx1030 per position, with a gfx1013 fallback for rejected encodings), and cross-decoded gfx1013 vs gfx1030 to catch semantic differences.
- Every adapter's pre + words + post assembles into `template.s` for gfx1036 at its wave size (`check_adapters.py`).
- The checker also verifies, per adapter: branch targets stay in range or are explicitly accepted with a traced reason; no memory/cache/`s_endpgm` instructions in the words; no writes to v0–v2/s4–s7 unless accepted; SGPRs < s48 and VGPRs < v128; a valid column map; rows expressions evaluate to 4 u32 values per lane; lds ≤ 4096.
- Each adapter author read the test's Run() (thread count, user data, input layout/stride, prologue) and Check() (asserted columns, tolerances).
- Rows expressions for MixPrecision, SdwaDestination, SdwaMoveSelectors, Vop3Aliases, Vop3CompareModifiers, Vop3Integer, VopcCompare*, IntegerDot, ScalarBitCount, Vop3CarryOut, Vop3Lerp and Vop3Sad were checked against a g++ build of the test's FillInput().
- Generated words (ScalarSopkCompare 3122, ScalarSopkWaitcnt 362, SdwaMoveSelectors 1491/1493, the to_array kernels) come from g++ builds of the test's own generator; their lengths match the manifest's word counts.

**Inferred, not measured:**
- That the gfx1036 oracle behaves like gfx1013 for everything the adapters keep. This rests on decode equivalence only.
- That cache/perf/trace hints (`s_inc/decperflevel`, `s_ttracedata*`, `s_sethalt 0`, `s_sleep`/`s_wakeup`, `s_cbranch_cdbg*` without a debugger) have no data effect.
- The hardware behaviours tested on purpose: out-of-range LDS reads and addtid probes at 4 KiB, out-of-range `movreld`, `exec_hi` under wave32 (ScalarSop1 parts 6–10), `v_readfirstlane` with EXEC=0, and the MODE bit layout for ScalarGetreg. A mismatch on any of these is a finding, not an adapter bug.
- That the template's zeroed v10..v25 stands in for registers the PS5 kernels leave uninitialised (e.g. Vop1Vop2Integer v22–v25).
- The HelperVectors/DivisionVectors rows were extracted by regex (all 256 rows parsed) and were not cross-checked against compiled C++. The Float16Misc/Float16Rounding `Vectors` tables are referenced by name with field indices; no rows file was generated for them.
- "ALU only" adapters (ScalarProgramEnd*, SetkillCode, ScalarSethalt, CallAfterEnd) check only the plain ALU result. The behaviour those tests target (stores after program end, a call after `s_endpgm`) cannot be replayed on the oracle.

## Files

- `*.json`: the adapters.
- `generated/*.words.txt`: kernel words (generated or with constants substituted).
- `generated/Division.*.rows.txt`: Division rows.
- `check_adapters.py`: offline checker.
- `disasm.py`: word-indexed disassembler (`python3 -I disasm.py File.cpp Kernel`). It doesn't handle `std::to_array` arrays or generated code; use `words_file` for those.
