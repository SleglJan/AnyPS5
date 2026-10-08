Second pass of the audit above, covering the kernels that needed adapting (results outside `v10..v25`, more than 16 results, prologue setup, user-data SGPRs, computed inputs), plus the Division kernels and the four whose code arrays hold a named constant. Same GPU, oracle and seven modes; `main` 4b6ab6bb's test sources and recompiler; upstream has since changed F32MinMaxNan.cpp and added ImagePckN, InterpolationF16, InterpolationModes, LdsSrc2 and PackedAlu, which this pass does not cover. 221 adapters run (43 whole kernels, 168 parts of 16 kernels split by result columns, store-free word ranges or input groups, 4 single test cases, 6 wave64), 11 skipped: five branch cases whose path runs through a store (BranchPastEndpgm, LoopEndingExits), and six kernels that need more than the oracle's 4 KiB of LDS (four), eight inputs per lane, or an opcode with no gfx103x encoding. 62 kernels, 69,504 cells; a second oracle run of every adapter reproduced all 486,528 values except ShaderClock's masked clock columns. 18 kernels with table inputs also ran through the recompiler on both hosts; the 42 with computed inputs and the two that take user-data SGPRs did not.

**Asserted cells whose pinned value differs from the hardware in the titles' mode: 163**, every one a signalling NaN kept unquieted by the hardware with `IEEE_MODE=0` and quieted in the table, reproduced by `IEEE_MODE=1` only, as in the first pass.

| Test | Cells | Instructions | Test's own check on the hardware value |
|---|---:|---|---|
| Vop1FloatUnary | 118 | f32 ceil, floor, trunc, rndne, fract, frexp_mant, sqrt, rsq, rcp, rcp_iflag, log, exp, sin, cos; the same in f16 except frexp_mant and rcp_iflag | rejects |
| Division | 21 | `v_div_fmas_f32`, `v_div_fixup_f32` | rejects |
| Float16Misc | 12 | `v_ldexp_f16`, `v_frexp_mant_f16`, `v_mul_legacy_f32` | accepts (NaN-tolerant) |
| Float16Rounding | 12 | `v_add_f16`, `v_mul_f16` | accepts (NaN-tolerant) |

The other 14 kernels with a pin table (13 tests) match in all 16,960 asserted cells. Both hosts return the pinned value in 147 of the 163 cells; in the other 16, all in the NaN-tolerant tests, NVIDIA returns its canonical NaN, and RADV the hardware value in Float16Misc's four. So Vop1FloatUnary's and Division's sNaN rows assert the IEEE 1 result exactly, and the recompiler at 4b6ab6bb gives it in every mode. Unasserted cells whose pin differs: 82, all captured with f32 denormals kept (72 `v_mul_legacy_f32` lanes with an f32 denormal, where the hosts return the hardware's +0 in 35, -0 in 19 and the pin in 13, and differ from each other in 5; and 10 masked Vop1FloatUnary cells).

**Host versus hardware** on the 18 host-run kernels: 358 of 22,944 cells, none outside what the test's own check accepts; the sources are the quiet bit above, NVIDIA's canonical NaN, f32 denormal inputs the hosts do not flush (unasserted `v_mul_legacy_f32` lanes and Vop1FloatUnary row 0), and differences within the tolerance of the transcendental columns. In every cell of an integer, scalar or LDS instruction the oracle and both hosts agree, which is the check that those 18 adapters reproduce their kernels; RelativeIndexing is checked by its table alone, and the other 43 kernels compute their expectation in code and are compared across modes only.

<details><summary>Which field changes which result, over the 1,224 cells whose bits depend on the mode</summary>

| Field | Cells | Instructions |
|---|---:|---|
| f16/f64 denormals | 478 | f16 norm, pknorm, frexp, ldexp; f16 results with output modifiers; f16 add, cvt, mul; the compare bits of VopcCompare (4) and VopcCompareF64 (50) |
| `IEEE_MODE` | 351 (163 the quiet bit) | sNaN quieting above; `v_cvt_f32_f16 mul:4`, `v_cvt_f32_ubyte0 mul:2`, `v_cvt_f32_ubyte1 div:2` (188, with f32 denormals: the output-modifier rule) |
| f32 denormals | 324 | the same 188; `v_mul_legacy_f32` with a denormal lane (72); the LDS float atomics' columns (54); 10 masked Vop1FloatUnary cells |
| `FP16_OVFL` | 247 | `v_ldexp_f16`, `v_cvt_f16_f32` (also with `div:2`), `v_cvt_f16_u16`, f16 add and mul, f16 mul, fma and rcp with `mul:2`, f16 exp, log, rcp, rsq |
| `DX10_CLAMP` | 21 | clamped f16 results, 16 of them `v_sqrt_f16 clamp mul:2` |

</details>

Not measurable here: FmaRounding columns 4 to 7 and 9 and Float16Misc column 10, whose instructions exist only on gfx10.1; Float16Misc word 36 decodes as `v_mac_legacy_f32` on gfx1013 and as the fused `v_fmac_legacy_f32` on gfx1036, so its column is excluded. Over both passes: 330 asserted cells in 13 tests whose pinned value is the `IEEE_MODE=1` result or, in F64Rounding, the ignored output modifier, and 50 Vop3PackLdexp cells of an opcode the oracle cannot run. The adapters, the driver, every run's body and raw output, the per-cell data and the listings for both passes are on a data branch of the fork: https://github.com/SleglJan/AnyPS5/tree/audit/oracle-2026-10-08/oracle-audit.

AI-assisted: yes (Claude Code).
