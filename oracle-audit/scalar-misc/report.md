# Scalar calls, null swappc, shader clocks, v_cmp_lg encodings (gfx1036, 2026-10-09)

Oracle: Ryzen 7950X3D iGPU (gfx1036, RDNA2), `tools/hw-oracle`, Linux (CachyOS, kernel 7.2.9). Script
`scripts/oracle/scalar_misc.py`; bodies, rows and outputs in this directory; `results.json`.

- **`s_call_b64`** (wave32 and wave64, 32 or 64 lanes, all uniform): the link pair receives `PC + 4` (the address of the
  instruction after the call; measured as 4 bytes past the preceding `s_getpc_b64` result), the branch goes to
  `PC + 4 + simm16 * 4` (simm16 = 3 skipped three dwords), and `s_setpc_b64` of the link returns to the instruction
  after the call (an order counter reads 3: callee +1, then the return path +2). This is the ISA model the recompiler
  follows (TechnicalDebt "s_call_b64").
- **`s_swappc_b64` with the null destination (SDST 125)**: the hardware branches to the source pair and writes no
  link (marker values in s14:s15 survive); the instruction at the skipped address is not executed. With an SGPR
  destination the link is `PC + 4` (12 bytes past the `s_getpc_b64` three dwords earlier) and the branch is taken.
  So decoding it as `s_setpc_b64` matches the hardware. Both wave sizes.
- **`s_memtime`, `s_memrealtime`, `HW_REG_SHADER_CYCLES`** before and after a counted loop (two `v_add_f32` and the
  loop control per iteration):

  | iterations | `s_memtime` delta | `s_memrealtime` delta | SHADER_CYCLES delta (20 bits) | ratio |
  |---:|---:|---:|---:|---:|
  | 1,000 | 23,022 | 3,808 | 23,022 | 6.0 |
  | 20,000 | 460,022 | 50,396 | 460,022 | 9.1 |
  | 400,000 | 9,200,022 | 430,902 | 811,414 | 21.4 |

  `s_memtime` is a 64-bit shader-cycle counter (23 cycles per iteration throughout, no wrap at 2^20);
  `SHADER_CYCLES` is its low 20 bits (811,414 = 9,200,022 mod 2^20); `s_memrealtime` counts the 100 MHz reference
  clock; their ratio is the shader clock, which ramped from 0.6 GHz to 2.1 GHz across the three runs. All reads
  uniform over the wave.
- **`v_cmp_lg_f32` / `v_cmp_lg_f16`, e32 (VCC) versus e64 (SGPR)**, wave32, 13 × 13 operand pairs per type (±0, ±1,
  ±inf, a denormal, quiet and signalling NaNs of both signs, 0xfffffffe, 1 + 1 ULP): identical in every pair; a NaN
  operand gives 0; `v_cmp_nlg` is the complement in every row; with f32 denormals flushed the denormal operand
  compares equal to zero (10 pairs), f16 denormals kept. Issue #825's RDNA 3.5 Windows failure is not a hardware
  property.

A first swappc body used literal markers (8-byte `s_mov_b32`), so a +16 target landed inside a literal and the wave
faulted (the oracle process died with SIGPIPE, like the misaligned LDS case). Inline-constant markers fixed it.
