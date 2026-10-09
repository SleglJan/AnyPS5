# Cube output modifier measurements (RDNA2)

Raw data behind SleglJan/AnyPS5#21 and the comment on boykopovar/AnyPS5#2463: how `v_cubeid_f32`, `v_cubesc_f32`,
`v_cubetc_f32` and `v_cubema_f32` apply the VOP3 output modifier (`mul:2`, `mul:4`, `div:2`).

Hardware: Ryzen 9 7950X3D integrated GPU (Raphael, gfx1036, RDNA2; not the PS5's gfx1013), run through hsa-rocr with
AnyPS5's `tools/hw-oracle/hw_oracle.py`. Each input row is loaded into `v4..v7`; results are read from `v10..v25`, one
line per row, one hex word per register. All inputs are synthetic.

## sweep/

- `rows.txt`: 2048 rows (x, y, z, unused): special values (zeros, denormals, smallest normals, largest finite, infinities,
  NaNs) in each position, then random words with exponents spread over the whole range.
- `body-a.s`: 16 columns, for each cube instruction: no modifier, `mul:2`, `mul:4`, `div:2`.
- `body-b.s`: 14 columns: negated-source forms with and without `div:2`, `clamp` alone and with each modifier, and two
  `v_mul_f32_e64` controls (`div:2`, `mul:2`).
- `out-{a,b}-i<IEEE>-d<f32 denorm>-h<f16/f64 denorm>.txt`: 16 modes, IEEE 0/1 x f32 denormal field 0-3 x f16/f64 field
  0 or 3; round to nearest even (round32 = round16 = 0), DX10_CLAMP = 1, fp16 overflow 0. Command:

  ```sh
  python3 -s tools/hw-oracle/hw_oracle.py --ieee I --denorm32 D --denorm16 H --dx10-clamp 1 \
      --round32 0 --round16 0 --fp16-overflow 0 --outs 16 body-a.s rows.txt
  ```

- `check.py <sweep dir>`: compares every modifier column with the unmodified column, using a Python port of AnyPS5's
  bit-level f32 output modifier (`applyF32ResultModifiers` in `OperandAccess.cpp`), and checks clamp after the modifier.

Result: with IEEE = 0 the modifier is applied in all eight denormal combinations and matches the port on every row
(21,768 of the 24,576 results per mode differ from the unmodified value); with IEEE = 1 it is ignored. The
`v_mul_f32_e64` controls follow AnyPS5 #1851's rule instead (ignored with f32 output denormals kept). Only round to
nearest even was measured; with the other rounding modes `mul` overflow gives the largest finite value.

## test-tables/

The 16-instruction kernel of `agc_driver_vop3_cube_clamp` (`kernel.s`, `body.s`), its 32 input rows and the hardware
results in IEEE 0/1 with f32 denormals flushed (`d0`) and kept (`d3`), f16/f64 denormals kept. The test's tables are
`hw-i0-d0.txt` and `hw-i1-d0.txt`; `d3` is identical to `d0` for both.
