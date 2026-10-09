# Transcendental sweep on gfx1036 (RDNA2), 2026-10-09

Oracle captures (`tools/hw-oracle`, Ryzen 7950X3D iGPU, IEEE 0, f32 denormals flushed or kept as named, f16 kept, DX10_CLAMP 1, RNE):
- `f16-title.txt.gz`: all 65,536 f16 inputs, columns v_rcp/rsq/sqrt/log/exp/sin/cos_f16.
- `f32-flushed.txt.gz`, `f32-kept.txt.gz`: the 32,832 rows of `f32-inputs.json` (every exponent with 64 mantissas of both signs, 13 specials, 26 cycle-count edges, 25 zero padding rows), columns v_rcp, v_rsq, v_sqrt, v_log, v_exp, v_sin, v_cos, v_rcp_iflag _f32.
- `dense-sincos.txt.gz`: 20,480 inputs in (0, 0.25] (`dense-sincos-rows.txt.gz`), columns v_sin_f32, v_cos_f32.
- `fold-constants.txt`: 1,920 inputs at q = 2^-k (1 + m/16), k = 8..22, m = 0..15, as q, 0.5 - q, 0.5 + q, 1 - q, -0.5 + q, 0.25 - q, 0.25 + q, 0.75 - q (`fold-constants-rows.txt`), columns v_sin_f32, v_cos_f32.
- `trig-test.txt`: the 32 rows of the execution test `agc_driver_sin_cos_near_zero` (`trig-test-rows.txt`: f32 in column 0, f16 in column 1); `trig-test-<variant>-<GPU>-results.txt`: the same rows through the recompiler at #1979 (`pr1979b`) and with the follow-up (`band`), columns v_sin_f32, v_cos_f32, v_sin_f16, v_cos_f16 (`scripts/trig_lane_distances.py`).

Host replays through the recompiler (`scripts/transcendental_hosts.py`, kernel-replay harness; Ryzen = RADV Mesa 26.2.4 on the iGPU, NVIDIA = RTX 3080 615.71.09):
- `hosts-report-main.json`: upstream main d70b8998 (fract-based reduction).
- `hosts-report-pr1979.json`: PR #1979 at e6eb64aa (x - roundEven(x)).
- `hosts-report-fold2.json`, `hosts-report-fold3.json`: the quarter-turn fold variants (linear below 2^-14 and 2^-12), measured and rejected.
- `hosts-report-band.json`: the follow-up as opened (linear within 2^-12 of a zero).
- `hosts-<variant>-<set>-<GPU>-results.txt.gz`: per-input host results (16 dwords per row, the op columns as above) for the last four variants; `dense-<variant>-<GPU>-results.txt.gz`: the dense sample through the same builds.
Scripts in `../scripts/`: `transcendental_sweep.py`, `transcendental_hosts.py`, `fold_constants.py`, `trig_make_test.py`, `trig_compare_variants.py`, `trig_dense_compare.py`, `trig_lane_distances.py`, `trig_apply_band.py`, `trig_apply_fold_wide.py`.
