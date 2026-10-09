# v_log_f32 near 1.0 on gfx1036 (RDNA2), 2026-10-09

Oracle captures (`tools/hw-oracle`, Ryzen 7950X3D iGPU, IEEE 0, f32 denormals flushed, f16 kept, DX10_CLAMP 1, RNE):
- `log-near-one.txt`: 2,368 rows x = 1 ± 2^-k (1 + m/64), k = 1..24, m = 0..63 (`log-near-one-rows.txt`), column v_log_f32.
- `log-dense.txt.gz`: 16,384 rows x = 0.875 + i × 2^-15 (`log-dense-rows.txt.gz`), column v_log_f32.
- `log-test.txt`: the 64 rows of the execution test `agc_driver_log_near_one` (`log-test-rows.txt`: f32 in column 0, f16 in column 1), columns v_log_f32, v_log_f16.
Host replays through the recompiler (kernel-replay harness; Ryzen = RADV Mesa 26.2.4 on the iGPU, NVIDIA = RTX 3080 615.71.09): `hosts-<main|fix>-near-one-<GPU>.txt`, `hosts-<main|fix>-dense-<GPU>.txt.gz`, `log-test-<main|fix>-<GPU>-results.txt` (16 dwords per row, column 0 = v_log_f32, 1 = v_log_f16), where `main` is upstream main 0fdda7be and `fix` the branch fix/shader-log2-near-one.
Scripts in `../scripts/`: `log_near_one.py`, `log_dense.py`, `log_make_test.py`, `log_lane_distances.py`, `log_apply_series.py` (the lowering patch).
