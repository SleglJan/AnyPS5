# Float-mode audit of AnyPS5's execution tests on RDNA2 (gfx1036)

Data behind the comment on boykopovar/AnyPS5#981. Measured 2026-10-08 on a Ryzen 7950X3D iGPU (Raphael, gfx1036) with
`tools/hw-oracle` (hsa-rocr 7.2.4, Linux), on `main` 4b6ab6bb's test sources. Hosts: RTX 3080 (NVIDIA) and the RADV iGPU.

Files:
- `audit.json`: per kernel, every flagged result cell with the oracle's value in seven modes (`title`, `old`, `ieee0_keep32`,
  `ieee1_flush32`, `title_flush16`, `title_noclamp`, `title_fp16ovfl`, defined in `audit_modes.py`) and both hosts' values;
  `cells` is the total measured per kernel (27,652 over the 47 driven kernels).
- `execution-manifest.json`: classification of the 186 execution tests' kernels for replay (`yes`/`partial`/`no`, word ranges, register maps).
- `pinned-vs-hardware.txt`: the tests' own `Expected` tables, bit-exact with `Checked` masks, against the titles'-mode hardware value,
  with the mode that reproduces each pin (`forensics_allpinned.py`).
- `mode-sensitivity.txt`: per kernel and column, which mode differs from the titles' mode (`forensics_sens.py`).
- `host-test-runs.txt`, `run-log.txt`: the 48 tests run on both hosts (96/96 pass) and the audit run header.
- `verify-bodies.s`, `verify-rows.txt`: the single-instruction checks quoted in the comment (omod, `v_min_f64` with sNaN, denormals).

Reproduce (from a workspace with `AnyPS5/` checked out and built, `build/` beside it):
1. `KernelReplay.cpp` goes in `core/libs/prx/libSceAgcDriver/tests/execution/`; build `agc_driver_kernel_replay_tests`.
   It reads `REPLAY_CODE` (kernel words), `REPLAY_ROWS`, `REPLAY_INPUTS`, `REPLAY_RESULTS`, `REPLAY_THREADS`, `REPLAY_WAVE` and
   prints each lane's results; `ANYPS5_GPU` selects the host.
2. `python3 -s audit_modes.py execution-manifest.json out/` runs every `yes` kernel on both hosts and the oracle in the seven modes.
3. `python3 -s forensics_allpinned.py out/` compares the tests' pinned tables against the titles'-mode hardware values.
`ANYPS5_AUDIT_ROOT` points the forensics scripts at the workspace root when they are not run from inside it.

## Layout of this branch (fork-only data branch, nothing here is meant for `main`)
- `run1/`: first pass (47 directly replayable kernels): audit.json, mode-sensitivity.txt, pinned-vs-hardware.txt, host-test-runs.txt, verify/.
- `adapters/`: the 232 adapter files for the partially replayable kernels, their README (schema and status), check_adapters.py, generated/.
- `run2/`: second pass (221 adapters): audit.json, values.json (every cell), pinned-vs-hardware.txt/.json, mode-sensitivity.txt, summary.txt,
  repeatability.txt, skipped.json, report.md, work/ (every run's body, rows and raw oracle output).
- `minmax/`: f64 and f16 min/max matrices (matrix.json); `fma-f64/`: v_fma_f64 subnormal matrix.
- `scripts/`: audit_modes.py (pass 1), audit_adapters.py (pass 2), srctables.py, extract_kernel.py, classify.py, minmax_matrix.py, minmax_fit.py,
  fma_f64_subnormal.py, forensics/ (allpinned.py, allpinned2.py, sens.py, sens2.py, summary2.py, expectations.py, colinstr.py, repeat2.py, ...).
- `KernelReplay.cpp`: the host replay harness (an execution-test target); `execution-manifest.json`: the replayability classification.
