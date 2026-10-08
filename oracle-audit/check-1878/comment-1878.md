Reproduced on a second RDNA2 part, the Ryzen 7950X3D iGPU (gfx1036) through `tools/hw-oracle` (hsa-rocr 7.2.4): both rows of the table give `v_fma_mix_f32` = `v_fma_f32` (`0xa8800000`, `0x34e00001`) and `v_mul_f32` + `v_add_f32` = `0x00000000`, `0x35000000`, with IEEE 0 and f32 denormals flushed as well as with IEEE 1 and denormals kept. The MixPrecision execution kernel (153 cells of its hardware table, f16-sourced) gives the same bits on both hosts (RTX 3080, RADV) before and after this change, as the description expects of a value test.

AI-assisted: yes (Claude Code).
