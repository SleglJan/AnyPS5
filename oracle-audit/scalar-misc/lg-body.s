v_cmp_lg_f32 vcc_lo, v4, v5
v_cndmask_b32 v10, 0, 1, vcc_lo
v_cmp_lg_f32_e64 s12, v4, v5
v_cndmask_b32_e64 v11, 0, 1, s12
v_cmp_lg_f16 vcc_lo, v6, v7
v_cndmask_b32 v12, 0, 1, vcc_lo
v_cmp_lg_f16_e64 s12, v6, v7
v_cndmask_b32_e64 v13, 0, 1, s12
v_cmp_nlg_f32 vcc_lo, v4, v5
v_cndmask_b32 v14, 0, 1, vcc_lo
v_cmp_neq_f32 vcc_lo, v4, v5
v_cndmask_b32 v15, 0, 1, vcc_lo
