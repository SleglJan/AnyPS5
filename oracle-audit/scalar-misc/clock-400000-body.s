s_memtime s[12:13]
s_memrealtime s[14:15]
s_waitcnt lgkmcnt(0)
s_getreg_b32 s22, hwreg(HW_REG_SHADER_CYCLES)
s_mov_b32 s20, 400000
1:
v_add_f32 v8, v4, v8
v_add_f32 v9, v5, v9
s_sub_u32 s20, s20, 1
s_cmp_lg_u32 s20, 0
s_cbranch_scc1 1b
s_memtime s[16:17]
s_memrealtime s[18:19]
s_waitcnt lgkmcnt(0)
s_getreg_b32 s23, hwreg(HW_REG_SHADER_CYCLES)
v_mov_b32 v10, s12
v_mov_b32 v11, s13
v_mov_b32 v12, s14
v_mov_b32 v13, s15
v_mov_b32 v14, s16
v_mov_b32 v15, s17
v_mov_b32 v16, s18
v_mov_b32 v17, s19
v_mov_b32 v18, s22
v_mov_b32 v19, s23
v_mov_b32 v20, v8
