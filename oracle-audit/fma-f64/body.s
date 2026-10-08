s_add_u32 s10, s4, 8192
s_addc_u32 s11, s5, 0
v_lshlrev_b32 v3, 3, v0
global_load_dwordx2 v[8:9], v3, s[10:11]
s_waitcnt vmcnt(0)
v_fma_f64 v[10:11], v[4:5], v[6:7], v[8:9]
v_mul_f64 v[12:13], v[4:5], v[6:7]
v_add_f64 v[14:15], v[4:5], v[8:9]
