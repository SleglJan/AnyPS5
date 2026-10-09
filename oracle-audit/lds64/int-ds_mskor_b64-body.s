v_lshlrev_b32 v3, 3, v0
s_add_u32 s10, s4, 1024
s_addc_u32 s11, s5, 0
global_load_dwordx2 v[8:9], v3, s[10:11]
s_waitcnt vmcnt(0)
ds_write_b64 v3, v[4:5]
s_waitcnt lgkmcnt(0)
ds_mskor_b64 v3, v[6:7], v[8:9]
s_waitcnt lgkmcnt(0)
ds_read_b64 v[10:11], v3
s_waitcnt lgkmcnt(0)
