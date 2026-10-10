v_lshlrev_b32 v3, 3, v0
s_add_u32 s10, s4, 4096
s_addc_u32 s11, s5, 0
global_load_dwordx2 v[8:9], v3, s[10:11]
s_waitcnt vmcnt(0)
ds_write_b32 v3, v4
s_waitcnt lgkmcnt(0)
ds_min_rtn_f32 v11, v3, v6
s_waitcnt lgkmcnt(0)
ds_read_b32 v10, v3
s_waitcnt lgkmcnt(0)
ds_write_b32 v3, v4
s_waitcnt lgkmcnt(0)
ds_max_rtn_f32 v13, v3, v6
s_waitcnt lgkmcnt(0)
ds_read_b32 v12, v3
s_waitcnt lgkmcnt(0)
ds_write_b32 v3, v4
s_waitcnt lgkmcnt(0)
ds_cmpst_rtn_f32 v15, v3, v6, v8
s_waitcnt lgkmcnt(0)
ds_read_b32 v14, v3
s_waitcnt lgkmcnt(0)
ds_write_b32 v3, v4
s_waitcnt lgkmcnt(0)
ds_min_f32 v3, v6
s_waitcnt lgkmcnt(0)
ds_read_b32 v16, v3
s_waitcnt lgkmcnt(0)
ds_write_b32 v3, v4
s_waitcnt lgkmcnt(0)
ds_max_f32 v3, v6
s_waitcnt lgkmcnt(0)
ds_read_b32 v17, v3
s_waitcnt lgkmcnt(0)
ds_write_b32 v3, v4
s_waitcnt lgkmcnt(0)
ds_cmpst_f32 v3, v6, v8
s_waitcnt lgkmcnt(0)
ds_read_b32 v18, v3
s_waitcnt lgkmcnt(0)
