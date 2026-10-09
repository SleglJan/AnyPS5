v_lshlrev_b32 v3, 3, v0
ds_write_b64 v3, v[4:5]
s_waitcnt lgkmcnt(0)
v_mov_b32 v8, 0x1000
v_add_nc_u32 v8, v8, v3
ds_add_rtn_u64 v[12:13], v8, v[6:7]
s_waitcnt lgkmcnt(0)
ds_read_b64 v[10:11], v3
v_mov_b32 v8, 0xfff8
ds_add_rtn_u64 v[14:15], v8, v[6:7]
s_waitcnt lgkmcnt(0)
