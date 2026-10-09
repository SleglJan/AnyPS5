v_lshlrev_b32 v3, 4, v0
ds_write_b64 v3, v[4:5] offset:8
s_waitcnt lgkmcnt(0)
ds_add_rtn_u64 v[12:13], v3, v[6:7] offset:8
s_waitcnt lgkmcnt(0)
ds_read_b64 v[10:11], v3 offset:8
ds_read_b64 v[14:15], v3
s_waitcnt lgkmcnt(0)
