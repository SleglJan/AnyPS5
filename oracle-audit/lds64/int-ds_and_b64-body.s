v_lshlrev_b32 v3, 3, v0
ds_write_b64 v3, v[4:5]
s_waitcnt lgkmcnt(0)
ds_and_b64 v3, v[6:7]
s_waitcnt lgkmcnt(0)
ds_read_b64 v[10:11], v3
s_waitcnt lgkmcnt(0)
