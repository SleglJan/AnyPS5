v_lshlrev_b32 v3, 3, v0
ds_write_b64 v3, v[4:5]
s_waitcnt lgkmcnt(0)
ds_min_rtn_f64 v[12:13], v3, v[6:7]
s_waitcnt lgkmcnt(0)
ds_read_b64 v[10:11], v3
s_waitcnt lgkmcnt(0)
ds_write_b64 v3, v[4:5]
s_waitcnt lgkmcnt(0)
ds_max_rtn_f64 v[16:17], v3, v[6:7]
s_waitcnt lgkmcnt(0)
ds_read_b64 v[14:15], v3
s_waitcnt lgkmcnt(0)
ds_write_b64 v3, v[4:5]
s_waitcnt lgkmcnt(0)
ds_min_f64 v3, v[6:7]
s_waitcnt lgkmcnt(0)
ds_read_b64 v[18:19], v3
s_waitcnt lgkmcnt(0)
ds_write_b64 v3, v[4:5]
s_waitcnt lgkmcnt(0)
ds_max_f64 v3, v[6:7]
s_waitcnt lgkmcnt(0)
ds_read_b64 v[20:21], v3
s_waitcnt lgkmcnt(0)
