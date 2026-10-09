v_mov_b32 v8, 0
v_mov_b32 v9, 0
v_mov_b32 v3, 0
ds_write_b64 v3, v[8:9]
ds_write_b64 v3, v[8:9] offset:8
ds_write_b64 v3, v[8:9] offset:16
ds_write_b64 v3, v[8:9] offset:24
s_waitcnt lgkmcnt(0)
s_barrier
ds_add_rtn_u64 v[10:11], v3, v[4:5]
ds_max_rtn_u64 v[12:13], v3, v[6:7] offset:8
ds_inc_rtn_u64 v[14:15], v3, v[6:7] offset:16
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b64 v[16:17], v3
ds_read_b64 v[18:19], v3 offset:8
ds_read_b64 v[20:21], v3 offset:16
s_waitcnt lgkmcnt(0)
