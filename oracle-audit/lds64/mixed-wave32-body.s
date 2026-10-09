v_mov_b32 v8, 0
v_mov_b32 v9, 0
v_mov_b32 v3, 0
ds_write_b64 v3, v[8:9]
ds_write_b64 v3, v[8:9] offset:8
ds_write_b64 v3, v[8:9] offset:16
ds_write_b64 v3, v[8:9] offset:24
s_waitcnt lgkmcnt(0)
s_barrier
v_and_b32 v8, 1, v0
v_cmp_eq_u32 vcc_lo, 0, v8
s_and_saveexec_b32 s12, vcc_lo
ds_add_u64 v3, v[4:5]
s_waitcnt lgkmcnt(0)
s_andn2_b32 exec_lo, s12, vcc_lo
ds_add_u32 v3, v6 offset:4
s_waitcnt lgkmcnt(0)
s_mov_b32 exec_lo, s12
s_waitcnt lgkmcnt(0)
s_barrier
ds_read_b64 v[10:11], v3
s_waitcnt lgkmcnt(0)
