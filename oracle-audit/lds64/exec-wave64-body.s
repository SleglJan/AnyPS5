v_lshlrev_b32 v3, 3, v0
ds_write_b64 v3, v[4:5]
s_waitcnt lgkmcnt(0)
s_mov_b32 exec_lo, 0x55555555
s_mov_b32 exec_hi, 0x55555555
ds_add_rtn_u64 v[12:13], v3, v[6:7]
s_waitcnt lgkmcnt(0)
s_mov_b64 exec, -1
ds_read_b64 v[10:11], v3
s_waitcnt lgkmcnt(0)
