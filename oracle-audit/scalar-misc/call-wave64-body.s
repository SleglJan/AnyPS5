s_mov_b32 s20, 0
s_getpc_b64 s[12:13]
s_call_b64 s[14:15], 3
s_add_u32 s20, s20, 2
s_branch 2f
s_nop 0
s_add_u32 s20, s20, 1
s_mov_b32 s19, 0x2222
s_setpc_b64 s[14:15]
2:
s_sub_u32 s16, s14, s12
s_subb_u32 s17, s15, s13
v_mov_b32 v10, s16
v_mov_b32 v11, s17
v_mov_b32 v12, s19
v_mov_b32 v13, s20
