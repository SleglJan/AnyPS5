s_mov_b32 s14, 0xabcd
s_mov_b32 s15, 0xef01
s_mov_b32 s24, 0
s_getpc_b64 s[12:13]
s_add_u32 s22, s12, 16
s_addc_u32 s23, s13, 0
s_swappc_b64 null, s[22:23]
s_mov_b32 s24, 17
s_mov_b32 s25, 34
s_getpc_b64 s[26:27]
s_add_u32 s28, s26, 16
s_addc_u32 s29, s27, 0
s_swappc_b64 s[30:31], s[28:29]
s_mov_b32 s25, 51
s_mov_b32 s34, 60
s_sub_u32 s32, s30, s26
s_subb_u32 s33, s31, s27
v_mov_b32 v10, s14
v_mov_b32 v11, s15
v_mov_b32 v12, s24
v_mov_b32 v13, s25
v_mov_b32 v14, s32
v_mov_b32 v15, s33
v_mov_b32 v16, s34
