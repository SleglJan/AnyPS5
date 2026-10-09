v_lshlrev_b32_e32 v1, 4, v0
v_lshlrev_b32_e32 v3, 6, v0
buffer_load_dwordx4 v[4:7], v1, s[0:3], 0 offen
s_waitcnt vmcnt(0)
v_cubeid_f32 v10, v4, v5, v6 mul:2
v_cubeid_f32 v11, v4, v5, v6 div:2
v_cubesc_f32 v12, v4, v5, v6 mul:2
v_cubesc_f32 v13, v4, v5, v6 mul:4
v_cubesc_f32 v14, v4, v5, v6 div:2
v_cubetc_f32 v15, v4, v5, v6 mul:2
v_cubetc_f32 v16, v4, v5, v6 mul:4
v_cubetc_f32 v17, v4, v5, v6 div:2
v_cubema_f32 v18, v4, v5, v6 mul:2
v_cubema_f32 v19, v4, v5, v6 mul:4
v_cubema_f32 v20, v4, v5, v6 div:2
v_cubeid_f32 v21, v4, -v5, -v6 div:2
v_cubesc_f32 v22, -v4, |v5|, v6 clamp mul:4
v_cubetc_f32 v23, |v4|, -v5, v6 clamp div:2
v_cubema_f32 v24, v4, v5, -v6 clamp mul:2
v_cubeid_f32 v25, v4, v5, v6 mul:4
buffer_store_dwordx4 v[10:13], v3, s[4:7], 0 offen offset:0
buffer_store_dwordx4 v[14:17], v3, s[4:7], 0 offen offset:16
buffer_store_dwordx4 v[18:21], v3, s[4:7], 0 offen offset:32
buffer_store_dwordx4 v[22:25], v3, s[4:7], 0 offen offset:48
s_endpgm
