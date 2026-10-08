  v_mov_b32 v12, v0
  s_movk_i32 s21, 0x1234
  s_mov_b32 exec_lo, 1
  v_writelane_b32 v12, s21, 37
  s_mov_b32 exec_lo, -1
  .long 0xbe940380
  .long 0xd7600008
  .long 0x00002904
  .long 0xd7600009
  .long 0x00002905
  .long 0x8096149f
  .long 0xd761000a
  .long 0x00002c08
  .long 0xd761000b
  .long 0x00002809
  .long 0x80148114
  .long 0xbf0aa014
  .long 0xbf85fff4
