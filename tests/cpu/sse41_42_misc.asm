; coverage: source cpu_aot_sse41_42_misc
; Source-owned differential fixture for 20 SSSE3/SSE4.1/SSE4.2 integer, insert/extract, ptest, crc32, and packed string compare instructions.
.code

pextrb_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    pextrb eax, xmm0, 11
    ret
pextrb_op ENDP

pextrw_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    pextrw eax, xmm0, 5
    ret
pextrw_op ENDP

pextrd_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    pextrd eax, xmm0, 2
    ret
pextrd_op ENDP

pextrq_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    pextrq rax, xmm0, 1
    ret
pextrq_op ENDP

pinsrb_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    mov eax, dword ptr [rdx]
    pinsrb xmm0, eax, 9
    movdqu xmmword ptr [r8], xmm0
    ret
pinsrb_op ENDP

pinsrw_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    mov eax, dword ptr [rdx]
    pinsrw xmm0, eax, 6
    movdqu xmmword ptr [r8], xmm0
    ret
pinsrw_op ENDP

pinsrd_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    mov eax, dword ptr [rdx]
    pinsrd xmm0, eax, 3
    movdqu xmmword ptr [r8], xmm0
    ret
pinsrd_op ENDP

pinsrq_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    mov rax, qword ptr [rdx]
    pinsrq xmm0, rax, 1
    movdqu xmmword ptr [r8], xmm0
    ret
pinsrq_op ENDP

pmaddubsw_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    pmaddubsw xmm0, xmm1
    movdqu xmmword ptr [r8], xmm0
    ret
pmaddubsw_op ENDP

pmulhrsw_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    pmulhrsw xmm0, xmm1
    movdqu xmmword ptr [r8], xmm0
    ret
pmulhrsw_op ENDP

phminposuw_op PROC FRAME
    .endprolog
    movdqu xmm1, xmmword ptr [rcx]
    phminposuw xmm0, xmm1
    movdqu xmmword ptr [r8], xmm0
    ret
phminposuw_op ENDP

ptest_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    xor eax, eax
    xor r9d, r9d
    ptest xmm0, xmm1
    sete al
    jb ptest_cf
    ret
ptest_cf:
    or eax, 2
    ret
ptest_op ENDP

mpsadbw_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    mpsadbw xmm0, xmm1, 05h
    movdqu xmmword ptr [r8], xmm0
    ret
mpsadbw_op ENDP

movntdqa_op PROC FRAME
    .endprolog
    movntdqa xmm0, xmmword ptr [rcx]
    movdqu xmmword ptr [r8], xmm0
    ret
movntdqa_op ENDP

pcmpgtq_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    pcmpgtq xmm0, xmm1
    movdqu xmmword ptr [r8], xmm0
    ret
pcmpgtq_op ENDP

crc32_8_op PROC FRAME
    .endprolog
    mov eax, ecx
    crc32 eax, dl
    ret
crc32_8_op ENDP

crc32_32_op PROC FRAME
    .endprolog
    mov eax, ecx
    crc32 eax, edx
    ret
crc32_32_op ENDP

crc32_64_op PROC FRAME
    .endprolog
    mov rax, rcx
    crc32 rax, rdx
    ret
crc32_64_op ENDP

pcmpestri_ub_eqany_pos_lsb PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    mov r9d, 1
    shl r9, 32
    mov eax, dword ptr [r8]
    or rax, r9
    mov edx, dword ptr [r8+4]
    or rdx, r9
    mov rcx, -1
    pcmpestri xmm0, xmm1, 00h
    mov qword ptr [r8], rcx
    setb al
    movzx eax, al
    sete r9b
    movzx r9d, r9b
    seto r10b
    movzx r10d, r10b
    shl r9d, 1
    shl r10d, 2
    or eax, r9d
    or eax, r10d
    ret
pcmpestri_ub_eqany_pos_lsb ENDP

pcmpestri_sw_ranges_neg_msb PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    mov r9d, 1
    shl r9, 32
    mov eax, dword ptr [r8]
    or rax, r9
    mov edx, dword ptr [r8+4]
    or rdx, r9
    mov rcx, -1
    pcmpestri xmm0, xmm1, 57h
    mov qword ptr [r8], rcx
    setb al
    movzx eax, al
    sete r9b
    movzx r9d, r9b
    seto r10b
    movzx r10d, r10b
    shl r9d, 1
    shl r10d, 2
    or eax, r9d
    or eax, r10d
    ret
pcmpestri_sw_ranges_neg_msb ENDP

pcmpestri_sb_eqeach_mpos_msb PROC FRAME
    .endprolog
    movdqu xmm1, xmmword ptr [rcx]
    movdqu xmm2, xmmword ptr [rdx]
    mov eax, dword ptr [r8]
    mov edx, dword ptr [r8+4]
    mov rcx, -1
    pcmpestri xmm1, xmm2, 6Ah
    mov qword ptr [r8], rcx
    setb al
    movzx eax, al
    sete r9b
    movzx r9d, r9b
    seto r10b
    movzx r10d, r10b
    shl r9d, 1
    shl r10d, 2
    or eax, r9d
    or eax, r10d
    ret
pcmpestri_sb_eqeach_mpos_msb ENDP

pcmpestri_uw_eqord_mneg_lsb PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    mov eax, dword ptr [r8]
    mov edx, dword ptr [r8+4]
    mov rcx, -1
    pcmpestri xmm0, xmm1, 3Dh
    mov qword ptr [r8], rcx
    setb al
    movzx eax, al
    sete r9b
    movzx r9d, r9b
    seto r10b
    movzx r10d, r10b
    shl r9d, 1
    shl r10d, 2
    or eax, r9d
    or eax, r10d
    ret
pcmpestri_uw_eqord_mneg_lsb ENDP

pcmpestrm_ub_eqany_pos_bit PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    mov r9d, 1
    shl r9, 32
    mov eax, dword ptr [r8]
    or rax, r9
    mov edx, dword ptr [r8+4]
    or rdx, r9
    pcmpestrm xmm0, xmm1, 00h
    movdqu xmmword ptr [r8], xmm0
    setb al
    movzx eax, al
    sete r9b
    movzx r9d, r9b
    seto r10b
    movzx r10d, r10b
    shl r9d, 1
    shl r10d, 2
    or eax, r9d
    or eax, r10d
    ret
pcmpestrm_ub_eqany_pos_bit ENDP

pcmpestrm_sb_ranges_neg_exp PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    mov eax, dword ptr [r8]
    mov edx, dword ptr [r8+4]
    pcmpestrm xmm0, xmm1, 56h
    movdqu xmmword ptr [r8], xmm0
    setb al
    movzx eax, al
    sete r9b
    movzx r9d, r9b
    seto r10b
    movzx r10d, r10b
    shl r9d, 1
    shl r10d, 2
    or eax, r9d
    or eax, r10d
    ret
pcmpestrm_sb_ranges_neg_exp ENDP

pcmpestrm_uw_eqeach_mpos_exp PROC FRAME
    .endprolog
    movdqu xmm1, xmmword ptr [rcx]
    movdqu xmm2, xmmword ptr [rdx]
    mov eax, dword ptr [r8]
    mov edx, dword ptr [r8+4]
    pcmpestrm xmm1, xmm2, 69h
    movdqu xmmword ptr [r8], xmm0
    movdqu xmmword ptr [r8+16], xmm1
    setb al
    movzx eax, al
    sete r9b
    movzx r9d, r9b
    seto r10b
    movzx r10d, r10b
    shl r9d, 1
    shl r10d, 2
    or eax, r9d
    or eax, r10d
    ret
pcmpestrm_uw_eqeach_mpos_exp ENDP

pcmpestrm_sw_eqord_mneg_bit PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    mov eax, dword ptr [r8]
    mov edx, dword ptr [r8+4]
    pcmpestrm xmm0, xmm1, 3Fh
    movdqu xmmword ptr [r8], xmm0
    setb al
    movzx eax, al
    sete r9b
    movzx r9d, r9b
    seto r10b
    movzx r10d, r10b
    shl r9d, 1
    shl r10d, 2
    or eax, r9d
    or eax, r10d
    ret
pcmpestrm_sw_eqord_mneg_bit ENDP

pcmpistri_ub_eqord_pos_lsb PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    mov rcx, -1
    pcmpistri xmm0, xmm1, 8Ch
    mov qword ptr [r8], rcx
    setb al
    movzx eax, al
    sete r9b
    movzx r9d, r9b
    seto r10b
    movzx r10d, r10b
    shl r9d, 1
    shl r10d, 2
    or eax, r9d
    or eax, r10d
    ret
pcmpistri_ub_eqord_pos_lsb ENDP

pcmpistri_sb_ranges_mneg_msb PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    mov rcx, -1
    pcmpistri xmm0, xmm1, 76h
    mov qword ptr [r8], rcx
    setb al
    movzx eax, al
    sete r9b
    movzx r9d, r9b
    seto r10b
    movzx r10d, r10b
    shl r9d, 1
    shl r10d, 2
    or eax, r9d
    or eax, r10d
    ret
pcmpistri_sb_ranges_mneg_msb ENDP

pcmpistri_uw_eqany_neg_msb PROC FRAME
    .endprolog
    movdqu xmm1, xmmword ptr [rcx]
    movdqu xmm2, xmmword ptr [rdx]
    mov rcx, -1
    pcmpistri xmm1, xmm2, 51h
    mov qword ptr [r8], rcx
    setb al
    movzx eax, al
    sete r9b
    movzx r9d, r9b
    seto r10b
    movzx r10d, r10b
    shl r9d, 1
    shl r10d, 2
    or eax, r9d
    or eax, r10d
    ret
pcmpistri_uw_eqany_neg_msb ENDP

pcmpistri_sw_eqeach_mpos_lsb PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    mov rcx, -1
    pcmpistri xmm0, xmm1, 2Bh
    mov qword ptr [r8], rcx
    setb al
    movzx eax, al
    sete r9b
    movzx r9d, r9b
    seto r10b
    movzx r10d, r10b
    shl r9d, 1
    shl r10d, 2
    or eax, r9d
    or eax, r10d
    ret
pcmpistri_sw_eqeach_mpos_lsb ENDP

pcmpistrm_ub_eqeach_pos_exp PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    pcmpistrm xmm0, xmm1, 48h
    movdqu xmmword ptr [r8], xmm0
    setb al
    movzx eax, al
    sete r9b
    movzx r9d, r9b
    seto r10b
    movzx r10d, r10b
    shl r9d, 1
    shl r10d, 2
    or eax, r9d
    or eax, r10d
    ret
pcmpistrm_ub_eqeach_pos_exp ENDP

pcmpistrm_sw_eqany_mneg_exp PROC FRAME
    .endprolog
    movdqu xmm1, xmmword ptr [rcx]
    movdqu xmm2, xmmword ptr [rdx]
    pcmpistrm xmm1, xmm2, 73h
    movdqu xmmword ptr [r8], xmm0
    movdqu xmmword ptr [r8+16], xmm1
    setb al
    movzx eax, al
    sete r9b
    movzx r9d, r9b
    seto r10b
    movzx r10d, r10b
    shl r9d, 1
    shl r10d, 2
    or eax, r9d
    or eax, r10d
    ret
pcmpistrm_sw_eqany_mneg_exp ENDP

pcmpistrm_uw_ranges_mpos_bit PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    pcmpistrm xmm0, xmm1, 25h
    movdqu xmmword ptr [r8], xmm0
    setb al
    movzx eax, al
    sete r9b
    movzx r9d, r9b
    seto r10b
    movzx r10d, r10b
    shl r9d, 1
    shl r10d, 2
    or eax, r9d
    or eax, r10d
    ret
pcmpistrm_uw_ranges_mpos_bit ENDP

pcmpistrm_sb_eqord_neg_bit PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    pcmpistrm xmm0, xmm1, 1Eh
    movdqu xmmword ptr [r8], xmm0
    setb al
    movzx eax, al
    sete r9b
    movzx r9d, r9b
    seto r10b
    movzx r10d, r10b
    shl r9d, 1
    shl r10d, 2
    or eax, r9d
    or eax, r10d
    ret
pcmpistrm_sb_eqord_neg_bit ENDP

pcmpestrm_ymm_preserve PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    mov eax, dword ptr [r8]
    mov edx, dword ptr [r8+4]
    pcmpestrm xmm0, xmm1, 40h
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    setb al
    movzx eax, al
    sete r9b
    movzx r9d, r9b
    seto r10b
    movzx r10d, r10b
    shl r9d, 1
    shl r10d, 2
    or eax, r9d
    or eax, r10d
    ret
pcmpestrm_ymm_preserve ENDP

pcmpistrm_ymm_preserve PROC FRAME
    .endprolog
    vmovups ymm0, ymmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    pcmpistrm xmm0, xmm1, 4Ch
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    setb al
    movzx eax, al
    sete r9b
    movzx r9d, r9b
    seto r10b
    movzx r10d, r10b
    shl r9d, 1
    shl r10d, 2
    or eax, r9d
    or eax, r10d
    ret
pcmpistrm_ymm_preserve ENDP

END
