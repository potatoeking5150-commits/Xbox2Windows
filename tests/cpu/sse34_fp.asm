; coverage: source cpu_aot_sse34_fp
; Source-owned differential fixture for 22 baseline 128-bit SSE3 and SSE4.1 floating-point instructions.
.code

addsubps_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    addsubps xmm0, xmm1
    movdqu xmmword ptr [r8], xmm0
    ret
addsubps_op ENDP

addsubpd_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    addsubpd xmm0, xmm1
    movdqu xmmword ptr [r8], xmm0
    ret
addsubpd_op ENDP

haddps_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    haddps xmm0, xmm1
    movdqu xmmword ptr [r8], xmm0
    ret
haddps_op ENDP

haddpd_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    haddpd xmm0, xmm1
    movdqu xmmword ptr [r8], xmm0
    ret
haddpd_op ENDP

hsubps_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    hsubps xmm0, xmm1
    movdqu xmmword ptr [r8], xmm0
    ret
hsubps_op ENDP

hsubpd_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    hsubpd xmm0, xmm1
    movdqu xmmword ptr [r8], xmm0
    ret
hsubpd_op ENDP

addsubps_mem_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    addsubps xmm0, xmmword ptr [rdx]
    movdqu xmmword ptr [r8], xmm0
    ret
addsubps_mem_op ENDP

addsubpd_mem_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    addsubpd xmm0, xmmword ptr [rdx]
    movdqu xmmword ptr [r8], xmm0
    ret
addsubpd_mem_op ENDP

movddup_op PROC FRAME
    .endprolog
    movdqu xmm1, xmmword ptr [rcx]
    movddup xmm0, xmm1
    movdqu xmmword ptr [r8], xmm0
    ret
movddup_op ENDP

movddup_mem_op PROC FRAME
    .endprolog
    movddup xmm0, qword ptr [rcx]
    movdqu xmmword ptr [r8], xmm0
    ret
movddup_mem_op ENDP

movsldup_op PROC FRAME
    .endprolog
    movdqu xmm1, xmmword ptr [rcx]
    movsldup xmm0, xmm1
    movdqu xmmword ptr [r8], xmm0
    ret
movsldup_op ENDP

movsldup_mem_op PROC FRAME
    .endprolog
    movsldup xmm0, xmmword ptr [rcx]
    movdqu xmmword ptr [r8], xmm0
    ret
movsldup_mem_op ENDP

movshdup_op PROC FRAME
    .endprolog
    movdqu xmm1, xmmword ptr [rcx]
    movshdup xmm0, xmm1
    movdqu xmmword ptr [r8], xmm0
    ret
movshdup_op ENDP

movshdup_mem_op PROC FRAME
    .endprolog
    movshdup xmm0, xmmword ptr [rcx]
    movdqu xmmword ptr [r8], xmm0
    ret
movshdup_mem_op ENDP

lddqu_op PROC FRAME
    .endprolog
    lddqu xmm0, xmmword ptr [rcx]
    movdqu xmmword ptr [r8], xmm0
    ret
lddqu_op ENDP

roundps_op PROC FRAME
    .endprolog
    movdqu xmm1, xmmword ptr [rcx]
    cmp edx, 0
    jne roundps_n1
    roundps xmm0, xmm1, 0
    jmp roundps_done
roundps_n1:
    cmp edx, 1
    jne roundps_n2
    roundps xmm0, xmm1, 1
    jmp roundps_done
roundps_n2:
    cmp edx, 2
    jne roundps_n3
    roundps xmm0, xmm1, 2
    jmp roundps_done
roundps_n3:
    cmp edx, 3
    jne roundps_n4
    roundps xmm0, xmm1, 3
    jmp roundps_done
roundps_n4:
    cmp edx, 4
    jne roundps_n5
    roundps xmm0, xmm1, 4
    jmp roundps_done
roundps_n5:
    cmp edx, 8
    jne roundps_n6
    roundps xmm0, xmm1, 8
    jmp roundps_done
roundps_n6:
    cmp edx, 9
    jne roundps_n7
    roundps xmm0, xmm1, 9
    jmp roundps_done
roundps_n7:
    cmp edx, 10
    jne roundps_n8
    roundps xmm0, xmm1, 10
    jmp roundps_done
roundps_n8:
    cmp edx, 11
    jne roundps_n9
    roundps xmm0, xmm1, 11
    jmp roundps_done
roundps_n9:
    cmp edx, 12
    jne roundps_done
    roundps xmm0, xmm1, 12
    jmp roundps_done
roundps_done:
    movdqu xmmword ptr [r8], xmm0
    ret
roundps_op ENDP

roundpd_op PROC FRAME
    .endprolog
    movdqu xmm1, xmmword ptr [rcx]
    cmp edx, 0
    jne roundpd_n1
    roundpd xmm0, xmm1, 0
    jmp roundpd_done
roundpd_n1:
    cmp edx, 1
    jne roundpd_n2
    roundpd xmm0, xmm1, 1
    jmp roundpd_done
roundpd_n2:
    cmp edx, 2
    jne roundpd_n3
    roundpd xmm0, xmm1, 2
    jmp roundpd_done
roundpd_n3:
    cmp edx, 3
    jne roundpd_n4
    roundpd xmm0, xmm1, 3
    jmp roundpd_done
roundpd_n4:
    cmp edx, 4
    jne roundpd_n5
    roundpd xmm0, xmm1, 4
    jmp roundpd_done
roundpd_n5:
    cmp edx, 8
    jne roundpd_n6
    roundpd xmm0, xmm1, 8
    jmp roundpd_done
roundpd_n6:
    cmp edx, 9
    jne roundpd_n7
    roundpd xmm0, xmm1, 9
    jmp roundpd_done
roundpd_n7:
    cmp edx, 10
    jne roundpd_n8
    roundpd xmm0, xmm1, 10
    jmp roundpd_done
roundpd_n8:
    cmp edx, 11
    jne roundpd_n9
    roundpd xmm0, xmm1, 11
    jmp roundpd_done
roundpd_n9:
    cmp edx, 12
    jne roundpd_done
    roundpd xmm0, xmm1, 12
    jmp roundpd_done
roundpd_done:
    movdqu xmmword ptr [r8], xmm0
    ret
roundpd_op ENDP

roundss_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    cmp r9d, 0
    jne roundss_n1
    roundss xmm0, xmm1, 0
    jmp roundss_done
roundss_n1:
    cmp r9d, 1
    jne roundss_n2
    roundss xmm0, xmm1, 1
    jmp roundss_done
roundss_n2:
    cmp r9d, 2
    jne roundss_n3
    roundss xmm0, xmm1, 2
    jmp roundss_done
roundss_n3:
    cmp r9d, 3
    jne roundss_n4
    roundss xmm0, xmm1, 3
    jmp roundss_done
roundss_n4:
    cmp r9d, 4
    jne roundss_n5
    roundss xmm0, xmm1, 4
    jmp roundss_done
roundss_n5:
    cmp r9d, 8
    jne roundss_n6
    roundss xmm0, xmm1, 8
    jmp roundss_done
roundss_n6:
    cmp r9d, 9
    jne roundss_n7
    roundss xmm0, xmm1, 9
    jmp roundss_done
roundss_n7:
    cmp r9d, 10
    jne roundss_n8
    roundss xmm0, xmm1, 10
    jmp roundss_done
roundss_n8:
    cmp r9d, 11
    jne roundss_n9
    roundss xmm0, xmm1, 11
    jmp roundss_done
roundss_n9:
    cmp r9d, 12
    jne roundss_done
    roundss xmm0, xmm1, 12
    jmp roundss_done
roundss_done:
    movdqu xmmword ptr [r8], xmm0
    ret
roundss_op ENDP

roundss_mem_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    cmp r9d, 0
    jne roundss_m_n1
    roundss xmm0, dword ptr [rdx], 0
    jmp roundss_m_done
roundss_m_n1:
    cmp r9d, 1
    jne roundss_m_n2
    roundss xmm0, dword ptr [rdx], 1
    jmp roundss_m_done
roundss_m_n2:
    cmp r9d, 2
    jne roundss_m_n3
    roundss xmm0, dword ptr [rdx], 2
    jmp roundss_m_done
roundss_m_n3:
    cmp r9d, 3
    jne roundss_m_done
    roundss xmm0, dword ptr [rdx], 3
    jmp roundss_m_done
roundss_m_done:
    movdqu xmmword ptr [r8], xmm0
    ret
roundss_mem_op ENDP

roundsd_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    cmp r9d, 0
    jne roundsd_n1
    roundsd xmm0, xmm1, 0
    jmp roundsd_done
roundsd_n1:
    cmp r9d, 1
    jne roundsd_n2
    roundsd xmm0, xmm1, 1
    jmp roundsd_done
roundsd_n2:
    cmp r9d, 2
    jne roundsd_n3
    roundsd xmm0, xmm1, 2
    jmp roundsd_done
roundsd_n3:
    cmp r9d, 3
    jne roundsd_n4
    roundsd xmm0, xmm1, 3
    jmp roundsd_done
roundsd_n4:
    cmp r9d, 4
    jne roundsd_n5
    roundsd xmm0, xmm1, 4
    jmp roundsd_done
roundsd_n5:
    cmp r9d, 8
    jne roundsd_n6
    roundsd xmm0, xmm1, 8
    jmp roundsd_done
roundsd_n6:
    cmp r9d, 9
    jne roundsd_n7
    roundsd xmm0, xmm1, 9
    jmp roundsd_done
roundsd_n7:
    cmp r9d, 10
    jne roundsd_n8
    roundsd xmm0, xmm1, 10
    jmp roundsd_done
roundsd_n8:
    cmp r9d, 11
    jne roundsd_n9
    roundsd xmm0, xmm1, 11
    jmp roundsd_done
roundsd_n9:
    cmp r9d, 12
    jne roundsd_done
    roundsd xmm0, xmm1, 12
    jmp roundsd_done
roundsd_done:
    movdqu xmmword ptr [r8], xmm0
    ret
roundsd_op ENDP

roundsd_mem_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    cmp r9d, 0
    jne roundsd_m_n1
    roundsd xmm0, qword ptr [rdx], 0
    jmp roundsd_m_done
roundsd_m_n1:
    cmp r9d, 1
    jne roundsd_m_n2
    roundsd xmm0, qword ptr [rdx], 1
    jmp roundsd_m_done
roundsd_m_n2:
    cmp r9d, 2
    jne roundsd_m_n3
    roundsd xmm0, qword ptr [rdx], 2
    jmp roundsd_m_done
roundsd_m_n3:
    cmp r9d, 3
    jne roundsd_m_done
    roundsd xmm0, qword ptr [rdx], 3
    jmp roundsd_m_done
roundsd_m_done:
    movdqu xmmword ptr [r8], xmm0
    ret
roundsd_mem_op ENDP

dpps_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    cmp r9d, 0
    jne dpps_n1
    dpps xmm0, xmm1, 0
    jmp dpps_done
dpps_n1:
    cmp r9d, 15
    jne dpps_n2
    dpps xmm0, xmm1, 15
    jmp dpps_done
dpps_n2:
    cmp r9d, 240
    jne dpps_n3
    dpps xmm0, xmm1, 240
    jmp dpps_done
dpps_n3:
    cmp r9d, 241
    jne dpps_n4
    dpps xmm0, xmm1, 241
    jmp dpps_done
dpps_n4:
    cmp r9d, 242
    jne dpps_n5
    dpps xmm0, xmm1, 242
    jmp dpps_done
dpps_n5:
    cmp r9d, 244
    jne dpps_n6
    dpps xmm0, xmm1, 244
    jmp dpps_done
dpps_n6:
    cmp r9d, 248
    jne dpps_n7
    dpps xmm0, xmm1, 248
    jmp dpps_done
dpps_n7:
    cmp r9d, 255
    jne dpps_n8
    dpps xmm0, xmm1, 255
    jmp dpps_done
dpps_n8:
    cmp r9d, 17
    jne dpps_n9
    dpps xmm0, xmm1, 17
    jmp dpps_done
dpps_n9:
    cmp r9d, 34
    jne dpps_n10
    dpps xmm0, xmm1, 34
    jmp dpps_done
dpps_n10:
    cmp r9d, 68
    jne dpps_n11
    dpps xmm0, xmm1, 68
    jmp dpps_done
dpps_n11:
    cmp r9d, 136
    jne dpps_n12
    dpps xmm0, xmm1, 136
    jmp dpps_done
dpps_n12:
    cmp r9d, 85
    jne dpps_n13
    dpps xmm0, xmm1, 85
    jmp dpps_done
dpps_n13:
    cmp r9d, 170
    jne dpps_n14
    dpps xmm0, xmm1, 170
    jmp dpps_done
dpps_n14:
    cmp r9d, 127
    jne dpps_n15
    dpps xmm0, xmm1, 127
    jmp dpps_done
dpps_n15:
    cmp r9d, 60
    jne dpps_n16
    dpps xmm0, xmm1, 60
    jmp dpps_done
dpps_n16:
    cmp r9d, 195
    jne dpps_n17
    dpps xmm0, xmm1, 195
    jmp dpps_done
dpps_n17:
    cmp r9d, 150
    jne dpps_done
    dpps xmm0, xmm1, 150
    jmp dpps_done
dpps_done:
    movdqu xmmword ptr [r8], xmm0
    ret
dpps_op ENDP

dppd_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    cmp r9d, 0
    jne dppd_n1
    dppd xmm0, xmm1, 0
    jmp dppd_done
dppd_n1:
    cmp r9d, 1
    jne dppd_n2
    dppd xmm0, xmm1, 1
    jmp dppd_done
dppd_n2:
    cmp r9d, 2
    jne dppd_n3
    dppd xmm0, xmm1, 2
    jmp dppd_done
dppd_n3:
    cmp r9d, 3
    jne dppd_n4
    dppd xmm0, xmm1, 3
    jmp dppd_done
dppd_n4:
    cmp r9d, 16
    jne dppd_n5
    dppd xmm0, xmm1, 16
    jmp dppd_done
dppd_n5:
    cmp r9d, 17
    jne dppd_n6
    dppd xmm0, xmm1, 17
    jmp dppd_done
dppd_n6:
    cmp r9d, 18
    jne dppd_n7
    dppd xmm0, xmm1, 18
    jmp dppd_done
dppd_n7:
    cmp r9d, 19
    jne dppd_n8
    dppd xmm0, xmm1, 19
    jmp dppd_done
dppd_n8:
    cmp r9d, 32
    jne dppd_n9
    dppd xmm0, xmm1, 32
    jmp dppd_done
dppd_n9:
    cmp r9d, 33
    jne dppd_n10
    dppd xmm0, xmm1, 33
    jmp dppd_done
dppd_n10:
    cmp r9d, 34
    jne dppd_n11
    dppd xmm0, xmm1, 34
    jmp dppd_done
dppd_n11:
    cmp r9d, 35
    jne dppd_n12
    dppd xmm0, xmm1, 35
    jmp dppd_done
dppd_n12:
    cmp r9d, 48
    jne dppd_n13
    dppd xmm0, xmm1, 48
    jmp dppd_done
dppd_n13:
    cmp r9d, 49
    jne dppd_n14
    dppd xmm0, xmm1, 49
    jmp dppd_done
dppd_n14:
    cmp r9d, 50
    jne dppd_n15
    dppd xmm0, xmm1, 50
    jmp dppd_done
dppd_n15:
    cmp r9d, 51
    jne dppd_n16
    dppd xmm0, xmm1, 51
    jmp dppd_done
dppd_n16:
    cmp r9d, 255
    jne dppd_done
    dppd xmm0, xmm1, 255
    jmp dppd_done
dppd_done:
    movdqu xmmword ptr [r8], xmm0
    ret
dppd_op ENDP

blendps_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    cmp r9d, 0
    jne blendps_n1
    blendps xmm0, xmm1, 0
    jmp blendps_done
blendps_n1:
    cmp r9d, 1
    jne blendps_n2
    blendps xmm0, xmm1, 1
    jmp blendps_done
blendps_n2:
    cmp r9d, 2
    jne blendps_n3
    blendps xmm0, xmm1, 2
    jmp blendps_done
blendps_n3:
    cmp r9d, 3
    jne blendps_n4
    blendps xmm0, xmm1, 3
    jmp blendps_done
blendps_n4:
    cmp r9d, 4
    jne blendps_n5
    blendps xmm0, xmm1, 4
    jmp blendps_done
blendps_n5:
    cmp r9d, 5
    jne blendps_n6
    blendps xmm0, xmm1, 5
    jmp blendps_done
blendps_n6:
    cmp r9d, 6
    jne blendps_n7
    blendps xmm0, xmm1, 6
    jmp blendps_done
blendps_n7:
    cmp r9d, 7
    jne blendps_n8
    blendps xmm0, xmm1, 7
    jmp blendps_done
blendps_n8:
    cmp r9d, 8
    jne blendps_n9
    blendps xmm0, xmm1, 8
    jmp blendps_done
blendps_n9:
    cmp r9d, 9
    jne blendps_n10
    blendps xmm0, xmm1, 9
    jmp blendps_done
blendps_n10:
    cmp r9d, 10
    jne blendps_n11
    blendps xmm0, xmm1, 10
    jmp blendps_done
blendps_n11:
    cmp r9d, 11
    jne blendps_n12
    blendps xmm0, xmm1, 11
    jmp blendps_done
blendps_n12:
    cmp r9d, 12
    jne blendps_n13
    blendps xmm0, xmm1, 12
    jmp blendps_done
blendps_n13:
    cmp r9d, 13
    jne blendps_n14
    blendps xmm0, xmm1, 13
    jmp blendps_done
blendps_n14:
    cmp r9d, 14
    jne blendps_n15
    blendps xmm0, xmm1, 14
    jmp blendps_done
blendps_n15:
    cmp r9d, 15
    jne blendps_done
    blendps xmm0, xmm1, 15
    jmp blendps_done
blendps_done:
    movdqu xmmword ptr [r8], xmm0
    ret
blendps_op ENDP

blendpd_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    cmp r9d, 0
    jne blendpd_n1
    blendpd xmm0, xmm1, 0
    jmp blendpd_done
blendpd_n1:
    cmp r9d, 1
    jne blendpd_n2
    blendpd xmm0, xmm1, 1
    jmp blendpd_done
blendpd_n2:
    cmp r9d, 2
    jne blendpd_n3
    blendpd xmm0, xmm1, 2
    jmp blendpd_done
blendpd_n3:
    cmp r9d, 3
    jne blendpd_done
    blendpd xmm0, xmm1, 3
    jmp blendpd_done
blendpd_done:
    movdqu xmmword ptr [r8], xmm0
    ret
blendpd_op ENDP

blendvps_op PROC FRAME
    .endprolog
    movdqu xmm1, xmmword ptr [rcx]
    movdqu xmm2, xmmword ptr [rdx]
    movdqu xmm0, xmmword ptr [r8]
    blendvps xmm1, xmm2, xmm0
    movdqu xmmword ptr [r9], xmm1
    ret
blendvps_op ENDP

blendvps_self_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    blendvps xmm0, xmm1, xmm0
    movdqu xmmword ptr [r8], xmm0
    ret
blendvps_self_op ENDP

blendvpd_op PROC FRAME
    .endprolog
    movdqu xmm1, xmmword ptr [rcx]
    movdqu xmm2, xmmword ptr [rdx]
    movdqu xmm0, xmmword ptr [r8]
    blendvpd xmm1, xmm2, xmm0
    movdqu xmmword ptr [r9], xmm1
    ret
blendvpd_op ENDP

blendvpd_self_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    blendvpd xmm0, xmm1, xmm0
    movdqu xmmword ptr [r8], xmm0
    ret
blendvpd_self_op ENDP

insertps_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    movdqu xmm1, xmmword ptr [rdx]
    cmp r9d, 0
    jne ins_n1
    insertps xmm0, xmm1, 0
    jmp ins_done
ins_n1:
    cmp r9d, 1
    jne ins_n2
    insertps xmm0, xmm1, 1
    jmp ins_done
ins_n2:
    cmp r9d, 2
    jne ins_n3
    insertps xmm0, xmm1, 2
    jmp ins_done
ins_n3:
    cmp r9d, 3
    jne ins_n4
    insertps xmm0, xmm1, 3
    jmp ins_done
ins_n4:
    cmp r9d, 4
    jne ins_n5
    insertps xmm0, xmm1, 4
    jmp ins_done
ins_n5:
    cmp r9d, 5
    jne ins_n6
    insertps xmm0, xmm1, 5
    jmp ins_done
ins_n6:
    cmp r9d, 6
    jne ins_n7
    insertps xmm0, xmm1, 6
    jmp ins_done
ins_n7:
    cmp r9d, 7
    jne ins_n8
    insertps xmm0, xmm1, 7
    jmp ins_done
ins_n8:
    cmp r9d, 8
    jne ins_n9
    insertps xmm0, xmm1, 8
    jmp ins_done
ins_n9:
    cmp r9d, 9
    jne ins_n10
    insertps xmm0, xmm1, 9
    jmp ins_done
ins_n10:
    cmp r9d, 10
    jne ins_n11
    insertps xmm0, xmm1, 10
    jmp ins_done
ins_n11:
    cmp r9d, 11
    jne ins_n12
    insertps xmm0, xmm1, 11
    jmp ins_done
ins_n12:
    cmp r9d, 12
    jne ins_n13
    insertps xmm0, xmm1, 12
    jmp ins_done
ins_n13:
    cmp r9d, 13
    jne ins_n14
    insertps xmm0, xmm1, 13
    jmp ins_done
ins_n14:
    cmp r9d, 14
    jne ins_n15
    insertps xmm0, xmm1, 14
    jmp ins_done
ins_n15:
    cmp r9d, 15
    jne ins_n16
    insertps xmm0, xmm1, 15
    jmp ins_done
ins_n16:
    cmp r9d, 16
    jne ins_n17
    insertps xmm0, xmm1, 16
    jmp ins_done
ins_n17:
    cmp r9d, 17
    jne ins_n18
    insertps xmm0, xmm1, 17
    jmp ins_done
ins_n18:
    cmp r9d, 18
    jne ins_n19
    insertps xmm0, xmm1, 18
    jmp ins_done
ins_n19:
    cmp r9d, 19
    jne ins_n20
    insertps xmm0, xmm1, 19
    jmp ins_done
ins_n20:
    cmp r9d, 20
    jne ins_n21
    insertps xmm0, xmm1, 20
    jmp ins_done
ins_n21:
    cmp r9d, 21
    jne ins_n22
    insertps xmm0, xmm1, 21
    jmp ins_done
ins_n22:
    cmp r9d, 22
    jne ins_n23
    insertps xmm0, xmm1, 22
    jmp ins_done
ins_n23:
    cmp r9d, 23
    jne ins_n24
    insertps xmm0, xmm1, 23
    jmp ins_done
ins_n24:
    cmp r9d, 24
    jne ins_n25
    insertps xmm0, xmm1, 24
    jmp ins_done
ins_n25:
    cmp r9d, 25
    jne ins_n26
    insertps xmm0, xmm1, 25
    jmp ins_done
ins_n26:
    cmp r9d, 26
    jne ins_n27
    insertps xmm0, xmm1, 26
    jmp ins_done
ins_n27:
    cmp r9d, 27
    jne ins_n28
    insertps xmm0, xmm1, 27
    jmp ins_done
ins_n28:
    cmp r9d, 28
    jne ins_n29
    insertps xmm0, xmm1, 28
    jmp ins_done
ins_n29:
    cmp r9d, 29
    jne ins_n30
    insertps xmm0, xmm1, 29
    jmp ins_done
ins_n30:
    cmp r9d, 30
    jne ins_n31
    insertps xmm0, xmm1, 30
    jmp ins_done
ins_n31:
    cmp r9d, 31
    jne ins_n32
    insertps xmm0, xmm1, 31
    jmp ins_done
ins_n32:
    cmp r9d, 32
    jne ins_n33
    insertps xmm0, xmm1, 32
    jmp ins_done
ins_n33:
    cmp r9d, 33
    jne ins_n34
    insertps xmm0, xmm1, 33
    jmp ins_done
ins_n34:
    cmp r9d, 34
    jne ins_n35
    insertps xmm0, xmm1, 34
    jmp ins_done
ins_n35:
    cmp r9d, 35
    jne ins_n36
    insertps xmm0, xmm1, 35
    jmp ins_done
ins_n36:
    cmp r9d, 36
    jne ins_n37
    insertps xmm0, xmm1, 36
    jmp ins_done
ins_n37:
    cmp r9d, 37
    jne ins_n38
    insertps xmm0, xmm1, 37
    jmp ins_done
ins_n38:
    cmp r9d, 38
    jne ins_n39
    insertps xmm0, xmm1, 38
    jmp ins_done
ins_n39:
    cmp r9d, 39
    jne ins_n40
    insertps xmm0, xmm1, 39
    jmp ins_done
ins_n40:
    cmp r9d, 40
    jne ins_n41
    insertps xmm0, xmm1, 40
    jmp ins_done
ins_n41:
    cmp r9d, 41
    jne ins_n42
    insertps xmm0, xmm1, 41
    jmp ins_done
ins_n42:
    cmp r9d, 42
    jne ins_n43
    insertps xmm0, xmm1, 42
    jmp ins_done
ins_n43:
    cmp r9d, 43
    jne ins_n44
    insertps xmm0, xmm1, 43
    jmp ins_done
ins_n44:
    cmp r9d, 44
    jne ins_n45
    insertps xmm0, xmm1, 44
    jmp ins_done
ins_n45:
    cmp r9d, 45
    jne ins_n46
    insertps xmm0, xmm1, 45
    jmp ins_done
ins_n46:
    cmp r9d, 46
    jne ins_n47
    insertps xmm0, xmm1, 46
    jmp ins_done
ins_n47:
    cmp r9d, 47
    jne ins_n48
    insertps xmm0, xmm1, 47
    jmp ins_done
ins_n48:
    cmp r9d, 48
    jne ins_n49
    insertps xmm0, xmm1, 48
    jmp ins_done
ins_n49:
    cmp r9d, 49
    jne ins_n50
    insertps xmm0, xmm1, 49
    jmp ins_done
ins_n50:
    cmp r9d, 50
    jne ins_n51
    insertps xmm0, xmm1, 50
    jmp ins_done
ins_n51:
    cmp r9d, 51
    jne ins_n52
    insertps xmm0, xmm1, 51
    jmp ins_done
ins_n52:
    cmp r9d, 52
    jne ins_n53
    insertps xmm0, xmm1, 52
    jmp ins_done
ins_n53:
    cmp r9d, 53
    jne ins_n54
    insertps xmm0, xmm1, 53
    jmp ins_done
ins_n54:
    cmp r9d, 54
    jne ins_n55
    insertps xmm0, xmm1, 54
    jmp ins_done
ins_n55:
    cmp r9d, 55
    jne ins_n56
    insertps xmm0, xmm1, 55
    jmp ins_done
ins_n56:
    cmp r9d, 56
    jne ins_n57
    insertps xmm0, xmm1, 56
    jmp ins_done
ins_n57:
    cmp r9d, 57
    jne ins_n58
    insertps xmm0, xmm1, 57
    jmp ins_done
ins_n58:
    cmp r9d, 58
    jne ins_n59
    insertps xmm0, xmm1, 58
    jmp ins_done
ins_n59:
    cmp r9d, 59
    jne ins_n60
    insertps xmm0, xmm1, 59
    jmp ins_done
ins_n60:
    cmp r9d, 60
    jne ins_n61
    insertps xmm0, xmm1, 60
    jmp ins_done
ins_n61:
    cmp r9d, 61
    jne ins_n62
    insertps xmm0, xmm1, 61
    jmp ins_done
ins_n62:
    cmp r9d, 62
    jne ins_n63
    insertps xmm0, xmm1, 62
    jmp ins_done
ins_n63:
    cmp r9d, 63
    jne ins_n64
    insertps xmm0, xmm1, 63
    jmp ins_done
ins_n64:
    cmp r9d, 64
    jne ins_n65
    insertps xmm0, xmm1, 64
    jmp ins_done
ins_n65:
    cmp r9d, 65
    jne ins_n66
    insertps xmm0, xmm1, 65
    jmp ins_done
ins_n66:
    cmp r9d, 66
    jne ins_n67
    insertps xmm0, xmm1, 66
    jmp ins_done
ins_n67:
    cmp r9d, 67
    jne ins_n68
    insertps xmm0, xmm1, 67
    jmp ins_done
ins_n68:
    cmp r9d, 68
    jne ins_n69
    insertps xmm0, xmm1, 68
    jmp ins_done
ins_n69:
    cmp r9d, 69
    jne ins_n70
    insertps xmm0, xmm1, 69
    jmp ins_done
ins_n70:
    cmp r9d, 70
    jne ins_n71
    insertps xmm0, xmm1, 70
    jmp ins_done
ins_n71:
    cmp r9d, 71
    jne ins_n72
    insertps xmm0, xmm1, 71
    jmp ins_done
ins_n72:
    cmp r9d, 72
    jne ins_n73
    insertps xmm0, xmm1, 72
    jmp ins_done
ins_n73:
    cmp r9d, 73
    jne ins_n74
    insertps xmm0, xmm1, 73
    jmp ins_done
ins_n74:
    cmp r9d, 74
    jne ins_n75
    insertps xmm0, xmm1, 74
    jmp ins_done
ins_n75:
    cmp r9d, 75
    jne ins_n76
    insertps xmm0, xmm1, 75
    jmp ins_done
ins_n76:
    cmp r9d, 76
    jne ins_n77
    insertps xmm0, xmm1, 76
    jmp ins_done
ins_n77:
    cmp r9d, 77
    jne ins_n78
    insertps xmm0, xmm1, 77
    jmp ins_done
ins_n78:
    cmp r9d, 78
    jne ins_n79
    insertps xmm0, xmm1, 78
    jmp ins_done
ins_n79:
    cmp r9d, 79
    jne ins_n80
    insertps xmm0, xmm1, 79
    jmp ins_done
ins_n80:
    cmp r9d, 80
    jne ins_n81
    insertps xmm0, xmm1, 80
    jmp ins_done
ins_n81:
    cmp r9d, 81
    jne ins_n82
    insertps xmm0, xmm1, 81
    jmp ins_done
ins_n82:
    cmp r9d, 82
    jne ins_n83
    insertps xmm0, xmm1, 82
    jmp ins_done
ins_n83:
    cmp r9d, 83
    jne ins_n84
    insertps xmm0, xmm1, 83
    jmp ins_done
ins_n84:
    cmp r9d, 84
    jne ins_n85
    insertps xmm0, xmm1, 84
    jmp ins_done
ins_n85:
    cmp r9d, 85
    jne ins_n86
    insertps xmm0, xmm1, 85
    jmp ins_done
ins_n86:
    cmp r9d, 86
    jne ins_n87
    insertps xmm0, xmm1, 86
    jmp ins_done
ins_n87:
    cmp r9d, 87
    jne ins_n88
    insertps xmm0, xmm1, 87
    jmp ins_done
ins_n88:
    cmp r9d, 88
    jne ins_n89
    insertps xmm0, xmm1, 88
    jmp ins_done
ins_n89:
    cmp r9d, 89
    jne ins_n90
    insertps xmm0, xmm1, 89
    jmp ins_done
ins_n90:
    cmp r9d, 90
    jne ins_n91
    insertps xmm0, xmm1, 90
    jmp ins_done
ins_n91:
    cmp r9d, 91
    jne ins_n92
    insertps xmm0, xmm1, 91
    jmp ins_done
ins_n92:
    cmp r9d, 92
    jne ins_n93
    insertps xmm0, xmm1, 92
    jmp ins_done
ins_n93:
    cmp r9d, 93
    jne ins_n94
    insertps xmm0, xmm1, 93
    jmp ins_done
ins_n94:
    cmp r9d, 94
    jne ins_n95
    insertps xmm0, xmm1, 94
    jmp ins_done
ins_n95:
    cmp r9d, 95
    jne ins_n96
    insertps xmm0, xmm1, 95
    jmp ins_done
ins_n96:
    cmp r9d, 96
    jne ins_n97
    insertps xmm0, xmm1, 96
    jmp ins_done
ins_n97:
    cmp r9d, 97
    jne ins_n98
    insertps xmm0, xmm1, 97
    jmp ins_done
ins_n98:
    cmp r9d, 98
    jne ins_n99
    insertps xmm0, xmm1, 98
    jmp ins_done
ins_n99:
    cmp r9d, 99
    jne ins_n100
    insertps xmm0, xmm1, 99
    jmp ins_done
ins_n100:
    cmp r9d, 100
    jne ins_n101
    insertps xmm0, xmm1, 100
    jmp ins_done
ins_n101:
    cmp r9d, 101
    jne ins_n102
    insertps xmm0, xmm1, 101
    jmp ins_done
ins_n102:
    cmp r9d, 102
    jne ins_n103
    insertps xmm0, xmm1, 102
    jmp ins_done
ins_n103:
    cmp r9d, 103
    jne ins_n104
    insertps xmm0, xmm1, 103
    jmp ins_done
ins_n104:
    cmp r9d, 104
    jne ins_n105
    insertps xmm0, xmm1, 104
    jmp ins_done
ins_n105:
    cmp r9d, 105
    jne ins_n106
    insertps xmm0, xmm1, 105
    jmp ins_done
ins_n106:
    cmp r9d, 106
    jne ins_n107
    insertps xmm0, xmm1, 106
    jmp ins_done
ins_n107:
    cmp r9d, 107
    jne ins_n108
    insertps xmm0, xmm1, 107
    jmp ins_done
ins_n108:
    cmp r9d, 108
    jne ins_n109
    insertps xmm0, xmm1, 108
    jmp ins_done
ins_n109:
    cmp r9d, 109
    jne ins_n110
    insertps xmm0, xmm1, 109
    jmp ins_done
ins_n110:
    cmp r9d, 110
    jne ins_n111
    insertps xmm0, xmm1, 110
    jmp ins_done
ins_n111:
    cmp r9d, 111
    jne ins_n112
    insertps xmm0, xmm1, 111
    jmp ins_done
ins_n112:
    cmp r9d, 112
    jne ins_n113
    insertps xmm0, xmm1, 112
    jmp ins_done
ins_n113:
    cmp r9d, 113
    jne ins_n114
    insertps xmm0, xmm1, 113
    jmp ins_done
ins_n114:
    cmp r9d, 114
    jne ins_n115
    insertps xmm0, xmm1, 114
    jmp ins_done
ins_n115:
    cmp r9d, 115
    jne ins_n116
    insertps xmm0, xmm1, 115
    jmp ins_done
ins_n116:
    cmp r9d, 116
    jne ins_n117
    insertps xmm0, xmm1, 116
    jmp ins_done
ins_n117:
    cmp r9d, 117
    jne ins_n118
    insertps xmm0, xmm1, 117
    jmp ins_done
ins_n118:
    cmp r9d, 118
    jne ins_n119
    insertps xmm0, xmm1, 118
    jmp ins_done
ins_n119:
    cmp r9d, 119
    jne ins_n120
    insertps xmm0, xmm1, 119
    jmp ins_done
ins_n120:
    cmp r9d, 120
    jne ins_n121
    insertps xmm0, xmm1, 120
    jmp ins_done
ins_n121:
    cmp r9d, 121
    jne ins_n122
    insertps xmm0, xmm1, 121
    jmp ins_done
ins_n122:
    cmp r9d, 122
    jne ins_n123
    insertps xmm0, xmm1, 122
    jmp ins_done
ins_n123:
    cmp r9d, 123
    jne ins_n124
    insertps xmm0, xmm1, 123
    jmp ins_done
ins_n124:
    cmp r9d, 124
    jne ins_n125
    insertps xmm0, xmm1, 124
    jmp ins_done
ins_n125:
    cmp r9d, 125
    jne ins_n126
    insertps xmm0, xmm1, 125
    jmp ins_done
ins_n126:
    cmp r9d, 126
    jne ins_n127
    insertps xmm0, xmm1, 126
    jmp ins_done
ins_n127:
    cmp r9d, 127
    jne ins_n128
    insertps xmm0, xmm1, 127
    jmp ins_done
ins_n128:
    cmp r9d, 128
    jne ins_n129
    insertps xmm0, xmm1, 128
    jmp ins_done
ins_n129:
    cmp r9d, 129
    jne ins_n130
    insertps xmm0, xmm1, 129
    jmp ins_done
ins_n130:
    cmp r9d, 130
    jne ins_n131
    insertps xmm0, xmm1, 130
    jmp ins_done
ins_n131:
    cmp r9d, 131
    jne ins_n132
    insertps xmm0, xmm1, 131
    jmp ins_done
ins_n132:
    cmp r9d, 132
    jne ins_n133
    insertps xmm0, xmm1, 132
    jmp ins_done
ins_n133:
    cmp r9d, 133
    jne ins_n134
    insertps xmm0, xmm1, 133
    jmp ins_done
ins_n134:
    cmp r9d, 134
    jne ins_n135
    insertps xmm0, xmm1, 134
    jmp ins_done
ins_n135:
    cmp r9d, 135
    jne ins_n136
    insertps xmm0, xmm1, 135
    jmp ins_done
ins_n136:
    cmp r9d, 136
    jne ins_n137
    insertps xmm0, xmm1, 136
    jmp ins_done
ins_n137:
    cmp r9d, 137
    jne ins_n138
    insertps xmm0, xmm1, 137
    jmp ins_done
ins_n138:
    cmp r9d, 138
    jne ins_n139
    insertps xmm0, xmm1, 138
    jmp ins_done
ins_n139:
    cmp r9d, 139
    jne ins_n140
    insertps xmm0, xmm1, 139
    jmp ins_done
ins_n140:
    cmp r9d, 140
    jne ins_n141
    insertps xmm0, xmm1, 140
    jmp ins_done
ins_n141:
    cmp r9d, 141
    jne ins_n142
    insertps xmm0, xmm1, 141
    jmp ins_done
ins_n142:
    cmp r9d, 142
    jne ins_n143
    insertps xmm0, xmm1, 142
    jmp ins_done
ins_n143:
    cmp r9d, 143
    jne ins_n144
    insertps xmm0, xmm1, 143
    jmp ins_done
ins_n144:
    cmp r9d, 144
    jne ins_n145
    insertps xmm0, xmm1, 144
    jmp ins_done
ins_n145:
    cmp r9d, 145
    jne ins_n146
    insertps xmm0, xmm1, 145
    jmp ins_done
ins_n146:
    cmp r9d, 146
    jne ins_n147
    insertps xmm0, xmm1, 146
    jmp ins_done
ins_n147:
    cmp r9d, 147
    jne ins_n148
    insertps xmm0, xmm1, 147
    jmp ins_done
ins_n148:
    cmp r9d, 148
    jne ins_n149
    insertps xmm0, xmm1, 148
    jmp ins_done
ins_n149:
    cmp r9d, 149
    jne ins_n150
    insertps xmm0, xmm1, 149
    jmp ins_done
ins_n150:
    cmp r9d, 150
    jne ins_n151
    insertps xmm0, xmm1, 150
    jmp ins_done
ins_n151:
    cmp r9d, 151
    jne ins_n152
    insertps xmm0, xmm1, 151
    jmp ins_done
ins_n152:
    cmp r9d, 152
    jne ins_n153
    insertps xmm0, xmm1, 152
    jmp ins_done
ins_n153:
    cmp r9d, 153
    jne ins_n154
    insertps xmm0, xmm1, 153
    jmp ins_done
ins_n154:
    cmp r9d, 154
    jne ins_n155
    insertps xmm0, xmm1, 154
    jmp ins_done
ins_n155:
    cmp r9d, 155
    jne ins_n156
    insertps xmm0, xmm1, 155
    jmp ins_done
ins_n156:
    cmp r9d, 156
    jne ins_n157
    insertps xmm0, xmm1, 156
    jmp ins_done
ins_n157:
    cmp r9d, 157
    jne ins_n158
    insertps xmm0, xmm1, 157
    jmp ins_done
ins_n158:
    cmp r9d, 158
    jne ins_n159
    insertps xmm0, xmm1, 158
    jmp ins_done
ins_n159:
    cmp r9d, 159
    jne ins_n160
    insertps xmm0, xmm1, 159
    jmp ins_done
ins_n160:
    cmp r9d, 160
    jne ins_n161
    insertps xmm0, xmm1, 160
    jmp ins_done
ins_n161:
    cmp r9d, 161
    jne ins_n162
    insertps xmm0, xmm1, 161
    jmp ins_done
ins_n162:
    cmp r9d, 162
    jne ins_n163
    insertps xmm0, xmm1, 162
    jmp ins_done
ins_n163:
    cmp r9d, 163
    jne ins_n164
    insertps xmm0, xmm1, 163
    jmp ins_done
ins_n164:
    cmp r9d, 164
    jne ins_n165
    insertps xmm0, xmm1, 164
    jmp ins_done
ins_n165:
    cmp r9d, 165
    jne ins_n166
    insertps xmm0, xmm1, 165
    jmp ins_done
ins_n166:
    cmp r9d, 166
    jne ins_n167
    insertps xmm0, xmm1, 166
    jmp ins_done
ins_n167:
    cmp r9d, 167
    jne ins_n168
    insertps xmm0, xmm1, 167
    jmp ins_done
ins_n168:
    cmp r9d, 168
    jne ins_n169
    insertps xmm0, xmm1, 168
    jmp ins_done
ins_n169:
    cmp r9d, 169
    jne ins_n170
    insertps xmm0, xmm1, 169
    jmp ins_done
ins_n170:
    cmp r9d, 170
    jne ins_n171
    insertps xmm0, xmm1, 170
    jmp ins_done
ins_n171:
    cmp r9d, 171
    jne ins_n172
    insertps xmm0, xmm1, 171
    jmp ins_done
ins_n172:
    cmp r9d, 172
    jne ins_n173
    insertps xmm0, xmm1, 172
    jmp ins_done
ins_n173:
    cmp r9d, 173
    jne ins_n174
    insertps xmm0, xmm1, 173
    jmp ins_done
ins_n174:
    cmp r9d, 174
    jne ins_n175
    insertps xmm0, xmm1, 174
    jmp ins_done
ins_n175:
    cmp r9d, 175
    jne ins_n176
    insertps xmm0, xmm1, 175
    jmp ins_done
ins_n176:
    cmp r9d, 176
    jne ins_n177
    insertps xmm0, xmm1, 176
    jmp ins_done
ins_n177:
    cmp r9d, 177
    jne ins_n178
    insertps xmm0, xmm1, 177
    jmp ins_done
ins_n178:
    cmp r9d, 178
    jne ins_n179
    insertps xmm0, xmm1, 178
    jmp ins_done
ins_n179:
    cmp r9d, 179
    jne ins_n180
    insertps xmm0, xmm1, 179
    jmp ins_done
ins_n180:
    cmp r9d, 180
    jne ins_n181
    insertps xmm0, xmm1, 180
    jmp ins_done
ins_n181:
    cmp r9d, 181
    jne ins_n182
    insertps xmm0, xmm1, 181
    jmp ins_done
ins_n182:
    cmp r9d, 182
    jne ins_n183
    insertps xmm0, xmm1, 182
    jmp ins_done
ins_n183:
    cmp r9d, 183
    jne ins_n184
    insertps xmm0, xmm1, 183
    jmp ins_done
ins_n184:
    cmp r9d, 184
    jne ins_n185
    insertps xmm0, xmm1, 184
    jmp ins_done
ins_n185:
    cmp r9d, 185
    jne ins_n186
    insertps xmm0, xmm1, 185
    jmp ins_done
ins_n186:
    cmp r9d, 186
    jne ins_n187
    insertps xmm0, xmm1, 186
    jmp ins_done
ins_n187:
    cmp r9d, 187
    jne ins_n188
    insertps xmm0, xmm1, 187
    jmp ins_done
ins_n188:
    cmp r9d, 188
    jne ins_n189
    insertps xmm0, xmm1, 188
    jmp ins_done
ins_n189:
    cmp r9d, 189
    jne ins_n190
    insertps xmm0, xmm1, 189
    jmp ins_done
ins_n190:
    cmp r9d, 190
    jne ins_n191
    insertps xmm0, xmm1, 190
    jmp ins_done
ins_n191:
    cmp r9d, 191
    jne ins_n192
    insertps xmm0, xmm1, 191
    jmp ins_done
ins_n192:
    cmp r9d, 192
    jne ins_n193
    insertps xmm0, xmm1, 192
    jmp ins_done
ins_n193:
    cmp r9d, 193
    jne ins_n194
    insertps xmm0, xmm1, 193
    jmp ins_done
ins_n194:
    cmp r9d, 194
    jne ins_n195
    insertps xmm0, xmm1, 194
    jmp ins_done
ins_n195:
    cmp r9d, 195
    jne ins_n196
    insertps xmm0, xmm1, 195
    jmp ins_done
ins_n196:
    cmp r9d, 196
    jne ins_n197
    insertps xmm0, xmm1, 196
    jmp ins_done
ins_n197:
    cmp r9d, 197
    jne ins_n198
    insertps xmm0, xmm1, 197
    jmp ins_done
ins_n198:
    cmp r9d, 198
    jne ins_n199
    insertps xmm0, xmm1, 198
    jmp ins_done
ins_n199:
    cmp r9d, 199
    jne ins_n200
    insertps xmm0, xmm1, 199
    jmp ins_done
ins_n200:
    cmp r9d, 200
    jne ins_n201
    insertps xmm0, xmm1, 200
    jmp ins_done
ins_n201:
    cmp r9d, 201
    jne ins_n202
    insertps xmm0, xmm1, 201
    jmp ins_done
ins_n202:
    cmp r9d, 202
    jne ins_n203
    insertps xmm0, xmm1, 202
    jmp ins_done
ins_n203:
    cmp r9d, 203
    jne ins_n204
    insertps xmm0, xmm1, 203
    jmp ins_done
ins_n204:
    cmp r9d, 204
    jne ins_n205
    insertps xmm0, xmm1, 204
    jmp ins_done
ins_n205:
    cmp r9d, 205
    jne ins_n206
    insertps xmm0, xmm1, 205
    jmp ins_done
ins_n206:
    cmp r9d, 206
    jne ins_n207
    insertps xmm0, xmm1, 206
    jmp ins_done
ins_n207:
    cmp r9d, 207
    jne ins_n208
    insertps xmm0, xmm1, 207
    jmp ins_done
ins_n208:
    cmp r9d, 208
    jne ins_n209
    insertps xmm0, xmm1, 208
    jmp ins_done
ins_n209:
    cmp r9d, 209
    jne ins_n210
    insertps xmm0, xmm1, 209
    jmp ins_done
ins_n210:
    cmp r9d, 210
    jne ins_n211
    insertps xmm0, xmm1, 210
    jmp ins_done
ins_n211:
    cmp r9d, 211
    jne ins_n212
    insertps xmm0, xmm1, 211
    jmp ins_done
ins_n212:
    cmp r9d, 212
    jne ins_n213
    insertps xmm0, xmm1, 212
    jmp ins_done
ins_n213:
    cmp r9d, 213
    jne ins_n214
    insertps xmm0, xmm1, 213
    jmp ins_done
ins_n214:
    cmp r9d, 214
    jne ins_n215
    insertps xmm0, xmm1, 214
    jmp ins_done
ins_n215:
    cmp r9d, 215
    jne ins_n216
    insertps xmm0, xmm1, 215
    jmp ins_done
ins_n216:
    cmp r9d, 216
    jne ins_n217
    insertps xmm0, xmm1, 216
    jmp ins_done
ins_n217:
    cmp r9d, 217
    jne ins_n218
    insertps xmm0, xmm1, 217
    jmp ins_done
ins_n218:
    cmp r9d, 218
    jne ins_n219
    insertps xmm0, xmm1, 218
    jmp ins_done
ins_n219:
    cmp r9d, 219
    jne ins_n220
    insertps xmm0, xmm1, 219
    jmp ins_done
ins_n220:
    cmp r9d, 220
    jne ins_n221
    insertps xmm0, xmm1, 220
    jmp ins_done
ins_n221:
    cmp r9d, 221
    jne ins_n222
    insertps xmm0, xmm1, 221
    jmp ins_done
ins_n222:
    cmp r9d, 222
    jne ins_n223
    insertps xmm0, xmm1, 222
    jmp ins_done
ins_n223:
    cmp r9d, 223
    jne ins_n224
    insertps xmm0, xmm1, 223
    jmp ins_done
ins_n224:
    cmp r9d, 224
    jne ins_n225
    insertps xmm0, xmm1, 224
    jmp ins_done
ins_n225:
    cmp r9d, 225
    jne ins_n226
    insertps xmm0, xmm1, 225
    jmp ins_done
ins_n226:
    cmp r9d, 226
    jne ins_n227
    insertps xmm0, xmm1, 226
    jmp ins_done
ins_n227:
    cmp r9d, 227
    jne ins_n228
    insertps xmm0, xmm1, 227
    jmp ins_done
ins_n228:
    cmp r9d, 228
    jne ins_n229
    insertps xmm0, xmm1, 228
    jmp ins_done
ins_n229:
    cmp r9d, 229
    jne ins_n230
    insertps xmm0, xmm1, 229
    jmp ins_done
ins_n230:
    cmp r9d, 230
    jne ins_n231
    insertps xmm0, xmm1, 230
    jmp ins_done
ins_n231:
    cmp r9d, 231
    jne ins_n232
    insertps xmm0, xmm1, 231
    jmp ins_done
ins_n232:
    cmp r9d, 232
    jne ins_n233
    insertps xmm0, xmm1, 232
    jmp ins_done
ins_n233:
    cmp r9d, 233
    jne ins_n234
    insertps xmm0, xmm1, 233
    jmp ins_done
ins_n234:
    cmp r9d, 234
    jne ins_n235
    insertps xmm0, xmm1, 234
    jmp ins_done
ins_n235:
    cmp r9d, 235
    jne ins_n236
    insertps xmm0, xmm1, 235
    jmp ins_done
ins_n236:
    cmp r9d, 236
    jne ins_n237
    insertps xmm0, xmm1, 236
    jmp ins_done
ins_n237:
    cmp r9d, 237
    jne ins_n238
    insertps xmm0, xmm1, 237
    jmp ins_done
ins_n238:
    cmp r9d, 238
    jne ins_n239
    insertps xmm0, xmm1, 238
    jmp ins_done
ins_n239:
    cmp r9d, 239
    jne ins_n240
    insertps xmm0, xmm1, 239
    jmp ins_done
ins_n240:
    cmp r9d, 240
    jne ins_n241
    insertps xmm0, xmm1, 240
    jmp ins_done
ins_n241:
    cmp r9d, 241
    jne ins_n242
    insertps xmm0, xmm1, 241
    jmp ins_done
ins_n242:
    cmp r9d, 242
    jne ins_n243
    insertps xmm0, xmm1, 242
    jmp ins_done
ins_n243:
    cmp r9d, 243
    jne ins_n244
    insertps xmm0, xmm1, 243
    jmp ins_done
ins_n244:
    cmp r9d, 244
    jne ins_n245
    insertps xmm0, xmm1, 244
    jmp ins_done
ins_n245:
    cmp r9d, 245
    jne ins_n246
    insertps xmm0, xmm1, 245
    jmp ins_done
ins_n246:
    cmp r9d, 246
    jne ins_n247
    insertps xmm0, xmm1, 246
    jmp ins_done
ins_n247:
    cmp r9d, 247
    jne ins_n248
    insertps xmm0, xmm1, 247
    jmp ins_done
ins_n248:
    cmp r9d, 248
    jne ins_n249
    insertps xmm0, xmm1, 248
    jmp ins_done
ins_n249:
    cmp r9d, 249
    jne ins_n250
    insertps xmm0, xmm1, 249
    jmp ins_done
ins_n250:
    cmp r9d, 250
    jne ins_n251
    insertps xmm0, xmm1, 250
    jmp ins_done
ins_n251:
    cmp r9d, 251
    jne ins_n252
    insertps xmm0, xmm1, 251
    jmp ins_done
ins_n252:
    cmp r9d, 252
    jne ins_n253
    insertps xmm0, xmm1, 252
    jmp ins_done
ins_n253:
    cmp r9d, 253
    jne ins_n254
    insertps xmm0, xmm1, 253
    jmp ins_done
ins_n254:
    cmp r9d, 254
    jne ins_n255
    insertps xmm0, xmm1, 254
    jmp ins_done
ins_n255:
    cmp r9d, 255
    jne ins_done
    insertps xmm0, xmm1, 255
    jmp ins_done
ins_done:
    movdqu xmmword ptr [r8], xmm0
    ret
insertps_op ENDP

insertps_mem_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    cmp r9d, 0
    jne insm_n1
    insertps xmm0, dword ptr [rdx], 0
    jmp insm_done
insm_n1:
    cmp r9d, 16
    jne insm_n2
    insertps xmm0, dword ptr [rdx], 16
    jmp insm_done
insm_n2:
    cmp r9d, 32
    jne insm_n3
    insertps xmm0, dword ptr [rdx], 32
    jmp insm_done
insm_n3:
    cmp r9d, 48
    jne insm_n4
    insertps xmm0, dword ptr [rdx], 48
    jmp insm_done
insm_n4:
    cmp r9d, 64
    jne insm_n5
    insertps xmm0, dword ptr [rdx], 64
    jmp insm_done
insm_n5:
    cmp r9d, 81
    jne insm_n6
    insertps xmm0, dword ptr [rdx], 81
    jmp insm_done
insm_n6:
    cmp r9d, 98
    jne insm_n7
    insertps xmm0, dword ptr [rdx], 98
    jmp insm_done
insm_n7:
    cmp r9d, 116
    jne insm_n8
    insertps xmm0, dword ptr [rdx], 116
    jmp insm_done
insm_n8:
    cmp r9d, 136
    jne insm_n9
    insertps xmm0, dword ptr [rdx], 136
    jmp insm_done
insm_n9:
    cmp r9d, 159
    jne insm_n10
    insertps xmm0, dword ptr [rdx], 159
    jmp insm_done
insm_n10:
    cmp r9d, 165
    jne insm_n11
    insertps xmm0, dword ptr [rdx], 165
    jmp insm_done
insm_n11:
    cmp r9d, 186
    jne insm_n12
    insertps xmm0, dword ptr [rdx], 186
    jmp insm_done
insm_n12:
    cmp r9d, 193
    jne insm_n13
    insertps xmm0, dword ptr [rdx], 193
    jmp insm_done
insm_n13:
    cmp r9d, 210
    jne insm_n14
    insertps xmm0, dword ptr [rdx], 210
    jmp insm_done
insm_n14:
    cmp r9d, 228
    jne insm_n15
    insertps xmm0, dword ptr [rdx], 228
    jmp insm_done
insm_n15:
    cmp r9d, 248
    jne insm_done
    insertps xmm0, dword ptr [rdx], 248
    jmp insm_done
insm_done:
    movdqu xmmword ptr [r8], xmm0
    ret
insertps_mem_op ENDP

extractps_gpr_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    mov rax, r8
    cmp edx, 0
    jne extg_n1
    extractps eax, xmm0, 0
    jmp extg_done
extg_n1:
    cmp edx, 1
    jne extg_n2
    extractps eax, xmm0, 1
    jmp extg_done
extg_n2:
    cmp edx, 2
    jne extg_n3
    extractps eax, xmm0, 2
    jmp extg_done
extg_n3:
    cmp edx, 3
    jne extg_done
    extractps eax, xmm0, 3
    jmp extg_done
extg_done:
    ret
extractps_gpr_op ENDP

extractps_mem_op PROC FRAME
    .endprolog
    movdqu xmm0, xmmword ptr [rcx]
    cmp edx, 0
    jne extm_n1
    extractps dword ptr [r8], xmm0, 0
    jmp extm_done
extm_n1:
    cmp edx, 1
    jne extm_n2
    extractps dword ptr [r8], xmm0, 1
    jmp extm_done
extm_n2:
    cmp edx, 2
    jne extm_n3
    extractps dword ptr [r8], xmm0, 2
    jmp extm_done
extm_n3:
    cmp edx, 3
    jne extm_done
    extractps dword ptr [r8], xmm0, 3
    jmp extm_done
extm_done:
    ret
extractps_mem_op ENDP

END
