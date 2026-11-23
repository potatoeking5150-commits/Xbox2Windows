; coverage: source cpu_aot_sse2_regs
; Source-built MASM fixture exercising 23 SSE2 forms from docs/coverage/x64_zen2.tsv
; group SSE2: cmppd, every packed/scalar convert expressible in the leaf ABI,
; maxsd/minsd/sqrtsd, lfence/mfence and the aligned, half-qword and
; non-temporal moves. The MMX-operand forms (cvtpd2pi, cvttpd2pi, movdq2q,
; movq2dq, cvtpi2pd mm source) and implicit-RDI maskmovdqu are outside the leaf
; ABI and are lifted as rejections, not exports.
; Every export uses the Windows x64 leaf ABI:
;   RCX = const void* input_a (32 bytes, 16-byte aligned)
;   RDX = const void* input_b (32 bytes, 16-byte aligned; second vector at +16)
;   R8  = void*       output  (32 bytes, 16-byte aligned)
.code

; --- cmppd: all eight predicates on register and aligned m128 sources ---

op_cmppd_0 PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movdqa xmm1, XMMWORD PTR [rdx]
    cmppd xmm0, xmm1, 0
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_cmppd_0 ENDP

op_cmppd_1 PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movdqa xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    cmppd xmm0, xmm1, 5
ELSE
    cmppd xmm0, xmm1, 1
ENDIF
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_cmppd_1 ENDP

op_cmppd_2 PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movdqa xmm1, XMMWORD PTR [rdx]
    cmppd xmm0, xmm1, 2
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_cmppd_2 ENDP

op_cmppd_3 PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movdqa xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    cmppd xmm0, xmm1, 7
ELSE
    cmppd xmm0, xmm1, 3
ENDIF
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_cmppd_3 ENDP

op_cmppd_4 PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movdqa xmm1, XMMWORD PTR [rdx]
    cmppd xmm0, xmm1, 4
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_cmppd_4 ENDP

op_cmppd_5 PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movdqa xmm1, XMMWORD PTR [rdx]
    cmppd xmm0, xmm1, 5
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_cmppd_5 ENDP

op_cmppd_6 PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movdqa xmm1, XMMWORD PTR [rdx]
IFDEF ALTER_IMM
    cmppd xmm0, xmm1, 2
ELSE
    cmppd xmm0, xmm1, 6
ENDIF
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_cmppd_6 ENDP

op_cmppd_7 PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movdqa xmm1, XMMWORD PTR [rdx]
    cmppd xmm0, xmm1, 7
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_cmppd_7 ENDP

op_cmppd_mem PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    cmppd xmm0, XMMWORD PTR [rdx], 5
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_cmppd_mem ENDP

; --- Scalar double ops: low-qword merge, high-qword preserved ---

op_maxsd_reg PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movdqa xmm1, XMMWORD PTR [rdx]
    maxsd xmm0, xmm1
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_maxsd_reg ENDP

op_maxsd_mem PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    maxsd xmm0, QWORD PTR [rdx]
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_maxsd_mem ENDP

op_minsd_reg PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movdqa xmm1, XMMWORD PTR [rdx]
    minsd xmm0, xmm1
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_minsd_reg ENDP

op_minsd_mem PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    minsd xmm0, QWORD PTR [rdx]
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_minsd_mem ENDP

op_sqrtsd_reg PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movdqa xmm1, XMMWORD PTR [rdx]
    sqrtsd xmm0, xmm1
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_sqrtsd_reg ENDP

op_sqrtsd_mem PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    sqrtsd xmm0, QWORD PTR [rdx]
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_sqrtsd_mem ENDP

; --- cvtsd2si: r32 and r64 destinations, xmm and m64 sources ---

op_cvtsd2si_r32_reg PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    cvtsd2si eax, xmm0
    mov DWORD PTR [r8], eax
    ret
op_cvtsd2si_r32_reg ENDP

op_cvtsd2si_r64_reg PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    cvtsd2si rax, xmm0
    mov QWORD PTR [r8], rax
    ret
op_cvtsd2si_r64_reg ENDP

op_cvtsd2si_r32_mem PROC FRAME
    .endprolog
    cvtsd2si eax, QWORD PTR [rdx]
    mov DWORD PTR [r8], eax
    ret
op_cvtsd2si_r32_mem ENDP

op_cvtsd2si_r64_mem PROC FRAME
    .endprolog
    cvtsd2si rax, QWORD PTR [rdx]
    mov QWORD PTR [r8], rax
    ret
op_cvtsd2si_r64_mem ENDP

; --- Packed converts ---

op_cvtdq2pd_reg PROC FRAME
    .endprolog
    movdqa xmm1, XMMWORD PTR [rdx]
    cvtdq2pd xmm0, xmm1
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_cvtdq2pd_reg ENDP

op_cvtdq2pd_mem PROC FRAME
    .endprolog
    cvtdq2pd xmm0, QWORD PTR [rdx]
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_cvtdq2pd_mem ENDP

op_cvtdq2ps_reg PROC FRAME
    .endprolog
    movdqa xmm1, XMMWORD PTR [rdx]
    cvtdq2ps xmm0, xmm1
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_cvtdq2ps_reg ENDP

op_cvtdq2ps_mem PROC FRAME
    .endprolog
    cvtdq2ps xmm0, XMMWORD PTR [rdx]
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_cvtdq2ps_mem ENDP

op_cvtpd2dq_reg PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movdqa xmm1, XMMWORD PTR [rdx]
    cvtpd2dq xmm0, xmm1
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_cvtpd2dq_reg ENDP

op_cvtpd2dq_mem PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    cvtpd2dq xmm0, XMMWORD PTR [rdx]
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_cvtpd2dq_mem ENDP

op_cvtpd2ps_reg PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movdqa xmm1, XMMWORD PTR [rdx]
    cvtpd2ps xmm0, xmm1
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_cvtpd2ps_reg ENDP

op_cvtpd2ps_mem PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    cvtpd2ps xmm0, XMMWORD PTR [rdx]
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_cvtpd2ps_mem ENDP

op_cvtpi2pd_mem PROC FRAME
    .endprolog
    cvtpi2pd xmm0, QWORD PTR [rdx]
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_cvtpi2pd_mem ENDP

op_cvtps2dq_reg PROC FRAME
    .endprolog
    movdqa xmm1, XMMWORD PTR [rdx]
    cvtps2dq xmm0, xmm1
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_cvtps2dq_reg ENDP

op_cvtps2dq_mem PROC FRAME
    .endprolog
    cvtps2dq xmm0, XMMWORD PTR [rdx]
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_cvtps2dq_mem ENDP

op_cvtps2pd_reg PROC FRAME
    .endprolog
    movdqa xmm1, XMMWORD PTR [rdx]
    cvtps2pd xmm0, xmm1
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_cvtps2pd_reg ENDP

op_cvtps2pd_mem PROC FRAME
    .endprolog
    cvtps2pd xmm0, QWORD PTR [rdx]
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_cvtps2pd_mem ENDP

op_cvttpd2dq_reg PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movdqa xmm1, XMMWORD PTR [rdx]
    cvttpd2dq xmm0, xmm1
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_cvttpd2dq_reg ENDP

op_cvttpd2dq_mem PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    cvttpd2dq xmm0, XMMWORD PTR [rdx]
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_cvttpd2dq_mem ENDP

op_cvttps2dq_reg PROC FRAME
    .endprolog
    movdqa xmm1, XMMWORD PTR [rdx]
    cvttps2dq xmm0, xmm1
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_cvttps2dq_reg ENDP

op_cvttps2dq_mem PROC FRAME
    .endprolog
    cvttps2dq xmm0, XMMWORD PTR [rdx]
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_cvttps2dq_mem ENDP

; --- Aligned moves: load, reg copy and store ---

op_movapd_ls PROC FRAME
    .endprolog
    movapd xmm0, XMMWORD PTR [rcx]
    movapd XMMWORD PTR [r8], xmm0
    ret
op_movapd_ls ENDP

op_movapd_reg PROC FRAME
    .endprolog
    movdqa xmm1, XMMWORD PTR [rcx]
    movapd xmm0, xmm1
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_movapd_reg ENDP

op_movdqa_ls PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_movdqa_ls ENDP

op_movdqa_reg PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movdqa xmm1, xmm0
    movdqa XMMWORD PTR [r8], xmm1
    ret
op_movdqa_reg ENDP

; --- Half-qword moves: m64 load merge and m64 store ---

op_movhpd_load PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movhpd xmm0, QWORD PTR [rdx]
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_movhpd_load ENDP

op_movhpd_store PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movhpd QWORD PTR [r8+8], xmm0
    ret
op_movhpd_store ENDP

op_movlpd_load PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movlpd xmm0, QWORD PTR [rdx]
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_movlpd_load ENDP

op_movlpd_store PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movlpd QWORD PTR [r8+8], xmm0
    ret
op_movlpd_store ENDP

op_movhpd_load_plus1 PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movhpd xmm0, QWORD PTR [rdx+1]
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_movhpd_load_plus1 ENDP

op_movlpd_load_plus1 PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movlpd xmm0, QWORD PTR [rdx+1]
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_movlpd_load_plus1 ENDP

op_movhpd_store_plus1 PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movhpd QWORD PTR [r8+9], xmm0
    ret
op_movhpd_store_plus1 ENDP

op_movlpd_store_plus1 PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movlpd QWORD PTR [r8+9], xmm0
    ret
op_movlpd_store_plus1 ENDP

; --- Unaligned +1 m64 scalar and convert sources ---

op_maxsd_mem_plus1 PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    maxsd xmm0, QWORD PTR [rdx+1]
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_maxsd_mem_plus1 ENDP

op_cvtsd2si_mem_plus1 PROC FRAME
    .endprolog
    cvtsd2si eax, QWORD PTR [rdx+1]
    mov DWORD PTR [r8], eax
    ret
op_cvtsd2si_mem_plus1 ENDP

; --- Non-temporal stores ---

op_movntdq PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movntdq XMMWORD PTR [r8], xmm0
    ret
op_movntdq ENDP

op_movntpd PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movntpd XMMWORD PTR [r8], xmm0
    ret
op_movntpd ENDP

op_movnti32 PROC FRAME
    .endprolog
    mov eax, DWORD PTR [rdx]
    movnti DWORD PTR [r8], eax
    ret
op_movnti32 ENDP

op_movnti64 PROC FRAME
    .endprolog
    mov rax, QWORD PTR [rdx]
    movnti QWORD PTR [r8], rax
    ret
op_movnti64 ENDP

; --- Fences: serializing instructions with no observable state ---

op_lfence PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    lfence
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_lfence ENDP

op_mfence PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    mfence
    movdqa XMMWORD PTR [r8], xmm0
    ret
op_mfence ENDP

; --- Chained pipeline: masks, convert, merges, scalar ops and GPR bridge ---

op_pipeline PROC FRAME
    .endprolog
    movdqa xmm0, XMMWORD PTR [rcx]
    movdqa xmm1, XMMWORD PTR [rdx]
    movdqa xmm2, XMMWORD PTR [rdx+16]
    cmppd xmm0, xmm1, 2
    cvtpd2ps xmm3, xmm0
    movhpd xmm3, QWORD PTR [rdx+24]
    maxsd xmm3, xmm1
    minsd xmm3, xmm2
    sqrtsd xmm2, xmm3
    cvtsd2si eax, xmm3
    movdqa XMMWORD PTR [r8], xmm2
    mov DWORD PTR [r8+16], eax
    ret
op_pipeline ENDP

END
