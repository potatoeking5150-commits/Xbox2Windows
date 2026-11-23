; coverage: source cpu_aot_avx_packed_int
; Source-owned differential fixture for 26 VEX AVX packed integer ALU instructions.
.code

vpaddb_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpaddb xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpaddb_op ENDP

vpaddw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpaddw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpaddw_op ENDP

vpaddq_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpaddq xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpaddq_op ENDP

vpsubb_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpsubb xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpsubb_op ENDP

vpsubw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpsubw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpsubw_op ENDP

vpsubd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpsubd xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpsubd_op ENDP

vpsubq_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpsubq xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpsubq_op ENDP

vpand_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpand xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpand_op ENDP

vpandn_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpandn xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpandn_op ENDP

vpor_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpor xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpor_op ENDP

vpcmpeqb_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpcmpeqb xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpcmpeqb_op ENDP

vpcmpeqw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpcmpeqw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpcmpeqw_op ENDP

vpcmpeqd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpcmpeqd xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpcmpeqd_op ENDP

vpcmpeqq_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpcmpeqq xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpcmpeqq_op ENDP

vpcmpgtb_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpcmpgtb xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpcmpgtb_op ENDP

vpcmpgtw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpcmpgtw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpcmpgtw_op ENDP

vpcmpgtd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpcmpgtd xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpcmpgtd_op ENDP

vpcmpgtq_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpcmpgtq xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpcmpgtq_op ENDP

vpabsb_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpabsb xmm0, xmm1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpabsb_op ENDP

vpabsw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpabsw xmm0, xmm1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpabsw_op ENDP

vpabsd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpabsd xmm0, xmm1
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpabsd_op ENDP

vpsignb_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpsignb xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpsignb_op ENDP

vpsignw_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpsignw xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpsignw_op ENDP

vpsignd_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpsignd xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpsignd_op ENDP

vpshufb_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpshufb xmm0, xmm1, xmm2
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpshufb_op ENDP

vpalignr_op PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vmovups ymm2, ymmword ptr [rdx]
    vpalignr xmm0, xmm1, xmm2, 5
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpalignr_op ENDP

vpshufb256_mem PROC FRAME
    .endprolog
    vmovups ymm1, ymmword ptr [rcx]
    vpshufb ymm0, ymm1, ymmword ptr [rdx]
    vmovups ymmword ptr [r8], ymm0
    vzeroupper
    ret
vpshufb256_mem ENDP

END
