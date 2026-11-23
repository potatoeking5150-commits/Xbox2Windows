; coverage: source cpu_aot_fma3
; Source-built MASM fixture exercising all 60 VEX FMA3 forms
; (docs/coverage/x64_zen2.tsv group FMA) across scalar ss/sd, packed XMM/YMM,
; and memory op2 forms.
; Every export uses the Windows x64 leaf ABI:
;   RCX = const void* inputs  (96 bytes: op0[32], op1[32], op2[32])
;   RDX = const void* mem_op2 (32 bytes for memory-operand exports)
;   R8  = void*       output  (32 bytes)
.code

op_vfmadd132ps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmadd132ps xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmadd132ps_xmm ENDP

op_vfmadd132ps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmadd132ps ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmadd132ps_ymm ENDP

op_vfmadd132pd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmadd132pd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmadd132pd_xmm ENDP

op_vfmadd132pd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmadd132pd ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmadd132pd_ymm ENDP

op_vfmadd132ss PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmadd132ss xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmadd132ss ENDP

op_vfmadd132sd PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmadd132sd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmadd132sd ENDP

op_vfmadd213ps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
IFDEF ALTER_FMA
    vfmsub213ps xmm0, xmm1, xmm2
ELSE
    vfmadd213ps xmm0, xmm1, xmm2
ENDIF
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmadd213ps_xmm ENDP

op_vfmadd213ps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmadd213ps ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmadd213ps_ymm ENDP

op_vfmadd213pd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmadd213pd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmadd213pd_xmm ENDP

op_vfmadd213pd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmadd213pd ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmadd213pd_ymm ENDP

op_vfmadd213ss PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmadd213ss xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmadd213ss ENDP

op_vfmadd213sd PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmadd213sd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmadd213sd ENDP

op_vfmadd231ps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmadd231ps xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmadd231ps_xmm ENDP

op_vfmadd231ps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmadd231ps ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmadd231ps_ymm ENDP

op_vfmadd231pd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmadd231pd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmadd231pd_xmm ENDP

op_vfmadd231pd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmadd231pd ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmadd231pd_ymm ENDP

op_vfmadd231ss PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmadd231ss xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmadd231ss ENDP

op_vfmadd231sd PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmadd231sd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmadd231sd ENDP

op_vfmsub132ps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsub132ps xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsub132ps_xmm ENDP

op_vfmsub132ps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsub132ps ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsub132ps_ymm ENDP

op_vfmsub132pd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsub132pd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsub132pd_xmm ENDP

op_vfmsub132pd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsub132pd ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsub132pd_ymm ENDP

op_vfmsub132ss PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsub132ss xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsub132ss ENDP

op_vfmsub132sd PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsub132sd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsub132sd ENDP

op_vfmsub213ps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsub213ps xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsub213ps_xmm ENDP

op_vfmsub213ps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsub213ps ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsub213ps_ymm ENDP

op_vfmsub213pd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsub213pd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsub213pd_xmm ENDP

op_vfmsub213pd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsub213pd ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsub213pd_ymm ENDP

op_vfmsub213ss PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsub213ss xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsub213ss ENDP

op_vfmsub213sd PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsub213sd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsub213sd ENDP

op_vfmsub231ps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsub231ps xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsub231ps_xmm ENDP

op_vfmsub231ps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsub231ps ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsub231ps_ymm ENDP

op_vfmsub231pd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsub231pd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsub231pd_xmm ENDP

op_vfmsub231pd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsub231pd ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsub231pd_ymm ENDP

op_vfmsub231ss PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsub231ss xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsub231ss ENDP

op_vfmsub231sd PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsub231sd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsub231sd ENDP

op_vfnmadd132ps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmadd132ps xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmadd132ps_xmm ENDP

op_vfnmadd132ps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmadd132ps ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmadd132ps_ymm ENDP

op_vfnmadd132pd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmadd132pd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmadd132pd_xmm ENDP

op_vfnmadd132pd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmadd132pd ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmadd132pd_ymm ENDP

op_vfnmadd132ss PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmadd132ss xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmadd132ss ENDP

op_vfnmadd132sd PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmadd132sd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmadd132sd ENDP

op_vfnmadd213ps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmadd213ps xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmadd213ps_xmm ENDP

op_vfnmadd213ps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmadd213ps ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmadd213ps_ymm ENDP

op_vfnmadd213pd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmadd213pd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmadd213pd_xmm ENDP

op_vfnmadd213pd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmadd213pd ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmadd213pd_ymm ENDP

op_vfnmadd213ss PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmadd213ss xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmadd213ss ENDP

op_vfnmadd213sd PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmadd213sd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmadd213sd ENDP

op_vfnmadd231ps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmadd231ps xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmadd231ps_xmm ENDP

op_vfnmadd231ps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmadd231ps ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmadd231ps_ymm ENDP

op_vfnmadd231pd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmadd231pd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmadd231pd_xmm ENDP

op_vfnmadd231pd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmadd231pd ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmadd231pd_ymm ENDP

op_vfnmadd231ss PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmadd231ss xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmadd231ss ENDP

op_vfnmadd231sd PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmadd231sd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmadd231sd ENDP

op_vfnmsub132ps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmsub132ps xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmsub132ps_xmm ENDP

op_vfnmsub132ps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmsub132ps ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmsub132ps_ymm ENDP

op_vfnmsub132pd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmsub132pd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmsub132pd_xmm ENDP

op_vfnmsub132pd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmsub132pd ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmsub132pd_ymm ENDP

op_vfnmsub132ss PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmsub132ss xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmsub132ss ENDP

op_vfnmsub132sd PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmsub132sd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmsub132sd ENDP

op_vfnmsub213ps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmsub213ps xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmsub213ps_xmm ENDP

op_vfnmsub213ps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmsub213ps ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmsub213ps_ymm ENDP

op_vfnmsub213pd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmsub213pd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmsub213pd_xmm ENDP

op_vfnmsub213pd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmsub213pd ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmsub213pd_ymm ENDP

op_vfnmsub213ss PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmsub213ss xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmsub213ss ENDP

op_vfnmsub213sd PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmsub213sd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmsub213sd ENDP

op_vfnmsub231ps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmsub231ps xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmsub231ps_xmm ENDP

op_vfnmsub231ps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmsub231ps ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmsub231ps_ymm ENDP

op_vfnmsub231pd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmsub231pd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmsub231pd_xmm ENDP

op_vfnmsub231pd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmsub231pd ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmsub231pd_ymm ENDP

op_vfnmsub231ss PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmsub231ss xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmsub231ss ENDP

op_vfnmsub231sd PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmsub231sd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfnmsub231sd ENDP

op_vfmaddsub132ps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmaddsub132ps xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmaddsub132ps_xmm ENDP

op_vfmaddsub132ps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmaddsub132ps ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmaddsub132ps_ymm ENDP

op_vfmaddsub132pd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmaddsub132pd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmaddsub132pd_xmm ENDP

op_vfmaddsub132pd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmaddsub132pd ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmaddsub132pd_ymm ENDP

op_vfmaddsub213ps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmaddsub213ps xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmaddsub213ps_xmm ENDP

op_vfmaddsub213ps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmaddsub213ps ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmaddsub213ps_ymm ENDP

op_vfmaddsub213pd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmaddsub213pd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmaddsub213pd_xmm ENDP

op_vfmaddsub213pd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmaddsub213pd ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmaddsub213pd_ymm ENDP

op_vfmaddsub231ps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmaddsub231ps xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmaddsub231ps_xmm ENDP

op_vfmaddsub231ps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmaddsub231ps ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmaddsub231ps_ymm ENDP

op_vfmaddsub231pd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmaddsub231pd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmaddsub231pd_xmm ENDP

op_vfmaddsub231pd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmaddsub231pd ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmaddsub231pd_ymm ENDP

op_vfmsubadd132ps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsubadd132ps xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsubadd132ps_xmm ENDP

op_vfmsubadd132ps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsubadd132ps ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsubadd132ps_ymm ENDP

op_vfmsubadd132pd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsubadd132pd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsubadd132pd_xmm ENDP

op_vfmsubadd132pd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsubadd132pd ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsubadd132pd_ymm ENDP

op_vfmsubadd213ps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsubadd213ps xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsubadd213ps_xmm ENDP

op_vfmsubadd213ps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsubadd213ps ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsubadd213ps_ymm ENDP

op_vfmsubadd213pd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsubadd213pd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsubadd213pd_xmm ENDP

op_vfmsubadd213pd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsubadd213pd ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsubadd213pd_ymm ENDP

op_vfmsubadd231ps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsubadd231ps xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsubadd231ps_xmm ENDP

op_vfmsubadd231ps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsubadd231ps ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsubadd231ps_ymm ENDP

op_vfmsubadd231pd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsubadd231pd xmm0, xmm1, xmm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsubadd231pd_xmm ENDP

op_vfmsubadd231pd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsubadd231pd ymm0, ymm1, ymm2
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_vfmsubadd231pd_ymm ENDP

op_mem_vfmadd132ss PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmadd132ss xmm0, xmm1, DWORD PTR [rdx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_mem_vfmadd132ss ENDP

op_mem_vfmsub213sd PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsub213sd xmm0, xmm1, QWORD PTR [rdx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_mem_vfmsub213sd ENDP

op_mem_vfnmadd231ps_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmadd231ps xmm0, xmm1, XMMWORD PTR [rdx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_mem_vfnmadd231ps_xmm ENDP

op_mem_vfnmsub132ps_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfnmsub132ps ymm0, ymm1, YMMWORD PTR [rdx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_mem_vfnmsub132ps_ymm ENDP

op_mem_vfmaddsub213pd_xmm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmaddsub213pd xmm0, xmm1, XMMWORD PTR [rdx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_mem_vfmaddsub213pd_xmm ENDP

op_mem_vfmsubadd231pd_ymm PROC FRAME
    .endprolog
    vmovdqu ymm0, YMMWORD PTR [rcx]
    vmovdqu ymm1, YMMWORD PTR [rcx+32]
    vmovdqu ymm2, YMMWORD PTR [rcx+64]
    vfmsubadd231pd ymm0, ymm1, YMMWORD PTR [rdx]
    vmovdqu YMMWORD PTR [r8], ymm0
    vzeroupper
    ret
op_mem_vfmsubadd231pd_ymm ENDP

END
