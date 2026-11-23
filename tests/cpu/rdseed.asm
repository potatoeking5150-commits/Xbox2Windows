; coverage: source cpu_aot_rdseed
; Source-owned differential fixture for RDSEED. rdseed_16, rdseed_32 and rdseed_64 take (a, b, preset, state) in RCX,
; RDX, R8 and R9. RAX starts as all ones, so the width rules show: a 16-bit result keeps bits 16-63, a 32-bit
; result clears them. ADD R8, R8 sets ZF, CF and OF from the preset before the read. After it, SETB, SETE and
; SETO store CF, ZF and OF to bytes 0-2 of state slot 1 (cleared first), and RAX goes to slot 0.
; rdseed_unused discards its result and returns 7. CPUID and the RFLAGS reads are in rdrand_harness.asm.
.code

rdseed_16 PROC FRAME
    .endprolog
    mov rax, -1
    mov qword ptr [r9+8], 0
    add r8, r8
    rdseed ax
    setb byte ptr [r9+8]
    sete byte ptr [r9+9]
    seto byte ptr [r9+10]
    mov qword ptr [r9], rax
    ret
rdseed_16 ENDP

rdseed_32 PROC FRAME
    .endprolog
    mov rax, -1
    mov qword ptr [r9+8], 0
    add r8, r8
    rdseed eax
    setb byte ptr [r9+8]
    sete byte ptr [r9+9]
    seto byte ptr [r9+10]
    mov qword ptr [r9], rax
    ret
rdseed_32 ENDP

rdseed_64 PROC FRAME
    .endprolog
    mov rax, -1
    mov qword ptr [r9+8], 0
    add r8, r8
    rdseed rax
    setb byte ptr [r9+8]
    sete byte ptr [r9+9]
    seto byte ptr [r9+10]
    mov qword ptr [r9], rax
    ret
rdseed_64 ENDP

; A read whose result is overwritten: the translation must still perform it, also at /O2.
rdseed_unused PROC FRAME
    .endprolog
    rdseed eax
    mov eax, 7
    ret
rdseed_unused ENDP

END
