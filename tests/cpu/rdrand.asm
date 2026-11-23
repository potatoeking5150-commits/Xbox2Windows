; coverage: source cpu_aot_rdrand
; Source-owned differential fixture for RDRAND. rdrand_16, rdrand_32 and rdrand_64 take (a, b, preset, state) in RCX,
; RDX, R8 and R9. RAX starts as all ones, so the width rules show: a 16-bit result keeps bits 16-63, a 32-bit
; result clears them. ADD R8, R8 sets ZF, CF and OF from the preset before the read. After it, SETB, SETE and
; SETO store CF, ZF and OF to bytes 0-2 of state slot 1 (cleared first), and RAX goes to slot 0.
; rdrand_unused discards its result and returns 7. CPUID and the RFLAGS reads are in rdrand_harness.asm.
.code

rdrand_16 PROC FRAME
    .endprolog
    mov rax, -1
    mov qword ptr [r9+8], 0
    add r8, r8
    rdrand ax
    setb byte ptr [r9+8]
    sete byte ptr [r9+9]
    seto byte ptr [r9+10]
    mov qword ptr [r9], rax
    ret
rdrand_16 ENDP

rdrand_32 PROC FRAME
    .endprolog
    mov rax, -1
    mov qword ptr [r9+8], 0
    add r8, r8
    rdrand eax
    setb byte ptr [r9+8]
    sete byte ptr [r9+9]
    seto byte ptr [r9+10]
    mov qword ptr [r9], rax
    ret
rdrand_32 ENDP

rdrand_64 PROC FRAME
    .endprolog
    mov rax, -1
    mov qword ptr [r9+8], 0
    add r8, r8
    rdrand rax
    setb byte ptr [r9+8]
    sete byte ptr [r9+9]
    seto byte ptr [r9+10]
    mov qword ptr [r9], rax
    ret
rdrand_64 ENDP

; A read whose result is overwritten: the translation must still perform it, also at /O2.
rdrand_unused PROC FRAME
    .endprolog
    rdrand eax
    mov eax, 7
    ret
rdrand_unused ENDP

END
