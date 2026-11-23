; coverage: source cpu_aot_rdtsc
; Source-owned differential fixture for RDTSC; RDTSCP is in rdtscp.asm. rdtsc_op and rdtsc_twice take
; (a, b, preset, state) in RCX, RDX, R8 and R9. ADD R8, R8 sets ZF and CF from the preset; RDTSC leaves them
; alone, and (CF << 1) | ZF goes to the state slot after the results. rdtsc_op fills RAX and RDX with all ones first, so a
; translation that does not zero-extend the 32-bit results leaves bits behind. The results go to state.
; rdtsc_unused ignores its arguments and overwrites the result with 7.
.code

TSC_FLAGS MACRO slot
    LOCAL no_carry, no_zero
    mov r10d, 0
    jae short no_carry
    mov r10d, 2
no_carry:
    jne short no_zero
    or r10d, 1
no_zero:
    mov qword ptr [r9+slot], r10
ENDM

rdtsc_op PROC FRAME
    .endprolog
    mov rax, -1
    mov rdx, -1
    add r8, r8
    rdtsc
    mov qword ptr [r9], rax
    mov qword ptr [r9+8], rdx
    TSC_FLAGS 24
    ret
rdtsc_op ENDP

; Two reads in one function: the second must not be earlier than the first, and neither may be merged away.
rdtsc_twice PROC FRAME
    .endprolog
    add r8, r8
    rdtsc
    mov qword ptr [r9], rax
    mov qword ptr [r9+8], rdx
    rdtsc
    mov qword ptr [r9+16], rax
    mov qword ptr [r9+24], rdx
    TSC_FLAGS 32
    ret
rdtsc_twice ENDP

; A read whose result is overwritten: the translation must still perform it, also at /O2.
rdtsc_unused PROC FRAME
    .endprolog
    rdtsc
    mov eax, 7
    xor edx, edx
    ret
rdtsc_unused ENDP

END
