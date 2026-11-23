; coverage: source cpu_aot_rdtscp
; Source-owned differential fixture for RDTSCP, in its own file so that its coverage follows cpu_aot_rdtscp,
; which exits 77 on a host without RDTSCP. rdtscp_op takes (a, b, preset, state) in RCX, RDX, R8 and R9.
; ADD R8, R8 sets ZF and CF from the preset; RDTSCP leaves them alone, and (CF << 1) | ZF goes to state
; slot 3. RAX, RDX and RCX start as all ones, so a translation that does not zero-extend the 32-bit results
; leaves bits behind; RAX, RDX and RCX go to slots 0 to 2. rdtscp_unused ignores its arguments and
; overwrites the result with 7.
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

rdtscp_op PROC FRAME
    .endprolog
    mov rax, -1
    mov rdx, -1
    mov rcx, -1
    add r8, r8
    rdtscp
    mov qword ptr [r9], rax
    mov qword ptr [r9+8], rdx
    mov qword ptr [r9+16], rcx
    TSC_FLAGS 24
    ret
rdtscp_op ENDP

rdtscp_unused PROC FRAME
    .endprolog
    rdtscp
    mov eax, 7
    xor edx, edx
    xor ecx, ecx
    ret
rdtscp_unused ENDP

END
