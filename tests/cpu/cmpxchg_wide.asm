; coverage: source cpu_aot_cmpxchg_wide
; Source-owned differential fixture for CMPXCHG8B and CMPXCHG16B without LOCK. RCX points to a block:
; +0 the 16-byte memory operand, +16 RAX, +24 RDX, +32 RBX and +40 RCX on entry; on return +48 RAX,
; +56 RDX, +64 ZF (a byte) and +65 CF (a byte, which both instructions leave unchanged). RDX on entry
; is the operand offset from the block (0 or 8, or 4 for a misaligned CMPXCHG16B operand) and bit 0 of
; R8 the CF to start with. The prolog saves RBX in the home area with unwind data, so an operand that
; faults still restores the caller's RBX.
.code

cmpxchg8b_op PROC FRAME
    mov qword ptr [rsp+8], rbx
    .savereg rbx, 8
    .endprolog
    mov r9, rcx
    add r9, rdx
    mov r10, rcx
    bt r8, 0
    mov rax, qword ptr [r10+16]
    mov rdx, qword ptr [r10+24]
    mov rbx, qword ptr [r10+32]
    mov rcx, qword ptr [r10+40]
    cmpxchg8b qword ptr [r9]
    sete byte ptr [r10+64]
    setb byte ptr [r10+65]
    mov qword ptr [r10+48], rax
    mov qword ptr [r10+56], rdx
    mov rbx, qword ptr [rsp+8]
    ret
cmpxchg8b_op ENDP

cmpxchg16b_op PROC FRAME
    mov qword ptr [rsp+8], rbx
    .savereg rbx, 8
    .endprolog
    mov r9, rcx
    add r9, rdx
    mov r10, rcx
    bt r8, 0
    mov rax, qword ptr [r10+16]
    mov rdx, qword ptr [r10+24]
    mov rbx, qword ptr [r10+32]
    mov rcx, qword ptr [r10+40]
    cmpxchg16b oword ptr [r9]
    sete byte ptr [r10+64]
    setb byte ptr [r10+65]
    mov qword ptr [r10+48], rax
    mov qword ptr [r10+56], rdx
    mov rbx, qword ptr [rsp+8]
    ret
cmpxchg16b_op ENDP

END
