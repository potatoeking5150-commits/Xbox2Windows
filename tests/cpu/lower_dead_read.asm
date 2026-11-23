; coverage: source cpu_aot_lower_dead_reads
; Source-owned control for guest memory reads in the shared integer lowering (ffx_version_query.lower).
; The processor performs each read of an instruction once, so an inaccessible operand faults whether or
; not the value is used. Every export takes RCX = address read, RDX = y and R8 = address of three
; writable QWORDs. Each stores 1 to [R8] before its read and 2 to [R8+8] after it.
.code
; Reads eight bytes at RCX, then overwrites the value. RAX = y.
read_dead PROC FRAME
.endprolog
    mov QWORD PTR [r8], 1
    mov rax, [rcx]
    mov QWORD PTR [r8+8], 2
    mov rax, rdx
    ret
read_dead ENDP

; Compares four bytes at RCX with zero; the flags are never read. RAX = y + 1.
cmp_dead PROC FRAME
.endprolog
    mov QWORD PTR [r8], 1
    cmp DWORD PTR [rcx], 0
    mov QWORD PTR [r8+8], 2
    lea rax, [rdx+1]
    ret
cmp_dead ENDP

; Tests one byte at RCX; the flags are never read. RAX = y.
test_dead PROC FRAME
.endprolog
    mov QWORD PTR [r8], 1
    test BYTE PTR [rcx], 1
    mov QWORD PTR [r8+8], 2
    mov rax, rdx
    ret
test_dead ENDP

; CMOVNE after CMP EDX, EDX never moves, but still reads its eight-byte source. RAX = 0.
cmov_dead PROC FRAME
.endprolog
    mov QWORD PTR [r8], 1
    xor eax, eax
    cmp edx, edx
    cmovne rax, [rcx]
    mov QWORD PTR [r8+8], 2
    ret
cmov_dead ENDP

; Adds EDX to the four bytes at RCX in place: one read, one write. RAX = 1 when the sum is zero.
add_in_place PROC FRAME
.endprolog
    mov QWORD PTR [r8], 1
    add DWORD PTR [rcx], edx
    sete al
    movzx eax, al
    mov QWORD PTR [r8+8], 2
    ret
add_in_place ENDP

; Control: the value read is returned, so no compiler can drop the read.
read_live PROC FRAME
.endprolog
    mov QWORD PTR [r8], 1
    mov rax, [rcx]
    mov QWORD PTR [r8+8], 2
    ret
read_live ENDP

; Through the x87 profile: loads the eight bytes at RCX and pops them unused. RAX = y.
fld_dead PROC FRAME
.endprolog
    mov QWORD PTR [r8], 1
    fld QWORD PTR [rcx]
    fstp st(0)
    mov QWORD PTR [r8+8], 2
    mov rax, rdx
    ret
fld_dead ENDP

; Reads sixteen bytes at RCX, then writes a new value to XMM0 without using the read.
read_vector_dead PROC FRAME
.endprolog
    mov QWORD PTR [r8], 1
    movdqu xmm0, XMMWORD PTR [rcx]
    pxor xmm0, xmm0
    mov QWORD PTR [r8+8], 2
    mov rax, rdx
    ret
read_vector_dead ENDP

; RCX is the source and RDX the byte count. Use the third trace word as destination.
; Preserve the actual Windows nonvolatile registers on either return or unwind.
copy_bytes PROC FRAME
    push rsi
    .pushreg rsi
    push rdi
    .pushreg rdi
.endprolog
    mov QWORD PTR [r8], 1
    mov rsi, rcx
    lea rdi, [r8+16]
    mov rcx, rdx
    rep movsb
    mov QWORD PTR [r8+8], 2
    mov rax, rdx
    pop rdi
    pop rsi
    ret
copy_bytes ENDP
END
