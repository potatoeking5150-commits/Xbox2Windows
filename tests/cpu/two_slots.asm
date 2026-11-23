; coverage: source cpu_aot_two_slots
; Ordinary imports from two providers in one translation, each through its own IAT slot and with its own argument
; count: mix3(first, second, output) from tail_provider.cpp and fold4(first, second, output, fourth) from
; slot_provider.cpp, which returns 16 bits. R8 is the address of three writable QWORDs. Each export takes x in RCX
; and y in RDX. The register probe seeds RBX and RSI, so a nonvolatile register that the translation loses shows up
; as a changed sentinel.
EXTERN __imp_mix3:QWORD
EXTERN __imp_fold4:QWORD
EXTERN fold4:PROC
EXTERN __ImageBase:BYTE
.data
ordinary_data_slot QWORD 0
.code

; Selects by x AND 3 through an MSVC jump table, with x kept in RBX:
; 0: (mix3(x, y, trace) + 1) XOR x, an ordinary call through the slot of mix3;
; 1: fold4(x, y + 1, trace, x + y + 1), a tail jump through the slot of fold4 after the epilogue;
; 2: fold4(x, y, trace, x XOR y) + x, a direct call of the import thunk of fold4, reading only AX of its result;
; 3: x + 5, no import.
route2 PROC FRAME
    push rbx
    .pushreg rbx
    sub rsp, 20h
    .allocstack 20h
    .endprolog
    mov rbx, rcx
    mov eax, ecx
    and eax, 3
    cmp eax, 2
    ja route_default
    movsxd r9, eax
    lea r10, __ImageBase
    mov r9d, DWORD PTR [(IMAGEREL route_table) + r10 + r9*4]
    add r9, r10
    jmp r9
route_first:
    call QWORD PTR [__imp_mix3]
    add rax, 1
    xor rax, rbx
    jmp route_return
route_tail:
    add rdx, 1
    lea r9, [rcx+rdx]
    add rsp, 20h
    pop rbx
    jmp QWORD PTR [__imp_fold4]
route_thunk:
    mov r9, rcx
    xor r9, rdx
    call fold4
    movzx eax, ax
    add rax, rbx
    jmp route_return
route_default:
    lea rax, [rbx+5]
route_return:
    add rsp, 20h
    pop rbx
    ret
    ALIGN 4
route_table LABEL DWORD
    DD IMAGEREL route_first
    DD IMAGEREL route_tail
    DD IMAGEREL route_thunk
route2 ENDP

; fold4(mix3(x, y, trace), y, trace, y + 1) + y: both slots on one path. y is kept in RBX and the trace pointer in
; RSI; the trace holds the arguments of fold4.
call_both PROC FRAME
    push rbx
    .pushreg rbx
    push rsi
    .pushreg rsi
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    mov rbx, rdx
    mov rsi, r8
    call QWORD PTR [__imp_mix3]
    mov rcx, rax
    mov rdx, rbx
    mov r8, rsi
    lea r9, [rbx+1]
    call QWORD PTR [__imp_fold4]
    movzx eax, ax
    add rax, rbx
    add rsp, 28h
    pop rsi
    pop rbx
    ret
call_both ENDP

; fold4(x, y, trace, y - x) zero-extended from AX. Only the slot of fold4 is reached.
call_second PROC FRAME
    sub rsp, 28h
    .allocstack 28h
    .endprolog
    mov r9, rdx
    sub r9, rcx
    call QWORD PTR [__imp_fold4]
    movzx eax, ax
    add rsp, 28h
    ret
call_second ENDP

; Rejection control only: a RIP-relative jump through ordinary data, outside the IAT.
jump_non_import PROC FRAME
    .endprolog
    jmp QWORD PTR [ordinary_data_slot]
jump_non_import ENDP
END
