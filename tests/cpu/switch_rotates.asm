; coverage: source cpu_aot_switch_rotates
; Source-owned differential fixture and negative controls for ROL and ROR in tools/cpu_switch_aot.py.
; Every differential export takes RCX = x, RDX = y and R8 = address of three writable QWORDs.
.code

rot_reg_cl_8 PROC FRAME
    .endprolog
    mov rax, rdx
    rol al, cl
    mov QWORD PTR [r8], rax
    mov r9, rdx
    ror r9b, cl
    mov QWORD PTR [r8+8], r9
    mov r10, rdx
    rol r10b, cl
    ror r10b, cl
    mov QWORD PTR [r8+16], r10
    ret
rot_reg_cl_8 ENDP

rot_reg_cl_16 PROC FRAME
    .endprolog
    mov rax, rdx
    rol ax, cl
    mov QWORD PTR [r8], rax
    mov r9, rdx
    ror r9w, cl
    mov QWORD PTR [r8+8], r9
    mov r10, rdx
    rol r10w, cl
    ror r10w, cl
    mov QWORD PTR [r8+16], r10
    ret
rot_reg_cl_16 ENDP

rot_reg_cl_32 PROC FRAME
    .endprolog
    mov rax, rdx
    rol eax, cl
    mov QWORD PTR [r8], rax
    mov r9, rdx
    ror r9d, cl
    mov QWORD PTR [r8+8], r9
    mov r10, rdx
    rol r10d, cl
    ror r10d, cl
    mov QWORD PTR [r8+16], r10
    ret
rot_reg_cl_32 ENDP

rot_reg_cl_64 PROC FRAME
    .endprolog
    mov rax, rdx
    rol rax, cl
    mov QWORD PTR [r8], rax
    mov r9, rdx
    ror r9, cl
    mov QWORD PTR [r8+8], r9
    mov r10, rdx
    rol r10, cl
    ror r10, cl
    mov QWORD PTR [r8+16], r10
    ret
rot_reg_cl_64 ENDP

rot_imm_1 PROC FRAME
    .endprolog
    mov rax, rcx
    rol al, 1
    ror ax, 1
    mov QWORD PTR [r8], rax
    mov r9, rcx
    rol r9d, 1
    mov QWORD PTR [r8+8], r9
    mov r10, rdx
    ror r10, 1
    mov QWORD PTR [r8+16], r10
    mov rax, rdx
    rol rax, 1
    ret
rot_imm_1 ENDP

rot_imm_edge PROC FRAME
    .endprolog
    mov rax, rcx
    rol al, 0
    mov r9, rcx
    ror r9b, 11
    mov QWORD PTR [r8], r9
    mov r10, rdx
    rol r10w, 19
    ror r10w, 0
    mov QWORD PTR [r8+8], r10
    mov r11, rdx
    rol r11d, 0
    ror r11d, 37
    mov QWORD PTR [r8+16], r11
    rol rax, 69
    ror rax, 0
    ret
rot_imm_edge ENDP

rot_mem_cl_8_16 PROC FRAME
    .endprolog
    mov QWORD PTR [r8], rdx
    mov QWORD PTR [r8+8], rdx
    mov QWORD PTR [r8+16], rdx
    rol BYTE PTR [r8], cl
    ror BYTE PTR [r8+8], 11
    rol WORD PTR [r8+16], cl
    mov rax, rdx
    ror ax, 1
    ret
rot_mem_cl_8_16 ENDP

rot_mem_cl_32_64 PROC FRAME
    .endprolog
    mov QWORD PTR [r8], rdx
    mov QWORD PTR [r8+8], rdx
    mov QWORD PTR [r8+16], rdx
    ror DWORD PTR [r8], cl
    rol DWORD PTR [r8+8], 37
    rol QWORD PTR [r8+16], cl
    ror QWORD PTR [r8+16], 1
    mov rax, QWORD PTR [r8+16]
    ret
rot_mem_cl_32_64 ENDP

rot_mem_imm PROC FRAME
    .endprolog
    mov QWORD PTR [r8], rdx
    mov QWORD PTR [r8+8], rdx
    mov QWORD PTR [r8+16], rdx
    rol BYTE PTR [r8], 1
    ror WORD PTR [r8+8], 19
    rol DWORD PTR [r8+16], 0
    mov rax, rcx
    ror rax, 13
    ret
rot_mem_imm ENDP

rot_narrow_ret PROC FRAME
    .endprolog
    mov al, dl
    rol al, cl
    ret
rot_narrow_ret ENDP

; --- Negative controls (untranslated by CMake; tested directly for rejection) ---

neg_rol_jcc PROC FRAME
    .endprolog
    rol ecx, 1
    jb taken
    xor eax, eax
    ret
taken:
    mov eax, 1
    ret
neg_rol_jcc ENDP

neg_ror_jcc PROC FRAME
    .endprolog
    ror ecx, 1
    jb taken
    xor eax, eax
    ret
taken:
    mov eax, 1
    ret
neg_ror_jcc ENDP

neg_rol_setcc PROC FRAME
    .endprolog
    rol ecx, cl
    setb al
    ret
neg_rol_setcc ENDP

neg_ror_setcc PROC FRAME
    .endprolog
    ror ecx, cl
    setb al
    ret
neg_ror_setcc ENDP

neg_rol_cmovcc PROC FRAME
    .endprolog
    mov eax, edx
    rol ecx, 1
    cmovb eax, r8d
    ret
neg_rol_cmovcc ENDP

neg_ror_cmovcc PROC FRAME
    .endprolog
    mov eax, edx
    ror ecx, 1
    cmovb eax, r8d
    ret
neg_ror_cmovcc ENDP

neg_cmp_rol_setcc PROC FRAME
    .endprolog
    cmp rcx, rdx
    rol r8d, 1
    setb al
    ret
neg_cmp_rol_setcc ENDP

neg_cmp_ror_setcc PROC FRAME
    .endprolog
    cmp rcx, rdx
    ror r8d, 1
    setb al
    ret
neg_cmp_ror_setcc ENDP

neg_rol_undef PROC FRAME
    .endprolog
    rol eax, 1
    ret
neg_rol_undef ENDP

neg_rol_ah PROC FRAME
    .endprolog
    mov rax, rcx
    rol ah, 1
    ret
neg_rol_ah ENDP

END
