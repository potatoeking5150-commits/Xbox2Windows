; coverage: source cpu_aot_signed_conds
; Source-owned differential fixture for signed and flag-conditional operations:
; CMOVcc (cmovl, cmovge, cmovle, cmovg, cmovs, cmovns, cmovo, cmovno in 32-bit and 64-bit),
; SETcc (setl, setge, setle, setg, sets, setns, seto, setno),
; Jcc (jl, jge, jle, jg, js, jns, jo, jno),
; NEG (32-bit and 64-bit), and CWD.
; 32-bit CMOVcc initializes RAX with upper-half noise to verify unconditional zero-extension;
; SETcc initializes RAX with 0x123456789abcdef0 to verify upper 56 bits are preserved.
.code

; --- CMOVcc 32-bit ---

cmovl32 PROC FRAME
    .endprolog
    mov rax, 0deadbeef00000011h
    cmp ecx, edx
    cmovl eax, r8d
    ret
cmovl32 ENDP

cmovge32 PROC FRAME
    .endprolog
    mov rax, 0deadbeef00000011h
    cmp ecx, edx
    cmovge eax, r8d
    ret
cmovge32 ENDP

cmovle32 PROC FRAME
    .endprolog
    mov rax, 0deadbeef00000011h
    cmp ecx, edx
    cmovle eax, r8d
    ret
cmovle32 ENDP

cmovg32 PROC FRAME
    .endprolog
    mov rax, 0deadbeef00000011h
    cmp ecx, edx
    cmovg eax, r8d
    ret
cmovg32 ENDP

cmovs32 PROC FRAME
    .endprolog
    mov rax, 0deadbeef00000011h
    cmp ecx, edx
    cmovs eax, r8d
    ret
cmovs32 ENDP

cmovns32 PROC FRAME
    .endprolog
    mov rax, 0deadbeef00000011h
    cmp ecx, edx
    cmovns eax, r8d
    ret
cmovns32 ENDP

cmovo32 PROC FRAME
    .endprolog
    mov rax, 0deadbeef00000011h
    cmp ecx, edx
    cmovo eax, r8d
    ret
cmovo32 ENDP

cmovno32 PROC FRAME
    .endprolog
    mov rax, 0deadbeef00000011h
    cmp ecx, edx
    cmovno eax, r8d
    ret
cmovno32 ENDP

; --- CMOVcc 64-bit ---

cmovl64 PROC FRAME
    .endprolog
    mov rax, 01122334455667788h
    cmp rcx, rdx
    cmovl rax, r8
    ret
cmovl64 ENDP

cmovge64 PROC FRAME
    .endprolog
    mov rax, 01122334455667788h
    cmp rcx, rdx
    cmovge rax, r8
    ret
cmovge64 ENDP

cmovle64 PROC FRAME
    .endprolog
    mov rax, 01122334455667788h
    cmp rcx, rdx
    cmovle rax, r8
    ret
cmovle64 ENDP

cmovg64 PROC FRAME
    .endprolog
    mov rax, 01122334455667788h
    cmp rcx, rdx
    cmovg rax, r8
    ret
cmovg64 ENDP

cmovs64 PROC FRAME
    .endprolog
    mov rax, 01122334455667788h
    cmp rcx, rdx
    cmovs rax, r8
    ret
cmovs64 ENDP

cmovns64 PROC FRAME
    .endprolog
    mov rax, 01122334455667788h
    cmp rcx, rdx
    cmovns rax, r8
    ret
cmovns64 ENDP

cmovo64 PROC FRAME
    .endprolog
    mov rax, 01122334455667788h
    cmp rcx, rdx
    cmovo rax, r8
    ret
cmovo64 ENDP

cmovno64 PROC FRAME
    .endprolog
    mov rax, 01122334455667788h
    cmp rcx, rdx
    cmovno rax, r8
    ret
cmovno64 ENDP

; --- SETcc ---

setl_op PROC FRAME
    .endprolog
    mov rax, 0123456789abcdef0h
    cmp rcx, rdx
    setl al
    ret
setl_op ENDP

setge_op PROC FRAME
    .endprolog
    mov rax, 0123456789abcdef0h
    cmp rcx, rdx
    setge al
    ret
setge_op ENDP

setle_op PROC FRAME
    .endprolog
    mov rax, 0123456789abcdef0h
    cmp rcx, rdx
    setle al
    ret
setle_op ENDP

setg_op PROC FRAME
    .endprolog
    mov rax, 0123456789abcdef0h
    cmp rcx, rdx
    setg al
    ret
setg_op ENDP

sets_op PROC FRAME
    .endprolog
    mov rax, 0123456789abcdef0h
    cmp rcx, rdx
    sets al
    ret
sets_op ENDP

setns_op PROC FRAME
    .endprolog
    mov rax, 0123456789abcdef0h
    cmp rcx, rdx
    setns al
    ret
setns_op ENDP

seto_op PROC FRAME
    .endprolog
    mov rax, 0123456789abcdef0h
    cmp rcx, rdx
    seto al
    ret
seto_op ENDP

setno_op PROC FRAME
    .endprolog
    mov rax, 0123456789abcdef0h
    cmp rcx, rdx
    setno al
    ret
setno_op ENDP

; --- Jcc branches ---

jl_branch PROC FRAME
    .endprolog
    cmp rcx, rdx
    jl short jl_taken
    mov eax, 0
    ret
jl_taken:
    mov eax, 1
    ret
jl_branch ENDP

jge_branch PROC FRAME
    .endprolog
    cmp rcx, rdx
    jge short jge_taken
    mov eax, 0
    ret
jge_taken:
    mov eax, 1
    ret
jge_branch ENDP

jle_branch PROC FRAME
    .endprolog
    cmp rcx, rdx
    jle short jle_taken
    mov eax, 0
    ret
jle_taken:
    mov eax, 1
    ret
jle_branch ENDP

jg_branch PROC FRAME
    .endprolog
    cmp rcx, rdx
    jg short jg_taken
    mov eax, 0
    ret
jg_taken:
    mov eax, 1
    ret
jg_branch ENDP

js_branch PROC FRAME
    .endprolog
    cmp rcx, rdx
    js short js_taken
    mov eax, 0
    ret
js_taken:
    mov eax, 1
    ret
js_branch ENDP

jns_branch PROC FRAME
    .endprolog
    cmp rcx, rdx
    jns short jns_taken
    mov eax, 0
    ret
jns_taken:
    mov eax, 1
    ret
jns_branch ENDP

jo_branch PROC FRAME
    .endprolog
    cmp rcx, rdx
    jo short jo_taken
    mov eax, 0
    ret
jo_taken:
    mov eax, 1
    ret
jo_branch ENDP

jno_branch PROC FRAME
    .endprolog
    cmp rcx, rdx
    jno short jno_taken
    mov eax, 0
    ret
jno_taken:
    mov eax, 1
    ret
jno_branch ENDP

; --- NEG ---

neg32 PROC FRAME
    .endprolog
    mov eax, ecx
    neg eax
    setb r8b
    seto r9b
    sets r10b
    sete r11b
    movzx r8d, r8b
    movzx r9d, r9b
    movzx r10d, r10b
    movzx r11d, r11b
    shl r9d, 1
    shl r10d, 2
    shl r11d, 3
    or r8d, r9d
    or r8d, r10d
    or r8d, r11d
    mov qword ptr [rdx], r8
    ret
neg32 ENDP

neg64 PROC FRAME
    .endprolog
    mov rax, rcx
    neg rax
    setb r8b
    seto r9b
    sets r10b
    sete r11b
    movzx r8d, r8b
    movzx r9d, r9b
    movzx r10d, r10b
    movzx r11d, r11b
    shl r9d, 1
    shl r10d, 2
    shl r11d, 3
    or r8d, r9d
    or r8d, r10d
    or r8d, r11d
    mov qword ptr [rdx], r8
    ret
neg64 ENDP

; --- CWD ---

cwd_op PROC FRAME
    .endprolog
    xor edx, edx
    mov rax, rcx
    cwd
    movzx eax, ax
    movzx edx, dx
    shl edx, 16
    or eax, edx
    ret
cwd_op ENDP

; --- Mixed-producer SF regression cases ---

xadd_sets PROC FRAME
    .endprolog
    xor eax, eax
    mov rax, rcx
    xadd rax, rdx
    sets al
    movzx eax, al
    ret
xadd_sets ENDP

xadd_jl PROC FRAME
    .endprolog
    xor eax, eax
    mov rax, rcx
    xadd rax, rdx
    jl short xadd_jl_taken
    mov eax, 0
    ret
xadd_jl_taken:
    mov eax, 1
    ret
xadd_jl ENDP

cmpxchg_sets PROC FRAME
    .endprolog
    xor r9d, r9d
    mov rax, rcx
    mov r9, rdx
    cmpxchg r9, r8
    sets al
    movzx eax, al
    ret
cmpxchg_sets ENDP

shld_sets PROC FRAME
    .endprolog
    xor eax, eax
    mov rax, rcx
    mov rcx, r8
    shld rax, rdx, cl
    sets al
    movzx eax, al
    ret
shld_sets ENDP

shrd_sets PROC FRAME
    .endprolog
    xor eax, eax
    mov rax, rcx
    mov rcx, r8
    shrd rax, rdx, cl
    sets al
    movzx eax, al
    ret
shrd_sets ENDP

shld_c0_sets PROC FRAME
    .endprolog
    mov rax, -1
    test rax, rax
    mov rax, rcx
    mov rcx, 0
    shld rax, rdx, cl
    sets al
    movzx eax, al
    ret
shld_c0_sets ENDP

andn_sets PROC FRAME
    .endprolog
    xor eax, eax
    andn rax, rcx, rdx
    sets al
    movzx eax, al
    ret
andn_sets ENDP

blsi_sets PROC FRAME
    .endprolog
    xor eax, eax
    blsi rax, rcx
    sets al
    movzx eax, al
    ret
blsi_sets ENDP

bzhi_sets PROC FRAME
    .endprolog
    xor eax, eax
    bzhi rax, rcx, rdx
    sets al
    movzx eax, al
    ret
bzhi_sets ENDP

popcnt_sets PROC FRAME
    .endprolog
    mov rax, -1
    test rax, rax
    popcnt rax, rcx
    sets al
    movzx eax, al
    ret
popcnt_sets ENDP

END
