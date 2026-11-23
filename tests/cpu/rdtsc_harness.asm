; Native-only harness for tests/cpu/rdtsc.py; it is never translated.
;
; serial_tsc returns one time-stamp read between two LFENCEs, so the measurement boundary cannot move across
; the translated read it brackets. has_rdtscp returns 1 when CPUID 80000000h reports leaf 80000001h and that
; leaf's EDX bit 27 (RDTSCP) is set.
.code

serial_tsc PROC FRAME
    .endprolog
    lfence
    rdtsc
    lfence
    shl rdx, 32
    or rax, rdx
    ret
serial_tsc ENDP

has_rdtscp PROC FRAME
    push rbx
    .pushreg rbx
    .endprolog
    mov eax, 80000000h
    cpuid
    cmp eax, 80000001h
    jb short none
    mov eax, 80000001h
    cpuid
    mov eax, edx
    shr eax, 27
    and eax, 1
    pop rbx
    ret
none:
    xor eax, eax
    pop rbx
    ret
has_rdtscp ENDP

END
