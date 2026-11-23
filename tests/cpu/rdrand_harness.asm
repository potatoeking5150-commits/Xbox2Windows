; Native-only harness for tests/cpu/rdrand.py; it is never translated.
;
; has_rdrand returns CPUID.01H:ECX bit 30. has_rdseed returns CPUID.(EAX=07H,ECX=0):EBX bit 18 when CPUID 0
; reports leaf 7, else 0. rdrand_rflags and rdseed_rflags take a selector in ECX. Selector 0 adds 1 to -1, which
; sets ZF, CF, PF and AF; any other selector adds 1 to 7FFFFFFFFFFFFFFFh, which sets SF, OF, PF and AF. Then
; they execute the 64-bit instruction and return RFLAGS, so the check can confirm on the host which flags the
; instruction clears. Call them only after the matching CPUID check.
.code

has_rdrand PROC FRAME
    push rbx
    .pushreg rbx
    .endprolog
    mov eax, 1
    cpuid
    mov eax, ecx
    shr eax, 30
    and eax, 1
    pop rbx
    ret
has_rdrand ENDP

has_rdseed PROC FRAME
    push rbx
    .pushreg rbx
    .endprolog
    xor eax, eax
    cpuid
    cmp eax, 7
    jb short none
    mov eax, 7
    xor ecx, ecx
    cpuid
    mov eax, ebx
    shr eax, 18
    and eax, 1
    pop rbx
    ret
none:
    xor eax, eax
    pop rbx
    ret
has_rdseed ENDP

PRESET_FLAGS MACRO
    LOCAL second, done
    test ecx, ecx
    jnz short second
    mov r10, -1
    add r10, 1
    jmp short done
second:
    mov r10, 7fffffffffffffffh
    add r10, 1
done:
ENDM

rdrand_rflags PROC FRAME
    .endprolog
    PRESET_FLAGS
    rdrand rax
    pushfq
    pop rax
    ret
rdrand_rflags ENDP

rdseed_rflags PROC FRAME
    .endprolog
    PRESET_FLAGS
    rdseed rax
    pushfq
    pop rax
    ret
rdseed_rflags ENDP

END
