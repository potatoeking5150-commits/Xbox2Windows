; #511's companion still supplies the baseline checks. With a compute request,
; this same EXE owns the allocation branch, dispatch arguments and cleanup.
EXTERN __imp_run_startup_shims:QWORD
EXTERN __imp_compute_request:QWORD
EXTERN __imp_compute_alloc:QWORD
EXTERN __imp_compute_dispatch:QWORD
EXTERN __imp_compute_free:QWORD
EXTERN __imp_draw_request:QWORD
EXTERN __imp_draw_frame:QWORD
EXTERN __imp_shadow_device_request:QWORD
EXTERN __imp_D3D12XboxCreateDevice:QWORD
EXTERN __imp_shadow_open:QWORD
EXTERN __imp_shadow_record:QWORD
EXTERN __imp_shadow_submit:QWORD
EXTERN __imp_shadow_wait:QWORD
EXTERN __imp_shadow_close:QWORD

.code
; The EXE owns the frame index and map slot. Each frame has one submission
; and wait. A rejected submission returns before the wait or any frame output.
shadow_frame PROC FRAME
    sub rsp, 56
    .allocstack 56
    .endprolog
    mov [rsp+32], rcx
    mov [rsp+40], r9d
    call qword ptr [__imp_shadow_record]
    test eax, eax
    jnz frame_done
    mov rcx, [rsp+32]
    mov edx, [rsp+40]
    call qword ptr [__imp_shadow_submit]
    test eax, eax
    jnz frame_done
    mov rcx, [rsp+32]
    call qword ptr [__imp_shadow_wait]
frame_done:
    add rsp, 56
    ret
shadow_frame ENDP

native_entry PROC FRAME
    sub rsp, 56
    .allocstack 56
    .endprolog
    call qword ptr [__imp_run_startup_shims]
    cmp eax, 42
    jne done
    call qword ptr [__imp_compute_request]
    test rax, rax
    jz draw_phase
    mov [rsp+32], rax
    mov rcx, rax
    mov edx, 0a8004000h
    call qword ptr [__imp_compute_alloc]
    test rax, rax
    jz no_memory
    mov [rsp+40], rax
    mov rcx, rax
    mov edx, dword ptr [rsp+32]
    xor edx, 012345678h
    mov r8d, 4
    call qword ptr [__imp_compute_dispatch]
    mov [rsp+48], eax
    mov rcx, [rsp+40]
    mov edx, 0a8004000h
    call qword ptr [__imp_compute_free]
    test eax, eax
    jz bad_release
    cmp dword ptr [rsp+48], 0
    je bad_dispatch
draw_phase:
    call qword ptr [__imp_draw_request]
    test eax, eax
    jz success
    cmp eax, 20
    jae draw_upload
    cmp eax, 14
    jae draw_shadow
    cmp eax, 10
    jae draw_texture
    cmp eax, 6
    jae draw_two
draw_single:
    mov ecx, eax
    call qword ptr [__imp_draw_frame]
    test eax, eax
    jnz bad_draw
    jmp success
draw_two:
    mov dword ptr [rsp+48], eax
    mov ecx, 6
    call qword ptr [__imp_draw_frame]
    test eax, eax
    jnz bad_draw
    mov ecx, dword ptr [rsp+48]
    add ecx, 1
    call qword ptr [__imp_draw_frame]
    cmp eax, -1
    je bad_second_draw
    test eax, eax
    jnz bad_draw
    jmp success
draw_upload:
    mov ecx, eax
    call qword ptr [__imp_draw_frame]
    test eax, eax
    jnz bad_draw
    jmp success
draw_shadow:
    mov [rsp+32], eax
    mov ecx, eax
    call qword ptr [__imp_shadow_device_request]
    test rax, rax
    jz bad_draw
    mov [rsp+40], rax
    mov rcx, [rax]
    xor edx, edx
    mov r8, [rax+8]
    lea r9, [rax+16]
    call qword ptr [__imp_D3D12XboxCreateDevice]
    mov edx, eax
    mov rcx, [rsp+40]
    call qword ptr [__imp_shadow_open]
    cmp eax, 0887a0004h
    je shadow_unsupported
    test eax, eax
    jnz bad_shadow
    mov rcx, [rsp+40]
    xor edx, edx
    xor r8d, r8d
    xor r9d, r9d
    call shadow_frame
    test eax, eax
    jnz bad_shadow
    mov rcx, [rsp+40]
    mov edx, 1
    mov r8d, 1
    mov r9d, 1
    cmp dword ptr [rsp+32], 15
    jne shadow_submit_index
    xor r8d, r8d
shadow_submit_index:
    cmp dword ptr [rsp+32], 17
    jne shadow_second
    mov r9d, 2
shadow_second:
    call shadow_frame
    cmp eax, 080070057h
    je bad_shadow_second
    test eax, eax
    jnz bad_shadow
    mov eax, 42
    jmp shadow_cleanup
shadow_unsupported:
    mov eax, 77
    jmp shadow_cleanup
bad_shadow_second:
    mov eax, 24
    jmp shadow_cleanup
bad_shadow:
    mov eax, 20
shadow_cleanup:
    mov [rsp+48], eax
    mov rcx, [rsp+40]
    call qword ptr [__imp_shadow_close]
    test eax, eax
    jnz bad_release
    mov eax, [rsp+48]
    jmp done
draw_texture:
    mov dword ptr [rsp+48], eax
    mov ecx, 10
    call qword ptr [__imp_draw_frame]
    test eax, eax
    jnz bad_draw
    mov ecx, dword ptr [rsp+48]
    add ecx, 1
    call qword ptr [__imp_draw_frame]
    test eax, eax
    jnz bad_draw
    mov ecx, 12
    call qword ptr [__imp_draw_frame]
    test eax, eax
    jnz bad_draw
success:
    mov eax, 42
    jmp done
no_memory:
    mov eax, 17
    jmp done
bad_release:
    mov eax, 18
    jmp done
bad_dispatch:
    mov eax, 19
    jmp done
bad_draw:
    mov eax, 20
    jmp done
bad_second_draw:
    mov eax, 24
done:
    add rsp, 56
    ret
native_entry ENDP
END
