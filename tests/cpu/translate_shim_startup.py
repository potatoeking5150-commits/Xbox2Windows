"""Lift the linked #511 EXE from its PE entry, binding only the fixture providers."""
import hashlib
from pathlib import Path
import sys

import pefile
from iced_x86 import Mnemonic as M

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_switch_aot import HostImport, PRELUDE, Program


COMPUTE = "cpu_shim_compute.dll"
DRAW = "cpu_shim_draw.dll"
IMPORTS = {
    ("cpu_shim_startup_guest.dll", b"run_startup_shims"): HostImport("run_startup_shims", (), 4),
    ("cpu_exe_io.dll", b"finish_exe"): HostImport("host_finish", (8, 4), 4),
    (COMPUTE, b"compute_request"): HostImport("compute_request", (), 8),
    (COMPUTE, b"compute_alloc"): HostImport("host_alloc", (8, 8), 8),
    (COMPUTE, b"compute_dispatch"): HostImport("host_dispatch", (8, 4, 4), 4),
    (COMPUTE, b"compute_free"): HostImport("host_free", (8, 8), 4),
    (DRAW, b"draw_request"): HostImport("draw_request", (), 4),
    (DRAW, b"draw_frame"): HostImport("host_draw_frame", (4,), 4),
    (DRAW, b"shadow_device_request"): HostImport("host_shadow_device_request", (4,), 8),
    ("d3d12_xs.dll", b"D3D12XboxCreateDevice"): HostImport("host_xbox_create_device", (8, 8, 8, 8), 4),
    (DRAW, b"shadow_open"): HostImport("host_shadow_open", (8, 4), 4),
    (DRAW, b"shadow_record"): HostImport("host_shadow_record", (8, 4, 4), 4),
    (DRAW, b"shadow_submit"): HostImport("host_shadow_submit", (8, 4), 4),
    (DRAW, b"shadow_wait"): HostImport("host_shadow_wait", (8,), 4),
    (DRAW, b"shadow_close"): HostImport("host_shadow_close", (8,), 4),
}

HOST = r"""
#include <windows.h>
#include <unknwn.h>
extern "C" __declspec(dllimport) DWORD run_startup_shims();
extern "C" __declspec(dllimport) DWORD finish_exe(const void*, DWORD);
extern "C" __declspec(dllimport) SIZE_T compute_request();
extern "C" __declspec(dllimport) void* compute_alloc(SIZE_T, ULONGLONG);
extern "C" __declspec(dllimport) DWORD compute_dispatch(void*, unsigned, unsigned);
extern "C" __declspec(dllimport) DWORD compute_free(void*, ULONGLONG);
extern "C" __declspec(dllimport) DWORD draw_request();
extern "C" __declspec(dllimport) DWORD draw_frame(DWORD);
extern "C" __declspec(dllimport) void* shadow_device_request(DWORD);
extern "C" __declspec(dllimport) HRESULT WINAPI D3D12XboxCreateDevice(IUnknown*, const void*, REFIID, void**);
extern "C" __declspec(dllimport) HRESULT shadow_open(void*, HRESULT);
extern "C" __declspec(dllimport) HRESULT shadow_record(void*, UINT, UINT);
extern "C" __declspec(dllimport) HRESULT shadow_submit(void*, UINT);
extern "C" __declspec(dllimport) HRESULT shadow_wait(void*);
extern "C" __declspec(dllimport) HRESULT shadow_close(void*);

static word bound_image() { return reinterpret_cast<word>(GetModuleHandleW(nullptr)); }
static word host_alloc(word size, word attr) { return reinterpret_cast<word>(compute_alloc(size, attr)); }
static word host_dispatch(word addr, word seed, word count) {
    return compute_dispatch(reinterpret_cast<void*>(addr), static_cast<unsigned>(seed), static_cast<unsigned>(count));
}
static word host_free(word addr, word attr) { return compute_free(reinterpret_cast<void*>(addr), attr); }
static word host_draw_frame(word mode) { return draw_frame(static_cast<DWORD>(mode)); }
static word host_shadow_device_request(word mode) {
    return reinterpret_cast<word>(shadow_device_request(static_cast<DWORD>(mode)));
}
static word host_xbox_create_device(word adapter, word parameters, word iid, word output) {
    return static_cast<DWORD>(D3D12XboxCreateDevice(reinterpret_cast<IUnknown*>(adapter),
        reinterpret_cast<const void*>(parameters), *reinterpret_cast<const IID*>(iid), reinterpret_cast<void**>(output)));
}
static word host_shadow_open(word request, word result) {
    return static_cast<DWORD>(shadow_open(reinterpret_cast<void*>(request), static_cast<HRESULT>(result)));
}
static word host_shadow_record(word request, word frame, word map) {
    return static_cast<DWORD>(shadow_record(reinterpret_cast<void*>(request), static_cast<UINT>(frame),
                                          static_cast<UINT>(map)));
}
static word host_shadow_submit(word request, word frame) {
    return static_cast<DWORD>(shadow_submit(reinterpret_cast<void*>(request), static_cast<UINT>(frame)));
}
static word host_shadow_wait(word request) {
    return static_cast<DWORD>(shadow_wait(reinterpret_cast<void*>(request)));
}
static word host_shadow_close(word request) {
    return static_cast<DWORD>(shadow_close(reinterpret_cast<void*>(request)));
}
static word host_finish(word base, word code) { return finish_exe(reinterpret_cast<void*>(base), static_cast<DWORD>(code)); }
"""


def emit(image, output):
    data = image.read_bytes()
    with pefile.PE(data=data, fast_load=True) as pe:
        opt = pe.OPTIONAL_HEADER
        if (pe.FILE_HEADER.Machine != 0x8664 or opt.Magic != 0x20b or opt.Subsystem != 3
                or pe.FILE_HEADER.Characteristics & 0x2000 or opt.DATA_DIRECTORY[9].VirtualAddress):
            raise ValueError("expected the CRT-free AMD64 console fixture without EXE TLS")
        pe.parse_data_directories(directories=[1])
        imports = {}
        for module in pe.DIRECTORY_ENTRY_IMPORT:
            for imp in module.imports:
                key = module.dll.decode("ascii").lower(), imp.name
                if key not in IMPORTS:
                    raise ValueError(f"unbound shim fixture import {key}")
                imports[imp.address - opt.ImageBase] = IMPORTS[key]
        program = Program(pe, imports=imports, image_data=True)
        entry = opt.AddressOfEntryPoint
        if entry not in program.entries:
            raise ValueError("unverified process entry")
        source, root = program.lift(entry, "process_entry", ())
        functions = [root, *program.callees.values()]
        used = {call for fn in functions for call in fn.host_calls.values()}
        if root.return_width is None or root.return_width < 4 or used != set(IMPORTS.values()):
            raise ValueError("incomplete shim startup import path")
        # Only the identity argument to finish_exe is relocated to the AOT process.
        # No original image bytes are mapped in the generated executable.
        for fn in functions:
            for ip, ins in fn.instructions.items():
                if ins.is_ip_rel_memory_operand and ip not in fn.host_calls:
                    if ins.mnemonic != M.LEA or ins.ip_rel_memory_address != 0:
                        raise ValueError(f"unsupported image access at {ip:#x}")
    digest = hashlib.sha256(data).hexdigest()
    output.write_text("\n".join([PRELUDE, HOST, *program.sources, source,
                                f'extern "C" __declspec(dllexport) const char aot_source_sha256[] = "{digest}";',
                                "int main() { return static_cast<int>(process_entry()); }", ""]), encoding="ascii")


if __name__ == "__main__":
    emit(Path(sys.argv[1]), Path(sys.argv[2]))
