"""Start the same EXE with the Windows loader and with explicit pe_map providers."""

import ctypes
import hashlib
from pathlib import Path
import subprocess
import sys

import pefile

from exe_entry_check import Observation, check
from mapped_exe_chain_check import run_mapped
from native_map_check import Mapper, STATUS


COMPANION = "cpu_shim_startup_guest.dll"
SHIM_IMPORTS = {
    "xmem.dll": {"XMemAlloc", "XMemFree", "XMemVirtualQuery", "XMemGetWorkingSetStatistics"},
    "d3d12_xs.dll": {"D3D12SerializeVersionedRootSignature", "D3D12CreateRootSignatureDeserializer"},
}
EXPECTED = Observation(
    b"",
    b"companion attached\nxmem allocation, query and free\nxmem working set unsupported\n"
    b"d3d12_xs root signature round trip\n",
    42,
)


def imports_by_dll(image):
    with pefile.PE(str(image), fast_load=True) as pe:
        pe.parse_data_directories(directories=[1])
        return {module.dll.decode().casefold(): [imp.name.decode() if imp.name else imp.ordinal
                                                   for imp in module.imports]
                for module in pe.DIRECTORY_ENTRY_IMPORT}


def mapped(mapper_path, exe, companion, io, xmem, d3d12_xs, *, withheld):
    mapper = Mapper(mapper_path)
    providers = {"kernel32.dll": ctypes.WinDLL("kernel32.dll")}
    providers.update({dll.name.casefold(): ctypes.WinDLL(str(dll))
                      for dll in (xmem, d3d12_xs) if dll.name.casefold() != withheld})

    base, status, detail, requests = mapper(companion, providers)
    try:
        if withheld:
            first = imports_by_dll(companion)[withheld][0]
            check("missing shim mapping", (base, status), (None, STATUS["unresolved"]))
            dll, symbol = requests[-1]
            check("missing shim request", (dll.casefold(), symbol), (withheld, first))
            check("missing shim diagnostic", detail, f"unresolved import {dll}!{symbol}")
            print(detail, file=sys.stderr)
            return 0

        if not base or status != STATUS["ok"]:
            raise AssertionError(f"companion mapping: {status} {detail}; requests={requests!r}")
        expected_imports = {(dll, symbol) for dll, symbols in SHIM_IMPORTS.items() for symbol in symbols}
        actual_imports = [(dll.casefold(), symbol) for dll, symbol in requests if dll.casefold() in SHIM_IMPORTS]
        check("six shim imports", (len(actual_imports), set(actual_imports)), (6, expected_imports))
        for dll, symbol in actual_imports:
            print(f"bound {dll}!{symbol}", file=sys.stderr)

        entry = mapper.export(base, b"run_startup_shims")
        if not entry:
            raise AssertionError("mapped companion entry export missing")
        code = run_mapped(mapper_path, exe, {
            COMPANION: {b"run_startup_shims": entry},
            "cpu_shim_compute.dll": ctypes.WinDLL(str(exe.with_name("cpu_shim_compute.dll"))),
            "cpu_shim_draw.dll": ctypes.WinDLL(str(exe.with_name("cpu_shim_draw.dll"))),
            d3d12_xs.name.casefold(): providers[d3d12_xs.name.casefold()],
            io.name.casefold(): ctypes.WinDLL(str(io)),
        })
        print("mapped host returned", file=sys.stderr)
        return code
    finally:
        if base:
            check("companion released", mapper.release(base), 1)


def compare(mapper_path, exe, companion, io, xmem, d3d12_xs):
    check("native DLL location", {p.parent for p in (companion, io, xmem, d3d12_xs)}, {exe.parent})
    for image in (exe, companion, xmem, d3d12_xs):
        print(f"{image.name} sha256={hashlib.sha256(image.read_bytes()).hexdigest()}")
    check("EXE companion import", imports_by_dll(exe)[COMPANION], ["run_startup_shims"])
    check("EXE device import", imports_by_dll(exe)["d3d12_xs.dll"], ["D3D12XboxCreateDevice"])
    actual = imports_by_dll(companion)
    for dll, symbols in SHIM_IMPORTS.items():
        check(dll, set(actual[dll]), symbols)

    native = subprocess.run([exe], capture_output=True, timeout=20)
    check("native startup", Observation(native.stderr, native.stdout, native.returncode), EXPECTED)
    print(f"native: {native.stdout.decode().strip().replace(chr(10), ' / ')}; exit {native.returncode}")

    command = [sys.executable, "-B", __file__, "--map", mapper_path, exe, companion, io, xmem, d3d12_xs]
    child = subprocess.run([*command, ""], capture_output=True, timeout=20)
    markers = child.stderr.decode().splitlines()
    check("mapped startup", Observation(b"", child.stdout, child.returncode), EXPECTED)
    check("mapped return marker", markers[-1:], ["mapped host returned"])
    expected_markers = {f"bound {dll}!{name}" for dll, names in SHIM_IMPORTS.items() for name in names}
    check("mapped providers", (len(markers) - 1, set(markers[:-1])), (6, expected_markers))
    print(f"mapped: matching startup output and exit {child.returncode}")
    print("\n".join(markers))

    for dll in SHIM_IMPORTS:
        failure = subprocess.run([*command, dll], capture_output=True, timeout=20)
        expected_error = f"unresolved import {dll}!{actual[dll][0]}"
        check(f"withheld {dll}", (failure.returncode, failure.stdout, failure.stderr.decode().strip()),
              (0, b"", expected_error))
        print(f"before entry: {expected_error}")


if __name__ == "__main__":
    if sys.argv[1] == "--map":
        sys.exit(mapped(*map(Path, sys.argv[2:-1]), withheld=sys.argv[-1]))
    compare(*map(Path, sys.argv[1:]))
