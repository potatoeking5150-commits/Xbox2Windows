"""Compare native and translated CMPXCHG8B and CMPXCHG16B without LOCK.

Each export loads RAX, RDX, RBX and RCX from a block, runs the instruction on an operand in the same
block and stores RAX, RDX, ZF and CF back. The cases cover equal operands, either half differing, upper
register halves that CMPXCHG8B must ignore or clear, and both CF values, which neither instruction
changes. A CMPXCHG16B operand that is not 16-byte aligned must fault with the same exception code in
both DLLs. A read-only operand must fault on the write cycle even on a compare mismatch.
"""
import ctypes
import mmap
from pathlib import Path
import struct
import sys

import pefile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from ffx_version_query import INTEGER_HEADER, lower, region

EXPORTS = ("cmpxchg8b_op", "cmpxchg16b_op")
M32, M64 = (1 << 32) - 1, (1 << 64) - 1
BLOCK = 80


def emit(image, output):
    parts = [INTEGER_HEADER]
    with pefile.PE(str(image), fast_load=True) as pe:
        for name in EXPORTS:
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            code = lower(region(pe, *bounds), ["rcx = block; rdx = offset; r8 = cf;"], {}, {}, {})
            parts.append(f'extern "C" __declspec(dllexport) word {name}(word block, word offset, word cf) {{\n{code}\n}}\n')
    output.write_text("".join(parts), encoding="ascii")


def load_lib(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    for name in EXPORTS:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64], ctypes.c_uint64
    return lib


def reference(name, memory, rax, rdx, rbx, rcx, cf):
    """(memory bytes after, RAX, RDX, ZF, CF) for an aligned operand at the start of memory."""
    if name == "cmpxchg8b_op":
        d, = struct.unpack_from("<Q", memory, 0)
        if d == ((rdx & M32) << 32) | (rax & M32):
            return struct.pack("<Q", ((rcx & M32) << 32) | (rbx & M32)), rax, rdx, 1, cf
        return memory, d & M32, d >> 32, 0, cf
    lo, hi = struct.unpack_from("<QQ", memory, 0)
    if (lo, hi) == (rax, rdx):
        return struct.pack("<QQ", rbx, rcx), rax, rdx, 1, cf
    return memory, lo, hi, 0, cf


def cases():
    a, b = 0x0123456789ABCDEF, 0xFEDCBA9876543210
    pairs = [(a, b), (0, 0), (M64, M64), (0x80000000, 0x7FFFFFFF), (0xDEADBEEF00000001, 0x00000002CAFEF00D)]
    for lo, hi in pairs:
        for name in EXPORTS:
            if name == "cmpxchg8b_op":
                d = lo
                compares = [(d & M32, d >> 32), ((d & M32) | 0x5555555500000000, (d >> 32) | 0xAAAAAAAA00000000),
                            (d & M32 ^ 1, d >> 32), (d & M32, (d >> 32) ^ 0x80000000)]
            else:
                compares = [(lo, hi), (lo ^ 1, hi), (lo, hi ^ (1 << 63)), (lo ^ M64, hi ^ M64)]
            for rax, rdx in compares:
                for cf in (0, 1):
                    for offset in ((0, 8) if name == "cmpxchg8b_op" else (0,)):
                        memory = struct.pack("<Q", lo) if name == "cmpxchg8b_op" else struct.pack("<QQ", lo, hi)
                        yield name, offset, memory, rax, rdx, 0x1111111122222222, 0x3333333344444444, cf


def run(lib, name, offset, memory, rax, rdx, rbx, rcx, cf):
    raw = (ctypes.c_uint8 * (BLOCK + 32))()
    base = (ctypes.addressof(raw) + 15) & ~15
    block = (ctypes.c_uint8 * BLOCK).from_address(base)
    ctypes.memset(base, 0xCC, BLOCK)
    block[offset:offset + len(memory)] = list(memory)
    struct.pack_into("<QQQQ", block, 16, rax, rdx, rbx, rcx)
    getattr(lib, name)(base, offset, cf)
    out_rax, out_rdx = struct.unpack_from("<QQ", block, 48)
    return bytes(block[offset:offset + len(memory)]), out_rax, out_rdx, block[64], block[65]


def check_readonly(native, lifted):
    protect = ctypes.WinDLL("kernel32", use_last_error=True).VirtualProtect
    protect.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32,
                        ctypes.POINTER(ctypes.c_uint32))
    protect.restype = ctypes.c_int
    page = mmap.PAGESIZE
    with mmap.mmap(-1, 2 * page) as block:
        base = ctypes.addressof(ctypes.c_char.from_buffer(block))
        target = base + page
        old = ctypes.c_uint32()
        if not protect(target, page, 0x02, ctypes.byref(old)):  # PAGE_READONLY
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            for name in EXPORTS:
                for compare in (0, 1):
                    struct.pack_into("<QQQQ", block, 16, compare, 0, 1, 2)
                    for label, lib in (("native", native), ("lifted", lifted)):
                        sentinels = b"\xcc" * 18
                        block[48:66] = sentinels
                        try:
                            getattr(lib, name)(base, page, 0)
                        except OSError as error:
                            prefix = "exception: access violation writing "
                            message = str(error)
                            if not message.startswith(prefix) or int(message[len(prefix):], 16) != target:
                                raise AssertionError(f"{label} {name} compare={compare}: {error}") from error
                        else:
                            raise AssertionError(f"{label} {name} compare={compare} did not fault")
                        if block[48:66] != sentinels:
                            raise AssertionError(f"{label} {name} wrote results after the fault")
        finally:
            if not protect(target, page, old.value, ctypes.byref(ctypes.c_uint32())):
                raise ctypes.WinError(ctypes.get_last_error())
    print("read-only CMPXCHG8B and CMPXCHG16B faulted on write for equal and unequal operands")


def check(native_path, lifted_path):
    native, lifted = load_lib(native_path), load_lib(lifted_path)
    count = 0
    for name, offset, memory, rax, rdx, rbx, rcx, cf in cases():
        mem, want_rax, want_rdx, zf, want_cf = reference(name, memory, rax, rdx, rbx, rcx, cf)
        for lib, label in ((native, "native"), (lifted, "lifted")):
            got = run(lib, name, offset, memory, rax, rdx, rbx, rcx, cf)
            expected = (mem, want_rax, want_rdx, zf, want_cf)
            if got != expected:
                raise AssertionError(f"{label} {name}(offset {offset}, memory {memory.hex()}, rax {rax:#x}, rdx {rdx:#x}, "
                                     f"cf {cf}) = {got}, expected {expected}")
        count += 1
    print(f"{count} CMPXCHG8B and CMPXCHG16B cases matched the reference in native and translated DLLs")
    # ctypes reports an unhandled exception as OSError("exception: ..."), naming an access violation with
    # its access type and address, so equal messages mean the same code and parameters.
    faults = {}
    for offset in (4, 8):
        for lib, label in ((native, "native"), (lifted, "lifted")):
            try:
                run(lib, "cmpxchg16b_op", offset, bytes(16), 0, 0, 1, 2, 0)
            except OSError as error:
                faults[(label, offset)] = str(error)
                continue
            raise AssertionError(f"{label} cmpxchg16b_op did not fault on an operand at offset {offset}")
        if faults[("native", offset)] != faults[("lifted", offset)]:
            raise AssertionError(f"offset {offset}: native {faults[('native', offset)]!r}, "
                                 f"lifted {faults[('lifted', offset)]!r}")
    print(f"a CMPXCHG16B operand at offset 4 or 8 raised {faults[('native', 4)]!r} in both DLLs")
    check_readonly(native, lifted)
    rejections()


def rejections():
    """CMPXCHG8B and CMPXCHG16B with LOCK stay rejected, like CMPXCHG."""
    from iced_x86 import Decoder
    for label, code in (("lock cmpxchg8b", "f00fc70ec3"), ("lock cmpxchg16b", "f0480fc70ec3")):
        try:
            lower(list(Decoder(64, bytes.fromhex(code))), [], {}, {}, {})
        except ValueError as error:
            if "prefix" not in str(error):
                raise AssertionError(f"{label}: unexpected rejection {error}")
            continue
        raise AssertionError(f"lower() accepted {label}")
    print("rejected lock cmpxchg8b and lock cmpxchg16b")


if __name__ == "__main__":
    mode, native, other = sys.argv[1:]
    if mode == "emit":
        emit(Path(native), Path(other))
    elif mode == "check":
        check(native, other)
    else:
        raise ValueError(mode)
