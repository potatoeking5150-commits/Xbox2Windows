"""Compare native and translated shuffle, unpacking, and vector move instructions."""
import ctypes
from pathlib import Path
import struct
import sys

import pefile
from iced_x86 import Decoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from ffx_version_query import INTEGER_HEADER, lower, region

U64 = ctypes.c_uint64

EXPORTS = (
    "shufps", "unpckhps", "unpcklps", "movhlps", "movlhps", "movmskps", "movups", "pmovmskb",
    "rcpps", "rcpss", "rsqrtps", "rsqrtss",
    "shufpd", "unpckhpd", "unpcklpd", "movmskpd", "movupd", "movapd", "movdqa",
    "pshufd", "pshufhw", "pshuflw", "pslldq", "psrldq", "punpckhqdq", "punpcklqdq",
    "vshufps", "vunpckhps", "vunpcklps", "vshufpd", "vunpckhpd", "vunpcklpd",
    "vpshufd", "vpshufhw", "vpshuflw", "vmovhlps", "vmovlhps", "vmovmskps", "vmovmskpd",
    "vmovaps", "vmovapd", "vmovdqa", "vpmovmskb", "vrcpps", "vrcpss", "vrsqrtps", "vrsqrtss",
    "vpslldq", "vpsrldq", "vpunpckhqdq", "vpunpcklqdq",
)
EXPORT_NAMES = tuple(f"{op}_op" for op in EXPORTS)


def emit(image, output):
    parts = [INTEGER_HEADER]
    with pefile.PE(str(image), fast_load=True) as pe:
        for name in EXPORT_NAMES:
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            code = lower(body, ["rcx = a; rdx = b; r8 = c;"], {}, {}, {})
            parts.append(f'extern "C" __declspec(dllexport) word {name}(word a, word b, word c) {{\n{code}\n}}\n')
    output.write_text("".join(parts), encoding="ascii")


def load_lib(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    for name in EXPORT_NAMES:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p], U64
    return lib


def aligned_buffer(data_bytes):
    raw = ctypes.create_string_buffer(len(data_bytes) + 32)
    addr = ctypes.addressof(raw)
    off = (16 - (addr % 16)) % 16
    ptr = addr + off
    ctypes.memmove(ptr, data_bytes, len(data_bytes))
    return raw, ptr


def check(native_path, lifted_path):
    # A legacy XMM load without VMOVD still needs an XMM declaration.
    old_load = list(Decoder(64, bytes.fromhex("c5 fa 6f 01 31 c0 c3")))
    source = lower(old_load, [], {}, {}, {})
    assert "__m128i xmm0" in source and "xmm0 = mem;" in source, source
    assert source.count("load_always<__m128i>(") == 1, source

    # Aligned memory moves require evidence that this profile does not have.
    for raw in ("66 0f 28 01 c3", "c5 f9 7f 01 c3"):
        try:
            lower(list(Decoder(64, bytes.fromhex(raw))), [], {}, {}, {})
        except ValueError as error:
            assert "unsupported" in str(error), error
        else:
            raise AssertionError(f"accepted aligned memory move {raw}")

    native, lifted = load_lib(native_path), load_lib(lifted_path)

    sentinel_a = b"\x5a" * 16
    sentinel_b = b"\xa5" * 16

    p1 = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0) + sentinel_a
    p2 = struct.pack("<4f", 10.0, 20.0, 30.0, 40.0) + sentinel_b
    p3 = struct.pack("<4I", 0x11223344, 0x55667788, 0x99aabbcc, 0xddeeff00) + sentinel_a
    p4 = struct.pack("<4I", 0x80808080, 0x01010101, 0x8f8f8f8f, 0x00000000) + sentinel_b

    p5 = struct.pack("<4I", 0x00000001, 0x80000001, 0x00000000, 0x80000000) + sentinel_a
    p6 = struct.pack("<4I", 0x80000001, 0x7f800000, 0xff800000, 0x7fc12345) + sentinel_b
    patterns = [(p1, p2), (p3, p4), (p1, p3), (p2, p4), (p5, p6), (p6, p5)]

    mask_ops = {"movmskps_op", "movmskpd_op", "pmovmskb_op",
                "vmovmskps_op", "vmovmskpd_op", "vpmovmskb_op"}

    comparisons = 0
    for a_bytes, b_bytes in patterns:
        raw_a, ptr_a = aligned_buffer(a_bytes)
        raw_b, ptr_b = aligned_buffer(b_bytes)
        raw_out_n, ptr_out_n = aligned_buffer(b"\xaa" * 32)
        raw_out_l, ptr_out_l = aligned_buffer(b"\x55" * 32)

        for name in EXPORT_NAMES:
            ctypes.memset(ptr_out_n, 0xaa, 32)
            ctypes.memset(ptr_out_l, 0x55, 32)
            getattr(native, name)(ptr_a, ptr_b, ptr_out_n)
            getattr(lifted, name)(ptr_a, ptr_b, ptr_out_l)

            if name in mask_ops:
                res_n = ctypes.string_at(ptr_out_n, 4)
                res_l = ctypes.string_at(ptr_out_l, 4)
                if res_n != res_l:
                    raise AssertionError(f"{name}: native mask {res_n.hex()} != lifted {res_l.hex()}")
            else:
                res_n = ctypes.string_at(ptr_out_n, 32)
                res_l = ctypes.string_at(ptr_out_l, 32)
                if res_n != res_l:
                    raise AssertionError(f"{name}: full 256-bit vector mismatch native {res_n.hex()} != lifted {res_l.hex()}")

                # Verify architectural high-half behavior: legacy XMM preserves, VEX.128 clears
                if name.startswith("v"):
                    if res_l[16:32] != b"\x00" * 16:
                        raise AssertionError(f"{name}: VEX.128 write did not clear high vector bits: {res_l[16:32].hex()}")
                else:
                    if res_l[16:32] != a_bytes[16:32]:
                        raise AssertionError(f"{name}: legacy XMM write did not preserve high vector bits: {res_l[16:32].hex()}")
            comparisons += 1

    # Aliased operand testing: both source pointers refer to the same buffer
    raw_aliased, ptr_aliased = aligned_buffer(p1)
    raw_out_n, ptr_out_n = aligned_buffer(b"\x00" * 32)
    raw_out_l, ptr_out_l = aligned_buffer(b"\x00" * 32)
    for name in EXPORT_NAMES:
        getattr(native, name)(ptr_aliased, ptr_aliased, ptr_out_n)
        getattr(lifted, name)(ptr_aliased, ptr_aliased, ptr_out_l)
        if name in mask_ops:
            if ctypes.string_at(ptr_out_n, 4) != ctypes.string_at(ptr_out_l, 4):
                raise AssertionError(f"{name}: aliased mask mismatch")
        else:
            if ctypes.string_at(ptr_out_n, 32) != ctypes.string_at(ptr_out_l, 32):
                raise AssertionError(f"{name}: aliased 256-bit vector mismatch")
        comparisons += 1

    print(f"All {comparisons} shuffle and vector move operations verified across full 256-bit vector state and aliased operands")


if __name__ == "__main__":
    mode, native, other = sys.argv[1:]
    if mode == "emit":
        emit(Path(native), Path(other))
    elif mode == "check":
        check(native, other)
    else:
        raise ValueError(mode)
