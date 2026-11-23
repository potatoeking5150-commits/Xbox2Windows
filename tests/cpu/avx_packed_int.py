"""Differential check for VEX AVX packed integer ALU lowering."""
import ctypes
import sys
from pathlib import Path

import pefile
from iced_x86 import Decoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from ffx_version_query import INTEGER_HEADER, lower, region

EXPORTS = (
    "vpaddb_op", "vpaddw_op", "vpaddq_op", "vpsubb_op", "vpsubw_op", "vpsubd_op", "vpsubq_op",
    "vpand_op", "vpandn_op", "vpor_op",
    "vpcmpeqb_op", "vpcmpeqw_op", "vpcmpeqd_op", "vpcmpeqq_op",
    "vpcmpgtb_op", "vpcmpgtw_op", "vpcmpgtd_op", "vpcmpgtq_op",
    "vpabsb_op", "vpabsw_op", "vpabsd_op",
    "vpsignb_op", "vpsignw_op", "vpsignd_op",
    "vpshufb_op", "vpalignr_op",
)

WIDE_EXPORT = "vpshufb256_mem"

CPUID_HELPER = """#include <intrin.h>
extern "C" __declspec(dllexport) uint32_t host_has_avx() {
    int info[4]{};
    __cpuid(info, 1);
    const int need = (1 << 27) | (1 << 28);
    if ((info[2] & need) != need) return 0u;
    if ((_xgetbv(0) & 6u) != 6u) return 0u;
    __cpuid(info, 0);
    if (info[0] < 7) return 1u;
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) ? 2u : 1u;
}
"""


def emit(native_dll: Path, output_cpp: Path) -> None:
    parts = [INTEGER_HEADER, CPUID_HELPER]
    with pefile.PE(str(native_dll), fast_load=True) as pe:
        for name in (*EXPORTS, WIDE_EXPORT):
            ranges = function_ranges(pe, export_rva(pe, name))
            body = [ins for lo, hi in ranges for ins in region(pe, lo, hi)]
            code = lower(body, ["rcx = a; rdx = b; r8 = dst;"], {}, {}, {})
            parts.append(
                f'extern "C" __declspec(dllexport) uint64_t {name}(uint64_t a, uint64_t b, uint64_t dst) {{\n{code}\n}}'
            )
    output_cpp.write_text("\n".join(parts) + "\n", encoding="utf-8")


def check_controls() -> None:
    # EVEX vpaddb xmm0{k1}, xmm1, xmm2 (62 F1 75 09 FC C2) followed by ret must be rejected.
    evex = list(Decoder(64, b"\x62\xf1\x75\x09\xfc\xc2\xc3", ip=0x1000))
    # VEX vpaddb xmm0, xmm1, xmm3 reaches the shared translator outside
    # its declared vector register set and must be rejected deliberately.
    untracked = list(Decoder(64, b"\xc5\xf1\xfc\xc3\xc3", ip=0x1000))
    for code in (evex, untracked):
        try:
            lower(code, [], {}, {}, {})
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected {code[0]} to be rejected")


def check(native_dll: Path, lifted_dll: Path) -> None:
    check_controls()
    lifted = ctypes.CDLL(str(lifted_dll))
    lifted.host_has_avx.restype = ctypes.c_uint32
    capability = lifted.host_has_avx()
    if not capability:
        print("host lacks AVX; skipping native differential check")
        raise SystemExit(77)
    native = ctypes.CDLL(str(native_dll))

    a_bytes = bytes([
        0x00, 0x80, 0x7f, 0xff, 0x00, 0x00, 0x80, 0x7f,
        0x11, 0x22, 0x33, 0x44, 0xfe, 0xdc, 0xba, 0x98,
    ]) + bytes([0xaa] * 16)
    b_bytes = bytes([
        0x00, 0x80, 0x01, 0xff, 0x00, 0xff, 0x80, 0x00,
        0x80, 0x0f, 0x02, 0x83, 0x00, 0x05, 0x0e, 0x81,
    ]) + bytes([0x55] * 16)
    a_buf = (ctypes.c_ubyte * 32)(*a_bytes)
    b_buf = (ctypes.c_ubyte * 32)(*b_bytes)
    out_n = (ctypes.c_ubyte * 32)()
    out_l = (ctypes.c_ubyte * 32)()
    for name in EXPORTS:
        fn_n, fn_l = getattr(native, name), getattr(lifted, name)
        fn_n.argtypes = fn_l.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn_n.restype = fn_l.restype = ctypes.c_uint64
        fn_n(ctypes.addressof(a_buf), ctypes.addressof(b_buf), ctypes.addressof(out_n))
        fn_l(ctypes.addressof(a_buf), ctypes.addressof(b_buf), ctypes.addressof(out_l))
        if bytes(out_n) != bytes(out_l):
            raise AssertionError(f"{name}: native={bytes(out_n).hex()} lifted={bytes(out_l).hex()}")
    print(f"{len(EXPORTS)} VEX.128 cases matched native and translated DLLs")
    if capability < 2:
        print("host lacks AVX2; VEX.256 shuffle not exercised")
        raise SystemExit(77)

    # Use unequal halves and controls that independently select bytes and zeros.
    a = (ctypes.c_ubyte * 32)(*range(32))
    pattern = [0x0f, 0x80, 0x00, 0x0d] * 4 + [0x02, 0xff, 0x04, 0x00] * 4
    control = (ctypes.c_ubyte * 32)(*pattern)
    expected = bytes(0 if index & 0x80 else (lane & 16) + (index & 15)
                     for lane, index in enumerate(pattern))
    for library in (native, lifted):
        shuffle = getattr(library, WIDE_EXPORT)
        shuffle.argtypes = [ctypes.c_uint64] * 3
        shuffle.restype = ctypes.c_uint64
        for i in range(32):
            out_n[i] = 0xff
        shuffle(ctypes.addressof(a), ctypes.addressof(control), ctypes.addressof(out_n))
        assert bytes(out_n) == expected, (bytes(out_n).hex(), expected.hex())
    print("VEX.256 shuffle with a 32-byte source matched both independent lanes")


if __name__ == "__main__":
    mode, native, other = sys.argv[1:4]
    if mode == "emit":
        emit(Path(native), Path(other))
    elif mode == "check":
        check(Path(native), Path(other))
    else:
        raise SystemExit(f"unknown mode {mode}")
