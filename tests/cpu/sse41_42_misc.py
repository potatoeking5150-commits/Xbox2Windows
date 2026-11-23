"""Differential check for SSSE3/SSE4.1/SSE4.2 integer, insert/extract, ptest, and crc32 lowering."""
import ctypes
import struct
import sys
from pathlib import Path

import pefile
from iced_x86 import Decoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from ffx_version_query import INTEGER_HEADER, lower, region

ALL_EXPORTS = (
    "pextrb_op", "pextrw_op", "pextrd_op", "pextrq_op",
    "pinsrb_op", "pinsrw_op", "pinsrd_op", "pinsrq_op",
    "pmaddubsw_op", "pmulhrsw_op", "phminposuw_op", "ptest_op",
    "mpsadbw_op", "pcmpgtq_op",
    "crc32_8_op", "crc32_32_op", "crc32_64_op",
    "pcmpestri_ub_eqany_pos_lsb", "pcmpestri_sw_ranges_neg_msb",
    "pcmpestri_sb_eqeach_mpos_msb", "pcmpestri_uw_eqord_mneg_lsb",
    "pcmpestrm_ub_eqany_pos_bit", "pcmpestrm_sb_ranges_neg_exp",
    "pcmpestrm_uw_eqeach_mpos_exp", "pcmpestrm_sw_eqord_mneg_bit",
    "pcmpistri_ub_eqord_pos_lsb", "pcmpistri_sb_ranges_mneg_msb",
    "pcmpistri_uw_eqany_neg_msb", "pcmpistri_sw_eqeach_mpos_lsb",
    "pcmpistrm_ub_eqeach_pos_exp", "pcmpistrm_sw_eqany_mneg_exp",
    "pcmpistrm_uw_ranges_mpos_bit", "pcmpistrm_sb_eqord_neg_bit",
    "pcmpestrm_ymm_preserve", "pcmpistrm_ymm_preserve",
)
VEC_EXPORTS = (
    "pinsrb_op", "pinsrw_op", "pinsrd_op", "pinsrq_op",
    "pmaddubsw_op", "pmulhrsw_op", "phminposuw_op",
    "mpsadbw_op", "pcmpgtq_op",
)

CPUID_HELPER = """#include <intrin.h>
extern "C" __declspec(dllexport) uint32_t host_has_sse42() {
    int info[4]{};
    __cpuid(info, 1);
    const int need = (1 << 9) | (1 << 19) | (1 << 20);
    return (info[2] & need) == need ? 1u : 0u;
}
extern "C" __declspec(dllexport) uint32_t host_has_avx() {
    int info[4]{};
    __cpuid(info, 1);
    const int need = (1 << 27) | (1 << 28);
    if ((info[2] & need) != need) return 0u;
    return (_xgetbv(0) & 6u) == 6u ? 1u : 0u;
}
"""


def emit(native_dll: Path, output_cpp: Path) -> None:
    parts = [INTEGER_HEADER, CPUID_HELPER]
    with pefile.PE(str(native_dll), fast_load=True) as pe:
        for name in ALL_EXPORTS:
            ranges = function_ranges(pe, export_rva(pe, name))
            body = [ins for lo, hi in ranges for ins in region(pe, lo, hi)]
            code = lower(body, ["rcx = a; rdx = b; r8 = dst;"], {}, {}, {})
            parts.append(
                f'extern "C" __declspec(dllexport) uint64_t {name}(uint64_t a, uint64_t b, uint64_t dst) {{\n{code}\n}}'
            )
    output_cpp.write_text("\n".join(parts) + "\n", encoding="utf-8")


def check_controls(native_dll: Path) -> None:
    with pefile.PE(str(native_dll), fast_load=True) as pe:
        ranges = function_ranges(pe, export_rva(pe, "movntdqa_op"))
        movntdqa = [ins for lo, hi in ranges for ins in region(pe, lo, hi)]
    try:
        lower(movntdqa, [], {}, {}, {})
    except ValueError as error:
        if "unsupported MOVNTDQA memory contract" not in str(error):
            raise
    else:
        raise AssertionError("expected MOVNTDQA memory form to be rejected")

    # 64-bit MMX pmulhrsw mm0, mm1 (0F 38 0B C1) followed by ret must be rejected.
    mmx = list(Decoder(64, b"\x0f\x38\x0b\xc1\xc3", ip=0x1000))
    try:
        lower(mmx, [], {}, {}, {})
    except ValueError:
        pass
    else:
        raise AssertionError("expected 64-bit MMX pmulhrsw to be rejected")

    # Unsupported packed string compare encodings (VEX, REX.W 64-bit, xmm3+, memory source) must be rejected.
    rejected_pcmp = (
        ("VEX vpcmpestri", "c4 e3 79 61 c1 00 c3", "unsupported query operation"),
        ("VEX vpcmpestrm", "c4 e3 79 60 c1 00 c3", "unsupported query operation"),
        ("VEX vpcmpistri", "c4 e3 79 63 c1 00 c3", "unsupported query operation"),
        ("VEX vpcmpistrm", "c4 e3 79 62 c1 00 c3", "unsupported query operation"),
        ("REX.W pcmpestri64", "66 48 0f 3a 61 c1 00 c3", "unsupported query operation"),
        ("REX.W pcmpestrm64", "66 48 0f 3a 60 c1 00 c3", "unsupported query operation"),
        ("untracked dst xmm3", "66 0f 3a 61 d9 00 c3", "unsupported packed string compare form"),
        ("untracked src xmm3", "66 0f 3a 63 c3 00 c3", "unsupported packed string compare form"),
        ("memory src pcmpestrm", "66 0f 3a 60 01 00 c3", "unsupported packed string compare form"),
    )
    for label, hex_bytes, expected_msg in rejected_pcmp:
        ins = list(Decoder(64, bytes.fromhex(hex_bytes), ip=0x1000))
        try:
            lower(ins, [], {}, {}, {})
        except ValueError as error:
            if expected_msg not in str(error):
                raise AssertionError(f"{label}: unexpected rejection {error}") from error
        else:
            raise AssertionError(f"expected {label} to be rejected")


def sdm_pcmp(a_bytes: bytes, b_bytes: bytes, imm: int, is_explicit: bool, raw_la: int = 0, raw_lb: int = 0):
    """Independent Intel SDM reference model for PCMPESTRI/M and PCMPISTRI/M."""
    is_word = bool(imm & 1)
    is_signed = bool(imm & 2)
    limit = 8 if is_word else 16
    agg = (imm >> 2) & 3
    pol = (imm >> 4) & 3
    high_sel = bool(imm & 0x40)

    fmt = f"<{limit}{'h' if is_signed else 'H'}" if is_word else f"<{limit}{'b' if is_signed else 'B'}"
    elems_a = list(struct.unpack(fmt, a_bytes))
    elems_b = list(struct.unpack(fmt, b_bytes))

    if is_explicit:
        def clamp_len(raw: int) -> int:
            s = struct.unpack("<i", struct.pack("<I", raw & 0xffffffff))[0]
            return min(abs(s), limit)
        len1 = clamp_len(raw_la)
        len2 = clamp_len(raw_lb)
    else:
        len1 = next((i for i, v in enumerate(elems_a) if v == 0), limit)
        len2 = next((i for i, v in enumerate(elems_b) if v == 0), limit)

    int_res1 = 0
    for i in range(limit):
        if agg == 0:  # Equal Any
            bit = i < len2 and any(elems_a[j] == elems_b[i] for j in range(len1))
        elif agg == 1:  # Ranges
            bit = i < len2 and any(
                elems_a[j] <= elems_b[i] <= elems_a[j + 1]
                for j in range(0, len1 - 1, 2)
            )
        elif agg == 2:  # Equal Each
            if i < len1 and i < len2:
                bit = elems_a[i] == elems_b[i]
            else:
                bit = i >= len1 and i >= len2
        else:  # Equal Ordered
            bit = all(
                (i + j) < len2 and elems_a[j] == elems_b[i + j]
                for j in range(min(len1, limit - i))
            )
        if bit:
            int_res1 |= 1 << i

    if pol == 1:
        int_res2 = int_res1 ^ ((1 << limit) - 1)
    elif pol == 3:
        int_res2 = int_res1 ^ ((1 << len2) - 1)
    else:
        int_res2 = int_res1

    if int_res2 == 0:
        index = limit
    elif not high_sel:
        index = (int_res2 & -int_res2).bit_length() - 1
    else:
        index = int_res2.bit_length() - 1

    if not high_sel:
        mask_bytes = struct.pack("<H", int_res2) + b"\x00" * 14
    elif not is_word:
        mask_bytes = bytes(0xff if ((int_res2 >> k) & 1) else 0x00 for k in range(16))
    else:
        mask_bytes = b"".join(struct.pack("<H", 0xffff if ((int_res2 >> k) & 1) else 0x0000) for k in range(8))

    flags = (1 if int_res2 != 0 else 0) | (2 if len2 < limit else 0) | (4 if (int_res2 & 1) else 0)
    return index, mask_bytes, flags


def check(native_dll: Path, lifted_dll: Path) -> None:
    check_controls(native_dll)
    lifted = ctypes.CDLL(str(lifted_dll))
    lifted.host_has_sse42.restype = ctypes.c_uint32
    if not lifted.host_has_sse42():
        print("host lacks SSSE3/SSE4.1/SSE4.2; skipping native differential check")
        raise SystemExit(77)
    native = ctypes.CDLL(str(native_dll))

    # Allocate 32 bytes and take a 16-byte aligned slice for movntdqa.
    raw_a = (ctypes.c_ubyte * 32)()
    off_a = (16 - (ctypes.addressof(raw_a) & 15)) & 15
    a_bytes = bytes([
        0xff, 0x80, 0x00, 0x80, 0x05, 0x00, 0x05, 0x00,
        0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
    ])
    b_bytes = bytes([
        0x7f, 0x80, 0x00, 0x80, 0xfe, 0x01, 0x02, 0x03,
        0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x7f,
    ])
    for i, v in enumerate(a_bytes):
        raw_a[off_a + i] = v
    a_ptr = ctypes.addressof(raw_a) + off_a
    b_buf = (ctypes.c_ubyte * 16)(*b_bytes)
    b_ptr = ctypes.addressof(b_buf)
    out_n = (ctypes.c_ubyte * 16)()
    out_l = (ctypes.c_ubyte * 16)()

    # Keep the authored aligned native control while translation rejects this form.
    native.movntdqa_op.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
    native.movntdqa_op.restype = ctypes.c_uint64
    native.movntdqa_op(a_ptr, 0, ctypes.addressof(out_n))
    if bytes(out_n) != a_bytes:
        raise AssertionError("native aligned MOVNTDQA did not copy all 16 bytes")

    cases = 0
    for name in VEC_EXPORTS:
        fn_n, fn_l = getattr(native, name), getattr(lifted, name)
        fn_n.argtypes = fn_l.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn_n.restype = fn_l.restype = ctypes.c_uint64
        fn_n(a_ptr, b_ptr, ctypes.addressof(out_n))
        fn_l(a_ptr, b_ptr, ctypes.addressof(out_l))
        if bytes(out_n) != bytes(out_l):
            raise AssertionError(f"{name}: native={bytes(out_n).hex()} lifted={bytes(out_l).hex()}")
        cases += 1

    for name in ("pextrb_op", "pextrw_op", "pextrd_op", "pextrq_op"):
        fn_n, fn_l = getattr(native, name), getattr(lifted, name)
        fn_n.argtypes = fn_l.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn_n.restype = fn_l.restype = ctypes.c_uint64
        got_n, got_l = fn_n(a_ptr, 0, 0), fn_l(a_ptr, 0, 0)
        if got_n != got_l:
            raise AssertionError(f"{name}: native={got_n:#x} lifted={got_l:#x}")
        cases += 1

    # Test all 4 (ZF, CF) combinations for ptest.
    for x, y, exp_flags in (
        (b"\x00" * 16, b"\x00" * 16, 3),  # ZF=1, CF=1
        (b"\x0f" * 16, b"\xf0" * 16, 1),  # ZF=1, CF=0
        (b"\xff" * 16, b"\x0f" * 16, 2),  # ZF=0, CF=1
        (b"\x0f" * 16, b"\xff" * 16, 0),  # ZF=0, CF=0
    ):
        xb = (ctypes.c_ubyte * 16)(*x)
        yb = (ctypes.c_ubyte * 16)(*y)
        native.ptest_op.argtypes = lifted.ptest_op.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        native.ptest_op.restype = lifted.ptest_op.restype = ctypes.c_uint64
        got_n = native.ptest_op(ctypes.addressof(xb), ctypes.addressof(yb), 0)
        got_l = lifted.ptest_op(ctypes.addressof(xb), ctypes.addressof(yb), 0)
        if got_n != exp_flags or got_l != exp_flags:
            raise AssertionError(f"ptest_op: expected={exp_flags} native={got_n} lifted={got_l}")
        cases += 1

    for name, seed, data in (
        ("crc32_8_op", 0x12345678, 0xAB),
        ("crc32_32_op", 0xDEADBEEF, 0x01234567),
        ("crc32_64_op", 0xFFFFFFFF12345678, 0x0123456789ABCDEF),
    ):
        fn_n, fn_l = getattr(native, name), getattr(lifted, name)
        fn_n.argtypes = fn_l.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn_n.restype = fn_l.restype = ctypes.c_uint64
        got_n, got_l = fn_n(seed, data, 0), fn_l(seed, data, 0)
        if got_n != got_l:
            raise AssertionError(f"{name}: native={got_n:#x} lifted={got_l:#x}")
        cases += 1

    # Packed string compare differential tests (PCMPESTRI, PCMPESTRM, PCMPISTRI, PCMPISTRM).
    pcmp_byte_pairs = (
        (b"aeiou\x00xxxxxxxxxx", b"hello world!\x00xyz"),
        (b"azAZ09\x80\x7f\x00abcdefg", b"mZ!\x80\x00hello\x7f\xff09ab"),
        (b"ab" + b"\x00" * 14, b"xxabxabyab" + b"\x00" * 6),
        (b"abc" + b"\x00" * 13, b"0123456789abcdab"),
        (b"\x00" * 16, b"\x00" * 16),
        (b"za" + b"\x80\x7f" + b"\x00" * 12, b"\x80\xff\x00\x01\x7f\x81azbcdefghi"),
        (b"0123456789abcdef", b"0123456789abcdef"),
        (b"0123456789abcdef", b"012345X789abcdef"),
    )
    pcmp_word_pairs = (
        (struct.pack("<8H", 10, 20, 30, 0x8000, 0, 1, 2, 3),
         struct.pack("<8H", 99, 20, 0x8000, 10, 0, 5, 10, 30)),
        (struct.pack("<8h", -100, -1, 10, 50, -32768, 32767, 0, 0),
         struct.pack("<8h", -101, -50, 0, 25, -32768, 32767, 100, -1)),
        (struct.pack("<8H", 1, 2, 3, 0, 0, 0, 0, 0),
         struct.pack("<8H", 9, 1, 2, 3, 9, 9, 1, 2)),
        (struct.pack("<8H", 0, 0, 0, 0, 0, 0, 0, 0),
         struct.pack("<8H", 1, 2, 3, 4, 5, 6, 7, 8)),
    )
    explicit_len_pairs = (
        (0, 0), (0, 5), (1, 5), (3, 16), (4, 8), (8, 8),
        (16, 16), (17, 5), (-1, 5), (4, -5), (-17, -17), (-0x80000000, 6),
    )

    pcmp_specs = (
        ("pcmpestri_ub_eqany_pos_lsb", 0x00, True, False, False, False),
        ("pcmpestri_sw_ranges_neg_msb", 0x57, True, False, True, False),
        ("pcmpestri_sb_eqeach_mpos_msb", 0x6A, True, False, False, False),
        ("pcmpestri_uw_eqord_mneg_lsb", 0x3D, True, False, True, False),
        ("pcmpestrm_ub_eqany_pos_bit", 0x00, True, True, False, False),
        ("pcmpestrm_sb_ranges_neg_exp", 0x56, True, True, False, False),
        ("pcmpestrm_uw_eqeach_mpos_exp", 0x69, True, True, True, True),
        ("pcmpestrm_sw_eqord_mneg_bit", 0x3F, True, True, True, False),
        ("pcmpistri_ub_eqord_pos_lsb", 0x8C, False, False, False, False),
        ("pcmpistri_sb_ranges_mneg_msb", 0x76, False, False, False, False),
        ("pcmpistri_uw_eqany_neg_msb", 0x51, False, False, True, False),
        ("pcmpistri_sw_eqeach_mpos_lsb", 0x2B, False, False, True, False),
        ("pcmpistrm_ub_eqeach_pos_exp", 0x48, False, True, False, False),
        ("pcmpistrm_sw_eqany_mneg_exp", 0x73, False, True, True, True),
        ("pcmpistrm_uw_ranges_mpos_bit", 0x25, False, True, True, False),
        ("pcmpistrm_sb_eqord_neg_bit", 0x1E, False, True, False, False),
    )

    io_n = (ctypes.c_ubyte * 32)()
    io_l = (ctypes.c_ubyte * 32)()
    for name, imm, is_exp, is_mask, is_word, check_xmm1 in pcmp_specs:
        fn_n, fn_l = getattr(native, name), getattr(lifted, name)
        fn_n.argtypes = fn_l.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
        fn_n.restype = fn_l.restype = ctypes.c_uint64
        vec_pairs = pcmp_word_pairs if is_word else pcmp_byte_pairs
        len_pairs = explicit_len_pairs if is_exp else ((0, 0),)
        for a_bytes, b_bytes in vec_pairs:
            a_buf = (ctypes.c_ubyte * 16)(*a_bytes)
            b_buf = (ctypes.c_ubyte * 16)(*b_bytes)
            for la, lb in len_pairs:
                exp_idx, exp_mask, exp_flags = sdm_pcmp(a_bytes, b_bytes, imm, is_exp, la, lb)
                header = struct.pack("<ii", la, lb) + b"\xaa" * 24 if is_exp else b"\xaa" * 32
                ctypes.memmove(io_n, header, 32)
                ctypes.memmove(io_l, header, 32)
                flags_n = fn_n(ctypes.addressof(a_buf), ctypes.addressof(b_buf), ctypes.addressof(io_n))
                flags_l = fn_l(ctypes.addressof(a_buf), ctypes.addressof(b_buf), ctypes.addressof(io_l))
                if flags_n != exp_flags or flags_l != exp_flags:
                    raise AssertionError(
                        f"{name} imm={imm:#x} la={la} lb={lb}: flags exp={exp_flags} native={flags_n} lifted={flags_l}"
                    )
                if is_mask:
                    got_n = bytes(io_n)[:16]
                    got_l = bytes(io_l)[:16]
                    if got_n != exp_mask or got_l != exp_mask:
                        raise AssertionError(
                            f"{name} imm={imm:#x} la={la} lb={lb}: mask exp={exp_mask.hex()} "
                            f"native={got_n.hex()} lifted={got_l.hex()}"
                        )
                    if check_xmm1:
                        if bytes(io_n)[16:32] != a_bytes or bytes(io_l)[16:32] != a_bytes:
                            raise AssertionError(f"{name}: xmm1 clobbered during implicit XMM0 mask write")
                else:
                    got_n = struct.unpack("<Q", bytes(io_n)[:8])[0]
                    got_l = struct.unpack("<Q", bytes(io_l)[:8])[0]
                    if got_n != exp_idx or got_l != exp_idx:
                        raise AssertionError(
                            f"{name} imm={imm:#x} la={la} lb={lb}: index exp={exp_idx:#x} "
                            f"native={got_n:#x} lifted={got_l:#x}"
                        )
                cases += 1

    lifted.host_has_avx.restype = ctypes.c_uint32
    if lifted.host_has_avx():
        sentinel_hi = bytes(range(0xA0, 0xB0))
        a32_bytes = b"aeiou\x00xxxxxxxxxx" + sentinel_hi
        b16_bytes = b"hello world!\x00xyz"
        a32_buf = (ctypes.c_ubyte * 32)(*a32_bytes)
        b16_buf = (ctypes.c_ubyte * 16)(*b16_bytes)
        for name, imm, is_exp, la, lb in (
            ("pcmpestrm_ymm_preserve", 0x40, True, 5, 12),
            ("pcmpistrm_ymm_preserve", 0x4C, False, 0, 0),
        ):
            fn_n, fn_l = getattr(native, name), getattr(lifted, name)
            fn_n.argtypes = fn_l.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]
            fn_n.restype = fn_l.restype = ctypes.c_uint64
            _, exp_mask, exp_flags = sdm_pcmp(a32_bytes[:16], b16_bytes, imm, is_exp, la, lb)
            header = struct.pack("<ii", la, lb) + b"\x00" * 24 if is_exp else b"\x00" * 32
            ctypes.memmove(io_n, header, 32)
            ctypes.memmove(io_l, header, 32)
            flags_n = fn_n(ctypes.addressof(a32_buf), ctypes.addressof(b16_buf), ctypes.addressof(io_n))
            flags_l = fn_l(ctypes.addressof(a32_buf), ctypes.addressof(b16_buf), ctypes.addressof(io_l))
            exp_ymm = exp_mask + sentinel_hi
            if flags_n != exp_flags or flags_l != exp_flags:
                raise AssertionError(f"{name}: flags exp={exp_flags} native={flags_n} lifted={flags_l}")
            if bytes(io_n) != exp_ymm or bytes(io_l) != exp_ymm:
                raise AssertionError(
                    f"{name}: YMM0 exp={exp_ymm.hex()} native={bytes(io_n).hex()} lifted={bytes(io_l).hex()}"
                )
            cases += 1

    print(f"{cases} SSSE3/SSE4.1/SSE4.2 misc cases matched native and translated DLLs")


if __name__ == "__main__":
    mode, native, other = sys.argv[1:4]
    if mode == "emit":
        emit(Path(native), Path(other))
    elif mode == "check":
        check(Path(native), Path(other))
    else:
        raise SystemExit(f"unknown mode {mode}")
