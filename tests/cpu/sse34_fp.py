"""Differential check for baseline 128-bit SSE3 and SSE4.1 floating-point lowering."""
import ctypes
from fractions import Fraction
import math
from pathlib import Path
import re
import struct
import sys

import pefile
from iced_x86 import Decoder, Mnemonic as M

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from ffx_version_query import INTEGER_HEADER, lower, region

U64 = ctypes.c_uint64

ROUND_IMMS = (0, 1, 2, 3, 4, 8, 9, 10, 11, 12)
ROUND_MEM_IMMS = (0, 1, 2, 3)
DPPS_IMMS = (
    0x00, 0x0F, 0xF0, 0xF1, 0xF2, 0xF4, 0xF8, 0xFF,
    0x11, 0x22, 0x44, 0x88, 0x55, 0xAA, 0x7F, 0x3C, 0xC3, 0x96,
)
DPPD_IMMS = (
    0x00, 0x01, 0x02, 0x03,
    0x10, 0x11, 0x12, 0x13,
    0x20, 0x21, 0x22, 0x23,
    0x30, 0x31, 0x32, 0x33,
    0xFF,
)
INSERTPS_MEM_IMMS = (
    0x00, 0x10, 0x20, 0x30,
    0x40, 0x51, 0x62, 0x74,
    0x88, 0x9F, 0xA5, 0xBA,
    0xC1, 0xD2, 0xE4, 0xF8,
)

ALL_EXPORTS = (
    "addsubps_op", "addsubps_mem_op",
    "addsubpd_op", "addsubpd_mem_op",
    "haddps_op", "haddpd_op", "hsubps_op", "hsubpd_op",
    "movddup_op", "movddup_mem_op",
    "movsldup_op", "movsldup_mem_op",
    "movshdup_op", "movshdup_mem_op",
    "lddqu_op",
    "roundps_op", "roundpd_op",
    "roundss_op", "roundss_mem_op",
    "roundsd_op", "roundsd_mem_op",
    "dpps_op", "dppd_op",
    "blendps_op", "blendpd_op",
    "blendvps_op", "blendvpd_op",
    "blendvps_self_op", "blendvpd_self_op",
    "insertps_op", "insertps_mem_op",
    "extractps_gpr_op", "extractps_mem_op",
)

CPUID_HELPER = """extern "C" __declspec(dllexport) uint32_t host_has_sse34() {
    int info[4]{};
    __cpuid(info, 1);
    const int need = (1 << 0) | (1 << 19);
    return (info[2] & need) == need ? 1u : 0u;
}
"""

# Mutated lifted negative controls derived from the lowered C++ of real exports.
MUTATED_CONTROLS = (
    ("addsubps_op", "addsubps_mutated_control", re.compile(r"_mm_addsub_ps"), "_mm_add_ps"),
    ("movsldup_op", "movsldup_mutated_control",
     re.compile(r"memcpy\(wide0\.data\(\), in2\.data\(\), 4\)"), "memcpy(wide0.data(), in2.data() + 4, 4)"),
    ("roundps_op", "roundps_mutated_control",
     re.compile(r"_mm_round_ps\(_mm_loadu_ps\(b\), 1\)"), "_mm_round_ps(_mm_loadu_ps(b), 2)"),
    ("insertps_op", "insertps_mutated_control",
     re.compile(r"memset\(wide0\.data\(\) \+ k \* 4, 0, 4\)"), "(void)k"),
)


def _dispatch_block(prefix: str, reg_sel: str, instr_fmt: str, imms, store_line: str) -> list[str]:
    out = []
    done = f"{prefix}_done"
    for idx, imm in enumerate(imms):
        nxt = f"{prefix}_n{idx + 1}" if idx + 1 < len(imms) else done
        out.append(f"    cmp {reg_sel}, {imm}")
        out.append(f"    jne {nxt}")
        out.append(f"    {instr_fmt.format(imm=imm)}")
        out.append(f"    jmp {done}")
        if idx + 1 < len(imms):
            out.append(f"{nxt}:")
    out.append(f"{done}:")
    if store_line:
        out.append(f"    {store_line}")
    out.append("    ret")
    return out


def generate_asm() -> str:
    lines = [
        "; coverage: source cpu_aot_sse34_fp",
        "; Source-owned differential fixture for 22 baseline 128-bit SSE3 and SSE4.1 floating-point instructions.",
        ".code",
        "",
    ]

    def add_proc(name: str, body: list[str]) -> None:
        lines.append(f"{name} PROC FRAME")
        lines.append("    .endprolog")
        lines.extend(body)
        lines.append(f"{name} ENDP")
        lines.append("")

    for mnem in ("addsubps", "addsubpd", "haddps", "haddpd", "hsubps", "hsubpd"):
        add_proc(f"{mnem}_op", [
            "    movdqu xmm0, xmmword ptr [rcx]",
            "    movdqu xmm1, xmmword ptr [rdx]",
            f"    {mnem} xmm0, xmm1",
            "    movdqu xmmword ptr [r8], xmm0",
            "    ret",
        ])
    for mnem in ("addsubps", "addsubpd"):
        add_proc(f"{mnem}_mem_op", [
            "    movdqu xmm0, xmmword ptr [rcx]",
            f"    {mnem} xmm0, xmmword ptr [rdx]",
            "    movdqu xmmword ptr [r8], xmm0",
            "    ret",
        ])

    add_proc("movddup_op", [
        "    movdqu xmm1, xmmword ptr [rcx]",
        "    movddup xmm0, xmm1",
        "    movdqu xmmword ptr [r8], xmm0",
        "    ret",
    ])
    add_proc("movddup_mem_op", [
        "    movddup xmm0, qword ptr [rcx]",
        "    movdqu xmmword ptr [r8], xmm0",
        "    ret",
    ])
    for mnem in ("movsldup", "movshdup"):
        add_proc(f"{mnem}_op", [
            "    movdqu xmm1, xmmword ptr [rcx]",
            f"    {mnem} xmm0, xmm1",
            "    movdqu xmmword ptr [r8], xmm0",
            "    ret",
        ])
        add_proc(f"{mnem}_mem_op", [
            f"    {mnem} xmm0, xmmword ptr [rcx]",
            "    movdqu xmmword ptr [r8], xmm0",
            "    ret",
        ])
    add_proc("lddqu_op", [
        "    lddqu xmm0, xmmword ptr [rcx]",
        "    movdqu xmmword ptr [r8], xmm0",
        "    ret",
    ])

    for mnem in ("roundps", "roundpd"):
        add_proc(f"{mnem}_op", [
            "    movdqu xmm1, xmmword ptr [rcx]",
            *_dispatch_block(mnem, "edx", f"{mnem} xmm0, xmm1, {{imm}}", ROUND_IMMS, "movdqu xmmword ptr [r8], xmm0"),
        ])
    add_proc("roundss_op", [
        "    movdqu xmm0, xmmword ptr [rcx]",
        "    movdqu xmm1, xmmword ptr [rdx]",
        *_dispatch_block("roundss", "r9d", "roundss xmm0, xmm1, {imm}", ROUND_IMMS, "movdqu xmmword ptr [r8], xmm0"),
    ])
    add_proc("roundss_mem_op", [
        "    movdqu xmm0, xmmword ptr [rcx]",
        *_dispatch_block("roundss_m", "r9d", "roundss xmm0, dword ptr [rdx], {imm}", ROUND_MEM_IMMS, "movdqu xmmword ptr [r8], xmm0"),
    ])
    add_proc("roundsd_op", [
        "    movdqu xmm0, xmmword ptr [rcx]",
        "    movdqu xmm1, xmmword ptr [rdx]",
        *_dispatch_block("roundsd", "r9d", "roundsd xmm0, xmm1, {imm}", ROUND_IMMS, "movdqu xmmword ptr [r8], xmm0"),
    ])
    add_proc("roundsd_mem_op", [
        "    movdqu xmm0, xmmword ptr [rcx]",
        *_dispatch_block("roundsd_m", "r9d", "roundsd xmm0, qword ptr [rdx], {imm}", ROUND_MEM_IMMS, "movdqu xmmword ptr [r8], xmm0"),
    ])

    add_proc("dpps_op", [
        "    movdqu xmm0, xmmword ptr [rcx]",
        "    movdqu xmm1, xmmword ptr [rdx]",
        *_dispatch_block("dpps", "r9d", "dpps xmm0, xmm1, {imm}", DPPS_IMMS, "movdqu xmmword ptr [r8], xmm0"),
    ])
    add_proc("dppd_op", [
        "    movdqu xmm0, xmmword ptr [rcx]",
        "    movdqu xmm1, xmmword ptr [rdx]",
        *_dispatch_block("dppd", "r9d", "dppd xmm0, xmm1, {imm}", DPPD_IMMS, "movdqu xmmword ptr [r8], xmm0"),
    ])
    add_proc("blendps_op", [
        "    movdqu xmm0, xmmword ptr [rcx]",
        "    movdqu xmm1, xmmword ptr [rdx]",
        *_dispatch_block("blendps", "r9d", "blendps xmm0, xmm1, {imm}", range(16), "movdqu xmmword ptr [r8], xmm0"),
    ])
    add_proc("blendpd_op", [
        "    movdqu xmm0, xmmword ptr [rcx]",
        "    movdqu xmm1, xmmword ptr [rdx]",
        *_dispatch_block("blendpd", "r9d", "blendpd xmm0, xmm1, {imm}", range(4), "movdqu xmmword ptr [r8], xmm0"),
    ])
    for mnem in ("blendvps", "blendvpd"):
        add_proc(f"{mnem}_op", [
            "    movdqu xmm1, xmmword ptr [rcx]",
            "    movdqu xmm2, xmmword ptr [rdx]",
            "    movdqu xmm0, xmmword ptr [r8]",
            f"    {mnem} xmm1, xmm2, xmm0",
            "    movdqu xmmword ptr [r9], xmm1",
            "    ret",
        ])
        add_proc(f"{mnem}_self_op", [
            "    movdqu xmm0, xmmword ptr [rcx]",
            "    movdqu xmm1, xmmword ptr [rdx]",
            f"    {mnem} xmm0, xmm1, xmm0",
            "    movdqu xmmword ptr [r8], xmm0",
            "    ret",
        ])

    add_proc("insertps_op", [
        "    movdqu xmm0, xmmword ptr [rcx]",
        "    movdqu xmm1, xmmword ptr [rdx]",
        *_dispatch_block("ins", "r9d", "insertps xmm0, xmm1, {imm}", range(256), "movdqu xmmword ptr [r8], xmm0"),
    ])
    add_proc("insertps_mem_op", [
        "    movdqu xmm0, xmmword ptr [rcx]",
        *_dispatch_block("insm", "r9d", "insertps xmm0, dword ptr [rdx], {imm}", INSERTPS_MEM_IMMS, "movdqu xmmword ptr [r8], xmm0"),
    ])
    add_proc("extractps_gpr_op", [
        "    movdqu xmm0, xmmword ptr [rcx]",
        "    mov rax, r8",
        *_dispatch_block("extg", "edx", "extractps eax, xmm0, {imm}", range(4), ""),
    ])
    add_proc("extractps_mem_op", [
        "    movdqu xmm0, xmmword ptr [rcx]",
        *_dispatch_block("extm", "edx", "extractps dword ptr [r8], xmm0, {imm}", range(4), ""),
    ])

    lines.append("END")
    return "\n".join(lines) + "\n"


def _verify_immediates(name: str, body) -> None:
    expected = {
        "roundps_op": (M.ROUNDPS, set(ROUND_IMMS)),
        "roundpd_op": (M.ROUNDPD, set(ROUND_IMMS)),
        "roundss_op": (M.ROUNDSS, set(ROUND_IMMS)),
        "roundss_mem_op": (M.ROUNDSS, set(ROUND_MEM_IMMS)),
        "roundsd_op": (M.ROUNDSD, set(ROUND_IMMS)),
        "roundsd_mem_op": (M.ROUNDSD, set(ROUND_MEM_IMMS)),
        "dpps_op": (M.DPPS, set(DPPS_IMMS)),
        "dppd_op": (M.DPPD, set(DPPD_IMMS)),
        "blendps_op": (M.BLENDPS, set(range(16))),
        "blendpd_op": (M.BLENDPD, set(range(4))),
        "insertps_op": (M.INSERTPS, set(range(256))),
        "insertps_mem_op": (M.INSERTPS, set(INSERTPS_MEM_IMMS)),
        "extractps_gpr_op": (M.EXTRACTPS, set(range(4))),
        "extractps_mem_op": (M.EXTRACTPS, set(range(4))),
    }
    if name not in expected:
        return
    mnem, want = expected[name]
    seen = {ins.immediate(2) & 0xff for ins in body if ins.mnemonic == mnem}
    if seen != want:
        raise AssertionError(f"{name}: decoded immediates {sorted(seen)} != expected {sorted(want)}")


def emit(native_dll: Path, output_cpp: Path) -> None:
    parts = [INTEGER_HEADER, CPUID_HELPER]
    lowered_by_name = {}
    with pefile.PE(str(native_dll), fast_load=True) as pe:
        for name in ALL_EXPORTS:
            ranges = function_ranges(pe, export_rva(pe, name))
            body = [ins for lo, hi in ranges for ins in region(pe, lo, hi)]
            _verify_immediates(name, body)
            code = lower(body, ["rcx = a; rdx = b; r8 = c; r9 = d;"], {}, {}, {})
            lowered_by_name[name] = code
            parts.append(
                f'extern "C" __declspec(dllexport) word {name}(word a, word b, word c, word d) {{\n{code}\n}}'
            )
    for src_name, ctrl_name, pattern, replacement in MUTATED_CONTROLS:
        mutated, count = pattern.subn(replacement, lowered_by_name[src_name])
        if not count:
            raise AssertionError(f"mutated control {ctrl_name} changed nothing")
        parts.append(
            f'extern "C" __declspec(dllexport) word {ctrl_name}(word a, word b, word c, word d) {{\n{mutated}\n}}'
        )
    output_cpp.write_text("\n".join(parts) + "\n", encoding="utf-8")


# --- Independent IEEE-754 binary32 / binary64 reference model (exact Fraction arithmetic) ---

class _FPFormat:
    def __init__(self, ebits: int, mbits: int):
        self.ebits = ebits
        self.mbits = mbits
        self.bits = 1 + ebits + mbits
        self.bias = (1 << (ebits - 1)) - 1
        self.emax = (1 << ebits) - 1
        self.sign_mask = 1 << (self.bits - 1)
        self.exp_mask = self.emax << mbits
        self.frac_mask = (1 << mbits) - 1
        self.quiet_bit = 1 << (mbits - 1)
        self.indef = self.sign_mask | self.exp_mask | self.quiet_bit

    def unpack(self, raw: int):
        sign = (raw >> (self.bits - 1)) & 1
        exp = (raw >> self.mbits) & self.emax
        frac = raw & self.frac_mask
        return sign, exp, frac

    def is_nan(self, raw: int) -> bool:
        _, exp, frac = self.unpack(raw)
        return exp == self.emax and frac != 0

    def is_inf(self, raw: int) -> bool:
        _, exp, frac = self.unpack(raw)
        return exp == self.emax and frac == 0

    def is_zero(self, raw: int) -> bool:
        _, exp, frac = self.unpack(raw)
        return exp == 0 and frac == 0

    def quiet(self, raw: int) -> int:
        return raw | self.quiet_bit

    def to_fraction(self, raw: int) -> Fraction:
        sign, exp, frac = self.unpack(raw)
        if exp == 0:
            mag = Fraction(frac, 1) * (Fraction(1, 2) ** (self.bias - 1 + self.mbits))
        else:
            mag = Fraction((1 << self.mbits) | frac, 1) * (Fraction(2, 1) ** (exp - self.bias - self.mbits))
        return -mag if sign else mag

    def from_fraction(self, val: Fraction, zero_sign: int = 0) -> int:
        if val == 0:
            return self.sign_mask if zero_sign else 0
        sign = 1 if val < 0 else 0
        mag = -val if sign else val
        # Find floor(log2(mag)): integer e such that 2^e <= mag < 2^(e+1).
        num, den = mag.numerator, mag.denominator
        e = num.bit_length() - den.bit_length()
        while (num << (-e) if e < 0 else num) < (den << e if e >= 0 else den):
            e -= 1
        while (num << (-(e + 1)) if e + 1 < 0 else num) >= (den << (e + 1) if e + 1 >= 0 else den):
            e += 1
        emin = 1 - self.bias
        if e < emin:
            # Subnormal (or underflow to signed zero): scale by 2^(mbits - emin)
            shift = self.mbits - emin
            scaled = mag * (Fraction(2, 1) ** shift)
            q = _round_nearest_even(scaled)
            if q == 0:
                return sign << (self.bits - 1)
            if q == (1 << self.mbits):
                return (sign << (self.bits - 1)) | (1 << self.mbits)
            return (sign << (self.bits - 1)) | q
        scaled = mag * (Fraction(2, 1) ** (self.mbits - e))
        q = _round_nearest_even(scaled)
        if q == (1 << (self.mbits + 1)):
            q >>= 1
            e += 1
        bexp = e + self.bias
        if bexp >= self.emax:
            return (sign << (self.bits - 1)) | self.exp_mask
        return (sign << (self.bits - 1)) | (bexp << self.mbits) | (q & self.frac_mask)

    def add(self, a: int, b: int) -> int:
        if self.is_nan(a):
            return self.quiet(a)
        if self.is_nan(b):
            return self.quiet(b)
        sa, _, _ = self.unpack(a)
        sb, _, _ = self.unpack(b)
        if self.is_inf(a) and self.is_inf(b):
            return self.indef if sa != sb else a
        if self.is_inf(a):
            return a
        if self.is_inf(b):
            return b
        if self.is_zero(a) and self.is_zero(b):
            return self.sign_mask if (sa and sb) else 0
        return self.from_fraction(self.to_fraction(a) + self.to_fraction(b), zero_sign=0)

    def sub(self, a: int, b: int) -> int:
        if self.is_nan(a):
            return self.quiet(a)
        if self.is_nan(b):
            return self.quiet(b)
        return self.add(a, b ^ self.sign_mask)

    def mul(self, a: int, b: int) -> int:
        if self.is_nan(a):
            return self.quiet(a)
        if self.is_nan(b):
            return self.quiet(b)
        sa, _, _ = self.unpack(a)
        sb, _, _ = self.unpack(b)
        rsign = sa ^ sb
        if (self.is_inf(a) and self.is_zero(b)) or (self.is_zero(a) and self.is_inf(b)):
            return self.indef
        if self.is_inf(a) or self.is_inf(b):
            return (rsign << (self.bits - 1)) | self.exp_mask
        if self.is_zero(a) or self.is_zero(b):
            return rsign << (self.bits - 1)
        return self.from_fraction(self.to_fraction(a) * self.to_fraction(b), zero_sign=rsign)

    def round_int(self, a: int, imm: int) -> int:
        if self.is_nan(a):
            return self.quiet(a)
        if self.is_inf(a) or self.is_zero(a):
            return a
        sa, _, _ = self.unpack(a)
        mode = 0 if (imm & 4) else (imm & 3)
        f = self.to_fraction(a)
        if mode == 0:
            n = _round_nearest_even(f)
        elif mode == 1:
            n = math.floor(f)
        elif mode == 2:
            n = math.ceil(f)
        else:
            n = int(f)
        return self.from_fraction(Fraction(n, 1), zero_sign=sa)


def _round_nearest_even(x: Fraction) -> int:
    sign = -1 if x < 0 else 1
    mag = -x if x < 0 else x
    q, r = divmod(mag.numerator, mag.denominator)
    twice_r = r * 2
    if twice_r > mag.denominator or (twice_r == mag.denominator and (q & 1)):
        q += 1
    return sign * q


F32 = _FPFormat(8, 23)
F64 = _FPFormat(11, 52)


def _u32s(buf: bytes) -> tuple[int, ...]:
    return struct.unpack("<4I", buf)


def _u64s(buf: bytes) -> tuple[int, ...]:
    return struct.unpack("<2Q", buf)


def _p32s(vals) -> bytes:
    return struct.pack("<4I", *vals)


def _p64s(vals) -> bytes:
    return struct.pack("<2Q", *vals)


def _dp_allowed(fmt, xa, xb, imm: int) -> set:
    # The product and pairwise reduction trees are architectural; the operand order
    # inside each commutative operation is vendor-defined, so enumerate every legal
    # result. NaN selection between two NaN operands differs across hosts.
    ts = [
        {fmt.mul(xa[k], xb[k]), fmt.mul(xb[k], xa[k])} if (imm & (1 << (4 + k))) else {0}
        for k in range(len(xa))
    ]
    if len(xa) == 2:
        return {fmt.add(x, y) for x in ts[0] for y in ts[1]} | {fmt.add(y, x) for x in ts[0] for y in ts[1]}
    p01 = {fmt.add(x, y) for x in ts[0] for y in ts[1]} | {fmt.add(y, x) for x in ts[0] for y in ts[1]}
    p23 = {fmt.add(x, y) for x in ts[2] for y in ts[3]} | {fmt.add(y, x) for x in ts[2] for y in ts[3]}
    return {fmt.add(x, y) for x in p01 for y in p23} | {fmt.add(y, x) for x in p01 for y in p23}


def ref_binary(name: str, a: bytes, b: bytes, imm: int = 0) -> bytes:
    if name in ("addsubps_op", "addsubps_mem_op"):
        xa, xb = _u32s(a), _u32s(b)
        return _p32s((F32.sub(xa[0], xb[0]), F32.add(xa[1], xb[1]),
                      F32.sub(xa[2], xb[2]), F32.add(xa[3], xb[3])))
    if name in ("addsubpd_op", "addsubpd_mem_op"):
        xa, xb = _u64s(a), _u64s(b)
        return _p64s((F64.sub(xa[0], xb[0]), F64.add(xa[1], xb[1])))
    if name == "haddps_op":
        xa, xb = _u32s(a), _u32s(b)
        return _p32s((F32.add(xa[0], xa[1]), F32.add(xa[2], xa[3]),
                      F32.add(xb[0], xb[1]), F32.add(xb[2], xb[3])))
    if name == "haddpd_op":
        xa, xb = _u64s(a), _u64s(b)
        return _p64s((F64.add(xa[0], xa[1]), F64.add(xb[0], xb[1])))
    if name == "hsubps_op":
        xa, xb = _u32s(a), _u32s(b)
        return _p32s((F32.sub(xa[0], xa[1]), F32.sub(xa[2], xa[3]),
                      F32.sub(xb[0], xb[1]), F32.sub(xb[2], xb[3])))
    if name == "hsubpd_op":
        xa, xb = _u64s(a), _u64s(b)
        return _p64s((F64.sub(xa[0], xa[1]), F64.sub(xb[0], xb[1])))
    if name in ("movddup_op", "movddup_mem_op"):
        return a[:8] + a[:8]
    if name in ("movsldup_op", "movsldup_mem_op"):
        return a[0:4] + a[0:4] + a[8:12] + a[8:12]
    if name in ("movshdup_op", "movshdup_mem_op"):
        return a[4:8] + a[4:8] + a[12:16] + a[12:16]
    if name == "lddqu_op":
        return a[:16]
    if name == "roundps_op":
        return _p32s(F32.round_int(x, imm) for x in _u32s(a))
    if name == "roundpd_op":
        return _p64s(F64.round_int(x, imm) for x in _u64s(a))
    if name in ("roundss_op", "roundss_mem_op"):
        xa = list(_u32s(a))
        sb0 = struct.unpack("<I", b[:4])[0]
        xa[0] = F32.round_int(sb0, imm)
        return _p32s(xa)
    if name in ("roundsd_op", "roundsd_mem_op"):
        xa = list(_u64s(a))
        sb0 = struct.unpack("<Q", b[:8])[0]
        xa[0] = F64.round_int(sb0, imm)
        return _p64s(xa)
    if name == "dpps_op":
        xa, xb = _u32s(a), _u32s(b)
        allowed = _dp_allowed(F32, xa, xb, imm)
        if len(allowed) != 1:
            raise AssertionError(f"dpps_op reference is vendor-dependent: {sorted(hex(v) for v in allowed)}")
        total = allowed.pop()
        return _p32s(total if (imm & (1 << k)) else 0 for k in range(4))
    if name == "dppd_op":
        xa, xb = _u64s(a), _u64s(b)
        allowed = _dp_allowed(F64, xa, xb, imm)
        if len(allowed) != 1:
            raise AssertionError(f"dppd_op reference is vendor-dependent: {sorted(hex(v) for v in allowed)}")
        total = allowed.pop()
        return _p64s(total if (imm & (1 << k)) else 0 for k in range(2))
    if name == "blendps_op":
        xa, xb = _u32s(a), _u32s(b)
        return _p32s(xb[k] if ((imm >> k) & 1) else xa[k] for k in range(4))
    if name == "blendpd_op":
        xa, xb = _u64s(a), _u64s(b)
        return _p64s(xb[k] if ((imm >> k) & 1) else xa[k] for k in range(2))
    if name in ("insertps_op", "insertps_mem_op"):
        xa = list(_u32s(a))
        if name == "insertps_mem_op":
            val = struct.unpack("<I", b[:4])[0]
        else:
            val = _u32s(b)[(imm >> 6) & 3]
        xa[(imm >> 4) & 3] = val
        for k in range(4):
            if (imm >> k) & 1:
                xa[k] = 0
        return _p32s(xa)
    raise AssertionError(name)


def ref_blendv(name: str, a: bytes, b: bytes, mask: bytes) -> bytes:
    if name in ("blendvps_op", "blendvps_self_op"):
        xa, xb, xm = _u32s(a), _u32s(b), _u32s(mask)
        return _p32s(xb[k] if (xm[k] & 0x80000000) else xa[k] for k in range(4))
    if name in ("blendvpd_op", "blendvpd_self_op"):
        xa, xb, xm = _u64s(a), _u64s(b), _u64s(mask)
        return _p64s(xb[k] if (xm[k] & 0x8000000000000000) else xa[k] for k in range(2))
    raise AssertionError(name)


def check_controls() -> None:
    reject_cases = (
        ("MMX paddb mm0, mm1", "0f fc c1 c3"),
        ("MMX phaddw mm0, mm1", "0f 38 01 c1 c3"),
        ("VEX vaddsubps xmm0, xmm1, xmm2", "c5 f3 d0 c2 c3"),
        ("VEX vroundps xmm0, xmm1, 1", "c4 e3 79 08 c1 01 c3"),
        ("VEX vdpps xmm0, xmm1, xmm2, 0xf1", "c4 e3 71 40 c2 f1 c3"),
        ("VEX vblendps xmm0, xmm1, xmm2, 5", "c4 e3 71 0c c2 05 c3"),
        ("VEX vextractps eax, xmm0, 2", "c4 e3 79 17 c0 02 c3"),
        ("untracked dst addsubps xmm3, xmm0", "f2 0f d0 d8 c3"),
        ("untracked src addsubps xmm0, xmm3", "f2 0f d0 c3 c3"),
        ("untracked src extractps eax, xmm3, 0", "66 0f 3a 17 d8 00 c3"),
    )
    for label, hx in reject_cases:
        insns = list(Decoder(64, bytes.fromhex(hx), ip=0x1000))
        try:
            lower(insns, [], {}, {}, {})
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected {label} to be rejected")

    # MOVNTDQA must remain rejected with its memory-contract diagnostic.
    movntdqa = list(Decoder(64, bytes.fromhex("66 0f 38 2a 01 c3"), ip=0x1000))
    try:
        lower(movntdqa, [], {}, {}, {})
    except ValueError as err:
        if "unsupported MOVNTDQA memory contract" not in str(err):
            raise AssertionError(f"unexpected MOVNTDQA diagnostic: {err}") from err
    else:
        raise AssertionError("expected MOVNTDQA to remain rejected")

    # Verify upper YMM bytes (bytes 16..31) are preserved across legacy SSE3/SSE4.1 FP ops.
    ymm_checks = (
        ("addsubps", "c5 fc 10 01 f2 0f d0 c1 c5 fc 11 00 c3"),
        ("movddup", "c5 fc 10 01 f2 0f 12 c1 c5 fc 11 00 c3"),
        ("roundps", "c5 fc 10 01 66 0f 3a 08 c1 01 c5 fc 11 00 c3"),
        ("dpps", "c5 fc 10 01 66 0f 3a 40 c1 f1 c5 fc 11 00 c3"),
        ("blendvps", "c5 fc 10 01 66 0f 38 14 c1 c5 fc 11 00 c3"),
        ("insertps", "c5 fc 10 01 66 0f 3a 21 c1 92 c5 fc 11 00 c3"),
    )
    for label, hx in ymm_checks:
        code = lower(list(Decoder(64, bytes.fromhex(hx), ip=0x1000)), [], {}, {}, {})
        op_line = [ln for ln in code.splitlines() if "block_1004:" in ln][0]
        if "wide0 = {};" in op_line or "wide0.data() + 16" in op_line or "wide0.data()+16" in op_line:
            raise AssertionError(f"{label} unexpectedly cleared upper YMM bytes: {op_line}")


def _aligned_buf(data: bytes):
    raw = (ctypes.c_ubyte * (len(data) + 32))()
    ctypes.memset(raw, 0xCC, len(raw))
    off = (16 - (ctypes.addressof(raw) & 15)) & 15
    for idx, b in enumerate(data):
        raw[off + idx] = b
    return raw, ctypes.addressof(raw) + off


def _guarded_buf(data: bytes, align_offset: int = 0):
    pad = 16
    raw = (ctypes.c_ubyte * (pad + align_offset + len(data) + pad + 16))()
    ctypes.memset(raw, 0xCC, len(raw))
    base = ((ctypes.addressof(raw) + 15) & ~15) + align_offset
    off = base - ctypes.addressof(raw)
    for idx, b in enumerate(data):
        raw[off + idx] = b
    return raw, base, off


def _load_lib(path: Path, exports):
    lib = ctypes.CDLL(str(path.resolve()))
    for name in exports:
        fn = getattr(lib, name)
        fn.argtypes = [U64, U64, U64, U64]
        fn.restype = U64
    return lib


def _f32_bytes(*vals) -> bytes:
    out = []
    for v in vals:
        out.append(struct.pack("<I", v) if isinstance(v, int) else struct.pack("<f", v))
    return b"".join(out)


def _f64_bytes(*vals) -> bytes:
    out = []
    for v in vals:
        out.append(struct.pack("<Q", v) if isinstance(v, int) else struct.pack("<d", v))
    return b"".join(out)


def check(native_dll: Path, lifted_dll: Path) -> None:
    check_controls()
    ctrl_names = tuple(c[1] for c in MUTATED_CONTROLS)
    lifted = _load_lib(lifted_dll, (*ALL_EXPORTS, *ctrl_names))
    lifted.host_has_sse34.argtypes = []
    lifted.host_has_sse34.restype = ctypes.c_uint32
    if not lifted.host_has_sse34():
        print("host lacks SSE3/SSE4.1 (CPUID.1 ECX bits 0 and 19); skipping native differential check")
        raise SystemExit(77)
    native = _load_lib(native_dll, ALL_EXPORTS)

    f32_pairs = (
        # Mixed signs, fractional values, ties to even
        (_f32_bytes(1.5, -2.5, 3.5, -0.5), _f32_bytes(0.25, -1.25, 2.5, -0.5)),
        # Signed zeros and exact cancellation
        (_f32_bytes(0x00000000, 0x80000000, 0x80000000, 1.0),
         _f32_bytes(0x80000000, 0x80000000, 0x00000000, -1.0)),
        # Denormals (smallest, largest, negative)
        (_f32_bytes(0x00000001, 0x80000001, 0x007fffff, 0x807fffff),
         _f32_bytes(0x00000001, 0x00000002, 0x00000001, 0x80000001)),
        # NaN payloads (qNaN + sNaN on both sides) and infinities
        (_f32_bytes(0x7fc12345, 0x7f800000, 0x7f800123, 0xff800000),
         _f32_bytes(0xffc54321, 0x7f800000, 1.0, 0x00000000)),
        # Second-operand NaNs (qNaN and sNaN) + negative qNaN
        (_f32_bytes(2.0, -3.0, 0x00000000, 0x80000000),
         _f32_bytes(0x7fc00abc, 0xff800def, 0x7f800000, 0xffc01234)),
        # Non-associative DPPS test: (t0+t1)+(t2+t3) != ((t0+t1)+t2)+t3
        (_f32_bytes(1.0e20, -1.0e20, 1.0, 2.0), _f32_bytes(1.0, 1.0, 1.5, -0.25)),
    )
    f64_pairs = (
        (_f64_bytes(1.5, -2.5), _f64_bytes(0.25, -1.25)),
        (_f64_bytes(0x0000000000000000, 0x8000000000000000),
         _f64_bytes(0x8000000000000000, 0x8000000000000000)),
        (_f64_bytes(0x0000000000000001, 0x800fffffffffffff),
         _f64_bytes(0x0000000000000001, 0x8000000000000001)),
        (_f64_bytes(0x7ff8000000012345, 0x7ff0000000000000),
         _f64_bytes(0xfff8000000054321, 0x7ff0000000000000)),
        (_f64_bytes(0x7ff0000000000123, 0x0000000000000000),
         _f64_bytes(-4.5, 0xfff0000000000000)),
        (_f64_bytes(3.5, -0.5), _f64_bytes(0xfff0000000000def, 0x7ff8000000000abc)),
    )

    out_n = (ctypes.c_ubyte * 16)()
    out_l = (ctypes.c_ubyte * 16)()
    cases = 0

    def run_vec(name: str, a_bytes: bytes, b_bytes: bytes, arg3: int = 0):
        nonlocal cases
        ra, pa = _aligned_buf(a_bytes)
        rb, pb = _aligned_buf(b_bytes)
        ctypes.memset(out_n, 0xAA, 16)
        ctypes.memset(out_l, 0x55, 16)
        getattr(native, name)(pa, pb, ctypes.addressof(out_n), arg3)
        getattr(lifted, name)(pa, pb, ctypes.addressof(out_l), arg3)
        got_n, got_l = bytes(out_n), bytes(out_l)
        if name in ("dpps_op", "dppd_op"):
            fmt, xs, lanes = (F32, _u32s, 4) if name == "dpps_op" else (F64, _u64s, 2)
            allowed = _dp_allowed(fmt, xs(a_bytes), xs(b_bytes), arg3)
            bad = None
            if got_n != got_l:
                bad = "native and lifted disagree"
            else:
                words = xs(got_n)
                enabled = {words[k] for k in range(lanes) if arg3 & (1 << k)}
                if any(words[k] != 0 for k in range(lanes) if not (arg3 & (1 << k))):
                    bad = "disabled lane is not +0.0"
                elif not enabled <= allowed:
                    bad = f"result not in architecturally allowed set {sorted(allowed)}"
            if bad:
                raise AssertionError(
                    f"{name}(arg3={arg3:#x}): {bad}: a={a_bytes.hex()} b={b_bytes.hex()} "
                    f"native={got_n.hex()} lifted={got_l.hex()}"
                )
            cases += 1
            return
        want = ref_binary(name, a_bytes, b_bytes, arg3)
        if got_n != want or got_l != want:
            raise AssertionError(
                f"{name}(arg3={arg3:#x}): a={a_bytes.hex()} b={b_bytes.hex()} "
                f"expected={want.hex()} native={got_n.hex()} lifted={got_l.hex()}"
            )
        cases += 1

    # A +4 pointer is readable but violates these legacy m128 instructions' alignment.
    # A failing load cannot publish output. The ordinary aligned cases below still run.
    for name, memory_argument in (("addsubps_mem_op", 1), ("addsubpd_mem_op", 1),
                                  ("movsldup_mem_op", 0), ("movshdup_mem_op", 0)):
        buffers = [_guarded_buf(bytes(16), align_offset=4 if j == memory_argument else 0)
                   for j in range(2)]
        args = [buffer[1] for buffer in buffers]
        for dll in (native, lifted):
            ctypes.memset(out_n, 0xAA, 16)
            try:
                getattr(dll, name)(*args, ctypes.addressof(out_n), 0)
            except OSError as exc:
                if "access violation reading 0xFFFFFFFFFFFFFFFF" not in str(exc):
                    raise
            else:
                raise AssertionError(f"{name}: unaligned m128 load did not fault")
            if bytes(out_n) != bytes([0xAA]) * 16:
                raise AssertionError(f"{name}: failed load changed output")

    # 1. Binary SSE3 arithmetic (reg and aligned memory)
    for name in ("addsubps_op", "addsubps_mem_op", "haddps_op", "hsubps_op"):
        for a, b in f32_pairs:
            run_vec(name, a, b)
    for name in ("addsubpd_op", "addsubpd_mem_op", "haddpd_op", "hsubpd_op"):
        for a, b in f64_pairs:
            run_vec(name, a, b)

    # 2. SSE3 duplication and unaligned load
    for a, _ in f64_pairs:
        run_vec("movddup_op", a, b"\x00" * 16)
        # Narrow 8-byte memory operand surrounded by 0xCC guard bytes
        ra, pa, _ = _guarded_buf(a[:8], align_offset=1)
        ctypes.memset(out_n, 0xAA, 16)
        ctypes.memset(out_l, 0x55, 16)
        native.movddup_mem_op(pa, 0, ctypes.addressof(out_n), 0)
        lifted.movddup_mem_op(pa, 0, ctypes.addressof(out_l), 0)
        want = ref_binary("movddup_mem_op", a[:8], b"")
        if bytes(out_n) != want or bytes(out_l) != want:
            raise AssertionError(f"movddup_mem_op: expected={want.hex()} native={bytes(out_n).hex()} lifted={bytes(out_l).hex()}")
        cases += 1

    for name in ("movsldup_op", "movsldup_mem_op", "movshdup_op", "movshdup_mem_op"):
        for a, _ in f32_pairs:
            run_vec(name, a, b"\x00" * 16)

    for offset in (0, 1, 3, 7):
        a = f32_pairs[offset % len(f32_pairs)][0]
        ra, pa, _ = _guarded_buf(a, align_offset=offset)
        ctypes.memset(out_n, 0xAA, 16)
        ctypes.memset(out_l, 0x55, 16)
        native.lddqu_op(pa, 0, ctypes.addressof(out_n), 0)
        lifted.lddqu_op(pa, 0, ctypes.addressof(out_l), 0)
        if bytes(out_n) != a or bytes(out_l) != a:
            raise AssertionError(f"lddqu_op(off={offset}): expected={a.hex()} native={bytes(out_n).hex()} lifted={bytes(out_l).hex()}")
        cases += 1

    # 3. SSE4.1 rounding (all four modes + MXCSR + NO_EXC + narrow memory operands)
    round_f32_vecs = (
        _f32_bytes(-3.5, -2.5, -0.25, -0.5),
        _f32_bytes(0.25, 0.5, 2.5, 3.5),
        _f32_bytes(0x00000000, 0x80000000, 0x00000001, 0x80000001),
        _f32_bytes(0x7fc12345, 0xff800456, 0x7f800000, 0xff800000),
        _f32_bytes(16777215.0, -16777215.0, 1.9999999, -1.9999999),
    )
    round_f64_vecs = (
        _f64_bytes(-3.5, -2.5),
        _f64_bytes(-0.25, -0.5),
        _f64_bytes(0.25, 0.5),
        _f64_bytes(2.5, 3.5),
        _f64_bytes(0x0000000000000000, 0x8000000000000000),
        _f64_bytes(0x0000000000000001, 0x8000000000000001),
        _f64_bytes(0x7ff8000000012345, 0xfff0000000000456),
        _f64_bytes(0x7ff0000000000000, 0xfff0000000000000),
    )
    for vec in round_f32_vecs:
        ra, pa = _aligned_buf(vec)
        for imm in ROUND_IMMS:
            ctypes.memset(out_n, 0xAA, 16)
            ctypes.memset(out_l, 0x55, 16)
            native.roundps_op(pa, imm, ctypes.addressof(out_n), 0)
            lifted.roundps_op(pa, imm, ctypes.addressof(out_l), 0)
            want = ref_binary("roundps_op", vec, b"", imm)
            if bytes(out_n) != want or bytes(out_l) != want:
                raise AssertionError(f"roundps_op(imm={imm}): expected={want.hex()} native={bytes(out_n).hex()} lifted={bytes(out_l).hex()}")
            cases += 1

    for vec in round_f64_vecs:
        ra, pa = _aligned_buf(vec)
        for imm in ROUND_IMMS:
            ctypes.memset(out_n, 0xAA, 16)
            ctypes.memset(out_l, 0x55, 16)
            native.roundpd_op(pa, imm, ctypes.addressof(out_n), 0)
            lifted.roundpd_op(pa, imm, ctypes.addressof(out_l), 0)
            want = ref_binary("roundpd_op", vec, b"", imm)
            if bytes(out_n) != want or bytes(out_l) != want:
                raise AssertionError(f"roundpd_op(imm={imm}): expected={want.hex()} native={bytes(out_n).hex()} lifted={bytes(out_l).hex()}")
            cases += 1

    upper_f32 = _f32_bytes(99.0, 0x7f800777, -42.5, 0x80000001)
    for vec in round_f32_vecs:
        for imm in ROUND_IMMS:
            run_vec("roundss_op", upper_f32, vec, imm)
        # Narrow 4-byte memory operand check
        rb, pb, _ = _guarded_buf(vec[:4], align_offset=1)
        ra, pa = _aligned_buf(upper_f32)
        for imm in ROUND_MEM_IMMS:
            ctypes.memset(out_n, 0xAA, 16)
            ctypes.memset(out_l, 0x55, 16)
            native.roundss_mem_op(pa, pb, ctypes.addressof(out_n), imm)
            lifted.roundss_mem_op(pa, pb, ctypes.addressof(out_l), imm)
            want = ref_binary("roundss_mem_op", upper_f32, vec[:4], imm)
            if bytes(out_n) != want or bytes(out_l) != want:
                raise AssertionError(f"roundss_mem_op(imm={imm}): expected={want.hex()} native={bytes(out_n).hex()} lifted={bytes(out_l).hex()}")
            cases += 1

    upper_f64 = _f64_bytes(99.0, 0x7ff0000000000777)
    for vec in round_f64_vecs:
        for imm in ROUND_IMMS:
            run_vec("roundsd_op", upper_f64, vec, imm)
        rb, pb, _ = _guarded_buf(vec[:8], align_offset=1)
        ra, pa = _aligned_buf(upper_f64)
        for imm in ROUND_MEM_IMMS:
            ctypes.memset(out_n, 0xAA, 16)
            ctypes.memset(out_l, 0x55, 16)
            native.roundsd_mem_op(pa, pb, ctypes.addressof(out_n), imm)
            lifted.roundsd_mem_op(pa, pb, ctypes.addressof(out_l), imm)
            want = ref_binary("roundsd_mem_op", upper_f64, vec[:8], imm)
            if bytes(out_n) != want or bytes(out_l) != want:
                raise AssertionError(f"roundsd_mem_op(imm={imm}): expected={want.hex()} native={bytes(out_n).hex()} lifted={bytes(out_l).hex()}")
            cases += 1

    # 4. SSE4.1 dot products
    for a, b in f32_pairs:
        for imm in DPPS_IMMS:
            run_vec("dpps_op", a, b, imm)
    for a, b in f64_pairs:
        for imm in DPPD_IMMS:
            run_vec("dppd_op", a, b, imm)

    # 5. SSE4.1 blends (all immediate masks + variable mask including dst==xmm0 aliasing)
    blend_a32 = _f32_bytes(0x7f800111, 0x80000000, 0x00000001, 1.5)
    blend_b32 = _f32_bytes(0xffc00222, 0x00000000, 0x807fffff, -2.5)
    for imm in range(16):
        run_vec("blendps_op", blend_a32, blend_b32, imm)

    blend_a64 = _f64_bytes(0x7ff0000000000111, 0x8000000000000000)
    blend_b64 = _f64_bytes(0xfff8000000000222, 0x0000000000000001)
    for imm in range(4):
        run_vec("blendpd_op", blend_a64, blend_b64, imm)

    # Exercise all 16 MSB mask patterns for blendvps (with non-MSB noise bits set)
    for pat in range(16):
        mask = _f32_bytes(*(0x80000000 | (k * 0x12345) if ((pat >> k) & 1) else (0x7fffffff ^ (k * 0x12345)) for k in range(4)))
        ra, pa = _aligned_buf(blend_a32)
        rb, pb = _aligned_buf(blend_b32)
        rm, pm = _aligned_buf(mask)
        ctypes.memset(out_n, 0xAA, 16)
        ctypes.memset(out_l, 0x55, 16)
        native.blendvps_op(pa, pb, pm, ctypes.addressof(out_n))
        lifted.blendvps_op(pa, pb, pm, ctypes.addressof(out_l))
        want = ref_blendv("blendvps_op", blend_a32, blend_b32, mask)
        if bytes(out_n) != want or bytes(out_l) != want:
            raise AssertionError(f"blendvps_op(pat={pat}): expected={want.hex()} native={bytes(out_n).hex()} lifted={bytes(out_l).hex()}")
        cases += 1
        # Aliasing check: dst == implicit xmm0 mask
        ctypes.memset(out_n, 0xAA, 16)
        ctypes.memset(out_l, 0x55, 16)
        native.blendvps_self_op(pm, pb, ctypes.addressof(out_n), 0)
        lifted.blendvps_self_op(pm, pb, ctypes.addressof(out_l), 0)
        want_s = ref_blendv("blendvps_self_op", mask, blend_b32, mask)
        if bytes(out_n) != want_s or bytes(out_l) != want_s:
            raise AssertionError(f"blendvps_self_op(pat={pat}): expected={want_s.hex()} native={bytes(out_n).hex()} lifted={bytes(out_l).hex()}")
        cases += 1

    # Exercise all 4 MSB mask patterns for blendvpd (with non-MSB noise bits)
    for pat in range(4):
        mask = _f64_bytes(*(0x8000000000000000 | (k * 0x123456789) if ((pat >> k) & 1) else (0x7fffffffffffffff ^ (k * 0x123456789)) for k in range(2)))
        ra, pa = _aligned_buf(blend_a64)
        rb, pb = _aligned_buf(blend_b64)
        rm, pm = _aligned_buf(mask)
        ctypes.memset(out_n, 0xAA, 16)
        ctypes.memset(out_l, 0x55, 16)
        native.blendvpd_op(pa, pb, pm, ctypes.addressof(out_n))
        lifted.blendvpd_op(pa, pb, pm, ctypes.addressof(out_l))
        want = ref_blendv("blendvpd_op", blend_a64, blend_b64, mask)
        if bytes(out_n) != want or bytes(out_l) != want:
            raise AssertionError(f"blendvpd_op(pat={pat}): expected={want.hex()} native={bytes(out_n).hex()} lifted={bytes(out_l).hex()}")
        cases += 1
        ctypes.memset(out_n, 0xAA, 16)
        ctypes.memset(out_l, 0x55, 16)
        native.blendvpd_self_op(pm, pb, ctypes.addressof(out_n), 0)
        lifted.blendvpd_self_op(pm, pb, ctypes.addressof(out_l), 0)
        want_s = ref_blendv("blendvpd_self_op", mask, blend_b64, mask)
        if bytes(out_n) != want_s or bytes(out_l) != want_s:
            raise AssertionError(f"blendvpd_self_op(pat={pat}): expected={want_s.hex()} native={bytes(out_n).hex()} lifted={bytes(out_l).hex()}")
        cases += 1

    # 6. SSE4.1 insertps (all 256 reg imm8 combinations + 16 narrow-m32 imm8 combinations)
    ins_a = _f32_bytes(0x7f800111, 0x80000000, 0x00000001, 0xbf800000)
    ins_b = _f32_bytes(0x3f800000, 0xffc00222, 0x807fffff, 0x40400000)
    for imm in range(256):
        run_vec("insertps_op", ins_a, ins_b, imm)
    rb, pb, _ = _guarded_buf(ins_b[:4], align_offset=1)
    ra, pa = _aligned_buf(ins_a)
    for imm in INSERTPS_MEM_IMMS:
        ctypes.memset(out_n, 0xAA, 16)
        ctypes.memset(out_l, 0x55, 16)
        native.insertps_mem_op(pa, pb, ctypes.addressof(out_n), imm)
        lifted.insertps_mem_op(pa, pb, ctypes.addressof(out_l), imm)
        want = ref_binary("insertps_mem_op", ins_a, ins_b[:4], imm)
        if bytes(out_n) != want or bytes(out_l) != want:
            raise AssertionError(f"insertps_mem_op(imm={imm:#x}): expected={want.hex()} native={bytes(out_n).hex()} lifted={bytes(out_l).hex()}")
        cases += 1

    # 7. SSE4.1 extractps (all 4 lanes, GPR zero-extension check + guarded 4-byte memory store)
    ext_src = _f32_bytes(0x7f800123, 0x80000000, 0x00000001, 0xdeadbeef)
    ra, pa = _aligned_buf(ext_src)
    poison = 0xfeedfacecafebabe
    for lane in range(4):
        want_u32 = _u32s(ext_src)[lane]
        got_n = native.extractps_gpr_op(pa, lane, poison, 0)
        got_l = lifted.extractps_gpr_op(pa, lane, poison, 0)
        if got_n != want_u32 or got_l != want_u32:
            raise AssertionError(f"extractps_gpr_op(lane={lane}): expected={want_u32:#x} native={got_n:#x} lifted={got_l:#x}")
        cases += 1

        # Guarded 16-byte buffer initialized to 0xCC; only bytes 0..3 may be written.
        dst_n = (ctypes.c_ubyte * 16)(*([0xCC] * 16))
        dst_l = (ctypes.c_ubyte * 16)(*([0xCC] * 16))
        native.extractps_mem_op(pa, lane, ctypes.addressof(dst_n), 0)
        lifted.extractps_mem_op(pa, lane, ctypes.addressof(dst_l), 0)
        want_mem = struct.pack("<I", want_u32) + b"\xCC" * 12
        if bytes(dst_n) != want_mem or bytes(dst_l) != want_mem:
            raise AssertionError(f"extractps_mem_op(lane={lane}): expected={want_mem.hex()} native={bytes(dst_n).hex()} lifted={bytes(dst_l).hex()}")
        cases += 1

    # 8. Mutated lifted negative controls: each must mismatch the native/reference output.
    def must_mismatch(ctrl_name: str, a_bytes: bytes, b_bytes: bytes, arg3: int, want: bytes) -> None:
        ra, pa = _aligned_buf(a_bytes)
        rb, pb = _aligned_buf(b_bytes)
        out_c = (ctypes.c_ubyte * 16)()
        ctypes.memset(out_c, 0x55, 16)
        getattr(lifted, ctrl_name)(pa, pb, ctypes.addressof(out_c), arg3)
        if bytes(out_c) == want:
            raise AssertionError(f"mutated control {ctrl_name} unexpectedly matched expected {want.hex()}")

    must_mismatch("addsubps_mutated_control", f32_pairs[0][0], f32_pairs[0][1], 0,
                  ref_binary("addsubps_op", f32_pairs[0][0], f32_pairs[0][1]))
    must_mismatch("movsldup_mutated_control", f32_pairs[0][0], b"\x00" * 16, 0,
                  ref_binary("movsldup_op", f32_pairs[0][0], b""))
    # roundps_op takes mode in arg1 (rdx) and dst in arg2 (r8); invoke directly
    ra, pa = _aligned_buf(round_f32_vecs[0])
    out_c = (ctypes.c_ubyte * 16)()
    lifted.roundps_mutated_control(pa, 1, ctypes.addressof(out_c), 0)
    if bytes(out_c) == ref_binary("roundps_op", round_f32_vecs[0], b"", 1):
        raise AssertionError("mutated control roundps_mutated_control unexpectedly matched floor output")
    must_mismatch("insertps_mutated_control", ins_a, ins_b, 0x1F,
                  ref_binary("insertps_op", ins_a, ins_b, 0x1F))

    print(
        f"{cases} baseline 128-bit SSE3/SSE4.1 FP cases across {len(ALL_EXPORTS)} exports "
        f"matched native DLL and reference; {len(MUTATED_CONTROLS)} mutated controls mismatched as required"
    )


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "asm":
        Path(sys.argv[2]).write_text(generate_asm(), encoding="ascii")
    elif mode == "emit":
        emit(Path(sys.argv[2]), Path(sys.argv[3]))
    elif mode == "check":
        check(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        raise SystemExit(f"unknown mode {mode}")
