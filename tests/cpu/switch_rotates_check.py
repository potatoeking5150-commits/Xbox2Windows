"""Native/AOT differential checks and rejection controls for ROL and ROR in cpu_switch_aot.py."""

import ctypes
from pathlib import Path
import random
import re
import sys

from iced_x86 import Decoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_switch_aot import Lifter, translate
from direct_check import GUARDS, MASK, SENTINELS, input_pairs, observe, register_probe

M32 = (1 << 32) - 1
WIDTHS = {
    "rot_reg_cl_8": 8,
    "rot_reg_cl_16": 8,
    "rot_reg_cl_32": 8,
    "rot_reg_cl_64": 8,
    "rot_imm_1": 8,
    "rot_imm_edge": 8,
    "rot_mem_cl_8_16": 8,
    "rot_mem_cl_32_64": 8,
    "rot_mem_imm": 8,
    "rot_narrow_ret": 1,
}

REJECTIONS = (
    ("neg_rol_jcc", "unsupported flag producer"),
    ("neg_ror_jcc", "unsupported flag producer"),
    ("neg_rol_setcc", "unsupported flag producer"),
    ("neg_ror_setcc", "unsupported flag producer"),
    ("neg_rol_cmovcc", "unsupported flag producer"),
    ("neg_ror_cmovcc", "unsupported flag producer"),
    ("neg_cmp_rol_setcc", "unsupported flag producer"),
    ("neg_cmp_ror_setcc", "unsupported flag producer"),
    ("neg_rol_undef", "undefined register rax"),
    ("neg_rol_ah", "unsupported register"),
)


def rol(value, bits, count):
    mask = (1 << bits) - 1
    count &= bits - 1
    value &= mask
    if count == 0:
        return value
    return ((value << count) | (value >> (bits - count))) & mask


def ror(value, bits, count):
    mask = (1 << bits) - 1
    count &= bits - 1
    value &= mask
    if count == 0:
        return value
    return ((value >> count) | (value << (bits - count))) & mask


def merge(register, value, width):
    low = (1 << width * 8) - 1
    return (register & ~low | value & low) & MASK


def reference(name, x, y, caller_rax):
    trace = [MASK] * 3
    if name == "rot_reg_cl_8":
        rax = merge(y, rol(y, 8, x), 1)
        trace = [rax, merge(y, ror(y, 8, x), 1), y]
    elif name == "rot_reg_cl_16":
        rax = merge(y, rol(y, 16, x), 2)
        trace = [rax, merge(y, ror(y, 16, x), 2), y]
    elif name == "rot_reg_cl_32":
        rax = rol(y, 32, x)
        trace = [rax, ror(y, 32, x), y & M32]
    elif name == "rot_reg_cl_64":
        rax = rol(y, 64, x)
        trace = [rax, ror(y, 64, x), y]
    elif name == "rot_imm_1":
        t0 = merge(x, rol(x, 8, 1), 1)
        t0 = merge(t0, ror(t0, 16, 1), 2)
        trace = [t0, rol(x, 32, 1), ror(y, 64, 1)]
        rax = rol(y, 64, 1)
    elif name == "rot_imm_edge":
        t0 = merge(x, ror(x, 8, 11), 1)
        t1 = merge(y, rol(y, 16, 19), 2)
        t2 = ror(rol(y, 32, 0), 32, 37)
        trace = [t0, t1, t2]
        rax = ror(rol(x, 64, 69), 64, 0)
    elif name == "rot_mem_cl_8_16":
        trace = [merge(y, rol(y, 8, x), 1), merge(y, ror(y, 8, 11), 1), merge(y, rol(y, 16, x), 2)]
        rax = merge(y, ror(y, 16, 1), 2)
    elif name == "rot_mem_cl_32_64":
        q2 = ror(rol(y, 64, x), 64, 1)
        trace = [merge(y, ror(y, 32, x), 4), merge(y, rol(y, 32, 37), 4), q2]
        rax = q2
    elif name == "rot_mem_imm":
        trace = [merge(y, rol(y, 8, 1), 1), merge(y, ror(y, 16, 19), 2), y]
        rax = ror(x, 64, 13)
    elif name == "rot_narrow_ret":
        rax = merge(caller_rax, rol(y, 8, x), 1)
    else:
        raise AssertionError(f"unknown export {name}")
    return rax, (GUARDS[0], *trace, GUARDS[1])


def cl_inputs():
    counts = sorted({0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 0x80, 0xff})
    values = (
        0,
        1,
        0x80,
        0x81,
        0xff,
        0x7fff,
        0x8000,
        0x8001,
        0xffff,
        0x1234,
        0xc3a5,
        0x7fffffff,
        0x80000000,
        0x80000001,
        0xffffffff,
        0x123456789abcdef0,
        0xdeadbeefcafebabe,
        0x8000000000000000,
        0x8000000000000001,
        MASK,
    )
    highs = (0, 0xdeadbeef00000000, 0xffffff00)
    pairs = [(c | high, v) for c in counts for high in highs for v in values]
    rng = random.Random(0x524F54)
    pairs.extend((rng.getrandbits(64), rng.getrandbits(64)) for _ in range(128))
    return pairs


def check_generated(original_path):
    source, functions = translate(original_path, list(WIDTHS))
    widths = {name: fn.return_width for name, fn in zip(WIDTHS, functions)}
    if widths != WIDTHS:
        raise AssertionError(f"return widths {widths} != {WIDTHS}")
    if "rotate_bits" not in source:
        raise AssertionError("expected rotate_bits helper in generated source")
    for name, expected_loads in (("rot_mem_cl_8_16", 3), ("rot_mem_cl_32_64", 5), ("rot_mem_imm", 3)):
        marker = f"lifted_{name}("
        body = source.split(marker, 1)[1].split("\n}", 1)[0]
        if body.count("load_always<") != expected_loads or re.search(r"\bload<", body):
            raise AssertionError(f"unexpected load count in {name}:\n{body}")


def check_rejections(original_path):
    for name, reason in REJECTIONS:
        try:
            translate(original_path, [name])
        except ValueError as exc:
            if reason not in str(exc):
                raise AssertionError(f"{name}: expected {reason!r}, got {exc}") from exc
            print(f"rejected {name}: {exc}")
        else:
            raise AssertionError(f"accepted negative control {name}")

    # Direct Lifter.lower(..., sets_flags=True) must also refuse ROL/ROR.
    for mnemonic_name, raw in (("rol eax, 1", "d1 c0"), ("ror eax, 1", "d1 c8")):
        ins = next(Decoder(64, bytes.fromhex(raw), ip=0x1000))
        try:
            Lifter().lower(ins, sets_flags=True)
        except ValueError as exc:
            if "unsupported flag producer" not in str(exc):
                raise AssertionError(f"{mnemonic_name} sets_flags: {exc}") from exc
        else:
            raise AssertionError(f"accepted {mnemonic_name} with sets_flags=True")
    print(f"{len(REJECTIONS) + 2} rejection controls passed")


def check(probe_path, original_path, lifted_path):
    check_generated(original_path)
    probe = register_probe(probe_path)
    native, lifted = ctypes.CDLL(original_path), ctypes.CDLL(lifted_path)
    cl_pairs = cl_inputs()
    imm_pairs = input_pairs()
    total = 0
    for name, width in WIDTHS.items():
        source, translation = getattr(native, name), getattr(lifted, "lifted_" + name)
        caller_rax = ctypes.cast(source, ctypes.c_void_p).value
        low = (1 << width * 8) - 1
        pairs = cl_pairs if "_cl_" in name or name == "rot_narrow_ret" else imm_pairs
        for x, y in pairs:
            want, trace = reference(name, x, y, caller_rax)
            original, actual = observe(probe, source, x, y), observe(probe, translation, x, y)
            if original != (want, trace, SENTINELS):
                raise AssertionError(f"original {name}({x:#x}, {y:#x}): {original} != {want:#x} {trace}")
            if (actual[0] & low, *actual[1:]) != (want & low, trace, SENTINELS):
                raise AssertionError(f"lifted_{name}({x:#x}, {y:#x}): {actual} != {want:#x} masked {low:#x}")
            total += 1
        print(f"{name}: {len(pairs)} inputs match native and reference")
    print(f"{total} differential comparisons passed across 8/16/32/64-bit reg/mem ROL/ROR")
    check_rejections(original_path)


if __name__ == "__main__":
    check(*sys.argv[1:])
