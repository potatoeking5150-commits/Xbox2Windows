"""Differential checks, single-rounding checks, NaN/edge-case checks and rejection controls for FMA3 AOT."""

import ctypes
from pathlib import Path
import random
import struct
import sys

from iced_x86 import Code, Encoder, Instruction, MemoryOperand, Register as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_avx2_aot import lift_code

FAMILIES = (
    ("vfmadd", True),
    ("vfmsub", True),
    ("vfnmadd", True),
    ("vfnmsub", True),
    ("vfmaddsub", False),
    ("vfmsubadd", False),
)

ALL_EXPORTS = []
for _fam, _has_scalar in FAMILIES:
    for _order in (132, 213, 231):
        for _sfx in ("ps", "pd"):
            ALL_EXPORTS.append((f"op_{_fam}{_order}{_sfx}_xmm", _sfx, False))
            ALL_EXPORTS.append((f"op_{_fam}{_order}{_sfx}_ymm", _sfx, False))
        if _has_scalar:
            for _sfx in ("ss", "sd"):
                ALL_EXPORTS.append((f"op_{_fam}{_order}{_sfx}", _sfx, False))

MEM_EXPORTS = (
    ("op_mem_vfmadd132ss", "ss", True),
    ("op_mem_vfmsub213sd", "sd", True),
    ("op_mem_vfnmadd231ps_xmm", "ps", True),
    ("op_mem_vfnmsub132ps_ymm", "ps", True),
    ("op_mem_vfmaddsub213pd_xmm", "pd", True),
    ("op_mem_vfmsubadd231pd_ymm", "pd", True),
)
ALL_EXPORTS.extend(MEM_EXPORTS)


def bind_fn(dll, name):
    fn = getattr(dll, name)
    fn.argtypes = (ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64)
    fn.restype = None
    return fn


def run_fn(fn, inputs_96, mem_32):
    in_buf = ctypes.create_string_buffer(inputs_96, 96)
    mem_buf = ctypes.create_string_buffer(mem_32, 32)
    out_buf = ctypes.create_string_buffer(32)
    fn(ctypes.addressof(in_buf), ctypes.addressof(mem_buf), ctypes.addressof(out_buf))
    return out_buf.raw


def pack_f32_bits(*words):
    assert len(words) == 8
    return struct.pack("<8I", *(w & 0xffffffff for w in words))


def pack_f64_bits(*words):
    assert len(words) == 4
    return struct.pack("<4Q", *(w & 0xffffffffffffffff for w in words))


def f32_bits(val):
    return struct.unpack("<I", struct.pack("<f", val))[0]


def f64_bits(val):
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def make_f32_cases():
    # Case 0: single-rounding FMA witness:
    # (1.0 + 2^-12) * (1.0 - 2^-12) - 1.0 = -2^-24 (exact with FMA, 0.0 if unfused!)
    a_sr = f32_bits(1.0 + 2.0**-12)
    b_sr = f32_bits(1.0 - 2.0**-12)
    c_sr = f32_bits(-1.0)
    pos_zero = 0x00000000
    neg_zero = 0x80000000
    pos_inf = 0x7f800000
    neg_inf = 0xff800000
    subnorm = 0x00000003
    qnan_a = 0x7fc01234
    qnan_b = 0xffc05678
    snan_c = 0x7f800042

    cases = [
        # Single-rounding witnesses + distinct values across all 8 lanes (proving order & addsub/subadd)
        (
            pack_f32_bits(a_sr, f32_bits(2.5), f32_bits(-3.25), f32_bits(0.5),
                          f32_bits(7.0), f32_bits(-0.125), f32_bits(100.5), f32_bits(-42.75)),
            pack_f32_bits(b_sr, f32_bits(-4.0), f32_bits(1.5), f32_bits(-6.0),
                          f32_bits(0.25), f32_bits(8.0), f32_bits(-0.5), f32_bits(2.0)),
            pack_f32_bits(c_sr, f32_bits(1.25), f32_bits(2.0), f32_bits(3.75),
                          f32_bits(-5.5), f32_bits(9.25), f32_bits(0.125), f32_bits(-11.0)),
        ),
        # Signed zeros, infinities, subnormals, 0*inf and inf-inf indefinite QNaN
        (
            pack_f32_bits(pos_zero, neg_zero, pos_zero, pos_inf, subnorm, f32_bits(1e20), f32_bits(-1e-20), neg_inf),
            pack_f32_bits(neg_zero, neg_zero, pos_inf, f32_bits(1.0), f32_bits(2.0), f32_bits(1e20), f32_bits(1e-20), f32_bits(2.0)),
            pack_f32_bits(neg_zero, pos_zero, f32_bits(1.0), neg_inf, subnorm, f32_bits(-1e38), pos_zero, pos_inf),
        ),
        # QNaN and SNaN priority across op0, op1, op2 (including negative QNaN sign preservation on fnmadd/fnmsub)
        (
            pack_f32_bits(qnan_a, f32_bits(1.0), f32_bits(2.0), qnan_b, snan_c, f32_bits(3.0), qnan_a, f32_bits(-5.0)),
            pack_f32_bits(qnan_b, qnan_b, f32_bits(3.0), snan_c, f32_bits(1.0), snan_c, f32_bits(4.0), qnan_a),
            pack_f32_bits(snan_c, f32_bits(4.0), snan_c, qnan_a, qnan_b, f32_bits(2.0), qnan_b, f32_bits(6.0)),
        ),
    ]
    rng = random.Random(0xF3A30032)
    for _ in range(4):
        vecs = []
        for _v in range(3):
            vals = [f32_bits(rng.uniform(-500.0, 500.0)) for _ in range(8)]
            vecs.append(pack_f32_bits(*vals))
        cases.append(tuple(vecs))
    return cases


def make_f64_cases():
    # Single-rounding double witness: (1 + 2^-26)*(1 - 2^-26) - 1 = -2^-52
    a_sr = f64_bits(1.0 + 2.0**-26)
    b_sr = f64_bits(1.0 - 2.0**-26)
    c_sr = f64_bits(-1.0)
    pos_zero = 0x0000000000000000
    neg_zero = 0x8000000000000000
    pos_inf = 0x7ff0000000000000
    neg_inf = 0xfff0000000000000
    subnorm = 0x0000000000000005
    qnan_a = 0x7ff8000012345678
    qnan_b = 0xfff8000087654321
    snan_c = 0x7ff0000000000099

    cases = [
        (
            pack_f64_bits(a_sr, f64_bits(-3.5), f64_bits(12.25), f64_bits(-0.125)),
            pack_f64_bits(b_sr, f64_bits(2.25), f64_bits(-0.5), f64_bits(16.0)),
            pack_f64_bits(c_sr, f64_bits(7.125), f64_bits(4.0), f64_bits(-9.5)),
        ),
        (
            pack_f64_bits(pos_zero, pos_inf, subnorm, neg_zero),
            pack_f64_bits(pos_inf, f64_bits(1.0), f64_bits(2.0), neg_zero),
            pack_f64_bits(f64_bits(1.0), neg_inf, subnorm, pos_zero),
        ),
        (
            pack_f64_bits(qnan_a, f64_bits(1.0), f64_bits(2.0), qnan_b),
            pack_f64_bits(qnan_b, qnan_b, f64_bits(3.0), snan_c),
            pack_f64_bits(snan_c, f64_bits(4.0), snan_c, qnan_a),
        ),
    ]
    rng = random.Random(0xF3A30064)
    for _ in range(4):
        vecs = []
        for _v in range(3):
            vals = [f64_bits(rng.uniform(-5000.0, 5000.0)) for _ in range(4)]
            vecs.append(pack_f64_bits(*vals))
        cases.append(tuple(vecs))
    return cases


def encode_seq(instructions):
    enc = Encoder(64)
    ip = 0x1000
    for ins in instructions:
        ip += enc.encode(ins, ip)
    return enc.take_buffer()


def check_rejections():
    # 1. Undefined op0 (FMA3 reads op0 as well as writing it)
    raw_undef = encode_seq([
        Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM1, MemoryOperand(R.RCX)),
        Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM2, MemoryOperand(R.RDX)),
        Instruction.create_reg_reg_reg(Code.VEX_VFMADD213PS_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM2),
        Instruction.create(Code.RETNQ),
    ])
    try:
        lift_code("bad_undef_op0", 0x1000, raw_undef)
        raise AssertionError("expected undefined op0 rejection")
    except ValueError as err:
        assert "undefined vector register" in str(err), err

    # 2. Nonvolatile YMM6 rejected
    raw_nonvol = encode_seq([
        Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM0, MemoryOperand(R.RCX)),
        Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM1, MemoryOperand(R.RDX)),
        Instruction.create_reg_reg_reg(Code.VEX_VFMADD213PS_YMM_YMM_YMMM256, R.YMM0, R.YMM1, R.YMM6),
        Instruction.create(Code.RETNQ),
    ])
    try:
        lift_code("bad_nonvol", 0x1000, raw_nonvol)
        raise AssertionError("expected nonvolatile register rejection")
    except ValueError as err:
        assert "nonvolatile" in str(err), err

    # 3. EVEX FMA3 rejected
    raw_evex = encode_seq([
        Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM0, MemoryOperand(R.RCX)),
        Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM1, MemoryOperand(R.RDX)),
        Instruction.create_reg_mem(Code.VEX_VMOVDQU_YMM_YMMM256, R.YMM2, MemoryOperand(R.R8)),
        Instruction.create_reg_reg_reg(Code.EVEX_VFMADD213PS_YMM_K1Z_YMM_YMMM256B32, R.YMM0, R.YMM1, R.YMM2),
        Instruction.create(Code.RETNQ),
    ])
    try:
        lift_code("bad_evex", 0x1000, raw_evex)
        raise AssertionError("expected EVEX rejection")
    except ValueError as err:
        assert "EVEX" in str(err), err

    # 4. Unsupported RSP memory base on FMA3 op2 rejected
    raw_rsp = encode_seq([
        Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX)),
        Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM1, MemoryOperand(R.RDX)),
        Instruction.create_reg_reg_mem(Code.VEX_VFMADD132SS_XMM_XMM_XMMM32, R.XMM0, R.XMM1, MemoryOperand(R.RSP)),
        Instruction.create(Code.RETNQ),
    ])
    try:
        lift_code("bad_rsp", 0x1000, raw_rsp)
        raise AssertionError("expected RSP memory rejection")
    except ValueError as err:
        assert "unsupported memory operand" in str(err), err

    # 5. AMD FMA4 4-operand instruction rejected
    raw_fma4 = encode_seq([
        Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM0, MemoryOperand(R.RCX)),
        Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM1, MemoryOperand(R.RDX)),
        Instruction.create_reg_mem(Code.VEX_VMOVDQU_XMM_XMMM128, R.XMM2, MemoryOperand(R.R8)),
        Instruction.create_reg_reg_reg_reg(Code.VEX_VFMADDPS_XMM_XMM_XMMM128_XMM, R.XMM0, R.XMM1, R.XMM2, R.XMM0),
        Instruction.create(Code.RETNQ),
    ])
    try:
        lift_code("bad_fma4", 0x1000, raw_fma4)
        raise AssertionError("expected FMA4 rejection")
    except ValueError as err:
        assert "unsupported instruction" in str(err), err


def main():
    capability_dll = ctypes.CDLL(sys.argv[4])
    capability_dll.host_has_fma.argtypes = []
    capability_dll.host_has_fma.restype = ctypes.c_int
    if not capability_dll.host_has_fma():
        print("FMA and OS-enabled XMM/YMM state are required for the native fixture")
        return 77

    orig_dll = ctypes.CDLL(sys.argv[1])
    lifted_dll = ctypes.CDLL(sys.argv[2])
    altered_dll = ctypes.CDLL(sys.argv[3])

    f32_cases = make_f32_cases()
    f64_cases = make_f64_cases()

    checked = 0
    for name, sfx, is_mem in ALL_EXPORTS:
        native = bind_fn(orig_dll, name)
        lifted = bind_fn(lifted_dll, f"lifted_{name}")
        cases = f32_cases if sfx in ("ps", "ss") else f64_cases
        for ci, (op0, op1, op2) in enumerate(cases):
            inputs_96 = op0 + op1 + (b"\x00" * 32 if is_mem else op2)
            mem_32 = op2 if is_mem else b"\x00" * 32
            got_native = run_fn(native, inputs_96, mem_32)
            got_lifted = run_fn(lifted, inputs_96, mem_32)
            if got_native != got_lifted:
                raise AssertionError(
                    f"{name} case {ci} mismatch:\n"
                    f"  native={got_native.hex()}\n"
                    f"  lifted={got_lifted.hex()}"
                )
            checked += 1

    # Single-rounding witness verification on op_vfmadd132ps_xmm and op_vfmadd132pd_xmm:
    # (1 + 2^-12)*(1 - 2^-12) + (-1.0) MUST equal exact -2^-24 (not 0.0)
    a32, b32, c32 = f32_cases[0]
    res32 = run_fn(bind_fn(lifted_dll, "lifted_op_vfmadd132ps_xmm"), a32 + c32 + b32, b"\x00" * 32)
    lane0_f32 = struct.unpack_from("<f", res32, 0)[0]
    assert lane0_f32 == -(2.0**-24), f"expected exact fused -2^-24, got {lane0_f32!r}"
    assert res32[16:] == b"\x00" * 16, "XMM VEX write must zero upper 128 bits"

    a64, b64, c64 = f64_cases[0]
    res64 = run_fn(bind_fn(lifted_dll, "lifted_op_vfmadd132pd_xmm"), a64 + c64 + b64, b"\x00" * 32)
    lane0_f64 = struct.unpack_from("<d", res64, 0)[0]
    assert lane0_f64 == -(2.0**-52), f"expected exact fused -2^-52, got {lane0_f64!r}"
    assert res64[16:] == b"\x00" * 16, "XMM VEX write must zero upper 128 bits"

    # Scalar upper-lane preservation + upper-128 zeroing check
    res_ss = run_fn(bind_fn(lifted_dll, "lifted_op_vfmadd213ss"), a32 + b32 + c32, b"\x00" * 32)
    assert res_ss[4:16] == a32[4:16], "scalar ss must preserve op0 bits 32..127"
    assert res_ss[16:] == b"\x00" * 16, "scalar ss must zero bits 128..255"

    res_sd = run_fn(bind_fn(lifted_dll, "lifted_op_vfmadd213sd"), a64 + b64 + c64, b"\x00" * 32)
    assert res_sd[8:16] == a64[8:16], "scalar sd must preserve op0 bits 64..127"
    assert res_sd[16:] == b"\x00" * 16, "scalar sd must zero bits 128..255"

    # Mutation negative control: ALTER_FMA changes op_vfmadd213ps_xmm to vfmsub213ps
    got_orig = run_fn(bind_fn(orig_dll, "op_vfmadd213ps_xmm"), a32 + b32 + c32, b"\x00" * 32)
    got_alt = run_fn(bind_fn(altered_dll, "lifted_op_vfmadd213ps_xmm"), a32 + b32 + c32, b"\x00" * 32)
    assert got_orig != got_alt, "altered FMA variant must differ from original"

    check_rejections()
    print(f"cpu_aot_fma3 passed ({len(ALL_EXPORTS)} exports, {checked} bit-exact differential evaluations)")


if __name__ == "__main__":
    sys.exit(main())
