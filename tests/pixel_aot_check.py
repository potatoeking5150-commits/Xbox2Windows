"""Exercise offline pixel translation with independently assembled test programs."""
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import tempfile
import zlib
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import pixel_aot
from pixel_aot import ALU, disassemble, native_program

# First word of each line of every program lower() accepted in this run. tools/form_coverage.py counts
# these when this test passes; a program a check expects to be rejected raises before it is recorded.
ACCEPTED = set()
SOPC_OPS = (
    "s_cmp_eq_i32", "s_cmp_lg_i32", "s_cmp_gt_i32", "s_cmp_ge_i32", "s_cmp_lt_i32", "s_cmp_le_i32",
    "s_cmp_eq_u32", "s_cmp_lg_u32", "s_cmp_gt_u32", "s_cmp_ge_u32", "s_cmp_lt_u32", "s_cmp_le_u32",
    "s_cmp_eq_u64", "s_cmp_lg_u64",
    "s_bitcmp0_b32", "s_bitcmp1_b32",
    "s_bitcmp0_b64", "s_bitcmp1_b64",
)
SOP2_LSHL_ADD_OPS = (
    "s_lshl1_add_u32", "s_lshl2_add_u32", "s_lshl3_add_u32", "s_lshl4_add_u32",
)
SOP2_ABSDIFF_OPS = (
    "s_absdiff_i32",
)
VOP2_INT_OPS = (
    "v_lshrrev_b32_e32", "v_ashrrev_i32_e32", "v_lshlrev_b32_e32",
    "v_and_b32_e32", "v_or_b32_e32", "v_xor_b32_e32",
    "v_add_nc_u32_e32", "v_sub_nc_u32_e32", "v_subrev_nc_u32_e32",
)
VOP3_ALIGN_OPS = (
    "v_alignbit_b32", "v_alignbyte_b32",
)
VOP3_TERNARY_INT_OPS = (
    "v_add3_u32", "v_or3_b32", "v_xor3_b32", "v_and_or_b32",
    "v_xad_u32", "v_lshl_or_b32", "v_lshl_add_u32", "v_add_lshl_u32",
)
# The 32-bit no-carry forms of nc_integer_group(1..3), as ACCEPTED records them (first word of the line).
NC_INT_OPS = (
    "v_add_nc_u32_e32", "v_sub_nc_u32_e32", "v_subrev_nc_u32_e32",
    "v_add_nc_u32_e64", "v_sub_nc_u32_e64", "v_subrev_nc_u32_e64",
    "v_add_nc_i32", "v_sub_nc_i32",
)


def lower(instructions, *args, **kwargs):
    source = pixel_aot.lower(instructions, *args, **kwargs)
    ACCEPTED.update((line[1] if isinstance(line, tuple) else line).split()[0]
                    for line in instructions
                    if (line[1] if isinstance(line, tuple) else line).strip())
    return source


def write_record(path):
    """Save ACCEPTED where CTest's FORM_COVERAGE_RECORD points; nothing is written without it."""
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("".join(f"{word}\n" for word in sorted(ACCEPTED)), encoding="utf-8")


RAMP = (0, 16, 37, 64, 101, 153, 219, 254)

# CPU references use plain Python floats, which keep binary32 subnormals: the fixtures assume the
# compiled desktop shader preserves them, and DXC runs with a matching -denorm flag. The Scarlett
# pipeline's FLOAT_MODE is unmeasured, so this records the harness's assumption rather than a claim
# about the console; binary16 and binary64 denormal behavior is established separately, not here.
F32_DENORM_MODE = "preserve"
# Binary16 subnormals are preserved by construction, whatever -denorm says. Packed f16 operations
# widen each half from its encoding (widen16), compute in binary32 and narrow in integer arithmetic
# (pack16_rne). Every binary16 subnormal and every product or sum of two of them is a binary32
# normal, so a binary32 flush never reaches them, and DXIL records no f16 denormal mode to request.
# The Python references keep binary16 subnormals too. Like the f32 mode, this is the harness's
# recorded assumption, not a claim about the Scarlett FLOAT_MODE f16 bits.
F16_DENORM_MODE = "preserve"

# Integer-to-binary32 references convert through Python's binary64 float, which holds every 32-bit
# integer exactly, then pack to binary32, rounding ties to even. The Scarlett pipeline's MODE.FP_ROUND
# is unmeasured, so this records the harness's assumption the same way F32_DENORM_MODE records its
# denormal assumption: an adapter disagreeing on a row would be recorded as a mismatch, not chosen.
F32_ROUND_ASSUMPTION = "nearest-even"

# Binary32-to-word references follow the issue #360 reading of the RDNA 2 ISA: truncate toward zero, NaN
# gives 0, out-of-range values saturate. That reading has not been measured on the console, and
# MODE.FP_ROUND is assumed not to affect truncation.
F32_TRUNC_ASSUMPTION = "toward-zero, NaN->0, saturating"


def u32f(word):
    """Reference for v_cvt_f32_u32_e32: read the word as unsigned and round to binary32."""
    return struct.unpack("<f", struct.pack("<f", float(word)))[0]


def i32f(word):
    """Reference for v_cvt_f32_i32_e32: sign-extend the word, then convert as u32f does."""
    return u32f(word - 2**32 if word & 0x80000000 else word)


def assembly(swap):
    red, blue = ("v14", "v12") if swap else ("v12", "v14")
    return [
        "s_mov_b64 s[50:51], exec",
        "s_wqm_b64 exec, exec",
        "s_mov_b32 m0, s24",
        "v_interp_p1_f32_e32 v8, v0, attr0.x",
        "v_interp_p1_f32_e32 v9, v0, attr0.y",
        "v_interp_p2_f32_e32 v9, v1, attr0.y",
        "v_interp_p2_f32_e32 v8, v1, attr0.x",
        "s_and_b64 exec, exec, s[50:51]",
        "s_mov_b32 s21, 0",
        "s_load_dwordx8 s[28:35], s[20:21], null",
        "s_mov_b32 s40, 1",
        "s_mov_b32 s41, 2",
        "s_mov_b32 s42, 3",
        "s_mov_b32 s43, 4",
        "s_waitcnt lgkmcnt(0)",
        "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
        "s_waitcnt vmcnt(0)",
        f"v_cvt_pkrtz_f16_f32_e32 v16, {red}, v13",
        f"v_cvt_pkrtz_f16_f32_e32 v17, {blue}, v15",
        "exp mrt0 v16, v16, v17, v17 done compr vm",
        "s_endpgm",
    ]


def rgb_sample() -> list[str]:
    program = assembly(False)
    sample = next(i for i, line in enumerate(program) if line.startswith("image_sample"))
    program[sample] = program[sample].replace("v[12:15]", "v[12:14]").replace("dmask:0xf", "dmask:0x7")
    program.insert(sample, "v_mov_b32_e32 v15, 0.5")
    return program


def lod_zero_sample() -> list[str]:
    mask_ops = ("s_mov_b64", "s_wqm_b64", "s_and_b64")
    return [line.replace("image_sample ", "image_sample_lz ") for line in assembly(False)
            if not line.startswith(mask_ops)]


DUAL_CASES = ("forward", "reverse", "reuse", "partial")


def dual_program(case="forward", *, swap_weights=False):
    program = assembly(False)
    first = next(i for i, line in enumerate(program) if line.startswith("image_sample"))
    sample_a = program[first]
    sample_b = sample_a.replace("v[12:15]", "v[20:23]").replace("s[28:35]", "s[56:63]")
    load_b = "s_load_dwordx8 s[56:63], s[20:21], 0x20"
    if case == "reuse":
        # Finish A before reloading the same descriptor registers for B.
        program[first:first+1] = [sample_a, "s_waitcnt vmcnt(0)",
                                  load_b.replace("56:63", "28:35"), "s_waitcnt lgkmcnt(0)",
                                  sample_b.replace("56:63", "28:35")]
    elif case in ("forward", "reverse", "partial"):
        program[first:first+1] = [sample_a, sample_b] if case != "reverse" else [sample_b, sample_a]
        program.insert(program.index("s_mov_b32 s40, 1"), load_b)
    else:
        raise ValueError(f"Unknown dual-texture case: {case}")
    pos = program.index("v_cvt_pkrtz_f16_f32_e32 v16, v12, v13")
    a, b = ("0x3e800000", "0x3f400000") if swap_weights else ("0x3f400000", "0x3e800000")
    ops = []
    for i in range(4):
        ops.extend([f"v_mul_legacy_f32_e32 v{12+i}, {a}, v{12+i}",
                    f"v_fma_f32 v{12+i}, {b}, v{20+i}, v{12+i}"])
    if case == "partial":
        # Scale A's red channel while B is still outstanding.
        wait = program.index("s_waitcnt vmcnt(0)")
        first_mul = ops.pop(0).replace(", v12", ", v24")
        program[wait:wait] = ["s_waitcnt vmcnt(1)", "v_mov_b32_e32 v24, v12", first_mul]
        pos += 3
    program[pos:pos] = ops
    return program


def translate_dual(instructions):
    return lower(instructions, 20, 24, [1, 2, 3, 4], second_texture_offset=32)


def shader_record(blob: bytes, name: bytes) -> bytes:
    encoder = zlib.compressobj(wbits=-15)
    payload = encoder.compress(blob) + encoder.flush()
    return (bytes([len(name) + 1]) + name + b"\0\0"
            + struct.pack("<4I", len(payload), 0, len(payload), len(blob)) + payload)


def envelope(code: bytes, name: bytes = b"fixture", *, private_data: bytes | None = None, directory_end=None,
             compute_width=None, mesh_limits=None) -> bytes:
    program_start = 288 if directory_end is not None else 256
    image = bytearray(program_start) + code
    image[:4] = b"FSSM"
    struct.pack_into("<I", image, 192, program_start - 40)
    if directory_end is not None:
        extent, end_field = directory_end
        struct.pack_into("<I", image, 4, 0x101)
        struct.pack_into("<Q", image, 200, extent)
        struct.pack_into("<Q", image, 232, end_field)
    native = b"XBCH" + bytes(12) + struct.pack("<II", len(image), 0) + zlib.compress(image)
    program = (bytes(24) + b"XDAT" + struct.pack("<II", 1, 0)
               + struct.pack("<III", 13, 48, len(native)) + native)
    if compute_width is not None:
        program = program[:2] + struct.pack("<H", 13 if mesh_limits else 5) + program[4:]
    sections = [b"DXIL" + struct.pack("<I", len(program)) + program]
    if compute_width is not None:
        psv = bytearray(52)
        struct.pack_into("<I", psv, 0, 48)
        psv[28] = 13 if mesh_limits else 5
        if mesh_limits:
            psv[31] = 2
            struct.pack_into("<HH", psv, 16, *mesh_limits)
        struct.pack_into("<3I", psv, 40, compute_width, 1, 1)
        sections.append(b"PSV0" + struct.pack("<I", len(psv)) + psv)
    if private_data is not None:
        sections.append(b"PRIV" + struct.pack("<I", len(private_data)) + private_data)
    offset = 32 + 4 * len(sections)
    offsets = bytearray()
    for section in sections:
        offsets.extend(struct.pack("<I", offset))
        offset += len(section)
    blob = b"DXBC" + bytes(16) + struct.pack("<3I", 1, offset, len(sections)) + offsets + b"".join(sections)
    return shader_record(blob, name)


def must_reject(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (ValueError, KeyError):
        return
    raise AssertionError(f"{function.__name__} accepted invalid input")


def expect_value_error(function, program, message):
    """Require a ValueError carrying message, so a rejection cannot pass for an unrelated reason."""
    try:
        function(program)
    except ValueError as error:
        assert message in str(error), error
    else:
        raise AssertionError(f"Expected rejection: {message}")


def compile_shader(compiler, source, shader, *, listing=None, mode=F32_DENORM_MODE, ieee=True):
    """Run DXC on one translated shader under an explicit binary32 denormal mode."""
    # DXC accepts -denorm only at shader model 6.2 and above; ps_6_2 is the minimum target.
    command = [compiler, *(["-Gis"] if ieee else []), "-denorm", mode,
               "-T", "ps_6_2", "-E", "main", "-Fo", shader]
    if listing is not None:
        command += ["-Fc", listing]
    command.append(source)
    subprocess.run(command, check=True)


def flush_ftz(value):
    """Flush one binary32 subnormal input or result to signed zero, as -denorm ftz would."""
    bits, = struct.unpack("<I", struct.pack("<f", value))
    if bits & 0x7f800000 == 0 and bits & 0x007fffff:
        return struct.unpack("<f", struct.pack("<I", bits & 0x80000000))[0]
    return value


# The NVIDIA driver (RTX 5080, drivers 32.0.16.2101 and 32.0.16.2124) refuses PSO creation for a pixel shader
# that uses log2 under -denorm preserve (E_OUTOFMEMORY, issue #828). The approximate programs that contain
# log2 record their mode as any, and assert that no input, intermediate or expected value is a binary32
# subnormal, so the mode cannot change what they compute or export. Every other program keeps preserve.
LOG2_DENORM_MODE = "any"


def log2_denorm_mode(label, hlsl, *values):
    """Mode for one translated program: any exactly when it contains log2, after checking values for subnormals."""
    if "log2(" not in hlsl:
        return F32_DENORM_MODE
    for value in values:
        assert flush_ftz(value) == value, (label, value)
    return LOG2_DENORM_MODE


def write_case(path, constants, pixels, *, nan_mask=None, integer_rt=False, extra_constants=None,
               title_memory=None, write_mask=None):
    """Write a renderer case: 256 constant bytes, then 64 expected float4 rows in row-major order.

    A single float4 broadcasts to every texel. The optional trailer appends one flags byte, whose
    bit 0 selects the lossless R32G32B32A32_UINT readback target. Floating-point cases may instead
    append a zero flags byte and 64 mask bytes for per-channel NaN-class checks. Cases written without
    integer_rt or nan_mask keep the original 1280-byte format exactly. extra_constants sets bit 3 of
    the flags byte and stores 512 bytes after it: the 256-byte buffers bound to b1 and b2.
    Bit 2 carries the title ranges and aliased buffers before any extra constants.
    The high nibble can hold an explicit RT0 mask for a floating-point case.
    """
    rows = [pixels] * 64 if len(pixels) == 4 and not isinstance(pixels[0], (list, tuple)) else list(pixels)
    if len(rows) != 64 or any(len(row) != 4 for row in rows):
        raise ValueError("A case needs 64 float4 expectations")
    data = bytearray(bytes(constants))
    for row in rows:
        data += struct.pack("<4f", *row)
    if write_mask is not None:
        if (not 0 < write_mask <= 0xF or integer_rt or
                any(item is not None for item in (nan_mask, extra_constants, title_memory))):
            raise ValueError("An explicit write mask requires a nonzero nibble and a plain floating-point render target")
        data.append(write_mask << 4)
    else:
        if extra_constants is not None and (integer_rt or nan_mask is not None or title_memory is not None):
            raise ValueError("Extra constant buffers are not combined with other trailer fields")
        if extra_constants is not None and len(extra_constants) != 512:
            raise ValueError("The b1 and b2 buffers need exactly 512 bytes")
        if integer_rt and nan_mask is not None:
            raise ValueError("NaN-class masks require a floating-point render target")
        flags = int(integer_rt) | (4 if title_memory is not None else 0) | (8 if extra_constants is not None else 0)
        if flags or nan_mask is not None:
            data.append(flags)
            if title_memory is not None:
                data += title_memory
            if extra_constants is not None:
                data += extra_constants
            if nan_mask is not None:
                if isinstance(nan_mask, int):
                    nan_mask = bytes([nan_mask]) * 64
                elif len(nan_mask) != 64:
                    raise ValueError("The NaN-class mask needs exactly 64 texel bytes")
                data += nan_mask
    Path(path).write_bytes(data)
    return Path(path)


def check_container_selector(directory: Path, code: bytes, library: str | None = None) -> None:
    archive = directory / "same-name.rpf"
    records = [envelope(code, private_data=metadata) for metadata in (b"first", b"second")]
    containers = []
    for record in records:
        archive.write_bytes(record)
        container, native = native_program(archive, "fixture")
        assert native == code
        containers.append(container)
    digests = [hashlib.sha256(blob).hexdigest() for blob in containers]
    first, second = records
    broken = b"\x08fixture\0\0" + struct.pack("<4I", 2, 0, 2, 8) + b"\x07\xff"
    archive.write_bytes(first + second)
    must_reject(native_program, archive, "fixture")
    for digest, blob in zip(digests, containers, strict=True):
        assert native_program(archive, "fixture", container_sha256=digest.upper()) == (blob, code)

    for order in (broken + first, first + broken):
        archive.write_bytes(order)
        assert native_program(archive, "fixture", container_sha256=digests[0]) == (containers[0], code)
    # A bad frame can claim bytes covering a subsequent record of the same name.
    archive.write_bytes(b"\x08fixture\0\0" + struct.pack("<4I", len(first) + 2, 0, len(first) + 2, 8)
                        + b"\x07\xff" + first)
    assert native_program(archive, "fixture", container_sha256=digests[0]) == (containers[0], code)
    archive.write_bytes(first * 257 + second)
    assert native_program(archive, "fixture", container_sha256=digests[1]) == (containers[1], code)

    archive.write_bytes(envelope(code))
    intact, _ = native_program(archive, "fixture")
    corrupt = intact[:-1] + bytes([intact[-1] ^ 1])
    archive.write_bytes(first + shader_record(corrupt, b"fixture"))
    try:
        native_program(archive, "fixture", container_sha256=hashlib.sha256(corrupt).hexdigest())
    except zlib.error as error:
        assert "incorrect data check" in str(error), error
    else:
        raise AssertionError("Container selection hid a corrupt native stream")

    for contents, digest, reason in (
        (first + second, hashlib.sha256(code).hexdigest(), "No record matches the name and container SHA-256"),
        (first + second, "bad", "Invalid decoded container SHA-256"),
        (first + broken + first + second, digests[0], "Multiple records match the name and container SHA-256"),
    ):
        archive.write_bytes(contents)
        try:
            native_program(archive, "fixture", container_sha256=digest)
        except ValueError as error:
            assert str(error) == reason, error
        else:
            raise AssertionError("Container selection accepted an invalid digest or ambiguous record")

    if library is None:
        return
    archive.write_bytes(first + second)
    output = directory / "selected.hlsl"
    command = [sys.executable, "-B", Path(pixel_aot.__file__), archive, "fixture", "--llvm", library,
               "--texture-sgpr", "20", "--interp-sgpr", "24", "--sampler-words", "1", "2", "3", "4",
               "--output", output]
    ambiguous = subprocess.run(command, capture_output=True, text=True)
    assert ambiguous.returncode == 1 and "Multiple records match the name" in ambiguous.stderr
    for digest in digests:
        result = subprocess.run([*command, "--container-sha256", digest], capture_output=True, text=True, check=True)
        summary = json.loads(result.stdout)
        assert summary["container_sha256"] == digest
        assert summary["native_sha256"] == hashlib.sha256(code).hexdigest()
        assert output.read_text(encoding="utf-8") == lower(assembly(False), 20, 24, [1, 2, 3, 4])
    wrong = subprocess.run([*command, "--container-sha256", "0" * 64], capture_output=True, text=True)
    assert wrong.returncode == 1 and "No record matches the name and container SHA-256" in wrong.stderr


def translate(instructions):
    return lower(instructions, 20, 24, [1, 2, 3, 4], constant_descriptor=64)


def translate_masked(instructions):
    return lower(instructions, 20, 24, [1, 2, 3, 4], constant_descriptor=64, return_write_mask=True)


def translate_int(instructions):
    """translate() for a program that exports through the bit-exact R32G32B32A32_UINT target."""
    return lower(instructions, 20, 24, [1, 2, 3, 4], constant_descriptor=64, integer_export=True)


def with_constants(body):
    """Load constant words 3 to 6 into s44 to s47, then run body on the sample in v12 to v15."""
    program = assembly(False)
    pos = program.index("s_mov_b32 s40, 1")
    program.insert(pos, "s_load_dwordx4 s[44:47], s[20:21], 0x40")
    pos = program.index("s_waitcnt vmcnt(0)")
    program[pos:pos] = [
        "s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc",
        "s_waitcnt lgkmcnt(0)",
    ]
    pos = program.index("v_cvt_pkrtz_f16_f32_e32 v16, v12, v13")
    program[pos:pos] = body
    return program


def compressed_export(mask):
    program = with_constants([f"v_sub_f32_e64 v{12 + i}, s{44 + i}, 0" for i in range(4)])
    pos = program.index("exp mrt0 v16, v16, v17, v17 done compr vm")
    program[pos] = {
        0x3: "exp mrt0 v16, v16, off, off done compr vm",
        0xC: "exp mrt0 off, off, v17, v17 done compr vm",
        0xF: "exp mrt0 v16, v16, v17, v17 done compr vm",
    }[mask]
    return program


# s_movk_i32 sign-extends its 16-bit immediate. s_bfe_u32 takes bytes 1 and 3 of each result, and each byte is
# widened as a binary16 subnormal and scaled by 2^24 to the exact integer: 0x8000 gives 0xffff8000 (bytes 0x80 and
# 0xff) and 0x7fff gives 0x00007fff (bytes 0x7f and 0x00). Zero extension would export 0 for byte 3 of 0x8000.
MOVK_BODY = [
    "s_movk_i32 s48, 0x8000",
    "s_movk_i32 s51, 0x7fff",
    "s_bfe_u32 s49, s48, 0x80018",
    "s_bfe_u32 s50, s48, 0x80008",
    "s_bfe_u32 s52, s51, 0x80008",
    "s_bfe_u32 s53, s51, 0x80018",
    *[line for channel, source in enumerate(("s49", "s50", "s52", "s53"))
      for line in (f"v_cvt_f32_f16_e32 v{12 + channel}, {source}", f"v_mul_f32_e32 v{12 + channel}, 0x4b800000, v{12 + channel}")],
]
MOVK_EXPECTED = [255.0, 128.0, 127.0, 0.0]


def check_movk_draw(directory, compiler, renderer, backend):
    source = directory / "movk.hlsl"
    shader = directory / "movk.cso"
    words = [(imm - 0x10000 if imm & 0x8000 else imm) & 0xffffffff for imm in (0x8000, 0x7fff)]
    assert [float((words[0] >> 24) & 0xff), float((words[0] >> 8) & 0xff),
            float((words[1] >> 8) & 0xff), float((words[1] >> 24) & 0xff)] == MOVK_EXPECTED
    source.write_text(translate(with_constants(MOVK_BODY)), encoding="ascii")
    compile_shader(compiler, source, shader)
    constants = bytes(256)
    cases = []
    # Zero extension clears byte 3 of 0x8000; sign-extending 0x7fff as negative sets byte 3.
    for channel, value in ((0, 0.0), (3, 255.0)):
        wrong = list(MOVK_EXPECTED)
        wrong[channel] = value
        path = directory / f"movk-wrong-{channel}.case"
        write_case(path, constants, tuple(wrong))
        cases.append((path, "mismatch"))
    pass_case = directory / "movk-pass.case"
    write_case(pass_case, constants, tuple(MOVK_EXPECTED))
    cases.append((pass_case, "pass"))
    run_pixel_batch(renderer, backend, shader, cases)


def uncompressed_export(body, mask=0xF):
    """Run body on the sample, then export the enabled v12 to v15 binary32 channels."""
    program = with_constants(body)
    pos = program.index("v_cvt_pkrtz_f16_f32_e32 v16, v12, v13")
    operands = [f"v{12 + ch}" if mask & (1 << ch) else "off" for ch in range(4)]
    program[pos:] = [f"exp mrt0 {', '.join(operands)} done vm", "s_endpgm"]
    return program


def masked_region_body():
    return [
        "v_cmp_lt_f32_e64 vcc_lo, v12, s44",
        "s_and_saveexec_b64 s[60:61], vcc",
        "v_mov_b32_e32 v13, s45",
        "v_mov_b32_e32 v14, s46",
        "s_or_b64 exec, exec, s[60:61]",
    ]


def masked_region():
    """A single forward masked-if: while sample.x is below s44, overwrite v13 and v14."""
    return uncompressed_export(masked_region_body())


def masked_complementary():
    """Two adjacent regions on complementary predicates, so exactly one writes v13 per lane."""
    return uncompressed_export([
        "v_cmp_lt_f32_e64 vcc_lo, v12, s44",
        "s_and_saveexec_b64 s[60:61], vcc",
        "v_mov_b32_e32 v13, s45",
        "s_or_b64 exec, exec, s[60:61]",
        "v_cmp_nlt_f32_e64 vcc_lo, v12, s44",
        "s_and_saveexec_b64 s[58:59], vcc",
        "v_mov_b32_e32 v13, s46",
        "s_or_b64 exec, exec, s[58:59]",
    ])


# Eight-byte encodings in the masked fixtures: SMEM and sample loads, VOP3 (e64) and exp.
def offset_pairs(program):
    pairs, offset = [], 0
    for line in program:
        pairs.append((offset, line))
        wide = line.startswith(("s_load_", "s_buffer_load_", "image_sample", "exp ")) or "_e64" in line
        offset += 8 if wide else 4
    return pairs


def masked_skip():
    """The masked region with its whole-wave empty-mask skip, as (offset, text) pairs."""
    program = masked_region()
    program.insert(program.index("s_and_saveexec_b64 s[60:61], vcc") + 1, "s_cbranch_execz 2")
    return offset_pairs(program)


def masked_skip_asm():
    """The skip program as assembler text: the branch names its restore label instead."""
    program = [line for _, line in masked_skip()]
    program[program.index("s_cbranch_execz 2")] = "s_cbranch_execz .Lmask_restore"
    program.insert(program.index("s_or_b64 exec, exec, s[60:61]"), ".Lmask_restore:")
    return program


MASKED_RAMP = RAMP


def masked_pixels(threshold, inner, complementary):
    pixels = []
    for g in MASKED_RAMP:
        for r in MASKED_RAMP:
            cond = single(r / 255) < threshold
            if complementary:
                pixels.append((single(r / 255), inner[0] if cond else inner[1], single(128 / 255), 1))
            else:
                pixels.append((single(r / 255), inner[0] if cond else single(g / 255),
                               inner[1] if cond else single(128 / 255), 1))
    return pixels


def check_masked_draw(directory, compiler, renderer, backend):
    """Draw masked regions on WARP or hardware: threshold 0.5 mixes the mask across the ramp."""
    source = directory / "masked.hlsl"
    shader = directory / "masked.cso"
    constants = bytearray(256)
    for label, program, complementary in (
            ("region", masked_region(), False), ("skip", masked_skip(), False),
            ("complementary", masked_complementary(), True)):
        source.write_text(translate(program), encoding="ascii")
        compile_shader(compiler, source, shader)
        cases = []
        for threshold in (0.5, 10.0, -1.0):
            struct.pack_into("<4f", constants, 12, threshold, 0.25, 0.75, 0.0)
            case = directory / f"masked-{label}-{len(cases)}.case"
            write_case(case, constants, masked_pixels(threshold, (0.25, 0.75), complementary))
            cases.append((case, "pass"))
        run_pixel_batch(renderer, backend, shader, cases)
    struct.pack_into("<4f", constants, 12, 0.5, 0.25, 0.75, 0.0)
    expected = masked_pixels(0.5, (0.25, 0.75), False)
    unmasked = [line for line in masked_region()
                if line not in ("s_and_saveexec_b64 s[60:61], vcc", "s_or_b64 exec, exec, s[60:61]")]
    for mutant in (("reversed", [line.replace("v_cmp_lt_f32_e64", "v_cmp_ge_f32_e64")
                                 for line in masked_region()]),
                   ("unmasked", unmasked)):
        source.write_text(translate(mutant[1]), encoding="ascii")
        compile_shader(compiler, source, shader)
        case = directory / f"masked-{mutant[0]}.case"
        write_case(case, constants, expected)
        run_pixel_batch(renderer, backend, shader, [(case, "mismatch")])


def with_scalar_constant_reads() -> list[str]:
    program = with_constants(["v_mov_b32_e32 v12, s61", "v_mov_b32_e32 v13, s62", "v_mov_b32_e32 v14, s63"])
    load = program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc")
    program[load:load + 1] = [
        "s_buffer_load_dword s61, s[44:47], null",
        "s_buffer_load_dword s62, s[44:47], 0x8",
        "s_buffer_load_dword s63, s[44:47], 0xfc",
    ]
    return program


# Two further descriptors beside the b0 one: another slot of the texture table, and a slot of the
# table whose low address word arrives in entry s9. Every read takes word 2, so only the buffer differs.
THREE_BUFFERS = dict(constant_descriptor=64, extra_constants=[(20, 0x60), (9, 0x20)])


def with_three_constant_buffers() -> list[str]:
    program = with_constants(["v_mov_b32_e32 v12, s61", "v_mov_b32_e32 v13, s62", "v_mov_b32_e32 v14, s63"])
    descriptor = program.index("s_load_dwordx4 s[44:47], s[20:21], 0x40")
    program[descriptor + 1:descriptor + 1] = [
        "s_load_dwordx4 s[52:55], s[20:21], 0x60",
        "s_mov_b32 s64, s9",
        "s_mov_b32 s65, 0",
        "s_load_dwordx4 s[56:59], s[64:65], 0x20",
    ]
    load = program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc")
    program[load:load + 1] = [
        "s_buffer_load_dword s61, s[44:47], 0x8",
        "s_buffer_load_dword s62, s[52:55], 0x8",
        "s_buffer_load_dword s63, s[56:59], 0x8",
    ]
    return program


def translate_three_buffers(instructions):
    return lower(instructions, 20, 24, [1, 2, 3, 4], **THREE_BUFFERS)


def affine_coordinates() -> list[str]:
    program = assembly(False)
    high_address = "s_mov_b32 s21, 0"
    program.remove(high_address)
    restore = program.index("s_and_b64 exec, exec, s[50:51]")
    program[restore:restore] = [
        high_address,
        "s_load_dwordx4 s[44:47], s[20:21], 0x40",
        "s_waitcnt lgkmcnt(0)",
        "s_buffer_load_dwordx4 s[60:63], s[44:47], null",
        "s_waitcnt lgkmcnt(0)",
        "v_add_f32_e32 v8, 0.5, v8",
        "v_mul_legacy_f32_e32 v8, s62, v8",
        "v_fma_f32 v8, 0.5, v8, 0",
    ]
    return program


PAIR_LOADS = ("s_buffer_load_dwordx2 s[60:61], s[44:47], 0xc",
              "s_buffer_load_dwordx2 s[62:63], s[44:47], 0x14",
              "s_buffer_load_dwordx2 s[64:65], s[44:47], 0xf8")


def with_scalar_pair_reads() -> list[str]:
    """Read constant words 3 and 4, 5 and 6, 62 and 63 as pairs; the first pair spans two float4 rows."""
    program = with_constants([
        "v_mul_f32_e32 v12, s60, v12",
        "v_max_f32_e32 v12, s64, v12",
        "v_add_f32_e32 v13, s61, v13",
        "v_sub_f32_e32 v14, s62, v14",
        "v_mov_b32_e32 v15, s63",
        "v_mul_f32_e32 v15, s65, v15",
    ])
    load = program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc")
    program[load:load + 1] = PAIR_LOADS
    return program


def scalar_pair_case(directory: Path, scale: float, bias: float, minuend: float, alpha: float,
                     floor: float, gain: float, *, name: str = "scalar-pairs.case") -> Path:
    """Expected colors for with_scalar_pair_reads(). Every step is exact in binary32."""
    def exact(value):
        assert single(value) == value, value
        return value
    case = bytearray(256)
    struct.pack_into("<4f", case, 12, *map(exact, (scale, bias, minuend, alpha)))
    struct.pack_into("<2f", case, 248, exact(floor), exact(gain))
    for y in RAMP:
        for x in RAMP:
            r, g, b = (single(c / 255) for c in (x, y, 128))
            color = (max(floor, exact(scale * r)), exact(bias + g), exact(minuend - b), exact(gain * alpha))
            case.extend(struct.pack("<4f", *map(half_rtz, color)))
    path = directory / name
    path.write_bytes(case)
    return path


def curve():
    return with_constants([
        "v_max_f32_e32 v18, s46, v12",
        "v_log_f32_e32 v18, v18",
        "v_rcp_f32_e32 v19, s44",
        "v_mul_legacy_f32_e32 v18, v19, v18",
        "v_exp_f32_e32 v18, v18",
        "v_sub_f32_e32 v19, 1.0, v13",
        "v_fma_f32 v12, s45, v18, -v19",
        "v_fmac_f32_e32 v12, s47, v13",
        "v_min_f32_e32 v12, 1.0, v12",
        "v_mul_legacy_f32_e64 v12, v12, 1.0 clamp",
    ])


def shaping():
    return with_constants([
        "v_max_f32_e32 v18, s46, v12",
        "v_sqrt_f32_e32 v18, v18",
        "v_mul_f32_e32 v18, s45, v18",
        "v_add_f32_e32 v19, s44, v13",
        "v_rsq_f32_e32 v19, v19",
        "v_add_f32_e32 v18, v19, v18",
        "v_mul_f32_e32 v20, s44, v13",
        "v_floor_f32_e32 v21, v20",
        "v_fract_f32_e32 v20, v20",
        "v_fma_f32 v12, s47, v20, v18",
        "v_sub_f32_e32 v12, v12, v21",
        "v_mul_f32_e64 v12, v12, 1.0 clamp",
    ])


def truncate_values() -> list[str]:
    return with_constants([
        "v_add_f32_e32 v12, s44, v12",
        "v_trunc_f32_e32 v12, v12",
        "v_trunc_f32_e32 v13, s45",
        "v_trunc_f32_e64 v14, -|s46|",
        "v_trunc_f32_e64 v15, s47 clamp",
    ])


def sopk_scalar() -> list[str]:
    return with_constants([
        "s_mov_b32 s50, s44",
        "s_mulk_i32 s50, 0x3",
        "s_mov_b32 s51, s45",
        "s_addk_i32 s51, 0xfffe",
        "v_mov_b32_e32 v12, s50",
        "v_mov_b32_e32 v13, s51",
    ])


SOPK_CMPK_OPS = ("eq", "lg", "gt", "ge", "lt", "le")
SOPK_CMPK_WORDS = (
    (0xffff7fff, 0x00007fff, 0x13579bdf, 0xffffffff),
    (0xffff8000, 0x00008000, 0x13579bdf, 0x0000ffff),
    (0x00008000, 0xffff8000, 0x2468ace0, 0xffffffff),
    (0xffff8001, 0x00008001, 0x2468ace0, 0),
    (0xffff8001, 0x00008001, 0xfffffffe, 0xffffffff),
    (0xffff8001, 0x00008001, 0x2468ace0, 1),
)


def sopk_cmpk_cmovk() -> list[str]:
    body = ["s_mov_b32 s52, 0", "s_mov_b32 s53, 0"]
    for bit, op in enumerate(SOPK_CMPK_OPS):
        flag = 1 << bit
        body += [
            f"s_cmpk_{op}_i32 s44, 0x8000",
            f"s_cselect_b32 s63, {flag}, 0",
            "s_or_b32 s52, s52, s63",
            f"s_cmpk_{op}_u32 s45, 0x8000",
            f"s_cselect_b32 s63, {flag}, 0",
            "s_or_b32 s53, s53, s63",
        ]
    body += [
        "s_cmpk_eq_u32 s45, 0x8001",
        "s_cselect_b32 s63, 64, 0",
        "s_or_b32 s53, s53, s63",
        "s_mov_b32 s50, s46",
        "s_mov_b32 s51, s46",
        "s_cmpk_eq_i32 s47, 0xffff",
        "s_cmovk_i32 s50, 0xfffe",
        "s_cmovk_i32 s51, 0x7fff",
        "s_cmpk_eq_i32 s47, 1",
        "s_cmovk_i32 s51, 0x8000",
        "v_mov_b32_e32 v12, s52",
        "v_mov_b32_e32 v13, s53",
        "v_mov_b32_e32 v14, s50",
        "v_mov_b32_e32 v15, s51",
    ]
    return uncompressed_export(body)


def sopk_cmpk_cmovk_reference(words):
    s44, s45, s46, s47 = words
    signed44 = s44 - (1 << 32) if s44 >> 31 else s44
    preds = {
        "eq": lambda a, b: a == b,
        "lg": lambda a, b: a != b,
        "gt": lambda a, b: a > b,
        "ge": lambda a, b: a >= b,
        "lt": lambda a, b: a < b,
        "le": lambda a, b: a <= b,
    }
    mask_i32 = sum((1 << bit) for bit, op in enumerate(SOPK_CMPK_OPS) if preds[op](signed44, -0x8000))
    mask_u32 = sum((1 << bit) for bit, op in enumerate(SOPK_CMPK_OPS) if preds[op](s45, 0x8000))
    mask_u32 |= (1 << len(SOPK_CMPK_OPS)) * (s45 == 0x8001)
    take = s47 == 0xffffffff
    first = 0xfffffffe if take else s46
    second = {-1: 0x7fff, 1: -32768}.get(s47 - (1 << 32) if s47 >> 31 else s47, s46)
    return (mask_i32, mask_u32, first, second % (1 << 32))


def rounding():
    """Round scaled samples up and to nearest even, and a constant tie through the VOP3 encoding."""
    return with_constants([
        "v_mul_f32_e32 v18, s44, v12",
        "v_ceil_f32_e32 v12, v18",
        "v_mul_f32_e32 v19, s44, v13",
        "v_rndne_f32_e32 v13, v19",
        "v_rndne_f32_e64 v14, s45",
        "v_ceil_f32_e64 v15, -s46",
    ])


def select(wave64=False):
    """Compare into lane masks and select; LLVM names the masks by wavefront width."""
    vcc, first, second = ("vcc", "s[52:53]", "s[54:55]") if wave64 else ("vcc_lo", "s52", "s53")
    return with_constants([
        f"v_cmp_nlt_f32_e64 {second}, v12, v13",
        f"v_cndmask_b32_e64 v14, 0, 1.0, {second}",
        f"v_cmp_eq_f32_e32 {vcc}, v12, v13",
        f"v_cndmask_b32_e32 v15, s47, v15, {vcc}",
        f"v_cmp_lt_f32_e32 {vcc}, s44, v12",
        "v_sub_f32_e32 v18, 1.0, v12",
        f"v_cmp_ge_f32_e64 {first}, v13, s45",
        f"v_cndmask_b32_e64 v13, v13, s46, {first}",
        f"v_cndmask_b32_e32 v12, v12, v18, {vcc}",
    ])


def three_way():
    """Median, minimum and maximum of three operands, each result one of its inputs."""
    return with_constants([
        "v_med3_f32 v15, s47, v13, v12",
        "v_max3_f32 v14, v12, v14, v13",
        "v_med3_f32 v12, v12, s44, s45",
        "v_min3_f32 v13, |s46|, v13, v12",
    ])


def absolute_value():
    program = assembly(False)
    pos = program.index("v_cvt_pkrtz_f16_f32_e32 v16, v12, v13")
    program[pos:pos] = [
        "v_sub_f32_e32 v12, 0.5, v12",
        "v_sub_f32_e64 v12, |v12|, 0",
    ]
    return program


def unpack_bytes():
    # Each output takes a different byte of a constant word, in e32 and e64 forms.
    return with_constants([
        "v_cvt_f32_ubyte0_e32 v12, s44",
        "v_cvt_f32_ubyte1_e64 v13, s45",
        "v_cvt_f32_ubyte2_e32 v14, s46",
        "v_cvt_f32_ubyte3_e64 v15, s47 clamp",
    ])


def offset_nibbles():
    # The low four bits of each constant word as a signed offset in sixteenths.
    return with_constants([
        "v_cvt_off_f32_i4_e32 v12, s44",
        "v_cvt_off_f32_i4_e64 v13, s45",
        "v_cvt_off_f32_i4_e32 v14, s46",
        "v_cvt_off_f32_i4_e64 v15, s47 clamp",
    ])


def legacy_multiply_add():
    # gfx10 allows two scalar operands per VOP3 instruction, so two constants move to VGPRs.
    return uncompressed_export([
        "v_mov_b32_e32 v18, s46",
        "v_mov_b32_e32 v19, s47",
        "v_mad_legacy_f32 v12, s44, s45, v18",
        "v_mad_legacy_f32 v13, s45, s44, -v19",
        "v_mad_legacy_f32 v14, |v18|, s47, s44",
        "v_mad_legacy_f32 v15, s44, v19, v18 clamp",
    ])


# v_mac_legacy_f32 is D = legacy_mul(S0, S1) + D, the destination being the accumulator (LLVM 15 VOP2Instructions.td:
# VOP_MAC_LEGACY_F32 under Constraints = "$vdst = $src2"). A, B, C and D are the constants in s44 to s47. Each channel
# starts as a copy of its accumulator: C, C, A and D. Channel 0 is the e32 form; channels 1 to 3 are e64 with a
# negated source, an absolute-value source and clamp.
MAC_LEGACY_BODY = [
    "v_mov_b32_e32 v18, s45",
    "v_mov_b32_e32 v19, s47",
    "v_mov_b32_e32 v12, s46",
    "v_mov_b32_e32 v13, s46",
    "v_mov_b32_e32 v14, s44",
    "v_mov_b32_e32 v15, s47",
    "v_mac_legacy_f32_e32 v12, s44, v18",
    "v_mac_legacy_f32_e64 v13, -v18, s44",
    "v_mac_legacy_f32_e64 v14, |s46|, v19",
    "v_mac_legacy_f32_e64 v15, s44, v18 clamp",
]
# The last set is (1 + 2^-23) * (1 - 2^-23) + -1: unfused it is 0, fused it is -2^-46.
MAC_LEGACY_SETS = ((0.0, math.inf, 1.5, -2.0), (2.0, 3.0, 0.5, 0.25), (0.0, math.nan, -0.75, 4.0),
                   (-1.5, 0.0, 2.0, 0.5), (3.0, -2.0, 1.0, -1.0), (1 + 2.0**-23, 1 - 2.0**-23, -1.0, 0.0))


def mac_legacy_reference(a, b, c, d, *, fused_first_channel=False, accumulate=lambda x, y, acc: acc):
    """The four exported values, from an independent model: a zero factor gives zero, then a rounded unfused add.
    accumulate chooses the added operand from (x, y, acc); the default is the destination, as the instruction has."""
    def legacy(x, y, acc):
        product = 0.0 if x == 0 or y == 0 else x * y
        return single(single(product) + accumulate(x, y, acc))

    values = [legacy(a, b, c), legacy(-b, a, c), legacy(abs(c), d, a), min(1.0, max(0.0, legacy(a, b, d)))]
    if fused_first_channel:
        values[0] = single(a * b + c)
    assert not any(math.isnan(v) for v in values), values
    return tuple(values)


def ceiling():
    return with_constants([
        "v_ceil_f32_e32 v12, s44",
        "v_ceil_f32_e64 v13, -|s45|",
        "v_ceil_f32_e64 v14, |s46|",
        "v_ceil_f32_e64 v15, |s47| clamp",
    ])


def int_minmax_pair(encoding):
    return with_constants([
        "v_mov_b32_e32 v18, s45",
        f"v_min_i32_{encoding} v12, s44, v18",
        f"v_max_i32_{encoding} v13, s44, v18",
        f"v_min_u32_{encoding} v14, s44, v18",
        f"v_max_u32_{encoding} v15, s44, v18",
    ])


def int_minmax_triple(kind):
    return with_constants([
        "v_mov_b32_e32 v18, s45",
        "v_mov_b32_e32 v19, s46",
        f"v_min3_{kind} v12, s44, v18, v19",
        f"v_max3_{kind} v13, s44, v18, v19",
        f"v_med3_{kind} v14, s44, v18, v19",
        f"v_med3_{kind} v15, s46, v18, v18",
    ])


def cvt_int_fixture(encoding):
    if encoding == "e32":
        return with_constants([
            "v_cvt_f32_i32_e32 v12, s44",
            "v_cvt_f32_u32_e32 v13, s45",
            "v_mov_b32_e32 v18, s46",
            "v_cvt_f32_i32_e32 v14, v18",
            "v_cvt_f32_u32_e32 v15, s47",
        ])
    return with_constants([
        "v_cvt_f32_i32_e64 v12, s44",
        "v_cvt_f32_u32_e64 v13, s45 clamp",
        "v_mov_b32_e32 v18, s46",
        "v_cvt_f32_i32_e64 v14, v18 clamp",
        "v_cvt_f32_u32_e64 v15, s47",
    ])


def cvt_int_literals():
    return with_constants([
        "v_cvt_f32_i32_e32 v12, -7",
        "v_cvt_f32_u32_e32 v13, 25",
        "v_cvt_f32_i32_e64 v14, -100 clamp",
        "v_cvt_f32_u32_e64 v15, 50 clamp",
    ])


def frexp_mant_values():
    return with_constants([
        "v_frexp_mant_f32_e32 v12, s44",
        "v_frexp_mant_f32_e64 v13, -|s45|",
        "v_frexp_mant_f32_e64 v14, |s46|",
        "v_frexp_mant_f32_e64 v15, s47 clamp",
    ])


def frexp_exp_values():
    # Compare the integer exponent before the binary16 pixel export.
    return with_constants([
        "v_frexp_exp_i32_f32_e32 v18, s44",
        "v_frexp_exp_i32_f32_e64 v19, -|s46|",
        "v_mov_b32_e32 v15, 1.0",
        "v_cmp_eq_i32_e32 vcc_lo, s45, v18",
        "v_cndmask_b32_e32 v12, 0, v15, vcc_lo",
        "v_cmp_eq_i32_e32 vcc_lo, s47, v19",
        "v_cndmask_b32_e32 v13, 0, v15, vcc_lo",
        "v_mov_b32_e32 v14, 0",
    ])


def ldexp_values():
    return with_constants([
        "v_mov_b32_e32 v18, s44",
        "v_ldexp_f32 v12, v18, s45",
        "v_ldexp_f32 v13, -|s44|, 2",
        "v_ldexp_f32 v14, s46, -2",
        "v_ldexp_f32 v15, s47, 3 clamp",
    ])
# v_sin_f32 and v_cos_f32 take revolutions, sin(2 * pi * x). D3D bounds sin and cos only to 0.0008
# absolute in [-100 pi, 100 pi], so each result is quantized to 1/32 steps by exact operations
# (times 32, plus 0.5, floor, times 1/32) and every reference lies at least 0.1 step from a boundary.
TRIG_BODY = [
    "v_sin_f32_e32 v12, s44",
    "v_cos_f32_e32 v13, s45",
    "v_sin_f32_e64 v14, -|s46|",
    "v_cos_f32_e64 v15, -s47 clamp",
] + [line for reg in ("v12", "v13", "v14", "v15") for line in (
    f"v_mul_f32_e32 {reg}, 0x42000000, {reg}",
    f"v_add_f32_e32 {reg}, 0.5, {reg}",
    f"v_floor_f32_e32 {reg}, {reg}",
    f"v_mul_f32_e32 {reg}, 0x3d000000, {reg}",
)]
# Each case covers all four quadrants across its channels and one input of millions of turns, where
# multiplying the unreduced input by 2 pi in binary32 moves the quantized result by 20 or more steps.
TRIG_CASES = (
    (-4000000.25, 0.125, -0.375, 0.5),
    (-0.125, 4000000.25, 2.75, -0.125),
    (0.25, -1.625, -4000000.5, 0.0),
    (-0.375, 0.375, 0.0625, 2000000.75),
)


def trig_reference(case, swap=False):
    """The quantized four channels of TRIG_BODY, from a double-precision reduction of each input."""
    channels = []
    for kind, x, saturate in zip(("sin", "cos", "sin", "cos"), (case[0], case[1], -abs(case[2]), -case[3]),
                                 (False, False, False, True)):
        if swap:
            kind = "cos" if kind == "sin" else "sin"
        value = (math.sin if kind == "sin" else math.cos)(2 * math.pi * math.remainder(x, 1.0))
        if saturate:
            value = min(1.0, max(0.0, value))
        scaled = value * 32 + 0.5
        # Only a drawn reference needs the margin; the swapped values merely have to differ from it.
        assert swap or abs(scaled - round(scaled)) >= 0.1, (case, kind, value)
        channels.append(math.floor(scaled) / 32)
    return channels


def reciprocal_iflag():
    return with_constants([
        "v_rcp_iflag_f32_e32 v12, s44",
        "v_rcp_iflag_f32_e64 v13, -|s45|",
        "v_rcp_iflag_f32_e64 v14, |s46|",
        "v_rcp_iflag_f32_e64 v15, |s47| clamp",
    ])


# The gfx10 predicate number is a four-bit truth table. Bits 0..3 select
# less, equal, greater and unordered respectively, independent of the HLSL.
PREDICATES = ("f", "lt", "eq", "le", "gt", "lg", "ge", "o",
              "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "tru")


def predicate_group(first, sdwa=False):
    # Copy runtime constants into vector registers using supported operations.
    body = ["v_sub_f32_e64 v18, s45, 0", "v_sub_f32_e64 v19, s47, 0"]
    for channel, op in enumerate(PREDICATES[first:first + 4]):
        compare = (f"v_cmp_{op}_f32_sdwa vcc_lo, s44, v18 src0_sel:DWORD src1_sel:DWORD" if sdwa
                   else f"v_cmp_{op}_f32_e32 vcc_lo, s44, v18")
        body.extend([compare,
                     f"v_cndmask_b32_e32 v{12 + channel}, 0, v19, vcc_lo"])
    return with_constants(body)


# Three-bit truth table for signed and unsigned 32-bit integer comparisons (bits 0..2: <, ==, >).
INT_PREDICATES = ("f", "lt", "eq", "le", "gt", "ne", "ge", "t")


def int_predicate_group(kind, encoding, first, *, wave64=False):
    # Copy runtime constants with v_mov_b32 so all 32 bits survive unchanged.
    dst = ("vcc" if wave64 else "vcc_lo") if encoding == "e32" else ("s[52:53]" if wave64 else "s52")
    body = ["v_mov_b32_e32 v18, s45", "v_mov_b32_e32 v19, s47"]
    for channel, op in enumerate(INT_PREDICATES[first:first + 4]):
        body.extend([f"v_cmp_{op}_{kind}_{encoding} {dst}, s44, v18",
                     f"v_cndmask_b32_{encoding} v{12 + channel}, 0, v19, {dst}"])
    return with_constants(body)


def unsigned_less_than(kind="u32"):
    return with_constants([
        "v_mov_b32_e32 v18, s44",
        "v_mov_b32_e32 v19, s45",
        "v_mov_b32_e32 v20, 1.0",
        f"v_cmp_lt_{kind}_e32 vcc_lo, s44, v19",
        "v_cndmask_b32_e32 v12, 0, v20, vcc_lo",
        f"v_cmp_lt_{kind}_e32 vcc_lo, s45, v18",
        "v_cndmask_b32_e32 v13, 0, v20, vcc_lo",
        f"v_cmp_lt_{kind}_e64 s52, v18, v19",
        "v_cndmask_b32_e64 v14, 0, v20, s52",
        f"v_cmp_lt_{kind}_e64 s52, v19, v18",
        "v_cndmask_b32_e64 v15, 0, v20, s52",
    ])


# 16-bit integer comparisons read only the low half of each source. L is s44, R is v18 (a copy of s45);
# other operands are an integer inline constant or a 16-bit literal. Each (kind, encoding) pair has two
# groups of four channels, so all six predicates appear with SGPR and VGPR operands, and the second group
# adds inline constants at both ends of their range and a literal with bit 15 set.
INT16_GROUPS = {
    "e32": ((("lt", "L", "R"), ("eq", "L", "R"), ("le", "L", "R"), ("gt", "L", "R")),
            (("ne", "L", "R"), ("ge", "L", "R"), ("lt", "-16", "R"), ("gt", "0x8000", "R"))),
    "e64": ((("lt", "L", "R"), ("eq", "L", "R"), ("le", "L", "R"), ("gt", "L", "R")),
            (("ne", "L", "R"), ("ge", "R", "L"), ("lt", "R", "64"), ("le", "R", "0x8000"))),
}


def int16_group(kind, encoding, group, *, wave64=False):
    dst = ("vcc" if wave64 else "vcc_lo") if encoding == "e32" else ("s[52:53]" if wave64 else "s52")
    body = ["v_mov_b32_e32 v18, s45", "v_mov_b32_e32 v19, s47"]
    names = {"L": "s44", "R": "v18"}
    for channel, (op, a, b) in enumerate(INT16_GROUPS[encoding][group]):
        body.extend([f"v_cmp_{op}_{kind}_{encoding} {dst}, {names.get(a, a)}, {names.get(b, b)}",
                     f"v_cndmask_b32_{encoding} v{12 + channel}, 0, v19, {dst}"])
    return with_constants(body)


def int16_reference(kind, encoding, group, left, right):
    """Expected channels of int16_group for 32-bit words left (s44) and right (s45)."""
    def value(token):
        word = {"L": left, "R": right}.get(token)
        word = int(token, 0) if word is None else word
        low = word & 0xffff
        return low - 0x10000 if kind == "i16" and low & 0x8000 else low

    tests = {"lt": lambda a, b: a < b, "eq": lambda a, b: a == b, "le": lambda a, b: a <= b,
             "gt": lambda a, b: a > b, "ne": lambda a, b: a != b, "ge": lambda a, b: a >= b}
    return [float(tests[op](value(a), value(b))) for op, a, b in INT16_GROUPS[encoding][group]]


# 64-bit integer comparisons. The constant words s44..s47 hold two 64-bit values: A = (s44 low, s45 high) and
# B = (s46 low, s47 high). v18..v21 copy them, so v[18:19] = A and v[20:21] = B, and v19 also starts the
# overlapping pair v[19:20] = (s45, s46). Each group has four channels, and every channel selects 1.0 (v22) or 0.
# Channel c of group g tests predicate CMP64_PREDICATES[4g + c] on operand shape CMP64_SHAPES[(c + g) % 4].
CMP64_PREDICATES = ("lt", "eq", "le", "gt", "ne", "ge", "f", "t")
CMP64_SHAPES = ("pairs", "sgpr", "overlap", "inline")


def cmp64_sources(shape, encoding):
    """src0 and src1 of one operand shape. An e32 src1 must be a VGPR pair, so its SGPR pair is src0."""
    return {"pairs": ("v[18:19]", "v[20:21]"),
            "sgpr": ("s[44:45]", "v[20:21]") if encoding == "e32" else ("v[18:19]", "s[46:47]"),
            "overlap": ("v[18:19]", "v[19:20]"),
            "inline": ("-1", "v[20:21]")}[shape]


def cmp64_prefix():
    return [f"v_mov_b32_e32 v{18 + n}, s{44 + n}" for n in range(4)] + ["v_mov_b32_e32 v22, 1.0"]


def cmp64_program(body):
    """body after the four word copies and v22 = 1.0, exported as four uncompressed channels."""
    return uncompressed_export(cmp64_prefix() + body)


def cmp64_group(kind, encoding, group, *, wave64=False):
    dst = ("vcc" if wave64 else "vcc_lo") if encoding == "e32" else ("s[52:53]" if wave64 else "s52")
    body = []
    for channel in range(4):
        op = CMP64_PREDICATES[4 * group + channel]
        src0, src1 = cmp64_sources(CMP64_SHAPES[(channel + group) % 4], encoding)
        body.extend([f"v_cmp_{op}_{kind}_{encoding} {dst}, {src0}, {src1}",
                     f"v_cndmask_b32_{encoding} v{12 + channel}, 0, v22, {dst}"])
    return cmp64_program(body)


# Pairs (s44, s45, s46, s47). Each 32-bit half alone gets some of them wrong: high words decide the first and
# fourth, low words the second and third; the fifth to seventh order differently signed and unsigned, and the sixth
# and seventh have high words that differ only in the sign bit. Overlap makes (s45, s46) the second operand.
CMP64_WORDS = (
    (0x00000000, 0x00000001, 0xffffffff, 0x00000000),  # 0x1_00000000 against 0x0_ffffffff
    (0x00000005, 0x12345678, 0x00000003, 0x12345678),  # identical high words, low words decide: A > B
    (0x00000003, 0x12345678, 0x00000005, 0x12345678),  # identical high words, low words decide: A < B
    (0xffffffff, 0xffffffff, 0x00000000, 0x00000000),  # -1 against 0
    (0xffffffff, 0x7fffffff, 0x00000000, 0x80000000),  # 0x7fffffff_ffffffff against 0x80000000_00000000
    (0x00000002, 0x00000001, 0x00000002, 0x80000001),  # high words differ only in the sign bit
    (0xffffffff, 0x80000001, 0x00000000, 0x00000001),  # as above, with the low words the other way round
    (0x89abcdef, 0x01234567, 0x89abcdef, 0x01234567),  # equal
    (0x00000001, 0x7fc00000, 0xffffffff, 0x80000000),  # words that are not normal binary32 values
    (0x80000000, 0x00000001, 0x7fc00000, 0x00000000),  # more non-normal bit patterns
    (0x00000003, 0x00000007, 0x00000007, 0x00000000),  # overlap: (3, 7) against (7, 7)
    (0x7fc00000, 0x7fc00000, 0x7fc00000, 0xffffffff),  # overlap: both operands are (0x7fc00000, 0x7fc00000)
    (0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff),  # all ones: the inline -1 equals both operands
)


def cmp64_truth(op, signed, left, right):
    """left and right are (low, high) word pairs."""
    def value(words):
        number = words[0] | words[1] << 32
        return number - (1 << 64) if signed and number >> 63 else number

    a, b = value(left), value(right)
    return {"f": False, "t": True, "lt": a < b, "eq": a == b, "le": a <= b,
            "gt": a > b, "ne": a != b, "ge": a >= b}[op]


def cmp64_expected(kind, encoding, group, words):
    """The four channels (float views of the exported words) of cmp64_group for one set of constant words."""
    s44, s45, s46, s47 = words
    channels = []
    for channel in range(4):
        shape = CMP64_SHAPES[(channel + group) % 4]
        left, right = {"overlap": ((s44, s45), (s45, s46)),
                       "inline": ((0xffffffff, 0xffffffff), (s46, s47))}.get(shape, ((s44, s45), (s46, s47)))
        truth = cmp64_truth(CMP64_PREDICATES[4 * group + channel], kind == "i64", left, right)
        channels.append(as_float(0x3f800000 if truth else 0))
    return tuple(channels)


# v_cmp_class_f32 is true when the class of S0 is in the mask S1 (LLVM SIInstrFlags::ClassFlags).
# Each channel tests one property: +inf against its own bit, a quiet NaN against both NaN bits,
# -0 against the +0 bit and a positive denormal against its bit, so neither the sign of zero
# nor a denormal flushed to zero can pass.
CLASS_BITS = (0x7f800000, 0x7fc00000, 0x80000000, 0x00000001)
CLASS_BODY = [
    "v_mov_b32_e32 v19, 1.0",
    "v_cmp_class_f32_e64 vcc_lo, s44, 0x200",
    "v_cndmask_b32_e32 v12, 0, v19, vcc_lo",
    "v_cmp_class_f32_e64 vcc_lo, s45, 3",
    "v_cndmask_b32_e32 v13, 0, v19, vcc_lo",
    "v_mov_b32_e32 v18, 64",
    "v_cmp_class_f32_e32 vcc_lo, s46, v18",
    "v_cndmask_b32_e32 v14, 0, v19, vcc_lo",
    "v_cmp_class_f32_e64 s0, s47, 0x80",
    "v_cndmask_b32_e64 v15, 0, v19, s0",
]


def check_class_draw(directory, compiler, renderer, backend):
    source = directory / "class.hlsl"
    shader = directory / "class.cso"
    source.write_text(translate(with_constants(CLASS_BODY)), encoding="ascii")
    compile_shader(compiler, source, shader)
    constants = bytearray(256)
    struct.pack_into("<4I", constants, 12, *CLASS_BITS)
    expected = [1.0, 1.0, 0.0, 1.0]
    cases = []
    # A wrong class in any channel flips its result; the renderer must report each flip.
    for k in range(4):
        wrong_values = list(expected)
        wrong_values[k] = 1.0 - wrong_values[k]
        path = directory / f"class-wrong-{k}.case"
        path.write_bytes(constants + struct.pack("<4f", *wrong_values) * 64)
        cases.append((path, "mismatch"))
    pass_case = directory / "class-pass.case"
    pass_case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
    cases.append((pass_case, "pass"))
    run_pixel_batch(renderer, backend, shader, cases)


# v_fract_f32 clamps x - floor(x) to 0x3f7fffff. For -2**-30 the unclamped difference rounds to 1.0;
# the other inputs are exact and stay below the clamp.
FRACT_INPUTS = (-2.0 ** -30, -0.25, 2.75, -2.0 ** -23)
FRACT_CLAMP = struct.unpack("<f", struct.pack("<I", 0x3f7fffff))[0]


def fract_values():
    return with_constants([
        "v_fract_f32_e32 v12, s44",
        "v_fract_f32_e64 v13, |s45|",
        "v_fract_f32_e64 v14, -|s46|",
        "v_fract_f32_e32 v15, s47",
    ])


RECIPROCAL_BODY = [
    "v_rcp_f32_e64 v12, |s44|",
    "v_rcp_f32_e64 v13, -|s45|",
    "v_rsq_f32_e64 v14, |s46|",
    "v_rsq_f32_e64 v15, |s47| clamp",
]


def check_reciprocal_draw(directory, compiler, renderer, backend):
    source = directory / "reciprocal.hlsl"
    shader = directory / "reciprocal.cso"
    case = directory / "reciprocal.case"
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, -4.0, 8.0, -16.0, -0.25)
    case.write_bytes(constants + struct.pack("<4f", 0.25, -0.125, 0.25, 1.0) * 64)
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    for mutation in (False, True):
        body = RECIPROCAL_BODY.copy()
        if mutation:
            # 1 / abs(-16) is 1/16, while its reciprocal square root is 1/4.
            body[2] = body[2].replace("v_rsq", "v_rcp")
        source.write_text(translate(with_constants(body)), encoding="ascii")
        compile_shader(compiler, source, shader)
        if mutation:
            wrong = subprocess.run(draw, capture_output=True, text=True)
            if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
                raise AssertionError("Substituting reciprocal for reciprocal square root escaped detection")
        else:
            subprocess.run(draw, check=True)
# VOP3 (e64) encodings that the translator already accepts but no other fixture draws: floor, fract, max, min and
# fmac with source modifiers, and the 16 comparisons writing an SGPR mask from negated operands. Negating both
# operands swaps the less and greater relations and keeps equality and unordered.
E64_ARITH = (
    (["v_floor_f32_e64 v12, -|s44|", "v_fract_f32_e64 v13, -s45",
      "v_max_f32_e64 v14, -s46, |s47|", "v_min_f32_e64 v15, s46, -|s47|"],
     [-3.0, 0.75, 4.0, -4.0], ((0, 2.0), (1, 0.25), (2, -3.0), (3, 3.0))),
    (["v_mov_b32_e32 v12, s46", "v_fmac_f32_e64 v12, -s44, |s45|",
      "v_mov_b32_e32 v13, 0", "v_mov_b32_e32 v14, 0", "v_mov_b32_e32 v15, 0"],
     [-0.125, 0.0, 0.0, 0.0], ((0, 6.125),)),
)
E64_ARITH_INPUTS = (2.5, 1.25, 3.0, -4.0)

# VOP3 (e64) encodings of v_exp_f32, v_log_f32 and v_mac_f32, which the translator accepted with no fixture
# drawing them. D3D allows exp2 and log2 a last-place error, so a raw result has no single binary32 reference:
# each input makes the true result a small integer, and adding 0.5 then taking the floor returns that integer
# for any result within 0.5 of it. That checks the operation and its source modifiers, which are what e64 adds
# over e32, and not last-place accuracy. v_mac_f32 is exact and is compared directly: channel 2 accumulates a
# product of modified sources, and channel 3 is the 4097 * 4097 case of MAD_BODY (0 unfused, 1.0 fused) with
# both sources negated. A, B, C and D are the constants in s44 to s47.
APPROX_E64_ACCUMULATOR = -16785408.0  # 0xcb801000, -(2**24 + 2**13)
APPROX_E64_BODY = [
    "v_exp_f32_e64 v12, -s44",
    "v_log_f32_e64 v13, |s45|",
    *[line for channel in (12, 13)
      for line in (f"v_add_f32_e32 v{channel}, 0.5, v{channel}", f"v_floor_f32_e32 v{channel}, v{channel}")],
    "v_mov_b32_e32 v14, s46",
    "v_mac_f32_e64 v14, -s44, |s45|",
    "v_mov_b32_e32 v15, 0xcb801000",
    "v_mac_f32_e64 v15, -s47, -s47",
]
APPROX_E64_OPS = ("v_exp_f32_e64", "v_log_f32_e64", "v_mac_f32_e64")
APPROX_E64_ROWS = ((-3.0, -32.0, 0.25, 4097.0), (-4.0, -8.0, 0.5, 4097.0))


def approx_e64_reference(a, b, c, d):
    """The four exported values: exp2(-A), log2(|B|), -A * |B| + C, and the unfused D * D + accumulator."""
    return (2.0 ** -a, math.log2(abs(b)), single(single(-a * abs(b)) + c),
            single(single(d * d) + APPROX_E64_ACCUMULATOR))


# Negative controls: (channel, value a wrong lowering would export, what went wrong). Each is drawn on the
# first row and must report a mismatch. The rounding step is applied to the wrong exp and log values as well.
APPROX_E64_WRONG = (
    (0, lambda a, b, c, d: math.floor(2.0 ** a + 0.5), "negation dropped on exp"),
    (0, lambda a, b, c, d: math.floor(math.exp(-a) + 0.5), "natural exponential for exp2"),
    (0, lambda a, b, c, d: math.floor(math.log2(-a) + 0.5), "log for exp"),
    (1, lambda a, b, c, d: math.floor(math.log(abs(b)) + 0.5), "natural logarithm for log2"),
    (1, lambda a, b, c, d: 2.0 ** abs(b), "exp for log"),
    (2, lambda a, b, c, d: a * abs(b) + c, "negation dropped on mac"),
    (2, lambda a, b, c, d: -a * abs(b), "accumulator dropped"),
    (3, lambda a, b, c, d: single(d * d + APPROX_E64_ACCUMULATOR), "fused multiply-add"),
)


def check_e64_approx_lowering():
    """Static checks for APPROX_E64_BODY: the modifiers reach exp2, log2 and the unfused helper."""
    body = translate(with_constants(APPROX_E64_BODY)).split("float4 main", 1)[1]
    assert set(APPROX_E64_OPS) <= ACCEPTED, set(APPROX_E64_OPS) - ACCEPTED
    assert "exp2((-params[0].w))" in body and "log2(abs(params[1].x))" in body, body
    assert "mad32((-params[0].w), abs(params[1].x), " in body and "mad32((-params[1].z), (-params[1].z), " in body, body
    assert body.count("mad32(") == 2 and " mad(" not in body, body
    # v_mac_f32 reads its destination, so an unwritten one is refused, as is a third source.
    must_reject(translate, with_constants(["v_mac_f32_e64 v20, -s44, |s45|"]))
    must_reject(translate, with_constants(["v_mov_b32_e32 v14, s46", "v_mac_f32_e64 v14, -s44, |s45|, v14"]))


def check_dpp_vop3_only_refused():
    """VOP3-only ALU ops have no DPP spelling; lowering one is a clean refusal, not a crash in the DPP path."""
    body = list(DPP_BODY)
    at = next(k for k, line in enumerate(body) if line.startswith("v_add_f32_dpp v23"))
    for op in ("mad", "mad_legacy", "min3", "max3", "med3"):
        assert op in ALU and op not in pixel_aot.DPP_ALU, op
        line = f"v_{op}_f32_dpp v23, v20, v20, v20 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf"
        program = with_constants(body[:at] + [line] + body[at + 1:])
        expect_value_error(translate, program, f"Unsupported instruction {program.index(line)}: {line}")


def check_mac_legacy_lowering():
    """Static checks for MAC_LEGACY_BODY and the forms LLVM 15 does not have for v_mac_legacy_f32."""
    program = uncompressed_export(MAC_LEGACY_BODY)
    body = translate(program).split("float4 main", 1)[1]
    assert {"v_mac_legacy_f32_e32", "v_mac_legacy_f32_e64"} <= ACCEPTED, ACCEPTED
    assert body.count("? 0.0f :") == 4 and body.count("saturate(") == 1 and "mad32(" not in body and " mad(" not in body, body
    # Each product is added to the value the destination held before the instruction: the copies of C, C, A and D.
    for line in ("alu26 = (((params[0].w == 0 || alu20 == 0) ? 0.0f : params[0].w * alu20) + alu22);",
                 "alu27 = ((((-alu20) == 0 || params[0].w == 0) ? 0.0f : (-alu20) * params[0].w) + alu23);",
                 "alu28 = (((abs(params[1].y) == 0 || alu21 == 0) ? 0.0f : abs(params[1].y) * alu21) + alu24);",
                 "alu29 = saturate((((params[0].w == 0 || alu20 == 0) ? 0.0f : params[0].w * alu20) + alu25));"):
        assert line in body, line
    # No DPP form exists in LLVM 15 (HasExtDPP = 0), and the SDWA table is an explicit list without it.
    assert "mac_legacy" in ALU and "mac_legacy" not in pixel_aot.DPP_ALU and "mac_legacy" not in pixel_aot.SDWA_ALU

    def swap(old, new):
        return [line.replace(old, new) for line in program]

    pos = program.index("v_mac_legacy_f32_e32 v12, s44, v18")
    for line in ("v_mac_legacy_f32_dpp v12, v18, v18 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
                 "v_mac_legacy_f32_sdwa v12, v18, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
                 "v_fmac_legacy_f32 v12, s44, v18"):
        expect_value_error(translate, program[:pos] + [line] + program[pos + 1:], f"Unsupported instruction {pos}: {line}")
    # Operands: the accumulator is implicit, so a third source or a missing second one is refused.
    for old, new in (("v_mac_legacy_f32_e32 v12, s44, v18", "v_mac_legacy_f32_e32 v12, s44, v18, v12"),
                     ("v_mac_legacy_f32_e32 v12, s44, v18", "v_mac_legacy_f32_e32 v12, s44"),
                     ("v_mac_legacy_f32_e64 v15, s44, v18 clamp", "v_mac_legacy_f32_e64 v15, s44, v18, v15 clamp")):
        changed = swap(old, new)
        assert changed != program
        expect_value_error(translate, changed, "Invalid arithmetic operation")
    # The accumulator must hold a tracked value, as for v_mac_f32: v20 was never written.
    changed = swap("v_mac_legacy_f32_e32 v12, s44, v18", "v_mac_legacy_f32_e32 v20, s44, v18")
    try:
        translate(changed)
    except KeyError as error:
        assert error.args == ("v20",), error
    else:
        raise AssertionError("An untracked accumulator was accepted")


def check_e64_approx_draws(directory, compiler, renderer, backend):
    """Draw APPROX_E64_BODY once per row of APPROX_E64_ROWS, and once per wrong reference on the first row.

    One compile and one renderer batch. A renderer that reports an unavailable adapter (exit 77) makes this a
    skip. On a desktop GPU the draws establish what the translated shader computes, not what the console does.
    """
    source = directory / "e64_approx.hlsl"
    shader = directory / "e64_approx.cso"
    hlsl = translate(with_constants(APPROX_E64_BODY))
    source.write_text(hlsl, encoding="ascii")
    values = [APPROX_E64_ACCUMULATOR]
    for a, b, c, d in APPROX_E64_ROWS:
        exp, log, product, square = 2.0 ** -a, math.log2(abs(b)), single(-a * abs(b)), single(d * d)
        values += [a, b, c, d, -a, abs(b), -d, exp, exp + 0.5, log, log + 0.5, product, square,
                   *approx_e64_reference(a, b, c, d)]
        values += [float(wrong(a, b, c, d)) for _, wrong, _ in APPROX_E64_WRONG]
    mode = log2_denorm_mode("e64_approx", hlsl, *values)
    assert mode == LOG2_DENORM_MODE
    compile_shader(compiler, source, shader, mode=mode)
    cases = []

    def add_case(row, values, status):
        constants = bytearray(256)
        struct.pack_into("<4f", constants, 12, *row)
        path = directory / f"e64_approx-{len(cases)}.case"
        path.write_bytes(bytes(constants) + struct.pack("<4f", *values) * 64)
        cases.append((path, status))

    for index, row in enumerate(APPROX_E64_ROWS):
        right = approx_e64_reference(*row)
        # The true exp2 and log2 results are integers, every right value survives the binary16 export
        # unchanged, and the unfused sum is exactly zero where a fused one is 1.0.
        assert right[0] == int(right[0]) and right[1] == int(right[1]) and right[3] == 0.0, right
        assert tuple(map(half_rtz, right)) == right, right
        add_case(row, right, "pass")
        if index:
            continue
        for channel, wrong, reason in APPROX_E64_WRONG:
            values = list(right)
            values[channel] = float(wrong(*row))
            assert values[channel] != right[channel], reason
            add_case(row, values, "mismatch")
    assert (cases[0][1], cases[-1][1]) == ("pass", "pass")
    try:
        run_pixel_batch(renderer, backend, shader, cases)
    except subprocess.CalledProcessError as error:
        if error.returncode == 77:
            return 77
        raise
    return 0


def predicate_group_e64(first):
    body = ["v_sub_f32_e64 v18, s45, 0", "v_sub_f32_e64 v19, s47, 0"]
    for channel, op in enumerate(PREDICATES[first:first + 4]):
        body.extend([f"v_cmp_{op}_f32_e64 s0, -s44, -v18", f"v_cndmask_b32_e64 v{12 + channel}, 0, v19, s0"])
    return with_constants(body)


def run_pixel_batch_statuses(renderer, backend, shader, paths):
    command = [renderer, backend, "--check-pixel-batch", shader, *paths]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 77:
        result.check_returncode()
    statuses = []
    for i, line in enumerate(line for line in result.stdout.splitlines() if line.startswith("case=")):
        prefix = f"case={i} status="
        if not line.startswith(prefix):
            raise AssertionError(f"Unexpected batch line {line!r}\n{result.stdout}\n{result.stderr}")
        statuses.append(line[len(prefix):])
    if len(statuses) != len(paths) or any(status not in ("pass", "mismatch") for status in statuses):
        raise AssertionError(f"Unexpected batch statuses {statuses}\n{result.stdout}\n{result.stderr}")
    if result.returncode != int("mismatch" in statuses):
        raise AssertionError(f"Unexpected batch exit code {result.returncode}\n{result.stdout}\n{result.stderr}")
    return statuses, result


def run_pixel_batch(renderer, backend, shader, cases):
    statuses, result = run_pixel_batch_statuses(renderer, backend, shader, [path for path, _ in cases])
    expected = [status for _, status in cases]
    if statuses != expected:
        raise AssertionError(f"Expected {expected}, got {statuses}\n{result.stdout}\n{result.stderr}")
    return result


def check_e64_draws(directory, compiler, renderer, backend):
    source = directory / "e64.hlsl"
    shader = directory / "e64.cso"
    def add_case(cases, constants, expected, status):
        path = directory / f"e64-{len(cases)}.case"
        path.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        cases.append((path, status))

    x, y, z, w = E64_ARITH_INPUTS
    assert [math.floor(-abs(x)), -y - math.floor(-y), max(-z, abs(w)), min(z, -abs(w))] == E64_ARITH[0][1]
    assert [-x * abs(y) + z, 0.0, 0.0, 0.0] == E64_ARITH[1][1]
    for body, expected, wrongs in E64_ARITH:
        source.write_text(translate(with_constants(body)), encoding="ascii")
        compile_shader(compiler, source, shader)
        constants = bytearray(256)
        struct.pack_into("<4f", constants, 12, *E64_ARITH_INPUTS)
        cases = []
        add_case(cases, constants, expected, "pass")
        for channel, value in wrongs:
            wrong_values = list(expected)
            wrong_values[channel] = value
            add_case(cases, constants, wrong_values, "mismatch")
        add_case(cases, constants, expected, "pass")
        run_pixel_batch(renderer, backend, shader, cases)
    inputs = ((-1, 1, 0), (-0.0, 0.0, 1), (1, -1, 2), (math.nan, 1, 3), (1, math.nan, 3))
    negated = {0: 2, 1: 1, 2: 0, 3: 3}
    for first in range(0, 16, 4):
        source.write_text(translate(predicate_group_e64(first)), encoding="ascii")
        compile_shader(compiler, source, shader)
        cases = []
        for left, right, relation in inputs:
            constants = bytearray(256)
            struct.pack_into("<4f", constants, 12, left, right, 0, 1)
            expected = [float((op >> negated[relation]) & 1) for op in range(first, first + 4)]
            add_case(cases, constants, expected, "pass")
            # A lowering that dropped the negations would follow the original relation.
            unnegated = [float((op >> relation) & 1) for op in range(first, first + 4)]
            if unnegated != expected:
                add_case(cases, constants, unnegated, "mismatch")
        run_pixel_batch(renderer, backend, shader, cases)


def bitwise_program():
    # Emit exact finite binary32 colors by placing result bits directly in their mantissas.
    return with_constants([
        "v_mov_b32_e32 v18, s46",
        "v_mov_b32_e32 v19, s47",
        "v_lshlrev_b32_e32 v20, s44, v18",
        "v_lshrrev_b32_e32 v20, s45, v20",
        "v_ashrrev_i32_e32 v21, 20, v19",
        "v_not_b32_e32 v22, v18",
        "v_bfi_b32 v23, 0x00700000, v18, v19",
        "v_xor_b32_e32 v22, v19, v22",
        *[instruction for index in range(4) for instruction in (
            f"v_and_b32_e32 v{12+index}, 0x007f0000, v{20+index}",
            f"v_or_b32_e32 v{12+index}, 0x3f800000, v{12+index}",
        )],
    ])


def check_bitwise_draw(directory, compiler, renderer, backend):
    shader = directory / "bits.cso"
    source = directory / "bits.hlsl"
    for logical in (False, True):
        program = bitwise_program()
        if logical:
            program = [line.replace("v_ashrrev_i32", "v_lshrrev_b32") for line in program]
        source.write_text(translate(program), encoding="ascii")
        compile_shader(compiler, source, shader)
        cases = []
        for idx, (left, right, a, b) in enumerate(((36, 33, 0x40d10000, 0xc0250000), (35, 37, 0x3e270000, 0xbf620000))):
            words = (((a << (left % 32)) & 0xffffffff) >> (right % 32),
                     (b - 2**32) >> 20, ~a ^ b, (a & 0x00700000) | (b & ~0x00700000))
            colors = struct.pack("<4I", *[0x3f800000 | (value & 0x007f0000) for value in words])
            constants = bytearray(256)
            struct.pack_into("<4I", constants, 12, left, right, a, b)
            case = directory / f"bits-{int(logical)}-{idx}.case"
            case.write_bytes(constants + colors * 64)
            cases.append((case, "mismatch" if logical else "pass"))
        run_pixel_batch(renderer, backend, shader, cases)


def check_fract_draw(directory, compiler, renderer, backend):
    source = directory / "fract.hlsl"
    shader = directory / "fract.cso"
    source.write_text(translate(fract_values()), encoding="ascii")
    compile_shader(compiler, source, shader)
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, *FRACT_INPUTS)
    a, b, c, d = FRACT_INPUTS
    expected = [min(single(x - math.floor(x)), FRACT_CLAMP) for x in (a, abs(b), -abs(c), d)]
    assert expected[0] == FRACT_CLAMP and single(FRACT_INPUTS[0] - math.floor(FRACT_INPUTS[0])) == 1.0
    pass_case = directory / "fract-pass.case"
    write_case(pass_case, constants, tuple(map(half_rtz, expected)))
    # An unclamped fract exports 1.0 for the first input; the renderer must tell it from the clamp.
    wrong_case = directory / "fract-wrong.case"
    write_case(wrong_case, constants, (1.0, *map(half_rtz, expected[1:])))
    run_pixel_batch(renderer, backend, shader, [(wrong_case, "mismatch"), (pass_case, "pass")])


# v_mad_f32 and v_mac_f32 round the product before the add and flush binary32 denormals.
# 4097 * 4097 = 2**24 + 2**13 + 1 rounds to 2**24 + 2**13, so the unfused sum is 0 where a fused one is 1.0.
# 2**-127 is a denormal input: flushed, the second sum is 0.5; preserved, it is 1.0.
MAD_INPUTS = (4097.0, -16785408.0, 2.0 ** -127, 2.0 ** 126)
MAD_BODY = [
    "v_mad_f32 v12, s44, s44, s45",
    "v_mov_b32_e32 v13, s47",
    "v_mad_f32 v13, s46, v13, 0.5",
    "v_mov_b32_e32 v15, s44",
    "v_mov_b32_e32 v14, s45",
    "v_mac_f32_e32 v14, s44, v15",
    "v_mad_f32 v15, 0.5, 4.0, 0.5",
]


def check_three_buffer_draw(directory, compiler, renderer, backend):
    """Each channel comes from word 2 of a different buffer; every other word holds a sentinel."""
    source = directory / "three-buffers.hlsl"
    shader = directory / "three-buffers.cso"
    program = with_three_constant_buffers()
    source.write_text(translate_three_buffers(program), encoding="ascii")
    compile_shader(compiler, source, shader)

    def buffers(values):
        data = [bytearray(struct.pack("<f", 0.0625) * 64) for _ in values]
        for buffer, value in zip(data, values, strict=True):
            struct.pack_into("<f", buffer, 8, value)
        return data[0], bytes(data[1] + data[2])

    cases = []
    for idx, (values, expected, status) in enumerate((
            ((.25, -.5, .75), (.25, -.5, .75), "pass"),
            ((-.125, .625, .875), (-.125, .625, .875), "pass"),
            # The reference reads b1 and b2 exchanged, then b0 in place of b1.
            ((.25, -.5, .75), (.25, .75, -.5), "mismatch"),
            ((.25, -.5, .75), (.25, .25, .75), "mismatch"))):
        first, extra = buffers(values)
        case = write_case(directory / f"three-buffers-{idx}.case", first, (*expected, 1), extra_constants=extra)
        cases.append((case, status))
    run_pixel_batch(renderer, backend, shader, cases)
    # Exchanging the two contract entries exchanges b1 and b2 in the translation.
    exchanged = dict(THREE_BUFFERS, extra_constants=THREE_BUFFERS["extra_constants"][::-1])
    source.write_text(lower(program, 20, 24, [1, 2, 3, 4], **exchanged), encoding="ascii")
    compile_shader(compiler, source, shader)
    run_pixel_batch(renderer, backend, shader, [(cases[0][0], "mismatch"), (cases[2][0], "pass")])
    # A truncated pair of extra buffers is a case-format error, not a draw.
    short = directory / "three-buffers-short.case"
    short.write_bytes(cases[0][0].read_bytes()[:-4])
    broken = subprocess.run([renderer, backend, "--check-pixel-case", shader, short], capture_output=True, text=True)
    if broken.returncode != 1 or "Expected two 256-byte constant buffers for b1 and b2" not in broken.stderr:
        raise AssertionError(f"A short extra-constant case was accepted\n{broken.stdout}\n{broken.stderr}")


def check_mad_draw(directory, compiler, renderer, backend):
    source = directory / "mad.hlsl"
    shader = directory / "mad.cso"
    source.write_text(translate(with_constants(MAD_BODY)), encoding="ascii")
    compile_shader(compiler, source, shader)
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, *MAD_INPUTS)
    a, c, tiny, large = MAD_INPUTS
    unfused = single(single(a * a) + c)
    assert unfused == 0.0 and single(a * a + c) == 1.0
    assert single(tiny) == tiny and 0 < tiny < 2.0 ** -126 and single(tiny * large) == 0.5
    expected = [unfused, 0.5, unfused, 2.5]
    cases = []
    # A fused multiply-add or a denormal-preserving one exports 1.0; the renderer must tell each case apart.
    for idx, wrong_values in enumerate(([1.0, 0.5, unfused, 2.5], [unfused, 1.0, unfused, 2.5], [unfused, 0.5, 1.0, 2.5])):
        path = directory / f"mad-wrong-{idx}.case"
        path.write_bytes(constants + struct.pack("<4f", *map(half_rtz, wrong_values)) * 64)
        cases.append((path, "mismatch"))
    pass_case = directory / "mad-pass.case"
    pass_case.write_bytes(constants + struct.pack("<4f", *map(half_rtz, expected)) * 64)
    cases.append((pass_case, "pass"))
    run_pixel_batch(renderer, backend, shader, cases)


# v_madak_f32 computes S0 * S1 + K and v_madmk_f32 computes S0 * K + S1, with K a 32-bit literal.
# Unlike the fma forms they round the product before the add and flush binary32 denormals, like v_mad_f32.
# 4097 * 4097 - 16785408 is 0 unfused and 1.0 fused; 2^-127 * 2^126 + 0.5 is 0.5 with the denormal input flushed.
MAD_LITERAL_BODY = [
    "v_mov_b32_e32 v13, s44",
    "v_madak_f32 v12, s44, v13, 0xcb801000",
    "v_mov_b32_e32 v13, s45",
    "v_madmk_f32 v13, s44, 0x45800800, v13",
    "v_mov_b32_e32 v15, s47",
    "v_madak_f32 v14, s46, v15, 0x3f000000",
    "v_mov_b32_e32 v15, 0.5",
    "v_madmk_f32 v15, s46, 0x7e800000, v15",
]


def check_mad_literal_draw(directory, compiler, renderer, backend):
    source = directory / "mad_literal.hlsl"
    shader = directory / "mad_literal.cso"
    source.write_text(translate(with_constants(MAD_LITERAL_BODY)), encoding="ascii")
    compile_shader(compiler, source, shader)
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, *MAD_INPUTS)
    a, c, tiny, large = MAD_INPUTS
    # Exact rational arithmetic gives the fused result; the unfused one rounds the product to binary32 first.
    fused = float(Fraction(a) * Fraction(a) + Fraction(c))
    unfused = single(single(a * a) + c)
    assert fused == 1.0 and unfused == 0.0
    assert single(tiny) == tiny and 0 < tiny < 2.0 ** -126 and single(tiny * large) == 0.5
    expected = [unfused, unfused, 0.5, 0.5]
    cases = []
    # A reference that fuses the first two rows or keeps the denormal in the last two exports 1.0 there.
    for k in range(4):
        wrong_values = list(expected)
        wrong_values[k] = 1.0
        path = directory / f"mad_literal-wrong-{k}.case"
        write_case(path, constants, tuple(map(half_rtz, wrong_values)))
        cases.append((path, "mismatch"))
    pass_case = directory / "mad_literal-pass.case"
    write_case(pass_case, constants, tuple(map(half_rtz, expected)))
    cases.append((pass_case, "pass"))
    run_pixel_batch(renderer, backend, shader, cases)


def check_predicate_draws(directory, compiler, renderer, backend):
    source = directory / "predicates.hlsl"
    shader = directory / "predicates.cso"
    # Isolate each negated relation so another predicate cannot supply the unordered test. An
    # unordered relational fcmp alone is not enough: the Windows 10 in-box WARP 10.0.19041
    # evaluates fcmp ult, ule, ugt and uge as their ordered forms.
    for op in ("nge", "nlg", "ngt", "nle", "nlt"):
        isolated = with_constants(["v_sub_f32_e64 v18, s45, 0",
                                   f"v_cmp_{op}_f32_e32 vcc_lo, s44, v18",
                                   "v_cndmask_b32_e32 v12, 0, 1.0, vcc_lo"])
        source.write_text(translate(isolated), encoding="ascii")
        listing = directory / f"{op}.ll"
        compile_shader(compiler, source, shader, listing=listing)
        dxil = listing.read_text(encoding="utf-8")
        if not any(text in dxil for text in ("@dx.op.isSpecialFloat.f32(i32 8,", "fcmp uno ", "fcmp ueq ")):
            raise AssertionError(f"{op.upper()} lacks an explicit unordered comparison in DXIL")
    inputs = (("lt", -1, 1, 0), ("signed-zero", -0.0, 0.0, 1), ("gt", 1, -1, 2),
              ("left-nan", math.nan, 1, 3), ("right-nan", 1, math.nan, 3))
    for first in range(0, 16, 4):
        source.write_text(translate(predicate_group(first)), encoding="ascii")
        compile_shader(compiler, source, shader)
        cases = []
        for label, left, right, relation in inputs:
            constants = bytearray(256)
            struct.pack_into("<4f", constants, 12, left, right, 0, 1)
            pixel = tuple((op >> relation) & 1 for op in range(first, first + 4))
            case = directory / f"predicates-{first}-{label}.case"
            write_case(case, constants, pixel)
            cases.append((case, "pass"))
        if first == 12:
            # Keep the unordered oracle and replace the final NaN operand with an equal value.
            struct.pack_into("<f", constants, 16, 1.0)
            wrong = directory / f"predicates-{first}-finite-instead-of-nan.case"
            write_case(wrong, constants, pixel)
            cases.insert(-1, (wrong, "mismatch"))
        try:
            run_pixel_batch(renderer, backend, shader, cases)
        except AssertionError as error:
            channels = ", ".join(f"{channel}={op}" for channel, op in enumerate(PREDICATES[first:first + 4]))
            error.add_note(f"Predicate draw on {backend}, first={first}, channels {channels}")
            raise


def check_int_predicate_draws(directory, compiler, renderer, backend):
    source = directory / "int_predicates.hlsl"
    shader = directory / "int_predicates.cso"
    # Normal and signed-zero IEEE-754 bit patterns avoid denormal/NaN canonicalization while covering
    # signed/unsigned order flips (-0.0 vs +0.0, +1.0 vs -1.0) and integer-vs-float order (-2.0 vs -1.0).
    pairs = (
        (0x80000000, 0x00000000),
        (0xbf800000, 0xbf800000),
        (0x3f800000, 0xbf800000),
        (0xc0000000, 0xbf800000),
    )

    def relation(left_u, right_u, kind):
        if kind == "i32":
            left_v = left_u - 0x100000000 if left_u & 0x80000000 else left_u
            right_v = right_u - 0x100000000 if right_u & 0x80000000 else right_u
        else:
            left_v, right_v = left_u, right_u
        return 0 if left_v < right_v else (1 if left_v == right_v else 2)

    for kind in ("i32", "u32"):
        for encoding in ("e32", "e64"):
            for first in (0, 4):
                source.write_text(translate(int_predicate_group(kind, encoding, first)), encoding="ascii")
                compile_shader(compiler, source, shader)
                cases = []
                for left_u, right_u in pairs:
                    rel = relation(left_u, right_u, kind)
                    constants = bytearray(256)
                    struct.pack_into("<IIff", constants, 12, left_u, right_u, 0.0, 1.0)
                    pixel = struct.pack("<4f", *((op >> rel) & 1 for op in range(first, first + 4)))
                    case = directory / f"int_predicates-{kind}-{encoding}-{first}-{left_u:08x}-{right_u:08x}.case"
                    case.write_bytes(constants + pixel * 64)
                    cases.append((case, "pass"))
                if (kind, encoding, first) == ("i32", "e32", 0):
                    # Negative control: the signed shader against the unsigned oracle on (0x80000000, 0x00000000).
                    constants = bytearray(256)
                    struct.pack_into("<IIff", constants, 12, 0x80000000, 0x00000000, 0.0, 1.0)
                    unsigned_rel = relation(0x80000000, 0x00000000, "u32")
                    wrong_pixel = struct.pack("<4f", *((op >> unsigned_rel) & 1 for op in range(4)))
                    wrong = directory / f"int_predicates-{kind}-{encoding}-{first}-unsigned-reference.case"
                    wrong.write_bytes(constants + wrong_pixel * 64)
                    cases.insert(-1, (wrong, "mismatch"))
                try:
                    run_pixel_batch(renderer, backend, shader, cases)
                except AssertionError as error:
                    error.add_note(f"Integer predicate draw on {backend}, {kind} {encoding} first={first}")
                    raise


def check_unsigned_less_than_draws(directory, compiler, renderer, backend):
    source = directory / "unsigned_less_than.hlsl"
    shader = directory / "unsigned_less_than.cso"
    source.write_text(translate(unsigned_less_than()), encoding="ascii")
    compile_shader(compiler, source, shader)

    cases = []
    # Quieting the first word of the NaN pair would reverse the unsigned ordering.
    pairs = (("sign-boundary", 0x80000000, 0x7fffffff),
             ("equal", 0x7fffffff, 0x7fffffff),
             ("nan-payload", 0x7f800002, 0x7fc00001))
    for label, left, right in pairs:
        constants = bytearray(256)
        struct.pack_into("<II", constants, 12, left, right)
        order = (float(left < right), float(right < left))
        case = directory / f"unsigned-less-than-{label}.case"
        write_case(case, constants, order * 2)
        cases.append((case, "pass"))
    run_pixel_batch(renderer, backend, shader, cases)

    source.write_text(translate(unsigned_less_than("i32")), encoding="ascii")
    compile_shader(compiler, source, shader)
    run_pixel_batch(renderer, backend, shader, [(cases[0][0], "mismatch")])


# Each word is a normal binary32 value, so no float move can flush or quiet it, and its upper half differs
# from the lower one wherever a 32-bit comparison would give another answer.
INT16_PAIRS = (
    (0x3f808000, 0x40000000),  # 0x8000 vs 0: signed less, unsigned greater
    (0x40000005, 0x3f800005),  # equal low halves; the whole words compare greater
    (0xbf80ffff, 0x3f800001),  # -1 vs 1 signed, 0xffff vs 1 unsigned
    (0x3f807fff, 0xbf808000),  # 0x7fff vs 0x8000; right equals the 0x8000 literal
    (0x3f80fff0, 0x3f800040),  # -16 vs 64; right equals the inline constant 64
    (0x3f800040, 0x3f80fff0),  # 64 vs -16; right equals the inline constant -16
)


def check_int16_predicate_draws(directory, compiler, renderer, backend):
    source = directory / "int16_predicates.hlsl"
    shader = directory / "int16_predicates.cso"
    case = directory / "int16_predicates.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]

    def build(program):
        source.write_text(translate(program), encoding="ascii")
        compile_shader(compiler, source, shader)

    def write_case(left, right, expected, path=case):
        constants = bytearray(256)
        struct.pack_into("<IIff", constants, 12, left, right, 0.0, 1.0)
        path.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        return path

    for kind in ("i16", "u16"):
        for encoding in ("e32", "e64"):
            for group in (0, 1):
                build(int16_group(kind, encoding, group))
                cases = []
                for left, right in INT16_PAIRS:
                    path = directory / f"int16_predicates-{kind}-{encoding}-{group}-{left:08x}-{right:08x}.case"
                    write_case(left, right, int16_reference(kind, encoding, group, left, right), path)
                    cases.append((path, "pass"))
                try:
                    run_pixel_batch(renderer, backend, shader, cases)
                except AssertionError as error:
                    error.add_note(f"16-bit predicate draw on {backend}, {kind} {encoding} group {group}")
                    raise

    def must_mismatch(program, left, right, expected, message):
        build(program)
        write_case(left, right, expected)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(message)

    left, right = INT16_PAIRS[0]
    must_mismatch(int16_group("i16", "e32", 0), left, right, int16_reference("u16", "e32", 0, left, right),
                  "A signed 16-bit comparison matched the unsigned reference")
    left, right = INT16_PAIRS[1]
    swapped = [line.replace("v_cmp_eq_i16", "v_cmp_ne_i16") for line in int16_group("i16", "e64", 0)]
    must_mismatch(swapped, left, right, int16_reference("i16", "e64", 0, left, right),
                  "Replacing eq with ne escaped detection")
    widened = [line.replace("_u16_", "_u32_") for line in int16_group("u16", "e32", 0)]
    must_mismatch(widened, left, right, int16_reference("u16", "e32", 0, left, right),
                  "A 32-bit comparison matched the 16-bit reference")


def check_cmp64_controls():
    """64-bit integer comparisons: lowering, destinations, sources and the refused neighbours."""
    forms = set()
    for kind in ("i64", "u64"):
        for encoding in ("e32", "e64"):
            for group in (0, 1):
                program = cmp64_group(kind, encoding, group)
                translate_int(program)
                forms.update(line.split()[0] for line in program if line.startswith("v_cmp_"))
                # The wave64 spellings (vcc, s[52:53]) lower to the same HLSL as the wave32 ones.
                assert translate_int(cmp64_group(kind, encoding, group, wave64=True)) == translate_int(program)
    assert forms == {f"v_cmp_{op}_{kind}_{enc}" for op in INT_PREDICATES for kind in ("i64", "u64")
                     for enc in ("e32", "e64")}, forms
    # Without a suffix LLVM 15 picks e32 when the operands fit it and e64 otherwise (it always prints the suffix).
    assert (translate_int(cmp64_program(["v_cmp_lt_u64 vcc_lo, v[18:19], v[20:21]"]))
            == translate_int(cmp64_program(["v_cmp_lt_u64_e32 vcc_lo, v[18:19], v[20:21]"])))

    def error(line, message, program=None):
        expect_value_error(translate_int, cmp64_program([line]) if program is None else program, message)

    # Raw words: each word is read with asuint (asint for a signed high word) and nothing else is applied.
    signed_base = cmp64_group("i64", "e32", 0)
    unsigned_base = cmp64_group("u64", "e32", 0)
    signed_source = translate_int(signed_base)
    unsigned_source = translate_int(unsigned_base)
    first = signed_base.index("v_cmp_lt_i64_e32 vcc_lo, v[18:19], v[20:21]")
    movs = [signed_base.index(f"v_mov_b32_e32 v{18 + n}, s{44 + n}") for n in range(4)]
    for source, high in ((signed_source, "int"), (unsigned_source, "uint")):
        convert = "asint" if high == "int" else "asuint"
        for line in (f"uint cmp{first}_l0 = asuint(alu{movs[0]});", f"{high} cmp{first}_h0 = {convert}(alu{movs[1]});",
                     f"uint cmp{first}_l1 = asuint(alu{movs[2]});", f"{high} cmp{first}_h1 = {convert}(alu{movs[3]});",
                     f"bool cmp{first} = (cmp{first}_h0 < cmp{first}_h1) || "
                     f"(cmp{first}_h0 == cmp{first}_h1 && cmp{first}_l0 < cmp{first}_l1);"):
            assert "    " + line + "\n" in source, (high, line)
    assert signed_source != unsigned_source
    # The overlapping pair v[18:19] against v[19:20] reads v19 twice, as the high word of the first operand and as
    # the low word of the second, and each read gets the register's single value.
    overlap = signed_base.index("v_cmp_le_i64_e32 vcc_lo, v[18:19], v[19:20]")
    for line in (f"int cmp{overlap}_h0 = asint(alu{movs[1]});", f"uint cmp{overlap}_l1 = asuint(alu{movs[1]});",
                 f"uint cmp{overlap}_l0 = asuint(alu{movs[0]});", f"int cmp{overlap}_h1 = asint(alu{movs[2]});",
                 f"bool cmp{overlap} = (cmp{overlap}_h0 < cmp{overlap}_h1) || "
                 f"(cmp{overlap}_h0 == cmp{overlap}_h1 && cmp{overlap}_l0 <= cmp{overlap}_l1);"):
        assert "    " + line + "\n" in signed_source, line
    # The remaining predicates, from the same four words.
    for op, expression in (("eq", "{h0} == {h1} && {l0} == {l1}"), ("ne", "!({h0} == {h1} && {l0} == {l1})"),
                           ("gt", "({h0} > {h1}) || ({h0} == {h1} && {l0} > {l1})"),
                           ("ge", "({h0} > {h1}) || ({h0} == {h1} && {l0} >= {l1})"), ("f", "false"), ("t", "true")):
        line = f"v_cmp_{op}_u64_e64 s52, v[18:19], v[20:21]"
        program = cmp64_program([line])
        number = program.index(line)
        names = {name: f"cmp{number}_{name}" for name in ("l0", "h0", "l1", "h1")}
        assert f"    bool cmp{number} = {expression.format(**names)};\n" in translate_int(program), op

    # Inline constants are sign-extended: -1 and -16 fill the high word, 0, 1 and 64 clear it.
    for token, low, high in (("0", 0, 0), ("1", 1, 0), ("64", 64, 0), ("-1", 0xffffffff, 0xffffffff),
                             ("-16", 0xfffffff0, 0xffffffff)):
        line = f"v_cmp_lt_u64_e64 s52, {token}, v[20:21]"
        program = cmp64_program([line])
        number = program.index(line)
        source = translate_int(program)
        assert f"    uint cmp{number}_l0 = {low}u;\n" in source and f"    uint cmp{number}_h0 = {high}u;\n" in source, token
        line = line.replace("u64", "i64")
        program = cmp64_program([line])
        number = program.index(line)
        assert f"    int cmp{number}_h0 = asint({high}u);\n" in translate_int(program), token
    # Destinations: vcc and vcc_lo on e32; vcc_lo, vcc, an SGPR or an aligned SGPR pair on e64.
    for dst in ("vcc_lo", "vcc", "s0", "s52", "s105", "s[52:53]", "s[104:105]"):
        translate_int(cmp64_program([f"v_cmp_ge_u64_e64 {dst}, v[18:19], v[20:21]"]))
    for dst in ("vcc_lo", "vcc"):
        translate_int(cmp64_program([f"v_cmp_ge_u64_e32 {dst}, v[18:19], v[20:21]"]))
    # Sources: the top VGPR pair, two SGPR pairs on e64 (constant-bus legality is not sourced) and an inline src1.
    translate_int(cmp64_program(["v_mov_b32_e32 v254, s44", "v_mov_b32_e32 v255, s45",
                                 "v_cmp_lt_i64_e64 s52, v[254:255], v[19:20]"]))
    translate_int(cmp64_program(["v_cmp_lt_i64_e64 s52, s[44:45], s[46:47]", "v_cmp_lt_i64_e64 s52, v[18:19], 64"]))

    # Destination rules.
    e32_destination = "64-bit comparison e32 destination must be vcc"
    for dst in ("s52", "s[52:53]"):
        error(f"v_cmp_lt_i64_e32 {dst}, v[18:19], v[20:21]", e32_destination)
        assert (translate_int(cmp64_program([f"v_cmp_lt_u64 {dst}, v[18:19], v[20:21]"]))
                == translate_int(cmp64_program([f"v_cmp_lt_u64_e64 {dst}, v[18:19], v[20:21]"]))), dst
    error("v_cmp_lt_i64_e64 s[45:46], v[18:19], v[20:21]", "Invalid 64-bit scalar register pair: s[45:46]")
    error("v_cmp_lt_i64_e64 s106, v[18:19], v[20:21]", "Scalar register out of range: s106")
    error("v_cmp_lt_i64_e64 s[106:107], v[18:19], v[20:21]", "Invalid 64-bit scalar register pair: s[106:107]")
    error("v_cmp_lt_i64_e64 s[52:54], v[18:19], v[20:21]", "Descriptor or vector width mismatch")
    error("v_cmp_lt_i64_e64 exec, v[18:19], v[20:21]", "Unsupported instruction")
    error("v_cmp_lt_i64_e64 vcc_hi, v[18:19], v[20:21]", "Unsupported instruction")
    # Source rules.
    invalid = "Invalid 64-bit comparison operand: "
    for kind, encoding, dst in (("i64", "e64", "s52"), ("u64", "e32", "vcc_lo")):
        for token in ("v[18:20]", "v[19:18]", "v[18:18]", "v[255:256]", "v[254:256]", "v[256:257]", "v[018:019]",
                      "v18", "s44", "vcc", "exec", "1.0", "-4.0", "0x10", "0x12345678", "0x100000000", "65", "-17",
                      "-0", "-v[18:19]", "|v[18:19]|", "abs(v[18:19])", "s44 clamp", "v[18:19] clamp"):
            error(f"v_cmp_lt_{kind}_{encoding} {dst}, {token}, v[20:21]", invalid + token)
    error("v_cmp_lt_i64_e64 s52, v[18:19], v[20:21] clamp", invalid + "v[20:21] clamp")
    error("v_cmp_lt_i64_e64 s52, v[18:19], v[20:21] op_sel:[1,0,0]", "Unsupported instruction")
    error("v_cmp_lt_i64_e64 s52, v[18:19], 1.0", invalid + "1.0")
    error("v_cmp_lt_i64_e64 s52, v[18:19], 0x40", invalid + "0x40")
    error("v_cmp_lt_i64_e64 s52, v[18:19], v[255:256]", invalid + "v[255:256]")
    error("v_cmp_lt_i64_e64 s52, s[45:46], v[20:21]", "Invalid 64-bit scalar register pair: s[45:46]")
    error("v_cmp_lt_i64_e64 s52, v[20:21], s[45:46]", "Invalid 64-bit scalar register pair: s[45:46]")
    error("v_cmp_lt_i64_e64 s52, s[44:46], v[20:21]", "Descriptor or vector width mismatch")
    src1 = "64-bit comparison e32 src1 must be a VGPR pair"
    for token in ("s[44:45]", "64", "-1", "s44", "v18", "1.0", "0x10"):
        error(f"v_cmp_lt_i64_e32 vcc_lo, v[18:19], {token}", src1)
        if token in ("s[44:45]", "64", "-1"):  # valid e64 sources: the suffix-less spelling becomes e64
            assert (translate_int(cmp64_program([f"v_cmp_lt_u64 vcc_lo, v[18:19], {token}"]))
                    == translate_int(cmp64_program([f"v_cmp_lt_u64_e64 vcc_lo, v[18:19], {token}"]))), token
        else:
            error(f"v_cmp_lt_u64 vcc_lo, v[18:19], {token}", invalid + token)
    # The exec-writing forms and the other compare families stay refused.
    for line in ("v_cmpx_lt_i64_e32 v[18:19], v[20:21]", "v_cmpx_lt_u64_e64 v[18:19], v[20:21]",
                 "v_cmpx_lt_i64_e64 s52, v[18:19], v[20:21]", "v_cmp_lt_f64_e64 s52, v[18:19], v[20:21]",
                 "v_cmp_class_f64_e32 vcc_lo, v[18:19], v20", "v_cmp_lt_i64_sdwa vcc_lo, v[18:19], v[20:21]",
                 "v_cmp_lt_i64_dpp vcc_lo, v[18:19], v[20:21] quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
                 "v_cmp_ngt_i64_e64 s52, v[18:19], v[20:21]", "v_cmp_lg_u64_e64 s52, v[18:19], v[20:21]"):
        error(line, "Unsupported instruction")
    # Untracked sources, including a pair with one tracked half.
    untracked = "Untracked 64-bit comparison source: "
    for line, token in (("v_cmp_lt_u64_e64 s52, v[60:61], v[20:21]", "v[60:61]"),
                        ("v_cmp_lt_u64_e64 s52, v[18:19], v[60:61]", "v[60:61]"),
                        ("v_cmp_lt_u64_e64 s52, v[254:255], v[20:21]", "v[254:255]"),
                        ("v_cmp_lt_u64_e32 vcc_lo, v[18:19], v[60:61]", "v[60:61]"),
                        ("v_cmp_lt_u64_e64 s52, s[60:61], v[20:21]", "s[60:61]"),
                        ("v_cmp_lt_u64_e64 s52, v[18:19], s[60:61]", "s[60:61]")):
        error(line, untracked + token)
    for tracked in ("v30", "v31"):
        compare = "v_cmp_lt_u64_e64 s52, v[30:31], v[20:21]"
        error("", untracked + "v[30:31]", cmp64_program([f"v_mov_b32_e32 {tracked}, s44", compare]))
    # A scalar load still pending, a pending destination, an outstanding sample and a masked region.
    base = cmp64_program(["v_cmp_lt_u64_e64 s52, v[18:19], v[20:21]"])
    early = base.index("s_waitcnt lgkmcnt(0)", base.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc"))
    for line, message in (("v_cmp_lt_u64_e64 s52, s[44:45], -1", "Constant read before load completion"),
                          ("v_cmp_lt_u64_e64 s52, -1, s[46:47]", "Constant read before load completion"),
                          ("v_cmp_lt_u64_e64 s45, -1, 64", "Invalid comparison"),
                          ("v_cmp_lt_u64_e64 s[44:45], -1, 64", "Invalid comparison"),
                          ("v_cmp_lt_u64_e32 vcc_lo, -1, v[20:21]", untracked + "v[20:21]")):
        error("", message, base[:early] + [line] + base[early:])
    wait = base.index("s_waitcnt vmcnt(0)")
    for line in ("v_cmp_lt_u64_e64 s52, v[12:13], v[14:15]", "v_cmp_lt_u64_e64 s52, v[14:15], 64",
                 "v_cmp_lt_u64_e32 vcc_lo, -1, v[14:15]"):
        error("", "Sample used before vmcnt wait", base[:wait] + [line] + base[wait:])
    masked = base.index("s_wqm_b64 exec, exec") + 1
    error("", "Invalid comparison", base[:masked] + ["v_cmp_lt_u64_e32 vcc_lo, -1, v[20:21]"] + base[masked:])

    # Python pre-check: the word list separates each wrong implementation from the right one. That the GPU
    # detects them is proven by the draw, which compares its output with these expected values.
    for kind in ("i64", "u64"):
        signed = kind == "i64"
        rows = [(words[:2], words[2:]) for words in CMP64_WORDS]
        for name, mutant in (("other signedness", lambda a, b: cmp64_truth("lt", not signed, a, b)),
                             ("swapped words", lambda a, b: cmp64_truth("lt", signed, a[::-1], b[::-1])),
                             ("low words only", lambda a, b: a[0] < b[0]),
                             ("high words only", lambda a, b: cmp64_truth("lt", signed, (0, a[1]), (0, b[1]))),
                             ("le", lambda a, b: cmp64_truth("le", signed, a, b))):
            assert any(mutant(a, b) != cmp64_truth("lt", signed, a, b) for a, b in rows), (kind, name)
    assert not cmp64_truth("lt", False, (0, 1), (0xffffffff, 0))
    assert cmp64_truth("lt", True, (0xffffffff, 0xffffffff), (0, 0))
    assert not cmp64_truth("lt", False, (0xffffffff, 0xffffffff), (0, 0))
    assert cmp64_truth("lt", False, (0xffffffff, 0x7fffffff), (0, 0x80000000))
    assert not cmp64_truth("lt", True, (0xffffffff, 0x7fffffff), (0, 0x80000000))
    assert cmp64_truth("lt", False, (2, 1), (2, 0x80000001)) and not cmp64_truth("lt", True, (2, 1), (2, 0x80000001))


def check_cmp64_draw(directory, compiler, renderer, backend):
    """64-bit integer comparisons against the Python reference, then mutations that must mismatch.

    The words travel through the constant buffer and v_mov_b32 copies and are compared as raw bits; they include
    patterns that are not normal binary32 values. The draw is compiled with -denorm preserve like the other
    bit-exact integer draws. It pins the translator's model on the desktop adapter; it is not evidence for gfx1013.
    """
    source = directory / "cmp64.hlsl"
    shader = directory / "cmp64.cso"

    def write_cases(kind, encoding, group, tag):
        paths = []
        for index, words in enumerate(CMP64_WORDS):
            constants = bytearray(256)
            struct.pack_into("<4I", constants, 12, *words)
            path = directory / f"cmp64-{tag}-{index}.case"
            write_case(path, bytes(constants), cmp64_expected(kind, encoding, group, words), integer_rt=True)
            paths.append(path)
        return paths

    def build(text):
        source.write_text(text, encoding="ascii")
        compile_shader(compiler, source, shader)

    for kind in ("i64", "u64"):
        for encoding in ("e32", "e64"):
            for group in (0, 1):
                tag = f"{kind}-{encoding}-{group}"
                build(translate_int(cmp64_group(kind, encoding, group)))
                try:
                    run_pixel_batch(renderer, backend, shader,
                                    [(path, "pass") for path in write_cases(kind, encoding, group, tag)])
                except AssertionError as error:
                    error.add_note(f"64-bit comparison draw on {backend}, {kind} {encoding} group {group}")
                    raise

    # Mutations of the translated lt (channel 0 of the signed e32 group): each must fail the reference.
    program = cmp64_group("i64", "e32", 0)
    lowered = translate_int(program)
    number = program.index("v_cmp_lt_i64_e32 vcc_lo, v[18:19], v[20:21]")
    names = {name: f"cmp{number}_{name}" for name in ("l0", "h0", "l1", "h1")}
    original = f"    bool cmp{number} = {pixel_aot.INT64_COMPARE['lt'].format(**names)};\n"
    assert original in lowered
    cases = write_cases("i64", "e32", 0, "mutant")

    def mismatches(mutated, label):
        assert mutated != lowered, label
        build(mutated)
        statuses, _ = run_pixel_batch_statuses(renderer, backend, shader, cases)
        if "mismatch" not in statuses:
            raise AssertionError(f"Mutating {label} escaped detection")

    def with_expression(expression):
        return lowered.replace(original, f"    bool cmp{number} = {expression};\n")

    swapped = dict(l0=names["h0"], h0=names["l0"], l1=names["h1"], h1=names["l1"])
    mismatches(with_expression(pixel_aot.INT64_COMPARE["lt"].format(**swapped)), "high and low words swapped")
    mismatches(with_expression(f"{names['l0']} < {names['l1']}"), "only the low words compared")
    mismatches(with_expression(f"{names['h0']} < {names['h1']}"), "only the high words compared")
    mismatches(with_expression(pixel_aot.INT64_COMPARE["le"].format(**names)), "lt replaced with le")
    unsigned = lowered
    for word in ("h0", "h1"):
        unsigned = unsigned.replace(f"    int {names[word]} = ", f"    uint {names[word]} = ")
    mismatches(unsigned, "signed compared as unsigned")
    # An overlapping pair that reads the wrong register is detected as well.
    overlap = program.index("v_cmp_le_i64_e32 vcc_lo, v[18:19], v[19:20]")
    moved = [line.replace("v_cmp_le_i64_e32 vcc_lo, v[18:19], v[19:20]", "v_cmp_le_i64_e32 vcc_lo, v[18:19], v[20:21]")
             for line in program]
    assert moved != program and program[overlap].endswith("v[19:20]")
    mismatches(translate_int(moved), "overlapping pair read as v[20:21]")


# The SDWA comparisons with whole-dword selections repeat the 16-predicate draws, including NaN on either side.
# For each group a reference that inverts the unordered (NaN) result must report a mismatch.
def check_sdwa_predicate_draws(directory, compiler, renderer, backend):
    source = directory / "sdwa_predicates.hlsl"
    shader = directory / "sdwa_predicates.cso"
    inputs = (("lt", -1, 1, 0), ("signed-zero", -0.0, 0.0, 1), ("gt", 1, -1, 2),
              ("left-nan", math.nan, 1, 3), ("right-nan", 1, math.nan, 3))
    for first in range(0, 16, 4):
        source.write_text(translate(predicate_group(first, sdwa=True)), encoding="ascii")
        compile_shader(compiler, source, shader)
        cases = []
        for label, left, right, relation in inputs:
            constants = bytearray(256)
            struct.pack_into("<4f", constants, 12, left, right, 0, 1)
            expected = [float((op >> relation) & 1) for op in range(first, first + 4)]
            case = directory / f"sdwa_predicates-{first}-{label}.case"
            case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
            cases.append((case, "pass"))
        # Keep the final NaN input and invert its reference; a passing case follows the control.
        wrong = directory / f"sdwa_predicates-{first}-inverted-unordered.case"
        wrong.write_bytes(constants + struct.pack("<4f", *(1.0 - value for value in expected)) * 64)
        cases.insert(-1, (wrong, "mismatch"))
        try:
            run_pixel_batch(renderer, backend, shader, cases)
        except AssertionError as error:
            channels = ", ".join(f"{channel}={op}" for channel, op in enumerate(PREDICATES[first:first + 4]))
            error.add_note(f"SDWA predicate draw on {backend}, first={first}, channels {channels}")
            raise


def half_rtz(x):
    bits, = struct.unpack("<H", struct.pack("<e", x))
    rounded, = struct.unpack("<e", struct.pack("<H", bits))
    return struct.unpack("<e", struct.pack("<H", bits - (abs(rounded) > abs(x))))[0]


# v_cvt_f32_f16 widens the binary16 in the low half of S0; the high half holds junk that must be ignored.
# The second channel applies -|x| to -5.0, the third is the largest finite half and the fourth is the
# smallest subnormal half, 2**-24, scaled by 2**24 so a flushed subnormal exports 0 instead of 1.0.
HALF_WORDS = (0xdead3c00, 0x0000c500, 0x00007bff, 0xbeef0001)
HALF_BODY = [
    "v_cvt_f32_f16_e32 v12, s44",
    "v_cvt_f32_f16_e64 v13, -|s45|",
    "v_cvt_f32_f16_e32 v14, s46",
    "v_cvt_f32_f16_e32 v15, s47",
    "v_mul_f32_e32 v15, 0x4b800000, v15",
]


def half_value(word):
    return struct.unpack("<e", struct.pack("<H", word & 0xffff))[0]


def check_half_draw(directory, compiler, renderer, backend):
    source = directory / "half.hlsl"
    shader = directory / "half.cso"
    source.write_text(translate(with_constants(HALF_BODY)), encoding="ascii")
    compile_shader(compiler, source, shader)
    constants = bytearray(256)
    struct.pack_into("<4I", constants, 12, *HALF_WORDS)
    first, second, third, fourth = map(half_value, HALF_WORDS)
    expected = [first, -abs(second), third, fourth * 2.0 ** 24]
    assert expected == [1.0, -5.0, 65504.0, 1.0] and fourth == 2.0 ** -24
    # Reading the high half, dropping the negation, a wrong exponent and a flushed subnormal.
    wrong = [half_value(HALF_WORDS[0] >> 16), 5.0, 32752.0, 0.0]
    for x in (*expected, *wrong):
        assert half_rtz(x) == x, x
    cases = []
    for k in range(4):
        wrong_values = list(expected)
        wrong_values[k] = wrong[k]
        path = directory / f"half-wrong-{k}.case"
        write_case(path, constants, tuple(wrong_values))
        cases.append((path, "mismatch"))
    pass_case = directory / "half-pass.case"
    write_case(pass_case, constants, tuple(expected))
    cases.append((pass_case, "pass"))
    run_pixel_batch(renderer, backend, shader, cases)

    # A separate export observes the literal and two scalar results as half encodings.
    body = [
        "v_cvt_f32_f16_e32 v12, 0x3c00",
        "s_ashr_i64 s[72:73], 0xf000, 2",
        "v_cvt_f32_f16_e32 v13, s72",
        "s_bfm_b64 s[72:73], 16, 0",
        "s_and_b64 s[72:73], s[72:73], 0x3c00",
        "v_cvt_f32_f16_e32 v14, s72",
        *HALF_BODY[3:],
    ]
    source.write_text(translate(with_constants(body)), encoding="ascii")
    compile_shader(compiler, source, shader)
    scalars = [half_value(0x3c00), half_value(0xf000 // 4), half_value((2**16 - 1) & 0x3c00), fourth * 2.0 ** 24]
    assert scalars == [1.0, 1.0, 1.0, 1.0]
    # A shift by zero leaves 0xf000 (-8192.0) in green.
    wrong_values = [scalars[0], half_value(0xf000), *scalars[2:]]
    wrong_scalar = directory / "half-scalar-wrong.case"
    wrong_scalar.write_bytes(constants + struct.pack("<4f", *wrong_values) * 64)
    pass_scalar = directory / "half-scalar-pass.case"
    pass_scalar.write_bytes(constants + struct.pack("<4f", *scalars) * 64)
    run_pixel_batch(renderer, backend, shader, [(wrong_scalar, "mismatch"), (pass_scalar, "pass")])


def pack_f16_pair(lo, hi):
    lo_bits = lo if isinstance(lo, int) else struct.unpack("<H", struct.pack("<e", lo))[0]
    hi_bits = hi if isinstance(hi, int) else struct.unpack("<H", struct.pack("<e", hi))[0]
    return lo_bits | (hi_bits << 16)


def rne_f16(value):
    if math.isnan(value):
        return math.nan
    try:
        return struct.unpack("<e", struct.pack("<e", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


# Packed binary16 VOP3P arithmetic: exercises all five v_pk_*_f16 instructions, op_sel/op_sel_hi,
# neg_lo/neg_hi, clamp, NaN minNum semantics, RNE tie-to-even rounding, and gfx1013 zero-extension
# of inline f16 constants (0.5 -> lo=0.5, hi=0.0).
VOP3P_BODY = [
    "v_mov_b32_e32 v19, s45",
    "v_mov_b32_e32 v20, s46",
    "v_mov_b32_e32 v21, s47",
    "v_pk_add_f16 v22, s44, v19 op_sel:[1,0] op_sel_hi:[0,1] neg_lo:[0,1]",
    "v_pk_mul_f16 v23, v22, v20 neg_hi:[1,0]",
    "v_pk_fma_f16 v24, v23, v21, s47 op_sel:[0,1,0] op_sel_hi:[1,0,1]",
    "v_pk_min_f16 v25, v24, 0x40007e00",
    "v_pk_max_f16 v26, v24, 0.5 clamp",
    "v_cvt_f32_f16_e32 v12, v25",
    "v_pk_add_f16 v27, v25, 0 op_sel:[1,0]",
    "v_cvt_f32_f16_e32 v13, v27",
    "v_cvt_f32_f16_e32 v14, v26",
    "v_pk_add_f16 v28, v26, 0 op_sel:[1,0]",
    "v_cvt_f32_f16_e32 v15, v28",
]
VOP3P_SETS = (
    # Set 1: v24=(0.875, 0.25); channel 0 tests minNum(0.875, qNaN)=0.875, channel 3=0.25 proves inline 0.5 hi=0.0.
    (pack_f16_pair(1.5, 2.0), pack_f16_pair(0.5, -1.0), pack_f16_pair(0.5, 1.0), pack_f16_pair(0.5, 0.5)),
    # Set 2: v24=(2.5, -1.0); channel 2 clamps 2.5 -> 1.0 and channel 3 clamps -1.0 -> 0.0.
    (pack_f16_pair(2.0, 3.0), pack_f16_pair(1.0, 0.0), pack_f16_pair(1.0, 2.0), pack_f16_pair(0.5, 1.0)),
    # Set 3: exact halfway ties at 1.0 + 2^-11 (-> even 1.0) and -(1.0 + 2^-10) - 2^-11 (-> even -(1.0 + 2^-9),
    # distinguishing RNE from RTZ f32tof16 in channel 1).
    (pack_f16_pair(0xbc01, 0x3c00), pack_f16_pair(0x9000, 0x9000), pack_f16_pair(4.0, 4.0), pack_f16_pair(0.25, 0.25)),
)


def vop3p_reference(words):
    def split16(w):
        return half_value(w), half_value(w >> 16)

    def min_num(a, b):
        return b if math.isnan(a) else (a if math.isnan(b) else min(a, b))

    def max_num(a, b):
        return b if math.isnan(a) else (a if math.isnan(b) else max(a, b))

    def sat(x):
        return 0.0 if math.isnan(x) or x <= 0.0 else (1.0 if x >= 1.0 else x)

    s44, s45, s46, s47 = (split16(w) for w in words)
    v22 = (rne_f16(s44[1] + (-s45[0])), rne_f16(s44[0] + s45[1]))
    v23 = (rne_f16(v22[0] * s46[0]), rne_f16((-v22[1]) * s46[1]))
    v24 = (rne_f16(v23[0] * s47[1] + s47[0]), rne_f16(v23[1] * s47[0] + s47[1]))
    v25 = (rne_f16(min_num(v24[0], math.nan)), rne_f16(min_num(v24[1], 2.0)))
    v26 = (rne_f16(sat(max_num(v24[0], 0.5))), rne_f16(sat(max_num(v24[1], 0.0))))
    return [v25[0], v25[1], v26[0], v26[1]]


def check_vop3p_draw(directory, compiler, renderer, backend):
    source = directory / "vop3p.hlsl"
    shader = directory / "vop3p.cso"
    program = with_constants(VOP3P_BODY)
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    cases = []
    for idx, words in enumerate(VOP3P_SETS):
        expected = vop3p_reference(words)
        assert all(not math.isnan(x) and half_rtz(x) == x for x in expected), expected
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *words)
        case = directory / f"vop3p-{idx}.case"
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        cases.append((case, "pass"))
    run_pixel_batch(renderer, backend, shader, cases)

    # Negative mutations against Set 1 and Set 2 must produce pixel_mismatch.
    for set_idx, old, new, label in (
        (0,
         "v_pk_add_f16 v22, s44, v19 op_sel:[1,0] op_sel_hi:[0,1] neg_lo:[0,1]",
         "v_pk_add_f16 v22, s44, v19 op_sel:[1,0] op_sel_hi:[0,1]", "neg_lo"),
        (0,
         "v_pk_mul_f16 v23, v22, v20 neg_hi:[1,0]",
         "v_pk_mul_f16 v23, v22, v20", "neg_hi"),
        (0,
         "v_pk_min_f16 v25, v24, 0x40007e00",
         "v_pk_max_f16 v25, v24, 0x40007e00", "min vs max"),
        (0,
         "v_pk_max_f16 v26, v24, 0.5 clamp",
         "v_pk_max_f16 v26, v24, 0.5 op_sel_hi:[1,0] clamp", "inline hi zero-extension"),
        (1,
         "v_pk_max_f16 v26, v24, 0.5 clamp",
         "v_pk_max_f16 v26, v24, 0.5", "clamp"),
    ):
        mutated = [line.replace(old, new) for line in program]
        assert mutated != program, label
        source.write_text(translate(mutated), encoding="ascii")
        compile_shader(compiler, source, shader)
        wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, cases[set_idx][0]],
                               capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Mutation of {label} escaped detection")


# v_cvt_f16_f32 rounds to nearest even and, on GFX10, writes only the low half of the destination. Channels 0
# and 1 read the whole word: v12 starts as 1.0, so after the conversion it is 1 + h * 2^-23 when the high half
# is kept, and (v12 - 1) * 2^23 splits into the low and high bytes of the binary16 encoding h. Channel 2
# widens -|S0| back to binary32 and channel 3 widens a subnormal result scaled by 2^24. Each set lists wrong
# references (channel, value) for the property it isolates.
NARROW_BODY = [
    "v_mov_b32_e32 v12, 1.0",
    "v_cvt_f16_f32_e32 v12, s44",
    "v_subrev_f32_e32 v12, 1.0, v12",
    "v_mul_f32_e32 v12, 0x4b000000, v12",
    "v_mul_f32_e32 v13, 0x3b800000, v12",
    "v_floor_f32_e32 v13, v13",
    "v_fmac_f32_e32 v12, 0xc3800000, v13",
    "v_cvt_f16_f32_e64 v14, -|s45|",
    "v_cvt_f32_f16_e32 v14, v14",
    "v_cvt_f16_f32_e32 v15, s46",
    "v_cvt_f32_f16_e32 v15, v15",
    "v_mul_f32_e32 v15, 0x4b800000, v15",
]
NARROW_SETS = (
    # Ties to even in the normal range and at the smallest subnormal.
    ((1 + 2.0 ** -11, 1 + 3 * 2.0 ** -11, 1.5 * 2.0 ** -24), ((0, 1.0), (2, -(1 + 2.0 ** -10)), (3, 1.0))),
    # The largest value below the overflow tie, -0, and the tie between zero and 2^-24.
    ((65520 - 2.0 ** -8, -0.0, 2.0 ** -25), ((1, 124.0), (2, 0.0), (3, 1.0))),
    # The overflow tie gives infinity; rounding carries a subnormal into the smallest normal; a negative subnormal.
    ((65520.0, 2.0 ** -14 - 2.0 ** -25, -3 * 2.0 ** -24), ((1, 123.0), (2, -1023 * 2.0 ** -24), (3, 3.0))),
    # A negative subnormal encoding, the negation modifier and a binary32 subnormal input.
    ((-3 * 2.0 ** -24, 3.0, 2.0 ** -149), ((1, 0.0), (2, 3.0))),
)


def narrow_reference(x):
    if abs(x) >= 65520:
        return 0xfc00 if x < 0 else 0x7c00
    return struct.unpack("<H", struct.pack("<e", x))[0]


def check_narrow_draw(directory, compiler, renderer, backend):
    source = directory / "narrow.hlsl"
    shader = directory / "narrow.cso"
    source.write_text(translate(with_constants(NARROW_BODY)), encoding="ascii")
    compile_shader(compiler, source, shader)
    cases = []
    for inputs, wrongs in NARROW_SETS:
        words = [struct.unpack("<I", struct.pack("<f", x))[0] for x in inputs]
        assert [struct.unpack("<f", struct.pack("<I", w))[0] for w in words] == list(inputs)
        constants = bytearray(256)
        struct.pack_into("<3I", constants, 12, *words)
        h = narrow_reference(inputs[0])
        expected = [float(h & 0xff), float(h >> 8), -abs(half_value(narrow_reference(-abs(inputs[1])))),
                    half_value(narrow_reference(inputs[2])) * 2.0 ** 24]
        for x in (*expected, *(value for _, value in wrongs)):
            assert half_rtz(x) == x, x
        for channel, value in wrongs:
            assert struct.pack("<f", value) != struct.pack("<f", expected[channel])
            wrong_values = list(expected)
            wrong_values[channel] = value
            path = directory / f"narrow-{len(cases)}.case"
            path.write_bytes(constants + struct.pack("<4f", *wrong_values) * 64)
            cases.append((path, "mismatch"))
        pass_path = directory / f"narrow-{len(cases)}.case"
        pass_path.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        cases.append((pass_path, "pass"))
    run_pixel_batch(renderer, backend, shader, cases)


def arithmetic():
    """Add, multiply, reversed subtraction and moves on the sample and constant words 8 to 11."""
    program = assembly(False)
    pos = program.index("s_mov_b32 s40, 1")
    program.insert(pos, "s_load_dwordx4 s[44:47], s[20:21], 0x40")
    pos = program.index("s_waitcnt vmcnt(0)")
    program[pos:pos] = [
        "s_buffer_load_dwordx4 s[44:47], s[44:47], 0x20",
        "s_waitcnt lgkmcnt(0)",
    ]
    pos = program.index("v_cvt_pkrtz_f16_f32_e32 v16, v12, v13")
    program[pos:pos] = [
        "v_mov_b32_e32 v18, v12",
        "v_mul_f32_e32 v18, s44, v18",
        "v_add_f32_e64 v12, v18, s45 clamp",
        "v_mov_b32_e64 v19, s46",
        "v_subrev_f32_e32 v19, v13, v19",
        "v_mul_f32_e64 v19, v19, -2.0",
        "v_subrev_f32_e64 v13, s47, v19 clamp",
        "v_add_f32_e32 v14, -0.5, v14",
        "v_mov_b32_e32 v20, 0x3e800000",
        "v_mul_f32_e32 v20, v20, v18",
        "v_mov_b32_e32 v15, v20",
    ]
    return program


def arithmetic_case(directory, *constants, name="arithmetic.case"):
    """Expected colors for arithmetic(). Each step must be exact in binary32, so neither the
    rounding of add, subtract and multiply nor fusion can change the result."""
    def exact(value):
        rounded, = struct.unpack("<f", struct.pack("<f", value))
        assert rounded == value, value
        return rounded
    scale, bias, pivot, offset = map(exact, constants)
    case = bytearray(256)
    struct.pack_into("<4f", case, 32, scale, bias, pivot, offset)
    ramp = (0, 16, 37, 64, 101, 153, 219, 254)
    pixels = []
    for y in ramp:
        for x in ramp:
            r, g, b = (struct.unpack("<f", struct.pack("<f", c / 255))[0] for c in (x, y, 128))
            scaled = exact(scale * r)
            red = min(1, max(0, exact(scaled + bias)))
            green = min(1, max(0, exact(exact(exact(pivot - g) * -2.0) - offset)))
            color = (red, green, exact(-0.5 + b), exact(0.25 * scaled))
            pixels.append(tuple(map(half_rtz, color)))
    return write_case(directory / name, case, pixels)


def curve_case(directory, gain, *, name="curve.case"):
    gamma, floor, mix = 2.0, .125, .25
    case = bytearray(256)
    struct.pack_into("<4f", case, 12, gamma, gain, floor, mix)
    pixels = []
    for y in RAMP:
        for x in RAMP:
            g = y / 255
            r = gain * math.sqrt(max(floor, x / 255)) - (1 - g) + mix * g
            pixels.append((half_rtz(min(1, max(0, r))), half_rtz(g), half_rtz(128/255), 1))
    return write_case(directory / name, case, pixels)


def shaping_case(directory, gain, *, name="shaping.case"):
    scale, floor, mix = 2.0, .125, .25
    case = bytearray(256)
    struct.pack_into("<4f", case, 12, scale, gain, floor, mix)
    pixels = []
    for y in RAMP:
        for x in RAMP:
            g = y / 255
            scaled = scale * g
            r = (gain * math.sqrt(max(floor, x / 255))
                 + 1.0 / math.sqrt(scale + g)
                 + mix * (scaled - math.floor(scaled))
                 - math.floor(scaled))
            pixels.append((half_rtz(min(1, max(0, r))), half_rtz(g), half_rtz(128/255), 1))
    return write_case(directory / name, case, pixels)


def truncate_case(directory: Path, bias: float, green: float, blue: float, alpha: float,
                  *, name: str = "truncate.case") -> Path:
    case = bytearray(256)
    struct.pack_into("<4f", case, 12, bias, green, blue, alpha)
    # modf returns the integer component as a float and preserves the sign of zero.
    green = math.modf(green)[1]
    blue = math.modf(-abs(blue))[1]
    alpha = max(0.0, min(1.0, math.modf(alpha)[1]))
    pixels = []
    for _ in RAMP:
        for x in RAMP:
            pixels.append((math.modf(bias + x / 255)[1], green, blue, alpha))
    return write_case(directory / name, case, pixels)


def sopk_case(directory, red: float, green: float, *, name: str = "sopk.case") -> Path:
    case = bytearray(256)
    struct.pack_into("<4f", case, 12, red, green, 0, 1)

    def bits(value):
        return struct.unpack("<I", struct.pack("<f", value))[0]

    results = (bits(red) * 3, bits(green) - 2)
    expected = [struct.unpack("<f", struct.pack("<I", value & 0xffffffff))[0]
                for value in results]
    for _ in RAMP:
        for x in RAMP:
            case.extend(struct.pack("<4f", half_rtz(expected[0]), half_rtz(expected[1]), half_rtz(128 / 255), 1))
    path = directory / name
    path.write_bytes(case)
    return path


def rndne(x):
    """Round a binary32 value to the nearest integer, ties to even (Python's round on a float)."""
    return float(round(x))


def rounding_case(directory, scale, tie, bias, *, name="rounding.case"):
    """Expected colors for rounding(). The scale is a power of two, so each product is exact and
    only the rounding mode decides the result. The constants keep every result away from -0."""
    assert math.frexp(scale)[0] == 0.5, scale
    case = bytearray(256)
    struct.pack_into("<4f", case, 12, scale, tie, bias, 0)
    for y in RAMP:
        for x in RAMP:
            r, g = single(x / 255), single(y / 255)
            color = (math.ceil(scale * r), rndne(scale * g), rndne(single(tie)), math.ceil(-single(bias)))
            case.extend(struct.pack("<4f", *map(half_rtz, map(float, color))))
    path = directory / name
    path.write_bytes(case)
    return path


def absolute_case(directory):
    case = bytearray(256)
    pixels = []
    for y in RAMP:
        for x in RAMP:
            red = abs(0.5 - x / 255)
            pixels.append((half_rtz(red), half_rtz(y / 255), half_rtz(128 / 255), 1))
    return write_case(directory / "absolute.case", case, pixels)


def single(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def select_case(directory, fold, limit, capped, unequal, *, name="select.case"):
    """Expected colors for select(): every step is exact in binary32, so no device tolerance applies."""
    case = bytearray(256)
    struct.pack_into("<4f", case, 12, fold, limit, capped, unequal)
    ramp = (0, 16, 37, 64, 101, 153, 219, 254)
    pixels = []
    for y in ramp:
        for x in ramp:
            r, g = single(x / 255), single(y / 255)
            color = (single(1 - r) if fold < r else r, capped if g >= limit else g,
                     0.0 if r < g else 1.0, 1.0 if r == g else unequal)
            pixels.append(tuple(map(half_rtz, color)))
    return write_case(directory / name, case, pixels)


# v_cube{id,sc,tc,ma}_f32 follow LLVM's ConstantFoldAMDGCNCubeIntrinsic. The direction sets cover all
# six faces, the y/x and z/x ties (the later axis in z, y, x order wins) and -0 on the major axis,
# which selects the positive face. Each set has one wrong reference for the property it isolates.
CUBE_BODY = [
    "v_mov_b32_e32 v18, s45",
    "v_mov_b32_e32 v19, s46",
    "v_cubeid_f32 v12, s44, v18, v19",
    "v_cubesc_f32 v13, s44, v18, v19",
    "v_cubetc_f32 v14, s44, v18, v19",
    "v_cubema_f32 v15, s44, v18, v19",
]
CUBE_SETS = (
    ((2.0, -2.0, 1.0), (0, 1.0)),
    ((-1.5, 0.25, -1.5), (0, 1.0)),
    ((-3.0, 1.0, 2.0), (3, 6.0)),
    ((0.0, 0.0, -0.0), (0, 5.0)),
    ((1.0, 0.5, -0.25), (1, -0.25)),
    ((0.25, 1.0, 0.5), (2, -0.5)),
)


def cube_reference(x, y, z):
    if abs(z) >= abs(x) and abs(z) >= abs(y):
        face, sc, tc, ma = (5.0, -x, -y, z) if z < 0 else (4.0, x, -y, z)
    elif abs(y) >= abs(x):
        face, sc, tc, ma = (3.0, x, -z, y) if y < 0 else (2.0, x, z, y)
    else:
        face, sc, tc, ma = (1.0, z, -y, x) if x < 0 else (0.0, -z, -y, x)
    return [face, sc, tc, ma + ma]


def check_cube_draw(directory, compiler, renderer, backend):
    source = directory / "cube.hlsl"
    shader = directory / "cube.cso"
    source.write_text(translate(with_constants(CUBE_BODY)), encoding="ascii")
    compile_shader(compiler, source, shader)
    faces = set()
    cases = []
    for direction, (channel, wrong_value) in CUBE_SETS:
        constants = bytearray(256)
        struct.pack_into("<3f", constants, 12, *direction)
        expected = cube_reference(*direction)
        faces.add(expected[0])
        assert all(half_rtz(x) == x for x in (*expected, wrong_value)) and wrong_value != expected[channel]
        wrong_values = list(expected)
        wrong_values[channel] = wrong_value
        wrong_path = directory / f"cube-{len(cases)}.case"
        write_case(wrong_path, constants, tuple(wrong_values))
        cases.append((wrong_path, "mismatch"))
        pass_path = directory / f"cube-{len(cases)}.case"
        write_case(pass_path, constants, tuple(expected))
        cases.append((pass_path, "pass"))
    run_pixel_batch(renderer, backend, shader, cases)
    assert faces == {0.0, 1.0, 2.0, 3.0, 4.0, 5.0}
    assert math.copysign(1.0, cube_reference(0.0, 0.0, -0.0)[3]) < 0


SCALAR_BFE64_OPS = ("s_bfe_u64", "s_bfe_i64")


def scalar_bfe64(op, src="s[44:45]", control="s46"):
    return uncompressed_export([
        "s_cmp_eq_u32 s47, 0",
        f"{op} s[44:45], {src}, {control}",
        "s_cselect_b32 s48, 1, 0",
        "v_mov_b32_e32 v12, s44",
        "v_mov_b32_e32 v13, s45",
        "v_mov_b32_e32 v14, s48",
        "v_mov_b32_e32 v15, 0",
    ])


def check_scalar_bfe64_draw(directory, compiler, renderer, backend):
    word = 0x81234567_89abcdef
    inputs = (
        (word, 0x000005),  # empty field on nonzero input
        (0, 0x400000),  # zero result from a non-empty field
        (word, 0x010000),
        (word, 0x02001f),  # negative two-bit field spanning the dwords
        (word, 0x200020),
        (1 << 40, 0x3f0001),  # only the high result dword sets SCC
        (word, 0x400000),
        (word, 0x7f003c),  # width beyond the source at a nonzero offset
        (word, 0xff7fffc0),  # ignored control bits, width 127, offset 0
        (word, 0x01003f),
        (0x7f800001, 0x200000),  # preserve raw bits of a signaling NaN
        (1, 0x800000),  # reserved bit 23 alone leaves width and offset zero
        # Width 60 at offset 1 includes source bit 60 as the sign bit.
        # The signed result has nonzero halves whose sum wraps to zero.
        (0x10000000_10000000, 0x3c0001),
    )
    cases = {}
    for op in SCALAR_BFE64_OPS:
        cases[op] = []
        for i, (value, control) in enumerate(inputs):
            offset = control & 63
            width = min((control >> 16) & 127, 64 - offset)
            field = (value >> offset) & ((1 << width) - 1)
            if op == "s_bfe_i64" and width and (field >> (width - 1)):
                field -= 1 << width
            scc = int(field != 0)
            constants = bytearray(256)
            # With s47=scc, the preceding comparison seeds the opposite condition.
            struct.pack_into("<4I", constants, 12, value & 0xffffffff, value >> 32, control, scc)
            pixel = struct.pack("<4I", field & 0xffffffff, (field >> 32) & 0xffffffff, scc, 0)
            path = directory / f"{op}-{i}.case"
            path.write_bytes(constants + pixel * 64 + b"\x01")  # UINT target, four raw words
            cases[op].append((path, "pass"))

    for op, wrong_op in (SCALAR_BFE64_OPS, SCALAR_BFE64_OPS[::-1]):
        source = directory / f"{op}.hlsl"
        shader = directory / f"{op}.cso"
        source.write_text(translate_int(scalar_bfe64(op)), encoding="ascii")
        compile_shader(compiler, source, shader)
        wrong_case, _ = cases[wrong_op][3]
        run_pixel_batch(renderer, backend, shader, [*cases[op], (wrong_case, "mismatch")])


QUADMASK_BITREPLICATE_INSTRUCTIONS = {
    "s_quadmask_b32": "s_quadmask_b32 s44, s44",
    "s_quadmask_b64": "s_quadmask_b64 s[44:45], s[44:45]",
    "s_bitreplicate_b64_b32": "s_bitreplicate_b64_b32 s[44:45], s44",
}

QUADMASK_BITREPLICATE_WORDS = {
    "sparse-quads": (0x81020408, 0x80010001),
    "high-only": (0, 0x80000000),
    "split-bits-15-16": (0x80000000, 0x00018000),
    "alternating-even": (0xaaaaaaaa, 0x55555555),
    "alternating-odd": (0x55555555, 0xaaaaaaaa),
    "all-set": (0xffffffff, 0xffffffff),
    "all-zero": (0, 0),
}

def quadmask_bitreplicate_program(op):
    return uncompressed_export([
        "s_cmp_lg_u32 s47, 0",
        QUADMASK_BITREPLICATE_INSTRUCTIONS[op],
        "s_cselect_b32 s48, 1, 0",
        "v_mov_b32_e32 v12, s44",
        "v_mov_b32_e32 v13, s45",
        "v_mov_b32_e32 v14, s48",
        "v_mov_b32_e32 v15, 0",
    ])


SCALAR_BITSET64_OPS = ("s_bitset0_b64", "s_bitset1_b64")


def scalar_bitset64(op, index="s46"):
    return uncompressed_export([
        "s_cmp_lg_u32 s47, 0",
        f"{op} s[44:45], {index}",
        "s_cselect_b32 s48, 1, 0",
        "v_mov_b32_e32 v12, s44",
        "v_mov_b32_e32 v13, s45",
        "v_mov_b32_e32 v14, s48",
        "v_mov_b32_e32 v15, 0",
    ])


def quadmask32(word):
    return sum((((word >> (4 * group)) & 0xf) != 0) << group for group in range(8))


def quadmask64(lo, hi):
    # Measured on RDNA2: the sixteen quad bits are contiguous in the low word.
    return quadmask32(lo) | (quadmask32(hi) << 8)


def bitreplicate16(word):
    value = word & 0xffff
    value = (value | (value << 8)) & 0x00ff00ff
    value = (value | (value << 4)) & 0x0f0f0f0f
    value = (value | (value << 2)) & 0x33333333
    value = (value | (value << 1)) & 0x55555555
    return value | (value << 1)


def bitreplicate(word):
    return bitreplicate16(word), bitreplicate16(word >> 16)


def quadmask_bitreplicate_reference(op, lo, hi, initial_scc):
    if op == "s_quadmask_b32":
        result = quadmask32(lo)
        return result, hi, int(result != 0), 0
    if op == "s_quadmask_b64":
        result = quadmask64(lo, hi)
        return result, 0, int(result != 0), 0
    assert op == "s_bitreplicate_b64_b32", op
    result_lo, result_hi = bitreplicate(lo)
    return result_lo, result_hi, initial_scc, 0


def run_quadmask_bitreplicate(directory, compiler, renderer, backend):
    for op in QUADMASK_BITREPLICATE_INSTRUCTIONS:
        source = directory / f"{op}.hlsl"
        shader = directory / f"{op}.cso"
        source.write_text(translate_int(quadmask_bitreplicate_program(op)), encoding="ascii")
        compile_shader(compiler, source, shader)

        def add_case(name, words, initial_scc, expected, status):
            constants = bytearray(256)
            expected_scc = expected[2]
            seed = initial_scc if op == "s_bitreplicate_b64_b32" else int(not expected_scc)
            struct.pack_into("<4I", constants, 12, words[0], words[1], 0, seed)
            path = directory / f"{op}-{name}.case"
            write_case(path, constants, tuple(as_float(word) for word in expected), integer_rt=True)
            cases.append((path, status))

        cases = []
        for name, words in QUADMASK_BITREPLICATE_WORDS.items():
            seeds = (0, 1) if op == "s_bitreplicate_b64_b32" else (0,)
            for seed in seeds:
                expected = quadmask_bitreplicate_reference(op, *words, seed)
                add_case(f"{name}-scc{seed}", words, seed, expected, "pass")

        lo, hi = QUADMASK_BITREPLICATE_WORDS["sparse-quads"]
        if op == "s_quadmask_b32":
            expected = quadmask_bitreplicate_reference(op, lo, hi, 0)
            # AND within each quad is zero for the sparse input.
            add_case("and-within-group", (lo, hi), 0, (expected[0] ^ 0xd5, *expected[1:]),
                     "mismatch")
            expected = quadmask_bitreplicate_reference(op, *QUADMASK_BITREPLICATE_WORDS["all-set"], 0)
            add_case("old-scc", QUADMASK_BITREPLICATE_WORDS["all-set"], 0,
                     (expected[0], expected[1], 0, 0), "mismatch")
        elif op == "s_quadmask_b64":
            expected = quadmask_bitreplicate_reference(op, lo, hi, 0)
            wrong_low = sum((((lo >> (4 * group)) & 0xf) == 0xf) << group for group in range(8))
            wrong_low |= sum((((hi >> (4 * group)) & 0xf) == 0xf) << (group + 8)
                             for group in range(8))
            add_case("and-within-group", (lo, hi), 0, (wrong_low, 0, expected[2], 0),
                     "mismatch")
            expected = quadmask_bitreplicate_reference(
                op, *QUADMASK_BITREPLICATE_WORDS["high-only"], 0)
            add_case("ignore-high-word", QUADMASK_BITREPLICATE_WORDS["high-only"], 0,
                     (quadmask32(0), 0, expected[2], 0), "mismatch")
            add_case("quad-in-high-word", QUADMASK_BITREPLICATE_WORDS["high-only"], 0,
                     (expected[0], 0x80000000, expected[2], 0), "mismatch")
            expected = quadmask_bitreplicate_reference(
                op, *QUADMASK_BITREPLICATE_WORDS["all-set"], 0)
            add_case("old-scc", QUADMASK_BITREPLICATE_WORDS["all-set"], 0,
                     (expected[0], expected[1], 0, 0), "mismatch")
        else:
            lo, hi = QUADMASK_BITREPLICATE_WORDS["split-bits-15-16"]
            expected = quadmask_bitreplicate_reference(op, lo, hi, 0)
            duplicate_lo, duplicate_hi = 0x00018001, 0x80010001
            duplicate_expected = quadmask_bitreplicate_reference(
                op, duplicate_lo, duplicate_hi, 0)
            duplicate_low_half = duplicate_lo & 0xffff
            wrong_stride = ((duplicate_low_half | (duplicate_low_half << 8)) & 0x00ff00ff)
            wrong_stride = (wrong_stride | (wrong_stride << 4)) & 0x0f0f0f0f
            wrong_stride = (wrong_stride | (wrong_stride << 2)) & 0x33333333
            wrong_stride = (wrong_stride | (wrong_stride << 1)) & 0x55555555
            add_case("omit-second-copy", (duplicate_lo, duplicate_hi), 0,
                     (wrong_stride, duplicate_expected[1], 0, 0), "mismatch")
            source_hi = bitreplicate(hi)
            add_case("source-high-word", (lo, hi), 0, (*source_hi, 0, 0), "mismatch")
            add_case("recomputed-scc", (lo, hi), 1, (expected[0], expected[1], 0, 0),
                     "mismatch")

        for path, status in cases:
            print(f"quadmask-bitreplicate {op}: {path.stem} expected={status}", flush=True)
        run_pixel_batch(renderer, backend, shader, cases)

def bitset64_reference(lo, hi, index, set_bit):
    """Set or clear one bit in a 64-bit pair without changing the other word."""
    bit = index & 63
    mask = 1 << bit
    value = (hi << 32) | lo
    value = value | mask if set_bit else value & ~mask
    return value & 0xffffffff, (value >> 32) & 0xffffffff


def check_scalar_bitset64_draw(directory, compiler, renderer, backend):
    source = directory / "scalar-bitset64.hlsl"
    shader = directory / "scalar-bitset64.cso"
    initial_lo, initial_hi = 0x81020408, 0x00010001

    for op in SCALAR_BITSET64_OPS:
        source.write_text(translate_int(scalar_bitset64(op)), encoding="ascii")
        compile_shader(compiler, source, shader)
        cases = []
        for index in (0, 31, 32, 63, 64):
            for initial_scc in (0, 1):
                constants = bytearray(256)
                struct.pack_into("<4I", constants, 12, initial_lo, initial_hi, index, initial_scc)
                result_lo, result_hi = bitset64_reference(
                    initial_lo, initial_hi, index, op == "s_bitset1_b64")
                expected = (result_lo, result_hi, initial_scc, 0)
                path = directory / f"{op}-{index}-scc{initial_scc}.case"
                write_case(path, constants, tuple(as_float(word) for word in expected), integer_rt=True)
                cases.append((path, "pass"))
        if op == "s_bitset1_b64":
            # Exact all-ones NaN-shaped input from PR #680. Setting bit 31 must
            # keep the full-width integer 0xffffffff instead of canonicalising it.
            for initial_scc in (0, 1):
                constants = bytearray(256)
                struct.pack_into("<4I", constants, 12, 0xFFFFFFFF, 0xFFFFFFFF, 31, initial_scc)
                expected = (0xFFFFFFFF, 0xFFFFFFFF, initial_scc, 0)
                path = directory / f"{op}-nan-bit31-scc{initial_scc}.case"
                write_case(path, constants, tuple(as_float(word) for word in expected), integer_rt=True)
                cases.append((path, "pass"))
        run_pixel_batch(renderer, backend, shader, cases)

        # At index 31, selecting the high destination word changes bit 63 instead of bit 31.
        boundary = directory / f"{op}-boundary-word.case"
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, initial_lo, initial_hi, 31, 0)
        wrong_value = (initial_hi << 32) | initial_lo
        wrong_mask = 1 << 63
        wrong_value = wrong_value | wrong_mask if op == "s_bitset1_b64" else wrong_value & ~wrong_mask
        wrong_expected = (wrong_value & 0xffffffff, wrong_value >> 32, 0, 0)
        write_case(boundary, constants, tuple(as_float(word) for word in wrong_expected),
                   integer_rt=True)
        run_pixel_batch(renderer, backend, shader, [(boundary, "mismatch")])

        # The index may alias either destination word. The lowering must read
        # the old pair before replacing it.
        alias_inputs = {
            "s_bitset1_b64": (("s44", 0x0000001F, 0xA5A5A5A5),
                              ("s45", 0x12345678, 0x00000021)),
            "s_bitset0_b64": (("s44", 0x8000001F, 0xA5A5A5A5),
                              ("s45", 0x12345678, 0x0000082B)),
        }
        for alias_index, alias_lo, alias_hi in alias_inputs[op]:
            alias_source = directory / f"{op}-{alias_index}.hlsl"
            alias_shader = directory / f"{op}-{alias_index}.cso"
            alias_source.write_text(translate_int(scalar_bitset64(op, alias_index)), encoding="ascii")
            compile_shader(compiler, alias_source, alias_shader)
            alias_cases = []
            for initial_scc in (0, 1):
                index = alias_lo if alias_index == "s44" else alias_hi
                result_lo, result_hi = bitset64_reference(
                    alias_lo, alias_hi, index, op == "s_bitset1_b64")
                constants = bytearray(256)
                struct.pack_into("<4I", constants, 12, alias_lo, alias_hi, 0, initial_scc)
                expected = (result_lo, result_hi, initial_scc, 0)
                path = directory / f"{op}-{alias_index}-scc{initial_scc}.case"
                write_case(path, constants, tuple(as_float(word) for word in expected), integer_rt=True)
                alias_cases.append((path, "pass"))
            run_pixel_batch(renderer, backend, alias_shader, alias_cases)

        if op == "s_bitset1_b64":
            # A read-after-write alias would take the index from 0x25 and set
            # bit 37 instead of bit 5. The reference encodes that wrong result.
            read_after_write = directory / "s_bitset1_b64-read-after-write.case"
            constants = bytearray(256)
            struct.pack_into("<4I", constants, 12, 5, 0x11111111, 0, 0)
            wrong_expected = (5, 0x11111131, 0, 0)
            write_case(read_after_write, constants,
                       tuple(as_float(word) for word in wrong_expected), integer_rt=True)
            alias_source = directory / "s_bitset1_b64-s44.hlsl"
            alias_shader = directory / "s_bitset1_b64-s44.cso"
            alias_source.write_text(translate_int(scalar_bitset64(op, "s44")), encoding="ascii")
            compile_shader(compiler, alias_source, alias_shader)
            run_pixel_batch(renderer, backend, alias_shader, [(read_after_write, "mismatch")])

    # PR #680 reproduced an NVIDIA sign-bit loss with an immediate index of 31.
    # Keep the original float-overwrite control so a raw bit copy can never reuse stale bits.
    program = scalar_bitset64("s_bitset1_b64", "31")
    constants = bytearray(256)
    struct.pack_into("<4I", constants, 12, 0xffffffff, 0xffffffff, 0, 0)
    for overwrite in (False, True):
        copies = ["v_mov_b32_e32 v20, v12", "v_mov_b32_e32 v12, 1.0"]
        copies.append("v_add_f32_e32 v12, 1.0, v12" if overwrite else "v_mov_b32_e32 v12, v20")
        source.write_text(translate_int(program[:-2] + copies + program[-2:]), encoding="ascii")
        compile_shader(compiler, source, shader)
        correct = 0x40000000 if overwrite else 0xffffffff
        wrong = 0xffffffff if overwrite else 0x7fffffff
        cases = []
        for low, status in ((correct, "pass"), (wrong, "mismatch")):
            path = directory / f"bitset31-transport-{overwrite}-{status}.case"
            path.write_bytes(constants + struct.pack("<4I", low, 0xffffffff, 0, 0) * 64 + b"\x01")
            cases.append((path, status))
        run_pixel_batch(renderer, backend, shader, cases)


S_CSELECT_B64 = "s_cselect_b64 s[44:45], s[44:45], s[46:47]"


def scalar_select64():
    return uncompressed_export([
        "s_cmp_eq_u32 s44, 0",
        S_CSELECT_B64,
        "s_cselect_b64 s[48:49], 0, -1",
        "v_mov_b32_e32 v12, s44",
        "v_mov_b32_e32 v13, s45",
        "v_mov_b32_e32 v14, s48",
        "v_mov_b32_e32 v15, s49",
    ])


def check_scalar_select64_draw(directory, compiler, renderer, backend):
    source = directory / "scalar-select64.hlsl"
    shader = directory / "scalar-select64.cso"
    program = scalar_select64()
    source.write_text(translate_int(program), encoding="ascii")
    compile_shader(compiler, source, shader)

    # Select a zero pair with SCC set and a nonzero pair with SCC clear to
    # distinguish preservation from recomputing SCC from the selected words.
    other = (0x87654321, 0x43218765)
    cases = []
    for i, (lo, hi) in enumerate(((0, 0), (0x89abcdef, 0x12345678), (0, 0x76543210))):
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, lo, hi, *other)
        expected = (*other, 0xffffffff, 0xffffffff) if lo else (lo, hi, 0, 0)
        case = directory / f"scalar-select64-{i}.case"
        write_case(case, constants, tuple(as_float(word) for word in expected), integer_rt=True)
        cases.append(case)
    run_pixel_batch(renderer, backend, shader, [(case, "pass") for case in cases])

    program[program.index(S_CSELECT_B64)] = "s_cselect_b64 s[44:45], s[46:47], s[44:45]"
    source.write_text(translate_int(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    run_pixel_batch(renderer, backend, shader, [(case, "mismatch") for case in cases])

    # s_cselect_b64 does not write SCC: capture SCC after a select that picks a zero pair.
    program = scalar_select64()
    pos = program.index("s_cselect_b64 s[48:49], 0, -1")
    program[pos + 1:pos + 1] = ["s_cselect_b32 s50, 1, 0"]
    program[program.index("v_mov_b32_e32 v14, s48")] = "v_mov_b32_e32 v14, s50"
    source.write_text(translate_int(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    scc_cases = []
    for i, (lo, hi) in enumerate(((0, 0), (0x89abcdef, 0x12345678))):
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, lo, hi, *other)
        expected = (*other, 0, 0xffffffff) if lo else (lo, hi, 1, 0)
        case = directory / f"scalar-select64-scc-{i}.case"
        write_case(case, constants, tuple(as_float(word) for word in expected), integer_rt=True)
        scc_cases.append((case, "pass"))
    run_pixel_batch(renderer, backend, shader, scc_cases)


def valu_div_fmas_program():
    return with_constants([
        "v_mov_b32_e32 v19, 2.0",
        "v_mov_b32_e32 v20, 4.0",
        "v_mov_b32_e32 v21, v19",
        "v_cmp_lt_f32 vcc_lo, 1.0, 0",
        "v_div_fmas_f32 v12, v21, 0.5, 1.0",
        "v_div_fmas_f32 v13, v21, 0.5, 1.0 clamp",
        "v_cmp_lt_f32 vcc_lo, 0, 1.0",
        "v_div_fmas_f32 v14, 1.0, 2.0, 3.0",
        "v_div_fmas_f32 v15, 0x00800000, 0.5, 0",
    ])


SCALAR64_SETS = (
    # (a_lo, a_hi, b_lo, b_hi, sh_lo, sh_hi, bfm_w, bfm_off, bfe_src, bfe_ctrl_a, bfe_ctrl_b, alt)
    (0x89abcdef, 0x91234567, 0x55aa55aa, 0x7fffffff, 5, 37, 12, 28, 0x9a345f50, 0x00080004, 0x00100018, 0x55aa55aa),
    (0x0000a5a5, 0x0f0f0000, 0xffff5a5a, 0xf0f0ffff, 0, 100, 40, 16, 0x9a345f50, 0x00000000, 0x00000000, 0x00000000),
)


def scalar64():
    """Exercise 64-bit SOP1/SOP2 scalar ALU and 32-bit BFE, packing 40 output bits into exact binary16 floats."""
    program = assembly(False)
    pos = program.index("s_mov_b32 s40, 1")
    program.insert(pos, "s_load_dwordx4 s[44:47], s[20:21], 0x40")
    pos = program.index("s_waitcnt vmcnt(0)")
    program[pos:pos] = [
        "s_buffer_load_dwordx4 s[52:55], s[44:47], 0x10",
        "s_buffer_load_dwordx4 s[56:59], s[44:47], 0x20",
        "s_buffer_load_dwordx4 s[60:63], s[44:47], 0x30",
        "s_waitcnt lgkmcnt(0)",
    ]

    body = [
        "s_mov_b32 s64, 0",
        "s_mov_b32 s65, 0",
        "s_mov_b32 s66, 0",
        "s_mov_b32 s67, 0",
        "s_mov_b32 s78, 0",
        "s_mov_b32 s79, 0",
        "s_mov_b32 s82, s63",
        "s_mov_b32 s83, 0",
    ]

    def capture_scc(bit_index):
        body.extend([
            "s_mov_b32 s70, 0",
            "s_mov_b32 s71, 0",
            f"s_bfm_b64 s[74:75], 1, {bit_index}",
            "s_cmov_b64 s[70:71], s[74:75]",
            "s_or_b64 s[66:67], s[66:67], s[70:71]",
        ])

    def mix_acc(pair):
        body.extend([
            "s_lshl_b64 s[68:69], s[64:65], 3",
            "s_lshr_b64 s[64:65], s[64:65], 61",
            "s_or_b64 s[64:65], s[64:65], s[68:69]",
            f"s_xor_b64 s[64:65], s[64:65], {pair}",
        ])

    # Bit 0: s_and_b64 + two consecutive s_cmov_b64 (inline -1 sign-extended and SCC preserved).
    body.extend([
        "s_and_b64 s[72:73], s[52:53], s[54:55]",
        "s_cmov_b64 s[78:79], -1",
    ])
    capture_scc(0)
    mix_acc("s[72:73]")
    mix_acc("s[78:79]")

    # Bit 1: s_or_b64 (full 64-bit union saved in s[84:85] + SCC probe on s[82:83]).
    body.append("s_or_b64 s[84:85], s[52:53], s[54:55]")
    mix_acc("s[84:85]")
    body.append("s_or_b64 s[72:73], s[82:83], 0")
    capture_scc(1)

    # Bit 2: s_xor_b64 (register pair and zero-extended 32-bit hex literal).
    body.append("s_xor_b64 s[72:73], s[52:53], s[54:55]")
    mix_acc("s[72:73]")
    body.append("s_xor_b64 s[72:73], s[82:83], 0x55aa55aa")
    capture_scc(2)
    mix_acc("s[72:73]")

    # Bit 3: s_andn2_b64 (register pair + SCC probe).
    body.append("s_andn2_b64 s[72:73], s[52:53], s[54:55]")
    mix_acc("s[72:73]")
    body.append("s_andn2_b64 s[72:73], s[82:83], 0xf0f0f0f")
    capture_scc(3)

    # Bit 4: s_orn2_b64 (register pair + inline -1 proving 64-bit all-ones inversion).
    body.append("s_orn2_b64 s[72:73], s[52:53], s[54:55]")
    mix_acc("s[72:73]")
    body.append("s_orn2_b64 s[72:73], s[82:83], -1")
    capture_scc(4)

    # Bit 5: s_not_b64 + s_brev_b64 (s_brev_b64 placed before capture_scc to verify SCC preservation).
    body.append("s_not_b64 s[72:73], s[52:53]")
    mix_acc("s[72:73]")
    body.extend([
        "s_not_b64 s[72:73], s[84:85]",
        "s_brev_b64 s[86:87], s[52:53]",
    ])
    capture_scc(5)
    mix_acc("s[86:87]")

    # Bit 6: s_lshl_b64 across small/zero and >=32/wrapped shift counts.
    body.append("s_lshl_b64 s[72:73], s[52:53], s56")
    mix_acc("s[72:73]")
    body.append("s_lshl_b64 s[72:73], s[52:53], s57")
    mix_acc("s[72:73]")
    body.append("s_lshl_b64 s[72:73], s[82:83], s57")
    capture_scc(6)

    # Bit 7: s_lshr_b64 across small/zero and >=32/wrapped shift counts.
    body.append("s_lshr_b64 s[72:73], s[52:53], s56")
    mix_acc("s[72:73]")
    body.append("s_lshr_b64 s[72:73], s[52:53], s57")
    mix_acc("s[72:73]")
    body.append("s_lshr_b64 s[72:73], s[82:83], s56")
    capture_scc(7)

    # Bit 8: s_ashr_i64 (signed) + s_bfm_b64 (preserving SCC, plus w=0 and w=32 boundaries).
    body.append("s_ashr_i64 s[72:73], s[52:53], s56")
    mix_acc("s[72:73]")
    body.append("s_ashr_i64 s[72:73], s[52:53], s57")
    mix_acc("s[72:73]")
    body.extend([
        "s_ashr_i64 s[72:73], s[82:83], s56",
        "s_bfm_b64 s[86:87], s58, s59",
    ])
    capture_scc(8)
    mix_acc("s[86:87]")
    body.append("s_bfm_b64 s[88:89], 0, 5")
    mix_acc("s[88:89]")
    body.append("s_bfm_b64 s[88:89], 32, 16")
    mix_acc("s[88:89]")

    # Bit 9: s_bfe_u32 (SCC observed via s_cmov_b64 into acc) and s_bfe_i32 (SCC observed in bit 9).
    body.extend([
        "s_mov_b32 s78, 0",
        "s_mov_b32 s79, 0",
        "s_bfe_u32 s72, s60, s61",
        "s_cmov_b64 s[78:79], 0x13572468",
        "s_bfe_u32 s73, s60, s62",
    ])
    mix_acc("s[72:73]")
    mix_acc("s[78:79]")
    body.extend([
        "s_bfe_i32 s72, s60, s61",
        "s_bfe_i32 s73, s60, s62",
    ])
    capture_scc(9)
    mix_acc("s[72:73]")

    # s_nand_b64, s_nor_b64 and s_xnor_b64: the register pair, then an SCC probe. Channel 3 exports
    # only SCC bits 0-9, so each probe's SCC reaches acc through s_cmov_b64 of a distinct literal.
    # nand(-1, -1) and nor(alt, -1) are zero, and xnor(alt, -1) = alt is zero only in Set 2.
    for op, probe, marker in (("s_nand_b64", "-1, -1", "0x2468ace0"),
                              ("s_nor_b64", "s[82:83], -1", "0x13579bdf"),
                              ("s_xnor_b64", "s[82:83], -1", "0x0fedcba9")):
        body.append(f"{op} s[72:73], s[52:53], s[54:55]")
        mix_acc("s[72:73]")
        body.extend([
            "s_mov_b32 s78, 0",
            "s_mov_b32 s79, 0",
            f"{op} s[72:73], {probe}",
            f"s_cmov_b64 s[78:79], {marker}",
        ])
        mix_acc("s[72:73]")
        mix_acc("s[78:79]")

    # Fold all 64 bits of acc into bits [29:0] of s64: acc ^ (acc >> 30) ^ (acc >> 60), both shifts of the
    # original acc, so every bit reaches the export. Repeating x ^= x >> 30 instead gives x ^ (x >> 60)
    # and drops bits 30-59. Then export 4 x 10-bit fields as 1 + k/1024.
    body.extend([
        "s_lshr_b64 s[68:69], s[64:65], 30",
        "s_lshr_b64 s[70:71], s[64:65], 60",
        "s_xor_b64 s[64:65], s[64:65], s[68:69]",
        "s_xor_b64 s[64:65], s[64:65], s[70:71]",
    ])
    for channel, (src, ctrl) in enumerate((("s64", "0xa0000"), ("s64", "0xa000a"),
                                           ("s64", "0xa0014"), ("s66", "0xa0000"))):
        body.extend([
            f"s_bfe_u32 s80, {src}, {ctrl}",
            "s_mov_b32 s81, 0",
            "s_lshl_b64 s[80:81], s[80:81], 13",
            "s_or_b64 s[80:81], s[80:81], 0x3f800000",
            f"v_mov_b32_e32 v{12 + channel}, s80",
        ])

    pos = program.index("v_cvt_pkrtz_f16_f32_e32 v16, v12, v13")
    program[pos:pos] = body
    return program


def scalar64_reference(program, words):
    """Independent Python ISA reference for scalar64() using wide integers."""
    mask32, mask64 = 0xffffffff, 0xffffffffffffffff
    regs = {}
    scc = None
    channels = {}

    def read32(tok):
        if tok.startswith("s"):
            return regs[tok] & mask32
        return int(tok, 0) & mask32

    def read64(tok, signed=False):
        if tok.startswith("s["):
            lo, hi = map(int, tok[2:-1].split(":"))
            return (regs[f"s{lo}"] & mask32) | ((regs[f"s{hi}"] & mask32) << 32)
        if tok.startswith("0x"):
            v = int(tok, 16)
            return v | (0xffffffff00000000 if signed and (v & 0x80000000) else 0)
        return int(tok, 0) & mask64

    def write64(tok, val):
        lo, hi = map(int, tok[2:-1].split(":"))
        val &= mask64
        regs[f"s{lo}"] = val & mask32
        regs[f"s{hi}"] = (val >> 32) & mask32

    for line in program:
        op, _, rest = line.partition(" ")
        ops = [t.strip() for t in rest.split(",")] if rest else []
        if op == "s_buffer_load_dwordx4":
            lo = int(ops[0][2:-1].split(":")[0])
            start = (0 if ops[2] == "null" else int(ops[2], 16)) // 4
            for j in range(4):
                regs[f"s{lo + j}"] = words[start + j]
        elif op == "s_mov_b32" and ops[0].startswith("s") and not ops[1].startswith("s2"):
            regs[ops[0]] = read32(ops[1])
        elif op == "s_and_b64" and ops[0].startswith("s["):
            v = read64(ops[1]) & read64(ops[2])
            write64(ops[0], v)
            scc = v != 0
        elif op == "s_or_b64":
            v = read64(ops[1]) | read64(ops[2])
            write64(ops[0], v)
            scc = v != 0
        elif op == "s_xor_b64":
            v = read64(ops[1]) ^ read64(ops[2])
            write64(ops[0], v)
            scc = v != 0
        elif op == "s_andn2_b64":
            v = read64(ops[1]) & (~read64(ops[2]) & mask64)
            write64(ops[0], v)
            scc = v != 0
        elif op == "s_orn2_b64":
            v = read64(ops[1]) | (~read64(ops[2]) & mask64)
            write64(ops[0], v)
            scc = v != 0
        elif op in ("s_nand_b64", "s_nor_b64", "s_xnor_b64"):
            a, b = read64(ops[1]), read64(ops[2])
            v = ~{"s_nand_b64": a & b, "s_nor_b64": a | b, "s_xnor_b64": a ^ b}[op] & mask64
            write64(ops[0], v)
            scc = v != 0
        elif op == "s_not_b64":
            v = ~read64(ops[1]) & mask64
            write64(ops[0], v)
            scc = v != 0
        elif op == "s_brev_b64":
            x = read64(ops[1])
            v = int(f"{x:064b}"[::-1], 2)
            write64(ops[0], v)
        elif op == "s_cmov_b64":
            assert scc is not None
            if scc:
                write64(ops[0], read64(ops[1]))
        elif op == "s_lshl_b64":
            v = (read64(ops[1]) << (read32(ops[2]) & 63)) & mask64
            write64(ops[0], v)
            scc = v != 0
        elif op == "s_lshr_b64":
            v = read64(ops[1]) >> (read32(ops[2]) & 63)
            write64(ops[0], v)
            scc = v != 0
        elif op == "s_ashr_i64":
            x = read64(ops[1], signed=True)
            if x >= (1 << 63):
                x -= 1 << 64
            v = (x >> (read32(ops[2]) & 63)) & mask64
            write64(ops[0], v)
            scc = v != 0
        elif op == "s_bfm_b64":
            w = read32(ops[1]) & 63
            off = read32(ops[2]) & 63
            v = (((1 << w) - 1) << off) & mask64
            write64(ops[0], v)
        elif op in ("s_bfe_u32", "s_bfe_i32"):
            src = read32(ops[1])
            ctrl = read32(ops[2])
            off = ctrl & 31
            w = (ctrl >> 16) & 127
            if w == 0:
                v = 0
            elif op == "s_bfe_u32":
                v = (src >> off) & ((1 << min(w, 32)) - 1)
            else:
                if off + w < 32:
                    raw = (src >> off) & ((1 << w) - 1)
                    if raw & (1 << (w - 1)):
                        raw -= 1 << w
                else:
                    raw = (src - (1 << 32) if src & 0x80000000 else src) >> off
                v = raw & mask32
            regs[ops[0]] = v
            scc = v != 0
        elif op == "v_mov_b32_e32" and ops[0] in ("v12", "v13", "v14", "v15"):
            bits = regs[ops[1]] & mask32
            val, = struct.unpack("<f", struct.pack("<I", bits))
            assert half_rtz(val) == val, hex(bits)
            channels[ops[0]] = val
    return tuple(channels[f"v{12 + c}"] for c in range(4)), regs["s66"] & 0x3ff


def scalar64_case(directory, values, *, name="scalar64.case"):
    constants = bytearray(256)
    struct.pack_into("<12I", constants, 16, *values)
    words = struct.unpack("<64I", constants)
    color, _ = scalar64_reference(scalar64(), words)
    return write_case(directory / name, constants, tuple(color))
# Scalar ALU producers in the scalar() fixture. Each SCC result sets its own bit of s60 through
# s_cselect_b32; each result value is mixed into the hash in s61. a to d are constant words 3 to 6.
SCALAR_STEPS = (
    ("s_add_u32 s62, s44, s45", True), ("s_sub_u32 s62, s46, s47", True),
    ("s_add_i32 s62, s46, s45", True), ("s_sub_i32 s62, s46, s45", True),
    ("s_min_u32 s62, s45, s44", True), ("s_min_i32 s62, s44, s45", True),
    ("s_max_u32 s62, s44, s45", True), ("s_max_i32 s62, s45, s44", True),
    ("s_and_b32 s62, s44, s47", True), ("s_xor_b32 s62, s44, s45", True),
    ("s_lshr_b32 s62, s46, 31", True),
    ("s_cmp_eq_u32 s44, s45", False), ("s_cmp_lg_u32 s44, s45", False),
    ("s_cmp_gt_i32 s45, s44", False), ("s_cmp_ge_i32 s44, s45", False),
    ("s_cmp_lt_i32 s44, s45", False), ("s_cmp_le_i32 s45, s44", False),
    ("s_cmp_gt_u32 s44, s45", False), ("s_cmp_ge_u32 s45, s44", False),
    ("s_cmp_lt_u32 s45, s44", False), ("s_cmp_le_u32 s44, s45", False),
    ("s_cmp_eq_i32 s47, -1", False), ("s_cmp_lg_i32 s46, 0x80000000", False),
)
# Results that leave SCC unused or unchanged; s_cselect_b32 here selects between two registers.
SCALAR_VALUES = ("s_mul_i32 s62, s44, s46", "s_or_b32 s62, s45, s46", "s_andn2_b32 s62, s44, s45",
                 "s_orn2_b32 s62, s47, s44", "s_not_b32 s62, s46", "s_lshl_b32 s62, s44, s47",
                 "s_ashr_i32 s62, s46, s44", "s_nand_b32 s62, s44, s45", "s_nor_b32 s62, s45, s46",
                 "s_xnor_b32 s62, s46, s47", "s_cmp_lt_u32 s44, s45", "s_cselect_b32 s62, s46, s47")
# Two constant sets; every SCC bit differs between them, so both arms of each select are drawn.
SCALAR_WORDS = ((0xfffffff0, 0x20, 0x80000000, 1), (5, 5, 0x7fffffff, 0xffffffff))


def scalar_body():
    body = ["s_mov_b32 s60, 0", "s_movk_i32 s61, 0xff80"]
    for bit, (step, value) in enumerate(SCALAR_STEPS):
        flag = 1 << bit
        body += [step, f"s_cselect_b32 s63, {flag if flag <= 64 else hex(flag)}, 0", "s_or_b32 s60, s60, s63"]
        if value:
            body += ["s_mul_i32 s61, s61, 31", "s_add_u32 s61, s61, s62"]
    for step in SCALAR_VALUES:
        body.append(step)
        if step.startswith("s_cselect") or not step.startswith("s_cmp"):
            body += ["s_mul_i32 s61, s61, 31", "s_add_u32 s61, s61, s62"]
    # Fold the 32-bit hash to 17 bits and spread 23 flag bits and 17 hash bits over four channels,
    # ten bits each, exported as 1 + k / 1024, which binary16 holds exactly.
    body += ["s_lshr_b32 s63, s61, 17", "s_xor_b32 s61, s61, s63", "s_and_b32 s61, s61, 0x1ffff",
             "s_and_b32 s56, s60, 0x3ff",
             "s_lshr_b32 s57, s60, 10", "s_and_b32 s57, s57, 0x3ff",
             "s_lshr_b32 s58, s60, 20", "s_lshl_b32 s63, s61, 3", "s_or_b32 s58, s58, s63",
             "s_and_b32 s58, s58, 0x3ff",
             "s_lshr_b32 s59, s61, 7"]
    for channel in range(4):
        body += [f"s_lshl_b32 s{56 + channel}, s{56 + channel}, 13",
                 f"s_or_b32 s{56 + channel}, s{56 + channel}, 1.0",
                 f"v_mov_b32_e32 v{12 + channel}, s{56 + channel}"]
    return body


PACK_HALVES = ("ll", "lh", "hh")


def scalar_carry_chain(subtract, *, initialize_scc=True):
    low, high = ("s_sub_u32", "s_subb_u32") if subtract else ("s_add_u32", "s_addc_u32")
    setup = [f"{low} s62, s44, s46", "s_cselect_b32 s63, 1, 0"] if initialize_scc else [
        "s_mov_b32 s62, 0", "s_mov_b32 s63, 0",
    ]
    return uncompressed_export([
        *setup,
        f"{high} s45, s45, s47", "s_cselect_b32 s64, 1, 0",
        "v_mov_b32_e32 v12, s62", "v_mov_b32_e32 v13, s45",
        "v_mov_b32_e32 v14, s63", "v_mov_b32_e32 v15, s64",
    ])


def check_scalar_carry_chain_draw(directory, compiler, renderer, backend):
    source, shader = directory / "carry-chain.hlsl", directory / "carry-chain.cso"
    for subtract in (False, True):
        program = scalar_carry_chain(subtract)
        source.write_text(translate_int(program), encoding="ascii")
        compile_shader(compiler, source, shader)
        pairs = ((0, 0), (0x100000000, 1), (0, 0x100000000), (0, 1),
                 (0, 0xffffffff00000001), (0x0002000000000000, 0x0001000000000001),
                 (0xffffffff00000000, 0xffffffff00000001)) if subtract else (
                     (0, 0), (0xffffffff, 1), (0xffffffff00000000, 0x100000000),
                     (0xffffffffffffffff, 1))
        cases = []
        for idx, (a, b) in enumerate(pairs):
            low_a, low_b = a & 0xffffffff, b & 0xffffffff
            low = low_a - low_b if subtract else low_a + low_b
            wide = a - b if subtract else a + b
            words = (wide & 0xffffffff, (wide >> 32) & 0xffffffff,
                     not 0 <= low < 1 << 32, not 0 <= wide < 1 << 64)
            constants = bytearray(256)
            struct.pack_into("<QQ", constants, 12, a, b)
            case = directory / f"carry-chain-{int(subtract)}-{idx}.case"
            write_case(case, constants, tuple(as_float(w) for w in words), integer_rt=True)
            cases.append((case, "pass"))
        run_pixel_batch(renderer, backend, shader, cases)
        # The final case wraps only because of the incoming SCC.
        old, new = ("s_subb_u32", "s_sub_u32") if subtract else ("s_addc_u32", "s_add_u32")
        source.write_text(translate_int([line.replace(old, new) for line in program]), encoding="ascii")
        compile_shader(compiler, source, shader)
        wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, cases[-1][0]],
                               capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Dropping SCC from {old} escaped detection")

    program = scalar_carry_chain(False)
    program[program.index("s_addc_u32 s45, s45, s47")] = "s_addc_u32 s45, s45, 0xffffffff"
    source.write_text(translate_int(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    # Carry in is set in both rows. With high A = 1 the first addition wraps.
    # With high A = 0 only the addition of carry in wraps.
    for high_a in (1, 0):
        constants = bytearray(256)
        struct.pack_into("<QQ", constants, 12, (high_a << 32) | 0xffffffff, 1)
        words = (0, high_a, 1, 1)
        write_case(case, constants, tuple(as_float(w) for w in words), integer_rt=True)
        subprocess.run([renderer, backend, "--check-pixel-case", shader, case], check=True)


def scalar_leading_bit_values(width):
    ops = ("s_flbit_i32_b32", "s_flbit_i32") if width == 32 else ("s_flbit_i32_b64", "s_flbit_i32_i64")
    src = "s44" if width == 32 else "s[44:45]"
    body = []
    for i, op in enumerate(ops):
        body.extend([
            # s_flbit does not write SCC: rows whose words are equal, including the
            # zero pair that finds no bit, must keep the false SCC seeded here.
            "s_cmp_lg_u32 s44, s45", f"{op} s62, {src}",
            "s_cselect_b32 s63, 0x3f800000, 0", f"v_mov_b32_e32 v{14 + i}, s63",
            f"s_cmp_eq_u32 s62, s{46 + i}",
            "s_cselect_b32 s63, 0x3f800000, 0", f"v_mov_b32_e32 v{12 + i}, s63",
        ])
    return with_constants(body)


def check_scalar_leading_bit_draw(directory, compiler, renderer, backend):
    source, shader = directory / "leading-bit.hlsl", directory / "leading-bit.cso"
    for width in (32, 64):
        source.write_text(translate(scalar_leading_bit_values(width)), encoding="ascii")
        compile_shader(compiler, source, shader, ieee=False)
        cases = []
        extra_rows = ((0x40000000, 0), (0x00010000, 0)) if width == 32 else ((0, 0x40000000),)
        for idx, (lo, hi) in enumerate(extra_rows + ((0, 0), (0xffffffff, 0xffffffff), (0x80000000, 0x80000000),
                                                     (0, 1), (0x80000000, 0), (1, 0), (0xfffffffe, 0xffffffff))):
            word = lo if width == 32 else lo | (hi << 32)
            bits = f"{word:0{width}b}"
            unsigned = bits.find("1")
            signed = bits.find("0" if bits[0] == "1" else "1")
            constants = bytearray(256)
            struct.pack_into("<4I", constants, 12, lo, hi, unsigned & 0xffffffff, signed & 0xffffffff)
            case = directory / f"leading-bit-{width}-{idx}.case"
            case.write_bytes(constants + struct.pack("<4f", 1, 1, lo != hi, lo != hi) * 64)
            cases.append((case, "pass"))
        run_pixel_batch(renderer, backend, shader, cases)
    # For -2, the signed search must cross into the low word; the unsigned search returns zero.
    program = [line.replace("s_flbit_i32_i64", "s_flbit_i32_b64") for line in scalar_leading_bit_values(64)]
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader, ieee=False)
    wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, cases[-1][0]],
                           capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Unsigned leading-bit scan matched the signed reference")


def scalar_first_bit_values(width):
    src = "s44" if width == 32 else "s[44:45]"
    body = []
    for bit in (0, 1):
        body.extend([
            "s_cmp_eq_u32 s44, s45",
            f"s_ff{bit}_i32_b{width} s62, {src}",
            "s_cselect_b32 s63, 0x3f800000, 0", f"v_mov_b32_e32 v{14 + bit}, s63",
            f"s_cmp_eq_u32 s62, s{46 + bit}",
            "s_cselect_b32 s63, 0x3f800000, 0", f"v_mov_b32_e32 v{12 + bit}, s63",
        ])
    return with_constants(body)


def check_scalar_first_bit_draw(directory, compiler, renderer, backend):
    source, shader, case = (directory / name for name in ("first-bit.hlsl", "first-bit.cso", "first-bit.case"))
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    for width in (32, 64):
        source.write_text(translate(scalar_first_bit_values(width)), encoding="ascii")
        compile_shader(compiler, source, shader, ieee=False)
        for lo, hi in ((0, 0), (0xffffffff, 0xffffffff), (0xffffffff, 0xfffffffe),
                       (0x80000000, 1), (0, 1), (0, 0x80000000)):
            word = lo if width == 32 else lo | (hi << 32)
            expected = [next((i for i in range(width) if (word >> i) & 1 == bit), 0xffffffff)
                        for bit in (0, 1)]
            constants = bytearray(256)
            struct.pack_into("<4I", constants, 12, lo, hi, *expected)
            case.write_bytes(constants + struct.pack("<4f", 1, 1, lo == hi, lo == hi) * 64)
            subprocess.run(draw, check=True)
    # For the final input, scanning only the low word misses the set bit at position 63.
    program = [line.replace("s_ff1_i32_b64 s62, s[44:45]", "s_ff1_i32_b32 s62, s44")
               for line in scalar_first_bit_values(64)]
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader, ieee=False)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Ignoring the high word escaped detection")


def scalar_bit_count_values(width):
    src = "s44" if width == 32 else "s[44:45]"
    body = []
    for bit in (0, 1):
        # Seed SCC from a runtime word. The reference always seeds the opposite of the expected SCC.
        body.extend([
            f"s_cmp_eq_u32 s{46 + bit}, 1",
            f"s_bcnt{bit}_i32_b{width} s62, {src}",
            "s_cselect_b32 s63, 0x3f800000, 0", f"v_mov_b32_e32 v{14 + bit}, s63",
            "s_lshl_b32 s62, s62, 13", "s_or_b32 s62, s62, 0x3f800000",
            f"v_mov_b32_e32 v{12 + bit}, s62",
        ])
    return with_constants(body)


def check_scalar_bit_count_draw(directory, compiler, renderer, backend):
    source, shader = directory / "bit-count.hlsl", directory / "bit-count.cso"
    register_rows = ((0, 0), (0xffffffff, 0xffffffff), (0x1234abcd, 0x5678ef90), (0, 0x80000000))
    # Keep the high-word-only register case last for the mismatch control below.
    for width, rows in (
        (32, ((0xfffffffe, 0), (1, 0)) + register_rows),
        (64, ((1, 1), (0xffffffff, 0xfffffffe)) + register_rows),
    ):
        program = scalar_bit_count_values(width)
        source.write_text(translate(program), encoding="ascii")
        compile_shader(compiler, source, shader, ieee=False)
        cases = []
        for idx, (lo, hi) in enumerate(rows):
            word = lo if width == 32 else lo | (hi << 32)
            ones = word.bit_count()
            zeros = width - ones
            constants = bytearray(256)
            struct.pack_into("<4I", constants, 12, lo, hi, zeros == 0, ones == 0)
            case = directory / f"bit-count-{width}-{idx}.case"
            case.write_bytes(constants + struct.pack("<4f", 1 + zeros / 1024, 1 + ones / 1024,
                                                      zeros != 0, ones != 0) * 64)
            cases.append((case, "pass"))
        run_pixel_batch(renderer, backend, shader, cases)
    # The last input has one bit set in the high word and none in the low word.
    program = [line.replace("s_bcnt1_i32_b64 s62, s[44:45]", "s_bcnt1_i32_b32 s62, s44")
               for line in scalar_bit_count_values(64)]
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader, ieee=False)
    wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, cases[-1][0]],
                           capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Dropping the high word matched the original reference")


def scalar_mul_hi_values(signed="i32"):
    return with_constants([
        "s_cmp_eq_u32 s44, s45",
        f"s_mul_hi_{signed} s62, s44, s45",
        "s_mul_hi_u32 s63, s44, s45",
        "s_cselect_b32 s64, 0x3f800000, 0",
        "v_mov_b32_e32 v14, s64",
        "s_cmp_eq_u32 s62, s46", "s_cselect_b32 s64, 0x3f800000, 0",
        "v_mov_b32_e32 v12, s64",
        "s_cmp_eq_u32 s63, s47", "s_cselect_b32 s64, 0x3f800000, 0",
        "v_mov_b32_e32 v13, s64", "v_mov_b32_e32 v15, 0",
    ])


def check_scalar_mul_hi_draw(directory, compiler, renderer, backend):
    source, shader = directory / "mul-hi.hlsl", directory / "mul-hi.cso"
    source.write_text(translate(scalar_mul_hi_values()), encoding="ascii")
    compile_shader(compiler, source, shader, ieee=False)
    cases = []
    for idx, (a, b) in enumerate(((1, 0), (0xffffffff, 1), (0, 0x80000000), (0x7fffffff, 0x7fffffff), (0x80000000, 3),
                                  (3, 0x80000000), (0x1234abcd, 0x5678ef90), (0xffffffff, 0xffffffff))):
        signed_a, signed_b = struct.unpack("<ii", struct.pack("<II", a, b))
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, a, b, (signed_a * signed_b >> 32) & 0xffffffff, a * b >> 32)
        case = directory / f"mul-hi-{idx}.case"
        case.write_bytes(constants + struct.pack("<4f", 1, 1, a == b, 0) * 64)
        cases.append((case, "pass"))
    run_pixel_batch(renderer, backend, shader, cases)
    # For the final inputs, signed high is zero and unsigned high is 0xfffffffe.
    source.write_text(translate(scalar_mul_hi_values("u32")), encoding="ascii")
    compile_shader(compiler, source, shader, ieee=False)
    wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, cases[-1][0]],
                           capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Unsigned high product matched the signed reference")


def scalar_pack_values(halves):
    return with_constants([
        "s_cmp_eq_u32 s47, 1", f"s_pack_{halves}_b32_b16 s44, s44, s45",
        "s_cselect_b32 s62, 0x3f800000, 0", "v_mov_b32_e32 v13, s62",
        "s_cmp_eq_u32 s44, s46", "s_cselect_b32 s62, 0x3f800000, 0",
        "v_mov_b32_e32 v12, s62", "v_mov_b32_e32 v14, 0", "v_mov_b32_e32 v15, 0",
    ])


def check_scalar_pack_draw(directory, compiler, renderer, backend):
    source, shader = directory / "pack.hlsl", directory / "pack.cso"
    for halves in PACK_HALVES:
        source.write_text(translate(scalar_pack_values(halves)), encoding="ascii")
        compile_shader(compiler, source, shader, ieee=False)
        cases = []
        # The third pair sets every bit of each half LL and LH select, so a mask that drops any one bit
        # of a selected half (0xfffd for the low half, 0xfffe0000 for the high half) breaks the equality.
        for index, words in enumerate(((0x1234abcd, 0x5678ef90), (0xffff0000, 0x0000ffff),
                                       (0x0000ffff, 0xffffffff))):
            flag = index % 2
            selected = [struct.unpack("<HH", struct.pack("<I", word))[half == "h"]
                        for word, half in zip(words, halves)]
            constants = bytearray(256)
            struct.pack_into("<IIHHI", constants, 12, *words, *selected, flag)
            case = directory / f"pack-{halves}-{index}.case"
            case.write_bytes(constants + struct.pack("<4f", 1, flag, 0, 0) * 64)
            cases.append((case, "pass"))
        run_pixel_batch(renderer, backend, shader, cases)
        if halves == "lh":
            # Compile LL against the unchanged LH reference; only the selected high half differs.
            source.write_text(translate(scalar_pack_values("ll")), encoding="ascii")
            mutant = directory / "wrong-pack.cso"
            compile_shader(compiler, source, mutant, ieee=False)
            wrong = subprocess.run([renderer, backend, "--check-pixel-case", mutant, cases[0][0]],
                                   capture_output=True, text=True)
            if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
                raise AssertionError("LL matched the LH reference")


def sext_values():
    body = []
    for index, width in enumerate((8, 16)):
        body.extend(["s_cmp_eq_u32 s47, 1", f"s_sext_i32_i{width} s62, s44",
                     "s_cselect_b32 s63, 0x3f800000, 0",
                     f"v_mov_b32_e32 v{14 + index}, s63",
                     f"s_cmp_eq_u32 s62, s{45 + index}", "s_cselect_b32 s63, 0x3f800000, 0",
                     f"v_mov_b32_e32 v{12 + index}, s63"])
    return with_constants(body)


def check_sext_draw(directory, compiler, renderer, backend):
    source = directory / "sext.hlsl"
    shader = directory / "sext.cso"
    source.write_text(translate(sext_values()), encoding="ascii")
    compile_shader(compiler, source, shader, ieee=False)
    cases = []
    for index, word in enumerate((0xabcd807f, 0xabcd7f80, 0xabcd0000, 0x1234ffff)):
        flag = index % 2
        raw = word.to_bytes(4, "little")
        expected = [int.from_bytes(raw[:size], "little", signed=True) for size in (1, 2)]
        constants = bytearray(256)
        struct.pack_into("<IiiI", constants, 12, word, *expected, flag)
        case = directory / f"sext-{index}.case"
        case.write_bytes(constants + struct.pack("<4f", 1, 1, flag, flag) * 64)
        cases.append((case, "pass"))
    # Replacing both signed references with their zero-extended values must change both channels.
    struct.pack_into("<II", constants, 16, 255, 65535)
    zero_ext = directory / "sext-zext.case"
    zero_ext.write_bytes(constants + struct.pack("<4f", 0, 0, flag, flag) * 64)
    cases.append((zero_ext, "pass"))
    run_pixel_batch(renderer, backend, shader, cases)


SOP1_WORDS = (
    (0, 0x2468ace1, 0x12400009, 3),
    (0xfffffffb, 0x2468ace1, 0x12400009, 35),
    (5, 0x11223344, 0x81400008, 0),
    (0x80000000, 0x55667788, 0x81400009, 31),
)


def sop1_scalar_ops() -> list[str]:
    return uncompressed_export([
        "s_abs_i32 s50, s44",
        "s_mov_b32 s51, s46",
        "s_cmov_b32 s51, s45",
        "s_mov_b32 s52, s46",
        "s_bitset0_b32 s52, s47",
        "s_bitset1_b32 s52, 31",
        "s_mov_b32 s53, s46",
        "s_cmov_b32 s53, 0x12345678",
        "v_mov_b32_e32 v12, s50",
        "v_mov_b32_e32 v13, s51",
        "v_mov_b32_e32 v14, s52",
        "v_mov_b32_e32 v15, s53",
    ])


def sop1_scalar_ops_reference(words, set_bit=31):
    s44, s45, s46, s47 = words
    signed44 = s44 - (1 << 32) if s44 >> 31 else s44
    abs44 = (-signed44 if signed44 < 0 else signed44) & 0xffffffff
    scc = abs44 != 0
    cmov_reg = s45 if scc else s46
    bitset = (s46 & ~(1 << (s47 & 31))) | (1 << set_bit)
    cmov_lit = 0x12345678 if scc else s46
    return (abs44, cmov_reg, bitset & 0xffffffff, cmov_lit)


# s_bfm_b32/s_bfm_b64: count words s44 (S0) and s45 (S1 offset) feed both widths; s46/s47 give a second 32-bit mask.
BFM_WORDS = (
    (0, 0, 1, 31),
    (31, 1, 32, 32),
    (32, 31, 33, 0xffffffff),
    (63, 33, 64, 63),
    (64, 5, 0xffffffe1, 0xffffffc3),
)


def bfm_scalar_ops() -> list[str]:
    return uncompressed_export([
        "s_bfm_b32 s50, s44, s45",
        "s_bfm_b32 s51, s46, s47",
        "s_bfm_b64 s[52:53], s44, s45",
        "v_mov_b32_e32 v12, s50",
        "v_mov_b32_e32 v13, s51",
        "v_mov_b32_e32 v14, s52",
        "v_mov_b32_e32 v15, s53",
    ])


def bfm_scalar_ops_reference(words):
    """Counts are masked to 5 (b32) or 6 (b64) bits; the mask is (1 << count) - 1, shifted left by the offset."""
    s44, s45, s46, s47 = words
    b32 = lambda count, offset: (((1 << (count & 31)) - 1) << (offset & 31)) & 0xffffffff
    b64 = (((1 << (s44 & 63)) - 1) << (s45 & 63)) & 0xffffffffffffffff
    return (b32(s44, s45), b32(s46, s47), b64 & 0xffffffff, b64 >> 32)


# s_brev_b32: s46 and s47 are reversed, s44 and s45 only set SCC. The words cover 0, all ones, the top bit, bit 0,
# alternating bits and an asymmetric pattern, with SCC set for equal s44/s45 and clear otherwise.
BREV_WORDS = (
    (5, 5, 0x00000000, 0xffffffff),
    (5, 6, 0xffffffff, 0x80000000),
    (7, 7, 0x80000000, 0x00000001),
    (1, 2, 0x00000001, 0xaaaaaaaa),
    (9, 9, 0xaaaaaaaa, 0x12345678),
    (3, 4, 0x12345678, 0x55555555),
    (8, 8, 0xffffffff, 0x80000000),
)
BREV_LITERAL = 0x12345678


def brev_scalar_ops() -> list[str]:
    """s_brev_b32 between an SCC-setting compare and the select that reads it: the select proves SCC is preserved."""
    return uncompressed_export([
        "s_cmp_eq_u32 s44, s45",
        "s_brev_b32 s50, s46",
        f"s_brev_b32 s51, {BREV_LITERAL:#x}",
        "s_cselect_b32 s52, s47, 0",
        "s_brev_b32 s53, s47",
        "v_mov_b32_e32 v12, s50",
        "v_mov_b32_e32 v13, s51",
        "v_mov_b32_e32 v14, s52",
        "v_mov_b32_e32 v15, s53",
    ])


def brev_scalar_ops_reference(words):
    """Independent bit loop; the select arm shows SCC from the compare (s44 == s45) survived both reversals."""
    s44, s45, s46, s47 = words

    def reverse(value):
        result = 0
        for bit in range(32):
            if value >> bit & 1:
                result |= 1 << (31 - bit)
        return result

    return (reverse(s46), reverse(BREV_LITERAL), s47 if s44 == s45 else 0, reverse(s47))


# Plain VOP1/VOP2 (and e64) v_bfrev_b32, v_ffbh_u32, v_ffbh_i32, v_ffbl_b32 and v_xnor_b32. The sources are the
# constant words in s44..s47, so no input passes through a float VGPR. Case k rotates the nine words over s44..s47.
PLAIN_BITSCAN_W = (0, 0xffffffff, 0x80000000, 0x00000001, 0x7fffffff, 0x0000ffff, 0xaaaaaaaa, 0x12345678, 0xfffffffe)
PLAIN_BITSCAN_WORDS = tuple(tuple(PLAIN_BITSCAN_W[(k + j) % 9] for j in range(4)) for k in range(9))
PLAIN_BITSCAN_LITERAL = 0x0f0f0f0f


def plain_bitscan_ops(second=False, vgpr_source=False) -> list[str]:
    """Fixture 1 (second=False) uses the plain spellings; fixture 2 mixes explicit _e32 and _e64 and a literal."""
    if second:
        ops = ["v_ffbl_b32_e32 v24, s44", "v_ffbh_u32_e64 v25, s45",
               "v_xnor_b32_e64 v26, s46, 0x0f0f0f0f", "v_bfrev_b32_e64 v27, s47"]
    else:
        ops = ["v_bfrev_b32 v24, s44", "v_ffbh_u32 v25, s45", "v_ffbh_i32 v26, s46", "v_xnor_b32 v27, s47, s44"]
    if vgpr_source:  # a VGPR source produced by an earlier integer op
        ops += ["v_xor_b32 v30, s44, s45", "v_bfrev_b32 v31, v30"]
    return uncompressed_export(ops + [f"v_mov_b32_e32 v{12 + i}, v{24 + i}" for i in range(4)])


def plain_rev(x):
    return sum(((x >> i) & 1) << (31 - i) for i in range(32))


def plain_ffbl(x):
    """Index of the lowest set bit; all ones when no bit is set."""
    return next((i for i in range(32) if (x >> i) & 1), 0xffffffff)


def plain_ffbh_u(x):
    """Zero-based position of the highest set bit counted from the MSB; all ones when no bit is set."""
    return next((i for i in range(32) if (x >> (31 - i)) & 1), 0xffffffff)


def plain_ffbh_i(x):
    """Position, counted from the MSB, of the first bit below the sign bit that differs from it; all ones if none."""
    sign = x >> 31
    return next((i for i in range(1, 32) if ((x >> (31 - i)) & 1) != sign), 0xffffffff)


def plain_bitscan_reference(words):
    """Expected (fixture 1, fixture 2) channels, from bit loops written independently of the HLSL helpers."""
    s44, s45, s46, s47 = words
    xnor = lambda a, b: ~(a ^ b) & 0xffffffff
    return ((plain_rev(s44), plain_ffbh_u(s45), plain_ffbh_i(s46), xnor(s47, s44)),
            (plain_ffbl(s44), plain_ffbh_u(s45), xnor(s46, PLAIN_BITSCAN_LITERAL), plain_rev(s47)))



# Non-packed 16-bit integer VOP3 ops (Refs #669). v18..v21 hold the constant words 0..3 (s44..s47). The translator
# tracks each result as ("low16", alu): only bits 15:0 are defined, and every reader that could observe the high
# half is refused. Groups 0..2 run four ops into v24..v27 and widen each result with
# v_mov_b32_sdwa ..., src0_sel:WORD_0 into v12..v15, so the export sees the zero-extended low half. The v24..v27
# registers start as 0xa5c3xxxx sentinels; they would catch a sentinel merged into the low half only; they do not
# detect a high-half leak (the widening never reads the high half). Group 3 runs a chain and 16-bit compares.
# Operand mapping, one row per op (dst = src0, src1), a = low half of src0, b = low half of src1:
#   group 0: v24 = add_nc_u16 v18, v19 (w0 + w1)       v25 = sub_nc_u16 s46, v18 (w2 - w0)
#            v26 = add_nc_i16 v20, 5   (w2 + 5)        v27 = sub_nc_i16 0x1234, v21 (0x1234 - w3)
#   group 1: v24 = min_u16 v18, v19 (w0, w1)           v25 = max_u16 s46, v21 (w2, w3)
#            v26 = min_i16 v18, v19 (w0, w1)           v27 = max_i16 v20, -2 (w2, 0xfffe)
#   group 2: v24 = mul_lo_u16 v18, v19 (w0 * w1)       v25 = lshlrev_b16 v20, v18 (w0 << (w2 & 15))
#            v26 = lshrrev_b16 v19, v21 (w3 >> (w1 & 15))   v27 = ashrrev_i16 s47, v18 (sext(w0) >> (w3 & 15))
#   group 3: v24 = add_nc_u16 v18, v19; v25 = mul_lo_u16 v24, v20 (a low16 result as a source);
#            v26 = v_cmp_lt_u16 v24, v21 selecting v18 or v19 (whole words);
#            v27 = v_cmp_lt_i16 v25, v24 selecting v20 or v21 (whole words)
INT16_VOP3_GROUPS = (
    (("add_nc_u16", "v18", "v19"), ("sub_nc_u16", "s46", "v18"),
     ("add_nc_i16", "v20", "5"), ("sub_nc_i16", "0x1234", "v21")),
    (("min_u16", "v18", "v19"), ("max_u16", "s46", "v21"),
     ("min_i16", "v18", "v19"), ("max_i16", "v20", "-2")),
    (("mul_lo_u16", "v18", "v19"), ("lshlrev_b16", "v20", "v18"),
     ("lshrrev_b16", "v19", "v21"), ("ashrrev_i16", "s47", "v18")),
)
INT16_VOP3_CHAIN = ("v_add_nc_u16 v24, v18, v19", "v_mul_lo_u16 v25, v24, v20",
                    "v_cmp_lt_u16 vcc_lo, v24, v21", "v_cndmask_b32 v26, v18, v19, vcc_lo",
                    "v_cmp_lt_i16 vcc_lo, v25, v24", "v_cndmask_b32 v27, v20, v21, vcc_lo")
INT16_VOP3_SENTINELS = (0xa5c31111, 0xa5c32222, 0xa5c33333, 0xa5c34444)
INT16_VOP3_WIDEN = "v_mov_b32_sdwa v{dst}, v{src} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0"
# Four constant words per set. Cover 0x7fff/0x8000, 0xffff + 1, 0 - 1, shift counts 15/16/17/31, negative b under
# ashrrev, signed/unsigned min/max disagreement, a mul_lo overflow (0x1234 * 0x0567) and high halves that differ from
# the low halves. Set index 2 gives lshrrev/ashrrev count 16 with b != 0 and lshlrev count 17 with b = 1.
INT16_VOP3_WORDS = (
    (0xdead7fff, 0x12348000, 0xbeef0001, 0x5a5a0002),
    (0x1111ffff, 0x22220001, 0x33330000, 0x4444ffff),
    (0xabcd0001, 0x3c1d0010, 0x77770011, 0x13570010),
    (0xf00f1234, 0x801f0567, 0x8000001f, 0x9999001f),
    (0x00008000, 0xffff7fff, 0x0000000f, 0xc0de0000),
    (0xfffe0000, 0x0001ffff, 0x80000010, 0x7ffe8001),
)


def int16_vop3_ops(group):
    """The group's ops on the constant words, widened with SDWA WORD_0 into v12..v15 and exported."""
    body = [f"v_mov_b32_e32 v{18 + k}, s{44 + k}" for k in range(4)]
    body += [f"v_mov_b32_e32 v{24 + k}, {word:#x}" for k, word in enumerate(INT16_VOP3_SENTINELS)]
    if group < 3:
        body += [f"v_{op} v{24 + k}, {src0}, {src1}" for k, (op, src0, src1) in enumerate(INT16_VOP3_GROUPS[group])]
        body += [INT16_VOP3_WIDEN.format(dst=12 + k, src=24 + k) for k in range(4)]
    else:
        body += INT16_VOP3_CHAIN
        body += [INT16_VOP3_WIDEN.format(dst=12, src=24), INT16_VOP3_WIDEN.format(dst=13, src=25),
                 "v_mov_b32_e32 v14, v26", "v_mov_b32_e32 v15, v27"]
    return uncompressed_export(body)


def int16_vop3_reference(group, words, *, mutation=None):
    """Four exported words from the semantics table (a, b = low halves of src0 and src1), without pixel_aot.

    Only the zero-extended low half of each 16-bit result is observed. mutation names a deliberate deviation that the
    draw's mutated shaders model: shift31, min_unsigned, source_high (a source read from bits 31:16), low16_chain or
    low16_compare (a low16 result read through >> 16u as a source of the chained op or of the first compare).
    """
    def signed(x):
        return x - 0x10000 if x & 0x8000 else x

    if group == 3:
        w0, w1, w2, w3 = words
        r24 = ((w0 & 0xffff) + (w1 & 0xffff)) & 0xffff
        r25 = ((r24 >> 16 if mutation == "low16_chain" else r24) * (w2 & 0xffff)) & 0xffff
        first = (0 if mutation == "low16_compare" else r24) < (w3 & 0xffff)
        second = signed(r25) < signed(r24)
        return (r24, r25, w1 if first else w0, w3 if second else w2)

    def operand(token):
        if token[0] == "v":
            return words[int(token[1:]) - 18]
        if token[0] == "s":
            return words[int(token[1:]) - 44]
        return int(token, 0) & 0xffffffff

    out = []
    for k, (op, src0, src1) in enumerate(INT16_VOP3_GROUPS[group]):
        a = (operand(src0) >> 16 if mutation == "source_high" else operand(src0)) & 0xffff
        b = operand(src1) & 0xffff
        if mutation == "min_unsigned" and op == "min_i16":
            op = "min_u16"
        count = a & (31 if mutation == "shift31" else 15)
        r = {
            "add_nc_u16": lambda: a + b, "add_nc_i16": lambda: a + b,
            "sub_nc_u16": lambda: a - b, "sub_nc_i16": lambda: a - b,
            "min_u16": lambda: min(a, b), "max_u16": lambda: max(a, b),
            "min_i16": lambda: min(signed(a), signed(b)), "max_i16": lambda: max(signed(a), signed(b)),
            "mul_lo_u16": lambda: a * b,
            "lshlrev_b16": lambda: b << count, "lshrrev_b16": lambda: b >> count,
            "ashrrev_i16": lambda: signed(b) >> count,
        }[op]() & 0xffff
        out.append(r)
    return tuple(out)


# Each mutation of the draw: the groups it is applied to and the lanes whose expected value it must change.
INT16_VOP3_MUTATIONS = {
    "shift31": ((2,), (1, 2, 3)),
    "min_unsigned": ((1,), (2,)),
    "source_high": ((0,), (0, 1, 2, 3)),
    "low16_chain": ((3,), (1, 3)),
    "low16_compare": ((3,), (2,)),
}


def check_int16_vop3_mutation_coverage():
    """Every mutation must change the affected lanes for some word set, so the draw can detect it."""
    for mutation, (groups, lanes) in INT16_VOP3_MUTATIONS.items():
        for group in groups:
            for lane in lanes:
                assert any(int16_vop3_reference(group, words, mutation=mutation)[lane]
                           != int16_vop3_reference(group, words)[lane] for words in INT16_VOP3_WORDS), \
                    (mutation, group, lane)
    assert len(INT16_VOP3_WORDS) >= 5 and all(len(words) == 4 for words in INT16_VOP3_WORDS)


# Three-operand 16-bit integer VOP3 forms (issue #669). Four ops per group on the constant words (w0..w3 in v18..v21 and
# s44..s47), widened with SDWA WORD_0 into v12..v15 and exported like INT16_VOP3_GROUPS:
#   group 0: v24 = mad_u16 v18, v19, v20 (w0 * w1 + w2)      v25 = mad_i16 v21, v18, s44 (w3 * w0 + w0, signed operands)
#            v26 = mad_u16 v20, 0x1234, v19                   v27 = mad_i16 s47, -3, v21
#   group 1: v24 = min3_i16 / v25 = min3_u16 of (w0, w1, w2)  v26 = max3_i16 (w0, w1, w3)   v27 = max3_u16 s44, v19, v21
#   group 2: v24 = med3_i16 / v25 = med3_u16 of (w0, w1, w2)  v26 = med3_i16 s46, v21, 4     v27 = med3_u16 v21, -2, s46
#   group 3: v24 = add_nc_u16 v18, v19 (a 12-op result)       v25 = mad_u16 v24, v20, v21 (low16 source)
#            v26 = med3_i16 v24, v25, v18                     v27 = max3_u16 v25, v26, v24
#   group 4: v24 = sub_nc_u16 v18, v19 (a low16 producer)     v25 = med3_u16 v24, v20, v21 (low16 into unsigned med3)
#            v26 = med3_i16 v24, v25, s46                     v27 = min3_i16 v25, v26, v24
INT16_TERNARY_GROUPS = (
    (("mad_u16", "v18", "v19", "v20"), ("mad_i16", "v21", "v18", "s44"),
     ("mad_u16", "v20", "0x1234", "v19"), ("mad_i16", "s47", "-3", "v21")),
    (("min3_i16", "v18", "v19", "v20"), ("min3_u16", "v18", "v19", "v20"),
     ("max3_i16", "v18", "v19", "v21"), ("max3_u16", "s44", "v19", "v21")),
    (("med3_i16", "v18", "v19", "v20"), ("med3_u16", "v18", "v19", "v20"),
     ("med3_i16", "s46", "v21", "4"), ("med3_u16", "v21", "-2", "s46")),
)
INT16_TERNARY_CHAIN = ("v_add_nc_u16 v24, v18, v19", "v_mad_u16 v25, v24, v20, v21",
                       "v_med3_i16 v26, v24, v25, v18", "v_max3_u16 v27, v25, v26, v24")
INT16_TERNARY_MED_CHAIN = ("v_sub_nc_u16 v24, v18, v19", "v_med3_u16 v25, v24, v20, v21",
                           "v_med3_i16 v26, v24, v25, s46", "v_min3_i16 v27, v25, v26, v24")
# Twelve word sets covering all six strict (w0, w1, w2) orderings for both signed and unsigned comparisons plus all four
# tie shapes (a == b, b == c, a == c, a == b == c). Sets 0..5 exercise signed/unsigned splits and wrapping mad; sets
# 6..7 complete the missing strict permutations (signed c < a < b and unsigned b < c < a); sets 8..11 exercise ties at
# the signed/unsigned boundary. Every high half differs from its low half.
INT16_TERNARY_WORDS = (
    (0xdead7fff, 0x12348000, 0xbeef0001, 0x5a5a0002),
    (0x1111ffff, 0x22220001, 0x33330000, 0x4444ffff),
    (0xabcd0003, 0x3c1d0004, 0x77770005, 0x13570006),
    (0xf00f1234, 0x801f0567, 0x8000ffff, 0x9999fffe),
    (0x00008000, 0xffff7fff, 0x0000fffe, 0xc0de0001),
    (0xfffe0002, 0x0001ffff, 0x80000010, 0x7ffe8001),
    (0x13570010, 0x24687fff, 0x35798000, 0x468a0003),
    (0x24688001, 0x35790002, 0x468a7fff, 0x579b0004),
    (0xaa558000, 0x55aa8000, 0x11227fff, 0x3344fffe),
    (0x66770001, 0x7788ffff, 0x8899ffff, 0x99aa8000),
    (0xbabe7fff, 0xcafe8000, 0xdead7fff, 0xfeed7fff),
    (0x11118001, 0x22228001, 0x33338001, 0x44448001),
)


def int16_ternary_ops(group):
    """The group's ops on the constant words, widened with SDWA WORD_0 into v12..v15 and exported."""
    body = [f"v_mov_b32_e32 v{18 + k}, s{44 + k}" for k in range(4)]
    body += [f"v_mov_b32_e32 v{24 + k}, {word:#x}" for k, word in enumerate(INT16_VOP3_SENTINELS)]
    if group < 3:
        body += [f"v_{op} v{24 + k}, {a}, {b}, {c}" for k, (op, a, b, c) in enumerate(INT16_TERNARY_GROUPS[group])]
    elif group == 3:
        body += INT16_TERNARY_CHAIN
    else:
        body += INT16_TERNARY_MED_CHAIN
    body += [INT16_VOP3_WIDEN.format(dst=12 + k, src=24 + k) for k in range(4)]
    return uncompressed_export(body)


def int16_ternary_reference(group, words, *, mutation=None):
    """Four exported words from the semantics (a, b, c = low halves of the sources), without pixel_aot.

    mad is (a * b + c) mod 2^16; min3, max3 and med3 are the smallest, largest and middle of the three halves, signed
    for the _i16 forms. mutation names a deviation the draw's mutated shaders model: signed_as_unsigned (the _i16 min3,
    max3 and med3 compared as unsigned), med3_as_min3, mad_no_addend or source_high (the first source read from 31:16).
    """
    def signed(x):
        return x - 0x10000 if x & 0x8000 else x

    def compute(op, a, b, c):
        kind = op[-3:]
        if op.startswith("mad"):
            if mutation == "mad_no_addend":
                c = 0
            r = (signed(a) * signed(b) + signed(c)) if kind == "i16" else (a * b + c)
            return r & 0xffff
        key = signed if kind == "i16" and mutation != "signed_as_unsigned" else (lambda x: x)
        ordered = sorted((a, b, c), key=key)
        pick = {"min3": 0, "max3": 2, "med3": 0 if mutation == "med3_as_min3" else 1}[op[:4]]
        return ordered[pick] & 0xffff

    if group == 3:
        w0, w1, w2, w3 = (w & 0xffff for w in words)
        r24 = (w0 + w1) & 0xffff
        r25 = compute("mad_u16", r24, w2, w3)
        r26 = compute("med3_i16", r24, r25, w0)
        return (r24, r25, r26, compute("max3_u16", r25, r26, r24))
    if group == 4:
        w0, w1, w2, w3 = (w & 0xffff for w in words)
        r24 = (w0 - w1) & 0xffff
        r25 = compute("med3_u16", r24, w2, w3)
        r26 = compute("med3_i16", r24, r25, w2)
        return (r24, r25, r26, compute("min3_i16", r25, r26, r24))

    def operand(token):
        if token[0] == "v":
            return words[int(token[1:]) - 18]
        if token[0] == "s":
            return words[int(token[1:]) - 44]
        return int(token, 0) & 0xffffffff

    out = []
    for op, *tokens in INT16_TERNARY_GROUPS[group]:
        a, b, c = (operand(t) for t in tokens)
        if mutation == "source_high":
            a >>= 16
        out.append(compute(op, a & 0xffff, b & 0xffff, c & 0xffff))
    return tuple(out)


# Each mutation of the draw: the groups it is applied to and the lanes whose expected value it must change.
INT16_TERNARY_MUTATIONS = {
    "signed_as_unsigned": ((1,), (0, 2)),
    "signed_as_unsigned_med3": ((2,), (0, 2)),
    "signed_as_unsigned_med_chain": ((4,), (2, 3)),
    "med3_as_min3": ((2,), (0, 1, 2, 3)),
    "med3_as_min3_chain": ((4,), (1, 2)),
    "mad_no_addend": ((0,), (0, 1, 2, 3)),
    "source_high": ((0, 1), (0, 1, 2, 3)),
}


def check_int16_ternary_mutation_coverage():
    """Every mutation must change the affected lanes for some word set, so the draw can detect it."""
    for mutation, (groups, lanes) in INT16_TERNARY_MUTATIONS.items():
        model = ("signed_as_unsigned" if mutation.startswith("signed_as_unsigned")
                 else "med3_as_min3" if mutation.startswith("med3_as_min3") else mutation)
        for group in groups:
            for lane in lanes:
                assert any(int16_ternary_reference(group, words, mutation=model)[lane]
                           != int16_ternary_reference(group, words)[lane] for words in INT16_TERNARY_WORDS), \
                    (mutation, group, lane)
    # Lanes that a mutation does not touch stay equal, so a mismatch names the mutated lane.
    for group in (1, 2):
        for words in INT16_TERNARY_WORDS:
            assert (int16_ternary_reference(group, words, mutation="signed_as_unsigned")[1]
                    == int16_ternary_reference(group, words)[1])
    for words in INT16_TERNARY_WORDS:
        assert (int16_ternary_reference(4, words, mutation="signed_as_unsigned")[:2]
                == int16_ternary_reference(4, words)[:2])
    # All six strict (w0, w1, w2) orderings are exercised for both signed and unsigned comparisons, plus all four tie
    # shapes (a == b, b == c, a == c, and a == b == c).
    strict_perms = {(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)}
    signed_perms, unsigned_perms, tie_shapes = set(), set(), set()
    for words in INT16_TERNARY_WORDS:
        a, b, c = (w & 0xffff for w in words[:3])
        sa, sb, sc = (x - 0x10000 if x & 0x8000 else x for x in (a, b, c))
        if len({a, b, c}) == 3:
            unsigned_perms.add(tuple(sorted((a, b, c)).index(x) for x in (a, b, c)))
            signed_perms.add(tuple(sorted((sa, sb, sc)).index(x) for x in (sa, sb, sc)))
        else:
            tie_shapes.add((a == b, b == c, a == c))
    assert signed_perms == strict_perms, signed_perms
    assert unsigned_perms == strict_perms, unsigned_perms
    assert tie_shapes == {(True, False, False), (False, True, False), (False, False, True), (True, True, True)}, \
        tie_shapes
    # mad_i16 and mad_u16 give the same low half, so a signed/unsigned mad mutation would be unobservable by design.
    for words in INT16_TERNARY_WORDS:
        a, b, c = (w & 0xffff for w in words[:3])
        sa, sb, sc = (x - 0x10000 if x & 0x8000 else x for x in (a, b, c))
        assert (a * b + c) & 0xffff == (sa * sb + sc) & 0xffff
    assert len(INT16_TERNARY_WORDS) >= 12 and all(len(words) == 4 for words in INT16_TERNARY_WORDS)


def check_int16_ternary_controls():
    """Lowering, spelling and refusals of the 8 three-operand 16-bit integer VOP3 forms and their low16 results."""
    check_int16_ternary_mutation_coverage()
    assert {op for group in INT16_TERNARY_GROUPS for op, *_ in group} == set(pixel_aot.VOP3_INT16_TERNARY)
    assert not set(pixel_aot.VOP3_INT16) & set(pixel_aot.VOP3_INT16_TERNARY)
    for group in range(5):
        lowered = translate_int(int16_ternary_ops(group))
        body = lowered.split("uint4 main", 1)[1]
        assert "0xffff0000u" not in body and "asfloat(0u | " not in body, group
        assert body.count("uint i16s") == (12 if group < 3 else 11), group
    prefix = ["v_mov_b32_e32 v18, s44", "v_mov_b32_e32 v19, s45", "v_mov_b32_e32 v20, s46"]

    def lowered_op(line):
        return translate_int(uncompressed_export([*prefix, line]))

    assert lowered_op("v_mad_u16_e64 v24, v18, v19, v20") == lowered_op("v_mad_u16 v24, v18, v19, v20")
    # Negative controls for the lowering itself: signed against unsigned, min3 against max3, med3 against min3.
    for kind in ("min3", "max3", "med3"):
        assert (lowered_op(f"v_{kind}_i16 v24, v18, v19, v20") != lowered_op(f"v_{kind}_u16 v24, v18, v19, v20")), kind
    assert lowered_op("v_min3_u16 v24, v18, v19, v20") != lowered_op("v_max3_u16 v24, v18, v19, v20")
    assert lowered_op("v_med3_u16 v24, v18, v19, v20") != lowered_op("v_min3_u16 v24, v18, v19, v20")
    # An inline constant, a 16-bit literal and a scalar register are accepted in every source slot.
    for line in ("v_max3_i16 v24, v18, v19, 4", "v_max3_i16 v24, 4, v19, v20", "v_med3_u16 v24, v18, -16, 64",
                 "v_mad_u16 v24, v18, 0xffff, v20", "v_mad_i16 v24, s44, v19, s46", "v_mad_u16 v24, v18, 0x0012, v20"):
        lowered_op(line)

    def refuse(line, message):
        expect_value_error(translate_int, uncompressed_export([*prefix, line]), message)

    for op in sorted(pixel_aot.VOP3_INT16_TERNARY):
        refuse(f"v_{op} v24, v18, v19, v20 clamp", "Unsupported instruction")
        refuse(f"v_{op} v24, v18, v19, v20 op_sel:[0,0,0,1]", "Unsupported instruction")
        refuse(f"v_{op} v24, v18, v19, v20 op_sel:[1,0,0,0]", "Unsupported instruction")
        refuse(f"v_{op} v24, -v18, v19, v20", "Unsupported 16-bit integer operand")
        refuse(f"v_{op} v24, v18, |v19|, v20", "Unsupported 16-bit integer operand")
        refuse(f"v_{op} v24, v18, v19, 1.0", "Unsupported 16-bit integer operand")
        refuse(f"v_{op} v24, 0x10000, v19, v20", "16-bit literal out of range")
        refuse(f"v_{op} v24, v18, v19, 65", "Unsupported 16-bit integer inline constant")
        refuse(f"v_{op} v24, v18, -17, v20", "Unsupported 16-bit integer inline constant")
        for slot in range(3):
            tokens = ["v18", "v19", "v20"]
            tokens[slot] = "v200"
            refuse(f"v_{op} v24, {', '.join(tokens)}", "Untracked 16-bit source")
        refuse(f"v_{op} v24, v18, v19", f"Wrong 16-bit integer operand count for v_{op}")
        refuse(f"v_{op} v256, v18, v19, v20", "Register out of range")
        refuse(f"v_{op} v24, v18, v19, s106", "Register out of range")
        refuse(f"v_{op}_dpp v24, v18, v19, v20 quad_perm:[0,1,2,3]", "Unsupported instruction")
        refuse(f"v_{op}_sdwa v24, v18, v19, v20 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_0 "
               "src1_sel:WORD_0 src2_sel:WORD_0", "Unsupported instruction")
    # The two-operand forms keep their operand count; the neighbours that stay unsupported are still refused.
    refuse("v_add_nc_u16 v24, v18, v19, v20", "Wrong 16-bit integer operand count for v_add_nc_u16")
    for line in ("v_mad_u32_u16 v24, v18, v19, v20", "v_mad_i32_i16 v24, v18, v19, v20",
                 "v_sad_u16 v24, v18, v19, v20", "v_med3_f16 v24, v18, v19, v20", "v_mad_f16 v24, v18, v19, v20"):
        refuse(line, "Unsupported instruction")
    sample = next(i for i, line in enumerate(with_constants([])) if line.startswith("image_sample"))
    overlap = with_constants([])
    overlap.insert(sample + 1, "v_mad_u16 v12, 1, 2, 3")
    expect_value_error(translate, overlap, "Write overlaps an outstanding sample")
    ok = with_constants([])
    ok.insert(sample + 1, "v_mad_u16 v20, 1, 2, 3")
    translate(ok)

    # Low16 sources: a 12-op result feeds a ternary op, a ternary result feeds a 12-op op, a 16-bit compare and the
    # SDWA widenings that stay inside the low half.
    low = [*prefix, "v_mov_b32_e32 v21, s47", "v_add_nc_u16 v24, v18, v19"]
    add_index = with_constants(low).index("v_add_nc_u16 v24, v18, v19")
    chain = translate_int(uncompressed_export([*low, "v_mad_u16 v25, v24, v20, v21"]))
    assert f"uint i16s{add_index + 1}_0 = asuint(alu{add_index});" in chain, chain
    translate_int(uncompressed_export([*low, "v_med3_i16 v25, v18, v24, v19"]))
    med_u16_chain = translate_int(uncompressed_export([*low, "v_med3_u16 v25, v24, v20, v21"]))
    assert f"uint i16s{add_index + 1}_0 = asuint(alu{add_index});" in med_u16_chain, med_u16_chain
    mad = [*prefix, "v_mad_u16 v24, v18, v19, v20"]
    mad_index = with_constants(mad).index("v_mad_u16 v24, v18, v19, v20")
    back = translate_int(uncompressed_export([*mad, "v_add_nc_u16 v25, v24, v18"]))
    assert f"uint i16s{mad_index + 1}_0 = asuint(alu{mad_index});" in back, back
    compare = translate_int(uncompressed_export([*mad, "v_cmp_lt_u16 vcc_lo, v24, v18"]))
    assert f"bool cmp{mad_index + 1} = (asuint(alu{mad_index}) & 0xffffu) <" in compare, compare
    for selection in ("WORD_0", "BYTE_0", "BYTE_1"):
        translate_int(uncompressed_export([*mad, "v_mov_b32_sdwa v25, v24 dst_sel:DWORD "
                                           f"dst_unused:UNUSED_PAD src0_sel:{selection}"]))
    # Full-word readers stay refused for the results of mad and both signed/unsigned med3.
    for producer in ("v_mad_u16 v24, v18, v19, v20", "v_med3_i16 v24, v18, v19, v20",
                     "v_med3_u16 v24, v18, v19, v20"):
        expect_value_error(translate_int, uncompressed_export([*prefix, producer, "v_add_nc_u32 v25, v24, v18"]),
                           "Expected a numeric register")
        expect_value_error(translate_int, uncompressed_export(
            [*prefix, producer, "v_mov_b32_sdwa v25, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1"]),
            "Reads the undefined high half of a 16-bit result")


def check_int16_vop3_controls():
    """Lowering, spelling and refusals of the 12 non-packed 16-bit integer VOP3 ops and their low16 results."""
    check_int16_vop3_mutation_coverage()
    assert {op for group in INT16_VOP3_GROUPS for op, _, _ in group} == set(pixel_aot.VOP3_INT16)
    for group in range(4):
        lowered = translate_int(int16_vop3_ops(group))
        body = lowered.split("uint4 main", 1)[1]
        # No old-value merge: the destination's high half is neither kept nor zeroed explicitly.
        assert "0xffff0000u" not in body and "asfloat(0u | " not in body, group
        assert body.count("uint i16s") == (8 if group < 3 else 4), group
    prefix = ["v_mov_b32_e32 v18, s44", "v_mov_b32_e32 v19, s45"]
    # An untracked destination and a sentinel destination lower alike; the destination value is never read.
    fresh = translate_int(uncompressed_export([*prefix, "v_add_nc_u16 v30, v18, v19"])).split("uint4 main", 1)[1]
    assert "0xffff0000u" not in fresh and "precise float alu" in fresh, fresh
    # The _e64 spelling lowers to the same shader as the unsuffixed one.
    plain = translate_int(uncompressed_export([*prefix, "v_add_nc_u16 v24, v18, v19"]))
    assert translate_int(uncompressed_export([*prefix, "v_add_nc_u16_e64 v24, v18, v19"])) == plain

    def refuse(line, message):
        expect_value_error(translate_int, uncompressed_export([*prefix, line]), message)

    for line in ("v_add_nc_u16 v24, v18, v19 op_sel:[0,0,1]", "v_add_nc_u16 v24, v18, v19 clamp",
                 "v_add_nc_u16_dpp v24, v18, v19 quad_perm:[0,1,2,3]",
                 "v_add_nc_u16_sdwa v24, v18, v19 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0",
                 "v_min_f16_e64 v24, v18, v19"):
        refuse(line, "Unsupported instruction")
    refuse("v_add_nc_u16 v24, v18, -v19", "Unsupported 16-bit integer operand")
    refuse("v_add_nc_u16 v24, v18, 1.0", "Unsupported 16-bit integer operand")
    refuse("v_add_nc_u16 v24, v18, 0x10000", "16-bit literal out of range")
    refuse("v_add_nc_u16 v24, v18, 65", "Unsupported 16-bit integer inline constant")
    refuse("v_add_nc_u16 v24, v18, -17", "Unsupported 16-bit integer inline constant")
    refuse("v_add_nc_u16 v24, v18, v200", "Untracked 16-bit source")
    refuse("v_add_nc_u16 v24, v200, v19", "Untracked 16-bit source")
    refuse("v_add_nc_u16 v256, v18, v19", "Register out of range")
    refuse("v_add_nc_u16 v24, v256, v19", "Register out of range")
    refuse("v_add_nc_u16 v24, s106, v19", "Register out of range")
    # A destination with an outstanding sample is refused; the same op on a free register is accepted.
    sample = next(i for i, line in enumerate(with_constants([])) if line.startswith("image_sample"))
    overlap = with_constants([])
    overlap.insert(sample + 1, "v_add_nc_u16 v12, 1, 2")
    expect_value_error(translate, overlap, "Write overlaps an outstanding sample")
    ok = with_constants([])
    ok.insert(sample + 1, "v_add_nc_u16 v20, 1, 2")
    translate(ok)
    # The helper rejects a key it does not know instead of falling back to ashrrev.
    try:
        pixel_aot.pk_int16_lane_expr("bogus", ["a", "b"], [0, 0], False)
    except ValueError as error:
        assert "Unknown 16-bit integer op" in str(error), error
    else:
        raise AssertionError("pk_int16_lane_expr accepted an unknown op")

    # Low16 results. The readers that see only bits 15:0 accept them: a chain of two ops, a 16-bit compare and a
    # v_mov_b32_sdwa that selects WORD_0, BYTE_0 or BYTE_1 (also sign-extending).
    low = [*prefix, "v_mov_b32_e32 v20, s46", "v_mov_b32_e32 v21, s47", "v_add_nc_u16 v24, v18, v19"]
    add_index = with_constants(low).index("v_add_nc_u16 v24, v18, v19")
    chain = translate_int(uncompressed_export([*low, "v_mul_lo_u16 v25, v24, v20"]))
    assert f"uint i16s{add_index + 1}_0 = asuint(alu{add_index});" in chain, chain
    compare = translate_int(uncompressed_export([*low, "v_cmp_lt_u16 vcc_lo, v24, v21"]))
    assert f"bool cmp{add_index + 1} = (asuint(alu{add_index}) & 0xffffu) <" in compare, compare
    translate_int(uncompressed_export([*low, "v_cmp_lt_i16 vcc_lo, v24, v21"]))
    for selection in ("WORD_0", "BYTE_0", "BYTE_1"):
        for source in ("v24", "sext(v24)"):
            translate_int(uncompressed_export([*low, f"v_mov_b32_sdwa v25, {source} dst_sel:DWORD "
                                                f"dst_unused:UNUSED_PAD src0_sel:{selection}"]))
    # A padded write into a low16 destination (any dst_sel) is a defined word and is accepted.
    translate_int(uncompressed_export([*low, "v_mov_b32_sdwa v24, v18 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD"]))

    def refuse_low(line, message, *, extra=(), function=translate_int):
        expect_value_error(function, uncompressed_export([*low, *extra, line]), message)

    numeric = "Expected a numeric register"
    undefined = "Reads the undefined high half of a 16-bit result"
    # Whole-word readers: integer (int_operand), float (fp_operand), move, compare data, packed and conversions.
    for line in ("v_add_nc_u32 v25, v24, v18", "v_xor_b32 v25, v24, v18", "v_add_nc_u32 v25, v18, v24",
                 "v_mul_f32 v25, v24, v18", "v_mov_b32_e32 v25, v24", "v_cvt_f32_u32_e32 v25, v24",
                 "v_cvt_f32_f16_e32 v25, v24", "v_cvt_off_f32_i4 v25, v24", "v_cvt_f32_ubyte0 v25, v24",
                 "v_cmp_lt_u32 vcc_lo, v24, v18", "v_cmp_lt_f32 vcc_lo, v24, v18",
                 "v_pk_add_u16 v25, v24, v18", "v_pk_add_f16 v25, v24, v18",
                 "v_cmp_class_f16 vcc_lo, v24, v18", "v_cmp_class_f16 vcc_lo, v18, v24",
                 "v_dot2_f32_f16 v25, v18, v19, v24"):
        refuse_low(line, numeric)
    refuse_low("v_cndmask_b32 v25, v24, v18, vcc_lo", numeric, extra=["v_cmp_lt_u32 vcc_lo, v18, v19"])
    refuse_low("v_cndmask_b32 v25, v18, v24, vcc_lo", numeric, extra=["v_cmp_lt_u32 vcc_lo, v18, v19"])
    refuse_low("v_add_f32_dpp v25, v24, v18 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
               "Expected a numeric VGPR in DPP")
    refuse_low("v_cvt_pkrtz_f16_f32_e32 v25, v24, v18", "Unsupported pack operand")
    refuse_low("v_cvt_f16_f32 v24, v18", "Unsupported conversion destination")
    # Exports: the compressed form needs the float target, the uncompressed form works on either.
    for tail, function in (("exp mrt0 v24, v25, v26, v27 done vm", translate_int),
                           ("exp mrt0 v24, v24, off, off done compr vm", translate)):
        program = with_constants([*low])
        end = program.index("v_cvt_pkrtz_f16_f32_e32 v16, v12, v13")
        program[end:] = [tail, "s_endpgm"]
        expect_value_error(function, program, "Unsupported export")
    # An image sample coordinate: v8 holds a low16 result when the sample reads it.
    coordinate = with_constants([])
    coordinate.insert(next(i for i, line in enumerate(coordinate) if line.startswith("image_sample")),
                      "v_add_nc_u16 v8, 1, 2")
    expect_value_error(translate, coordinate, "Unsupported texture coordinate")
    # v_mov_b32_sdwa: every selection that reaches the undefined bits 31:16 is refused, as is merging with a low16
    # destination.
    for selection in ("WORD_1", "BYTE_2", "BYTE_3", "DWORD"):
        for source in ("v24", "sext(v24)"):
            refuse_low(f"v_mov_b32_sdwa v25, {source} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:{selection}",
                       undefined)
    refuse_low("v_mov_b32_sdwa v24, v18 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD", undefined)
    refuse_low("v_mov_b32_sdwa v24, v18 dst_sel:BYTE_1 dst_unused:UNUSED_PRESERVE src0_sel:BYTE_0", undefined)
    # The same selections on a whole-word source and a preserve on a tracked destination stay accepted.
    translate_int(uncompressed_export([*low, "v_mov_b32_e32 v26, s44",
                                  "v_mov_b32_sdwa v25, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1",
                                  "v_mov_b32_sdwa v26, v18 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD"]))


# v_swap_b32: s44..s47 move to v18..v21 through float VGPRs, so every word is a normal binary32 pattern (no
# subnormal, infinity or NaN) and the four words of a set are distinct.
SWAP_WORDS = (
    (0x3f800000, 0x40000000, 0x40400000, 0x40800000),
    (0xc2f6e979, 0x41200000, 0x3dcccccd, 0x7f000000),
    (0x00800000, 0x80800000, 0x3f000000, 0x12345678),
    (0x9abcdef0, 0x7f7fffff, 0xff7fffff, 0x0dedbeef),
)

SWAP_BODY = [
    "v_mov_b32_e32 v18, s44",
    "v_mov_b32_e32 v19, s45",
    "v_mov_b32_e32 v20, s46",
    "v_mov_b32_e32 v21, s47",
    "v_swap_b32 v18, v19",
    "v_swap_b32 v20, v21",
    "v_swap_b32 v18, v20",
    "v_swap_b32 v21, v21",
    "v_mov_b32_e32 v12, v18",
    "v_mov_b32_e32 v13, v19",
    "v_mov_b32_e32 v14, v20",
    "v_mov_b32_e32 v15, v21",
]


def swap_ops(body=None) -> list[str]:
    return uncompressed_export(list(SWAP_BODY if body is None else body))


def swap_ops_reference(words):
    """a b c d -> swap(1,2): b a c d -> swap(3,4): b a d c -> swap(1,3): d a b c -> self swap: d a b c."""
    a, b, c, d = words
    return (d, a, b, c)


def swap_emulate(body, words):
    """Independent model of the body's moves and swaps on Python integers; the other lines are not needed here."""
    regs = dict(zip(("s44", "s45", "s46", "s47"), words))
    for line in body:
        op, _, operands = line.partition(" ")
        args = operands.split(", ")
        if op == "v_swap_b32":
            regs[args[0]], regs[args[1]] = regs[args[1]], regs[args[0]]
        elif op == "v_mov_b32_e32":
            regs[args[0]] = regs[args[1]]
        else:
            raise AssertionError(line)
    return tuple(regs[f"v{n}"] for n in range(12, 16))


def swap_mutants():
    """Wrong lowerings of the same body: (label, mutated body)."""
    swaps = [i for i, line in enumerate(SWAP_BODY) if line.startswith("v_swap_b32")]
    one_way = list(SWAP_BODY)
    one_way[swaps[0]] = "v_mov_b32_e32 v18, v19"
    no_snapshot = list(SWAP_BODY)
    no_snapshot[swaps[0]:swaps[0] + 1] = ["v_mov_b32_e32 v18, v19", "v_mov_b32_e32 v19, v18"]
    reordered = list(SWAP_BODY)
    reordered[swaps[1]], reordered[swaps[2]] = reordered[swaps[2]], reordered[swaps[1]]
    no_self = list(SWAP_BODY)
    no_self[swaps[3]] = "v_mov_b32_e32 v21, v20"
    return (("one-way copy", one_way), ("no snapshot", no_snapshot), ("reordered chain", reordered),
            ("self swap as copy", no_self))


def scalar():
    return with_constants(scalar_body())


def scalar_reference(body, words):
    """Run body on Python integers; returns the four exported channels and every select's arm."""
    mask = 0xffffffff
    regs = dict(zip(("s44", "s45", "s46", "s47"), words))
    scc, arms, out = None, [], {}

    def signed(x):
        return x - (1 << 32) if x >> 31 else x

    def read(token):
        if token in regs:
            return regs[token]
        if token == "1.0":
            return 0x3f800000
        return int(token, 0) & mask

    for line in body:
        op, operands = line.split(" ", 1)
        args = operands.split(", ")
        if op.startswith("v_mov_b32"):
            out[args[0]] = regs[args[1]]
            continue
        x, *rest = map(read, args[1:] if not op.startswith("s_cmp") else args)
        y = rest[0] if rest else None
        if op.startswith("s_cmp_"):
            relation, kind = op[6:8], op[9]
            left, right = (signed(x), signed(y)) if kind == "i" else (x, y)
            scc = {"eq": left == right, "lg": left != right, "gt": left > right, "ge": left >= right,
                   "lt": left < right, "le": left <= right}[relation]
            continue
        name = op[2:]
        if name == "mov_b32":
            result = x
        elif name == "movk_i32":
            result = (x - 0x10000 if x & 0x8000 else x) & mask
        elif name == "cselect_b32":
            assert scc is not None
            arms.append(scc)
            result = x if scc else y
        elif name in ("add_u32", "sub_u32"):
            wide = x + y if name == "add_u32" else x - y
            result, scc = wide & mask, not 0 <= wide <= mask
        elif name in ("add_i32", "sub_i32"):
            wide = signed(x) + signed(y) if name == "add_i32" else signed(x) - signed(y)
            result, scc = wide & mask, not -2**31 <= wide < 2**31
        elif name == "mul_i32":
            result = (x * y) & mask
        elif name in ("min_u32", "max_u32", "min_i32", "max_i32"):
            key = signed if name.endswith("i32") else (lambda v: v)
            scc = key(x) < key(y) if name.startswith("min") else key(x) > key(y)
            result = x if scc else y
        elif name == "not_b32":
            result = ~x & mask
            scc = result != 0
        else:
            result = {"and_b32": lambda: x & y, "or_b32": lambda: x | y, "xor_b32": lambda: x ^ y,
                      "andn2_b32": lambda: x & ~y, "orn2_b32": lambda: x | ~y,
                      "nand_b32": lambda: ~(x & y), "nor_b32": lambda: ~(x | y), "xnor_b32": lambda: ~(x ^ y),
                      "lshl_b32": lambda: x << (y & 31), "lshr_b32": lambda: x >> (y & 31),
                      "ashr_i32": lambda: signed(x) >> (y & 31)}[name]() & mask
            scc = result != 0
        regs[args[0]] = result
    channels = []
    for n in range(12, 16):
        value = struct.unpack("<f", struct.pack("<I", out[f"v{n}"]))[0]
        assert 1 <= value < 2 and half_rtz(value) == value, value
        channels.append(value)
    return channels, arms


def scalar_case(directory, words, *, name="scalar.case"):
    """Constant words 3 to 6 as raw 32-bit patterns, then the uniform color expected at every texel."""
    case = bytearray(256)
    struct.pack_into("<4I", case, 12, *words)
    channels, _ = scalar_reference(scalar_body(), words)
    case.extend(struct.pack("<4f", *channels) * 64)
    path = directory / name
    path.write_bytes(case)
    return path


# Mutations a draw must detect: signedness of a shift and of a comparison, and the sources of a subtraction.
SCALAR_MUTATIONS = (("s_ashr_i32", "s_lshr_b32"), ("s_cmp_lt_i32", "s_cmp_lt_u32"),
                    ("s_sub_u32 s62, s46, s47", "s_sub_u32 s62, s47, s46"),
                    ("s_nand_b32 s62, s44, s45", "s_and_b32 s62, s44, s45"),
                    ("s_nor_b32 s62, s45, s46", "s_or_b32 s62, s45, s46"),
                    ("s_xnor_b32 s62, s46, s47", "s_xor_b32 s62, s46, s47"))


def scalar_detecting_words(old, new):
    """The first constant set whose reference changes under the mutation."""
    body = scalar_body()
    changed = [line.replace(old, new) for line in body]
    assert changed != body, old
    return next(words for words in SCALAR_WORDS
                if scalar_reference(changed, words)[0] != scalar_reference(body, words)[0])


def check_scalar_controls():
    body = scalar_body()
    program = scalar()
    source = translate(program)
    for expected in ("= asuint(params[0].w) + asuint(params[1].x);", " < asuint(params[0].w)) ? 1u : 0u;",
                     "= (asuint(params[1].y) < asuint(params[1].z)) ? 2u : 0u;",
                     "= ((~(asuint(params[1].y) ^ asuint(params[1].x)) & (asuint(params[1].y) ^ salu",
                     "= asuint(min(asint(asuint(params[0].w)), asint(asuint(params[1].x))));",
                     "= (asint(asuint(params[1].x)) > asint(asuint(params[0].w)));", "? 8192u : 0u;",
                     "= (asint(asuint(params[1].z)) == asint(4294967295u));", "? 2097152u : 0u;",
                     "= asuint(params[0].w) & ~asuint(params[1].x);", "= ~asuint(params[1].y);",
                     "= asuint(asint(asuint(params[1].y)) >> (asuint(params[0].w) & 31u));",
                     "= ~(asuint(params[0].w) & asuint(params[1].x));",
                     "= ~(asuint(params[1].x) | asuint(params[1].y));",
                     "= ~(asuint(params[1].y) ^ asuint(params[1].z));",
                     "= 4294967168u * 31u;", "? asuint(params[1].y) : asuint(params[1].z);",
                     " | 1065353216u;", "= asfloat(salu"):
        assert expected in source, expected
    # Both arms of every flag select are drawn across the two constant sets.
    flags = [scalar_reference(body, words)[1] for words in SCALAR_WORDS]
    assert len(flags[0]) == len(SCALAR_STEPS) + 1
    assert all(a != b for a, b in zip(flags[0][:-1], flags[1][:-1])), flags
    for old, new in SCALAR_MUTATIONS:
        scalar_detecting_words(old, new)
    # SCC exists only after a tracked write; the exec-mask operations overwrite it.
    base = assembly(False)
    wqm = base.index("s_wqm_b64 exec, exec")
    andexec = base.index("s_and_b64 exec, exec, s[50:51]")
    translate(base[:andexec] + ["s_cmp_eq_u32 1, 1", "s_cselect_b32 s62, 1, 0"] + base[andexec:])
    must_reject(translate, base[:andexec] + ["s_cselect_b32 s62, 1, 0"] + base[andexec:])
    must_reject(translate, base[:andexec] + ["s_cmp_eq_u32 1, 1"] + base[andexec:andexec + 1]
                + ["s_cselect_b32 s62, 1, 0"] + base[andexec + 1:])
    must_reject(translate, base[:wqm] + ["s_cmp_eq_u32 1, 1"] + base[wqm:wqm + 1]
                + ["s_cselect_b32 s62, 1, 0"] + base[wqm + 1:])
    # A scalar result is not a known constant, so it cannot supply a sampler word or address.
    for old, new in (("s_mov_b32 s40, 1", "s_add_u32 s40, 0, 1"), ("s_mov_b32 s21, 0", "s_and_b32 s21, 0, 0")):
        must_reject(translate, [line.replace(old, new) for line in base])
    # Neighbouring forms that the lowerer does not implement, operand kinds it cannot read, and
    # destinations other than plain scalar registers.
    first = "s_add_u32 s62, s44, s45"
    for subtract in (False, True):
        must_reject(translate_int, scalar_carry_chain(subtract, initialize_scc=False))
    for new in ("s_addc_u32 s62, s44, s45", "s_subb_u32 s62, s44, s45",
                "s_nand_b32 s62, v12, s45", "s_nor_b32 v62, s44, s45", "s_xnor_b32 s62, s44",
                "s_bfe_u32 s62, v12, s45", "s_lshl5_add_u32 s62, s44, s45",
                "s_bitreplicate_b64_b32 s[62:63], v12", "s_bitset0_b64 s[62:63], s44",
                "s_quadmask_b32 s62, v12", "s_bitset0_b64 s[62:63], v12",
                "s_brev_b32 s62, v12", "s_and_b64 s[62:63], s[44:46], s[46:47]",
                "s_add_u32 s62, s[44:45], s45", "s_add_u32 s62, s44", "s_add_u32 s62, s44, s45, s46",
                "s_add_u32 s62, -s44, s45", "s_add_u32 s62, s28, s45", "s_add_u32 s62, s20, s45",
                "s_add_u32 s62, s90, s45", "s_add_u32 s62, vcc_lo, s45", "s_add_u32 s62, exec_lo, s45",
                "s_add_u32 s62, v12, s45", "s_add_u32 m0, s44, s45", "s_add_u32 vcc_lo, s44, s45",
                "s_add_u32 s62, s44, |s45|",
                "s_cmp_eq_u64 s[44:46], s[46:47]", "s_bitcmp1_b32 v12, s45",
                "s_cmpk_eq_u32 s44, 0x10000", "s_cmpk_eq_u32 s44, -1", "s_cmpk_eq_f32 s44, 0x10",
                "s_cmp_lt_i32 s44", "s_cmp_lt_f32 s44, s45"):
        changed = [line.replace(first, new) for line in program]
        assert changed != program, new
        must_reject(translate, changed)
    flag = program.index(first) + 1
    for new in ("s_cselect_b64 s[62:63], s44, s45", "s_cmovk_i32 s63, 0x10000", "s_cmovk_i32 vcc_lo, 0x1",
                "s_cselect_b32 s63, s28, 0", "s_cselect_b32 vcc_lo, 1, 0"):
        must_reject(translate, program[:flag] + [new] + program[flag + 1:])
    # Before lgkmcnt(0) the constant words are still loading: neither read nor overwrite them.
    load = program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc") + 1
    for line in ("s_add_u32 s62, s44, 1", "s_add_u32 s45, 1, 1", "s_cmp_eq_u32 s46, 0",
                 "s_cselect_b32 s47, 1, 0"):
        must_reject(translate, program[:load] + ["s_cmp_eq_u32 1, 1", line] + program[load:])
    translate(program[:load] + ["s_cmp_eq_u32 1, 1", "s_cselect_b32 s62, 1, 0"] + program[load:])

    sop1_program = sop1_scalar_ops()
    sop1_source = translate_int(sop1_program)
    for expected in ("(asint(params[0].w) < 0 ? (0u - params[0].w) : params[0].w)",
                     "& ~(1u << (", "| (1u << (31u & 31u))", "? 305419896u :"):
        assert expected in sop1_source, expected
    sop1_refs = [sop1_scalar_ops_reference(words) for words in SOP1_WORDS]
    assert any(r[0] == 0 for r in sop1_refs) and any(r[0] == 0x80000000 for r in sop1_refs)
    assert any(r[3] == 0x12400009 for r in sop1_refs) and any(r[3] == 0x12345678 for r in sop1_refs)
    must_reject(translate, base[:andexec] + ["s_mov_b32 s51, 0", "s_cmov_b32 s51, s40"] + base[andexec:])
    must_reject(translate, base[:andexec] + ["s_cmp_eq_u32 1, 1"] + base[andexec:andexec + 1]
                + ["s_mov_b32 s51, 0", "s_cmov_b32 s51, s40"] + base[andexec + 1:])
    must_reject(translate, base[:wqm] + ["s_cmp_eq_u32 1, 1"] + base[wqm:wqm + 1]
                + ["s_mov_b32 s51, 0", "s_cmov_b32 s51, s40"] + base[wqm + 1:])
    for uninit in ("s_cmov_b32 s60, s44", "s_bitset0_b32 s60, 1", "s_bitset1_b32 s60, 1"):
        must_reject(translate, base[:andexec] + ["s_cmp_eq_u32 1, 1", uninit] + base[andexec:])
    sop1_wait = sop1_program.index("s_waitcnt lgkmcnt(0)", sop1_program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc"))
    for bad in ("s_abs_i32 s50, s44", "s_cmov_b32 s44, 1", "s_bitset0_b32 s44, 1", "s_bitset1_b32 s44, 1"):
        must_reject(translate_int, sop1_program[:sop1_wait] + ["s_cmp_eq_u32 1, 1", bad] + sop1_program[sop1_wait:])
    for old, new in (("s_abs_i32 s50, s44", "s_abs_i32 v12, s44"),
                     ("s_abs_i32 s50, s44", "s_abs_i32 s106, s44"),
                     ("s_cmov_b32 s51, s45", "s_cmov_b32 vcc_lo, s45"),
                     ("s_cmov_b32 s51, s45", "s_cmov_b32 s106, s45"),
                     ("s_bitset0_b32 s52, s47", "s_bitset0_b32 v12, 1"),
                     ("s_bitset1_b32 s52, 31", "s_bitset1_b32 s106, 1"),
                     ("s_bitset0_b32 s52, s47", "s_bitset0_b64 s[52:53], 1"),
                     ("s_bitset1_b32 s52, 31", "s_bitset1_b64 s[52:53], 1"),
                     ("s_bitset0_b32 s52, s47", "s_bitset0_b32 s52, v12")):
        changed = [line.replace(old, new) for line in sop1_program]
        assert changed != sop1_program, new
        must_reject(translate_int, changed)

    # s_bfm_b32: the lowering masks both counts to five bits and leaves SCC untouched.
    bfm_program = bfm_scalar_ops()
    bfm_source = translate_int(bfm_program)
    assert ("= ((1u << (params[0].w & 31u)) - 1u) << (params[1].x & 31u);" in bfm_source), bfm_source
    assert "s_bfm_b64(" in bfm_source
    refs = [bfm_scalar_ops_reference(words) for words in BFM_WORDS]
    # Counts 0, 1, 31, 32 and 33 for b32 and 63 and 64 for b64 are all covered with distinct results.
    assert [r[0] for r in refs] == [0, 0xfffffffe, 0, 0xfffffffe, 0]
    assert [r[1] for r in refs] == [0x80000000, 0, 0x80000000, 0, 8]
    assert refs[1][2:] == (0xfffffffe, 0) and refs[0][2:] == (0, 0)
    assert refs[2][2:] == (0x80000000, 0x7fffffff) and refs[3][2:] == (0, 0xfffffffe) and refs[4][2:] == (0, 0)
    with_scc = ["s_cmp_eq_u32 s44, s44"]
    scc_program = bfm_program[:]
    at = scc_program.index("v_mov_b32_e32 v12, s50")
    scc_program[at:at] = with_scc + ["s_bfm_b32 s54, 4, 8", "s_cselect_b32 s55, s44, s45"]
    scc_source = translate_int(scc_program)
    assert re.search(r"= scc\d+ \? params\[0\]\.w : params\[1\]\.x;", scc_source), scc_source
    # Without a tracked SCC the select is rejected even after s_bfm_b32 (it does not create one).
    at = bfm_program.index("v_mov_b32_e32 v12, s50")
    must_reject(translate_int, bfm_program[:at] + ["s_bfm_b32 s54, 4, 8", "s_cselect_b32 s55, s44, s45"] + bfm_program[at:])
    wait = bfm_program.index("s_waitcnt lgkmcnt(0)", bfm_program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc"))
    # Before the s_waitcnt, s44..s47 are still pending scalar loads: reading them, or overwriting s44, must be refused.
    for bad in ("s_bfm_b32 s50, s44, 1", "s_bfm_b32 s44, 1, 1", "s_bfm_b32 s50, 1, s47"):
        must_reject(translate_int, bfm_program[:wait] + [bad] + bfm_program[wait:])
    for old, new in (("s_bfm_b32 s50, s44, s45", "s_bfm_b32 v12, s44, s45"),
                     ("s_bfm_b32 s50, s44, s45", "s_bfm_b32 s50, v12, s45"),
                     ("s_bfm_b32 s50, s44, s45", "s_bfm_b32 s50, s44, v12"),
                     ("s_bfm_b32 s50, s44, s45", "s_bfm_b32 s50, s44"),
                     ("s_bfm_b32 s50, s44, s45", "s_bfm_b32 s50, s44, s45, s46"),
                     ("s_bfm_b32 s50, s44, s45", "s_bfm_b32 s106, s44, s45"),
                     ("s_bfm_b32 s50, s44, s45", "s_bfm_b32 s[50:51], s44, s45"),
                     ("s_bfm_b32 s50, s44, s45", "s_bfm_b32 s50, s[44:45], s45"),
                     ("s_bfm_b32 s50, s44, s45", "s_bfm_b32 s50, s90, s45")):
        changed = [line.replace(old, new) for line in bfm_program]
        assert changed != bfm_program, new
        must_reject(translate_int, changed)

    # s_brev_b32: reverses all 32 bits and leaves SCC untouched, so the select still reads the compare's SCC.
    brev_program = brev_scalar_ops()
    brev_source = translate_int(brev_program)
    assert re.search(r"uint salu\d+ = reversebits\(params\[1\]\.y\);", brev_source), brev_source
    assert re.search(r"uint salu\d+ = reversebits\(params\[1\]\.z\);", brev_source), brev_source
    assert brev_source.count("reversebits(") == 3, brev_source
    assert len(re.findall(r"\bscc\d+ =", brev_source)) == 1, brev_source  # only the compare writes SCC
    assert re.search(r"= scc\d+ \? params\[1\]\.z : 0u;", brev_source), brev_source
    brev_refs = [brev_scalar_ops_reference(words) for words in BREV_WORDS]
    assert brev_refs[0] == (0, 0x1e6a2c48, 0xffffffff, 0xffffffff), brev_refs[0]
    assert brev_refs[1] == (0xffffffff, 0x1e6a2c48, 0, 1), brev_refs[1]
    assert brev_refs[2] == (1, 0x1e6a2c48, 1, 0x80000000), brev_refs[2]
    assert brev_refs[3][3] == 0x55555555 and brev_refs[4][0] == 0x55555555, brev_refs
    assert brev_refs[4][3] == 0x1e6a2c48 and brev_refs[5][0] == 0x1e6a2c48, brev_refs
    # Both SCC outcomes appear, so a brev that wrote SCC would change the select arm for some word set.
    assert {words[0] == words[1] for words in BREV_WORDS} == {True, False}
    # Without a tracked SCC the select is rejected even after s_brev_b32 (it does not create one).
    brev_at = brev_program.index("s_cmp_eq_u32 s44, s45")
    must_reject(translate_int, brev_program[:brev_at] + brev_program[brev_at + 1:])
    brev_wait = brev_program.index("s_waitcnt lgkmcnt(0)", brev_program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc"))
    # Before the s_waitcnt, s44..s47 are still pending scalar loads: reading or overwriting them must be refused.
    for bad in ("s_brev_b32 s50, s46", "s_brev_b32 s47, 1", "s_brev_b32 s50, s44"):
        must_reject(translate_int, brev_program[:brev_wait] + [bad] + brev_program[brev_wait:])
    for old, new in (("s_brev_b32 s50, s46", "s_brev_b32 v12, s46"),
                     ("s_brev_b32 s50, s46", "s_brev_b32 s50, v12"),
                     ("s_brev_b32 s50, s46", "s_brev_b32 s50"),
                     ("s_brev_b32 s50, s46", "s_brev_b32 s50, s46, s47"),
                     ("s_brev_b32 s50, s46", "s_brev_b32 s106, s46"),
                     ("s_brev_b32 s50, s46", "s_brev_b32 s[50:51], s46"),
                     ("s_brev_b32 s50, s46", "s_brev_b32 s50, s[46:47]"),
                     ("s_brev_b32 s50, s46", "s_brev_b32 s50, s90")):
        changed = [line.replace(old, new) for line in brev_program]
        assert changed != brev_program, new
        must_reject(translate_int, changed)

    # Plain bfrev/ffbh/ffbl/xnor: both fixtures lower; each ffbh helper is defined exactly once, and only when used.
    bitscan_first = translate_int(plain_bitscan_ops())
    bitscan_second = translate_int(plain_bitscan_ops(True))
    assert "reversebits(" in bitscan_first and "firstbitlow(" in bitscan_second
    assert "~(params[1].z ^ params[0].w)" in bitscan_first, bitscan_first
    assert bitscan_first.count("uint dpp_ffbh_u32(") == 1 and bitscan_first.count("uint dpp_ffbh_i32(") == 1
    assert bitscan_second.count("uint dpp_ffbh_u32(") == 1 and "uint dpp_ffbh_i32(" not in bitscan_second
    assert "uint dpp_ffbh_" not in translate_int(uncompressed_export(["v_bfrev_b32 v24, s44"]))
    assert "reversebits(" in translate_int(plain_bitscan_ops(vgpr_source=True))  # a VGPR source lowers too
    bitscan_program = plain_bitscan_ops()
    for old, new, message in (("v_bfrev_b32 v24, s44", "v_bfrev_b32 v24, s44 clamp", "Unsupported 32-bit scalar operand"),
                              ("v_xnor_b32 v27, s47, s44", "v_xnor_b32 v27, s47", "Invalid integer operation"),
                              ("v_xnor_b32 v27, s47, s44", "v_xnor_b32 v27, s47, s44, s45", "Invalid integer operation"),
                              ("v_bfrev_b32 v24, s44", "v_bfrev_b32 v24, s44, s45", "Invalid integer operation"),
                              ("v_bfrev_b32 v24, s44", "v_bfrev_b32 v256, s44", "Invalid integer operation")):
        changed = [line.replace(old, new) for line in bitscan_program]
        assert changed != bitscan_program, new
        expect_value_error(translate_int, changed, message)



    # v_swap_b32: the independent model matches the reference, and every wrong lowering changes the result.
    swap_program = swap_ops()
    swap_source = translate_int(swap_program)
    assert "v_swap_b32 v18, v19" in swap_program
    for words in SWAP_WORDS:
        assert len(set(words)) == 4 and all(0 < (w >> 23 & 0xff) < 0xff for w in words), words
        assert swap_emulate(SWAP_BODY, words) == swap_ops_reference(words) != words
    for label, body in swap_mutants():
        assert body != SWAP_BODY, label
        assert any(swap_emulate(body, words) != swap_ops_reference(words) for words in SWAP_WORDS), label
        assert translate_int(swap_ops(body)) != swap_source, label
    # A self swap is accepted and is a no-op: the program is the same as with an s_nop in its place.
    nop_body = [("s_nop 0" if line == "v_swap_b32 v21, v21" else line) for line in SWAP_BODY]
    assert nop_body != SWAP_BODY
    assert translate_int(swap_ops(nop_body)) == swap_source
    for old, new, message in (
            ("v_swap_b32 v18, v19", "v_swap_b32 v18, s44", "Unsupported instruction"),  # SGPR source
            ("v_swap_b32 v18, v19", "v_swap_b32 s44, v19", "Unsupported instruction"),  # SGPR destination
            ("v_swap_b32 v18, v19", "v_swap_b32 v18, 1", "Unsupported instruction"),  # inline constant
            ("v_swap_b32 v18, v19", "v_swap_b32_e64 v18, v19", "Unsupported instruction"),
            ("v_swap_b32 v18, v19", "v_swap_b32_dpp v18, v19 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
             "Unsupported instruction"),
            ("v_swap_b32 v18, v19", "v_swap_b32_sdwa v18, v19 dst_sel:DWORD dst_unused:UNUSED_PAD "
             "src0_sel:DWORD src1_sel:DWORD", "Unsupported instruction"),
            ("v_swap_b32 v18, v19", "v_swap_b32 v18, -v19", "Unsupported instruction"),
            ("v_swap_b32 v18, v19", "v_swap_b32 v18, v19 clamp", "Unsupported instruction"),
            ("v_swap_b32 v18, v19", "v_swap_b32 v18", "Unsupported instruction"),
            ("v_swap_b32 v18, v19", "v_swap_b32 v18, v19, v20", "Unsupported instruction"),
            ("v_swap_b32 v18, v19", "v_swap_b32 v256, v19", "Invalid register swap"),
            ("v_swap_b32 v18, v19", "v_swap_b32 v18, v256", "Invalid register swap"),
            ("v_swap_b32 v18, v19", "v_swap_b32 v18, v40", "Expected a numeric register"),  # never written
            ("v_swap_b32 v18, v19", "v_swap_b32 v40, v19", "Expected a numeric register"),
            ("v_swap_b32 v18, v19", "v_swap_b32 v0, v19", "Expected a numeric register"),  # barycentric, not numeric
            ("v_swap_b32 v21, v21", "v_swap_b32 v40, v40", "Expected a numeric register")):
        changed = [line.replace(old, new) for line in SWAP_BODY]
        assert changed != SWAP_BODY, new
        expect_value_error(translate_int, swap_ops(changed), message)
    # The sample in v12..v15 is outstanding before its vmcnt wait: a swap touching it must be refused.
    wait = swap_program.index("s_waitcnt vmcnt(0)")
    for bad in ("v_swap_b32 v12, v13", "v_swap_b32 v13, v12"):
        expect_value_error(translate_int, swap_program[:wait] + [bad] + swap_program[wait:],
                           "Swap overlaps an outstanding sample")
    # Once the wait has completed it, the same swap is accepted.
    translate_int(swap_ops(["v_swap_b32 v12, v13"] + SWAP_BODY))

    quadmask_source = translate_int(quadmask_bitreplicate_program("s_quadmask_b32"))
    assert "s_quadmask_b32(" in quadmask_source, quadmask_source
    quadmask64_source = translate_int(quadmask_bitreplicate_program("s_quadmask_b64"))
    assert "uint2 s_quadmask_b64(uint2 value)" in quadmask64_source, quadmask64_source
    bitreplicate_source = translate_int(quadmask_bitreplicate_program("s_bitreplicate_b64_b32"))
    assert "uint2 s_bitreplicate_b64_b32(uint value)" in bitreplicate_source, bitreplicate_source
    # A bitreplicate leaves the seeded SCC available to the following select.
    assert re.search(r"=?\s*scc\d+ \? 1u : 0u", bitreplicate_source), bitreplicate_source
    for bad in ("s_quadmask_b32 s62, v12",
                "s_quadmask_b32 s62, s[44:45]",
                "s_quadmask_b64 s44, s[44:45]",
                "s_quadmask_b64 s[44:46], s[44:45]",
                "s_bitreplicate_b64_b32 s[62:63], v12",
                "s_bitreplicate_b64_b32 s62, s44",
                "s_bitreplicate_b64_b32 s[62:63], s[44:45]"):
        changed = [line.replace(QUADMASK_BITREPLICATE_INSTRUCTIONS[
                     "s_quadmask_b32" if "quadmask_b32" in bad else
                     "s_quadmask_b64" if "quadmask_b64" in bad else
                     "s_bitreplicate_b64_b32"], bad)
                   for line in quadmask_bitreplicate_program(
                       "s_quadmask_b32" if "quadmask_b32" in bad else
                       "s_quadmask_b64" if "quadmask_b64" in bad else
                       "s_bitreplicate_b64_b32")]
        must_reject(translate_int, changed)


# SDWA moves and half widening. In SDWA_PRESERVE_BODY each move writes a 16-bit field into WORD_0 of a register
# holding 1.0 and keeps its high half, so the word is 1 + f * 2^-23; (v - 1) * 2^23 is split into the low and high
# bytes of f. Channels 0 and 1 read WORD_1 of S0; channels 2 and 3 read BYTE_2 of S1 sign-extended to 16 bits.
# SDWA_PAD_BODY writes BYTE_0 of S0 to BYTE_3 and WORD_0 of S2 to WORD_1 with zero padding, and widens each half
# of S1, the low one through -|x|. Each constant set has wrong references (channel, value) for one selection.
def sdwa_readback(value, high):
    return [line.format(r=value, h=high) for line in SDWA_READBACK]


SDWA_READBACK = ["v_subrev_f32_e32 {r}, 1.0, {r}", "v_mul_f32_e32 {r}, 0x4b000000, {r}", "v_mul_f32_e32 {h}, 0x3b800000, {r}", "v_floor_f32_e32 {h}, {h}", "v_fmac_f32_e32 {r}, 0xc3800000, {h}"]
SDWA_PRESERVE_BODY = [
    "v_mov_b32_e32 v12, 1.0",
    "v_mov_b32_sdwa v12, s44 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1",
    *sdwa_readback("v12", "v13"),
    "v_mov_b32_e32 v14, 1.0",
    "v_mov_b32_sdwa v14, sext(s45) dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:BYTE_2",
    *sdwa_readback("v14", "v15"),
]
SDWA_PAD_BODY = [
    "v_mov_b32_sdwa v12, s44 dst_sel:BYTE_3 dst_unused:UNUSED_PAD src0_sel:BYTE_0",
    "v_cvt_f32_f16_sdwa v13, s45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1",
    "v_cvt_f32_f16_sdwa v14, -|s45| dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0",
    "v_mov_b32_sdwa v15, s46 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_0",
]
SDWA_SETS = (
    # A wrong WORD_0 selection; a zero-extended byte.
    (SDWA_PRESERVE_BODY, (0x12345678, 0x00c00000, 0), [52.0, 18.0, 192.0, 255.0], ((0, 120.0), (1, 86.0), (3, 0.0))),
    (SDWA_PRESERVE_BODY, (0xfedc0000, 0x007f0000, 0), [220.0, 254.0, 127.0, 0.0], ((0, 0.0), (3, 255.0))),
    # BYTE_1 instead of BYTE_0; the other half of S1 in channels 1 and 2; WORD_1 instead of WORD_0 of S2.
    (SDWA_PAD_BODY, (0x0000c340, 0xc5003c00, 0x3f80c0a0), [2.0, -5.0, -1.0, -5.0], ((0, -128.0), (1, 1.0), (2, -5.0), (3, 1.0))),
    (SDWA_PAD_BODY, (0x000047c0, 0x38004900, 0x00004120), [-2.0, 0.5, -10.0, 10.0], ((0, 32768.0), (1, 10.0), (2, -0.5), (3, 0.0))),
)


def sdwa_reference(body, words):
    """Selections and widening on the integer words, independent of the lowering."""
    x, y, z = words
    if body is SDWA_PRESERVE_BODY:
        first = x >> 16
        byte = (y >> 16) & 0xff
        second = (byte - 0x100 if byte & 0x80 else byte) & 0xffff
        return [float(first & 0xff), float(first >> 8), float(second & 0xff), float(second >> 8)]
    as_float = lambda w: struct.unpack("<f", struct.pack("<I", w))[0]
    return [as_float((x & 0xff) << 24), half_value(y >> 16), -abs(half_value(y)), as_float((z & 0xffff) << 16)]


def check_sdwa_draw(directory, compiler, renderer, backend):
    cases_by_name = {"sdwa_preserve": [], "sdwa_pad": []}
    for name, body in (("sdwa_preserve", SDWA_PRESERVE_BODY), ("sdwa_pad", SDWA_PAD_BODY)):
        source = directory / f"{name}.hlsl"
        shader = directory / f"{name}.cso"
        source.write_text(translate(with_constants(body)), encoding="ascii")
        compile_shader(compiler, source, shader)
    for body, words, expected, wrongs in SDWA_SETS:
        assert sdwa_reference(body, words) == expected, (words, expected)
        name = "sdwa_preserve" if body is SDWA_PRESERVE_BODY else "sdwa_pad"
        cases = cases_by_name[name]
        constants = bytearray(256)
        struct.pack_into("<3I", constants, 12, *words)
        for x in (*expected, *(value for _, value in wrongs)):
            assert half_rtz(x) == x, x
        for channel, value in wrongs:
            assert value != expected[channel]
            wrong_values = list(expected)
            wrong_values[channel] = value
            path = directory / f"{name}-{len(cases)}.case"
            path.write_bytes(constants + struct.pack("<4f", *wrong_values) * 64)
            cases.append((path, "mismatch"))
        pass_path = directory / f"{name}-{len(cases)}.case"
        pass_path.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        cases.append((pass_path, "pass"))
    for name, cases in cases_by_name.items():
        run_pixel_batch(renderer, backend, directory / f"{name}.cso", cases)


# SDWA moves and half widening. In SDWA_PRESERVE_BODY each move writes a 16-bit field into WORD_0 of a register
# holding 1.0 and keeps its high half, so the word is 1 + f * 2^-23; (v - 1) * 2^23 is split into the low and high
# bytes of f. Channels 0 and 1 read WORD_1 of S0; channels 2 and 3 read BYTE_2 of S1 sign-extended to 16 bits.
# SDWA_PAD_BODY writes BYTE_0 of S0 to BYTE_3 and WORD_0 of S2 to WORD_1 with zero padding, and widens each half
# of S1, the low one through -|x|. Each constant set has wrong references (channel, value) for one selection.
def sdwa_arith(op, dst, operands, dst_sel="DWORD", unused="UNUSED_PAD", clamp=False):
    sels = " src0_sel:DWORD src1_sel:DWORD" if len(operands) == 2 else " src0_sel:DWORD"
    return (f"v_{op}_f32_sdwa {dst}, {', '.join(operands)}{' clamp' if clamp else ''}"
            f" dst_sel:{dst_sel} dst_unused:{unused}{sels}")


SDWA_ARITH = (
    ([
        "v_mov_b32_e32 v18, s45",
        sdwa_arith("add", "v12", ["s44", "v18"]),
        sdwa_arith("sub", "v13", ["-s44", "v18"]),
        sdwa_arith("subrev", "v14", ["s46", "v18"], clamp=True),
        sdwa_arith("mul", "v15", ["-|s46|", "v18"]),
    ], (1.5, 2.25, -3.0, 0.0), [3.75, -3.75, 1.0, -6.75], ((0, 0.75), (1, -0.75), (2, 5.25), (3, 6.75))),
    ([
        "v_mov_b32_e32 v18, s45",
        "v_mov_b32_e32 v19, s47",
        sdwa_arith("max", "v12", ["s44", "v18"]),
        sdwa_arith("min", "v13", ["0.5", "v18"]),
        sdwa_arith("mul_legacy", "v14", ["s46", "v19"]),
        sdwa_arith("floor", "v15", ["-s45"]),
    ], (-0.75, 2.5, 0.0, math.inf), [2.5, 0.5, 0.0, -3.0], ((0, -0.75), (1, 2.5), (2, -0.0), (3, -2.0))),
    ([
        sdwa_arith("ceil", "v12", ["s44"]),
        sdwa_arith("fract", "v13", ["-s45"]),
        sdwa_arith("trunc", "v14", ["|s46|"]),
        "v_mov_b32_e32 v19, 0",
        sdwa_arith("add", "v15", ["s47", "v19"], dst_sel="BYTE_3"),
    ], (-1.25, 2.75, -2.5, 0x3f800040), [-1.0, 0.25, 2.0, 2.0], ((0, -2.0), (1, 0.75), (2, -2.0), (3, 1.0))),
    ([
        "v_mov_b32_e32 v18, s45",
        "v_mov_b32_e32 v12, 1.0",
        sdwa_arith("mul", "v12", ["s44", "v18"], "WORD_0", "UNUSED_PRESERVE"),
        *sdwa_readback("v12", "v13"),
        "v_mov_b32_e32 v14, 1.0",
        sdwa_arith("floor", "v14", ["s46"], "BYTE_2", "UNUSED_PRESERVE"),
        sdwa_arith("mul", "v15", ["s47", "v18"], "WORD_1"),
    ], (0x3f801234, 1.0, 8388613.0, 0xc0a03fff), [52.0, 18.0, 0.51953125, 1.9921875],
     ((0, 128.0), (1, 63.0), (2, 1.0), (3, -5.0))),
)


def sdwa_arith_reference(index, inputs):
    """Python binary32 arithmetic on the constant words, independent of the lowering."""
    as_float = lambda w: struct.unpack("<f", struct.pack("<I", w))[0]
    bits = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]
    x, y, z, w = (as_float(v) if isinstance(v, int) else v for v in inputs)
    if index == 0:
        return [x + y, -x - y, min(max(y - z, 0.0), 1.0), -abs(z) * y]
    if index == 1:
        return [max(x, y), min(0.5, y), 0.0 if z == 0 or w == 0 else z * w, float(math.floor(-y))]
    if index == 2:
        return [float(math.ceil(x)), -y - math.floor(-y), float(math.trunc(abs(z))), as_float((bits(w + 0.0) & 0xff) << 24)]
    product = bits(x * y) & 0xffff
    return [float(product & 0xff), float(product >> 8),
            as_float((0x3f800000 & ~0xff0000) | ((bits(float(math.floor(z))) & 0xff) << 16)), as_float((bits(w * y) & 0xffff) << 16)]


def check_sdwa_arith_draws(directory, compiler, renderer, backend):
    for index, (body, inputs, expected, wrongs) in enumerate(SDWA_ARITH):
        assert sdwa_arith_reference(index, inputs) == expected, (index, sdwa_arith_reference(index, inputs))
        source = directory / f"sdwa_arith{index}.hlsl"
        shader = directory / f"sdwa_arith{index}.cso"
        source.write_text(translate(with_constants(body)), encoding="ascii")
        compile_shader(compiler, source, shader)
        words = [v if isinstance(v, int) else struct.unpack("<I", struct.pack("<f", v))[0] for v in inputs]
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *words)
        for x in (*expected, *(value for _, value in wrongs)):
            assert half_rtz(x) == x, x
        cases = []
        for channel, value in wrongs:
            assert struct.pack("<f", value) != struct.pack("<f", expected[channel])
            wrong_values = list(expected)
            wrong_values[channel] = value
            path = directory / f"sdwa_arith{index}-wrong-{channel}.case"
            path.write_bytes(constants + struct.pack("<4f", *wrong_values) * 64)
            cases.append((path, "mismatch"))
        pass_path = directory / f"sdwa_arith{index}-pass.case"
        pass_path.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        cases.append((pass_path, "pass"))
        run_pixel_batch(renderer, backend, shader, cases)


# SDWA forms of the remaining unary binary32 operations, with whole-dword sources. D3D allows exp2, log2, a
# reciprocal, rsqrt, sqrt, sin and cos an error, so a raw result has no single binary32 reference: in the first
# two programs each input makes the true result an integer, and adding 0.5 then taking the floor returns that
# integer for any result within 0.5 of it. That checks which operation ran and its source modifiers, not
# last-place accuracy. v_rndne_f32 and v_frexp_mant_f32 are exact and are compared directly, with clamp, a byte
# kept in 1.0 under UNUSED_PRESERVE and a byte padded into BYTE_3. A, B, C and D are the constants in s44 to s47.
SDWA_UNARY_OPS = tuple(f"v_{op}_f32_sdwa" for op in (
    "exp", "log", "rcp", "rcp_iflag", "rsq", "sqrt", "sin", "cos", "rndne", "frexp_mant"))
SDWA_UNARY_ROUND = [line for channel in (12, 13, 14, 15)
                    for line in (f"v_add_f32_e32 v{channel}, 0.5, v{channel}", f"v_floor_f32_e32 v{channel}, v{channel}")]


def sdwa_unary_place(result, old, offset, width):
    """Bits of result's low field written at offset into old (0 for UNUSED_PAD), as a float."""
    bits, = struct.unpack("<I", struct.pack("<f", result))
    mask = (1 << width) - 1
    word = (old & ~(mask << offset) & 0xffffffff) | ((bits & mask) << offset)
    return struct.unpack("<f", struct.pack("<I", word))[0]


def sdwa_unary_rounded(*values):
    return tuple(float(math.floor(value + 0.5)) for value in values)


# (name, body, rows of constants, reference(a, b, c, d), wrong references). A wrong reference is
# (row, channel, value(a, b, c, d), what a lowering exporting it got wrong) and must report a mismatch.
SDWA_UNARY = (
    ("sdwa_unary_exp", [
        sdwa_arith("exp", "v12", ["-s44"]),
        sdwa_arith("log", "v13", ["|s45|"]),
        sdwa_arith("rcp", "v14", ["s46"]),
        sdwa_arith("rcp_iflag", "v15", ["-|s47|"]),
        *SDWA_UNARY_ROUND,
    ], ((-3.0, -32.0, 0.125, 0.25), (-4.0, -8.0, -0.5, -0.0625)),
     lambda a, b, c, d: sdwa_unary_rounded(2.0 ** -a, math.log2(abs(b)), 1.0 / c, 1.0 / -abs(d)),
     ((0, 0, lambda a, b, c, d: math.floor(2.0 ** a + 0.5), "negation dropped on exp"),
      (0, 0, lambda a, b, c, d: math.floor(math.exp(-a) + 0.5), "natural exponential for exp2"),
      (0, 1, lambda a, b, c, d: math.floor(math.log(abs(b)) + 0.5), "natural logarithm for log2"),
      (0, 1, lambda a, b, c, d: math.floor(abs(b) + 0.5), "source copied for log"),
      (0, 2, lambda a, b, c, d: math.floor(c ** -0.5 + 0.5), "rsq for rcp"),
      (0, 2, lambda a, b, c, d: math.floor(c + 0.5), "source copied for rcp"),
      (0, 3, lambda a, b, c, d: math.floor(1.0 / abs(d) + 0.5), "negation dropped on rcp_iflag"),
      (0, 3, lambda a, b, c, d: math.floor(-abs(d) + 0.5), "source copied for rcp_iflag"))),
    ("sdwa_unary_root", [
        sdwa_arith("rsq", "v12", ["|s44|"]),
        sdwa_arith("sqrt", "v13", ["s45"]),
        sdwa_arith("sin", "v14", ["s46"]),
        sdwa_arith("cos", "v15", ["-s47"]),
        *SDWA_UNARY_ROUND,
    ], ((-0.0625, 49.0, 0.25, -0.5), (-0.25, 9.0, -0.25, 1.0)),
     lambda a, b, c, d: sdwa_unary_rounded(abs(a) ** -0.5, math.sqrt(b), math.sin(2 * math.pi * c),
                                           math.cos(2 * math.pi * -d)),
     ((0, 0, lambda a, b, c, d: math.floor(math.sqrt(abs(a)) + 0.5), "sqrt for rsq"),
      (0, 0, lambda a, b, c, d: math.floor(1.0 / abs(a) + 0.5), "rcp for rsq"),
      (0, 1, lambda a, b, c, d: math.floor(b ** -0.5 + 0.5), "rsq for sqrt"),
      (0, 1, lambda a, b, c, d: b, "source copied for sqrt"),
      (0, 2, lambda a, b, c, d: math.floor(math.cos(2 * math.pi * c) + 0.5), "cos for sin"),
      (0, 2, lambda a, b, c, d: math.floor(math.sin(c) + 0.5), "radians for revolutions in sin"),
      (0, 3, lambda a, b, c, d: math.floor(math.sin(2 * math.pi * -d) + 0.5), "sin for cos"),
      (0, 3, lambda a, b, c, d: math.floor(math.cos(-d) + 0.5), "radians for revolutions in cos"))),
    ("sdwa_unary_exact", [
        sdwa_arith("rndne", "v12", ["s44"]),
        sdwa_arith("frexp_mant", "v13", ["-s45"], clamp=True),
        "v_mov_b32_e32 v14, 1.0",
        sdwa_arith("rndne", "v14", ["s46"], "BYTE_2", "UNUSED_PRESERVE"),
        sdwa_arith("frexp_mant", "v15", ["s47"], dst_sel="BYTE_3"),
    ], ((2.5, -12.0, 8388613.0, 0x41400040), (-3.5, 12.0, 4194306.5, 0x414000c0)),
     lambda a, b, c, d: (float(round(a)), min(max(math.frexp(-b)[0], 0.0), 1.0),
                         sdwa_unary_place(float(round(c)), 0x3f800000, 16, 8),
                         sdwa_unary_place(math.frexp(d)[0], 0, 24, 8)),
     ((0, 0, lambda a, b, c, d: math.floor(a + 0.5), "ties rounded up"),
      (1, 0, lambda a, b, c, d: math.trunc(a), "ties rounded toward zero"),
      (0, 1, lambda a, b, c, d: min(max(math.frexp(b)[0], 0.0), 1.0), "negation dropped on frexp_mant"),
      (0, 1, lambda a, b, c, d: 1.0, "source copied for frexp_mant, then clamped"),
      (1, 1, lambda a, b, c, d: math.frexp(-b)[0], "clamp dropped"),
      (0, 2, lambda a, b, c, d: half_rtz(sdwa_unary_place(float(round(c)), 0, 16, 8)), "padded instead of preserved"),
      (0, 2, lambda a, b, c, d: half_rtz(sdwa_unary_place(float(round(c)), 0x3f800000, 8, 8)),
       "placed in the neighbouring byte"),
      (1, 2, lambda a, b, c, d: sdwa_unary_place(float(math.floor(c + 0.5)), 0x3f800000, 16, 8),
       "ties rounded up, placed"),
      (0, 3, lambda a, b, c, d: half_rtz(math.frexp(d)[0]), "result not placed"),
      (0, 3, lambda a, b, c, d: sdwa_unary_place(d, 0, 24, 8) * 2.0, "wrong byte value placed"))),
)


# Raw results of the operations, before rounding, of the one SDWA_UNARY program that contains log2.
SDWA_UNARY_RAW = {
    "sdwa_unary_exp": lambda a, b, c, d: (-a, abs(b), -abs(d), 2.0 ** -a, math.log2(abs(b)), 1.0 / c,
                                          1.0 / -abs(d)),
}


def sdwa_unary_inputs(row):
    """Constants of a row as floats; an int is a binary32 bit pattern."""
    return tuple(struct.unpack("<f", struct.pack("<I", v))[0] if isinstance(v, int) else v for v in row)


def check_sdwa_unary_lowering():
    """Static checks for SDWA_UNARY: each form reuses its e32/e64 lowering, and the refusals stay."""
    bodies = [translate(with_constants(body)).split("float4 main", 1)[1] for _, body, *_ in SDWA_UNARY]
    assert set(SDWA_UNARY_OPS) <= ACCEPTED, set(SDWA_UNARY_OPS) - ACCEPTED
    for text in ("exp2((-params[0].w))", "log2(abs(params[1].x))", "(1.0f / params[1].y)",
                 "(1.0f / (-abs(params[1].z)))"):
        assert text in bodies[0], (text, bodies[0])
    for text in ("rsqrt(abs(params[0].w))", "sqrt(params[1].x)", "sin32(params[1].y)", "cos32((-params[1].z))"):
        assert text in bodies[1], (text, bodies[1])
    for text in ("round(params[0].w)", "saturate(frexp_mant32((-params[1].x)))",
                 "& 0xff00ffffu) | ((asuint(round(params[1].y)) & 0xffu) << 16u)",
                 "((asuint(frexp_mant32(params[1].z)) & 0xffu) << 24u)"):
        assert text in bodies[2], (text, bodies[2])
    for op in SDWA_UNARY_OPS:
        # The SDWA and e32 spellings of each operation produce the same shader.
        sdwa = translate(with_constants([f"{op} v12, s44 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD"]))
        assert sdwa == translate(with_constants([f"{op.replace('_sdwa', '_e32')} v12, s44"])), op
        # Byte and word sources stay refused, as do a second source selection, a second operand, an output
        # modifier, UNUSED_SEXT on a partial destination and UNUSED_PRESERVE over an untracked destination.
        name = op[2:-len("_f32_sdwa")]
        for field in ("BYTE_0", "BYTE_3", "WORD_0", "WORD_1"):
            must_reject(translate, with_constants(
                [sdwa_arith(name, "v12", ["s44"]).replace("src0_sel:DWORD", f"src0_sel:{field}")]))
        must_reject(translate, with_constants([sdwa_arith(name, "v12", ["s44"]) + " src1_sel:DWORD"]))
        must_reject(translate, with_constants(["v_mov_b32_e32 v18, s45", sdwa_arith(name, "v12", ["s44", "v18"])]))
        must_reject(translate, with_constants([sdwa_arith(name, "v12", ["s44 mul:2"])]))
        must_reject(translate, with_constants(
            ["v_mov_b32_e32 v12, 1.0", sdwa_arith(name, "v12", ["s44"], "WORD_0", "UNUSED_SEXT")]))
        must_reject(translate, with_constants([sdwa_arith(name, "v20", ["s44"], "BYTE_2", "UNUSED_PRESERVE")]))


def check_sdwa_unary_draws(directory, compiler, renderer, backend):
    """Draw each program of SDWA_UNARY once per row and once per wrong reference.

    One compile and one renderer batch per program. A renderer that reports an unavailable adapter (exit 77)
    makes this a skip. On a desktop GPU the draws establish what the translated shader computes, not what the
    console does.
    """
    for name, body, rows, reference, wrongs in SDWA_UNARY:
        source = directory / f"{name}.hlsl"
        shader = directory / f"{name}.cso"
        hlsl = translate(with_constants(body))
        source.write_text(hlsl, encoding="ascii")
        values = []
        for index, row in enumerate(rows):
            inputs = sdwa_unary_inputs(row)
            raw = SDWA_UNARY_RAW[name](*inputs) if name in SDWA_UNARY_RAW else ()
            values += [*inputs, *raw, *(value + 0.5 for value in raw), *reference(*inputs)]
            values += [float(wrong(*inputs)) for wrong_row, _, wrong, _ in wrongs if wrong_row == index]
        mode = log2_denorm_mode(name, hlsl, *values)
        # Only sdwa_unary_exp contains log2; sdwa_unary_root and sdwa_unary_exact stay under preserve.
        assert (mode == LOG2_DENORM_MODE) == (name in SDWA_UNARY_RAW), (name, mode)
        compile_shader(compiler, source, shader, mode=mode)
        cases = []

        def add_case(row, values, status):
            words = [v if isinstance(v, int) else struct.unpack("<I", struct.pack("<f", v))[0] for v in row]
            constants = bytearray(256)
            struct.pack_into("<4I", constants, 12, *words)
            path = directory / f"{name}-{len(cases)}.case"
            path.write_bytes(bytes(constants) + struct.pack("<4f", *values) * 64)
            cases.append((path, status))

        for index, row in enumerate(rows):
            right = tuple(reference(*sdwa_unary_inputs(row)))
            # Every right value survives the binary16 export unchanged.
            assert tuple(map(half_rtz, right)) == right, (name, right)
            for wrong_row, channel, wrong, reason in wrongs:
                if wrong_row != index:
                    continue
                values = list(right)
                values[channel] = float(wrong(*sdwa_unary_inputs(row)))
                assert half_rtz(values[channel]) == values[channel], (name, reason, values[channel])
                assert struct.pack("<f", values[channel]) != struct.pack("<f", right[channel]), (name, reason)
                add_case(row, values, "mismatch")
            add_case(row, right, "pass")
        assert cases[-1][1] == "pass" and len(cases) == len(rows) + len(wrongs), name
        try:
            run_pixel_batch(renderer, backend, shader, cases)
        except subprocess.CalledProcessError as error:
            if error.returncode == 77:
                return 77
            raise
    return 0


def check_constant_descriptor_bounds() -> None:
    for offset in (0, 28, 32):
        program = [line.replace("s[20:21], 0x40", f"s[20:21], {offset:#x}")
                   for line in curve()]
        if offset == 32:
            assert lower(program, 20, 24, [1, 2, 3, 4], offset) == translate(curve())
        else:
            try:
                lower(program, 20, 24, [1, 2, 3, 4], offset)
            except ValueError as error:
                assert "descriptors overlap" in str(error), error
            else:
                raise AssertionError(f"Constant descriptor at {offset} overlaps texture 0")


def check_three_constant_buffers() -> None:
    program = with_three_constant_buffers()
    source = translate_three_buffers(program)
    for slot, name in enumerate(("params", "params1", "params2")):
        suffix = str(slot) if slot else ""
        assert f"cbuffer Constants{suffix} : register(b{slot}) {{ float4 {name}[16]; }};" in source
        assert source.count(f"{name}[0].z") == 1

    def bound(instructions, constant_descriptor=64, extra_constants=((20, 0x60), (9, 0x20))):
        return lower(instructions, 20, 24, [1, 2, 3, 4], constant_descriptor, extra_constants=extra_constants)

    def changed(old, new):
        assert old in program
        return [new if line == old else line for line in program]

    def exchange(text, a, b):
        return text.replace(a, "#").replace(b, a).replace("#", b)

    # The contract order alone selects the register, and b0 is optional.
    swapped = bound(program, extra_constants=((9, 0x20), (20, 0x60)))
    assert swapped != source and swapped == exchange(source, "params1[0].z", "params2[0].z")
    only_extra = bound(program, None, ((20, 0x40), (20, 0x60), (9, 0x20)))
    assert "register(b0)" not in only_extra and "params3[0].z" in only_extra
    # One offset on two tables names two descriptors.
    same_offset = changed("s_load_dwordx4 s[56:59], s[64:65], 0x20", "s_load_dwordx4 s[56:59], s[64:65], 0x60")
    assert bound(same_offset, extra_constants=((20, 0x60), (9, 0x60))) == source

    # A descriptor the contract does not name is refused, on either table.
    for old in ("s_load_dwordx4 s[52:55], s[20:21], 0x60", "s_load_dwordx4 s[56:59], s[64:65], 0x20"):
        expect_value_error(bound, changed(old, old[:-2] + "70"), "Constant descriptor differs from binding contract")
    expect_value_error(translate, program, "Constant descriptor differs from binding contract")
    expect_value_error(lambda p: bound(p, extra_constants=((9, 0x20),)), program,
                       "Constant descriptor differs from binding contract")
    expect_value_error(lambda p: bound(p, None, ()), program, "Unresolved constant buffer binding")
    # An entry register outside the contract stays unbound.
    try:
        bound(program, extra_constants=((20, 0x60),))
    except KeyError as error:
        assert error.args == ("s9",), error
    else:
        raise AssertionError("An unbound table register was accepted")
    # The table address needs a zero high word, and its low word is not a number.
    for high in ("s_mov_b32 s65, 1", "s_mov_b32 s65, s9", "s_mov_b32 s65, s20"):
        expect_value_error(bound, changed("s_mov_b32 s65, 0", high), "Unresolved constant buffer binding")
    expect_value_error(bound, changed("s_mov_b32 s64, s9", "s_add_i32 s64, s9, 4"),
                       "Expected a numeric scalar register: s9")
    # A texture descriptor is not read from a constant table.
    expect_value_error(bound, changed("s_load_dwordx4 s[56:59], s[64:65], 0x20",
                                      "s_load_dwordx8 s[56:63], s[64:65], 0x20"),
                       "Unbound texture descriptor offset")
    expect_value_error(lambda p: bound(p, extra_constants=((20, 0x60), (9, 0x20))),
                       changed("s_load_dwordx4 s[56:59], s[64:65], 0x20", "s_load_dwordx8 s[56:63], s[64:65], null"),
                       "Texture address differs from binding contract")
    # Descriptor words from two buffers do not form one descriptor.
    read = program.index("s_buffer_load_dword s62, s[52:55], 0x8")
    wait = "s_waitcnt lgkmcnt(0)"
    for mixed in ("s_mov_b32 s53, s57", "s_mov_b32 s52, s56", "s_mov_b32 s55, 0"):
        expect_value_error(bound, program[:read] + [wait, mixed] + program[read:], "Unresolved constant buffer binding")
    # A whole descriptor copied word by word keeps its buffer.
    copied = program[:read] + [wait, *(f"s_mov_b32 s{52 + n}, s{56 + n}" for n in range(4))] + program[read:]
    assert "params1" not in bound(copied).split("main(")[1] and bound(copied).count("params2[0].z") == 2
    # A descriptor is read only after its load completes.
    first_wait = program.index(wait)
    must_reject(bound, program[:first_wait] + ["s_buffer_load_dword s62, s[52:55], 0x8"] + program[first_wait:])

    for extra, message in (
            (((20, 0x40),), "Constant descriptors overlap"), (((20, 0x4c),), "Constant descriptors overlap"),
            (((20, 0x34),), "Constant descriptors overlap"), (((9, 0x20), (9, 0x2c)), "Constant descriptors overlap"),
            (((20, 0),), "Texture and constant descriptors overlap"), (((20, 0x1c),), "Texture and constant descriptors overlap"),
            (((24, 0x20),), "Invalid constant descriptor binding"), (((9, 0x22),), "Invalid constant descriptor binding"),
            (((9, -4),), "Invalid constant descriptor binding"), (((106, 0),), "Invalid constant descriptor binding"),
            (((-1, 0),), "Invalid constant descriptor binding"), (((9,),), "Invalid constant descriptor binding"),
            (((9, 0x20, 0),), "Invalid constant descriptor binding"), (((9.0, 0x20),), "Invalid constant descriptor binding"),
            (((True, 0x20),), "Invalid constant descriptor binding")):
        expect_value_error(lambda p: bound(p, extra_constants=extra), program, message)
    # Adjacent 16-byte descriptors on one table are distinct.
    adjacent = changed("s_load_dwordx4 s[52:55], s[20:21], 0x60", "s_load_dwordx4 s[52:55], s[20:21], 0x50")
    assert bound(adjacent, extra_constants=((20, 0x50), (9, 0x20))) == source


def check_scalar_constant_reads() -> None:
    program = with_scalar_constant_reads()
    source = translate(program)
    for field in ("params[0].x", "params[0].z", "params[15].w"):
        assert f"= {field};" in source

    load = "s_buffer_load_dword s62, s[44:47], 0x8"
    for bad in (load.replace("0x8", "0x9"), load.replace("0x8", "0x100"), load.replace("0x8", "s60"),
                load.replace("s[44:47]", "s[28:31]"), load.replace("s[44:47]", "s[44:46]")):
        must_reject(translate, [bad if line == load else line for line in program])

    first = program.index("s_buffer_load_dword s61, s[44:47], null")
    wait = program.index("s_waitcnt lgkmcnt(0)", first)
    descriptor = program.index("s_load_dwordx4 s[44:47], s[20:21], 0x40")
    must_reject(translate, program[:descriptor + 1] + [load] + program[descriptor + 1:])
    must_reject(translate, program[:wait] + program[wait + 1:])  # Result read before the wait.
    must_reject(translate, program[:wait] + [load] + program[wait:])  # Outstanding write to the same SGPR.
    must_reject(translate, program[:first] + ["s_mov_b32 s46, 0"] + program[first:])
    must_reject(translate, [line.replace("s[44:47]", "s[46:49]") for line in program])

    alias = [line.replace("s63", "s46") for line in program]
    assert translate(alias) == source
    must_reject(translate, alias[:wait + 1] + [load] + alias[wait + 1:])


def check_affine_coordinates() -> None:
    program = affine_coordinates()
    final = "v_fma_f32 v8, 0.5, v8, 0"
    final_index = program.index(final)
    source = translate(program)
    assert f".Sample(samp, float2(alu{final_index}, uv.y))" in source

    exact_input = ["v_mov_b32_e32 v22, 0.5", *program]
    exact_input[final_index + 1] = final.replace("0.5", "v22")
    must_reject(translate, exact_input)
    mask_input = ["v_cmp_eq_f32_e64 s52, 0.5, 0.5", *program]
    mask_input[final_index + 1] = final.replace("0.5", "s52")
    must_reject(translate, mask_input)

    restore = program.index("s_and_b64 exec, exec, s[50:51]")
    exact_overwrite = "v_add_f32_e32 v8, -0.5, v8"
    must_reject(translate, program[:restore + 1] + [exact_overwrite] + program[restore + 1:])
    duplicate = [*program[:final_index], final.replace("v8, 0.5", "v22, 0.5"), *program[final_index:restore],
                 "v_add_f32_e32 v23, 0.0, v9", program[restore], exact_overwrite, *program[restore + 1:]]
    separate_coords = [line.replace("v[8:9]", "v[22:23]") for line in duplicate]
    assert f".Sample(samp, float2(alu{final_index}, alu{restore + 1}))" in translate(separate_coords)
    partial = program.index("v_interp_p2_f32_e32 v8, v1, attr0.x")
    must_reject(translate, program[:partial] + ["v_add_f32_e32 v8, 0.0, v8"] + program[partial:])
    wait = program.index("s_buffer_load_dwordx4 s[60:63], s[44:47], null") + 1
    must_reject(translate, program[:wait] + program[wait + 1:])

    for opcode in ("v_mul_legacy_f32_e32 v8, v9, v8", "v_fma_f32 v8, v9, v8, 0",
                   "v_fma_f32 v8, 0.5, |v8|, 0", "v_fma_f32 v8, 0.5, v8, 0 clamp"):
        changed = program.copy()
        changed[final_index] = opcode
        must_reject(translate, changed)


def check_scalar_helpers_with_vector_arithmetic() -> None:
    entry = assembly(False)
    entry_add = "v_add_f32_e32 v12, 0.0, v12"
    entry.insert(entry.index("s_waitcnt vmcnt(0)") + 1, entry_add)
    quad_add = "v_add_f32_e32 v8, 0.5, v8"
    scalar = "s_bfe_u32 s70, 1, 0x10000"
    for program, vector in ((entry, entry_add), (affine_coordinates(), quad_add)):
        position = program.index(vector)
        for index in (position, position + 1):
            source = translate(program[:index] + [scalar] + program[index:])
            assert "uint s_bfe_u32(" in source and " = s_bfe_u32(" in source


# v_fmaak_f32 computes S0 * S1 + K and v_fmamk_f32 computes S0 * K + S1, with K a 32-bit literal.
# Every product and sum is exact in binary32, so the reference does not depend on fusion. Swapping
# the literal with S1 changes each result.
LITERAL_FMA_INPUTS = (1.5, 2.5, -3.0, 0.75)
LITERAL_FMA_BODY = [
    "v_mov_b32_e32 v13, s45",
    "v_fmaak_f32 v12, s44, v13, 0x3e800000",
    "v_fmamk_f32 v13, s44, 0x40400000, v13",
    "v_mov_b32_e32 v15, s47",
    "v_fmaak_f32 v14, s46, v15, 0xbf000000",
    "v_fmamk_f32 v15, s46, 0x3f000000, v15",
]


def check_literal_fma_draw(directory, compiler, renderer, backend):
    source = directory / "literal_fma.hlsl"
    shader = directory / "literal_fma.cso"
    source.write_text(translate(with_constants(LITERAL_FMA_BODY)), encoding="ascii")
    compile_shader(compiler, source, shader)
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, *LITERAL_FMA_INPUTS)
    a, b, c, d = LITERAL_FMA_INPUTS
    expected = [a * b + 0.25, a * 3.0 + b, c * d - 0.5, c * 0.5 + d]
    swapped = [a * 0.25 + b, a * b + 3.0, c * -0.5 + d, c * d + 0.5]
    for x in (*expected, *swapped):
        assert single(x) == x == half_rtz(x), x
    assert all(x != y for x, y in zip(expected, swapped))
    cases = []
    # Treating the literal as S1, or S1 as the literal, must change the exported color.
    for k in range(4):
        wrong_values = list(expected)
        wrong_values[k] = swapped[k]
        path = directory / f"literal_fma-wrong-{k}.case"
        write_case(path, constants, tuple(wrong_values))
        cases.append((path, "mismatch"))
    pass_case = directory / "literal_fma-pass.case"
    write_case(pass_case, constants, tuple(expected))
    cases.append((pass_case, "pass"))
    run_pixel_batch(renderer, backend, shader, cases)
def check_scalar_pair_reads() -> None:
    program = with_scalar_pair_reads()
    source = translate(program)
    for field in ("(params[0].w * sample", "max(params[15].z, ", "(params[1].x + sample", "(params[1].y - sample",
                  "= params[1].z;", "(params[15].w * alu"):
        assert field in source, field

    load = PAIR_LOADS[1]
    for old, new in (("0x14", "0x15"), ("0x14", "0x16"),  # not dword-aligned
                     ("0x14", "0xfc"), ("0x14", "0x100"),  # the second word or both pass the 256 bound bytes
                     ("0x14", "s60"), ("0x14", "s60, 0x14"),  # register offset
                     ("s[62:63]", "s62"), ("s[62:63]", "s[62:62]"), ("s[62:63]", "s[62:64]"),
                     ("s[62:63]", "s[60:63]"), ("s[62:63]", "s[61:62]"), ("s[62:63]", "s[106:107]"),
                     ("s[44:47]", "s[28:31]"), ("s[44:47]", "s[44:45]"), ("s[44:47]", "s[44:46]"),
                     ("dwordx2", "dword"), ("dwordx2", "dwordx4"), ("s_buffer_load", "s_load"),
                     ("dwordx2 s[62:63]", "dwordx8 s[56:63]"), ("dwordx2 s[62:63]", "dwordx16 s[48:63]")):
        bad = load.replace(old, new)
        assert bad != load
        must_reject(translate, [bad if line == load else line for line in program])

    first = program.index(PAIR_LOADS[0])
    wait = program.index("s_waitcnt lgkmcnt(0)", first)
    descriptor = program.index("s_load_dwordx4 s[44:47], s[20:21], 0x40")
    must_reject(translate, program[:descriptor + 1] + [load] + program[descriptor + 1:])  # Descriptor still loading.
    must_reject(translate, program[:wait] + program[wait + 1:])  # Results read before the wait.
    for word in ("s62", "s63"):  # Neither word of the pair can be read or written until the wait.
        must_reject(translate, program[:wait] + [f"v_mov_b32_e32 v20, {word}"] + program[wait:])
        translate(program[:wait + 1] + [f"v_mov_b32_e32 v20, {word}"] + program[wait + 1:])
        must_reject(translate, program[:wait] + [f"s_mov_b32 {word}, 0"] + program[wait:])
        must_reject(translate, program[:wait] + [f"s_buffer_load_dword {word}, s[44:47], 0x8"] + program[wait:])
    must_reject(translate, program[:wait] + [load] + program[wait:])
    must_reject(translate, program[:wait] + ["s_mov_b64 s[62:63], exec"] + program[wait:])
    translate(program[:wait + 1] + [load, "s_waitcnt lgkmcnt(0)"] + program[wait + 1:])  # Reloading after the wait.
    must_reject(translate, program[:first] + ["s_mov_b32 s46, 0"] + program[first:])

    # A pair may land in the descriptor's own registers, which ends the descriptor.
    alias = [line.replace("s[64:65]", "s[46:47]").replace("s64", "s46").replace("s65", "s47") for line in program]
    assert alias != program and translate(alias) == source
    must_reject(translate, alias[:wait + 1] + [load] + alias[wait + 1:])


def check_readfirstlane_controls() -> None:
    program = readfirstlane_program()
    source = translate_int(program)
    assert "WaveReadLaneFirst(asuint(" in source
    assert "salu" in source
    assert "v_readfirstlane_b32" in ACCEPTED

    # Positive control: self-move of a uniform register retains uniform provenance
    self_move_prog = uncompressed_export([
        "v_mov_b32_e32 v18, s44",
        "v_mov_b32_e32 v18, v18",
        "v_readfirstlane_b32 s48, v18",
        "s_add_u32 s49, s48, 5",
        "v_mov_b32_e32 v12, s48",
        "v_mov_b32_e32 v13, s49",
        "v_mov_b32_e32 v14, 0x12345678",
        "v_mov_b32_e32 v15, 0",
    ])
    assert "WaveReadLaneFirst(asuint(" in translate_int(self_move_prog)

    # Rejection controls with uniform source register
    must_reject(translate, with_constants(["v_mov_b32_e32 v18, s44", "v_readfirstlane_b32 v12, v18"]))
    must_reject(translate, with_constants(["v_mov_b32_e32 v18, s44", "v_readfirstlane_b32 s200, v18"]))
    must_reject(translate, with_constants(["v_mov_b32_e32 v18, s44", "v_readfirstlane_b32 s48, v18 clamp"]))
    must_reject(translate, with_constants(["v_mov_b32_e32 v18, s44", "v_readfirstlane_b32 s48, abs(v18)"]))
    must_reject(translate, with_constants(["v_mov_b32_e32 v18, s44", "v_readfirstlane_b32 s48, -v18"]))
    must_reject(translate, with_constants(["v_mov_b32_e32 v18, s44", "v_readfirstlane_b32 s48, v18, v13"]))

    # Varying VGPR rejection: interpolated coordinates and barycentric inputs are not uniform
    varying_prog = list(assembly(False))
    pos = varying_prog.index("v_interp_p2_f32_e32 v8, v1, attr0.x")
    varying_prog.insert(pos + 1, "v_readfirstlane_b32 s48, v8")
    must_reject(translate, varying_prog)
    must_reject(translate, with_constants(["v_readfirstlane_b32 s48, v0"]))
    must_reject(translate, with_constants(["v_readfirstlane_b32 s48, v1"]))
    varying_self_prog = list(assembly(False))
    pos_self = varying_self_prog.index("v_interp_p2_f32_e32 v8, v1, attr0.x")
    varying_self_prog.insert(pos_self + 1, "v_mov_b32_e32 v8, v8")
    varying_self_prog.insert(pos_self + 2, "v_readfirstlane_b32 s48, v8")
    must_reject(translate, varying_self_prog)

    # Non-entry EXEC rejection (e.g. during WQM)
    wqm_prog = list(assembly(False))
    pos = wqm_prog.index("s_wqm_b64 exec, exec")
    wqm_prog.insert(pos + 1, "v_mov_b32_e32 v18, s44")
    wqm_prog.insert(pos + 2, "v_readfirstlane_b32 s48, v18")
    must_reject(translate, wqm_prog)

    # Unresolved scalar memory dependency
    pending_prog = list(assembly(False))
    pos = pending_prog.index("s_load_dwordx8 s[28:35], s[20:21], null")
    pending_prog.insert(pos + 1, "v_mov_b32_e32 v18, s44")
    pending_prog.insert(pos + 2, "v_readfirstlane_b32 s28, v18")
    must_reject(translate, pending_prog)


# s_setreg_imm32_b32 writes hardware state. The program opens with the scratch base writes, which no lowered
# instruction reads. MODE[3:2] then selects round toward zero for an exact widening, restores round to nearest
# even for two narrowings, and selects round toward zero again before a second exact widening and the
# truncating pack. Channel 0 widens S0, channels 1 and 2 narrow and widen back (channel 2 a subnormal scaled
# by 2^24) and channel 3 widens -|S0|.
FLAT_SCRATCH_WRITES = [
    "s_setreg_imm32_b32 hwreg(HW_REG_FLAT_SCR_LO), 0",
    "s_setreg_imm32_b32 hwreg(HW_REG_FLAT_SCR_HI), 0x400",
]
HALF_RTZ = "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3"
HALF_RNE = "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0"
MODE_BODY = [
    HALF_RTZ,
    "v_cvt_f32_f16_e32 v12, s44",
    HALF_RNE,
    "v_cvt_f16_f32_e32 v13, s45",
    "v_cvt_f32_f16_e32 v13, v13",
    "v_cvt_f16_f32_e32 v14, s46",
    "v_cvt_f32_f16_e32 v14, v14",
    "v_mul_f32_e32 v14, 0x4b800000, v14",
    HALF_RTZ,
    "v_cvt_f32_f16_e64 v15, -|s47|",
]
# Each set: the half in S44, two binary32 inputs whose nearest-even and truncated halves differ, the half in
# S47, and the wrong references (channel, value) that truncation after the restore would give.
MODE_SETS = (
    ((0xdeadc500, 1 + 3 * 2.0 ** -11, 1.5 * 2.0 ** -24, 0xbeef7bff), ((1, 1 + 2.0 ** -10), (2, 1.0))),
    ((0x00003c01, -(2.0 ** -14 - 2.0 ** -25), 65519.0, 0x00000001), ((1, -1023 * 2.0 ** -24),)),
)


def mode_program():
    return FLAT_SCRATCH_WRITES + with_constants(MODE_BODY)


def check_mode_register_draw(directory, compiler, renderer, backend):
    source = directory / "mode_register.hlsl"
    shader = directory / "mode_register.cso"
    source.write_text(translate(mode_program()), encoding="ascii")
    compile_shader(compiler, source, shader)
    cases = []
    for (first, second, third, last), wrongs in MODE_SETS:
        words = [first, *(struct.unpack("<I", struct.pack("<f", x))[0] for x in (second, third)), last]
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *words)
        narrowed = half_value(narrow_reference(third))
        expected = [half_value(first), half_value(narrow_reference(second)),
                    narrowed * 2.0 ** 24 if abs(narrowed) < 2.0 ** -14 else narrowed, -abs(half_value(last))]
        for x in (*expected, *(value for _, value in wrongs)):
            assert half_rtz(x) == x, x
        for channel, value in wrongs:
            assert struct.pack("<f", value) != struct.pack("<f", expected[channel])
            wrong_values = list(expected)
            wrong_values[channel] = value
            path = directory / f"mode_register-{len(cases)}.case"
            path.write_bytes(constants + struct.pack("<4f", *wrong_values) * 64)
            cases.append((path, "mismatch"))
        pass_path = directory / f"mode_register-{len(cases)}.case"
        pass_path.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        cases.append((pass_path, "pass"))
    run_pixel_batch(renderer, backend, shader, cases)


def check_mode_register_controls():
    """s_setreg_imm32_b32 is accepted for the scratch base and for MODE[3:2] as tracked rounding state."""
    program = mode_program()
    as_nops = [("s_nop 0" if line.startswith("s_setreg_imm32_b32 ") else line) for line in program]
    assert as_nops.count("s_nop 0") == 5
    # Every accepted write here leaves the lowered result unchanged: generated names carry the instruction
    # index, so the program must translate exactly like s_nop 0 at the same positions.
    assert translate(program) == translate(as_nops)
    plain = with_constants(NARROW_BODY)
    nop = translate(["s_nop 0"] + plain)
    # The scratch base is a whole 32-bit register with any immediate; a field of it is refused.
    for value in ("0", "0x400", "-1", "0x41", "0xfffffffe", "-2147483648", "4294967295"):
        for register in ("LO", "HI"):
            assert translate([f"s_setreg_imm32_b32 hwreg(HW_REG_FLAT_SCR_{register}), {value}"] + plain) == nop
    for write in ("hwreg(HW_REG_FLAT_SCR_LO, 0, 16), 0", "hwreg(HW_REG_FLAT_SCR_HI, 16, 16), 0",
                  "hwreg(HW_REG_FLAT_SCR_LO), s48", "hwreg(HW_REG_FLAT_SCR_LO), 0x100000000",
                  "hwreg(HW_REG_FLAT_SCR_LO), 4294967296", "hwreg(HW_REG_FLAT_SCR_HI), -2147483649",
                  "hwreg(HW_REG_MODE, 0, 2), 3", "hwreg(HW_REG_MODE, 0, 2), 0", "hwreg(HW_REG_MODE, 2, 1), 0",
                  "hwreg(HW_REG_MODE, 3, 2), 0", "hwreg(HW_REG_MODE, 0, 4), 12", "hwreg(HW_REG_MODE, 4, 4), 15",
                  "hwreg(HW_REG_MODE), 0", "hwreg(HW_REG_MODE), 0x2f0", "hwreg(HW_REG_STATUS, 2, 2), 0",
                  "hwreg(HW_REG_STATUS), 0", "hwreg(HW_REG_TRAPSTS, 2, 2), 3", "hwreg(HW_REG_LDS_ALLOC), 0",
                  "hwreg(HW_REG_SHADER_CYCLES), 0", "hwreg(HW_REG_GPR_ALLOC), 0", "hwreg(HW_REG_IB_STS), 0"):
        expect_value_error(translate, [f"s_setreg_imm32_b32 {write}"] + plain, "Unsupported hardware register write")
    # The scratch base stays unobservable: s_getreg_b32 is not lowered, and flat_load_dword reads
    # only through the title ranges, so the writes translate like s_nop 0 before it.
    expect_value_error(translate, with_constants([*FLAT_SCRATCH_WRITES, "s_getreg_b32 s48, hwreg(HW_REG_FLAT_SCR_LO)"]),
                       "Unsupported instruction")
    expect_value_error(translate, with_constants([*FLAT_SCRATCH_WRITES, "flat_load_dword v12, v[12:13]"]),
                       "Title memory is not bound")
    assert (translate_title(FLAT_SCRATCH_WRITES + title_program())
            == translate_title(["s_nop 0", "s_nop 0"] + title_program()))
    # Nearest even is the entry state, so writing it first is no change. The directed modes are refused.
    assert translate([HALF_RNE] + plain) == nop
    assert translate([HALF_RTZ, HALF_RNE] + plain) == translate(["s_nop 0", "s_nop 0"] + plain)
    for value in ("1", "2", "4", "7", "-1", "0x103"):
        expect_value_error(translate, [f"s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), {value}"] + plain,
                           "Unsupported binary16 rounding mode")
    # The register forms of the same access stay unsupported.
    expect_value_error(translate, ["s_setreg_b32 hwreg(HW_REG_MODE, 2, 2), s48"] + plain, "Unsupported instruction")
    expect_value_error(translate, ["s_getreg_b32 s48, hwreg(HW_REG_MODE, 2, 2)"] + plain, "Unsupported instruction")
    # Like the hints, nothing may sit between the color export and s_endpgm.
    must_reject(translate, plain[:-1] + [HALF_RNE, plain[-1]])
    must_reject(translate, plain[:-1] + [FLAT_SCRATCH_WRITES[0], plain[-1]])
    # The write is state, not a hint: under round toward zero the nearest-even narrowing and the packed
    # binary16 arithmetic are refused by name, the last write wins, and restoring nearest even first makes
    # the same programs translate exactly as without the writes.
    expect_value_error(translate, [HALF_RTZ] + plain, "under round toward zero: v_cvt_f16_f32_e32")
    expect_value_error(translate, [HALF_RNE, HALF_RTZ] + plain, "under round toward zero: v_cvt_f16_f32_e32")
    expect_value_error(translate, [line for line in program if line != HALF_RNE],
                       "under round toward zero: v_cvt_f16_f32_e32")
    packed = with_constants(VOP3P_BODY)
    expect_value_error(translate, [HALF_RTZ] + packed, "under round toward zero: v_pk_add_f16")
    assert translate([HALF_RTZ, HALF_RNE] + packed) == translate(["s_nop 0", "s_nop 0"] + packed)
    for operation in ("v_cvt_f16_f32_e64 v12, -|s44|", "v_cvt_f16_u16_e32 v12, v12", "v_add_f16_e32 v12, v12, v13",
                      "v_cvt_f32_f64_e32 v12, v[12:13]", "v_dot2_f32_f16 v12, v12, v13, v14",
                      "v_fma_mixlo_f16 v12, v12, v13, v14", "v_interp_p2_f16 v12, v1, attr0.x"):
        expect_value_error(translate, with_constants([HALF_RTZ, operation]), "under round toward zero")
    # The exact widening and the pack that always truncates do not depend on the mode.
    assert translate([HALF_RTZ] + with_constants(HALF_BODY)) == translate(["s_nop 0"] + with_constants(HALF_BODY))


HINT_NAMES = ("s_clause", "s_setprio", "s_sleep")


def scheduling_hints():
    # Hints the compiler places between instructions. native_program ends the code at s_endpgm,
    # so s_code_end padding never reaches the translator and is not part of the fixture.
    program = assembly(False)
    program.insert(program.index("s_waitcnt vmcnt(0)"), "s_clause 0x1")
    program.insert(1, "s_setprio 2")
    program.insert(program.index("s_waitcnt vmcnt(0)"), "v_nop")
    program.insert(program.index("s_waitcnt vmcnt(0)"), "s_sleep 1")
    return program


def check_scheduling_hints():
    """Scheduling hints change timing only. Generated names carry the instruction index, so
    each hint must translate exactly like s_nop 0 at the same position."""
    plain = assembly(False)
    hinted = scheduling_hints()
    as_nops = [("s_nop 0" if line.split(" ")[0] in (*HINT_NAMES, "v_nop") else line)
               for line in hinted]
    assert translate(hinted) == translate(as_nops)
    nop = translate(["s_nop 0"] + plain)
    for hint, limit in (("s_clause", 0x3f), ("s_setprio", 3), ("s_sleep", 0x7f)):
        assert translate([f"{hint} {limit}"] + plain) == nop
        assert translate([f"{hint} {hex(limit)}"] + plain) == nop
        must_reject(translate, [f"{hint} {limit + 1}"] + plain)
    assert translate(["v_nop"] + plain) == nop
    must_reject(translate, ["v_nop 0"] + plain)
    # s_code_end stays unknown, and nothing may follow the program end or sit between the color
    # export and s_endpgm.
    must_reject(translate, ["s_code_end"] + plain)
    must_reject(translate, plain + ["s_code_end"])
    must_reject(translate, plain + ["s_nop 0"])
    must_reject(translate, plain + ["v_nop"])
    must_reject(translate, plain[:-1] + ["s_setprio 0", plain[-1]])
    must_reject(translate, plain[:-1] + ["v_nop", plain[-1]])
    check_e64_spelling_lowering()


def check_adapter_skip(directory: Path) -> None:
    compiler, renderer = "stub-dxc", "stub-offscreen"
    work = directory / "no-adapter"
    work.mkdir()

    def run_without_adapter(args: list[str | Path], *, check: bool = False) -> subprocess.CompletedProcess[None]:
        assert args[0] in (compiler, renderer)
        result = subprocess.CompletedProcess(args, 77 if args[0] == renderer else 0)
        if check:
            result.check_returncode()
        return result

    with patch.object(subprocess, "run", side_effect=run_without_adapter):
        assert check_render(work, compiler, renderer, "hardware") == 77


# Bit operations are observed as four exact byte-valued output channels. Half
# widening converts a byte's raw bits to byte * 2^-24; scaling avoids relying on
# another integer-to-float instruction under development in a separate PR.
BIT_OPERATIONS = ("v_bfe_u32", "v_bfe_i32", "v_bfm_b32", "v_bcnt_u32_b32", *(
    op + suffix for op in ("v_bfrev_b32", "v_ffbh_u32", "v_ffbh_i32", "v_ffbl_b32")
    for suffix in ("_e32", "_e64")))
BIT_CASES = (
    (0, 0, 0), (0xffffffff, 0, 0), (0x80000000, 31, 1),
    (0x7fffffff, 28, 31), (0xdeadbeef, 4, 8), (0x89abcdef, 36, 40),
    (0xfedcba98, 63, 31), (0xffffffff, 0xffffffff, 0xffffffff),
    (0x80000001, 1, 31), (0x12345678, 16, 16), (0x01010101, 63, 32),
    (0x7f00ff00, 8, 9), (1, 0xffffffff, 33), (0xfffffffe, 32, 63),
    (0x40000000, 2, 30), (0x00010000, 31, 31),
)


def bit_program(op):
    base = op.removesuffix("_e32").removesuffix("_e64")
    arity = pixel_aot.INTEGER[base[2:]][0]
    body = [f"v_mov_b32_e32 v{20+i}, s{44+i}" for i in range(3)]
    body.append(f"{op} v18, " + ", ".join(f"v{20+i}" for i in range(arity)))
    for channel in range(4):
        dst = f"v{12+channel}"
        body += [f"v_bfe_u32 {dst}, v18, {8*channel}, 8",
                 f"v_cvt_f32_f16_e32 {dst}, {dst}",
                 f"v_mul_f32_e32 {dst}, 0x4b800000, {dst}"]
    return with_constants(body)


def bit_reference(op, a, b, c):
    op = op.removesuffix("_e32").removesuffix("_e64")
    if op in ("v_bfe_u32", "v_bfe_i32"):
        offset, width = b % 32, c % 32
        if not width:
            return 0
        field = 0
        for i in range(width):
            pos = offset + i
            bit = (a >> pos) & 1 if pos < 32 else (a >> 31 if op == "v_bfe_i32" else 0)
            field |= bit << i
        if op == "v_bfe_i32" and field & (1 << (width-1)):
            field -= 1 << width
        return field & 0xffffffff
    if op == "v_bfm_b32":
        return sum(1 << i for i in range(b % 32, b % 32 + a % 32) if i < 32)
    if op == "v_bcnt_u32_b32":
        return (sum((a >> i) & 1 for i in range(32)) + b) & 0xffffffff
    if op == "v_bfrev_b32":
        return sum(((a >> i) & 1) << (31-i) for i in range(32))
    positions = range(32) if op == "v_ffbl_b32" else range(31, -1, -1)
    sought = 0 if op == "v_ffbh_i32" and a >> 31 else 1
    for i in positions:
        if ((a >> i) & 1) == sought:
            return i if op == "v_ffbl_b32" else 31-i
    return 0xffffffff


def check_bit_controls():
    # Acceptance controls run in every test mode. Only the binary/draw fixtures
    # below record these forms, so an unrelated successful draw cannot cover them.
    from scan_unsupported_shaders import is_instruction_supported

    def validate(program):
        return pixel_aot.lower(program, 20, 24, [1, 2, 3, 4], constant_descriptor=64)
    for op in BIT_OPERATIONS:
        program = bit_program(op)
        source = validate(program)
        assert "asfloat(" in source
        for bad in (op + "_sdwa", op + "_dpp"):
            must_reject(validate, [line.replace(op + " v18", bad + " v18") for line in program])
        for old, replacement in (("v18, v20", "v18, -v20"),
                                 ("v18, v20", "v18, |v20|"),
                                 ("v18, v20", "v18, v0")):
            must_reject(validate, [line.replace(old, replacement) if line.startswith(op + " ") else line
                                    for line in program])
        must_reject(validate, [line + " clamp" if line.startswith(op + " ") else line for line in program])
        must_reject(validate, [line.replace("s_waitcnt lgkmcnt(0)", "s_nop 0") for line in program])
        wait = program.index("s_waitcnt vmcnt(0)")
        operation = next(line for line in program if line.startswith(op + " "))
        assert is_instruction_supported(operation), operation
        must_reject(validate, program[:wait] + [operation.replace("v18,", "v13,")] + program[wait:])
        must_reject(validate, program[:wait] + [operation.replace("v20", "v12")] + program[wait:])
        wqm = program.index("s_and_b64 exec, exec, s[50:51]")
        must_reject(validate, program[:wqm] + [operation] + program[wqm:])
    # Masked lane counts require wave state, which these lane-local operations do not introduce.
    must_reject(validate, with_constants(["v_mbcnt_lo_u32_b32 v12, 0, 0"]))
    for invalid in ("v_bfe_u32_e32 v12, 0, 0, 0", "v_bfm_b32_e32 v12, 0, 0",
                    "v_bcnt_u32_b32_e32 v12, 0, 0"):
        must_reject(validate, with_constants([invalid]))
        assert not is_instruction_supported(invalid), invalid
    assert bit_reference("v_ffbh_i32", 0x80000000, 0, 0) == 1
    assert bit_reference("v_ffbh_i32", 0xffffffff, 0, 0) == 0xffffffff
    assert bit_reference("v_bfe_i32", 0x80000000, 31, 31) == 0xffffffff


def check_bit_draws(directory, compiler, renderer, backend):
    source, shader = (directory / name for name in ("bits.hlsl", "bits.cso"))
    for op in BIT_OPERATIONS:
        source.write_text(translate(bit_program(op)), encoding="ascii")
        subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
        cases = []
        for index, (a, b, c) in enumerate(BIT_CASES):
            constants = bytearray(256)
            struct.pack_into("<3I", constants, 12, a, b, c)
            result = bit_reference(op, a, b, c)
            rgba = tuple(float((result >> (8 * i)) & 255) for i in range(4))
            assert all(half_rtz(x) == x for x in rgba)
            path = write_case(directory / f"bits_{index}.case", constants, rgba)
            cases.append((path, "pass"))
        # Reuse the last draw and flip one byte in its reference.
        wrong = (float(int(rgba[0]) ^ 1), *rgba[1:])
        path = write_case(directory / "bits_wrong.case", constants, wrong)
        cases.append((path, "mismatch"))
        batch = run_pixel_batch(renderer, backend, shader, cases)
        if "pixel_mismatch x=0 y=0" not in batch.stderr:
            raise AssertionError(f"{op}: wrong reference did not report the first changed pixel")
        if op == BIT_OPERATIONS[0]:
            for line in batch.stdout.splitlines():
                if line.startswith(("backend=", "adapter=", "driver=")):
                    print(line)
        print(f"vector_bits={op} cases={len(BIT_CASES)} matched", flush=True)
    return 0


def attribute_color(rg):
    code = ["s_mov_b32 s30, s14", "s_mov_b32 m0, s30"]
    for phase in (1, 2):
        for i, channel in enumerate("wzxy"):
            code.append(f"v_interp_p{phase}_f32_e32 v{8+i}, v{phase-1}, attr0.{channel}")
    code.extend(["v_cvt_pkrtz_f16_f32_e32 v12, v8, v9", "v_cvt_pkrtz_f16_f32_e32 v13, v10, v11"])
    tail = "off, off" if rg else "v13, v13"
    return code + [f"exp mrt0 v12, v12, {tail} done compr vm", "s_endpgm"]


def translate_attribute(code):
    return lower(code, 8, 14, None)


# VOP3 (e64) spellings left over after the other e64 draws: v_interp_p1_f32_e64, v_interp_p2_f32_e64 and
# v_nop_e64. LLVM defines each as a second encoding of the instruction already lowered, and the translator
# accepts the interpolation pair only in the operand shape of the e32 form: a plain VGPR source and attr0, with
# no source modifier, clamp or output modifier. e64_spellings rewrites the interpolation lines of a drawn
# program and adds a v_nop_e64 at entry mask, so the translated shader must equal the e32 one.
E64_SPELLING_OPS = ("v_interp_p1_f32_e64", "v_interp_p2_f32_e64", "v_nop_e64")


def e64_spellings(program, *, phases=("p1", "p2"), nop_before=None):
    """program with its v_interp_<phase>_f32 lines in the VOP3 spelling and one v_nop_e64 inserted.

    The v_nop_e64 goes before the line nop_before, or before the first export when that is None.
    """
    spelled = [line.replace("_f32_e32", "_f32_e64") if line.startswith(tuple(f"v_interp_{p}_f32_e32" for p in phases))
               else line for line in program]
    at = (spelled.index(nop_before) if nop_before is not None
          else next(i for i, line in enumerate(spelled) if line.startswith("v_cvt_pkrtz")))
    return spelled[:at] + ["v_nop_e64"] + spelled[at:]


def e64_spelling_programs():
    """(name, program, translate function, the same program with e32 interpolation and s_nop 0)."""
    programs = []
    # Whole-quad interpolation of the texture coordinates. The second program mixes the encodings in each
    # pair: e32 for p1 and e64 for p2.
    for swap, phases in ((False, ("p1", "p2")), (True, ("p2",))):
        program = e64_spellings(assembly(swap), phases=phases, nop_before="s_waitcnt vmcnt(0)")
        programs.append((f"sampled-{'bgra' if swap else 'rgba'}", program, translate))
    # Entry-mask interpolation of all four components.
    programs.append(("attribute", e64_spellings(attribute_color(False)), translate_attribute))
    return [(name, program, function,
             [line.replace("_f32_e64", "_f32_e32") if line.startswith("v_interp_") else
              "s_nop 0" if line == "v_nop_e64" else line for line in program])
            for name, program, function in programs]


def check_e64_spelling_lowering():
    """Static checks: each e64 spelling lowers exactly like the e32 form, and nothing wider is accepted."""
    for name, program, function, plain in e64_spelling_programs():
        assert sum(line.startswith("v_interp_") and "_e64 " in line for line in program) in (2, 4, 8), name
        assert program.count("v_nop_e64") == 1, name
        assert function(program) == function(plain), name
    assert set(E64_SPELLING_OPS) <= ACCEPTED, set(E64_SPELLING_OPS) - ACCEPTED
    # The other mix, e64 for p1 and e32 for p2, is the same pair as well.
    assert (translate_attribute(e64_spellings(attribute_color(True), phases=("p1",)))
            == translate_attribute(e64_spellings(attribute_color(True), phases=())))
    # The VOP3 encoding has source modifiers, clamp and output modifiers that the e32 one lacks. None is
    # lowered, and neither is an SGPR source, another attribute, or an operand on v_nop_e64.
    sampled = e64_spellings(assembly(False), nop_before="s_waitcnt vmcnt(0)")
    attribute = e64_spellings(attribute_color(False))
    first, second = "v_interp_p1_f32_e64 v8, v0, attr0.x", "v_interp_p2_f32_e64 v8, v1, attr0.x"
    assert first in sampled and second in sampled
    for old, new in ((first, "v_interp_p1_f32_e64 v8, -v0, attr0.x"),
                     (first, "v_interp_p1_f32_e64 v8, |v0|, attr0.x"),
                     (first, "v_interp_p1_f32_e64 v8, -|v0|, attr0.x"),
                     (first, first + " clamp"),
                     (first, first + " mul:2"),
                     (first, "v_interp_p1_f32_e64 v8, s0, attr0.x"),
                     (first, "v_interp_p1_f32_e64 v8, v0, attr1.x"),
                     (first, "v_interp_p1_f32_e64 v8, v0"),
                     (second, "v_interp_p2_f32_e64 v8, -v1, attr0.x"),
                     (second, second + " clamp"),
                     (second, second + " div:2"),
                     (second, "v_interp_p2_f32_e64 v8, v1, attr1.x"),
                     # Pairing rules are those of the e32 form: p2 needs the j barycentric and its own p1.
                     (second, "v_interp_p2_f32_e64 v8, v0, attr0.x"),
                     (second, "v_interp_p2_f32_e64 v8, v1, attr0.y"),
                     ("v_nop_e64", "v_nop_e64 0"),
                     ("v_nop_e64", "v_nop_e64 v0")):
        assert old in sampled, old
        must_reject(translate, [new if line == old else line for line in sampled])
    must_reject(translate_attribute, [line.replace("v8, v1, attr0.w", "v8, v1, attr0.z") for line in attribute])
    must_reject(translate_attribute, [line.replace("attr0.z", "attr1.z") for line in attribute])
    # v_nop_e64 follows v_nop: refused after the export and under the whole-quad mask.
    plain = assembly(False)
    must_reject(translate, plain + ["v_nop_e64"])
    must_reject(translate, plain[:-1] + ["v_nop_e64", plain[-1]])
    must_reject(translate, plain[:3] + ["v_nop_e64"] + plain[3:])
    must_reject(translate, plain[:3] + ["v_nop"] + plain[3:])


def check_e64_spelling_draws(directory, compiler, renderer, backend):
    """Draw the three programs of e64_spelling_programs against the references their e32 forms use.

    The sampled programs use the renderer's texture reference, one per channel order; each checked against
    the other order must report a mismatch. The attribute program uses a per-pixel CPU reference for all four
    interpolated components; a zeroed channel at an interior pixel must report a mismatch there.
    A renderer that reports an unavailable adapter (exit 77) makes this a skip. On a desktop GPU the draws
    establish what the translated shader computes, not what the console does.
    """
    source = directory / "e64_spelling.hlsl"
    shader = directory / "e64_spelling.cso"
    for name, program, function, _ in e64_spelling_programs():
        source.write_text(function(program), encoding="ascii")
        if name.startswith("sampled"):
            compile_shader(compiler, source, shader, ieee=False)
            order = name.split("-")[1]
            draw = [renderer, backend, "--check-pixel", shader]
            result = subprocess.run([*draw, order])
            if result.returncode:
                return result.returncode
            other = subprocess.run([*draw, "rgba" if order == "bgra" else "bgra"], capture_output=True, text=True)
            if other.returncode != 1 or "pixel_mismatch" not in other.stderr:
                raise AssertionError(f"{name}: the other channel order escaped detection")
            continue
        subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)
        case = directory / "e64_spelling.case"
        draw = [renderer, backend, "--check-pixel-attribute", shader, case]
        data = bytearray(256)
        for y in range(32):
            for x in range(32):
                u, v = (x + 0.5) / 32, (y + 0.5) / 32
                data += struct.pack("<4f", 1 - v, 0.25 + 0.5 * u, u, v)
        case.write_bytes(data)
        result = subprocess.run(draw)
        if result.returncode:
            return result.returncode
        # One interior pixel per channel: attr0.w, attr0.z, attr0.x and attr0.y.
        for channel, (x, y) in enumerate(((17, 9), (5, 22), (26, 3), (11, 30))):
            wrong = bytearray(data)
            struct.pack_into("<f", wrong, 256 + (y * 32 + x) * 16 + channel * 4, 0)
            case.write_bytes(wrong)
            mismatch = subprocess.run(draw, capture_output=True, text=True)
            if mismatch.returncode != 1 or f"pixel_mismatch x={x} y={y}" not in mismatch.stderr:
                raise AssertionError(f"{name}: a wrong interpolated channel {channel} escaped detection")
    return 0


def buffer_color(rg):
    # Use a copied low table address and an explicit zero high word.
    tail = "off, off" if rg else "v26, v26"
    return [
        "s_mov_b32 s30, s9", "s_mov_b32 s31, 0",
        "s_load_dwordx4 s[12:15], s[30:31], null", "s_waitcnt lgkmcnt(0)",
        "s_buffer_load_dwordx4 s[4:7], s[12:15], 0x10", "s_waitcnt lgkmcnt(0)",
        "v_mov_b32_e32 v20, s7", "v_mov_b32_e32 v21, s4",
        "v_cvt_pkrtz_f16_f32_e32 v25, v20, v21",
        "v_cvt_pkrtz_f16_f32_e32 v26, v21, v20",
        f"exp mrt0 v25, v25, {tail} done compr vm", "s_endpgm",
    ]


def unsigned_conversion(quad):
    prefix = buffer_color(False)[:6]  # word at byte 16 is in s4 after the wait
    if quad:
        prefix += ["s_mov_b64 s[40:41], exec", "s_wqm_b64 exec, exec",
                   "v_cvt_u32_f32_e32 v32, s4", "s_and_b64 exec, exec, s[40:41]"]
    else:
        prefix += ["v_mov_b32_e32 v32, s4", "v_cvt_u32_f32_e32 v32, v32"]
    # Every possible byte is an exact binary16 and binary32 value.
    return prefix + [f"v_cvt_f32_ubyte{i}_e32 v{20+i}, v32" for i in range(4)] + [
        "v_cvt_pkrtz_f16_f32_e32 v24, v20, v21", "v_cvt_pkrtz_f16_f32_e32 v25, v22, v23",
        "exp mrt0 v24, v24, v25, v25 done compr vm", "s_endpgm",
    ]


def translate_buffer_color(instructions):
    return lower(instructions, 9, 24, None, 0)


def three_way_case(directory, low, high, cap, middle, *, name="three_way.case"):
    """Expected colors for three_way(); no rounding occurs before the half export."""
    case = bytearray(256)
    struct.pack_into("<4f", case, 12, low, high, cap, middle)
    for y in RAMP:
        for x in RAMP:
            r, g, b = single(x / 255), single(y / 255), single(128 / 255)
            clamped = sorted((r, low, high))[1]
            color = (clamped, min(abs(cap), g, clamped), max(r, b, g), sorted((middle, g, r))[1])
            case.extend(struct.pack("<4f", *map(half_rtz, color)))
    path = directory / name
    path.write_bytes(case)
    return path


def check_three_way_templates():
    """The HLSL forms must pick the expected operand for every operand order."""
    values = (-1.5, 0.0, 0.25, 0.25, 3.0)
    for a in values:
        for b in values:
            for c in values:
                pick = {name: eval(ALU[name][1].format("a", "b", "c"), {"min": min, "max": max},
                                   {"a": a, "b": b, "c": c}) for name in ("min3", "max3", "med3")}
                assert pick == {"min3": min(a, b, c), "max3": max(a, b, c),
                                "med3": sorted((a, b, c))[1]}, (a, b, c)
# Quad-local DPP16 and DPP8 integer/bitwise/shift/bit-scan fixture covering 23 VOP1/VOP2 DPP forms.
# Derives integer quad lane ID {0, 1, 2, 3} from interpolated uv.x/uv.y (v8/v9), verifies quad
# swizzles across lanes, folds all 23 ops into a 32-bit accumulator, and exports all 4 bytes as
# exact [0, 255] integers.
DPP_INT32_BODY = [
    "v_mov_b32_e32 v17, 0",
    "v_or_b32_dpp v18, v8, v17 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf",
    "v_or_b32_dpp v19, v9, v17 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf",
    "v_or_b32_dpp v20, v8, v17 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_or_b32_dpp v21, v9, v17 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_sub_f32_e32 v18, v20, v18",
    "v_sub_f32_e32 v19, v21, v19",
    "v_mul_f32_e32 v18, 0x42800000, v18",
    "v_mul_f32_e32 v19, 0x43000000, v19",
    "v_add_f32_e32 v20, v18, v19",
    "v_add_f32_e32 v20, 0x4b000000, v20",
    "v_mov_b32_e32 v17, 3",
    "v_and_b32_dpp v20, v20, v17 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_sub_nc_u32_dpp v21, v20, v20 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
    "v_mul_i32_i24_dpp v21, v21, v21 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_subrev_nc_u32_dpp v22, v20, v20 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf",
    "v_mul_i32_i24_dpp v22, v22, v22 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_add_nc_u32_dpp v23, v20, v20 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf",
    "v_sub_nc_u32_dpp v24, v20, v20 dpp8:[1,2,3,0,5,6,7,4]",
    "v_and_b32_dpp v24, v24, v17 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_mov_b32_e32 v25, s44",
    "v_mov_b32_e32 v26, s45",
    "v_mov_b32_e32 v27, s46",
    "v_mov_b32_e32 v28, s47",
    "v_min_i32_dpp v29, v25, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
    "v_max_i32_dpp v30, v25, v26 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf",
    "v_min_u32_dpp v31, v25, v26 dpp8:[3,2,1,0,7,6,5,4]",
    "v_max_u32_dpp v32, v25, v26 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf",
    "v_mul_u32_u24_dpp v33, v25, v26 dpp8:[1,0,3,2,5,4,7,6]",
    "v_mul_hi_u32_u24_dpp v34, v25, v26 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
    "v_mul_hi_i32_i24_dpp v35, v25, v26 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf",
    "v_xor_b32_dpp v36, v29, v30 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
    "v_xnor_b32_dpp v37, v31, v32 dpp8:[2,3,0,1,6,7,4,5]",
    "v_not_b32_dpp v38, v27 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf",
    "v_lshlrev_b32_dpp v39, v28, v25 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
    "v_lshrrev_b32_dpp v40, v28, v26 dpp8:[1,0,3,2,5,4,7,6]",
    "v_ashrrev_i32_dpp v41, v28, v25 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf",
    "v_bfrev_b32_dpp v42, v27 dpp8:[3,2,1,0,7,6,5,4]",
    "v_ffbl_b32_dpp v43, v27 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
    "v_ffbh_u32_dpp v44, v27 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf",
    "v_ffbh_i32_dpp v45, v25 dpp8:[2,3,0,1,6,7,4,5]",
    "v_add_nc_u32_dpp v30, v21, v22 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_add_nc_u32_dpp v30, v30, v23 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_add_nc_u32_dpp v30, v30, v24 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_xor_b32_dpp v30, v30, v33 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_add_nc_u32_dpp v30, v30, v34 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_xor_b32_dpp v30, v30, v35 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_add_nc_u32_dpp v30, v30, v36 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_xor_b32_dpp v30, v30, v37 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_add_nc_u32_dpp v30, v30, v38 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_xor_b32_dpp v30, v30, v39 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_add_nc_u32_dpp v30, v30, v40 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_xor_b32_dpp v30, v30, v41 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_add_nc_u32_dpp v30, v30, v42 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_xor_b32_dpp v30, v30, v43 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_add_nc_u32_dpp v30, v30, v44 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_xor_b32_dpp v30, v30, v45 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_mov_b32_e32 v46, 0xff",
    "v_mov_b32_e32 v47, 8",
    "v_mov_b32_e32 v48, 16",
    "v_mov_b32_e32 v49, 24",
    "v_and_b32_dpp v50, v30, v46 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_lshrrev_b32_dpp v51, v47, v30 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_and_b32_dpp v51, v51, v46 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_lshrrev_b32_dpp v52, v48, v30 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_and_b32_dpp v52, v52, v46 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_lshrrev_b32_dpp v53, v49, v30 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_cvt_f32_f16_e32 v12, v50",
    "v_mul_f32_e32 v12, 0x4b800000, v12",
    "v_cvt_f32_f16_e32 v13, v51",
    "v_mul_f32_e32 v13, 0x4b800000, v13",
    "v_cvt_f32_f16_e32 v14, v52",
    "v_mul_f32_e32 v14, 0x4b800000, v14",
    "v_cvt_f32_f16_e32 v15, v53",
    "v_mul_f32_e32 v15, 0x4b800000, v15",
]

DPP_INT32_SETS = (
    # Set 1: negative i24/i32 in s44, positive in s45, mixed bits in s46, shift >= 32 (37 -> 5).
    (0xff801234, 0x007f5678, 0x00108040, 37),
    # Set 2: extreme i24 (-2^23 * 2^23-1), zero input for ffbl/ffbh (-1), all-ones sign for ffbh_i32.
    (0xffffffff, 0x007fffff, 0x00000000, 13),
    # Set 3: negative * negative i24 (-2^23 * -2^23 -> +0x4000 high), MSB=1 for ffbh_u32=0.
    (0x00800000, 0x00800000, 0x80000001, 31),
)


def dpp_int32_reference(a, b, c, shift):
    m32 = 0xffffffff

    def i32(x):
        x &= m32
        return x - 0x100000000 if x & 0x80000000 else x

    def i24(x):
        x &= 0xffffff
        return x - 0x1000000 if x & 0x800000 else x

    def ffbl(x):
        x &= m32
        return m32 if x == 0 else (x & -x).bit_length() - 1

    def ffbh_u(x):
        x &= m32
        return m32 if x == 0 else 32 - x.bit_length()

    def ffbh_i(x):
        x &= m32
        m = (~x & m32) if (x & 0x80000000) else x
        return m32 if m == 0 else 32 - m.bit_length()

    s = shift & 31
    v21, v22, v23, v24 = 1, 4, 3, 1
    v29 = min(i32(a), i32(b)) & m32
    v30_max = max(i32(a), i32(b)) & m32
    v31 = min(a & m32, b & m32)
    v32 = max(a & m32, b & m32)
    v33 = ((a & 0xffffff) * (b & 0xffffff)) & m32
    v34 = (((a & 0xffffff) * (b & 0xffffff)) >> 32) & m32
    v35 = ((i24(a) * i24(b)) >> 32) & m32
    v36 = (v29 ^ v30_max) & m32
    v37 = (~(v31 ^ v32)) & m32
    v38 = (~c) & m32
    v39 = ((a & m32) << s) & m32
    v40 = (b & m32) >> s
    v41 = (i32(a) >> s) & m32
    v42 = int(f"{c & m32:032b}"[::-1], 2)
    v43 = ffbl(c)
    v44 = ffbh_u(c)
    v45 = ffbh_i(a)

    acc = (v21 + v22 + v23 + v24) & m32
    acc = (acc ^ v33) & m32
    acc = (acc + v34) & m32
    acc = (acc ^ v35) & m32
    acc = (acc + v36) & m32
    acc = (acc ^ v37) & m32
    acc = (acc + v38) & m32
    acc = (acc ^ v39) & m32
    acc = (acc + v40) & m32
    acc = (acc ^ v41) & m32
    acc = (acc + v42) & m32
    acc = (acc ^ v43) & m32
    acc = (acc + v44) & m32
    acc = (acc ^ v45) & m32
    return tuple(float((acc >> (8 * k)) & 0xff) for k in range(4))


def ds_swizzle_program():
    # Keep interpolation and the DS instructions inside the verified whole-quad region.
    program = assembly(False)[:7]
    for op in (
        "ds_swizzle_b32 v12, v8 offset:swizzle(QUAD_PERM,2,0,3,1)",
        "ds_swizzle_b32 v13, v9 offset:swizzle(QUAD_PERM,2,0,3,1)",
        "ds_swizzle_b32 v14, v8 offset:swizzle(QUAD_PERM,1,0,3,2)",
        "ds_swizzle_b32 v15, v9 offset:swizzle(QUAD_PERM,2,2,2,2)",
    ):
        program.extend((op, "s_waitcnt lgkmcnt(0)"))
    return program + ["s_and_b64 exec, exec, s[50:51]", "exp mrt0 v12, v13, v14, v15 done vm", "s_endpgm"]


def translate_ds_swizzle(program):
    return lower(program, 20, 24, None, integer_export=True, wave_size=64)


def check_ds_swizzle_controls():
    program = ds_swizzle_program()
    source = translate_ds_swizzle(program)
    assert "QuadReadAcrossX" in source and "QuadReadLaneAt" in source
    i = next(i for i, line in enumerate(program) if line.startswith("ds_swizzle_b32 "))
    op = program[i]
    for wrong in (op + " gds", op.replace("2,0,3,1", "4,0,3,1"),
                  op.replace("swizzle(QUAD_PERM,2,0,3,1)", "swizzle(SWAP,16)"),
                  op.replace("v8", "v255"), op.replace("v12", "v256")):
        must_reject(translate_ds_swizzle, program[:i] + [wrong] + program[i + 1:])
    for wait in ([], ["s_waitcnt lgkmcnt(1)"], ["s_waitcnt vmcnt(0)"]):
        expect_value_error(translate_ds_swizzle, program[:i + 1] + wait + program[i + 2:],
                           "ds_swizzle_b32 requires an immediate s_waitcnt lgkmcnt(0)")
    restore = program[-3]
    must_reject(translate_ds_swizzle, program[:i] + [restore] + program[i:-3] + program[-2:])


def ds_nop_program():
    program = ds_swizzle_program()
    # Exercise entry EXEC and whole-quad EXEC, with each DS swizzle wait intact.
    return ["ds_nop"] + program[:7] + ["ds_nop"] + program[7:]


def check_ds_nop_controls():
    base = ds_swizzle_program()
    program = ds_nop_program()
    # Keep the instruction indices used in generated HLSL temporary names.
    control = ["s_nop 0" if line == "ds_nop" else line for line in program]
    assert translate_ds_swizzle(program) == translate_ds_swizzle(control)
    for invalid in ("ds_nop gds", "ds_nop offset:1", "ds_nop v0"):
        must_reject(translate_ds_swizzle, [invalid] + base)
    expect_value_error(translate_ds_swizzle, base[:-1] + ["ds_nop", base[-1]],
                       "Unexpected instruction after color export")
    expect_value_error(translate_ds_swizzle, base + ["ds_nop"], "Instruction follows program end")
    wait = base.index("s_waitcnt lgkmcnt(0)")
    expect_value_error(translate_ds_swizzle, base[:wait] + ["ds_nop"] + base[wait:],
                       "ds_swizzle_b32 requires an immediate s_waitcnt lgkmcnt(0)")


def check_ds_swizzle_draw(directory, compiler, renderer, backend):
    program = ds_swizzle_program()
    source, shader = directory / "ds_swizzle.hlsl", directory / "ds_swizzle.cso"
    case = directory / "ds_swizzle.case"
    pixels = []
    for y in range(8):
        for x in range(8):
            # Pixel centers for the oversized triangle in pixel_aot_vs.hlsl.
            lane = (y % 2) * 2 + x % 2
            shuffled = (2, 0, 3, 1)[lane]
            quad_x, quad_y = x & ~1, y & ~1
            floats = ((quad_x + shuffled % 2 + 0.5) / 8,
                      (quad_y + shuffled // 2 + 0.5) / 8,
                      ((x ^ 1) + 0.5) / 8,
                      (quad_y + 1.5) / 8)
            pixels.append(floats)
    # Existing case trailer bits select UINT output and an explicit 8x8 viewport.
    case.write_bytes(bytes(256) + b"".join(struct.pack("<4f", *pixel) for pixel in pixels)
                     + struct.pack("<B4I", 3, 0, 0, 8, 8))
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    for variant in (program, ds_nop_program()):
        source.write_text(translate_ds_swizzle(variant), encoding="ascii")
        compile_shader(compiler, source, shader)
        subprocess.run(draw, check=True)

    # Remove the horizontal swap from only the blue channel.
    wrong = [line.replace("QUAD_PERM,1,0,3,2", "QUAD_PERM,0,1,2,3") for line in program]
    source.write_text(translate_ds_swizzle(wrong), encoding="ascii")
    compile_shader(compiler, source, shader)
    result = subprocess.run(draw, capture_output=True, text=True)
    if (result.returncode != 1 or "pixel_mismatch x=0 y=0" not in result.stderr
            or "channel=2 expected=0.1875 actual=0.0625" not in result.stderr):
        raise AssertionError(f"Expected the identity mutation to differ at (0, 0): {result.stderr}")
    print(result.stderr, end="")


def check_dpp_int32_draw(directory, compiler, renderer, backend):
    program = with_constants(DPP_INT32_BODY)
    source = directory / "dpp_int32.hlsl"
    shader = directory / "dpp_int32.cso"
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    cases = []
    for idx, values in enumerate(DPP_INT32_SETS):
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *values)
        expected = dpp_int32_reference(*values)
        case = directory / f"dpp_int32-{idx}.case"
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        cases.append((case, "pass"))
    run_pixel_batch(renderer, backend, shader, cases)
    for set_idx, old, new, label in (
        (0, "v_sub_nc_u32_dpp v21, v20, v20 quad_perm:[1,0,3,2]",
         "v_sub_nc_u32_dpp v21, v20, v20 quad_perm:[2,3,0,1]", "AcrossX vs AcrossY"),
        (0, "dpp8:[1,2,3,0,5,6,7,4]", "dpp8:[3,0,1,2,7,4,5,6]", "cyclic dpp8 rotation"),
        (0, "v_ashrrev_i32_dpp v41, v28, v25", "v_lshrrev_b32_dpp v41, v28, v25", "ashrrev vs lshrrev"),
        (0, "v_mul_hi_i32_i24_dpp v35", "v_mul_hi_u32_u24_dpp v35", "mul_hi signed vs unsigned"),
        (0, "v_ffbh_i32_dpp v45, v25", "v_ffbh_u32_dpp v45, v25", "ffbh signed vs unsigned"),
    ):
        mutated = [line.replace(old, new) for line in program]
        assert mutated != program, label
        source.write_text(translate(mutated), encoding="ascii")
        compile_shader(compiler, source, shader)
        wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, cases[set_idx][0]],
                               capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Mutation of {label} escaped detection")


# Quad-local DPP16 (quad_perm) and DPP8 (dpp8) fixture covering all 20 supported VOP1/VOP2 DPP forms.
# Derives per-quad lane ID {0, 1, 2, 3} from interpolated uv.x (v8) and uv.y (v9) so every 2x2 quad
# carries 4 distinct values across all 4,096 pixels while exporting uniform exact binary16 results.
DPP_BODY = [
    "v_subrev_f32_dpp v18, v8, v8 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf",
    "v_subrev_f32_dpp v19, v9, v9 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf",
    "v_mul_f32_e32 v18, 0x42800000, v18",
    "v_fma_f32 v20, 0x43000000, v19, v18",
    "v_sub_f32_dpp v21, v20, v20 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
    "v_add_f32_dpp v21, |v21|, |v21| quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_sub_f32_dpp v22, v20, v20 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf",
    "v_max_f32_dpp v22, |v22|, v22 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf",
    "v_add_f32_dpp v23, v20, v20 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf",
    "v_mov_b32_dpp v24, v20 dpp8:[1,2,3,0,5,6,7,4]",
    "v_sub_f32_e32 v25, v24, v20",
    "v_add_f32_e32 v26, 4.0, v25",
    "v_cmp_gt_f32_e32 vcc_lo, 0, v25",
    "v_cndmask_b32_dpp v27, v25, v26, vcc_lo quad_perm:[1,2,0,3] row_mask:0xf bank_mask:0xf",
    "v_sub_f32_dpp v49, v20, v20 quad_perm:[1,2,3,2] row_mask:0xf bank_mask:0xf",
    "v_max_f32_dpp v50, |v49|, v49 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
    "v_add_f32_e32 v12, v21, v22",
    "v_add_f32_e32 v12, v12, v23",
    "v_add_f32_e32 v12, v12, v27",
    "v_add_f32_e32 v12, v12, v50",
    # DPP16 neg on src1: v23 is 3 in every lane, so v45 = 3 + -3 = 0 and leaves channel 0 unchanged;
    # a dropped src1 neg gives 6.
    "v_add_f32_dpp v45, v23, -v23 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf",
    "v_add_f32_e32 v12, v12, v45",
    "v_add_f32_e32 v12, s44, v12",
    "v_mov_b32_e32 v28, s45",
    "v_mov_b32_e32 v29, s46",
    "v_mov_b32_e32 v30, s47",
    "v_mul_f32_dpp v31, v20, v28 quad_perm:[3,3,3,3] row_mask:0xf bank_mask:0xf",
    "v_mul_legacy_f32_dpp v32, v20, v28 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf",
    "v_min_f32_dpp v33, -v31, v30 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
    "v_cvt_f32_f16_dpp v34, -|v29| quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf",
    "v_mov_b32_e32 v13, v32",
    "v_fmac_f32_dpp v13, v33, v34 dpp8:[1,0,3,2,5,4,7,6]",
    "v_cvt_f32_f16_dpp v51, v29 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
    "v_add_f32_e32 v13, v13, v51",
    "v_add_f32_e32 v35, s44, v30",
    "v_floor_f32_dpp v36, v35 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
    "v_ceil_f32_dpp v37, v35 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf",
    "v_trunc_f32_dpp v38, v35 dpp8:[3,2,1,0,7,6,5,4]",
    "v_fract_f32_dpp v39, v35 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf",
    "v_fma_f32 v14, 2.0, v37, v36",
    "v_fma_f32 v14, 4.0, v38, v14",
    "v_add_f32_e32 v14, v14, v39",
    "v_log_f32_dpp v40, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
    "v_exp_f32_dpp v41, v40 dpp8:[2,3,0,1,6,7,4,5]",
    "v_rcp_f32_dpp v42, v41 quad_perm:[3,2,1,0] row_mask:0xf bank_mask:0xf",
    "v_sqrt_f32_dpp v43, v28 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
    "v_rsq_f32_dpp v44, v28 dpp8:[3,2,1,0,7,6,5,4]",
    "v_add_f32_e32 v15, v40, v41",
    "v_fma_f32 v15, 0x41800000, v42, v15",
    "v_add_f32_e32 v15, v15, v43",
    "v_fma_f32 v15, 4.0, v44, v15",
    "v_exp_f32_dpp v52, v30 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf",
    "v_fma_f32 v15, 4.0, v52, v15",
    "v_add_f32_e32 v15, 0.5, v15",
    "v_floor_f32_e32 v15, v15",
]

DPP_SETS = (
    # (a: f32, b: power-of-2 f32, h_bits: packed f16 in low half, c: f32)
    (0.25, 4.0, 0x4000, 1.5),
    (-2.25, 16.0, 0xbe00, 0.5),
)


def dpp_reference(a, b, h_bits, c):
    h = struct.unpack("<e", struct.pack("<H", h_bits & 0xffff))[0]
    # quad_perm:[1,2,3,2] read of the lane id, minus the lane id, through max(|s|, s): 1 in every lane.
    perm = (1, 2, 3, 2)
    quad_terms = {max(abs(perm[lane] - lane), perm[lane] - lane) for lane in range(4)}
    assert len(quad_terms) == 1, quad_terms
    ch0 = 8.0 + a + quad_terms.pop()
    ch1 = 3.0 * b * abs(h) + h
    x = a + c
    ch2 = math.floor(x) + 2.0 * math.ceil(x) + 4.0 * math.trunc(x) + (x - math.floor(x))
    log_b = math.log2(b)
    ch3 = math.floor(log_b + (2.0 ** log_b) + 16.0 / (2.0 ** log_b) + math.sqrt(b) + 4.0 / math.sqrt(b)
                     + 4.0 * (2.0 ** c) + 0.5)
    return (half_rtz(ch0), half_rtz(ch1), half_rtz(ch2), half_rtz(ch3))


def check_dpp_draw(directory, compiler, renderer, backend):
    program = with_constants(DPP_BODY)
    source = directory / "dpp.hlsl"
    shader = directory / "dpp.cso"
    source.write_text(translate(program), encoding="ascii")
    # The NVIDIA driver (32.0.16.2101, RTX 5080) refuses PSO creation for the dpp program
    # under -denorm preserve (E_OUTOFMEMORY), matching the recorded curve exception. Its rows
    # contain no binary32 subnormals, so record this fixture's mode as any and assert that
    # flushing leaves every expected word unchanged.
    compile_shader(compiler, source, shader, mode="any")
    cases = []
    for idx, (a, b, h_bits, c) in enumerate(DPP_SETS):
        constants = bytearray(256)
        struct.pack_into("<2fIf", constants, 12, a, b, h_bits, c)
        expected = dpp_reference(a, b, h_bits, c)
        assert all(flush_ftz(word) == word for word in expected), expected
        case = directory / f"dpp-{idx}.case"
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        cases.append((case, "pass"))
    run_pixel_batch(renderer, backend, shader, cases)
    for set_idx, old, new, label in (
        (0, "v_sub_f32_dpp v21, v20, v20 quad_perm:[1,0,3,2]",
         "v_sub_f32_dpp v21, v20, v20 quad_perm:[2,3,0,1]", "AcrossX vs AcrossY"),
        (0, "v_sub_f32_dpp v22, v20, v20 quad_perm:[2,3,0,1]",
         "v_sub_f32_dpp v22, v20, v20 quad_perm:[1,0,3,2]", "AcrossY vs AcrossX"),
        (0, "dpp8:[1,2,3,0,5,6,7,4]", "dpp8:[3,0,1,2,7,4,5,6]", "cyclic dpp8 rotation"),
        (0, "v_mul_f32_dpp v31, v20, v28 quad_perm:[3,3,3,3]",
         "v_mul_f32_dpp v31, v20, v28 quad_perm:[1,1,1,1]", "lane broadcast"),
        (0, "v_min_f32_dpp v33, -v31, v30", "v_min_f32_dpp v33, v31, v30", "DPP16 neg modifier"),
        (0, "v_add_f32_dpp v45, v23, -v23", "v_add_f32_dpp v45, v23, v23", "DPP16 src1 neg modifier"),
        (1, "v_floor_f32_dpp v36, v35", "v_ceil_f32_dpp v36, v35", "floor vs ceil DPP"),
        (0, "quad_perm:[1,2,3,2]", "quad_perm:[1,2,3,0]", "generic quad_perm lane 3"),
        (0, "v_max_f32_dpp v50, |v49|, v49", "v_max_f32_dpp v50, v49, v49", "DPP16 abs on max"),
        (1, "v_cvt_f32_f16_dpp v51, v29", "v_cvt_f32_f16_dpp v51, |v29|", "cvt_f32_f16 DPP sign"),
        (0, "v_exp_f32_dpp v52, v30", "v_exp_f32_dpp v52, v38", "exp of fractional DPP input"),
    ):
        mutated = [line.replace(old, new) for line in program]
        assert mutated != program, label
        source.write_text(translate(mutated), encoding="ascii")
        compile_shader(compiler, source, shader, mode="any")
        wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, cases[set_idx][0]],
                               capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Mutation of {label} escaped detection")


def pack_f16_pair(lo, hi):
    lo_b = struct.unpack("<H", struct.pack("<e", lo))[0]
    hi_b = struct.unpack("<H", struct.pack("<e", hi))[0]
    return lo_b | (hi_b << 16)


def rne_f16(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


# Mixed-precision VOP3P FMA fixture covering v_fma_mix_f32, v_fma_mixlo_f16, v_fma_mixhi_f16,
# op_sel / op_sel_hi, |.| / -, clamp, inline f16 constants, and untouched-half preservation.
FMA_MIX_BODY = [
    "v_mov_b32_e32 v19, s45",
    "v_mov_b32_e32 v20, s46",
    "v_mov_b32_e32 v21, s47",
    "v_mov_b32_e32 v22, 0x40003c00",
    "v_fma_mix_f32 v12, |s44|, -v19, v20 op_sel:[1,1,0] op_sel_hi:[1,0,0]",
    "v_fma_mixlo_f16 v22, s44, 0.5, v21 op_sel:[1,1,0]",
    "v_fma_mix_f32 v23, v22, 1.0, 0 op_sel:[1,0,0] op_sel_hi:[1,0,0]",
    "v_fma_mixhi_f16 v22, -|v19|, v23, v20 op_sel:[1,0,0] op_sel_hi:[1,0,0] clamp",
    "v_cvt_f32_f16_e32 v13, v22",
    "v_fma_mix_f32 v14, v22, 1.0, 0 op_sel:[1,0,0] op_sel_hi:[1,0,0]",
    "v_fma_mix_f32 v15, v12, 1.0, 0.5 clamp",
]

FMA_MIX_SETS = (
    # Set 1: mixed f16/f32 with |.|, -, op_sel_hi, and untouched-half preservation on both mixlo & mixhi.
    (pack_f16_pair(1.5, -2.0), pack_f16_pair(0.5, -0.25), 0.75, 0.25),
    # Set 2: saturates both v_fma_mixhi_f16 clamp and v_fma_mix_f32 clamp to [0, 1].
    (pack_f16_pair(3.0, -4.0), pack_f16_pair(-0.5, 0.5), 2.5, -1.0),
    # Set 3: exact binary16 halfway ties (odd mantissa -> round up under RNE vs down under RTZ).
    (pack_f16_pair(2.0 + 2 ** -9, 1.0), pack_f16_pair(0.5, 0.25), 1.0 + 2 ** -11 + 2 ** -12, 2 ** -11),
)


def fma_mix_reference(a_bits, b_bits, c_val, d_val):
    a_lo = struct.unpack("<e", struct.pack("<H", a_bits & 0xffff))[0]
    a_hi = struct.unpack("<e", struct.pack("<H", (a_bits >> 16) & 0xffff))[0]
    b_lo = struct.unpack("<e", struct.pack("<H", b_bits & 0xffff))[0]
    b_hi = struct.unpack("<e", struct.pack("<H", (b_bits >> 16) & 0xffff))[0]
    v12 = single(abs(a_hi) * (-b_lo) + c_val)
    lo = rne_f16(single(a_lo * 0.5 + d_val))
    hi = rne_f16(min(max(single(-abs(b_hi) * 2.0 + c_val), 0.0), 1.0))
    v15 = min(max(single(v12 + 0.5), 0.0), 1.0)
    return (half_rtz(v12), half_rtz(lo), half_rtz(hi), half_rtz(v15))


def check_fma_mix_draw(directory, compiler, renderer, backend):
    program = with_constants(FMA_MIX_BODY)
    source = directory / "fma_mix.hlsl"
    shader = directory / "fma_mix.cso"
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    cases = []
    for idx, (a_bits, b_bits, c_val, d_val) in enumerate(FMA_MIX_SETS):
        constants = bytearray(256)
        struct.pack_into("<2I2f", constants, 12, a_bits, b_bits, c_val, d_val)
        expected = fma_mix_reference(a_bits, b_bits, c_val, d_val)
        case = directory / f"fma_mix-{idx}.case"
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        cases.append((case, "pass"))
    run_pixel_batch(renderer, backend, shader, cases)
    # Mutations verified against Set 1 (modifiers, half selection, preservation) and Set 2 (clamp).
    for set_idx, old, new, label in (
        (0, "v_fma_mix_f32 v12, |s44|, -v19, v20", "v_fma_mix_f32 v12, s44, -v19, v20", "abs modifier"),
        (0, "v_fma_mix_f32 v12, |s44|, -v19, v20", "v_fma_mix_f32 v12, |s44|, v19, v20", "neg modifier"),
        (0, "op_sel:[1,1,0] op_sel_hi:[1,0,0]", "op_sel:[1,1,0]", "op_sel_hi half select"),
        (0, "v_fma_mixhi_f16 v22, -|v19|", "v_fma_mixlo_f16 v22, -|v19|", "mixhi half preservation"),
        (1, "v_fma_mixhi_f16 v22, -|v19|, v23, v20 op_sel:[1,0,0] op_sel_hi:[1,0,0] clamp",
         "v_fma_mixhi_f16 v22, -|v19|, v23, v20 op_sel:[1,0,0] op_sel_hi:[1,0,0]", "mixhi clamp"),
    ):
        mutated = [line.replace(old, new) for line in program]
        assert mutated != program, label
        source.write_text(translate(mutated), encoding="ascii")
        compile_shader(compiler, source, shader)
        wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, cases[set_idx][0]],
                               capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Mutation of {label} escaped detection")


def sopc_group() -> list[str]:
    body = [
        "s_mov_b32 s70, 0",
        "s_mov_b32 s71, 0",
        "s_mov_b32 s72, 1",
        "s_mov_b32 s73, 0",
    ]
    for op in (
        "s_cmp_eq_i32", "s_cmp_lg_i32", "s_cmp_gt_i32", "s_cmp_ge_i32", "s_cmp_lt_i32", "s_cmp_le_i32",
        "s_cmp_eq_u32", "s_cmp_lg_u32", "s_cmp_gt_u32", "s_cmp_ge_u32", "s_cmp_lt_u32", "s_cmp_le_u32",
    ):
        body.extend([f"{op} s44, s45", "s_cmov_b64 s[70:71], s[72:73]"])
    body.extend([
        "s_cmp_eq_u64 s[44:45], s[46:47]", "s_cmov_b64 s[70:71], s[72:73]",
        "s_cmp_lg_u64 s[44:45], s[46:47]", "s_cmov_b64 s[70:71], s[72:73]",
        "s_bitcmp0_b32 s44, 5", "s_cmov_b64 s[70:71], s[72:73]",
        "s_bitcmp1_b32 s44, 5", "s_cmov_b64 s[70:71], s[72:73]",
        "s_bitcmp0_b64 s[44:45], 37", "s_cmov_b64 s[70:71], s[72:73]",
        "s_bitcmp1_b64 s[44:45], 37", "s_cmov_b64 s[70:71], s[72:73]",
    ])
    return with_constants(body)


def vop2_integer_group() -> list[str]:
    return with_constants([
        "v_mov_b32_e32 v18, s45",
        "v_lshrrev_b32_e32 v12, s44, v18",
        "v_ashrrev_i32_e32 v13, s44, v18",
        "v_lshlrev_b32_e32 v14, s44, v18",
        "v_and_b32_e32 v15, s44, v18",
        "v_or_b32_e32 v12, s44, v18",
        "v_xor_b32_e32 v13, s44, v18",
        "v_add_nc_u32_e32 v14, s44, v18",
        "v_sub_nc_u32_e32 v15, s44, v18",
        "v_subrev_nc_u32_e32 v16, s44, v18",
    ])


def align_integer_group(byte=False, swapped=False) -> list[str]:
    op = "v_alignbyte_b32" if byte else "v_alignbit_b32"
    first, last = (1, 3) if byte else (4, 31)
    left, right = ("v18", "s44") if swapped else ("s44", "v18")
    return uncompressed_export([
        "v_mov_b32_e32 v18, s45",
        f"{op} v12, {left}, {right}, {first}",
        f"{op} v13, {left}, {right}, 0",
        f"{op} v14, {left}, {right}, {last}",
        f"{op} v18, {left}, {right}, s46",
        "v_mov_b32_e32 v15, v18",
    ])


def readfirstlane_program() -> list[str]:
    return uncompressed_export([
        "v_mov_b32_e32 v18, s44",
        "v_readfirstlane_b32 s48, v18",
        "s_add_u32 s49, s48, 5",
        "v_mov_b32_e32 v12, s48",
        "v_mov_b32_e32 v13, s49",
        "v_mov_b32_e32 v14, 0x12345678",
        "v_mov_b32_e32 v15, 0",
    ])


def align_reference(a, b, count, byte=False):
    shifts = (1, 0, 3, count & 3) if byte else (4, 0, 31, count & 31)
    joined = (a << 32) | b
    return tuple((joined >> (shift * (8 if byte else 1))) & 0xffffffff for shift in shifts)


def vop3_ternary_int_group(group=1, mutated=False) -> list[str]:
    if group == 1:
        ops = (
            ("v_xor3_b32", "v_and_or_b32", "v_add3_u32", "v_or3_b32")
            if mutated
            else ("v_add3_u32", "v_or3_b32", "v_xor3_b32", "v_and_or_b32")
        )
    else:
        ops = (
            ("v_lshl_add_u32", "v_add_lshl_u32", "v_xad_u32", "v_lshl_or_b32")
            if mutated
            else ("v_xad_u32", "v_lshl_or_b32", "v_lshl_add_u32", "v_add_lshl_u32")
        )
    return uncompressed_export([
        "v_mov_b32_e32 v18, s45",
        f"{ops[0]} v12, s44, v18, s46",
        f"{ops[1]} v13, s44, v18, s46",
        f"{ops[2]} v14, s44, v18, s46",
        f"{ops[3]} v18, s44, v18, s46",
        "v_mov_b32_e32 v15, v18",
    ])


def vop3_ternary_int_reference(group, a, b, c):
    if group == 1:
        return (
            (a + b + c) & 0xffffffff,
            (a | b | c) & 0xffffffff,
            (a ^ b ^ c) & 0xffffffff,
            ((a & b) | c) & 0xffffffff,
        )
    return (
        ((a ^ b) + c) & 0xffffffff,
        ((a << (b & 31)) | c) & 0xffffffff,
        ((a << (b & 31)) + c) & 0xffffffff,
        (((a + b) & 0xffffffff) << (c & 31)) & 0xffffffff,
    )


# Bodies for uncompressed_export(): s44 = A and s45 = B from the constants, v18 = B, and each channel v12..v15
# holds its own result. Group 3 uses inline constants and register-first operands.
NC_INT_BODIES = {
    1: ["v_add_nc_u32_e64 v12, s44, v18", "v_sub_nc_u32_e64 v13, s44, v18",
        "v_subrev_nc_u32_e32 v14, s44, v18", "v_subrev_nc_u32_e64 v15, s44, v18"],
    2: ["v_add_nc_i32 v12, s44, v18", "v_sub_nc_i32 v13, s44, v18",
        "v_add_nc_u32_e32 v14, s44, v18", "v_sub_nc_u32_e32 v15, s44, v18"],
    3: ["v_subrev_nc_u32_e64 v12, 64, v18", "v_add_nc_i32 v13, 64, v18",
        "v_sub_nc_u32_e64 v14, s44, 64", "v_subrev_nc_u32_e64 v15, v18, s44"],
}
# Reference results by group, from the operand order A (s44) and B (s45): sub is src0 - src1 and subrev is
# src1 - src0, both modulo 2**32. Both follow the translator's float subrev convention; see the docs.
NC_INT_RESULTS = {
    1: lambda a, b: (a + b, a - b, b - a, b - a),
    2: lambda a, b: (a + b, a - b, a + b, a - b),
    3: lambda a, b: (b - 64, 64 + b, a - 64, a - b),
}
# (A, B) as binary32 words; the inputs and every result are normal numbers or zero (asserted at the draw).
NC_INT_ROWS = (
    (0xC0000000, 0x7F000000),  # unsigned wrap
    (0x7F000000, 0x3F800000),  # signed overflow
    (0x40400000, 0x3E800000),  # sub differs from subrev
    (0x3E800000, 0x40400000),  # borrow
    (0x80800000, 0x80800000),  # high-bit wrap, sub is zero
)


def nc_integer_group(group):
    return ["v_mov_b32_e32 v18, s45", *NC_INT_BODIES[group]]


# Integer VOP3 (e64) encodings that the translator accepted with no fixture drawing them, and v_xnor_b32 in
# both encodings. Bodies for uncompressed_export(): A, B and C are the constants in s44, s45 and s46, v18 = B,
# and each channel v12..v15 holds its own result. The shift and bitwise groups take both sources from SGPRs,
# which the VOP2 (e32) encoding cannot express for the second source.
E64_INT_BODIES = {
    "shift": ["v_lshlrev_b32_e64 v12, s44, s45", "v_lshrrev_b32_e64 v13, s44, s45",
              "v_ashrrev_i32_e64 v14, s44, s45", "v_not_b32_e64 v15, s45"],
    "bitwise": ["v_and_b32_e64 v12, s44, s45", "v_or_b32_e64 v13, s44, s45",
                "v_xor_b32_e64 v14, s44, s45", "v_xnor_b32_e64 v15, s44, s45"],
    "xnor_byte": ["v_xnor_b32_e32 v12, s44, v18", "v_cvt_f32_ubyte2_e64 v13, s46",
                  "v_cvt_f32_ubyte2_e64 v14, v18", "v_cvt_f32_ubyte2_e64 v15, s46 clamp"],
}
E64_INT_OPS = tuple(sorted({line.split()[0] for body in E64_INT_BODIES.values() for line in body}))


def _f32_word(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _sar32(word, count):
    return ((word - (1 << 32) if word & 0x80000000 else word) >> count) & 0xffffffff


def _byte(word, index):
    return (word >> (8 * index)) & 0xff


# Reference words by group. The reversed shifts move src1 (B) by the low five bits of src0 (A).
E64_INT_RESULTS = {
    "shift": lambda a, b, c: (b << (a & 31), b >> (a & 31), _sar32(b, a & 31), ~b),
    "bitwise": lambda a, b, c: (a & b, a | b, a ^ b, ~(a ^ b)),
    "xnor_byte": lambda a, b, c: (~(a ^ b), _f32_word(float(_byte(c, 2))), _f32_word(float(_byte(b, 2))),
                                  _f32_word(min(float(_byte(c, 2)), 1.0))),
}
# Negative controls by group: (channel, word a wrong lowering would give, what went wrong). Each is drawn on
# the first row of the group and must report a mismatch.
E64_INT_WRONG = {
    "shift": (
        (0, lambda a, b, c: a << (b & 31), "shift operands swapped"),
        (0, lambda a, b, c: 0, "left-shift count not reduced modulo 32"),
        (1, lambda a, b, c: _sar32(b, a & 31), "arithmetic shift for logical"),
        (2, lambda a, b, c: b >> (a & 31), "logical shift for arithmetic"),
        (3, lambda a, b, c: b, "not dropped"),
    ),
    "bitwise": (
        (0, lambda a, b, c: a | b, "or for and"),
        (1, lambda a, b, c: a & b, "and for or"),
        (2, lambda a, b, c: ~(a ^ b), "xnor for xor"),
        (3, lambda a, b, c: a ^ b, "xor for xnor"),
    ),
    "xnor_byte": (
        (0, lambda a, b, c: a ^ b, "xor for xnor"),
        (1, lambda a, b, c: _f32_word(float(_byte(c, 1))), "byte 1 for byte 2"),
        (1, lambda a, b, c: _f32_word(float(_byte(c, 3))), "byte 3 for byte 2"),
        (2, lambda a, b, c: _f32_word(float(_byte(c, 2))), "wrong source register"),
        (3, lambda a, b, c: _f32_word(float(_byte(c, 2))), "clamp dropped"),
    ),
}
# (A, B, C) as binary32 words; the inputs and every right result are normal numbers or zero (asserted at the
# draw). The shift counts 0x3f800024 and 0x40000023 exceed 31, so only their low five bits (4 and 3) may count;
# B is negative in the first shift row and positive in the second. The four bytes of each C differ.
E64_INT_ROWS = {
    "shift": ((0x3F800024, 0xC1700000, 0), (0x40000023, 0x4C003A00, 0)),
    "bitwise": ((0x40D10000, 0xC0250000, 0), (0x3E270000, 0xBF620000, 0)),
    "xnor_byte": ((0x40D10000, 0xC0250000, 0x40C32A11), (0x3E270000, 0xBF620000, 0x3F00FF80)),
}


def e64_integer_group(group):
    return ["v_mov_b32_e32 v18, s45", *E64_INT_BODIES[group]]


# SDWA forms of the 32-bit integer operations, four results per group in v12 to v15, with A, B, C and D in
# s44 to s47 and B also in v18. Every form appears once; the selections cover each byte and word, zero and
# sign extension on either source, a selected shift count, and results placed into a byte or word with
# UNUSED_PAD and UNUSED_PRESERVE.
SDWA_INT_BODIES = {
    "bitwise": [
        "v_and_b32_sdwa v12, s44, sext(s45) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1",
        "v_or_b32_sdwa v13, s44, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1",
        "v_xor_b32_sdwa v14, s44, s45 dst_sel:BYTE_3 dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_2",
        "v_xnor_b32_sdwa v15, sext(s44), s45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD",
    ],
    "shift": [
        "v_lshlrev_b32_sdwa v12, s44, s45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:DWORD",
        "v_lshrrev_b32_sdwa v13, s44, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:DWORD",
        "v_ashrrev_i32_sdwa v14, s44, sext(s45) dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:WORD_1",
        "v_mov_b32_e32 v15, s47",
        "v_not_b32_sdwa v15, s46 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:BYTE_2",
    ],
    "arith": [
        "v_add_nc_u32_sdwa v12, s44, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3",
        "v_sub_nc_u32_sdwa v13, s44, sext(v18) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0",
        "v_subrev_nc_u32_sdwa v14, s44, s45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD",
        "v_mov_b32_e32 v15, s47",
        "v_add_nc_u32_sdwa v15, s44, s45 dst_sel:BYTE_1 dst_unused:UNUSED_PRESERVE src0_sel:BYTE_0 src1_sel:BYTE_0",
    ],
    "minmax": [
        "v_min_i32_sdwa v12, sext(s44), sext(s45) dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_0",
        "v_max_i32_sdwa v13, sext(s44), v18 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_0",
        "v_min_u32_sdwa v14, s44, s45 dst_sel:BYTE_3 dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_1",
        "v_max_u32_sdwa v15, s44, s45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
    ],
}
SDWA_INT_OPS = tuple(sorted({line.split()[0] for body in SDWA_INT_BODIES.values() for line in body
                             if line.split()[0].endswith("_sdwa")}))


def _word(word, index):
    return (word >> (16 * index)) & 0xffff


def _sext(value, width):
    """The width-bit field as a two's-complement Python integer."""
    return value - (1 << width) if value >> (width - 1) & 1 else value


def _s32(word):
    return _sext(word & 0xffffffff, 32)


# Reference words by group: the e32 operation on the selected fields, then the placement.
SDWA_INT_RESULTS = {
    "bitwise": lambda a, b, c, d: (
        a & _sext(_byte(b, 1), 8), a | _word(b, 1), ((_byte(a, 0) ^ _byte(b, 2)) & 0xff) << 24,
        ~(_sext(_word(a, 0), 16) ^ b)),
    "shift": lambda a, b, c, d: (
        b << (_byte(a, 1) & 31), b >> (_byte(a, 0) & 31),
        ((_sext(_word(b, 1), 16) >> (_byte(a, 2) & 31)) & 0xffff) << 16,
        (d & 0xffff0000) | (~_byte(c, 2) & 0xffff)),
    "arith": lambda a, b, c, d: (
        a + _byte(b, 3), a - _sext(_byte(b, 0), 8), b - _word(a, 0),
        (d & 0xffff00ff) | (((_byte(a, 0) + _byte(b, 0)) & 0xff) << 8)),
    "minmax": lambda a, b, c, d: (
        (min(_sext(_word(a, 1), 16), _sext(_word(b, 0), 16)) & 0xffff) << 16,
        (max(_sext(_word(a, 1), 16), _word(b, 0)) & 0xffff) << 16,
        min(_byte(a, 3), _byte(b, 1)) << 24, max(a, b)),
}
# Negative controls by group: (row, channel, word a wrong lowering would give, what went wrong). Each must
# report a mismatch.
SDWA_INT_WRONG = {
    "bitwise": (
        (0, 0, lambda a, b, c, d: a & _byte(b, 1), "sign extension dropped on src1"),
        (0, 0, lambda a, b, c, d: a & _sext(_byte(b, 0), 8), "BYTE_0 for BYTE_1"),
        (0, 0, lambda a, b, c, d: a | _sext(_byte(b, 1), 8), "or for and"),
        (0, 1, lambda a, b, c, d: a | _word(b, 0), "WORD_0 for WORD_1"),
        (0, 1, lambda a, b, c, d: a | b, "src1 selection ignored"),
        (0, 1, lambda a, b, c, d: a ^ _word(b, 1), "xor for or"),
        (0, 2, lambda a, b, c, d: ((_byte(a, 2) ^ _byte(b, 0)) & 0xff) << 24, "source selections swapped"),
        (0, 2, lambda a, b, c, d: ((_byte(a, 0) ^ _byte(b, 2)) & 0xff) << 16, "result placed in BYTE_2"),
        (0, 2, lambda a, b, c, d: ((_byte(a, 0) | _byte(b, 2)) & 0xff) << 24, "or for xor"),
        (0, 3, lambda a, b, c, d: ~(_word(a, 0) ^ b), "sign extension dropped on src0"),
        (0, 3, lambda a, b, c, d: _sext(_word(a, 0), 16) ^ b, "xor for xnor"),
    ),
    "shift": (
        (0, 0, lambda a, b, c, d: b << (_byte(a, 0) & 31), "count from BYTE_0 for BYTE_1"),
        (0, 0, lambda a, b, c, d: 0, "count not reduced modulo 32"),
        (0, 1, lambda a, b, c, d: _sar32(b, _byte(a, 0) & 31), "arithmetic shift for logical"),
        (0, 1, lambda a, b, c, d: b >> (_byte(a, 1) & 31), "count from BYTE_1 for BYTE_0"),
        (0, 2, lambda a, b, c, d: ((_word(b, 1) >> (_byte(a, 2) & 31)) & 0xffff) << 16, "sign extension dropped on src1"),
        (0, 2, lambda a, b, c, d: (_sext(_word(b, 1), 16) >> (_byte(a, 2) & 31)) & 0xffff, "result placed in WORD_0"),
        (0, 2, lambda a, b, c, d: ((_sext(_word(b, 0), 16) >> (_byte(a, 2) & 31)) & 0xffff) << 16, "WORD_0 for WORD_1"),
        (0, 3, lambda a, b, c, d: ~_byte(c, 2) & 0xffff, "padding for UNUSED_PRESERVE"),
        (0, 3, lambda a, b, c, d: (d & 0xffff0000) | _byte(c, 2), "not dropped"),
        (0, 3, lambda a, b, c, d: (d & 0xffff0000) | (~_byte(c, 1) & 0xffff), "BYTE_1 for BYTE_2"),
        (0, 3, lambda a, b, c, d: (d & 0xffff0000) | (~_byte(c, 2) & 0xff), "byte written for word"),
    ),
    "arith": (
        (0, 0, lambda a, b, c, d: a + _byte(b, 2), "BYTE_2 for BYTE_3"),
        (0, 0, lambda a, b, c, d: a - _byte(b, 3), "sub for add"),
        (0, 1, lambda a, b, c, d: a - _byte(b, 0), "sign extension dropped on src1"),
        (0, 1, lambda a, b, c, d: _sext(_byte(b, 0), 8) - a, "subrev order for sub"),
        (0, 2, lambda a, b, c, d: _word(a, 0) - b, "sub order for subrev"),
        (0, 2, lambda a, b, c, d: b - _word(a, 1), "WORD_1 for WORD_0"),
        (0, 2, lambda a, b, c, d: b - _sext(_word(a, 0), 16), "sign extension added on src0"),
        (0, 3, lambda a, b, c, d: (d & 0xffff00ff) | ((_byte(a, 0) + _byte(b, 0)) << 8), "carry leaves the byte"),
        (0, 3, lambda a, b, c, d: ((_byte(a, 0) + _byte(b, 0)) & 0xff) << 8, "padding for UNUSED_PRESERVE"),
        (0, 3, lambda a, b, c, d: (d & 0xffffff00) | ((_byte(a, 0) + _byte(b, 0)) & 0xff), "result placed in BYTE_0"),
    ),
    "minmax": (
        (0, 0, lambda a, b, c, d: (min(_word(a, 1), _word(b, 0)) & 0xffff) << 16, "sign extension dropped on src0"),
        (1, 0, lambda a, b, c, d: (min(_sext(_word(a, 1), 16), _word(b, 0)) & 0xffff) << 16, "sign extension dropped on src1"),
        (0, 1, lambda a, b, c, d: (max(_word(a, 1), _word(b, 0)) & 0xffff) << 16, "sign extension dropped on src0"),
        (1, 1, lambda a, b, c, d: (max(_sext(_word(a, 1), 16), _sext(_word(b, 0), 16)) & 0xffff) << 16, "sign extension added on src1"),
        (0, 2, lambda a, b, c, d: max(_byte(a, 3), _byte(b, 1)) << 24, "max for min"),
        (0, 2, lambda a, b, c, d: min(_byte(a, 3), _byte(b, 0)) << 24, "BYTE_0 for BYTE_1"),
        (0, 3, lambda a, b, c, d: max(_s32(a), _s32(b)), "signed max for unsigned"),
        (1, 3, lambda a, b, c, d: min(a, b), "min for max"),
    ),
}
# (A, B, C, D) as binary32 words; the inputs and every right result are normal numbers or zero (asserted at
# the draw). The selected bytes and words differ from their neighbours, the sign-extended fields are negative
# in at least one row, and the shift counts (bytes of A in the shift group) exceed 31.
SDWA_INT_ROWS = {
    "bitwise": ((0x40D18A11, 0xC025B37E, 0, 0), (0x3E27F0C4, 0xBF62915A, 0, 0)),
    "shift": ((0x3F232425, 0xC1700000, 0x40C32A11, 0x41A00000), (0x40214263, 0x4C003A00, 0x3F00FF80, 0xC2480000)),
    "arith": ((0x40D18AF1, 0xC025B38E, 0, 0x41A00000), (0x3E27F0C4, 0xBF62915A, 0, 0xC2480000)),
    "minmax": ((0xC0258A11, 0x40D13A00, 0, 0), (0x3E27F0C4, 0xBF62915A, 0, 0)),
}


def sdwa_integer_group(group):
    return ["v_mov_b32_e32 v18, s45", *SDWA_INT_BODIES[group]]


def sdwa_integer_cases(group):
    """(row, expected words, status, reason) for every draw of a group: the right rows, then the controls."""
    rows = SDWA_INT_ROWS[group]
    right = [tuple(word & 0xffffffff for word in SDWA_INT_RESULTS[group](*row)) for row in rows]
    cases = [(row, words, "pass", "") for row, words in zip(rows, right)]
    for index, channel, wrong, reason in SDWA_INT_WRONG[group]:
        words = list(right[index])
        words[channel] = wrong(*rows[index]) & 0xffffffff
        assert words[channel] != right[index][channel], (group, reason)
        cases.append((rows[index], tuple(words), "mismatch", reason))
    # Integer export carries any pattern, but normal rows keep the draw independent of NaN and denormal handling.
    for row, words in zip(rows, right):
        assert all(binary32_normal_or_zero(word) for word in (*row, *words)), (group, [hex(w) for w in (*row, *words)])
    return cases


# Saturating 16-bit packs (v_cvt_pk_u16_u32, v_cvt_pk_i16_i32). Bodies for uncompressed_export(): s44 = A and
# s45 = B from the constants, v18 = B, v19 = B >> 16 (the high half, always inside both 16-bit ranges when
# read as unsigned or sign-extended below), v20 = B >> 12 and v21 = B >> 16 arithmetic. A normal binary32
# word is always outside both ranges, so the in-range operands are derived in the shader. A result whose
# high half saturates is not a normal binary32 word, so those channels are exported through an xor with
# PACK_FLIP, which moves the exponent field off all-ones and all-zeroes.
PACK_FLIP = 0x40000000
PACK_LITERAL = 0x12345
PACK_PREFIX = ["v_mov_b32_e32 v18, s45", "v_lshrrev_b32_e32 v19, 16, v18", "v_lshrrev_b32_e32 v20, 12, v18",
               "v_ashrrev_i32_e32 v21, 16, v18"]
PACK_BODIES = {
    "unsigned": [f"v_cvt_pk_u16_u32 v12, {PACK_LITERAL:#x}, v19", "v_cvt_pk_u16_u32 v13, v20, v19",
                 "v_cvt_pk_u16_u32 v22, v19, s44", f"v_xor_b32_e32 v14, {PACK_FLIP:#x}, v22",
                 "v_cvt_pk_u16_u32 v15, v21, v19"],
    "signed": ["v_cvt_pk_i16_i32 v12, s44, v21", "v_cvt_pk_i16_i32 v22, v21, s44",
               f"v_xor_b32_e32 v13, {PACK_FLIP:#x}, v22", "v_cvt_pk_i16_i32 v14, v19, v21",
               "v_cvt_pk_i16_i32 v15, -16, v21"],
}
PACK_OPS = ("v_cvt_pk_i16_i32", "v_cvt_pk_u16_u32")


def _sat_u16(word):
    return min(word & 0xffffffff, 0xffff)


def _sat_i16(word):
    return max(-32768, min(_s32(word & 0xffffffff), 32767)) & 0xffff


def _pk(lo, hi):
    return (lo & 0xffff) | ((hi & 0xffff) << 16)


# Reference words by group: src0 saturates into the low half and src1 into the high half.
PACK_RESULTS = {
    "unsigned": lambda a, b: (_pk(_sat_u16(PACK_LITERAL), _sat_u16(b >> 16)), _pk(_sat_u16(b >> 12), _sat_u16(b >> 16)),
                              _pk(_sat_u16(b >> 16), _sat_u16(a)) ^ PACK_FLIP,
                              _pk(_sat_u16(_sar32(b, 16)), _sat_u16(b >> 16))),
    "signed": lambda a, b: (_pk(_sat_i16(a), _sat_i16(_sar32(b, 16))),
                            _pk(_sat_i16(_sar32(b, 16)), _sat_i16(a)) ^ PACK_FLIP,
                            _pk(_sat_i16(b >> 16), _sat_i16(_sar32(b, 16))),
                            _pk(_sat_i16(-16), _sat_i16(_sar32(b, 16)))),
}
# Negative controls by group: (channel, word a wrong lowering would give, what went wrong). Each is drawn on
# the first row of the group and must report a mismatch.
PACK_WRONG = {
    "unsigned": (
        (0, lambda a, b: _pk(PACK_LITERAL, b >> 16), "literal truncated instead of saturated"),
        (0, lambda a, b: _pk(b >> 16, _sat_u16(PACK_LITERAL)), "halves swapped"),
        (1, lambda a, b: _pk(b >> 12, b >> 16), "register truncated instead of saturated"),
        (2, lambda a, b: _pk(b >> 16, a) ^ PACK_FLIP, "high half truncated instead of saturated"),
        (2, lambda a, b: _pk(_sat_u16(a), b >> 16) ^ PACK_FLIP, "halves swapped"),
        (2, lambda a, b: _pk(b >> 16, _sat_i16(a)) ^ PACK_FLIP, "signed bound for the unsigned pack"),
        (3, lambda a, b: _pk(0, b >> 16), "negative source clamped to zero"),
        (3, lambda a, b: _pk(_sar32(b, 16), b >> 16), "source read as signed and kept"),
    ),
    "signed": (
        (0, lambda a, b: _pk(a, _sar32(b, 16)), "truncated instead of saturated"),
        (0, lambda a, b: _pk(_sat_u16(a), _sar32(b, 16)), "unsigned bound for the signed pack"),
        (0, lambda a, b: _pk(_sar32(b, 16), _sat_i16(a)), "halves swapped"),
        (1, lambda a, b: _pk(_sar32(b, 16), a) ^ PACK_FLIP, "high half truncated instead of saturated"),
        (2, lambda a, b: _pk(b >> 16, _sar32(b, 16)), "value above 32767 kept"),
        (3, lambda a, b: _pk(0x7fff, _sar32(b, 16)), "inline -16 read as unsigned"),
        (3, lambda a, b: _pk(0, _sar32(b, 16)), "negative source clamped to zero"),
    ),
}
# (A, B) as binary32 words; the inputs and every right result are normal numbers or zero (asserted at the
# draw). First row: A is positive (it saturates to 0x7fff as signed) and B is negative, so B >> 16 is above
# 32767, B >> 12 is above 65535 and the arithmetic shift is negative. Second row: A is negative (0x8000 as
# signed) and B >> 12 is 0xfff, which both packs must keep.
PACK_ROWS = ((0x40D12345, 0xC1257F90), (0xC0251234, 0x00FFF123))


def pack_group(group):
    return [*PACK_PREFIX, *PACK_BODIES[group]]


# v_add_co_u32, v_sub_co_u32 and v_subrev_co_u32: a 32-bit add or subtract that also writes the unsigned carry
# or borrow of each lane to a lane mask. Bodies for uncompressed_export(): A, B, C and D are the constants in
# s44 to s47, v18 = B and v19 = D. v12 holds op(A, B) with its carry in vcc_lo and v14 holds op(B, A) with its
# carry in an SGPR, in the Wave32 spelling LLVM 15 assembles for gfx1013. v13 and v15 select D where the carry
# is set and C where it is clear; D is read from a VGPR because gfx10 allows two scalar operands at most.
CARRY_OUT_OPS = ("v_add_co_u32", "v_sub_co_u32", "v_subrev_co_u32")


def carry_out_group(op):
    return ["v_mov_b32_e32 v18, s45", "v_mov_b32_e32 v19, s47",
            f"{op} v12, vcc_lo, s44, v18", "v_cndmask_b32_e64 v13, s46, v19, vcc_lo",
            f"{op} v14, s60, v18, s44", "v_cndmask_b32_e64 v15, s46, v19, s60"]


def _carry_out(op, src0, src1):
    """The 32-bit result and the carry-out, from unbounded integers: bit 32 of the sum, or a negative difference."""
    wide = {"v_add_co_u32": src0 + src1, "v_sub_co_u32": src0 - src1, "v_subrev_co_u32": src1 - src0}[op]
    return wide & 0xffffffff, not 0 <= wide <= 0xffffffff


def carry_out_words(op, a, b, c, d):
    first, first_carry = _carry_out(op, a, b)
    second, second_carry = _carry_out(op, b, a)
    return first, d if first_carry else c, second, d if second_carry else c


# (A, B) as binary32 words, drawn with C and D from _CARRY_SELECT; the inputs and every right result are normal
# numbers or zero (asserted at the draw). Add: a carry, no carry, a zero addend (the sum equals src0 and must
# not carry) and a sum that wraps to exactly zero (it must carry). Subtract: a borrow in one operand order and
# none in the other, equal operands (no borrow in either order), and a row with the larger operand in A.
_CARRY_SELECT = (0x3E270000, 0xBF620000)
CARRY_OUT_ROWS = {
    "v_add_co_u32": ((0xC1700000, 0x7F000000), (0x3F800024, 0x01000000), (0x40D10000, 0), (0xC0000000, 0x40000000)),
    "v_sub_co_u32": ((0x3F800024, 0xC1700000), (0x40D10000, 0x40D10000), (0xC0250000, 0x3E270000)),
    "v_subrev_co_u32": ((0x3F800024, 0xC1700000), (0x40D10000, 0x40D10000), (0xC0250000, 0x3E270000)),
}
# Negative controls by operation: (row, channel, word a wrong lowering would give, what went wrong). Each must
# report a mismatch.
CARRY_OUT_WRONG = {
    "v_add_co_u32": (
        (0, 0, lambda a, b, c, d: a - b, "subtract for add"),
        (0, 1, lambda a, b, c, d: c, "carry dropped"),
        (0, 3, lambda a, b, c, d: c, "carry dropped on the SGPR mask"),
        (1, 1, lambda a, b, c, d: d, "carry without overflow"),
        (2, 1, lambda a, b, c, d: d, "carry on a sum equal to src0"),
        (3, 1, lambda a, b, c, d: c, "carry dropped when the sum wraps to zero"),
    ),
    "v_sub_co_u32": (
        (0, 0, lambda a, b, c, d: b - a, "operands reversed"),
        (0, 0, lambda a, b, c, d: a + b, "add for subtract"),
        (0, 1, lambda a, b, c, d: c, "borrow dropped"),
        (0, 3, lambda a, b, c, d: d, "borrow of the other operand order"),
        (1, 1, lambda a, b, c, d: d, "borrow on equal operands"),
        (2, 3, lambda a, b, c, d: c, "borrow dropped on the SGPR mask"),
    ),
    "v_subrev_co_u32": (
        (0, 0, lambda a, b, c, d: a - b, "subtract for reverse subtract"),
        (0, 1, lambda a, b, c, d: d, "borrow of the unreversed subtract"),
        (0, 3, lambda a, b, c, d: c, "borrow dropped on the SGPR mask"),
        (1, 1, lambda a, b, c, d: d, "borrow on equal operands"),
        (2, 1, lambda a, b, c, d: c, "borrow dropped"),
    ),
}
# v_add_co_ci_u32, v_sub_co_ci_u32 and v_subrev_co_ci_u32: the carry-out add and subtracts above, plus the carry or
# borrow of each lane read from a lane mask. Bodies for uncompressed_export(): A, B, C and D are the constants in
# s44 to s47, v18 = B and v19 = D. The carry-out form of the same operation leaves the carry of op(A, B) in vcc_lo
# and that of op(A, D) in s60; its results in v16 and v17 are not exported. v12 = op(C, D, vcc_lo) in the VOP2
# (e32) encoding, whose masks are always VCC, and v14 = op(D, C, s60) in the VOP3 (e64) encoding with its carry-out
# in s61. v13 and v15 are 1.0 where the carry-out is set and 0.5 where it is clear. 2.0 (0x40000000) is then added
# to v12 and v14: a bijection modulo 2^32, so no bit is lost, and it moves the all-ones word that a borrow taken
# only from the borrow-in gives (C = D) to a normal number.
CARRY_IN_OPS = ("v_add_co_ci_u32", "v_sub_co_ci_u32", "v_subrev_co_ci_u32")
_CARRY_IN_OFFSET = 0x40000000
_CARRY_IN_SELECT = (0x3F000000, 0x3F800000)


def carry_in_group(op):
    base = op.replace("_co_ci_", "_co_")
    return ["v_mov_b32_e32 v18, s45", "v_mov_b32_e32 v19, s47",
            f"{base} v16, vcc_lo, s44, v18", f"{base} v17, s60, s44, v19",
            f"{op}_e32 v12, vcc_lo, s46, v19, vcc_lo", f"{op}_e64 v14, s61, v19, s46, s60",
            "v_cndmask_b32_e64 v13, 0.5, 1.0, vcc_lo", "v_cndmask_b32_e64 v15, 0.5, 1.0, s61",
            "v_add_nc_u32_e32 v12, 2.0, v12", "v_add_nc_u32_e32 v14, 2.0, v14"]


def _carry_in(op, src0, src1, carry):
    """The 32-bit result and the carry-out, from unbounded integers: op(src0, src1) plus or minus the carry-in."""
    wide = {"v_add_co_ci_u32": src0 + src1 + carry, "v_sub_co_ci_u32": src0 - src1 - carry,
            "v_subrev_co_ci_u32": src1 - src0 - carry}[op]
    return wide & 0xffffffff, not 0 <= wide <= 0xffffffff


def carry_in_masks(op, a, b, c, d):
    """The carry-in of each channel: the carry-out of op(A, B) in vcc_lo and of op(A, D) in s60."""
    base = op.replace("_co_ci_", "_co_")
    return _carry_out(base, a, b)[1], _carry_out(base, a, d)[1]


def carry_in_words(op, a, b, c, d):
    first_in, second_in = carry_in_masks(op, a, b, c, d)
    first, first_carry = _carry_in(op, c, d, first_in)
    second, second_carry = _carry_in(op, d, c, second_in)
    low, high = _CARRY_IN_SELECT
    return ((first + _CARRY_IN_OFFSET) & 0xffffffff, high if first_carry else low,
            (second + _CARRY_IN_OFFSET) & 0xffffffff, high if second_carry else low)


# (A, B, C, D) as binary32 words; the inputs, every result before and after the offset (except the all-ones word
# of a borrow taken only from the borrow-in) and every exported word are normal numbers or zero (asserted at the
# draw). Row 0 sets both carry-ins and makes the carry-out of both channels come from the carry-in alone (C + D
# is 0xffffffff, or C = D). Row 1 clears both carry-ins and carries or borrows from the operands. Rows 2 and 3 set
# only one carry-in, vcc_lo in row 2 and s60 in row 3, so a carry-in read from the other mask changes a word.
CARRY_IN_ROWS = {
    "v_add_co_ci_u32": ((0xC0000000, 0x40000000, 0x7F7FFFFF, 0x80800000),
                        (0x7F7FFFFF, 0x3F000000, 0xC1700000, 0x40D10000),
                        (0x3F800000, 0xC1700000, 0x40000000, 0x40D10000),
                        (0x7F7FFFFF, 0x01000000, 0x00800000, 0xC0250000)),
    "v_sub_co_ci_u32": ((0x01000000, 0x7F000000, 0x3F800000, 0x3F800000),
                        (0x7F7FFFFF, 0x3F800024, 0x3F800000, 0x40D10000),
                        (0x3F800024, 0x80000000, 0x40800000, 0x3F800000),
                        (0x3F800000, 0x3F800000, 0x3F800000, 0x7F000000)),
    "v_subrev_co_ci_u32": ((0x7F7FFFFF, 0x40000000, 0x00800000, 0x00800000),
                           (0x01000000, 0xBF620000, 0xC0000000, 0x80000000),
                           (0x7F000000, 0x3F800024, 0x7F7FFFFF, 0xC0000000),
                           (0x3F800000, 0x3F800000, 0x3F800000, 0x01000000)),
}


def _offset(word):
    return (word + _CARRY_IN_OFFSET) & 0xffffffff


# Negative controls by operation: (row, channel, word a wrong lowering would give, what went wrong). Each must
# report a mismatch. Channels 1 and 3 give the word for a carry-out that is clear (0.5) or set (1.0).
_CLEAR, _SET = _CARRY_IN_SELECT
CARRY_IN_WRONG = {
    "v_add_co_ci_u32": (
        (0, 0, lambda a, b, c, d: _offset(c + d), "carry-in dropped from the sum"),
        (0, 1, lambda a, b, c, d: _CLEAR, "carry-out without the carry-in term"),
        (0, 3, lambda a, b, c, d: _CLEAR, "carry-out without the carry-in term on the SGPR masks"),
        (1, 1, lambda a, b, c, d: _CLEAR, "carry-out of the operands dropped"),
        (1, 0, lambda a, b, c, d: _offset(c + d + 1), "carry-in read as set"),
        (2, 0, lambda a, b, c, d: _offset(c - d - 1), "subtract for add"),
        (2, 2, lambda a, b, c, d: _offset(d + c + 1), "carry-in of the e64 form read from vcc_lo"),
        (3, 0, lambda a, b, c, d: _offset(c + d + 1), "carry-in of the e32 form read from s60"),
        (3, 2, lambda a, b, c, d: _offset(d + c), "carry-in of the e64 form dropped"),
    ),
    "v_sub_co_ci_u32": (
        (0, 0, lambda a, b, c, d: _offset(c - d), "borrow-in dropped"),
        (0, 1, lambda a, b, c, d: _CLEAR, "borrow-out without the borrow-in term"),
        (0, 3, lambda a, b, c, d: _CLEAR, "borrow-out without the borrow-in term on the SGPR masks"),
        (1, 1, lambda a, b, c, d: _CLEAR, "borrow-out of the operands dropped"),
        (2, 0, lambda a, b, c, d: _offset(d - c - 1), "operands reversed"),
        (2, 0, lambda a, b, c, d: _offset(c + d + 1), "add for subtract"),
        (2, 0, lambda a, b, c, d: _offset(c - d + 1), "borrow-in added"),
        (2, 2, lambda a, b, c, d: _offset(d - c - 1), "borrow-in of the e64 form read from vcc_lo"),
        (3, 0, lambda a, b, c, d: _offset(c - d - 1), "borrow-in of the e32 form read from s60"),
        (3, 3, lambda a, b, c, d: _SET, "borrow-out of the reversed operands"),
    ),
    "v_subrev_co_ci_u32": (
        (0, 0, lambda a, b, c, d: _offset(d - c), "borrow-in dropped"),
        (0, 1, lambda a, b, c, d: _CLEAR, "borrow-out without the borrow-in term"),
        (0, 3, lambda a, b, c, d: _CLEAR, "borrow-out without the borrow-in term on the SGPR masks"),
        (1, 1, lambda a, b, c, d: _CLEAR, "borrow-out of the operands dropped"),
        (2, 0, lambda a, b, c, d: _offset(c - d - 1), "subtract for reverse subtract"),
        (2, 2, lambda a, b, c, d: _offset(c - d - 1), "borrow-in of the e64 form read from vcc_lo"),
        (3, 0, lambda a, b, c, d: _offset(d - c - 1), "borrow-in of the e32 form read from s60"),
        (3, 3, lambda a, b, c, d: _SET, "borrow-out of the unreversed subtract"),
    ),
}
# Sums of absolute differences (v_sad_hi_u8, v_sad_u32, v_msad_u8). Bodies for uncompressed_export(): s44 = A,
# s45 = B and s46 = C from the constants, v18 = B, and each channel v12..v15 holds its own result. The second
# source is the reference of v_msad_u8, here once a register and once a literal whose byte 2 is zero. No
# instruction uses more than two scalar registers and literals, the gfx10 constant bus limit LLVM 15 enforces.
SAD_LITERAL = 0x7f00ff01
SAD_BODIES = {
    "bytes": ["v_sad_hi_u8 v12, s44, v18, s46", "v_sad_hi_u8 v13, v18, s44, 0",
              "v_msad_u8 v14, s44, v18, s46", "v_msad_u8 v15, v18, s44, s46"],
    "dword": ["v_sad_u32 v12, s44, v18, s46", "v_sad_u32 v13, v18, s44, 64",
              f"v_msad_u8 v14, s44, {SAD_LITERAL:#x}, v18", "v_sad_hi_u8 v15, s45, s44, v18"],
}
SAD_OPS = tuple(sorted({line.split()[0] for body in SAD_BODIES.values() for line in body}))


def _sad4(a, b, masked=False, lanes=4, signed=False):
    """Sum of |a.byte[k] - b.byte[k]| over the low lanes; masked leaves out lanes whose b byte is zero."""
    def lane(word, k):
        value = _byte(word, k)
        return value - 256 if signed and value >= 128 else value
    return sum(abs(lane(a, k) - lane(b, k)) for k in range(lanes) if not masked or _byte(b, k))


def _signed_absdiff32(a, b):
    return abs((a - (1 << 32) if a & 0x80000000 else a) - (b - (1 << 32) if b & 0x80000000 else b))


# Reference words by group, from the definitions in the AMD RDNA 2 ISA reference (see docs/experiments.md).
SAD_RESULTS = {
    "bytes": lambda a, b, c: ((_sad4(a, b) << 16) + c, _sad4(b, a) << 16, _sad4(a, b, True) + c, _sad4(b, a, True) + c),
    "dword": lambda a, b, c: (abs(a - b) + c, abs(b - a) + 64, _sad4(a, SAD_LITERAL, True) + b, (_sad4(b, a) << 16) + b),
}
# Negative controls by group: (channel, word a wrong lowering would give, what went wrong). Each is drawn on
# the first row of the group and must report a mismatch.
SAD_WRONG = {
    "bytes": (
        (0, lambda a, b, c: _sad4(a, b) + c, "sum not moved to the high half"),
        (0, lambda a, b, c: (_sad4(a, b, True) << 16) + c, "v_sad_hi_u8 masked like v_msad_u8"),
        (0, lambda a, b, c: (_sad4(a, b, signed=True) << 16) + c, "signed byte lanes"),
        (0, lambda a, b, c: (_sad4(a, b, lanes=3) << 16) + c, "top byte lane dropped"),
        (0, lambda a, b, c: _sad4(a, b) << 16, "accumulator dropped"),
        (1, lambda a, b, c: (_sad4(b, a) << 16) + c, "s46 read for the inline zero accumulator"),
        (2, lambda a, b, c: _sad4(a, b) + c, "zero reference bytes not left out"),
        (2, lambda a, b, c: _sad4(b, a, True) + c, "mask taken from src0"),
        (2, lambda a, b, c: (_sad4(a, b, True) << 16) + c, "v_msad_u8 moved to the high half"),
        (3, lambda a, b, c: _sad4(a, b, True) + c, "operands not swapped"),
    ),
    "dword": (
        (0, lambda a, b, c: _signed_absdiff32(a, b) + c, "signed comparison"),
        (0, lambda a, b, c: abs(a - b), "accumulator dropped"),
        (0, lambda a, b, c: _sad4(a, b) + c, "byte lanes for the dword difference"),
        (1, lambda a, b, c: abs(a - b) + c, "s46 read for the inline accumulator"),
        (2, lambda a, b, c: _sad4(a, SAD_LITERAL) + b, "zero byte of the literal reference not left out"),
        (2, lambda a, b, c: _sad4(SAD_LITERAL, a, True) + b, "mask taken from src0"),
        (2, lambda a, b, c: _sad4(a, SAD_LITERAL, True) + c, "s46 read for the register accumulator"),
        (3, lambda a, b, c: (_sad4(b, a) << 16) + c, "wrong accumulator register"),
    ),
}
# (A, B, C) as binary32 words; the inputs and every right result are normal numbers or zero (asserted at the
# draw). In the first bytes row A and B each have one zero byte in a different lane and lanes on both sides of
# 0x80; in the first dword row A and B differ in the sign bit and the sum with C passes 2^32.
SAD_ROWS = {
    "bytes": ((0xC1007F90, 0x40C30011, 0x3F80FFFF), (0x3E0027FF, 0xBF62FF00, 0x40000000)),
    "dword": ((0xC1700000, 0x3F800024, 0x7F000000), (0x40000023, 0x4C003A00, 0x3F800000)),
}


def sad_group(group):
    return ["v_mov_b32_e32 v18, s45", *SAD_BODIES[group]]


def sop2_lshl_add_group() -> list[str]:
    return with_constants([
        "s_lshl1_add_u32 s60, s44, s45",
        "s_cselect_b32 s71, 1, 0",
        "s_lshl2_add_u32 s61, s44, s45",
        "s_cselect_b32 s72, 2, 0",
        "s_lshl3_add_u32 s62, s44, s45",
        "s_cselect_b32 s73, 4, 0",
        "s_lshl4_add_u32 s63, s44, s45",
        "s_cselect_b32 s74, 8, 0",
        "v_mov_b32_e32 v12, s60",
        "v_mov_b32_e32 v13, s61",
        "v_mov_b32_e32 v14, s62",
        "v_mov_b32_e32 v15, s63",
    ])


def sop2_absdiff_group() -> list[str]:
    return with_constants([
        "s_absdiff_i32 s60, s44, s45",
        "s_cselect_b32 s71, 1, 0",
        "v_mov_b32_e32 v12, s60",
        "v_mov_b32_e32 v13, s71",
        "v_mov_b32_e32 v14, 0",
        "v_mov_b32_e32 v15, 0",
    ])


def pack_f16_pair(lo, hi):
    lo_b = struct.unpack("<H", struct.pack("<e", lo))[0]
    hi_b = struct.unpack("<H", struct.pack("<e", hi))[0]
    return lo_b | (hi_b << 16)


# VOP3P dot-product fixture covering all 7 RDNA 2 (gfx1030) dot-product forms:
# v_dot2_f32_f16 (op_sel, op_sel_hi, neg_lo, neg_hi, clamp), v_dot2_{i,u}32_{i,u}16,
# v_dot4_{i,u}32_{i,u}8, and v_dot8_{i,u}32_{i,u}4 (wrapping and saturating clamp).
DOT_BODY = [
    "v_mov_b32_e32 v18, s44",
    "v_mov_b32_e32 v19, s45",
    "v_mov_b32_e32 v20, s46",
    "v_mov_b32_e32 v21, s47",
    "v_dot2_f32_f16 v22, v18, 0x40003800, 0.5 op_sel:[1,0,0] op_sel_hi:[0,1,1] neg_lo:[1,0,1] neg_hi:[0,1,0]",
    "v_dot2_f32_f16 v12, v18, 0x3c003800, v22 clamp",
    "v_dot2_i32_i16 v23, v19, v20, v21 op_sel:[0,1,0] op_sel_hi:[1,0,1] clamp",
    "v_dot2_u32_u16 v24, v19, v20, v21 clamp",
    "v_dot4_i32_i8 v25, v19, v20, v21 clamp",
    "v_dot4_u32_u8 v26, v19, v20, v21",
    "v_dot8_i32_i4 v27, v19, v20, v21",
    "v_dot8_u32_u4 v28, v19, v20, v21 clamp",
    "v_dot8_u32_u4 v29, v23, 0x12341234, 0",
    "v_dot8_u32_u4 v29, v24, 0x43214321, v29",
    "v_cvt_f32_f16_e32 v13, v29",
    "v_mul_f32_e32 v13, 0x4b800000, v13",
    "v_dot8_u32_u4 v30, v25, 0x12341234, 0",
    "v_dot8_u32_u4 v30, v26, 0x43214321, v30",
    "v_cvt_f32_f16_e32 v14, v30",
    "v_mul_f32_e32 v14, 0x4b800000, v14",
    "v_dot8_u32_u4 v31, v27, 0x12341234, 0",
    "v_dot8_u32_u4 v31, v28, 0x43214321, v31",
    "v_cvt_f32_f16_e32 v15, v31",
    "v_mul_f32_e32 v15, 0x4b800000, v15",
]

DOT_SETS = (
    # Set 1: mixed signs, interior v_dot2_f32_f16 value, non-overflowing signed & unsigned integer dots.
    (pack_f16_pair(-0.5, 1.0), 0xfffc0005, 0x0003fffe, 0x00000010),
    # Set 2: positive signed saturation (INT32_MAX) and unsigned saturation (UINT32_MAX).
    (pack_f16_pair(2.0, 2.0), 0x7fff7fff, 0x7fff7fff, 0x7ffffff0),
    # Set 3: negative signed saturation (INT32_MIN) and high-end clamp on v_dot2_f32_f16.
    (pack_f16_pair(-2.0, 2.0), 0x8000ffff, 0x7fff7fff, 0x80000010),
    # Set 4: v_dot8_u32_u4 unsigned saturation (UINT32_MAX) vs unwrapped v_dot4_u32_u8 wraparound.
    (pack_f16_pair(0.5, -1.0), 0xffffffff, 0xffffffff, 0xfffffe00),
    # Set 5: signed dots one below INT32_MAX (0x7ffffffe), no saturation.
    (pack_f16_pair(-0.5, 1.0), 0x00070007, 0x0003ffff, 0x7ffffff0),
    # Set 6: signed dots one above INT32_MIN (0x80000001), no saturation.
    (pack_f16_pair(-0.5, 1.0), 0x00050005, 0xfffeffff, 0x80000010),
)


def dot_reference(f16_bits, x, y, acc):
    def sext(val, bits):
        mask = (1 << bits) - 1
        v = val & mask
        return v - (1 << bits) if v & (1 << (bits - 1)) else v

    def sat_i32(v):
        return max(-0x80000000, min(0x7fffffff, v)) & 0xffffffff

    def sat_u32(v):
        return min(0xffffffff, v) & 0xffffffff

    def nibble_dot(a, b):
        return sum(((a >> (4 * k)) & 0xf) * ((b >> (4 * k)) & 0xf) for k in range(8))

    a_lo = struct.unpack("<e", struct.pack("<H", f16_bits & 0xffff))[0]
    a_hi = struct.unpack("<e", struct.pack("<H", (f16_bits >> 16) & 0xffff))[0]
    v22 = single((-a_hi) * 0.5 + a_lo * (-2.0) + (-0.5))
    v12 = min(max(single(a_lo * 0.5 + a_hi * 1.0 + v22), 0.0), 1.0)

    acc_s = sext(acc, 32)
    v23 = sat_i32(sext(x, 16) * sext(y >> 16, 16) + sext(x >> 16, 16) * sext(y, 16) + acc_s)
    v24 = sat_u32((x & 0xffff) * (y & 0xffff) + ((x >> 16) & 0xffff) * ((y >> 16) & 0xffff) + acc)
    v25 = sat_i32(sum(sext(x >> (8 * k), 8) * sext(y >> (8 * k), 8) for k in range(4)) + acc_s)
    v26 = (sum(((x >> (8 * k)) & 0xff) * ((y >> (8 * k)) & 0xff) for k in range(4)) + acc) & 0xffffffff
    v27 = (sum(sext(x >> (4 * k), 4) * sext(y >> (4 * k), 4) for k in range(8)) + acc_s) & 0xffffffff
    v28 = sat_u32(nibble_dot(x, y) + acc)

    c1 = nibble_dot(v23, 0x12341234) + nibble_dot(v24, 0x43214321)
    c2 = nibble_dot(v25, 0x12341234) + nibble_dot(v26, 0x43214321)
    c3 = nibble_dot(v27, 0x12341234) + nibble_dot(v28, 0x43214321)
    assert c1 < 1024 and c2 < 1024 and c3 < 1024, (c1, c2, c3)
    return (half_rtz(v12), float(c1), float(c2), float(c3))


def check_dot_draw(directory, compiler, renderer, backend):
    program = with_constants(DOT_BODY)
    source = directory / "dot.hlsl"
    shader = directory / "dot.cso"
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    cases = []
    for idx, values in enumerate(DOT_SETS):
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *values)
        expected = dot_reference(*values)
        case = directory / f"dot-{idx}.case"
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        cases.append((case, "pass"))
    run_pixel_batch(renderer, backend, shader, cases)
    for set_idx, old, new, label in (
        (0, "neg_lo:[1,0,1]", "neg_lo:[0,0,1]", "dot2_f32_f16 neg_lo"),
        (0, "v_dot2_i32_i16 v23", "v_dot2_u32_u16 v23", "dot2 signed vs unsigned"),
        (1, "v_dot4_i32_i8 v25, v19, v20, v21 clamp", "v_dot4_i32_i8 v25, v19, v20, v21", "dot4_i32_i8 clamp"),
        (0, "v_dot8_i32_i4 v27", "v_dot8_u32_u4 v27", "dot8 signed vs unsigned"),
        (3, "v_dot8_u32_u4 v28, v19, v20, v21 clamp", "v_dot8_u32_u4 v28, v19, v20, v21", "dot8_u32_u4 clamp"),
    ):
        mutated = [line.replace(old, new) for line in program]
        assert mutated != program, label
        source.write_text(translate(mutated), encoding="ascii")
        compile_shader(compiler, source, shader)
        wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, cases[set_idx][0]],
                               capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Mutation of {label} escaped detection")
# Packed 16-bit integer VOP3P fixture covering all 14 forms: add/sub/min/max (i16/u16),
# mul_lo_u16, mad (i16/u16), and lshlrev/lshrrev/ashrrev shifts, plus op_sel/op_sel_hi, clamp,
# and inline -1. Exports all four bytes of the packed accumulator as exact [0, 255] integers.
PK_INT16_BODY = [
    "v_mov_b32_e32 v18, s44",
    "v_mov_b32_e32 v19, s45",
    "v_mov_b32_e32 v20, s46",
    "v_mov_b32_e32 v21, s47",
    "v_pk_add_i16 v22, v18, v19 op_sel:[1,0] op_sel_hi:[0,1] clamp",
    "v_pk_add_u16 v23, v18, v19 clamp",
    "v_pk_sub_i16 v24, v18, v19 clamp",
    "v_pk_sub_u16 v25, v18, v19 clamp",
    "v_pk_mad_i16 v26, v18, v19, v20 clamp",
    "v_pk_mad_u16 v27, v18, v19, v20 clamp",
    "v_pk_min_i16 v28, v22, v24",
    "v_pk_max_i16 v29, v28, v26",
    "v_pk_min_u16 v28, v23, v25",
    "v_pk_max_u16 v28, v28, v27",
    "v_pk_mul_lo_u16 v29, v29, v28",
    "v_pk_add_i16 v29, v29, -1",
    "v_pk_lshlrev_b16 v22, v21, v29 op_sel:[0,1] op_sel_hi:[1,0]",
    "v_pk_lshrrev_b16 v23, v21, v28",
    "v_pk_ashrrev_i16 v24, v21, v18",
    "v_pk_add_u16 v30, v22, v23",
    "v_pk_sub_u16 v30, v30, v24",
    "v_pk_add_u16 v30, v30, v25",
    "v_pk_add_u16 v30, v30, v26",
    "v_pk_add_u16 v30, v30, v29",
    "v_pk_lshlrev_b16 v25, 8, v30 op_sel_hi:[0,1]",
    "v_pk_lshrrev_b16 v25, 8, v25 op_sel_hi:[0,1]",
    "v_pk_lshrrev_b16 v26, 8, v30 op_sel_hi:[0,1]",
    "v_pk_add_u16 v27, v25, 0 op_sel:[1,0] op_sel_hi:[0,0]",
    "v_pk_add_u16 v28, v26, 0 op_sel:[1,0] op_sel_hi:[0,0]",
    "v_cvt_f32_f16_e32 v12, v25",
    "v_mul_f32_e32 v12, 0x4b800000, v12",
    "v_cvt_f32_f16_e32 v13, v26",
    "v_mul_f32_e32 v13, 0x4b800000, v13",
    "v_cvt_f32_f16_e32 v14, v27",
    "v_mul_f32_e32 v14, 0x4b800000, v14",
    "v_cvt_f32_f16_e32 v15, v28",
    "v_mul_f32_e32 v15, 0x4b800000, v15",
]

PK_INT16_SETS = (
    # Set 1: mixed signed/unsigned lanes + shift counts >= 16 (testing & 15 masking and ashrrev sign fill).
    (0xf0080015, 0x0010fffe, 0x0007fff9, 0x00130012),
    # Set 2: saturates all 6 clamping ops (signed +/-32767/-32768 and unsigned 0/65535 boundaries).
    (0x70009000, 0x7000a000, 0x4000c000, 0x00020003),
    # Set 3: signed/unsigned extremes (INT16_MIN/MAX, UINT16_MAX) and max 4-bit shift count 15.
    (0x80007fff, 0xffff0001, 0x0002fffd, 0x000f0001),
)


def pk_int16_reference(w18, w19, w20, w21):
    def u16(w, hi):
        return (w >> 16) & 0xffff if hi else w & 0xffff

    def i16(w, hi):
        v = u16(w, hi)
        return v - 0x10000 if v >= 0x8000 else v

    def pack(lo, hi):
        return (lo & 0xffff) | ((hi & 0xffff) << 16)

    def sat_i(v):
        return max(-32768, min(32767, v)) & 0xffff

    def sat_u(v):
        return max(0, min(65535, v)) & 0xffff

    v22 = pack(sat_i(i16(w18, 1) + i16(w19, 0)), sat_i(i16(w18, 0) + i16(w19, 1)))
    v23 = pack(sat_u(u16(w18, 0) + u16(w19, 0)), sat_u(u16(w18, 1) + u16(w19, 1)))
    v24 = pack(sat_i(i16(w18, 0) - i16(w19, 0)), sat_i(i16(w18, 1) - i16(w19, 1)))
    v25 = pack(sat_u(u16(w18, 0) - u16(w19, 0)), sat_u(u16(w18, 1) - u16(w19, 1)))
    v26 = pack(sat_i(i16(w18, 0) * i16(w19, 0) + i16(w20, 0)), sat_i(i16(w18, 1) * i16(w19, 1) + i16(w20, 1)))
    v27 = pack(sat_u(u16(w18, 0) * u16(w19, 0) + u16(w20, 0)), sat_u(u16(w18, 1) * u16(w19, 1) + u16(w20, 1)))
    v28 = pack(min(i16(v22, 0), i16(v24, 0)), min(i16(v22, 1), i16(v24, 1)))
    v29 = pack(max(i16(v28, 0), i16(v26, 0)), max(i16(v28, 1), i16(v26, 1)))
    v28 = pack(min(u16(v23, 0), u16(v25, 0)), min(u16(v23, 1), u16(v25, 1)))
    v28 = pack(max(u16(v28, 0), u16(v27, 0)), max(u16(v28, 1), u16(v27, 1)))
    v29 = pack((u16(v29, 0) * u16(v28, 0)) & 0xffff, (u16(v29, 1) * u16(v28, 1)) & 0xffff)
    v29 = pack((i16(v29, 0) - 1) & 0xffff, (i16(v29, 1) - 1) & 0xffff)
    v22 = pack((u16(v29, 1) << (u16(w21, 0) & 15)) & 0xffff, (u16(v29, 0) << (u16(w21, 1) & 15)) & 0xffff)
    v23 = pack(u16(v28, 0) >> (u16(w21, 0) & 15), u16(v28, 1) >> (u16(w21, 1) & 15))
    v24 = pack((i16(w18, 0) >> (u16(w21, 0) & 15)) & 0xffff, (i16(w18, 1) >> (u16(w21, 1) & 15)) & 0xffff)
    v30 = pack((u16(v22, 0) + u16(v23, 0)) & 0xffff, (u16(v22, 1) + u16(v23, 1)) & 0xffff)
    v30 = pack((u16(v30, 0) - u16(v24, 0)) & 0xffff, (u16(v30, 1) - u16(v24, 1)) & 0xffff)
    v30 = pack((u16(v30, 0) + u16(v25, 0)) & 0xffff, (u16(v30, 1) + u16(v25, 1)) & 0xffff)
    v30 = pack((u16(v30, 0) + u16(v26, 0)) & 0xffff, (u16(v30, 1) + u16(v26, 1)) & 0xffff)
    v30 = pack((u16(v30, 0) + u16(v29, 0)) & 0xffff, (u16(v30, 1) + u16(v29, 1)) & 0xffff)
    return tuple(float((v30 >> (8 * k)) & 0xff) for k in range(4))


def check_pk_int16_draw(directory, compiler, renderer, backend):
    program = with_constants(PK_INT16_BODY)
    source = directory / "pk_int16.hlsl"
    shader = directory / "pk_int16.cso"
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    cases = []
    for idx, values in enumerate(PK_INT16_SETS):
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *values)
        expected = pk_int16_reference(*values)
        case = directory / f"pk_int16-{idx}.case"
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        cases.append((case, "pass"))
    run_pixel_batch(renderer, backend, shader, cases)
    for set_idx, old, new, label in (
        (0, "v_pk_ashrrev_i16 v24, v21, v18", "v_pk_lshrrev_b16 v24, v21, v18", "ashrrev sign extension"),
        (0, "v_pk_min_i16 v28, v22, v24", "v_pk_min_u16 v28, v22, v24", "signed vs unsigned min"),
        (1, "v_pk_sub_u16 v25, v18, v19 clamp", "v_pk_sub_u16 v25, v18, v19", "sub_u16 clamp"),
        (0, "op_sel:[1,0] op_sel_hi:[0,1] clamp", "op_sel_hi:[1,1] clamp", "op_sel cross-lane swap"),
        (1, "v_pk_mad_i16 v26, v18, v19, v20 clamp", "v_pk_mad_i16 v26, v18, v19, v20", "mad_i16 clamp"),
    ):
        mutated = [line.replace(old, new) for line in program]
        assert mutated != program, label
        source.write_text(translate(mutated), encoding="ascii")
        compile_shader(compiler, source, shader)
        wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, cases[set_idx][0]],
                               capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Mutation of {label} escaped detection")


def check_f16_denorm_controls():
    """CPU-side checks behind F16_DENORM_MODE: references and the lowering it relies on."""
    assert F16_DENORM_MODE == "preserve"
    assert f16_denorm_words(flush=False) == (0x84018001, 0x03ff0001, 0x80000200, 0x05fe0002)
    assert f16_denorm_words(flush=True) == (0x84018000, 0x00000000, 0x80000000, 0x00000000)
    body = translate(uncompressed_export(F16_DENORM_BODY)).split("main(", 1)[1]
    # The mode argument holds only while binary16 values are built and narrowed from their encodings;
    # a driver conversion or a native half type would make the result depend on the device's f16 mode.
    for token in ("f32tof16", "f16tof32", "half", "min16float", "float16_t"):
        assert token not in body, token
    # Presence only: the guarantee is that halves go through these helpers, not how many calls appear.
    assert "widen16(" in body and "pack16_rne(" in body, body


def check_e64_integer_lowering():
    """Static checks for the integer e64 forms of e64_integer_group() and for v_xnor_b32 in both encodings."""
    # v18 is written by the mov at index 20 and the four group instructions follow at 21..24.
    a, b, c, v = "params[0].w", "params[1].x", "asuint(asfloat(params[1].y))", "asuint(alu20)"
    expected = {
        "shift": (f"asfloat({b} << ({a} & 31u))", f"asfloat({b} >> ({a} & 31u))",
                  f"asfloat(asuint(asint({b}) >> ({a} & 31u)))", f"asfloat(~{b})"),
        "bitwise": (f"asfloat({a} & {b})", f"asfloat({a} | {b})", f"asfloat({a} ^ {b})", f"asfloat(~({a} ^ {b}))"),
        "xnor_byte": (f"asfloat(~({a} ^ {v}))", f"float(({c} >> 16u) & 0xffu)", f"float(({v} >> 16u) & 0xffu)",
                      f"saturate(float(({c} >> 16u) & 0xffu))"),
    }
    assert expected.keys() == E64_INT_BODIES.keys() == E64_INT_RESULTS.keys() == E64_INT_WRONG.keys() == E64_INT_ROWS.keys()
    for group, expressions in expected.items():
        program = uncompressed_export(e64_integer_group(group))
        assert program.index("v_mov_b32_e32 v18, s45") == 20, group
        source = translate_int(program)
        for offset, expression in enumerate(expressions, start=21):
            statement = f"    precise float alu{offset} = {expression};"
            assert statement in source.splitlines(), (group, statement)
    assert set(E64_INT_OPS) <= ACCEPTED, set(E64_INT_OPS) - ACCEPTED
    assert len(E64_INT_OPS) == 10
    # v_xnor_b32 takes the operand shapes of v_xor_b32 and nothing else: no float modifier, no clamp, two
    # sources, a written register, and no SDWA spelling without all of its fields.
    for good, bad in (
        ("v_xnor_b32_e32 v12, s44, v18", "v_xnor_b32_e32 v12, s44, |v18|"),
        ("v_xnor_b32_e32 v12, s44, v18", "v_xnor_b32_e32 v12, s44, -v18"),
        ("v_xnor_b32_e64 v12, s44, v18", "v_xnor_b32_e64 v12, abs(s44), v18"),
        ("v_xnor_b32_e64 v12, s44, v18", "v_xnor_b32_e64 v12, s44, v18 clamp"),
        ("v_xnor_b32_e64 v12, 64, v18", "v_xnor_b32_e64 v12, s44"),
        ("v_xnor_b32_e64 v12, v18, s44", "v_xnor_b32_e64 v12, s44, v18, s45"),
        ("v_xnor_b32_e32 v12, s44, v18", "v_xnor_b32_e32 v12, s44, v106"),
        ("v_xnor_b32_e32 v12, s44, v18", "v_xnor_b32_sdwa v12, v18, v18 src0_sel:DWORD src1_sel:DWORD"),
    ):
        translate_int(uncompressed_export(["v_mov_b32_e32 v18, s45", good]))
        must_reject(translate_int, uncompressed_export(["v_mov_b32_e32 v18, s45", bad]))
    xnor_source = translate_int(uncompressed_export(["v_mov_b32_e32 v18, s45", "v_xnor_b32_e64 v12, s44, v18"]))
    xor_source = translate_int(uncompressed_export(["v_mov_b32_e32 v18, s45", "v_xor_b32_e64 v12, s44, v18"]))
    assert xnor_source != xor_source
    # A float modifier on the byte conversion stays refused in the e64 spelling as well.
    must_reject(translate_int, uncompressed_export(["v_cvt_f32_ubyte2_e64 v12, -s46"]))


def check_sdwa_integer_lowering():
    """Static checks for the SDWA forms of the 32-bit integer operations in sdwa_integer_group()."""
    assert SDWA_INT_BODIES.keys() == SDWA_INT_RESULTS.keys() == SDWA_INT_WRONG.keys() == SDWA_INT_ROWS.keys()
    for group in SDWA_INT_BODIES:
        translate_int(uncompressed_export(sdwa_integer_group(group)))
        sdwa_integer_cases(group)
    assert set(SDWA_INT_OPS) <= ACCEPTED, set(SDWA_INT_OPS) - ACCEPTED
    assert len(SDWA_INT_OPS) == 15 and set(SDWA_INT_OPS) == {f"v_{op}_sdwa" for op in pixel_aot.SDWA_INT}

    def statement(*body):
        """The HLSL of the last instruction of body, with its own temporary renamed to `r`."""
        program = uncompressed_export(["v_mov_b32_e32 v18, s45", *body])
        index = program.index(body[-1])
        lines = [line for line in translate_int(program).splitlines()
                 if line.startswith(f"    precise float alu{index} = ")]
        assert len(lines) == 1, (body, lines)
        return lines[0].replace(f"alu{index} ", "r ", 1)

    a, b, v = "params[0].w", "params[1].x", "asuint(alu20)"
    sdwa = "dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD"
    # With whole-dword selections every form is its e32 statement, up to the parentheses of the placement.
    for op, (arity, _) in pixel_aot.SDWA_INT.items():
        if arity == 1:
            plain = statement(f"v_{op}_e32 v12, v18")
            whole = statement(f"v_{op}_sdwa v12, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD")
        else:
            plain = statement(f"v_{op}_e32 v12, s44, v18")
            whole = statement(f"v_{op}_sdwa v12, s44, v18 {sdwa}")
        inner = plain.removeprefix("    precise float r = asfloat(").removesuffix(");")
        if op in ("min_i32", "max_i32"):
            # The e32 statement converts the signed result directly; the SDWA one goes through its unsigned word.
            inner = f"asuint({inner})".replace("asint(params[0].w)", f"asint({a})").replace("asint(alu20)", f"asint({v})")
        assert whole == f"    precise float r = asfloat(({inner}));", (op, plain, whole)
    # Selections: zero extension is a shift and a mask, sign extension an arithmetic shift pair, and the
    # placement masks the result into its field.
    assert statement("v_and_b32_sdwa v12, s44, sext(s45) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD "
                     "src1_sel:BYTE_1") == f"    precise float r = asfloat(({a} & asuint(asint({b} << 16u) >> 24u)));"
    assert statement("v_or_b32_sdwa v13, s44, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD "
                     "src1_sel:WORD_1") == f"    precise float r = asfloat(({a} | (({v} >> 16u) & 0xffffu)));"
    assert statement("v_xor_b32_sdwa v14, s44, s45 dst_sel:BYTE_3 dst_unused:UNUSED_PAD src0_sel:BYTE_0 "
                     "src1_sel:BYTE_2") == (f"    precise float r = asfloat(((((({a} >> 0u) & 0xffu) ^ "
                                            f"(({b} >> 16u) & 0xffu)) & 0xffu) << 24u));")
    assert statement("v_mov_b32_e32 v15, s47", "v_not_b32_sdwa v15, s46 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE "
                     "src0_sel:BYTE_2") == ("    precise float r = asfloat(((asuint(alu21) & 0xffff0000u) | "
                                            "(((~((params[1].y >> 16u) & 0xffu)) & 0xffffu) << 0u)));")
    good = f"v_and_b32_sdwa v12, s44, v18 {sdwa}"
    translate_int(uncompressed_export(["v_mov_b32_e32 v18, s45", good]))
    for bad, message in (
        # UNUSED_SEXT stays refused, as for the SDWA moves.
        (good.replace("dst_sel:DWORD dst_unused:UNUSED_PAD", "dst_sel:WORD_0 dst_unused:UNUSED_SEXT"),
         "UNUSED_SEXT is not lowered"),
        # UNUSED_PRESERVE keeps bits of a tracked numeric destination only.
        (f"v_and_b32_sdwa v20, s44, v18 {sdwa}".replace("dst_sel:DWORD dst_unused:UNUSED_PAD",
                                                         "dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE"),
         "UNUSED_PRESERVE needs a tracked numeric destination"),
        (f"v_and_b32_sdwa v0, s44, v18 {sdwa}".replace("dst_sel:DWORD dst_unused:UNUSED_PAD",
                                                        "dst_sel:BYTE_1 dst_unused:UNUSED_PRESERVE"),
         "UNUSED_PRESERVE needs a tracked numeric destination"),
        # clamp, inline constants, literals, float modifiers and incomplete or malformed fields.
        (good.replace(" dst_sel", " clamp dst_sel"), "Unsupported SDWA integer operands"),
        (good.replace("s44,", "64,"), "Unsupported SDWA integer operands"),
        (good.replace("s44,", "1.0,"), "Unsupported SDWA integer operands"),
        (good.replace("s44,", "0x12345,"), "Unsupported SDWA integer operands"),
        (good.replace("s44,", "-s44,"), "Unsupported SDWA integer operands"),
        (good.replace("v18 ", "|v18| "), "Unsupported SDWA integer operands"),
        (good.replace("s44,", "sext(-s44),"), "Unsupported SDWA integer operands"),
        ("v_and_b32_sdwa v12, s44, v18", "Unsupported SDWA integer operands"),
        ("v_and_b32_sdwa v12, s44, v18 src0_sel:DWORD src1_sel:DWORD", "Unsupported SDWA integer operands"),
        (good.replace("src0_sel:DWORD", "src0_sel:WORD_2"), "Unsupported SDWA integer operands"),
        (good.replace("s44,", "sext(s44,"), "Unbalanced sext()"),
        (good.replace("s44,", "s44),"), "Unbalanced sext()"),
        # The source count and the selections must agree with the operation.
        (good.replace(" src1_sel:DWORD", ""), "takes 2 source(s)"),
        (good.replace("s44, v18", "v18"), "takes 2 source(s)"),
        (f"v_not_b32_sdwa v12, s44, v18 {sdwa}", "takes 1 source(s)"),
        (f"v_not_b32_sdwa v12, v18 {sdwa}", "takes 1 source(s)"),
        # An unwritten register is not a numeric source.
        (good.replace("v18", "v0"), "Expected a numeric register"),
    ):
        expect_value_error(translate_int, uncompressed_export(["v_mov_b32_e32 v18, s45", bad]), message)
    must_reject(translate_int, uncompressed_export(["v_mov_b32_e32 v18, s45", good.replace("v18", "v106")]))
    # Operations outside the table keep their refusal in the SDWA spelling.
    for bad in (f"v_bfi_b32_sdwa v12, s44, v18 {sdwa}", f"v_mul_u32_u24_sdwa v12, s44, v18 {sdwa}",
                f"v_add_nc_i32_sdwa v12, s44, v18 {sdwa}", f"v_min3_u32_sdwa v12, s44, v18 {sdwa}"):
        must_reject(translate_int, uncompressed_export(["v_mov_b32_e32 v18, s45", bad]))


def check_pack_int16_lowering():
    """Static checks for v_cvt_pk_u16_u32 and v_cvt_pk_i16_i32 in the programs of pack_group()."""
    a = "params[0].w"

    def u16(x, y):
        return f"asfloat(min({x}, 65535u) | (min({y}, 65535u) << 16u));"

    def i16(x, y):
        return (f"asfloat((asuint(clamp(asint({x}), -32768, 32767)) & 0xffffu) | "
                f"(asuint(clamp(asint({y}), -32768, 32767)) << 16u));")

    # The prefix writes alu20..alu23 (B, B >> 16, B >> 12, B >> 16 arithmetic); the body follows at 24.
    hi, mid, sext = "asuint(alu21)", "asuint(alu22)", "asuint(alu23)"
    expected = {
        "unsigned": (u16(f"{PACK_LITERAL}u", hi), u16(mid, hi), u16(hi, a), u16(sext, hi)),
        "signed": (i16(a, sext), i16(sext, a), i16(hi, sext), i16(f"{(-16) & 0xffffffff}u", sext)),
    }
    assert expected.keys() == PACK_BODIES.keys() == PACK_RESULTS.keys() == PACK_WRONG.keys()
    for group, statements in expected.items():
        program = uncompressed_export(pack_group(group))
        assert program.index("v_mov_b32_e32 v18, s45") == 20, group
        source = translate_int(program)
        for statement in statements:
            assert sum(line.endswith(" = " + statement) for line in source.splitlines()) == 1, (group, statement)
        assert {line.split()[0] for line in PACK_BODIES[group]} & set(PACK_OPS), group
    assert set(PACK_OPS) <= ACCEPTED, set(PACK_OPS) - ACCEPTED
    # The Python references at the bounds and one step past them.
    assert [_sat_u16(v) for v in (0, 0xffff, 0x10000, 0xffffffff)] == [0, 0xffff, 0xffff, 0xffff]
    assert [_sat_i16(v) for v in (0x7fff, 0x8000, 0xffff8000, 0xffff7fff, 0xffffffff, 0x80000000)] == [
        0x7fff, 0x7fff, 0x8000, 0x8000, 0xffff, 0x8000]
    assert PACK_RESULTS["signed"](*PACK_ROWS[0]) == (0xC1257FFF, 0x3FFFC125, 0xC1257FFF, 0xC125FFF0)
    assert PACK_RESULTS["unsigned"](*PACK_ROWS[1]) == (0x00FFFFFF, 0x00FF0FFF, 0xBFFF00FF, 0x00FF00FF)
    # Two sources in the VOP3 encoding and nothing else: no e32 spelling (gfx10 has none), no clamp, no float
    # modifier, an in-range destination, written registers and 32-bit literals only.
    for op in PACK_OPS:
        good = f"{op} v12, s44, v18"
        translate_int(uncompressed_export(["v_mov_b32_e32 v18, s45", good]))
        translate_int(uncompressed_export(["v_mov_b32_e32 v18, s45", f"{op}_e64 v12, s44, v18"]))
        for bad in (
            f"{op}_e32 v12, s44, v18",
            f"{op} v12, s44, v18 clamp",
            f"{op} v12, s44",
            f"{op} v12, s44, v18, s46",
            f"{op} v12, s44, -v18",
            f"{op} v12, |v18|, s44",
            f"{op} v256, s44, v18",
            f"{op} v12, 0x100000000, v18",
            f"{op} v12, s44, v106",
            f"{op}_sdwa v12, v18, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
        ):
            must_reject(translate_int, uncompressed_export(["v_mov_b32_e32 v18, s45", bad]))
    for op in PACK_OPS:
        expect_value_error(translate_int, uncompressed_export(["v_mov_b32_e32 v18, s45", f"{op}_e32 v12, s44, v18"]),
                           "does not support e32 encoding")
    # The float packs stay unsupported: no public source found here states their rounding and bounds.
    for neighbour in (
        "v_cvt_pknorm_i16_f32 v12, s44, v18",
        "v_cvt_pknorm_u16_f32 v12, s44, v18",
        "v_cvt_pknorm_i16_f16 v12, s44, v18",
        "v_cvt_pknorm_u16_f16 v12, s44, v18",
        "v_cvt_pk_u8_f32 v12, s44, v18, v18",
        "v_sat_pk_u8_i16_e32 v12, v18",
        "v_sat_pk_u8_i16_e64 v12, v18",
    ):
        must_reject(translate_int, uncompressed_export(["v_mov_b32_e32 v18, s45", neighbour]))


def check_carry_out_lowering():
    """Static checks for v_add_co_u32, v_sub_co_u32 and v_subrev_co_u32 as written by carry_out_group()."""
    # v18 and v19 are written by the movs at 20 and 21; the first operation is 22, its select 23, the second
    # 24, its select 25.
    a, v, c, d = "params[0].w", "asuint(alu20)", "asfloat(params[1].y)", "alu21"
    expected = {
        "v_add_co_u32": ((f"{a} + {v}", f"co22 < {a}"), (f"{v} + {a}", f"co24 < {v}")),
        "v_sub_co_u32": ((f"{a} - {v}", f"{v} > {a}"), (f"{v} - {a}", f"{a} > {v}")),
        "v_subrev_co_u32": ((f"{v} - {a}", f"{a} > {v}"), (f"{a} - {v}", f"{v} > {a}")),
    }
    assert set(expected) == set(CARRY_OUT_OPS) == set(CARRY_OUT_ROWS) == set(CARRY_OUT_WRONG)
    for op, pairs in expected.items():
        program = uncompressed_export(carry_out_group(op))
        assert program.index("v_mov_b32_e32 v18, s45") == 20, op
        source = translate_int(program).splitlines()
        for index, (result, carry) in zip((22, 24), pairs):
            for statement in (f"    uint co{index} = {result};", f"    bool cmp{index} = {carry};",
                              f"    precise float alu{index} = asfloat(co{index});",
                              f"    precise float alu{index + 1} = cmp{index} ? {d} : {c};"):
                assert statement in source, (op, statement)
    assert set(CARRY_OUT_OPS) <= ACCEPTED, set(CARRY_OUT_OPS) - ACCEPTED
    # The carry goes to vcc, vcc_lo, one SGPR or an SGPR pair, and both sources take a register, an inline
    # constant or a literal. Nothing else is accepted: no suffix, clamp, float modifier, exec or VGPR as the
    # mask, missing or extra operand, unwritten register, DPP controls, or the carry-in forms.
    for op in CARRY_OUT_OPS:
        for good in (f"{op} v12, vcc, s44, v18", f"{op} v12, vcc_lo, v18, s44", f"{op} v12, s60, v18, v18",
                     f"{op} v12, s[60:61], 64, v18", f"{op} v12, vcc, v18, 0x12345678", f"{op} v12, vcc, -16, s45"):
            translate_int(uncompressed_export(["v_mov_b32_e32 v18, s45", good]))
        for bad in (f"{op} v12, vcc, s44, v18 clamp", f"{op}_e64 v12, vcc, s44, v18", f"{op}_e32 v12, vcc, s44, v18",
                    f"{op} v12, s44, v18", f"{op} v12, exec, s44, v18", f"{op} v12, v13, s44, v18",
                    f"{op} v12, vcc, -s44, v18", f"{op} v12, vcc, s44, |v18|", f"{op} v12, vcc, s44, v106",
                    f"{op} v12, vcc, s44", f"{op} v12, vcc, s44, v18, vcc", f"{op} vcc, s44, v18",
                    f"{op} v12, vcc, s44, v18 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
                    f"{op.replace('_co_', '_co_ci_')}_e64 v12, vcc, s44, v18, vcc"):
            must_reject(translate_int, uncompressed_export(["v_mov_b32_e32 v18, s45", bad]))
    # The mask the operation wrote is the one a select reads: a select on another register is refused, and a
    # later comparison into the same register replaces the carry.
    must_reject(translate_int, uncompressed_export(
        ["v_mov_b32_e32 v18, s45", "v_add_co_u32 v12, s[60:61], s44, v18", "v_cndmask_b32_e64 v13, s46, s47, s[62:63]"]))
    replaced = translate_int(uncompressed_export(
        ["v_mov_b32_e32 v18, s45", "v_add_co_u32 v12, vcc, s44, v18", "v_cmp_lt_u32_e64 vcc, s44, v18",
         "v_cndmask_b32_e64 v13, s46, s47, vcc"])).splitlines()
    assert f"    precise float alu23 = cmp22 ? asfloat(params[1].z) : {c};" in replaced, replaced
    sources = {op: translate_int(uncompressed_export(carry_out_group(op))) for op in CARRY_OUT_OPS}
    assert len(set(sources.values())) == 3


def check_carry_in_lowering():
    """Static checks for v_add_co_ci_u32, v_sub_co_ci_u32 and v_subrev_co_ci_u32 as written by carry_in_group()."""
    # v18 and v19 are written by the movs at 20 and 21, the carry-out forms at 22 (vcc_lo) and 23 (s60) write the
    # carry-ins, the e32 form is 24, the e64 form 25 and the selects 26 and 27.
    c, d = "params[1].y", "asuint(alu21)"
    expected = {
        "v_add_co_ci_u32": (f"{c} + {d}", f"{d} + {c}", "cp{0} + ci{0}", "cp{0} < {1} || co{0} < cp{0}"),
        "v_sub_co_ci_u32": (f"{c} - {d}", f"{d} - {c}", "cp{0} - ci{0}", "{2} > {1} || ci{0} > cp{0}"),
        "v_subrev_co_ci_u32": (f"{d} - {c}", f"{c} - {d}", "cp{0} - ci{0}", "{1} > {2} || ci{0} > cp{0}"),
    }
    assert set(expected) == set(CARRY_IN_OPS) == set(CARRY_IN_ROWS) == set(CARRY_IN_WRONG)
    for op, (first, second, result, carry) in expected.items():
        program = uncompressed_export(carry_in_group(op))
        assert program.index("v_mov_b32_e32 v18, s45") == 20, op
        source = translate_int(program).splitlines()
        for index, partial, cin, src0, src1 in ((24, first, "cmp22", c, d), (25, second, "cmp23", d, c)):
            for statement in (f"    uint ci{index} = {cin} ? 1u : 0u;", f"    uint cp{index} = {partial};",
                              f"    uint co{index} = {result.format(index)};",
                              f"    bool cmp{index} = {carry.format(index, src0, src1)};",
                              f"    precise float alu{index} = asfloat(co{index});"):
                assert statement in source, (op, statement)
        assert any(line.startswith("    precise float alu26 = cmp24 ?") for line in source), (op, source)
        assert any(line.startswith("    precise float alu27 = cmp25 ?") for line in source), (op, source)
    carry_in_mnemonics = {f"{op}_{encoding}" for op in CARRY_IN_OPS for encoding in ("e32", "e64")}
    assert carry_in_mnemonics <= ACCEPTED, carry_in_mnemonics - ACCEPTED
    sources = {op: translate_int(uncompressed_export(carry_in_group(op))) for op in CARRY_IN_OPS}
    assert len(set(sources.values())) == 3
    # The e32 form takes vcc or vcc_lo for both masks and a VGPR as src1, which is all its encoding has room for.
    # The e64 form takes vcc, vcc_lo, one SGPR or an SGPR pair for each mask, both of the same width. Both sources
    # take a register, an inline constant or a literal. The carry-in must be a lane mask the translator tracks.
    setup = ["v_mov_b32_e32 v18, s45", "v_add_co_u32 v16, vcc_lo, s44, v18", "v_add_co_u32 v17, s60, s44, v18",
             "v_add_co_u32 v20, s[62:63], s44, v18"]
    for op in CARRY_IN_OPS:
        for good in (f"{op}_e32 v12, vcc_lo, s44, v18, vcc_lo", f"{op}_e32 v12, vcc, v18, v18, vcc",
                     f"{op}_e32 v12, vcc_lo, 0x12345678, v18, vcc_lo", f"{op}_e32 v12, vcc_lo, -16, v18, vcc_lo",
                     f"{op}_e64 v12, s61, v18, s44, s60", f"{op}_e64 v12, s60, 64, v18, s60",
                     f"{op}_e64 v12, vcc_lo, s44, v18, vcc_lo", f"{op}_e64 v12, s[64:65], s44, v18, s[62:63]",
                     f"{op}_e64 v12, vcc_lo, v18, 0x12345678, s60", f"{op}_e64 v12, s61, 2.0, v18, vcc_lo"):
            translate_int(uncompressed_export([*setup, good]))
        for bad in (f"{op} v12, vcc_lo, s44, v18, vcc_lo", f"{op}_e32 v12, vcc_lo, v18, s44, vcc_lo",
                    f"{op}_e32 v12, s61, s44, v18, vcc_lo", f"{op}_e32 v12, vcc_lo, s44, v18, s60",
                    f"{op}_e64 v12, s61, v18, s44, s60 clamp", f"{op}_e64 v12, s61, -v18, s44, s60",
                    f"{op}_e64 v12, s61, v18, |s44|, s60", f"{op}_e64 v12, exec_lo, v18, s44, s60",
                    f"{op}_e64 v12, s61, v18, s44, exec_lo", f"{op}_e64 v12, v13, v18, s44, s60",
                    f"{op}_e64 v12, s61, v18, s44, v13", f"{op}_e64 v12, s61, v18, s44",
                    f"{op}_e64 v12, s61, v18, s44, s60, s60", f"{op}_e64 v12, s61, v18, v106, s60",
                    f"{op}_e64 v12, s[64:65], v18, s44, s60", f"{op}_e64 v12, s61, v18, s44, s[62:63]",
                    f"{op}_e64 v12, s61, v18, s44, s58", f"{op}_e64 v12, s61, v18, s44, s61",
                    f"{op}_dpp v12, vcc_lo, v18, v18, vcc_lo quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
                    f"{op}_sdwa v12, vcc_lo, v18, v18, vcc_lo dst_sel:DWORD dst_unused:UNUSED_PAD "
                    "src0_sel:DWORD src1_sel:DWORD"):
            must_reject(translate_int, uncompressed_export([*setup, bad]))
    # A carry-in that is no lane mask at all: vcc before anything writes it, and an SGPR holding a constant.
    for op in CARRY_IN_OPS:
        must_reject(translate_int, uncompressed_export(
            ["v_mov_b32_e32 v18, s45", f"{op}_e32 v12, vcc_lo, s44, v18, vcc_lo"]))
        must_reject(translate_int, uncompressed_export(
            ["v_mov_b32_e32 v18, s45", f"{op}_e64 v12, s61, v18, s44, s46"]))
    # The carry-out replaces the mask it writes: a select on that mask reads the new carry, and a later carry-in
    # from the same register reads it too.
    chained = translate_int(uncompressed_export(
        [*setup, "v_add_co_ci_u32_e32 v12, vcc_lo, s44, v18, vcc_lo", "v_add_co_ci_u32_e32 v13, vcc_lo, s44, v18, vcc_lo",
         "v_cndmask_b32_e64 v14, s46, s47, vcc_lo"])).splitlines()
    assert "    uint ci25 = cmp24 ? 1u : 0u;" in chained, chained
    assert any(line.startswith("    precise float alu26 = cmp25 ?") for line in chained), chained


def check_sad_lowering():
    """Static checks for v_sad_hi_u8, v_sad_u32 and v_msad_u8 in the programs of sad_group()."""
    # v18 is written by the mov at index 20 and the four group instructions follow at 21..24.
    a, b, c, v = "params[0].w", "params[1].x", "params[1].y", "asuint(alu20)"
    expected = {
        "bytes": (f"(sad4_u8({a}, {v}, false) << 16u) + {c}", f"(sad4_u8({v}, {a}, false) << 16u) + 0u",
                  f"sad4_u8({a}, {v}, true) + {c}", f"sad4_u8({v}, {a}, true) + {c}"),
        "dword": (f"(max({a}, {v}) - min({a}, {v})) + {c}", f"(max({v}, {a}) - min({v}, {a})) + 64u",
                  f"sad4_u8({a}, {SAD_LITERAL}u, true) + {v}", f"(sad4_u8({b}, {a}, false) << 16u) + {v}"),
    }
    assert expected.keys() == SAD_BODIES.keys() == SAD_RESULTS.keys() == SAD_WRONG.keys() == SAD_ROWS.keys()
    for group, expressions in expected.items():
        program = uncompressed_export(sad_group(group))
        assert program.index("v_mov_b32_e32 v18, s45") == 20, group
        source = translate_int(program)
        for offset, expression in enumerate(expressions, start=21):
            statement = f"    precise float alu{offset} = asfloat({expression});"
            assert statement in source.splitlines(), (group, statement)
        # The byte-lane helper is emitted once, and only for a program that uses it.
        assert source.count("uint sad4_u8(uint a, uint b, bool masked)") == 1, group
    only_dword = translate_int(uncompressed_export(["v_mov_b32_e32 v18, s45", "v_sad_u32 v12, s44, v18, s46"]))
    assert "sad4_u8" not in only_dword
    assert set(SAD_OPS) <= ACCEPTED, set(SAD_OPS) - ACCEPTED
    assert SAD_OPS == ("v_msad_u8", "v_sad_hi_u8", "v_sad_u32")
    # The Python references agree with the worked values of the first rows.
    assert _sad4(0xC1007F90, 0x40C30011) == 0x242 and _sad4(0xC1007F90, 0x40C30011, True) == 0x1C3
    assert _sad4(0x40C30011, 0xC1007F90, True) == 0x17F
    assert SAD_RESULTS["dword"](*SAD_ROWS["dword"][0])[0] & 0xffffffff == 0x00EFFFDC
    # Three sources in the VOP3 encoding and nothing else: no e32 spelling, no clamp (LLVM 15 assembles the
    # bit, but the ISA text gives it no meaning for an integer result), no float modifier, an in-range
    # destination, written registers and 32-bit literals only.
    good = "v_sad_u32 v12, s44, v18, s46"
    for bad in (
        "v_sad_u32_e32 v12, s44, v18, s46",
        "v_sad_u32 v12, s44, v18, s46 clamp",
        "v_sad_u32 v12, s44, v18",
        "v_sad_u32 v12, s44, v18, s46, s46",
        "v_sad_u32 v12, s44, -v18, s46",
        "v_sad_hi_u8 v12, |v18|, s44, s46",
        "v_sad_hi_u8_e32 v12, s44, v18, s46",
        "v_sad_hi_u8 v12, s44, v18, s46 clamp",
        "v_msad_u8_e32 v12, s44, v18, s46",
        "v_msad_u8 v12, s44, v18, s46 clamp",
        "v_msad_u8 v12, s44, abs(v18), s46",
        "v_msad_u8 v256, s44, v18, s46",
        "v_msad_u8 v12, s44, 0x100000000, s46",
        "v_msad_u8 v12, s44, v106, s46",
    ):
        translate_int(uncompressed_export(["v_mov_b32_e32 v18, s45", good]))
        must_reject(translate_int, uncompressed_export(["v_mov_b32_e32 v18, s45", bad]))
    # The neighbours stay unsupported: the 8- and 16-bit lane sums without a shift, and the wide packed forms.
    for neighbour in (
        "v_sad_u8 v12, s44, v18, s46",
        "v_sad_u16 v12, s44, v18, s46",
        "v_qsad_pk_u16_u8 v[12:13], v[18:19], s44, v[20:21]",
        "v_mqsad_pk_u16_u8 v[12:13], v[18:19], s44, v[20:21]",
        "v_mqsad_u32_u8 v[12:15], v[18:19], s44, v[20:23]",
    ):
        must_reject(translate_int, uncompressed_export(["v_mov_b32_e32 v18, s45", neighbour]))


def check_nc_integer_lowering():
    """Static checks for the 32-bit no-carry add/sub forms of nc_integer_group(1..3)."""
    # v18 is written by the mov at index 20, and the four group instructions follow at 21..24. A and B are
    # s44 (params[0].w) and s45. sub is src0 - src1; subrev swaps the operands and is src1 - src0.
    a, b = "params[0].w", "asuint(alu20)"
    expected = {
        1: (f"{a} + {b}", f"{a} - {b}", f"{b} - {a}", f"{b} - {a}"),
        2: (f"{a} + {b}", f"{a} - {b}", f"{a} + {b}", f"{a} - {b}"),
        3: (f"{b} - 64u", f"64u + {b}", f"{a} - 64u", f"{a} - {b}"),
    }
    for group, expressions in expected.items():
        program = uncompressed_export(nc_integer_group(group))
        assert program.index("v_mov_b32_e32 v18, s45") == 20, group
        source = translate_int(program)
        for offset, expression in enumerate(expressions, start=21):
            statement = f"    precise float alu{offset} = asfloat({expression});"
            assert statement in source.splitlines(), (group, statement)
    assert set(NC_INT_OPS) <= ACCEPTED, set(NC_INT_OPS) - ACCEPTED
    # sub and subrev with the same operands must not lower alike.
    sub_source = translate_int(uncompressed_export(["v_mov_b32_e32 v18, s45", "v_sub_nc_u32_e64 v12, s44, v18"]))
    rev_source = translate_int(uncompressed_export(["v_mov_b32_e32 v18, s45", "v_subrev_nc_u32_e64 v12, s44, v18"]))
    assert sub_source != rev_source
    # A float-written inline constant on an integer form is taken as its binary32 bits, as for the #495 forms.
    float_source = translate_int(uncompressed_export(["v_mov_b32_e32 v18, s45", "v_add_nc_u32_e64 v12, 1.0, v18"]))
    assert "1065353216u" in float_source, float_source
    neg_source = translate_int(uncompressed_export(["v_mov_b32_e32 v18, s45", "v_add_nc_u32_e64 v12, -1.0, v18"]))
    assert "3212836864u" in neg_source, neg_source
    # Each refused spelling is paired with its nearest accepted one, so a refusal for another reason shows up.
    for bad, good in (
        ("v_add_nc_u32_e64 v12, s44, v18 clamp", "v_add_nc_u32_e64 v12, s44, v18"),
        ("v_subrev_nc_u32_e64 v12, s44, v18 clamp", "v_subrev_nc_u32_e64 v12, s44, v18"),
        ("v_add_nc_i32 v12, s44, v18 clamp", "v_add_nc_i32 v12, s44, v18"),
        ("v_sub_nc_i32 v12, s44, v18 clamp", "v_sub_nc_i32 v12, s44, v18"),
        ("v_add_nc_i32_e64 v12, s44, v18", "v_add_nc_i32 v12, s44, v18"),
        ("v_sub_nc_u32_e64 v12, abs(v18), v18", "v_sub_nc_u32_e64 v12, v18, v18"),
        ("v_add_nc_u32_e64 v12, s44, -v18", "v_add_nc_u32_e64 v12, s44, v18"),
        ("v_subrev_nc_u32_dpp v12, v18, v18 row_shl:1 row_mask:0xf bank_mask:0xf",
         "v_subrev_nc_u32_e32 v12, v18, v18"),
        ("v_subrev_nc_u32_sdwa v12, v18, v18 src0_sel:DWORD src1_sel:DWORD",
         "v_subrev_nc_u32_e32 v12, v18, v18"),
    ):
        translate_int(uncompressed_export(["v_mov_b32_e32 v18, s45", good]))
        must_reject(translate_int, uncompressed_export(["v_mov_b32_e32 v18, s45", bad]))


def check_masked_controls():
    """Exec-masked regions (#338): one forward masked-if, merged with per-lane selects."""
    save = "s_and_saveexec_b64 s[60:61], vcc"
    restore = "s_or_b64 exec, exec, s[60:61]"
    region = masked_region()
    inside = region.index(save) + 1
    source = translate(region)
    assert "cmp" in source and " ? " in source
    assert "precise float alu" in source
    assert "cmp" in translate(masked_skip()) and "cmp" in translate(masked_complementary())
    # An aligned SGPR pair holding a tracked comparison is a valid mask source too.
    pair_source = translate(uncompressed_export([
        "v_cmp_lt_f32_e64 s[52:53], v12, s44",
        "s_and_saveexec_b64 s[60:61], s[52:53]", "v_mov_b32_e32 v13, s45", restore]))
    assert " ? " in pair_source
    # A register written twice inside still merges against its value at the save.
    double = translate(uncompressed_export([
        "v_cmp_lt_f32_e64 vcc_lo, v12, s44", save,
        "v_mov_b32_e32 v13, s45", "v_add_f32_e32 v13, v13, s46", restore]))
    assert "_v13 = (cmp" in double and "sample16.y" in double.split("_v13")[1][:80]
    # A bare skip line carries no offset; only pair input can place it.
    must_reject(translate, [line for _, line in masked_skip()])
    for extra in (
            "s_mov_b32 s62, 1", "s_waitcnt lgkmcnt(0)", "s_and_saveexec_b64 s[58:59], vcc",
            "s_buffer_load_dword s70, s[44:47], 0x10", "exp mrt0 v12, v13, v14, v15 done vm",
            "image_sample v[16:19], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "v_cmp_eq_f32_e64 vcc_lo, v12, v13", "v_interp_p1_f32_e32 v17, v0, attr0.x",
            "v_readfirstlane_b32 s62, v12", "v_readlane_b32 s62, v12, 0",
            "v_writelane_b32 v12, s45, 3", "v_permlane_b32 v13, v12, s60",
            "v_mov_b32_dpp v13, v12 dpp8:[1,2,3,0,5,6,7,4]",
            "s_and_b64 exec, exec, s[50:51]", "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13"):
        must_reject(translate, region[:inside] + [extra] + region[inside:])
    unclosed = [line for line in region if line != restore]
    for variant in (
            unclosed[:unclosed.index("exp mrt0 v12, v13, v14, v15 done vm")],
            [line if line != save else "s_and_saveexec_b64 s[58:59], vcc" for line in region],
            ["s_and_saveexec_b64 s[60:61], s[50:51]" if line == save else line for line in region],
            [line.replace("s[60:61]", "s[61:62]") if line == save else line for line in region],
            uncompressed_export(["v_cmp_lt_f32_e64 vcc_lo, v12, s44", save,
                                 "v_mov_b32_e32 v13, s45", "s_and_saveexec_b64 s[58:59], vcc",
                                 "v_mov_b32_e32 v14, s46", restore, restore]),
    ):
        must_reject(translate, variant)
    # A pending scalar load may not overlap the saved pair.
    expect_value_error(translate, uncompressed_export([
        "s_load_dwordx4 s[60:63], s[20:21], 0x40",
        "v_cmp_lt_f32_e64 vcc_lo, v12, s44", save, restore]),
        "Unsupported execution mask save")
    # A register defined only inside the region may not be read afterwards...
    undefined = uncompressed_export([
        "v_cmp_lt_f32_e64 vcc_lo, v12, s44", save, "v_mov_b32_e32 v30, 1.0", restore,
        "v_add_f32_e32 v12, v12, v30"])
    must_reject(translate, undefined)
    # ...but an unconditional write makes it defined on every path.
    translate(uncompressed_export([
        "v_cmp_lt_f32_e64 vcc_lo, v12, s44", save, "v_mov_b32_e32 v30, 1.0", restore,
        "v_mov_b32_e32 v30, s47", "v_add_f32_e32 v12, v12, v30"]))
    # An empty region lowers to the untouched program.
    empty = uncompressed_export(["v_cmp_lt_f32_e64 vcc_lo, v12, s44", save, restore])
    assert translate(empty)
    # SCC after a restore is unknown: a consumer works without the region but rejects after it.
    translate(uncompressed_export(["s_cmp_eq_u32 s44, s45", "s_cmov_b64 s[44:45], s[46:47]"]))
    expect_value_error(translate, uncompressed_export([
        "s_cmp_eq_u32 s44, s45", "v_cmp_lt_f32_e64 vcc_lo, v12, s44", save, restore,
        "s_cmov_b64 s[44:45], s[46:47]"]), "Conditional move requires a tracked SCC")
    # Skip-branch displacement variants, all on pair input.
    pairs = masked_skip()
    branch = next(i for i, (_, line) in enumerate(pairs) if line.startswith("s_cbranch_execz"))
    for imm, why in ((1, "target lands mid-region"), (3, "target is the export"), (40, "no boundary")):
        wrong = [(o, "s_cbranch_execz %d" % imm if j == branch else line)
                 for j, (o, line) in enumerate(pairs)]
        must_reject(translate, wrong)
    two = masked_region()
    pos = two.index(save) + 1
    two[pos:pos] = ["s_cbranch_execz 3", "s_cbranch_execz 2"]
    must_reject(translate, offset_pairs(two))
    # Multiple independently closed regions may follow each other with vector work between.
    translate(uncompressed_export([
        "v_cmp_lt_f32_e64 vcc_lo, v12, s44", save, "v_mov_b32_e32 v13, s45", restore,
        "v_add_f32_e32 v14, 1.0, v14",
        "v_cmp_ge_f32_e64 vcc_lo, v13, s45", save, "v_mov_b32_e32 v14, s46", restore]))
    # A conditional register left by an earlier region does not taint later region writes.
    translate(uncompressed_export([
        "v_cmp_lt_f32_e64 vcc_lo, v12, s44", save, "v_mov_b32_e32 v30, s45", restore,
        "v_cmp_lt_f32_e64 vcc_lo, v12, s44", save, "v_mov_b32_e32 v13, s45", restore]))
    # A register whose outer value is not an emitted name cannot be merged.
    must_reject(translate, uncompressed_export([
        "v_interp_p1_f32_e32 v10, v0, attr0.x",
        "v_cmp_lt_f32_e64 vcc_lo, v12, s44", save, "v_mov_b32_e32 v10, s45", restore]))
    must_reject(translate, uncompressed_export([
        "v_cmp_lt_f32_e64 vcc_lo, v12, s44", save, "v_mov_b32_e32 v0, s45", restore]))
    # The skip only points forward: a negative displacement onto an earlier restore rejects.
    backward = uncompressed_export([
        "v_cmp_lt_f32_e64 vcc_lo, v12, s44", save, "v_mov_b32_e32 v13, s45", restore,
        "v_cmp_lt_f32_e64 vcc_lo, v12, s44", save, "s_cbranch_execz 0", "v_mov_b32_e32 v14, s45", restore])
    pairs = offset_pairs(backward)
    cbr = next(i for i, (_, line) in enumerate(pairs) if line == "s_cbranch_execz 0")
    target = next(o for o, line in pairs if line == restore)
    imm = (target - pairs[cbr][0] - 4) // 4
    pairs[cbr] = (pairs[cbr][0], f"s_cbranch_execz {imm}")
    assert imm < 0
    must_reject(translate, pairs)
    # Nor may it land on a later restore of the same pair: the close must be the target.
    two_same = uncompressed_export([
        "v_cmp_lt_f32_e64 vcc_lo, v12, s44", save, "s_cbranch_execz 0",
        "v_mov_b32_e32 v13, s45", restore,
        "v_cmp_ge_f32_e64 vcc_lo, v12, s44", save, "v_mov_b32_e32 v14, s46", restore])
    pairs = offset_pairs(two_same)
    cbr = next(i for i, (_, line) in enumerate(pairs) if line == "s_cbranch_execz 0")
    last = next(o for o, line in reversed(pairs) if line == restore)
    pairs[cbr] = (pairs[cbr][0], f"s_cbranch_execz {(last - pairs[cbr][0] - 4) // 4}")
    must_reject(translate, pairs)
    # The skip must sit immediately after the save; a mid-region placement rejects even
    # when its displacement still names the matching restore.
    mid = uncompressed_export([
        "v_cmp_lt_f32_e64 vcc_lo, v12, s44", save, "v_mov_b32_e32 v13, s45",
        "s_cbranch_execz 1", "v_mov_b32_e32 v14, s46", restore])
    expect_value_error(translate, offset_pairs(mid), "Instruction cannot run in a masked region")
    # A second skip anywhere in the program rejects, even in a later region.
    two_skips = uncompressed_export([
        "v_cmp_lt_f32_e64 vcc_lo, v12, s44", save, "s_cbranch_execz 1",
        "v_mov_b32_e32 v13, s45", restore,
        "v_cmp_ge_f32_e64 vcc_lo, v12, s44", save, "s_cbranch_execz 1",
        "v_mov_b32_e32 v14, s46", restore])
    expect_value_error(translate, offset_pairs(two_skips), "Unsupported execution skip")
    # The displacement is a signed 16-bit immediate.
    pairs = masked_skip()
    pos = next(i for i, (_, line) in enumerate(pairs) if line.startswith("s_cbranch_execz"))
    pairs[pos] = (pairs[pos][0], "s_cbranch_execz 0x8000")
    expect_value_error(translate, pairs, "Execution skip displacement out of range")
    # A 64-bit save needs a 64-bit mask source: vcc or an aligned pair, nothing else.
    for dst_cmp, src in (("s62", "s62"), ("vcc_lo", "vcc_lo")):
        must_reject(translate, uncompressed_export([
            f"v_cmp_lt_f32_e64 {dst_cmp}, v12, s44",
            f"s_and_saveexec_b64 s[60:61], {src}", "v_mov_b32_e32 v13, s45", restore]))
    expect_value_error(translate, uncompressed_export([
        "v_cmp_lt_f32_e64 vcc_lo, v12, s44",
        "s_and_saveexec_b64 s[60:61], s[61:62]", "v_mov_b32_e32 v13, s45", restore]),
        "Invalid 64-bit scalar register pair")
    # A register still conditional from an earlier region stays conditional when a later
    # region overwrites it; no merge is emitted against the stale inner value.
    cond2 = uncompressed_export([
        "v_cmp_lt_f32_e64 vcc_lo, v12, s44", save, "v_mov_b32_e32 v30, s45", restore,
        "v_cmp_lt_f32_e64 vcc_lo, v12, s44", save, "v_mov_b32_e32 v30, s46", restore])
    source2 = translate(cond2)
    assert "_v30" not in source2
    expect_value_error(translate, uncompressed_export([
        "v_cmp_lt_f32_e64 vcc_lo, v12, s44", save, "v_mov_b32_e32 v30, s45", restore,
        "v_cmp_lt_f32_e64 vcc_lo, v12, s44", save, "v_mov_b32_e32 v30, s46", restore,
        "v_add_f32_e32 v12, v12, v30"]), "Read of a register defined only on the masked path")
    # A conditional register may not read-modify-write itself either.
    expect_value_error(translate, uncompressed_export([
        "v_cmp_lt_f32_e64 vcc_lo, v12, s44", save, "v_mov_b32_e32 v30, s45", restore,
        "v_add_f32_e32 v30, v30, 1.0"]), "Read of a register defined only on the masked path")
    # All positions in a branch-bearing program must come from one ordered stream.
    positioned = masked_skip()
    branch = next(i for i, (_, line) in enumerate(positioned) if line.startswith("s_cbranch_execz"))
    duplicate = positioned[:]
    duplicate[branch + 1] = (duplicate[branch][0], duplicate[branch + 1][1])
    unaligned = positioned[:]
    unaligned[branch + 1] = (unaligned[branch + 1][0] + 1, unaligned[branch + 1][1])
    for invalid in (duplicate, unaligned, positioned[:branch] + [positioned[branch][1]] + positioned[branch + 1:]):
        must_reject(translate, invalid)
    assert translate(iter(positioned)) == translate(positioned)
    # Pair elements must be (int offset, str text); anything else is a clean rejection.
    for bad in ([(0, 42)], [("x", "s_nop 0")], [()], [[0, "s_nop 0"]], [42]):
        must_reject(translate, bad)


def sopk_wait_program():
    """The sampled program with every wait spelled as a gfx10 SOPK wait, plus a store-counter wait."""
    spelled = {"s_waitcnt lgkmcnt(0)": "s_waitcnt_lgkmcnt null, 0x0", "s_waitcnt vmcnt(0)": "s_waitcnt_vmcnt null, 0x0"}
    program = [spelled.get(line, line) for line in assembly(False)]
    program.insert(program.index("s_waitcnt_vmcnt null, 0x0") + 1, "s_waitcnt_vscnt null, 0x0")
    return program


def check_sopk_waitcnt(program):
    """The gfx10 SOPK wait forms with null act like the s_waitcnt tokens; vscnt is a no-op."""
    def same(function, old, token_line, new, source):
        """The new spelling must give the same HLSL as the token form, or be rejected as it is."""
        token = [token_line if line == old else line for line in source]
        sopk = [new if line == old else line for line in source]
        try:
            expected = function(token)
        except ValueError:
            must_reject(function, sopk)
            return None
        assert function(sopk) == expected
        return expected

    for count in ("0", "1", "2", "63"):
        for spelling in (f"s_waitcnt_vmcnt null, 0x{int(count):x}", f"s_waitcnt_vmcnt null, {count}"):
            same(translate, "s_waitcnt vmcnt(0)", f"s_waitcnt vmcnt({count})", spelling, program)
        for spelling in (f"s_waitcnt_lgkmcnt null, 0x{int(count):x}", f"s_waitcnt_lgkmcnt null, {count}"):
            same(translate, "s_waitcnt lgkmcnt(0)", f"s_waitcnt lgkmcnt({count})", spelling, program)
    # Full drain gives the baseline output; a partial count leaves the sample or the descriptor load
    # outstanding, so the single-sample program refuses it (the dependent use is not covered).
    base = translate(program)
    assert same(translate, "s_waitcnt vmcnt(0)", "s_waitcnt vmcnt(0)", "s_waitcnt_vmcnt null, 0x0", program) == base
    assert same(translate, "s_waitcnt lgkmcnt(0)", "s_waitcnt lgkmcnt(0)", "s_waitcnt_lgkmcnt null, 0x0", program) == base
    must_reject(translate, [("s_waitcnt_vmcnt null, 0x1" if line == "s_waitcnt vmcnt(0)" else line) for line in program])
    must_reject(translate, [("s_waitcnt_lgkmcnt null, 0x1" if line == "s_waitcnt lgkmcnt(0)" else line) for line in program])
    # Partial counts with two outstanding samples behave like vmcnt(1).
    partial = dual_program("partial")
    wait = partial.index("s_waitcnt vmcnt(1)")
    legal = same(translate_dual, "s_waitcnt vmcnt(1)", "s_waitcnt vmcnt(1)", "s_waitcnt_vmcnt null, 0x1", partial)
    assert legal is not None and legal == translate_dual(partial)
    for early_use in ("v_mov_b32_e32 v24, v20", "v_mov_b32_e32 v20, v12", "v_mov_b32_e32 v20, 0"):
        changed = partial.copy()
        changed[wait] = "s_waitcnt_vmcnt null, 0x1"
        changed.insert(wait + 1, early_use)
        must_reject(translate_dual, changed)
    for count in ("0x2", "0x3f"):
        changed = partial.copy()
        changed[wait] = f"s_waitcnt_vmcnt null, {count}"
        must_reject(translate_dual, changed)
    # A later, larger wait after draining does not invalidate completed samples.
    end_wait = partial.index("s_waitcnt vmcnt(0)")
    translate_dual(partial[:end_wait + 1] + ["s_waitcnt_vmcnt null, 0x3f"] + partial[end_wait + 1:])
    # vscnt: nothing is outstanding on the store counter, so it is accepted anywhere before the export
    # and changes nothing.
    export = next(i for i, line in enumerate(program) if line.startswith("exp "))
    # HLSL temporaries are named after the index of the instruction defining them, so an insertion
    # shifts later names; renumber by first appearance before comparing. After the last definition
    # the output must be byte-identical.
    def renumbered(source):
        names = {}
        return re.sub(r"\b(sample|alu)\d+\b", lambda m: names.setdefault(m[0], f"{m[1]}#{len(names)}"), source)

    for form in ("s_waitcnt_vscnt null, 0x0", "s_waitcnt_vscnt null, 0", "s_waitcnt_vscnt null, 0x3f"):
        for at in range(export + 1):
            assert renumbered(translate(program[:at] + [form] + program[at:])) == renumbered(base), (form, at)
        assert translate(program[:export] + [form] + program[export:]) == base, form
    must_reject(translate, program + ["s_waitcnt_vscnt null, 0x0"])
    assert translate(sopk_wait_program()) == base
    # Refusals, with exact messages.
    for counter, label in (("vmcnt", "VM"), ("lgkmcnt", "LGKM"), ("vscnt", "VS")):
        for operand in ("s5", "s0", "vcc_lo", "m0", "0"):
            expect_value_error(translate, [f"s_waitcnt_{counter} {operand}, 0x1"] + program,
                               f"Unsupported s_waitcnt_{counter} SGPR operand (only null is modelled): "
                               f"s_waitcnt_{counter} {operand}, 0x1")
        for count in ("0x40", "64", "0xffff"):
            expect_value_error(translate, [f"s_waitcnt_{counter} null, {count}"] + program,
                               f"{label} wait count exceeds the gfx10 six-bit field")
        for bad in (f"s_waitcnt_{counter}", f"s_waitcnt_{counter} null", f"s_waitcnt_{counter} null,",
                    f"s_waitcnt_{counter} 0x1", f"s_waitcnt_{counter} null, null", f"s_waitcnt_{counter} null, -1",
                    f"s_waitcnt_{counter} null, 05", f"s_waitcnt_{counter} null, 0x1, 0x2"):
            expect_value_error(translate, [bad] + program, f"Malformed s_waitcnt_{counter} operands: {bad}")
    # The neighbouring SOPK wait spelling stays unsupported.
    must_reject(translate, ["s_waitcnt_depctr 0x0"] + program)


def s_version_program():
    """The cmpk/cselect program (SCC, a pending sample and descriptor loads in flight) with s_version markers,
    one of them after the color export."""
    program = sopk_cmpk_cmovk()
    program.insert(program.index("s_cmpk_eq_i32 s47, 0xffff") + 1, "s_version 0x1004")
    program.insert(program.index("s_cselect_b32 s63, 1, 0"), "s_version 0")
    program.insert(program.index("image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D") + 1,
                   "s_version 0xffff")
    program.insert(len(program) - 1, "s_version 4100")
    return program


def check_s_version():
    """SOPK s_version K is a marker with no destination that leaves SCC, the pending sample and the loads alone."""
    plain = sopk_cmpk_cmovk()
    base = translate_int(plain)
    export = next(i for i, line in enumerate(plain) if line.startswith("exp "))

    # HLSL temporaries are named after the defining instruction index, so an insertion shifts later names.
    def renumbered(source):
        names = {}
        return re.sub(r"\b(sample|alu|salu|scc)(\d+)\b", lambda m: names.setdefault(m[0], f"{m[1]}#{len(names)}"), source)

    assert re.search(r"\b(?:sample|alu|salu|scc)\d+\b", base), "no renumberable names, the control would be vacuous"
    for form in ("s_version 0x1004", "s_version 0", "s_version 0x0", "s_version 0xffff", "s_version 65535",
                 "s_version 4100", "s_version 0x00ff"):
        # Between every pair of instructions up to the export, including between s_cmpk and s_cselect and
        # between the image_sample and its wait: the HLSL is the one of the program without the marker.
        for at in range(export + 1):
            assert renumbered(translate_int(plain[:at] + [form] + plain[at:])) == renumbered(base), (form, at)
        # After the color export it is accepted exactly like s_nop, and the output is byte-identical.
        assert translate_int(plain[:export + 1] + [form] + plain[export + 1:]) == base, form
        assert translate_int(plain[:export + 1] + ["s_nop 0"] + plain[export + 1:]) == base
    assert translate_int(plain[:export + 1] + ["s_version 0x1004", "s_nop 0"] + plain[export + 1:]) == base
    # The marker must not repair state: dropping the wait before the sample is used is still refused.
    wait = plain.index("s_waitcnt vmcnt(0)")
    must_reject(translate_int, plain[:wait] + ["s_version 0"] + plain[wait + 1:])
    # Another instruction after the export is still refused, with or without a marker before it.
    must_reject(translate_int, plain[:export + 1] + ["s_version 0", "s_mov_b32 s50, 0"] + plain[export + 1:])
    # The marker also works for the compressed-export and float programs.
    sampled = assembly(False)
    assert renumbered(translate(sampled[:-3] + ["s_version 0x1004"] + sampled[-3:])) == renumbered(translate(sampled))
    # Refusals, with exact messages.
    for bad in ("s_version 0x10000", "s_version 65536", "s_version 0xfffff", "s_version 4294967295",
                "s_version -1", "s_version -0x1", "s_version s0", "s_version s5", "s_version null", "s_version v0",
                "s_version vcc_lo", "s_version 0, 0", "s_version 0 0", "s_version 0x1004, 0x1", "s_version",
                "s_version ", "s_version 0x", "s_version 1.0", "s_version 007", "s_version abc", "s_version 0X1"):
        for source in (plain[:export] + [bad], plain[:export + 1] + [bad] + plain[export + 1:]):
            expect_value_error(translate_int, source, "Invalid s_version operand")
    assert renumbered(translate_int(s_version_program())) == renumbered(base)


def check_vopc_e32_destination():
    """VOPC e32 writes VCC only: LLVM 15 rejects an SGPR destination on the explicit spelling."""
    message = "VOPC e32 destination must be vcc"
    f32_all = [f"v_cmp_{op}_f32" for op in PREDICATES]
    families = [f"v_cmp_{op}_{kind}" for kind in ("f16", "i32", "u32", "i16", "u16")
                for op in ("lt", "eq")] + ["v_cmp_class_f32", "v_cmp_class_f16"] + f32_all
    for name in families:
        for dst in ("s52", "s[52:53]"):
            expect_value_error(translate, with_constants(["v_mov_b32_e32 v18, 64", f"{name}_e32 {dst}, s44, v18"]), message)
            # The suffix-less spelling is what LLVM turns into e64; both lower identically.
            plain = translate(with_constants(["v_mov_b32_e32 v18, 64", f"{name} {dst}, s44, v18"]))
            assert plain == translate(with_constants(["v_mov_b32_e32 v18, 64", f"{name}_e64 {dst}, s44, v18"])), (name, dst)
        # vcc and vcc_lo stay valid on e32 (wave32 and wave64 spellings).
        for dst in ("vcc_lo", "vcc"):
            translate(with_constants(["v_mov_b32_e32 v18, 64", f"{name}_e32 {dst}, s44, v18"]))


# Plain non-fused binary16 VOP2 arithmetic (issue #839): v_add_f16, v_sub_f16, v_subrev_f16 and v_mul_f16, e32 and e64.
# Both sources must be statically proven finite, so every operand is baked into the program as an inline constant, a
# 16-bit literal, a tracked SGPR or a VGPR copied from a literal; a result that would overflow is refused. The lowering
# is only valid with the entry denormal state declared (lower(half_denorm="preserve")).
F16_ARITH_OPS = ("add", "sub", "subrev", "mul")
# 40 pairs (a, b) of finite binary16 encodings. No lane of any of the four ops overflows (asserted below).
F16_ARITH_PAIRS = (
    (0x3c00, 0x1000), (0x6800, 0x3c00), (0xbc00, 0x9000),              # halfway, down to even
    (0x3c01, 0x1000), (0x6801, 0x3c00), (0xbc01, 0x9000),              # halfway, up to even
    (0x6800, 0x3c01), (0x6801, 0x3bff),                                # just above and below a tie
    (0x7bff, 0x0001), (0xfbff, 0x8001),                                # sticky bit beyond binary32
    (0x0001, 0x3800), (0x8001, 0x3800), (0x0003, 0x3800), (0x0005, 0x3800), (0x0006, 0x3400), (0x0001, 0x3400),
    (0x0401, 0x0400), (0x8400, 0x0401), (0x0400, 0x3800),              # normal to subnormal
    (0x03ff, 0x0001), (0x0001, 0x0001), (0x8001, 0x8001), (0x03ff, 0x03fe), (0x03ff, 0x4000),
    (0x0001, 0x3c00), (0x0001, 0xbc00),                                # subnormal inputs and the carry into 0x0400
    (0x0000, 0x0000), (0x8000, 0x8000), (0x0000, 0x8000), (0x8000, 0x0000),
    (0x3c00, 0x3c00), (0xbc00, 0x3c00), (0x3c00, 0x8000), (0xbc00, 0x8000), (0x8001, 0x0001),   # signed zero
    (0x3c01, 0x3c01), (0x3e00, 0x3e00), (0x7bff, 0x3800),             # plain
    (0x8003, 0x0003), (0x3555, 0xb555),
)
# Junk for bits 31:16 of every source word, so that a source read from the high half changes lanes.
F16_ARITH_JUNK = (0x3555, 0xd1a7, 0x4248, 0xa5c3)
# Pairs read through the four accepted readers: v_cvt_f32_f16 e32 and e64, its SDWA WORD_0 form and v_mov_b32_sdwa
# WORD_0. Reader r of program j reads the result of op (r + j) % 4.
F16_ARITH_READER_PAIRS = ((0x0003, 0x3800), (0x8001, 0x8001), (0x8000, 0x8000), (0x6801, 0x3c00),
                          (0x03ff, 0x0001), (0xbc00, 0x3c00))
# Chains: v24 = add(a, b), v25 = mul(v24, c), v26 = sub(v25, v24), v27 = subrev(v24, v26), each a result of an
# accepted op read as the source of the next one.
F16_ARITH_CHAINS = ((0x3c01, 0x1000, 0x0401), (0x8001, 0x8001, 0x4000))
F16_ARITH_MUTATIONS = ("truncate", "ties_away", "flush_out", "flush_in", "neg_subnormal_in", "zero_sign_out",
                       "src_high", "subrev_as_sub", "sub_as_add", "mul_as_add", "zero_sum_sign")


def f16_arith_reference(op, word_a, word_b, *, mutation=None, double=False):
    """Binary16 result bits of one op on the low halves of two source words, from the semantics only.

    Exact integer arithmetic (value = integer * 2^-24 per operand, a product in units of 2^-48) and one round to
    nearest even on the binary16 grid, with the IEEE rule for the sign of an exact zero. No pixel_aot import and no
    struct 'e'. Returns None for a NaN or infinity operand and for an overflow, which no fixture uses.
    mutation models a deliberate deviation of the draw's mutated shaders; a mutation (or double=True) first rounds the
    exact value to binary32 as the lowering does.
    """
    op = {"subrev_as_sub": {"subrev": "sub"}, "sub_as_add": {"sub": "add"},
          "mul_as_add": {"mul": "add"}}.get(mutation, {}).get(op, op)

    def operand(word, high):
        half = (word >> 16 if high else word) & 0xffff
        sign, exponent, mantissa = half >> 15, (half >> 10) & 31, half & 1023
        if exponent == 31:
            return None
        if exponent == 0:
            return (0 if mutation == "neg_subnormal_in" else sign), (0 if mutation == "flush_in" else mantissa)
        return sign, (mantissa | 1024) << (exponent - 1)

    first, second = operand(word_a, mutation == "src_high"), operand(word_b, False)
    if first is None or second is None:
        return None
    (sign_a, mag_a), (sign_b, mag_b) = first, second
    x, y = (-mag_a if sign_a else mag_a), (-mag_b if sign_b else mag_b)
    if op == "mul":
        exact, zero_negative = x * y, sign_a != sign_b
    else:
        exact, zero_negative = {"add": ((x + y) << 24, bool(sign_a and sign_b)),
                                "sub": ((x - y) << 24, bool(sign_a and not sign_b)),
                                "subrev": ((y - x) << 24, bool(sign_b and not sign_a))}[op]
    magnitude = abs(exact)
    if (double or mutation is not None) and magnitude.bit_length() > 24:
        drop = magnitude.bit_length() - 24
        kept, rest = divmod(magnitude, 1 << drop)
        if rest > 1 << (drop - 1) or (rest == 1 << (drop - 1) and kept & 1):
            kept += 1
        magnitude = kept << drop
    sign = 0x8000 if exact < 0 or (exact == 0 and zero_negative) else 0
    if mutation == "zero_sign_out" and magnitude < 1 << 23:  # below 2^-25
        return 0
    if mutation == "flush_out" and magnitude < 1 << 34:  # below 2^-14
        return sign
    if magnitude == 0:
        bits = 0
    else:
        exponent = magnitude.bit_length() - 1 - 48
        quantum = max(exponent, -14) - 10
        kept, rest = divmod(magnitude, 1 << (quantum + 48))
        half = 1 << (quantum + 47)
        if mutation == "ties_away":
            kept += rest >= half
        elif mutation != "truncate" and (rest > half or (rest == half and kept & 1)):
            kept += 1
        exponent = max(exponent, -14)
        if kept >= 2048:
            kept, exponent = kept >> 1, exponent + 1
        if kept < 1024:
            bits = kept
        elif exponent + 15 >= 31:
            return None
        else:
            bits = ((exponent + 15) << 10) | (kept - 1024)
    result = sign | bits
    if mutation == "zero_sum_sign" and op != "mul" and result == 0x8000:
        return 0
    return result


def f16_widen_word(bits):
    """The binary32 encoding of a finite binary16 encoding (v_cvt_f32_f16), exact."""
    sign, exponent, mantissa = bits >> 15, (bits >> 10) & 31, bits & 1023
    magnitude = (Fraction(mantissa, 1 << 24) if exponent == 0
                 else Fraction(mantissa | 1024) * Fraction(2) ** (exponent - 25))
    return struct.unpack("<I", struct.pack("<f", math.copysign(float(magnitude), -1.0 if sign else 1.0)))[0]


def f16_arith_pair_words(pair):
    """The 32-bit source words of F16_ARITH_PAIRS[pair]: the binary16 value in bits 15:0, junk in bits 31:16."""
    half_a, half_b = F16_ARITH_PAIRS[pair]
    return ((F16_ARITH_JUNK[pair % 4] << 16) | half_a, (F16_ARITH_JUNK[(pair + 1) % 4] << 16) | half_b)


def f16_arith_token(half):
    """The src0 token of an e32 literal form, spelled as LLVM 15 prints it, or None where LLVM folds it to an integer."""
    inverse = {value: token for token, value in pixel_aot.INLINE_F16.items()}
    if half in inverse:
        return inverse[half]
    if half == 0:
        return "0"
    return None if 1 <= half <= 0x40 else f"{half:#x}"


def f16_arith_program(kind, index):
    """One draw program. Kinds: main (two pairs, four ops each, packed two results per word), reader (one pair read
    through the four readers) and chain (four chained results widened with v_cvt_f32_f16)."""
    pack = "v_mov_b32_sdwa v{dst}, v{src} dst_sel:{sel} dst_unused:{unused} src0_sel:WORD_0"
    body = []
    if kind == "main":
        for slot in (0, 1):
            pair = 2 * index + slot
            word_a, word_b = f16_arith_pair_words(pair)
            reg_a, reg_b, sgpr = 18 + 2 * slot, 19 + 2 * slot, 60 + slot
            body += [f"v_mov_b32_e32 v{reg_a}, {word_a:#x}", f"v_mov_b32_e32 v{reg_b}, {word_b:#x}",
                     f"s_mov_b32 s{sgpr}, {word_a:#x}"]
            literal = f16_arith_token(word_a & 0xffff)
            for k, op in enumerate(F16_ARITH_OPS):
                dst, form = 24 + 4 * slot + k, (pair + k) % 4
                if form == 3 and literal is None:
                    form = 0
                body.append([f"v_{op}_f16_e32 v{dst}, v{reg_a}, v{reg_b}", f"v_{op}_f16_e64 v{dst}, v{reg_a}, v{reg_b}",
                             f"v_{op}_f16_e64 v{dst}, s{sgpr}, v{reg_b}",
                             f"v_{op}_f16_e32 v{dst}, {literal}, v{reg_b}"][form])
        for word in range(4):
            body.append(pack.format(dst=12 + word, src=24 + 2 * word, sel="WORD_0", unused="UNUSED_PAD"))
            body.append(pack.format(dst=12 + word, src=25 + 2 * word, sel="WORD_1", unused="UNUSED_PRESERVE"))
    elif kind == "reader":
        word_a, word_b = f16_arith_pair_words(F16_ARITH_PAIRS.index(F16_ARITH_READER_PAIRS[index]))
        body += [f"v_mov_b32_e32 v18, {word_a:#x}", f"v_mov_b32_e32 v19, {word_b:#x}",
                 "v_add_f16_e32 v24, v18, v19", "v_sub_f16_e64 v25, v18, v19",
                 "v_subrev_f16_e32 v26, v18, v19", "v_mul_f16_e64 v27, v18, v19"]
        readers = ("v_cvt_f32_f16_e32 v12, v{}", "v_cvt_f32_f16_e64 v13, v{}",
                   "v_cvt_f32_f16_sdwa v14, v{} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0",
                   "v_mov_b32_sdwa v15, v{} dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0")
        body += [line.format(24 + (r + index) % 4) for r, line in enumerate(readers)]
    else:
        word_a, word_b, word_c = ((F16_ARITH_JUNK[k] << 16) | half for k, half in enumerate(F16_ARITH_CHAINS[index]))
        body += [f"v_mov_b32_e32 v18, {word_a:#x}", f"v_mov_b32_e32 v19, {word_b:#x}", f"v_mov_b32_e32 v20, {word_c:#x}",
                 "v_add_f16_e32 v24, v18, v19", "v_mul_f16_e32 v25, v24, v20",
                 "v_sub_f16_e32 v26, v25, v24", "v_subrev_f16_e32 v27, v24, v26"]
        body += [f"v_cvt_f32_f16_e32 v{12 + k}, v{24 + k}" for k in range(4)]
    return uncompressed_export(body)


def f16_arith_source_halves(kind, index):
    """The binary16 source encodings of one draw program."""
    if kind == "main":
        return [half for pair in F16_ARITH_PAIRS[2 * index:2 * index + 2] for half in pair]
    return list(F16_ARITH_READER_PAIRS[index] if kind == "reader" else F16_ARITH_CHAINS[index])


def f16_arith_expected(kind, index, mutation=None):
    """The four exported words of f16_arith_program(kind, index); an entry is None where the model gives NaN, infinity
    or overflow (a mutated model only)."""
    def result(op, word_a, word_b):
        return f16_arith_reference(op, word_a, word_b, mutation=mutation)

    def widened(bits):
        return None if bits is None else f16_widen_word(bits)

    if kind == "main":
        results = []
        for slot in (0, 1):
            word_a, word_b = f16_arith_pair_words(2 * index + slot)
            results += [result(op, word_a, word_b) for op in F16_ARITH_OPS]
        return tuple(None if results[2 * w] is None or results[2 * w + 1] is None
                     else results[2 * w] | results[2 * w + 1] << 16 for w in range(4))
    if kind == "reader":
        word_a, word_b = f16_arith_pair_words(F16_ARITH_PAIRS.index(F16_ARITH_READER_PAIRS[index]))
        results = [result(op, word_a, word_b) for op in F16_ARITH_OPS]
        chosen = [results[(r + index) % 4] for r in range(4)]
        return (*(widened(bits) for bits in chosen[:3]), chosen[3])
    word_a, word_b, word_c = ((F16_ARITH_JUNK[k] << 16) | half for k, half in enumerate(F16_ARITH_CHAINS[index]))
    chain = [result("add", word_a, word_b)]
    chain.append(None if chain[0] is None else result("mul", chain[0], word_c))
    chain.append(None if chain[1] is None else result("sub", chain[1], chain[0]))
    chain.append(None if chain[2] is None else result("subrev", chain[0], chain[2]))
    return tuple(widened(bits) for bits in chain)


F16_ARITH_PROGRAMS = (*(("main", i) for i in range(len(F16_ARITH_PAIRS) // 2)),
                      *(("reader", i) for i in range(len(F16_ARITH_READER_PAIRS))),
                      *(("chain", i) for i in range(len(F16_ARITH_CHAINS))))


def translate_f16(instructions):
    """translate_int() for a program that declares the binary16 denormal state at entry as preserved."""
    return lower(instructions, 20, 24, [1, 2, 3, 4], constant_descriptor=64, integer_export=True,
                 half_denorm=pixel_aot.HALF_DENORM_PRESERVE)


# The 11 deviations of the draw. Each text edit applies to the lowered source of every program and asserts that it
# changed it; the three op-level ones rewrite the program instead.
F16_ARITH_TEXT_EDITS = {
    "truncate": (("return sign | ((full + (1u << (shift - 1u)) - 1u + ((full >> shift) & 1u)) >> shift);",
                  "return sign | (full >> shift);"),
                 ("uint rounded = (full + 0xfffu + ((full >> 13) & 1u)) >> 13;", "uint rounded = full >> 13;")),
    "ties_away": (("(full + (1u << (shift - 1u)) - 1u + ((full >> shift) & 1u)) >> shift",
                   "(full + (1u << (shift - 1u))) >> shift"),
                  ("(full + 0xfffu + ((full >> 13) & 1u)) >> 13", "(full + 0x1000u) >> 13")),
    "flush_out": (("return sign | ((full + (1u << (shift - 1u)) - 1u + ((full >> shift) & 1u)) >> shift);",
                   "return sign;"),),
    "flush_in": (("precise float magnitude = float(mantissa) * asfloat(0x33800000u);",
                  "precise float magnitude = 0.0f;"),),
    "neg_subnormal_in": (("return sign != 0u ? -magnitude : magnitude;", "return magnitude;"),),
    "zero_sign_out": (("    if (exponent < 102u) {\n        return sign;\n", "    if (exponent < 102u) {\n        return 0u;\n"),),
}
F16_ARITH_OP_MUTATIONS = {"subrev_as_sub": ("v_subrev_f16", "v_sub_f16"), "sub_as_add": ("v_sub_f16", "v_add_f16"),
                          "mul_as_add": ("v_mul_f16", "v_add_f16")}


def f16_arith_mutated(mutation, program):
    """The lowered source of program (instruction lines) with one deviation applied."""
    if mutation in F16_ARITH_OP_MUTATIONS:
        old, new = F16_ARITH_OP_MUTATIONS[mutation]
        lines = [line.replace(old, new) for line in program]
        assert lines != program, mutation
        return translate_f16(lines)
    text = translate_f16(program)
    if mutation == "src_high":
        mutated = re.sub(r"uint (hf\d+_0) = \((.+)\) & 0xffffu;", r"uint \1 = ((\2) >> 16u) & 0xffffu;", text)
        assert mutated != text
        return mutated
    if mutation == "zero_sum_sign":
        mutated = re.sub(r"(precise float hs\d+ = )(widen16\(hf\d+_\d\) [+-] widen16\(hf\d+_\d\));", r"\1(\2) + 0.0f;", text)
        assert mutated != text
        return mutated
    for old, new in F16_ARITH_TEXT_EDITS[mutation]:
        assert text.count(old) == 1, (mutation, old)
        text = text.replace(old, new)
    return text


def f16_arith_detecting_program(mutation):
    """The first program whose expected words the modelled mutation changes."""
    for kind, index in F16_ARITH_PROGRAMS:
        if f16_arith_expected(kind, index, mutation) != f16_arith_expected(kind, index):
            return kind, index
    raise AssertionError(f"No program separates {mutation}")


def check_f16_arith_reference():
    """The Python side of the draw: the reference against pinned rows, double rounding and the translator's own copy."""
    assert F16_DENORM_MODE == pixel_aot.HALF_DENORM_PRESERVE
    assert len(F16_ARITH_PAIRS) == 40 and len(set(F16_ARITH_PAIRS)) == 40
    # Pinned rows, from the IEEE 754 round-to-nearest-even rules and worked by hand: (op, a, b, result).
    for op, half_a, half_b, expected in (
            ("add", 0x3c00, 0x1000, 0x3c00), ("add", 0x3c01, 0x1000, 0x3c02),                 # tie to even, down and up
            ("add", 0x6800, 0x3c01, 0x6801), ("add", 0x7bff, 0x0001, 0x7bff),
            ("add", 0x03ff, 0x0001, 0x0400), ("add", 0x8001, 0x8001, 0x8002),                 # carry into the normals
            ("add", 0x8000, 0x8000, 0x8000), ("add", 0x0000, 0x8000, 0x0000), ("sub", 0x3c00, 0x3c00, 0x0000),
            ("sub", 0x0401, 0x0400, 0x0001), ("subrev", 0x3c00, 0x4000, 0x3c00), ("subrev", 0x0001, 0x0003, 0x0002),
            ("sub", 0x8000, 0x0000, 0x8000), ("subrev", 0x0000, 0x8000, 0x8000),
            ("mul", 0x0001, 0x3800, 0x0000), ("mul", 0x8001, 0x3800, 0x8000), ("mul", 0x0003, 0x3800, 0x0002),
            ("mul", 0x0005, 0x3800, 0x0002), ("mul", 0x0001, 0xbc00, 0x8001), ("mul", 0x3c01, 0x3c01, 0x3c02),
            ("mul", 0x03ff, 0x4000, 0x07fe), ("mul", 0x8001, 0x0001, 0x8000), ("mul", 0xbc00, 0x8000, 0x0000)):
        assert f16_arith_reference(op, half_a, half_b) == expected, (op, hex(half_a), hex(half_b))
    # Overflow is refused, not computed: 65504 + 16 is the halfway case that rounds to infinity.
    assert f16_arith_reference("add", 0x7bff, 0x4c00) is None and f16_arith_reference("add", 0x7bff, 0x4bff) == 0x7bff
    assert f16_arith_reference("mul", 0x7bff, 0x4000) is None and f16_arith_reference("add", 0x7c00, 0x3c00) is None
    for pair, (half_a, half_b) in enumerate(F16_ARITH_PAIRS):
        words = f16_arith_pair_words(pair)
        for op in F16_ARITH_OPS:
            result = f16_arith_reference(op, *words)
            assert result is not None and result & 0x7c00 != 0x7c00, (op, hex(half_a), hex(half_b))
            assert result == pixel_aot.f16_vop2_result(op, half_a, half_b), (op, hex(half_a), hex(half_b))
            assert result == f16_arith_reference(op, *words, double=True), (op, hex(half_a), hex(half_b))
    # One binary32 operation followed by one binary16 rounding equals the single rounding: the claim of the lowering.
    # Every 97th finite value (plus the edges of each binade) against 48 boundary values, all four ops.
    finite = [h for h in range(0x10000) if (h >> 10) & 31 != 31]
    edges = [e << 10 | m for e in range(31) for m in (0, 1, 0x3ff)]
    boundary = sorted({h | s for h in (0x0000, 0x0001, 0x0002, 0x0003, 0x03ff, 0x0400, 0x0401, 0x0800, 0x1000, 0x3bff,
                                       0x3c00, 0x3c01, 0x4000, 0x6800, 0x7bff, 0x77ff, 0x0200, 0x0100, 0x2400, 0x5800,
                                       0x1400, 0x0555, 0x0001 << 3, 0x01ff) for s in (0, 0x8000)})
    for half_a in sorted({*finite[::97], *edges, *(e | 0x8000 for e in edges)}):
        if (half_a >> 10) & 31 == 31:
            continue
        for half_b in boundary:
            for op in F16_ARITH_OPS:
                single = f16_arith_reference(op, half_a, half_b)
                assert single == f16_arith_reference(op, half_a, half_b, double=True), (op, hex(half_a), hex(half_b))
                assert single == pixel_aot.f16_vop2_result(op, half_a, half_b), (op, hex(half_a), hex(half_b))
    assert [f16_widen_word(h) for h in (0x3c00, 0x0001, 0x8000, 0xbc00, 0x7bff)] == [
        0x3f800000, 0x33800000, 0x80000000, 0xbf800000, 0x477fe000]


def check_f16_arith_mutation_coverage():
    """Every deviation changes at least one expected word of some program, so the draw can detect it, and the
    programs expect finite values only."""
    for kind, index in F16_ARITH_PROGRAMS:
        words = f16_arith_expected(kind, index)
        assert len(words) == 4 and all(word is not None for word in words), (kind, index)
    assert set(F16_ARITH_MUTATIONS) == {*F16_ARITH_TEXT_EDITS, *F16_ARITH_OP_MUTATIONS, "src_high", "zero_sum_sign"}
    for mutation in F16_ARITH_MUTATIONS:
        f16_arith_detecting_program(mutation)
    # Each deviation moves a specific class of row: the rows named here are the ones the model must separate.
    def differs(mutation, op, half_a, half_b):
        words = (0xa5c30000 | half_a, 0x3555_0000 | half_b)
        return f16_arith_reference(op, *words, mutation=mutation) != f16_arith_reference(op, *words)

    assert differs("truncate", "add", 0x3c01, 0x1000) and differs("ties_away", "add", 0x3c00, 0x1000)
    assert differs("flush_out", "sub", 0x0401, 0x0400) and differs("flush_in", "add", 0x0001, 0x0001)
    assert differs("neg_subnormal_in", "add", 0x8001, 0x8001) and differs("zero_sign_out", "add", 0x8000, 0x8000)
    assert differs("zero_sign_out", "mul", 0x8001, 0x0001) and differs("zero_sum_sign", "add", 0x8000, 0x8000)
    assert differs("subrev_as_sub", "subrev", 0x3c00, 0x4000) and differs("sub_as_add", "sub", 0x3c00, 0x3c00)
    assert differs("mul_as_add", "mul", 0x3c01, 0x3c01) and differs("src_high", "add", 0x3c00, 0x1000)
    # A mutation must not move rows it should leave alone (a negative control for the model itself).
    assert not differs("truncate", "add", 0x3c00, 0x3c00) and not differs("zero_sum_sign", "mul", 0x8000, 0x3c00)
    assert not differs("flush_in", "add", 0x3c00, 0x3c00) and not differs("ties_away", "add", 0x3c00, 0x3c00)


def check_f16_arith_controls():
    """Lowering, spelling, input proof, overflow, mode and reader refusals of the plain binary16 VOP2 forms."""
    check_f16_arith_reference()
    check_f16_arith_mutation_coverage()
    prefix = ["v_mov_b32_e32 v18, 0x35553c00", "v_mov_b32_e32 v19, 0xd1a74000", "s_mov_b32 s60, 0x3555c000"]

    def program(*lines):
        return uncompressed_export([*prefix, *lines])

    def refuse(line, message, translator=translate_f16):
        expect_value_error(translator, program(line), message)

    # Lowering and spelling: the bare, _e32 and _e64 spellings are one lowering (names carry the instruction index).
    for op in F16_ARITH_OPS:
        spelled = [translate_f16(program(f"v_{op}_f16{suffix} v24, v18, v19")) for suffix in ("", "_e32", "_e64")]
        assert spelled[0] == spelled[1] == spelled[2], op
        body = spelled[0].split("uint4 main", 1)[1]
        assert "widen16(" in body and "narrow16(" in body and "precise float hs" in body, op
        for forbidden in ("f32tof16", "f16tof32", "float16_t", "half ", "0xffff0000u"):
            assert forbidden not in body, (op, forbidden)
    body = translate_f16(program("v_sub_f16 v24, v18, v19")).split("uint4 main", 1)[1]
    assert re.search(r"hs\d+ = widen16\(hf\d+_0\) - widen16\(hf\d+_1\);", body), body
    body = translate_f16(program("v_subrev_f16 v24, v18, v19")).split("uint4 main", 1)[1]
    assert re.search(r"hs\d+ = widen16\(hf\d+_1\) - widen16\(hf\d+_0\);", body), body
    assert re.search(r"hs\d+ = widen16\(hf\d+_0\) \* widen16\(hf\d+_1\);",
                     translate_f16(program("v_mul_f16 v24, v18, v19"))), "mul"
    assert re.search(r"hs\d+ = widen16\(hf\d+_0\) \+ widen16\(hf\d+_1\);",
                     translate_f16(program("v_add_f16 v24, v18, v19"))), "add"
    # Operand kinds: VGPR, SGPR, every inline constant (the emitted word is the table value), 0, 16-bit literals.
    for token, bits in (*pixel_aot.INLINE_F16.items(), ("0", 0), ("0x3555", 0x3555), ("0x8000", 0x8000), ("0x7bff", 0x7bff)):
        text = translate_f16(program(f"v_add_f16_e64 v24, {token}, v19"))
        assert f"uint hf" in text and f"_0 = ({bits}u) & 0xffffu;" in text, token
        assert f"_1 = (asuint(alu" in text.split("uint4 main", 1)[1]
        assert f"_1 = ({bits}u) & 0xffffu;" in translate_f16(program(f"v_mul_f16_e64 v24, v18, {token}")), token
    assert f"_0 = ({0x3555c000}u) & 0xffffu;" in translate_f16(program("v_add_f16_e64 v24, s60, s60")), "SGPR word"
    translate_f16(program("v_add_f16_e64 v24, s60, v19"))
    translate_f16(program("v_add_f16_e64 v24, 0x3555, 0x3555"))
    # A VGPR copy of a proven constant keeps the proof; a result keeps it too and chains through the readers.
    translate_f16(program("v_mov_b32_e32 v20, v18", "v_sub_f16 v24, v20, v19"))
    translate_f16(program("s_mov_b32 s61, s60", "v_add_f16_e64 v24, s61, v19"))
    chained = translate_f16(program("v_add_f16 v24, v18, v19", "v_mul_f16 v25, v24, v24", "v_cvt_f32_f16 v12, v25"))
    assert "asuint(alu" in chained.split("uint4 main", 1)[1]
    # Entry profile: unknown or an unsupported declaration refuses every spelling; the other forms are unaffected.
    for op in F16_ARITH_OPS:
        for suffix in ("", "_e32", "_e64"):
            expect_value_error(lambda p: lower(p, 20, 24, [1, 2, 3, 4], constant_descriptor=64, integer_export=True),
                               program(f"v_{op}_f16{suffix} v24, v18, v19"), "binary16 denormal mode unknown at entry")
    for state in ("flush", "ftz", "", False):
        expect_value_error(lambda p: lower(p, 20, 24, [1, 2, 3, 4], constant_descriptor=64, integer_export=True,
                                           half_denorm=state), program("v_add_f16 v24, v18, v19"),
                           "Invalid binary16 denormal entry state")
    lower(program("v_add_nc_u16 v24, v18, v19"), 20, 24, [1, 2, 3, 4], constant_descriptor=64, integer_export=True)
    # No finite proof: a constant-buffer word, an uninitialized register, a sample, a copy without provenance.
    for line, token in (("v_add_f16_e64 v24, s44, v19", "s44"), ("v_add_f16_e64 v24, v18, s45", "s45"),
                        ("v_add_f16 v24, v30, v19", "v30"), ("v_add_f16 v24, v18, v30", "v30"),
                        ("v_add_f16_e64 v24, s70, v19", "s70"), ("v_add_f16 v24, v12, v19", "v12")):
        refuse(line, f"binary16 operand lacks a finite proof: {token}")
    expect_value_error(translate_f16, program("v_mov_b32_e32 v20, s44", "v_add_f16 v24, v20, v19"),
                       "binary16 operand lacks a finite proof: v20")
    expect_value_error(translate_f16, program("v_add_nc_u16 v25, v18, v19", "v_add_f16 v24, v25, v19"),
                       "binary16 operand lacks a finite proof: v25")
    # NaN and infinity are refused wherever the proof finds them, in either source and through every provenance.
    for token in ("0x7c00", "0xfc00", "0x7e00", "0x7c01", "0xffff"):
        refuse(f"v_add_f16_e64 v24, {token}, v19", f"binary16 operand is not finite: {token}")
        refuse(f"v_mul_f16_e64 v24, v18, {token}", f"binary16 operand is not finite: {token}")
    expect_value_error(translate_f16, program("v_mov_b32_e32 v20, 0x7e00", "v_sub_f16 v24, v20, v19"),
                       "binary16 operand is not finite: v20")
    expect_value_error(translate_f16, program("s_mov_b32 s61, 0xdead7c00", "v_add_f16_e64 v24, s61, v19"),
                       "binary16 operand is not finite: s61")
    expect_value_error(translate_f16, program("v_mov_b32_e32 v20, 0x7e00", "v_mov_b32_e32 v21, v20", "v_sub_f16 v24, v21, v19"),
                       "binary16 operand is not finite: v21")
    # Negative controls for the finite proof: the largest finite values, a subnormal literal and a high half that is
    # an infinity (only the low half is a source) are accepted.
    for token in ("0x7bff", "0xfbff", "0x0001", "0x8001", "0x0400", "-2.0", "-0.5", "0"):
        translate_f16(program(f"v_add_f16_e64 v24, {token}, v19"))
    translate_f16(program("v_mov_b32_e32 v20, 0x7c003c00", "v_add_f16 v24, v20, v19"))
    translate_f16(program("s_mov_b32 s61, 0x7e003c00", "v_add_f16_e64 v24, s61, v19"))
    # Finite operands whose result overflows are refused, with both signs and at the rounding boundary.
    for line, name in (("v_add_f16_e64 v24, 0x7bff, 0x4c00", "add"), ("v_add_f16_e64 v24, 0xfbff, 0xcc00", "add"),
                       ("v_sub_f16_e64 v24, 0x7bff, 0xcc00", "sub"), ("v_sub_f16_e64 v24, 0xfbff, 0x4c00", "sub"),
                       ("v_subrev_f16_e64 v24, 0xcc00, 0x7bff", "subrev"), ("v_subrev_f16_e64 v24, 0x4c00, 0xfbff", "subrev"),
                       ("v_mul_f16_e64 v24, 0x7bff, 0x4000", "mul"), ("v_mul_f16_e64 v24, 0xfbff, 0x4000", "mul"),
                       ("v_mul_f16_e64 v24, 0x7bff, 0xc000", "mul")):
        refuse(line, f"binary16 result overflows: v_{name}_f16")
    for line in ("v_add_f16_e64 v24, 0x7bff, 0x4bff", "v_add_f16_e64 v24, 0xfbff, 0xcbff", "v_sub_f16_e64 v24, 0x7bff, 0xcbff",
                 "v_subrev_f16_e64 v24, 0xcbff, 0x7bff", "v_mul_f16_e64 v24, 0x7bff, 0x3c00", "v_mul_f16_e64 v24, 0xfbff, 0x3c00"):
        translate_f16(program(line))
    expect_value_error(translate_f16, program("v_mov_b32_e32 v20, 0x7bff", "v_mov_b32_e32 v21, 0x3c00",
                                              "v_mul_f16 v25, v20, v21", "v_add_f16 v26, v25, v25"),
                       "binary16 result overflows: v_add_f16")
    # Operand grammar. A leading minus is a negative inline constant first and a refused modifier otherwise.
    for token in ("-v18", "|v18|", "abs(v18)", "neg(v18)", "-|v18|", "-s60", "-3.0"):
        refuse(f"v_add_f16_e64 v24, {token}, v19", f"Unsupported binary16 source modifier {token}")
    for token in ("1", "64", "-1", "-16", "5", "-0"):
        refuse(f"v_add_f16_e64 v24, {token}, v19", f"Unsupported binary16 integer inline constant {token}")
    for token in ("3.0", "0.25", "vcc_lo", "m0", "exec_lo", "null", "v[18:19]"):
        refuse(f"v_add_f16_e64 v24, {token}, v19", f"Unsupported binary16 operand {token}")
    refuse("v_add_f16_e64 v24, 0x10000, v19", "16-bit literal out of range: 0x10000")
    refuse("v_add_f16_e64 v256, 1.0, 1.0", "Register out of range: v256")
    refuse("v_add_f16_e64 v24, v256, 1.0", "Register out of range: v256")
    refuse("v_add_f16_e64 v24, s106, 1.0", "Register out of range: s106")
    for line in ("v_add_f16_e32 v24, v18, s60", "v_add_f16 v24, v18, s60", "v_mul_f16 v24, v18, 1.0", "v_sub_f16_e32 v24, v18, 0x3555"):
        refuse(line, "Binary16 e32 src1 must be a VGPR")
    # Forms the lowering does not accept fall through to the generic refusal, and so do the neighbouring binary16 ops.
    for line in ("v_add_f16_e64 v24, v18, v19 clamp", "v_mul_f16_e64 v24, v18, v19 mul:2", "v_mul_f16_e64 v24, v18, v19 div:2",
                 "v_add_f16_e64 v24, v18, v19 op_sel:[0,1]", "v_add_f16_dpp v24, v18, v19 quad_perm:[0,1,2,3]",
                 "v_add_f16_sdwa v24, v18, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0",
                 "v_add_f16 v24, v18, v19, v19", "v_add_f16 v24, v18", "v_min_f16_e64 v24, v18, v19",
                 "v_max_f16_e32 v24, v18, v19", "v_fmac_f16_e32 v24, v18, v19", "v_ldexp_f16_e32 v24, v18, v19"):
        refuse(line, "Unsupported instruction")

    # Scheduling around the load and the samples: a pending SGPR, an outstanding sample, a non-entry mask.
    def float_f16(instructions):
        return lower(instructions, 20, 24, [1, 2, 3, 4], constant_descriptor=64, half_denorm=pixel_aot.HALF_DENORM_PRESERVE)

    base = with_constants([])
    load = next(i for i, line in enumerate(base) if line.startswith("s_buffer_load_dwordx4"))
    expect_value_error(float_f16, base[:load + 1] + ["v_add_f16_e64 v24, s44, 1.0"] + base[load + 1:],
                       "Constant read before load completion")
    sample = next(i for i, line in enumerate(base) if line.startswith("image_sample"))
    expect_value_error(float_f16, base[:sample + 1] + ["v_add_f16_e64 v12, 1.0, 1.0"] + base[sample + 1:],
                       "Write overlaps an outstanding sample")
    expect_value_error(float_f16, base[:sample + 1] + ["v_add_f16_e64 v24, v12, 1.0"] + base[sample + 1:],
                       "Sample used before vmcnt wait")
    float_f16(base[:sample + 1] + ["v_add_f16_e64 v20, 1.0, 1.0"] + base[sample + 1:])
    masked = base.index("s_wqm_b64 exec, exec") + 1
    expect_value_error(float_f16, base[:masked] + ["v_add_f16_e64 v24, 1.0, 1.0"] + base[masked:],
                       "Invalid binary16 arithmetic operation")
    # Mode: every spelling is refused under round toward zero, a restored mode lowers like no write at all, and every
    # instruction that could change the denormal state stays refused before and after the op.
    for spelling in (f"v_{op}_f16{suffix}" for op in F16_ARITH_OPS for suffix in ("_e32", "_e64")):
        line = f"{spelling} v24, v18, v19"
        expect_value_error(translate_f16, uncompressed_export([HALF_RTZ, *prefix, line]),
                           f"under round toward zero: {spelling}")
        assert translate_f16(uncompressed_export([HALF_RTZ, HALF_RNE, *prefix, line])) == translate_f16(
            uncompressed_export(["s_nop 0", "s_nop 0", *prefix, line]))
    for write, message in (("s_denorm_mode 15", "Unsupported instruction"), ("s_round_mode 3", "Unsupported instruction"),
                           ("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 6, 2), 3", "Unsupported hardware register write"),
                           ("s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 4), 15", "Unsupported hardware register write"),
                           ("s_setreg_b32 hwreg(HW_REG_MODE, 6, 2), s48", "Unsupported instruction")):
        for lines in ([write, *prefix, "v_add_f16 v24, v18, v19"], [*prefix, "v_add_f16 v24, v18, v19", write]):
            expect_value_error(translate_f16, uncompressed_export(lines), message)
    # Readers. The audited ones read a binary16 result; every other reader of the tag refuses it.
    produced = [*prefix, "v_add_f16_e32 v24, v18, v19", "v_add_nc_u16 v25, v18, v19"]

    def reader(line):
        return uncompressed_export([*produced, line])

    for line in ("v_cvt_f32_f16_e32 v12, v24", "v_cvt_f32_f16_e64 v12, v24",
                 "v_cvt_f32_f16_sdwa v12, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0",
                 "v_cvt_f32_f16_sdwa v12, v24 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:WORD_0",
                 "v_mov_b32_sdwa v12, v24 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0",
                 "v_mov_b32_sdwa v12, v24 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_0",
                 "v_add_f16_e32 v13, v24, v18"):
        if line.startswith("v_add_f16"):
            expect_value_error(translate_f16, reader(line), "Unsupported export")  # a chained op writes v13: not exportable
        else:
            translate_f16(reader(line))
    numeric = "Expected a numeric register"
    undefined = "Reads the undefined high half of a 16-bit result"
    modifiers = "Modifiers on a binary16 arithmetic result are not lowered"
    sdwa = "dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:"
    for line, message in (
            ("v_mov_b32_e32 v12, v24", numeric), ("v_add_f32_e32 v12, v24, v18", numeric),
            ("v_swap_b32 v24, v18", numeric), ("v_add_nc_u32_e32 v12, v24, v18", numeric),
            ("v_add_nc_u16 v12, v24, v18", numeric), ("v_cmp_lt_f16_e32 vcc_lo, v24, v18", numeric),
            ("v_cmp_lt_f16_e64 s52, v24, v18", numeric), ("v_cmp_class_f16_e32 vcc_lo, v24, v18", numeric),
            ("v_cmp_lt_u16_e32 vcc_lo, v24, v18", numeric), ("v_pk_add_f16 v12, v24, v18", numeric),
            ("v_fma_mixlo_f16 v12, v24, v18, v18", numeric), ("v_dot2_f32_f16 v12, v24, v18, v18", numeric),
            ("v_cvt_f16_f32_e32 v24, v18", "Unsupported conversion destination"),
            ("v_cvt_pkrtz_f16_f32_e32 v12, v24, v18", "Unsupported pack operand"),
            ("v_mov_b32_dpp v12, v24 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf", "Expected a numeric VGPR in DPP"),
            ("v_cvt_f32_f16_dpp v12, v24 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf", "Expected a numeric VGPR in DPP"),
            (f"v_cvt_f32_f16_sdwa v12, v24 {sdwa}WORD_1", undefined),
            *((f"v_mov_b32_sdwa v12, v24 {sdwa}{select}", undefined)
              for select in ("WORD_1", "BYTE_0", "BYTE_1", "BYTE_2", "BYTE_3", "DWORD")),
            (f"v_mov_b32_sdwa v12, sext(v24) {sdwa}WORD_0", undefined),
            ("v_mov_b32_sdwa v24, v18 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:WORD_0", undefined),
            ("v_cvt_f32_f16_e32 v12, -v24", modifiers), ("v_cvt_f32_f16_e64 v12, |v24|", modifiers),
            ("v_cvt_f32_f16_e64 v12, -|v24|", modifiers), ("v_cvt_f32_f16_e64 v12, v24 clamp", modifiers),
            (f"v_cvt_f32_f16_sdwa v12, -v24 {sdwa}WORD_0", modifiers),
            (f"v_cvt_f32_f16_sdwa v12, v24 clamp {sdwa}WORD_0", modifiers),
            ("v_cvt_f32_f16_e32 v12, v25", numeric)):
        expect_value_error(translate_f16, reader(line), message)
    # Exports: an uncompressed export of a binary16 result, and the compressed form, whose string-pair check would
    # accept a tag starting with "sample" or "alu".
    for registers in ("v24, v13, v14, v15", "v12, v24, v14, v15", "v12, v13, v14, v24"):
        export = uncompressed_export(produced)
        export[next(i for i, line in enumerate(export) if line.startswith("exp ")):] = [
            f"exp mrt0 {registers} done vm", "s_endpgm"]
        expect_value_error(translate_f16, export, "Unsupported export")
    export = uncompressed_export(produced)
    export[next(i for i, line in enumerate(export) if line.startswith("exp ")):] = [
        "exp mrt0 v24, v24, off, off done compr vm", "s_endpgm"]
    expect_value_error(float_f16, export, "Unsupported export")


def check_controls(directory):
    check_masked_controls()
    check_title_load_controls()
    check_ds_swizzle_controls()
    check_ds_nop_controls()
    for mask in (0x3, 0xC, 0xF):
        program = compressed_export(mask)
        modern = [line.replace("exp mrt0 ", "exp mrt0, ") for line in program]
        source, actual = translate_masked(program)
        assert actual == mask and source == translate_masked(modern)[0]
        assert "float4 main(" in source
    partial = compressed_export(0x3)
    assert "float2 main(" in translate(partial)
    assert translate_masked(uncompressed_export([]))[1] == 0xF
    must_reject(translate, compressed_export(0xC))
    must_reject(translate_int, partial)
    export = partial.index("exp mrt0 v16, v16, off, off done compr vm")
    for invalid in ("exp mrt0 off, off, off, off done compr vm",
                    "exp mrt0 v16, v16, v17, off done compr vm",
                    "exp mrt0 v16, off, off, off done compr vm",
                    "exp mrt0 off, off, v17, off done compr vm"):
        must_reject(translate_masked, partial[:export] + [invalid] + partial[export + 1:])
    for mask in (0x1, 0x8, 0x5, 0xA, 0xF):
        program = uncompressed_export([], mask)
        modern = [line.replace("exp mrt0 ", "exp mrt0, ") for line in program]
        assert translate_masked(program) == translate_masked(modern)
        assert translate_masked(program)[1] == mask
        if mask != 0xF:
            must_reject(translate, program)
            must_reject(translate_int, program)
    must_reject(translate_masked, uncompressed_export([], 0))
    duplicate = uncompressed_export([], 0x5)
    must_reject(translate_masked, [line.replace("v12, off, v14", "v12, off, v12") for line in duplicate])
    # A syntactically valid DPP VGPR must already have a tracked numeric value.
    must_reject(translate, with_constants([
        "v_mov_b32_dpp v12, v255 dpp8:[1,2,3,0,5,6,7,4]",
    ]))
    # Fast-path quad swizzles must hoist QuadRead* into an unconditional local before any ternary.
    dpp_ternary = translate(with_constants([
        "v_cmp_gt_f32_e32 vcc_lo, 0, v12",
        "v_cndmask_b32_dpp v13, v12, v13, vcc_lo quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
        "v_mul_legacy_f32_dpp v12, v12, v13 quad_perm:[0,0,0,0] row_mask:0xf bank_mask:0xf",
    ]))
    assert dpp_ternary.count("QuadReadAcrossX(") == 1 and dpp_ternary.count("QuadReadLaneAt(") == 1
    assert not any("?" in line and "QuadRead" in line for line in dpp_ternary.splitlines())
    check_f16_denorm_controls()
    check_vopc_e32_destination()
    program = bitwise_program()
    for before, after in (("s44, v18", "s44, |v18|"), ("20, v19", "20, v19 clamp"),
                          ("0x00700000, v18, v19", "0x00700000, v18")):
        altered = [line.replace(before, after) for line in program]
        assert altered != program
        must_reject(translate, altered)
    program = buffer_color(True)
    assert "float2 main(" in translate_buffer_color(program)
    assert "SamplerState" not in translate_buffer_color(program)
    must_reject(lambda p: lower(p, 8, 24, None, 0), program)
    must_reject(lambda p: lower(p, 9, 24, None, 32), program)
    must_reject(lambda p: lower(p, 9, 24, None, 0, second_texture_offset=32), program)
    must_reject(lambda p: lower(p, 9, 24, [0] * 4, 0), program)
    must_reject(lambda p: lower(p, 20, 24, None), assembly(False))
    must_reject(translate_buffer_color, [s for s in program if s != "s_waitcnt lgkmcnt(0)"])
    must_reject(translate_buffer_color, [s.replace("off, off", "off, v26") for s in program])
    check_bit_controls()
    check_scheduling_hints()
    check_mode_register_controls()
    check_constant_descriptor_bounds()
    check_scalar_constant_reads()
    check_three_constant_buffers()
    check_affine_coordinates()
    check_scalar_helpers_with_vector_arithmetic()
    check_adapter_skip(directory)
    check_scalar_pair_reads()
    check_readfirstlane_controls()
    check_flat_scratch_controls()
    # Whole-dword SDWA comparisons lower exactly as the e32 forms; byte and word selections are refused.
    for first in range(0, 16, 4):
        assert translate(predicate_group(first, sdwa=True)) == translate(predicate_group(first)), first
    must_reject(translate, with_constants(["v_sub_f32_e64 v18, s45, 0",
        "v_cmp_lt_f32_sdwa vcc_lo, s44, v18 src0_sel:WORD_1 src1_sel:DWORD"]))
    sopc_source = translate(sopc_group())
    assert set(SOPC_OPS) <= ACCEPTED
    for expected in (
        "asint(asuint(params[0].w)) == asint(asuint(params[1].x))",
        "asint(asuint(params[0].w)) < asint(asuint(params[1].x))",
        "asuint(params[0].w) == asuint(params[1].x)",
        "asuint(params[0].w) < asuint(params[1].x)",
        "asuint(params[0].w) == asuint(params[1].y) && asuint(params[1].x) == asuint(params[1].z)",
        "asuint(params[0].w) != asuint(params[1].y) || asuint(params[1].x) != asuint(params[1].z)",
        "((asuint(params[0].w) & (1u << (5u & 31u))) == 0u)",
        "((asuint(params[0].w) & (1u << (5u & 31u))) != 0u)",
        "(((37u & 32u) != 0u ? (asuint(params[1].x) & (1u << (37u & 31u))) : (asuint(params[0].w) & (1u << (37u & 31u)))) == 0u)",
        "(((37u & 32u) != 0u ? (asuint(params[1].x) & (1u << (37u & 31u))) : (asuint(params[0].w) & (1u << (37u & 31u)))) != 0u)",
    ):
        assert expected in sopc_source, expected
    for bad_sopc in (
        "s_cmp_gt_u64 s[44:45], s[46:47]",
        "s_cmp_eq_i64 s[44:45], s[46:47]",
        "s_cmp_eq_u64 s[65:66], s[46:47]",
        "s_cmp_eq_u32 s106, s44",
        "s_cmp_eq_u64 s[105:106], s[46:47]",
    ):
        must_reject(translate, with_constants([bad_sopc]))
    vop2_source = translate(vop2_integer_group())
    assert set(VOP2_INT_OPS) <= ACCEPTED
    for expected in (
        "(asuint(alu20) >> (asuint(params[0].w) & 31u))",
        "asuint(asint(asuint(alu20)) >> (asuint(params[0].w) & 31u))",
        "(asuint(alu20) << (asuint(params[0].w) & 31u))",
        "(asuint(params[0].w) & asuint(alu20))",
        "(asuint(params[0].w) | asuint(alu20))",
        "(asuint(params[0].w) ^ asuint(alu20))",
        "(asuint(params[0].w) + asuint(alu20))",
        "(asuint(params[0].w) - asuint(alu20))",
        "(asuint(alu20) - asuint(params[0].w))",
    ):
        assert expected in vop2_source, expected
    for bad_vop2 in (
        "v_sub_nc_u32_e32 v12, abs(v18), v18",
        "v_subrev_nc_u32_e32 v12, abs(v18), v18",
        "v_and_b32_e32 v12, -v18, v18",
        "v_and_b32_e32 v12, abs(v18), v18",
        "v_and_b32_e32 v12, v106, v18",
        "v_and_b32_e32 v12, s44, v106",
    ):
        must_reject(translate, with_constants(["v_mov_b32_e32 v18, s45", bad_vop2]))
    A, B = 0x12345678, 0x9abcdef0
    assert align_reference(A, B, 0)[3] == B
    assert align_reference(A, B, 0, True)[3] == B
    assert align_reference(A, B, 31)[3] == 0x2468acf1
    assert align_reference(A, B, 32)[3] == B
    assert align_reference(A, B, 32)[3] != A
    assert align_reference(A, B, 3, True)[3] == 0x3456789a
    assert align_reference(A, B, 4, True)[3] == B
    assert align_reference(A, B, 4, True)[3] != A
    assert align_reference(A, B, 4) != align_reference(B, A, 4)
    assert align_reference(A, B, 1, True) != align_reference(B, A, 1, True)
    assert align_reference(0x12345678, 0x9abcdef0, 32) == (
        0x89abcdef, 0x9abcdef0, 0x2468acf1, 0x9abcdef0)
    assert align_reference(0x12345678, 0x9abcdef0, 4, True) == (
        0x789abcde, 0x9abcdef0, 0x3456789a, 0x9abcdef0)
    align_source = "\n".join(translate_int(align_integer_group(byte)) for byte in (False, True))
    assert set(VOP3_ALIGN_OPS) <= ACCEPTED
    for expected in (
        "((4u & 31u) == 0u ? asuint(alu20) : ((asuint(alu20) >> (4u & 31u)) | (params[0].w << (32u - (4u & 31u)))))",
        "((0u & 31u) == 0u ? asuint(alu20) : ((asuint(alu20) >> (0u & 31u)) | (params[0].w << (32u - (0u & 31u)))))",
        "((31u & 31u) == 0u ? asuint(alu20) : ((asuint(alu20) >> (31u & 31u)) | (params[0].w << (32u - (31u & 31u)))))",
        "((params[1].y & 31u) == 0u ? asuint(alu20) : ((asuint(alu20) >> (params[1].y & 31u)) | (params[0].w << (32u - (params[1].y & 31u)))))",
        "((1u & 3u) == 0u ? asuint(alu20) : ((asuint(alu20) >> ((1u & 3u) * 8u)) | (params[0].w << (32u - ((1u & 3u) * 8u)))))",
        "((0u & 3u) == 0u ? asuint(alu20) : ((asuint(alu20) >> ((0u & 3u) * 8u)) | (params[0].w << (32u - ((0u & 3u) * 8u)))))",
        "((3u & 3u) == 0u ? asuint(alu20) : ((asuint(alu20) >> ((3u & 3u) * 8u)) | (params[0].w << (32u - ((3u & 3u) * 8u)))))",
        "((params[1].y & 3u) == 0u ? asuint(alu20) : ((asuint(alu20) >> ((params[1].y & 3u) * 8u)) | (params[0].w << (32u - ((params[1].y & 3u) * 8u)))))",
    ):
        assert expected in align_source, expected
    for bad_align in (
        "v_alignbit_b32 v12, s44, v18",
        "v_alignbit_b32 v12, s44, v18, 4, 5",
        "v_alignbyte_b32 v12, s44, v18",
        "v_alignbyte_b32 v12, s44, v18, 1, 2",
        "v_alignbit_b32_e32 v12, s44, v18, 4",
        "v_alignbyte_b32_e32 v12, s44, v18, 1",
        "v_alignbit_b32 v256, s44, v18, 4",
        "v_alignbyte_b32 v256, s44, v18, 1",
        "v_alignbit_b32 v12, abs(s44), v18, 4",
        "v_alignbyte_b32 v12, s44, -v18, 1",
        "v_alignbit_b32 v12, s44, v18, 4 clamp",
        "v_alignbyte_b32 v12, s44, v18, 1 clamp",
    ):
        must_reject(translate, with_constants(["v_mov_b32_e32 v18, s45", bad_align]))
    vop3_source = "\n".join(translate_int(vop3_ternary_int_group(group)) for group in (1, 2))
    assert set(VOP3_TERNARY_INT_OPS) <= ACCEPTED
    for expected in (
        "(params[0].w + asuint(alu20) + params[1].y)",
        "(params[0].w | asuint(alu20) | params[1].y)",
        "(params[0].w ^ asuint(alu20) ^ params[1].y)",
        "((params[0].w & asuint(alu20)) | params[1].y)",
        "((params[0].w ^ asuint(alu20)) + params[1].y)",
        "((params[0].w << (asuint(alu20) & 31u)) | params[1].y)",
        "((params[0].w << (asuint(alu20) & 31u)) + params[1].y)",
        "((params[0].w + asuint(alu20)) << (params[1].y & 31u))",
    ):
        assert expected in vop3_source, expected
    for op in VOP3_TERNARY_INT_OPS:
        for bad_vop3 in (
            f"{op}_e32 v12, s44, v18, s46",
            f"{op} v12, s44, v18",
            f"{op} v12, s44, v18, s46, 5",
            f"{op} v256, s44, v18, s46",
            f"{op} v12, abs(s44), v18, s46",
            f"{op} v12, s44, -v18, s46",
            f"{op} v12, s44, v18, s46 clamp",
        ):
            must_reject(translate, with_constants(["v_mov_b32_e32 v18, s45", bad_vop3]))
    check_nc_integer_lowering()
    check_e64_integer_lowering()
    check_carry_out_lowering()
    check_carry_in_lowering()
    check_sdwa_integer_lowering()
    check_pack_int16_lowering()
    check_sad_lowering()
    lshl_add_source = translate(sop2_lshl_add_group())
    assert set(SOP2_LSHL_ADD_OPS) <= ACCEPTED
    for expected in (
        "(asuint(params[0].w) << 1u) + asuint(params[1].x)",
        "(asuint(params[0].w) << 2u) + asuint(params[1].x)",
        "(asuint(params[0].w) << 3u) + asuint(params[1].x)",
        "(asuint(params[0].w) << 4u) + asuint(params[1].x)",
        "asuint(params[0].w) >> 31u != 0u",
        "asuint(params[0].w) >> 30u != 0u",
        "asuint(params[0].w) >> 29u != 0u",
        "asuint(params[0].w) >> 28u != 0u",
    ):
        assert expected in lshl_add_source, expected
    for bad_lshl_add in (
        "s_lshl0_add_u32 s62, s44, s45",
        "s_lshl5_add_u32 s62, s44, s45",
        "s_lshl1_add_u32 s62, s44",
        "s_lshl2_add_u32 s62, s44, s45, s46",
        "s_lshl3_add_u32 s62, v12, s45",
        "s_lshl4_add_u32 s62, s44, v13",
        "s_lshl1_add_u32 v12, s44, s45",
    ):
        must_reject(translate, with_constants([bad_lshl_add]))
    absdiff_source = translate(sop2_absdiff_group())
    assert set(SOP2_ABSDIFF_OPS) <= ACCEPTED
    assert "asuint(abs(asint(asuint(params[0].w)) - asint(asuint(params[1].x))))" in absdiff_source, absdiff_source
    assert "!= 0u) ? 1u : 0u;" in absdiff_source, absdiff_source
    for bad_absdiff in (
        "s_absdiff_i64 s[62:63], s[44:45], s[46:47]",
        "s_absdiff_u32 s62, s44, s45",
        "s_absdiff_i32 s62, s44",
        "s_absdiff_i32 s62, s44, s45, s46",
        "s_absdiff_i32 s62, v12, s45",
        "s_absdiff_i32 s62, s44, v13",
        "s_absdiff_i32 v12, s44, s45",
    ):
        must_reject(translate, with_constants([bad_absdiff]))
    rgb = rgb_sample()
    translate(rgb)
    rga = translate([line.replace("dmask:0x7", "dmask:0xb") for line in rgb])
    assert re.search(r"sample\d+\.w", rga), rga
    for old, new in (("v[12:14]", "v[12:15]"), ("dmask:0x7", "dmask:0x3")):
        must_reject(translate, [line.replace(old, new) for line in rgb])
    wait = rgb.index("s_waitcnt vmcnt(0)")
    must_reject(translate, rgb[:wait] + rgb[wait+1:])
    must_reject(translate, rgb[:wait] + ["v_mov_b32_e32 v14, 1.0"] + rgb[wait:])
    translate(rgb[:wait] + ["v_mov_b32_e32 v15, 1.0"] + rgb[wait:])
    lod_zero = lod_zero_sample()
    assert ".SampleLevel(samp, float2(uv.x, uv.y), 0.0f)" in translate(lod_zero)
    must_reject(translate, [line.replace("image_sample_lz ", "image_sample ") for line in lod_zero])
    mixed_masks = [line.replace("image_sample ", "image_sample_lz ") for line in assembly(False)]
    restore = mixed_masks.pop(mixed_masks.index("s_and_b64 exec, exec, s[50:51]"))
    phase_two = next(i for i, line in enumerate(mixed_masks) if line.startswith("v_interp_p2"))
    mixed_masks.insert(phase_two, restore)
    must_reject(translate, mixed_masks)
    # v_cvt_f32_f16 widens the low half through widen16 and applies modifiers after widening.
    body = translate(with_constants(HALF_BODY)).split("float4 main", 1)[1]
    assert body.count("widen16(asuint(") == 4 and "(-abs(widen16(" in body, body
    must_reject(translate, with_constants(["v_cvt_f32_f16_e32 v12, 1.0"]))
    must_reject(translate, with_constants(["v_cvt_f32_f16_sdwa v12, s44 src0_sel:WORD_1"]))
    check_scalar_controls()
    # v_cvt_f16_f32 keeps the destination's high half and narrows the modified source through narrow16.
    body = translate(with_constants(NARROW_BODY)).split("float4 main", 1)[1]
    assert body.count("narrow16(") == 3 and body.count("& 0xffff0000u) | narrow16(") == 3, body
    assert "narrow16((-abs(" in body, body
    must_reject(translate, with_constants(["v_cvt_f16_f32_e64 v12, s44 clamp"]))
    must_reject(translate, with_constants(["v_cvt_f16_f32_sdwa v12, s44 dst_sel:WORD_1"]))
    # An inline integer keeps its bit pattern, and a destination holding interpolation state is refused.
    body = translate(with_constants(["v_cvt_f16_f32_e32 v12, -1", "v_cvt_f16_f32_e64 v13, -|0.5|"])).split("float4 main", 1)[1]
    assert "narrow16(asfloat(4294967295u))" in body and "narrow16((-abs(0.5f)))" in body, body
    for register in ("v0", "v1", "v8", "v9"):
        must_reject(translate, with_constants([f"v_cvt_f16_f32_e32 {register}, s44"]))
    # v_cvt_f32_u32/v_cvt_f32_i32 lower the plain e32 encoding without modifiers and the e64
    # encoding with its optional clamp; every other modifier or variant is refused.
    program = uncompressed_export(cvt_int_body(0xFFFFFFFF, 0x80000000))
    body = translate(program).split("float4 main", 1)[1]
    assert body.count("float(asuint(") == 2 and body.count("float(asint(") == 2, body
    modern = [line.replace("exp mrt0 ", "exp mrt0, ") for line in program]
    assert modern != program and translate(modern) == translate(program)
    wait = program.index("s_waitcnt lgkmcnt(0)", program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc"))
    must_reject(translate, program[:wait] + ["v_cvt_f32_u32_e32 v19, s45"] + program[wait:])
    program = uncompressed_export(cvt_int_body(0xFFFFFFFF, 0x80000000, e64=True))
    body = translate(program).split("float4 main", 1)[1]
    assert body.count("float(asuint(") == 2 and body.count("float(asint(") == 2, body
    modern = [line.replace("exp mrt0 ", "exp mrt0, ") for line in program]
    assert modern != program and translate(modern) == translate(program)
    program = uncompressed_export(cvt_int_body(0xFFFFFFFF, 0x80000000, e64=True, clamp=True))
    body = translate(program).split("float4 main", 1)[1]
    assert body.count("saturate(float(asuint(") == 2 and body.count("saturate(float(asint(") == 2, body
    # The e64 pending-load guard mirrors the e32 one; the modifier list pins the assembler-measured
    # refusals: omod in every probed spelling (LLVM 15 rejects it, so it cannot round-trip), abs/neg
    # on the integer source, the DPP/SDWA variants, and clamp on the modifier-free e32 encoding.
    must_reject(translate, program[:wait] + ["v_cvt_f32_u32_e64 v19, s45"] + program[wait:])
    for bad in ("v_cvt_f32_u32_e32 v12, s44 clamp", "v_cvt_f32_i32_e32 v12, s44 clamp",
                "v_cvt_f32_u32_e64 v12, |s44|", "v_cvt_f32_i32_e64 v12, -s44",
                "v_cvt_f32_u32_e64 v12, s44 omod:1", "v_cvt_f32_i32_e64 v12, s44 omod:2",
                "v_cvt_f32_u32_e64 v12, s44 omod:3", "v_cvt_f32_i32_e64 v12, s44 clamp omod:2",
                "v_cvt_f32_u32_e64 v12, s44 omod(2)", "v_cvt_f32_i32_e64 v12, s44 mul:2",
                "v_cvt_f32_u32_dpp v12, v13 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
                "v_cvt_f32_i32_sdwa v12, v13 dst_sel:DWORD src0_sel:DWORD"):
        must_reject(translate, with_constants([bad]))
    # v_cvt_u32_f32_e32/v_cvt_i32_f32_e32 lower only the plain e32 encoding, through one helper each.
    program = uncompressed_export(cvt_float_body(0x3fc00000, 0xc0300000))
    full = translate_int(program)
    body = full.split("uint4 main", 1)[1]
    assert body.count("cvt_u32_f32(") == 2 and body.count("cvt_i32_f32(") == 2, body
    assert full.count("uint cvt_u32_f32(uint bits)") == 1 and full.count("uint cvt_i32_f32(uint bits)") == 1
    modern = [line.replace("exp mrt0 ", "exp mrt0, ") for line in program]
    assert modern != program and translate_int(modern) == full
    assert "asfloat(1077936128u)" in translate_int(uncompressed_export(["v_cvt_u32_f32_e32 v12, 0x40400000"]))
    wait = program.index("s_waitcnt lgkmcnt(0)", program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc"))
    expect_value_error(translate_int, program[:wait] + ["v_cvt_u32_f32_e32 v19, s45"] + program[wait:],
                       "Constant read before load completion")
    for bad in ("v_cvt_u32_f32_e64 v12, s44 clamp", "v_cvt_i32_f32_e64 v12, s44 omod:2",
                "v_cvt_flr_i32_f32_e64 v12, s44 clamp", "v_cvt_rpi_i32_f32_e64 v12, s44 omod:2",
                "v_cvt_u32_f32_dpp v12, v13 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
                "v_cvt_i32_f32_sdwa v12, v13 dst_sel:DWORD src0_sel:DWORD",
                "v_cvt_flr_i32_f32_dpp v12, v13 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
                "v_cvt_rpi_i32_f32_sdwa v12, v13 dst_sel:DWORD src0_sel:DWORD",
                "v_cvt_u32_f32_e32 v12, 0.5", "v_cvt_u32_f32_e32 v12, 1", "v_cvt_i32_f32_e32 v12, -1",
                "v_cvt_flr_i32_f32_e32 v12, 0.5", "v_cvt_rpi_i32_f32_e32 v12, -1",
                "v_cvt_u32_f32_e64 v12, 0.5", "v_cvt_u32_f32_e64 v12, 1", "v_cvt_i32_f32_e64 v12, -1",
                "v_cvt_flr_i32_f32_e64 v12, 0.5", "v_cvt_rpi_i32_f32_e64 v12, 1",
                "v_cvt_i32_f32_e64 v12, -0x40400000", "v_cvt_u32_f32_e64 v12, |-s44|",
                "v_cvt_flr_i32_f32_e64 v12, -0x40400000", "v_cvt_rpi_i32_f32_e64 v12, |-s44|",
                "v_cvt_i32_f32_e64 v12, --s44", "v_cvt_flr_i32_f32_e64 v12, --s44"):
        expect_value_error(translate, with_constants([bad]), "Unsupported instruction")
    for inst, diagnostic in (("v_cvt_i32_f32_e32 v256, s44", "Invalid signed conversion register"),
                             ("v_cvt_i32_f32_e32 v12, v255", "Unbound pixel input v255"),
                             ("v_cvt_flr_i32_f32_e32 v256, s44", "Invalid signed conversion register"),
                             ("v_cvt_rpi_i32_f32_e32 v12, v255", "Unbound pixel input v255"),
                             ("v_cvt_u32_f32_e64 v256, |s44|", "Invalid unsigned conversion register"),
                             ("v_cvt_i32_f32_e64 v12, -s106", "Invalid signed conversion register"),
                             ("v_cvt_flr_i32_f32_e64 v256, |s44|", "Invalid signed conversion register"),
                             ("v_cvt_rpi_i32_f32_e64 v12, -s106", "Invalid signed conversion register"),
                             ("v_cvt_flr_i32_f32_e64 v12, -|v255|", "Unbound pixel input v255"),
                             ("v_cvt_u32_f32_e64 v12, -|v255|", "Unbound pixel input v255")):
        expect_value_error(translate, with_constants([inst]), diagnostic)
    # Positive control: plain e32 and e64 lower identically through their shared/lazy helpers.
    for good in ("v_cvt_u32_f32_e32 v12, s44", "v_cvt_i32_f32_e32 v12, s44",
                 "v_cvt_u32_f32_e32 v12, 0x40400000", "v_cvt_flr_i32_f32_e32 v12, 0x40400000",
                 "v_cvt_rpi_i32_f32_e32 v12, 0xbf000000"):
        e32_hlsl = translate(with_constants([good]))
        assert translate(with_constants([good.replace("_e32", "_e64")])) == e32_hlsl
    for op in ("flr_i32", "rpi_i32"):
        e32_s = translate(with_constants(["s_mov_b32 s44, 0x3effffff", f"v_cvt_{op}_f32_e32 v12, s44"]))
        assert translate(with_constants(["s_mov_b32 s44, 0x3effffff", f"v_cvt_{op}_f32_e64 v12, s44"])) == e32_s
    flr_rpi_hlsl = translate_int(uncompressed_export(cvt_flr_rpi_pair_body(0x3effffff, 0xbf000001, e64=False)))
    assert flr_rpi_hlsl.count("uint cvt_flr_i32_f32(uint bits)") == 1
    assert flr_rpi_hlsl.count("uint cvt_rpi_i32_f32(uint bits)") == 1
    assert "uint cvt_i32_f32(uint bits)" not in flr_rpi_hlsl
    flr_mod_hlsl = translate_int(uncompressed_export(cvt_flr_rpi_mod_body("flr_i32", CVT_FLR_RPI_MOD_SETS[0])))
    assert flr_mod_hlsl.count("uint cvt_flr_i32_f32(uint bits)") == 1
    assert "uint cvt_rpi_i32_f32(uint bits)" not in flr_mod_hlsl
    assert "& 0x7fffffffu)" in flr_mod_hlsl
    assert "^ 0x80000000u)" in flr_mod_hlsl
    proven_s44 = ["s_mov_b32 s44, 0xc0200000"]
    assert translate(with_constants([*proven_s44, "v_cvt_flr_i32_f32_e64 v12, abs(s44)"])) == translate(with_constants([*proven_s44, "v_cvt_flr_i32_f32_e64 v12, |s44|"]))
    assert translate(with_constants([*proven_s44, "v_cvt_rpi_i32_f32_e64 v12, neg(s44)"])) == translate(with_constants([*proven_s44, "v_cvt_rpi_i32_f32_e64 v12, -s44"]))
    for alt in ("-abs(s44)", "neg(abs(s44))", "neg(|s44|)"):
        assert translate(with_constants([*proven_s44, f"v_cvt_rpi_i32_f32_e64 v12, {alt}"])) == translate(with_constants([*proven_s44, "v_cvt_rpi_i32_f32_e64 v12, -|s44|"]))
    # Post-modifier in-range control: -(+2^31) = -2^31 is exact in signed i32.
    translate(with_constants(["v_cvt_flr_i32_f32_e64 v12, neg(0x4f000000)"]))
    translate(with_constants(["v_cvt_rpi_i32_f32_e64 v12, -|0x4f000000|"]))
    # Explicit refusals for unproven runtime inputs and exceptional/out-of-range constants (including after modifiers).
    for unproven in ("v_cvt_flr_i32_f32_e32 v12, s44", "v_cvt_rpi_i32_f32_e64 v12, -s44",
                     "v_cvt_flr_i32_f32_e32 v12, v12", "v_cvt_rpi_i32_f32_e64 v12, |v12|"):
        expect_value_error(translate, with_constants([unproven]), "Unproven runtime input")
    expect_value_error(translate_int, uncompressed_export(["v_mov_b32_e32 v18, s44", "v_cvt_flr_i32_f32_e32 v12, v18"]),
                       "Unproven runtime input")
    expect_value_error(translate_int, uncompressed_export(["v_mov_b32_e32 v18, 0x3f800000", "v_add_f32_e32 v18, v18, v18",
                                                           "v_cvt_rpi_i32_f32_e32 v12, v18"]),
                       "Unproven runtime input")
    for bad_const in ("v_cvt_flr_i32_f32_e32 v12, 0x7fc00000", "v_cvt_rpi_i32_f32_e32 v12, 0xff800001",
                      "v_cvt_flr_i32_f32_e32 v12, 0x7f800000", "v_cvt_rpi_i32_f32_e32 v12, 0xff800000",
                      "v_cvt_flr_i32_f32_e32 v12, 0x4f000000", "v_cvt_rpi_i32_f32_e32 v12, 0xcf000001",
                      "v_cvt_flr_i32_f32_e64 v12, |0xcf000000|", "v_cvt_rpi_i32_f32_e64 v12, neg(0xcf000000)",
                      "v_cvt_flr_i32_f32_e64 v12, -|0x7f800000|"):
        expect_value_error(translate, with_constants([bad_const]), "Unproven exceptional or out-of-range input")
    expect_value_error(translate_int, uncompressed_export(["s_mov_b32 s44, 0xcf000000", "v_cvt_flr_i32_f32_e64 v12, |s44|"]),
                       "Unproven exceptional or out-of-range input")
    expect_value_error(translate_int, uncompressed_export(["v_mov_b32_e32 v18, 0x4f000000", "v_cvt_rpi_i32_f32_e32 v12, v18"]),
                       "Unproven exceptional or out-of-range input")
    # e64 source modifiers apply bitwise sign-bit ops before the shared helper, never FP abs/neg.
    e64_mod_hlsl = translate_int(uncompressed_export(CVT_FLOAT_E64_I32_MOD_BODY))
    assert e64_mod_hlsl.count("uint cvt_i32_f32(uint bits)") == 1
    assert "(asuint(asfloat(params[0].w)) & 0x7fffffffu)" in e64_mod_hlsl, e64_mod_hlsl
    assert "(asuint(asfloat(params[1].x)) ^ 0x80000000u)" in e64_mod_hlsl, e64_mod_hlsl
    assert "((asuint(asfloat(params[1].y)) & 0x7fffffffu) ^ 0x80000000u)" in e64_mod_hlsl, e64_mod_hlsl
    assert "abs(" not in e64_mod_hlsl.split("uint4 main", 1)[1], e64_mod_hlsl
    assert translate(with_constants(["v_cvt_u32_f32_e64 v12, abs(s44)"])) == translate(with_constants(["v_cvt_u32_f32_e64 v12, |s44|"]))
    assert translate(with_constants(["v_cvt_i32_f32_e64 v12, neg(s44)"])) == translate(with_constants(["v_cvt_i32_f32_e64 v12, -s44"]))
    for alt in ("-abs(s44)", "neg(abs(s44))", "neg(|s44|)"):
        assert translate(with_constants([f"v_cvt_i32_f32_e64 v12, {alt}"])) == translate(with_constants(["v_cvt_i32_f32_e64 v12, -|s44|"]))
    lit_e64 = translate_int(uncompressed_export([
        "v_cvt_u32_f32_e64 v12, |0xc0300000|",
        "v_cvt_i32_f32_e64 v13, neg(0x40400000)",
        "v_cvt_i32_f32_e64 v14, -|0xc0300000|",
        "v_cvt_u32_f32_e64 v15, 0x40400000",
    ]))
    assert "(asuint(asfloat(3224371200u)) & 0x7fffffffu)" in lit_e64
    assert "(asuint(asfloat(1077936128u)) ^ 0x80000000u)" in lit_e64
    assert "((asuint(asfloat(3224371200u)) & 0x7fffffffu) ^ 0x80000000u)" in lit_e64
    e64_wait = program.index("s_waitcnt lgkmcnt(0)", program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc"))
    expect_value_error(translate_int, program[:e64_wait] + ["v_cvt_i32_f32_e64 v19, -|s45|"] + program[e64_wait:],
                       "Constant read before load completion")
    # Guards: the destination may not overlap an outstanding image_sample, and the lowering needs the entry mask.
    sample = next(i for i, line in enumerate(with_constants([])) if line.startswith("image_sample"))
    for op in ("u32", "i32", "flr_i32", "rpi_i32"):
        for suffix, src in (("_e32", "0x40400000"), ("_e64", "-|0x40400000|")):
            overlap = with_constants([])
            overlap.insert(sample + 1, f"v_cvt_{op}_f32{suffix} v12, {src}")
            expect_value_error(translate, overlap, "Write overlaps an outstanding sample")
            ok = with_constants([])
            ok.insert(sample + 1, f"v_cvt_{op}_f32{suffix} v20, {src}")
            translate(ok)
            if op != "u32" or suffix == "_e64":
                quad = with_constants([])
                quad.insert(quad.index("s_wqm_b64 exec, exec") + 1, f"v_cvt_{op}_f32{suffix} v20, {src}")
                expect_value_error(translate, quad, "Invalid conversion")
            entry = with_constants([])
            entry.insert(entry.index("s_and_b64 exec, exec, s[50:51]") + 1, f"v_cvt_{op}_f32{suffix} v20, {src}")
            translate(entry)
    pending_src = with_constants([])
    pending_src.insert(sample + 1, "v_cvt_u32_f32_e64 v20, |v12|")
    expect_value_error(translate, pending_src, "Sample used before vmcnt wait")
    pending_flr = with_constants([])
    pending_flr.insert(sample + 1, "v_cvt_flr_i32_f32_e64 v20, -|v12|")
    expect_value_error(translate, pending_flr, "Sample used before vmcnt wait")
    # v_cvt_f32_u32 and v_cvt_f32_i32 reject float modifiers, float literals, out-of-range immediates and bad registers.
    body = translate(with_constants([
        "v_cvt_f32_i32_e32 v12, -42",
        "v_cvt_f32_u32_e32 v13, 42",
        "v_cvt_f32_i32_e64 v14, 0x80000000",
        "v_cvt_f32_u32_e64 v15, 0x80000000 clamp",
    ])).split("float4 main", 1)[1]
    assert "float(asint(asfloat(4294967254u)))" in body and "float(asuint(asfloat(42u)))" in body, body
    assert "float(asint(asfloat(2147483648u)))" in body and "saturate(float(asuint(asfloat(2147483648u))))" in body, body
    plain_body = translate(with_constants(["v_cvt_f32_i32_e32 v12, s44", "v_cvt_f32_u32_e32 v13, s45"]))
    assert "float(asint(params[0].w))" in plain_body and "float(asuint(params[1].x))" in plain_body, plain_body
    for op in ("v_cvt_f32_u32_e32", "v_cvt_f32_i32_e32"):
        must_reject(translate, with_constants([f"{op} v12, -v18"]))
        must_reject(translate, with_constants([f"{op} v12, |v18|"]))
        must_reject(translate, with_constants([f"{op} v12, abs(v18)"]))
        must_reject(translate, with_constants([f"{op} v12, -s44"]))
        must_reject(translate, with_constants([f"{op} v12, |s44|"]))
        must_reject(translate, with_constants([f"{op} v12, abs(s44)"]))
        must_reject(translate, with_constants([f"{op} v12, 1.5"]))
        must_reject(translate, with_constants([f"{op} v12, 0x100000000"]))
        must_reject(translate, with_constants([f"{op} s40, v18"]))
    must_reject(translate, with_constants(["v_cvt_f32_u32_sdwa v12, s44 src0_sel:WORD_1"]))
    must_reject(translate, with_constants(["v_cvt_f32_i32_sdwa v12, s44 src0_sel:WORD_1"]))
    ctrl_prog = assembly(False)
    ctrl_wait = ctrl_prog.index("s_waitcnt vmcnt(0)")
    must_reject(translate, ctrl_prog[:ctrl_wait] + ["v_cvt_f32_u32_e32 v12, 0"] + ctrl_prog[ctrl_wait:])
    must_reject(translate, ctrl_prog[:ctrl_wait] + ["v_cvt_f32_i32_e32 v12, 0"] + ctrl_prog[ctrl_wait:])
    ctrl_pos = ctrl_prog.index("s_wqm_b64 exec, exec") + 1
    must_reject(translate, ctrl_prog[:ctrl_pos] + ["v_cvt_f32_u32_e32 v20, 0"] + ctrl_prog[ctrl_pos:])
    must_reject(translate, ctrl_prog[:ctrl_pos] + ["v_cvt_f32_i32_e32 v20, 0"] + ctrl_prog[ctrl_pos:])
    ctrl_base = with_constants([])
    ctrl_lgkm = ctrl_base.index("s_waitcnt lgkmcnt(0)")
    must_reject(translate, ctrl_base[:ctrl_lgkm] + ["v_cvt_f32_u32_e32 v12, s44"] + ctrl_base[ctrl_lgkm:])
    must_reject(translate, ctrl_base[:ctrl_lgkm] + ["v_cvt_f32_i32_e32 v12, s44"] + ctrl_base[ctrl_lgkm:])
    archive = directory / "input.bin"
    code = bytes.fromhex("000080bf 000081bf")  # s_nop 0 followed by s_endpgm
    check_container_selector(directory, code)
    record = envelope(code)
    archive.write_bytes(b"another_fixture\0" + record)
    assert native_program(archive, "fixture")[1] == code
    long_name = b"synthetic_pixel_with_a_printable_length"
    archive.write_bytes(envelope(code, long_name))
    assert native_program(archive, long_name.decode())[1] == code
    for invalid in (record[:-1], record + b"\0" + record, envelope(code + code)):
        archive.write_bytes(invalid)
        must_reject(native_program, archive, "fixture")

    partial = dual_program("partial")
    wait = partial.index("s_waitcnt vmcnt(1)")
    # Keep the early read of A but issue B first. Now A is still pending.
    reversed_samples = partial.copy()
    reversed_samples[wait - 2:wait] = reversed(reversed_samples[wait - 2:wait])
    must_reject(translate_dual, reversed_samples)
    for count in (2, 63):
        changed = partial.copy()
        changed[wait] = f"s_waitcnt vmcnt({count})"
        must_reject(translate_dual, changed)
    for early_use in ("v_mov_b32_e32 v24, v20", "v_mov_b32_e32 v20, v12",
                      "v_mul_legacy_f32_e32 v12, 1.0, v20",
                      "v_cvt_pkrtz_f16_f32_e32 v24, v12, v20"):
        changed = partial.copy()
        changed.insert(wait + 1, early_use)
        must_reject(translate_dual, changed)
    # Each of B's four destinations remains protected from a constant write.
    for reg in range(20, 24):
        changed = partial.copy()
        changed.insert(wait + 1, f"v_mov_b32_e32 v{reg}, 0")
        must_reject(translate_dual, changed)
    # Packing A is legal here even though B is outstanding.
    translate_dual(partial[:wait + 1] + ["v_cvt_pkrtz_f16_f32_e32 v24, v12, v13"] + partial[wait + 1:])
    # A larger legal wait after draining does not invalidate completed samples.
    end_wait = partial.index("s_waitcnt vmcnt(0)")
    drained = partial[:end_wait + 1] + ["s_waitcnt vmcnt(63)"] + partial[end_wait + 1:]
    translate_dual(drained)
    must_reject(translate_dual, [op.replace("vmcnt(63)", "vmcnt(64)") for op in drained])

    # Composite waitcnt forms with multiple counters, single counters, or raw 0.
    for legal_wait in (
        "s_waitcnt vmcnt(7) lgkmcnt(0)",
        "s_waitcnt vmcnt(7) lgkmcnt(63)",
        "s_waitcnt lgkmcnt(0) vmcnt(0)",
        "s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)",
        "s_waitcnt 0",
        "s_waitcnt 0x0",
    ):
        translate_dual(partial[:end_wait + 1] + [legal_wait] + partial[end_wait + 1:])
    for illegal_wait in (
        "s_waitcnt lgkmcnt(64)",
        "s_waitcnt expcnt(8)",
        "s_waitcnt vmcnt(0) vmcnt(1)",
        "s_waitcnt lgkmcnt(0) lgkmcnt(1)",
        "s_waitcnt expcnt(0) expcnt(1)",
        "s_waitcnt invalid",
    ):
        must_reject(translate_dual, partial[:end_wait + 1] + [illegal_wait] + partial[end_wait + 1:])

    # Verify that composite waitcnt with lgkmcnt(0) satisfies constant read requirements,
    # while lgkmcnt(63) keeps pending scalar loads outstanding.
    cbuf_program = with_constants(["v_mov_b32_e32 v8, s44"])
    cbuf_wait = cbuf_program.index("s_waitcnt lgkmcnt(0)")
    cbuf_composite_ok = cbuf_program.copy()
    cbuf_composite_ok[cbuf_wait] = "s_waitcnt vmcnt(7) lgkmcnt(0)"
    translate(cbuf_composite_ok)
    cbuf_composite_pending = cbuf_program.copy()
    cbuf_composite_pending[cbuf_wait] = "s_waitcnt vmcnt(7) lgkmcnt(63)"
    must_reject(translate, cbuf_composite_pending)

    # 32-bit vector integer ALU operations: lshl_add_u32, lshl_or_b32, and_or_b32, or3_b32, add3_u32, mul_lo_u32.
    alu_prog = uncompressed_export([
        "v_lshl_add_u32 v12, s44, 2, s45",
        "v_lshl_or_b32 v13, s44, 4, s45",
        "v_and_or_b32 v14, s44, s45, s46",
        "v_or3_b32 v15, s44, s45, s46",
        "v_mul_lo_u32 v12, v12, s47",
        "v_add3_u32 v13, v13, v14, v15",
    ])
    alu_src = translate(alu_prog)
    assert "((asuint(params[0].w) << (2u & 31u)) + asuint(params[1].x))" in alu_src
    assert "((asuint(params[0].w) << (4u & 31u)) | asuint(params[1].x))" in alu_src
    assert "((asuint(params[0].w) & asuint(params[1].x)) | asuint(params[1].y))" in alu_src
    assert "(asuint(params[0].w) | asuint(params[1].x) | asuint(params[1].y))" in alu_src
    assert "(asuint(alu20) * asuint(params[1].z))" in alu_src
    assert "(asuint(alu21) + asuint(alu22) + asuint(alu23))" in alu_src
    # Negative controls: arity mismatch and register out-of-range
    must_reject(translate, uncompressed_export(["v_lshl_add_u32 v12, s44, 2"]))
    must_reject(translate, uncompressed_export(["v_lshl_add_u32 v12, s44, 2, s45, s46"]))
    must_reject(translate, uncompressed_export(["v_mul_lo_u32 v12, s44"]))
    must_reject(translate, uncompressed_export(["v_mul_lo_u32 v12, s44, s45, s46"]))
    must_reject(translate, uncompressed_export(["v_or3_b32 v256, s44, s45, s46"]))

    # LLVM 15 prints "exp mrt0 v16, ..."; newer LLVM prints "exp mrt0, v16, ...".
    for program in (assembly(False), assembly(True), curve(), shaping(), arithmetic(), three_way(), rounding()):
        modern = [line.replace("exp mrt0 ", "exp mrt0, ") for line in program]
        assert modern != program
        assert translate(modern) == translate(program)
        must_reject(translate, [line.replace("exp mrt0 ", "exp mrt0,, ") for line in program])

    # Uncompressed binary32 exports need four distinct computed or sampled registers; an integer
    # export refuses the compressed form and nothing may follow the export.
    program = uncompressed_export(["v_mov_b32_e32 v12, s44", "v_mov_b32_e32 v13, s45",
                                   "v_mov_b32_e32 v14, s46", "v_mov_b32_e32 v15, s47"])
    modern = [line.replace("exp mrt0 ", "exp mrt0, ") for line in program]
    assert translate(modern) == translate(program)
    source = translate(program)
    assert "return float4(alu" in source and "export16(" not in source.split("float4 main", 1)[1]
    integral = lower(program, 20, 24, [1, 2, 3, 4], constant_descriptor=64, integer_export=True)
    assert "uint4 main" in integral and "return uint4(asuint(" in integral
    must_reject(lower, with_constants([]), 20, 24, [1, 2, 3, 4],
                constant_descriptor=64, integer_export=True)
    export = program.index("exp mrt0 v12, v13, v14, v15 done vm")
    must_reject(translate, program[:export] + ["exp mrt0 v12, v12, v13, v14 done vm"] + program[export+1:])
    must_reject(translate, program[:export] + ["exp mrt0 v12, v13, v14, v255 done vm"] + program[export+1:])
    sampled = assembly(False)
    sample_wait = sampled.index("s_waitcnt vmcnt(0)")
    must_reject(translate, sampled[:sample_wait] + ["exp mrt0 v12, v13, v14, v15 done vm", "s_endpgm"])
    must_reject(translate, program[:export+1] + ["v_mov_b32_e32 v20, 1.0"] + program[export+1:])

    # The literal of v_fmaak_f32 is its last operand and that of v_fmamk_f32 its second; a register there is refused.
    body = translate(with_constants(LITERAL_FMA_BODY)).split("float4 main", 1)[1]
    assert body.count("mad(") == 4 and "asfloat(1048576000u)" in body, body
    must_reject(translate, with_constants(["v_fmaak_f32 v12, s44, v13, v14"]))
    must_reject(translate, with_constants(["v_fmamk_f32 v12, s44, v13, 0x3e800000"]))
    must_reject(translate, with_constants(["v_fmaak_f32 v12, s44, 0x3e800000"]))
    for old, new in (("v_sqrt_f32_e32 v18, v18", "v_sqrt_f32_e32 v18, v18, v18"),
                     ("v_add_f32_e32 v19, s44, v13", "v_add_f32_e32 v19, v13")):
        must_reject(translate, [line.replace(old, new) for line in shaping()])
    check_three_way_templates()
    source = translate(three_way())
    for expected in ("max(min(params[1].z, sample16.y), min(max(params[1].z, sample16.y), sample16.x))",
                     "max(max(sample16.x, sample16.z), sample16.y)",
                     "max(min(sample16.x, params[0].w), min(max(sample16.x, params[0].w), params[1].x))",
                     "min(min(abs(params[1].y), sample16.y), alu"):
        assert expected in source, expected
    for old, new in (("v_med3_f32 v12, v12, s44, s45", "v_med3_f32 v12, v12, s44"),
                     ("v_med3_f32 v15, s47", "v_med3_i32 v15, |s47|"),
                     ("v_max3_f32 v14, v12", "v_max3_u32 v14, |v12|"),
                     ("v_min3_f32 v13, |s46|", "v_min3_f16 v13, |s46|")):
        changed = [line.replace(old, new) for line in three_way()]
        assert changed != three_way()
        must_reject(translate, changed)

    program = truncate_values()
    source = translate(program)
    assert "trunc(params[1].x)" in source
    must_reject(translate, [line.replace("v13, s45", "v13, s45, v12") for line in program])
    wait = program.index("s_waitcnt vmcnt(0)")
    program.remove("v_trunc_f32_e32 v13, s45")
    program.insert(wait, "v_trunc_f32_e32 v13, s45")
    must_reject(translate, program)

    # v_cmp_class_f32 classifies the raw encoding; a mask with modifiers or a non-numeric operand is refused.
    body = translate(with_constants(CLASS_BODY)).split("float4 main", 1)[1]
    assert body.count("class32(") == 4 and "& 512u" in body and "asuint(alu" in body, body
    must_reject(translate, with_constants(["v_cmp_class_f32_e64 vcc_lo, s44, -s45"]))
    must_reject(translate, with_constants(["v_cmp_class_f32_e64 vcc_lo, s44"]))
    program = sopk_scalar()
    source = translate(program)
    assert "(asuint(params[0].w) * 3u)" in source and "(asuint(params[1].x) + 4294967294u)" in source

    scc_program = with_constants([
        "s_mov_b32 s52, 0", "s_mov_b32 s53, 0", "s_mov_b32 s54, 0", "s_mov_b32 s55, 0",
        "s_mov_b32 s50, s44", "s_addk_i32 s50, 0xfffe",
        "s_mov_b32 s70, 0", "s_mov_b32 s71, 0", "s_mov_b32 s74, 1", "s_mov_b32 s75, 0",
        "s_cmov_b64 s[70:71], s[74:75]",
    ])

    assert "0x80000000u" in translate(scc_program)
    cmov = scc_program.index("s_cmov_b64 s[70:71], s[74:75]")
    after_mul = scc_program[:cmov] + ["s_mulk_i32 s50, 0x3"] + scc_program[cmov:]
    assert "0x80000000u" in translate(after_mul)

    for old, new in (("s_mulk_i32 s50, 0x3", "s_mulk_i32 s50, 0x10000"),
                     ("s_addk_i32 s51, 0xfffe", "s_addk_i32 s51, 0x10001"),
                     ("s_mov_b32 s50, s44", "s_mov_b32 s50, s28"),
                     ("s_mulk_i32 s50, 0x3", "s_mulk_i32 v50, 0x3"),
                     ("s_addk_i32 s51, 0xfffe", "s_addk_i32 s51, v45")):
        changed = [line.replace(old, new) for line in program]
        assert changed != program
        must_reject(translate, changed)

    wait = program.index("s_waitcnt lgkmcnt(0)", program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc"))
    must_reject(translate, program[:wait] + ["s_mulk_i32 s44, 0x3"] + program[wait:])

    cmpk_program = sopk_cmpk_cmovk()
    cmpk_source = translate_int(cmpk_program)
    for expected in ("(asint(params[0].w) < asint(4294934528u))",
                     "(params[1].x < 32768u)",
                     "? 4294967294u : ",
                     "? 32767u : "):
        assert expected in cmpk_source, expected
    refs = [sopk_cmpk_cmovk_reference(words) for words in SOPK_CMPK_WORDS]
    for bit in range(len(SOPK_CMPK_OPS)):
        assert any(r[0] & (1 << bit) for r in refs) and any(not (r[0] & (1 << bit)) for r in refs)
        assert any(r[1] & (1 << bit) for r in refs) and any(not (r[1] & (1 << bit)) for r in refs)
    assert any(r[2] == 0xfffffffe for r in refs) and any(r[2] == 0x13579bdf for r in refs)
    base = assembly(False)
    wqm = base.index("s_wqm_b64 exec, exec")
    andexec = base.index("s_and_b64 exec, exec, s[50:51]")
    must_reject(translate, base[:andexec] + ["s_mov_b32 s50, 0", "s_cmovk_i32 s50, 1"] + base[andexec:])
    must_reject(translate, base[:andexec] + ["s_cmp_eq_u32 1, 1"] + base[andexec:andexec + 1]
                + ["s_mov_b32 s50, 0", "s_cmovk_i32 s50, 1"] + base[andexec + 1:])
    must_reject(translate, base[:wqm] + ["s_cmp_eq_u32 1, 1"] + base[wqm:wqm + 1]
                + ["s_mov_b32 s50, 0", "s_cmovk_i32 s50, 1"] + base[wqm + 1:])
    must_reject(translate_int, [line.replace("s_mov_b32 s50, s46", "s_nop 0") for line in cmpk_program])
    cmpk_wait = cmpk_program.index("s_waitcnt lgkmcnt(0)", cmpk_program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc"))
    for bad in ("s_cmpk_eq_i32 s44, 0x0", "s_cmovk_i32 s44, 0x1"):
        must_reject(translate_int, cmpk_program[:cmpk_wait] + ["s_cmp_eq_u32 1, 1", bad] + cmpk_program[cmpk_wait:])
    for old, new in (("s_cmpk_eq_i32 s44, 0x8000", "s_cmpk_eq_i32 s44, 0x10000"),
                     ("s_cmpk_eq_i32 s44, 0x8000", "s_cmpk_eq_i32 s44, -32769"),
                     ("s_cmpk_eq_u32 s45, 0x8000", "s_cmpk_eq_u32 s45, 0x10000"),
                     ("s_cmpk_eq_u32 s45, 0x8000", "s_cmpk_eq_u32 s45, -1"),
                     ("s_cmpk_eq_i32 s44, 0x8000", "s_cmpk_eq_i32 s106, 0x0"),
                     ("s_cmpk_eq_i32 s44, 0x8000", "s_cmpk_eq_i32 v12, 0x0"),
                     ("s_cmovk_i32 s50, 0xfffe", "s_cmovk_i32 s50, 0x10000"),
                     ("s_cmovk_i32 s50, 0xfffe", "s_cmovk_i32 s50, -32769"),
                     ("s_cmovk_i32 s50, 0xfffe", "s_cmovk_i32 s106, 0x1"),
                     ("s_cmovk_i32 s50, 0xfffe", "s_cmovk_i32 v12, 0x1")):
        changed = [line.replace(old, new) for line in cmpk_program]
        assert changed != cmpk_program, new
        must_reject(translate_int, changed)
    source = translate(rounding())
    for expected in ("= ceil(alu", "= round(alu", "= round(params[1].x);", "= ceil((-params[1].y));"):
        assert expected in source, expected
    # Neighbouring forms: other widths, output modifiers, a second source and the IEEE-flag reciprocal.
    for old, new in (("v_ceil_f32_e32 v12, v18", "v_ceil_f16_e32 v12, v18"),
                     ("v_ceil_f32_e32 v12, v18", "v_ceil_f64_e32 v[12:13], v[18:19]"),
                     ("v_rndne_f32_e32 v13, v19", "v_rndne_f16_e32 v13, v19"),
                     ("v_rndne_f32_e32 v13, v19", "v_rndne_f32_e32 v13, v19, v18"),
                     ("v_rndne_f32_e64 v14, s45", "v_rndne_f32_e64 v14, s45 mul:2"),
                     ("v_ceil_f32_e64 v15, -s46", "v_ceil_f32_e64 v15, -s46 div:2"),
                     ("v_ceil_f32_e64 v15, -s46", "v_rcp_iflag_f32_e64 v15, -s46, v12"),
                     ("v_ceil_f32_e64 v15, -s46", "v_ceil_f32_e64 v15, -s28")):
        changed = [line.replace(old, new) for line in rounding()]
        assert changed != rounding()
        must_reject(translate, changed)

    program = absolute_value()
    alternate = [line.replace("|v12|", "abs(v12)") for line in program]
    assert translate(program) == translate(alternate)
    # Each cube operation reads the three coordinates once and needs all three operands.
    body = translate(with_constants(CUBE_BODY)).split("float4 main", 1)[1]
    assert body.count("precise float3 cube") == 4 and body.count("* 2.0f)") == 1, body
    must_reject(translate, with_constants(["v_mov_b32_e32 v18, s45", "v_cubeid_f32 v12, s44, v18"]))
    # SDWA moves select and place fields; UNUSED_SEXT, byte selections for half widening and
    # non-canonical SDWA text are refused.
    body = translate(with_constants(SDWA_PRESERVE_BODY + SDWA_PAD_BODY)).split("float4 main", 1)[1]
    assert body.count("& 0xffff0000u) | ") == 2 and body.count("asint(") == 1 and body.count("widen16(") == 2, body
    must_reject(translate, with_constants(["v_mov_b32_sdwa v12, s44 dst_sel:BYTE_0 dst_unused:UNUSED_SEXT src0_sel:WORD_1"]))
    must_reject(translate, with_constants(["v_cvt_f32_f16_sdwa v12, s44 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1"]))
    must_reject(translate, with_constants(["v_mov_b32_sdwa v12, s44 src0_sel:WORD_1"]))
    # UNUSED_PRESERVE keeps bits only from a tracked numeric value; DWORD and padded writes may replace anything.
    for register in ("v0", "v1", "v8", "v9", "v20"):
        must_reject(translate, with_constants([f"v_mov_b32_sdwa {register}, s44 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1"]))
    body = translate(with_constants(["v_mov_b32_sdwa v12, -0.5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD",
                                     "v_mov_b32_sdwa v20, s44 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_0",
                                     "v_mov_b32_sdwa v13, v20 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD"])).split("float4 main", 1)[1]
    assert "asuint((-0.5f))" in body, body
    # SDWA arithmetic takes whole-dword sources only and refuses output modifiers and a src1_sel on a unary op.
    body = translate(with_constants(SDWA_ARITH[3][0])).split("float4 main", 1)[1]
    assert body.count("& 0xffff0000u) | ") == 1 and body.count("& 0xff00ffffu) | ") == 1, body
    must_reject(translate, with_constants(["v_mov_b32_e32 v18, s45", sdwa_arith("add", "v12", ["s44", "v18 mul:2"])]))
    must_reject(translate, with_constants([sdwa_arith("add", "v12", ["s44", "v13"]).replace("src0_sel:DWORD", "src0_sel:WORD_1")]))
    must_reject(translate, with_constants([sdwa_arith("floor", "v12", ["s44"]) + " src1_sel:DWORD"]))
    # A partial UNUSED_PRESERVE result needs a tracked numeric destination, as for SDWA moves.
    for register in ("v0", "v8", "v20"):
        must_reject(translate, with_constants([sdwa_arith("floor", register, ["s44"], "BYTE_2", "UNUSED_PRESERVE")]))
    check_sdwa_unary_lowering()

    program = assembly(False)
    for old, new in (("s_mov_b32 s21, 0", "s_mov_b32 s21, 1"),
                     ("s_mov_b32 s40, 1", "s_mov_b32 s40, 8"),
                     ("attr0.y", "attr1.y"),
                     ("s_wqm_b64 exec, exec", "s_branch 0")):
        must_reject(translate, [line.replace(old, new) for line in program])
    must_reject(translate, program[:-1])
    for wait in ("s_waitcnt vmcnt(0)", "s_waitcnt lgkmcnt(0)"):
        must_reject(translate, [line for line in program if line != wait])
    for exp_inst in ("s_waitcnt_expcnt null, 0x0", "s_waitcnt_expcnt null, 0x3", "s_waitcnt_expcnt null, 0",
                     "s_waitcnt_expcnt s0, 0x0", "s_waitcnt_expcnt 0x3", "s_waitcnt_expcnt 0",
                     "s_waitcnt expcnt(0)", "s_waitcnt expcnt(7)"):
        assert "return float4(" in translate([exp_inst] + program)
        assert translate(program[:-1] + [exp_inst, program[-1]]) == translate(program)
    must_reject(translate, program + ["s_waitcnt_expcnt null, 0"])
    must_reject(translate, ["s_waitcnt_expcnt null, 0x8"] + program)
    must_reject(translate, ["s_waitcnt_expcnt null, 8"] + program)
    must_reject(translate, ["s_waitcnt_expcnt 0x8"] + program)
    must_reject(translate, ["s_waitcnt_expcnt invalid"] + program)
    must_reject(translate, ["s_waitcnt_expcnt null, invalid"] + program)
    must_reject(translate, ["s_waitcnt expcnt(8)"] + program)
    check_sopk_waitcnt(program)
    check_s_version()

    wait = program.index("s_waitcnt vmcnt(0)")
    overwrite = "v_floor_f32_e32 v13, 1.0"
    must_reject(translate, program[:wait] + [overwrite] + program[wait:])
    free = translate(program[:wait] + ["v_floor_f32_e32 v24, 1.0"] + program[wait:])
    late = translate(program[:wait+1] + [overwrite] + program[wait+1:])
    assert "export16(sample15.y)" in free
    assert "export16(alu17)" in late

    program = arithmetic()
    source = translate(program)
    for expected in ("= sample16.x;", "= (params[2].x * alu20);", "= saturate((alu21 + params[2].y));",
                     "= params[2].z;", "= (alu23 - sample16.y);", "= saturate((alu25 - params[2].w));",
                     "= asfloat(1048576000u);", "= alu29;"):
        assert expected in source, expected
    # A move is a bit copy: LLVM prints integer inline constants as integers.
    for operand, expected in (("1", "asfloat(1u)"), ("-4", "asfloat(4294967292u)"),
                              ("-0.5", "(-0.5f)"), ("0.15915494", "0.15915494f"), ("s40", "asfloat(1u)")):
        changed = [line.replace("v20, 0x3e800000", f"v20, {operand}") for line in program]
        assert f"alu28 = {expected};" in translate(changed), operand
    for old, new in (("v18, v12", "v18, v0"), ("v18, v12", "v18, v8"), ("v18, v12", "v18, v30"),
                     ("v19, s46", "v19, s28"), ("v19, s46", "v19, -s46"), ("v19, s46", "v19, s46 clamp"),
                     ("v18, v12", "v18, v12, v13"), ("v14, -0.5, v14", "v14, v14"),
                     ("v19, v13, v19", "v19, v13, v19, v18"), ("s45 clamp", "s45 clamp mul:2")):
        changed = [line.replace(old, new) for line in program]
        assert changed != program
        must_reject(translate, changed)
    first = program.index("v_mov_b32_e32 v18, v12")
    early = program[:first-1] + [program[first], program[first-1]] + program[first+1:]
    assert early[first] == "s_waitcnt vmcnt(0)"
    must_reject(translate, early)  # copies the sample before the wait
    # Until vmcnt(0), the sample may still fill v12 to v15, so a move into one of them has no defined
    # order with it. The same move into a free register, or into v12 after the wait, is accepted.
    wait = program.index("s_waitcnt vmcnt(0)")
    for register in ("v12", "v13", "v14", "v15"):
        must_reject(translate, program[:wait] + [f"v_mov_b32_e32 {register}, 1.0"] + program[wait:])
    must_reject(translate, program[:wait] + ["v_ceil_f32_e32 v12, 1.0"] + program[wait:])
    must_reject(translate, program[:wait] + ["v_sin_f32_e32 v12, 1.0"] + program[wait:])
    must_reject(translate, program[:wait] + ["v_cos_f32_e64 v13, 1.0"] + program[wait:])
    trig = translate(with_constants(TRIG_BODY))
    assert trig.count("sin32(") == 3 and trig.count("cos32(") == 3, trig  # two calls plus each definition
    assert "saturate(cos32((-params" in trig and "sin32((-abs(" in trig and "cos32(params" in trig, trig
    free = translate(program[:wait] + ["v_mov_b32_e32 v21, 1.0"] + program[wait:])
    assert free.index("tex.Sample") < free.index("= 1.0f;") < free.index("= sample16.x;")
    late = translate(program[:wait+1] + ["v_mov_b32_e32 v12, 1.0"] + program[wait+1:])
    assert "= 1.0f;" in late and "= sample16.x;" not in late
    pos = program.index("v_cvt_pkrtz_f16_f32_e32 v17, v14, v15")
    must_reject(translate, program[:pos] + ["v_mov_b32_e32 v20, v16"] + program[pos:])
    pos = program.index("s_wqm_b64 exec, exec") + 1
    must_reject(translate, program[:pos] + ["v_mov_b32_e32 v20, 1.0"] + program[pos:])
    pos = program.index("v_interp_p1_f32_e32 v9, v0, attr0.y")
    partial = program[:pos] + ["v_interp_p1_f32_e32 v10, v0, attr0.x"] + program[pos:]
    translate(partial)  # an unpaired first phase is allowed, but its partial value is not numeric
    must_reject(translate, [line.replace("v18, v12", "v18, v10") for line in partial])
    pos = program.index("s_waitcnt lgkmcnt(0)", program.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0x20"))
    must_reject(translate, program[:pos] + ["v_mov_b32_e64 v19, s46"] + program[pos:])

    for old, new in (("0x40", "0x48"), ("0xc", "0xfe"), ("0xc", "0x100"),
                     ("s_waitcnt lgkmcnt(0)", "s_nop 0"), ("s[44:47], 0xc", "s[40:43], 0xc")):
        must_reject(translate, [line.replace(old, new) for line in curve()])
    source = translate(select())
    assert translate(select(True)) == source
    for expected in ("(sample16.x >= sample16.y || isnan(sample16.x) || isnan(sample16.y))",
                     "sample16.x == sample16.y",
                     "params[0].w < sample16.x", "sample16.y >= params[1].x",
                     "? 1.0f : asfloat(0u);", "? sample16.w : params[1].z;"):
        assert expected in source, expected
    for wave64 in (False, True):
        program = select(wave64)
        mask = "s[52:53]" if wave64 else "s52"
        absolute_compare = [line.replace("v13, s45", "v13, |v12|") for line in program]
        assert "abs(" in translate(absolute_compare)
        pos = program.index(f"v_cndmask_b32_e64 v13, v13, s46, {mask}")
        must_reject(translate, program[:pos] + ["s_mov_b32 s52, 0"] + program[pos:])
        for old, new in ((f"s46, {mask}", "s46, s[53:54]" if wave64 else "s46, s50"),
                         ("v_cmp_eq_f32_e32", "v_cmp_eq_f64_e32"),
                         ("v_cmp_ge_f32_e64", "v_cmpx_ge_f32_e64"),
                         (f"v_cmp_ge_f32_e64 {mask}", "v_cmp_ge_f32_e64 " + ("s[56:57]" if wave64 else "s56")),
                         ("v13, s45", "v13, |v12"),
                         ("v15, s47, v15", "v15, s50, v15")):
            changed = [line.replace(old, new) for line in program]
            assert changed != program
            must_reject(translate, changed)
        pos = program.index("s_waitcnt vmcnt(0)")
        must_reject(translate, program[:pos] + program[pos+1:])
    # v_fract_f32 goes through the clamped helper, never through a bare D3D frac.
    body = translate(shaping()).split("float4 main", 1)[1]
    assert "fract32(" in body and "frac(" not in body, body
    # v_mad_f32 and v_mac_f32 use the unfused, denormal-flushing helper, never the fma lowering.
    body = translate(with_constants(MAD_BODY)).split("float4 main", 1)[1]
    assert body.count("mad32(") == 4 and " mad(" not in body, body
    must_reject(translate, with_constants(["v_mad_f32 v12, s44, s45"]))
    must_reject(translate, with_constants(["v_mac_f32_e32 v14, s44, v15, v14"]))
    check_e64_approx_lowering()
    check_mac_legacy_lowering()
    check_dpp_vop3_only_refused()
    # v_madak_f32 and v_madmk_f32 use the same helper. The literal is the last operand of madak and the
    # second of madmk; a register there, a missing operand, and the e64, DPP and SDWA spellings are refused.
    body = translate(with_constants(MAD_LITERAL_BODY)).split("float4 main", 1)[1]
    assert body.count("mad32(") == 4 and " mad(" not in body and "asfloat(3414167552u)" in body, body
    must_reject(translate, with_constants(["v_madak_f32 v12, s44, v13, v14"]))
    must_reject(translate, with_constants(["v_madmk_f32 v12, s44, v13, 0x3e800000"]))
    must_reject(translate, with_constants(["v_madak_f32 v12, s44, 0x3e800000"]))
    must_reject(translate, with_constants(["v_madak_f32_e64 v12, s44, v13, 0x3e800000"]))
    must_reject(translate, with_constants(["v_madmk_f32_e64 v12, s44, 0x3e800000, v13"]))
    must_reject(translate, with_constants(["v_madak_f32_dpp v12, v13, v14, 0x3e800000 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf"]))
    must_reject(translate, with_constants(["v_madmk_f32_sdwa v12, v13, 0x3e800000, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD"]))

    # 3-operand vector float math: v_div_fmas_f32 and rejection of unverified v_fma_f16 / v_div_scale_f32
    for opcode in ("v_fma_f16", "v_fma_f16_e64"):
        for insn in (
            f"{opcode} v22, v19, v20, v21",
            f"{opcode} v22, v19, v20, v21 op_sel:[0,0,0]",
            f"{opcode} v22, v19, v20, v21 clamp",
        ):
            try:
                translate(with_constants([insn]))
            except ValueError as error:
                assert "Unverified v_fma_f16" in str(error), error
            else:
                raise AssertionError(f"Unverified {insn} was accepted")

    # Rounding controls for binary16: independent case where RNE and RTZ differ.
    val_diff = 1.0 + 2.0 ** -11 + 2.0 ** -12
    rne_packed = struct.unpack("<H", struct.pack("<e", val_diff))[0]
    rtz_packed = struct.unpack("<H", struct.pack("<e", half_rtz(val_diff)))[0]
    assert rne_packed == 0x3c01 and rtz_packed == 0x3c00
    assert rne_packed != rtz_packed

    # Fused-half halfway case distinguishing single rounding to binary16 from rounding to float32 first:
    # a = 0x3c69 (1.1025390625), b = 0x3faa (1.916015625), c = 0x89d1 (-0.00017750263214111328).
    # Exact a * b + c = 2.11230456829071044921875.
    # Midpoint between 0x4039 (2.111328125) and 0x403a (2.11328125) is exactly 2.1123046875.
    # Single rounding to binary16 rounds down to 0x4039.
    # Intermediate rounding to float32 produces 2.1123046875 (0x40073000), which lands exactly on
    # the halfway tie and rounds up to even mantissa 0x403a.
    a_f16 = half_value(0x3c69)
    b_f16 = half_value(0x3faa)
    c_f16 = half_value(0x89d1)
    exact_fma = a_f16 * b_f16 + c_f16
    single_rounded_f16 = struct.unpack("<H", struct.pack("<e", exact_fma))[0]
    float32_intermediate = struct.unpack("<f", struct.pack("<f", exact_fma))[0]
    double_rounded_f16 = struct.unpack("<H", struct.pack("<e", float32_intermediate))[0]
    assert single_rounded_f16 == 0x4039
    assert double_rounded_f16 == 0x403a
    assert single_rounded_f16 != double_rounded_f16

    div_source = translate(valu_div_fmas_program())
    assert div_source.count("float div_fmas32(") == 1
    assert "div_fmas32" not in translate(assembly(False))
    assert "4294967296.0" in div_source
    # v_div_fmas_f32 scaling boundary: intermediate subnormal value (2^-127) before multiplying by 2^32.
    assert "asfloat(8388608u)" in div_source or "asfloat(0x00800000u)" in div_source
    s0_bound = struct.unpack("<f", struct.pack("<I", 0x00800000))[0]
    inter_fma = s0_bound * 0.5
    assert 0 < inter_fma < 2.0 ** -126
    scaled_fma = inter_fma * (2.0 ** 32)
    assert scaled_fma == 2.0 ** -95

    # Cancellation check: (1 + 2^-23) * (1 - 2^-23) - 1.0 = -2^-46
    s0_cancel = struct.unpack("<f", struct.pack("<I", 0x3f800001))[0]
    s1_cancel = struct.unpack("<f", struct.pack("<I", 0x3f7ffffe))[0]
    s2_cancel = struct.unpack("<f", struct.pack("<I", 0xbf800000))[0]
    f32_cancel = struct.unpack("<f", struct.pack("<f", s0_cancel * s1_cancel))[0]
    assert f32_cancel == 1.0 and f32_cancel + s2_cancel == 0.0  # unfused float32 rounds to 1.0
    cancel_exact = s0_cancel * s1_cancel + s2_cancel  # double maintains exact product + addend
    assert cancel_exact == -(2.0 ** -46)
    assert struct.unpack("<I", struct.pack("<f", cancel_exact))[0] == 0xa8800000
    cancel_scaled = cancel_exact * (2.0 ** 32)
    assert cancel_scaled == -(2.0 ** -14)
    assert struct.unpack("<I", struct.pack("<f", cancel_scaled))[0] == 0xb8800000

    # Pre-scale underflow check: 2^-126 * 2^-25 = 2^-151 (underflows float32 to 0.0 before scale)
    s0_under = struct.unpack("<f", struct.pack("<I", 0x00800000))[0]
    s1_under = struct.unpack("<f", struct.pack("<I", 0x33000000))[0]
    f32_under = struct.unpack("<f", struct.pack("<f", s0_under * s1_under))[0]
    assert f32_under == 0.0  # intermediate underflow in float32
    under_scaled = (s0_under * s1_under) * (2.0 ** 32)  # double preserves 2^-151, scaling to 2^-119
    assert under_scaled == 2.0 ** -119
    assert struct.unpack("<I", struct.pack("<f", under_scaled))[0] == 0x04000000

    # Midpoint double-rounding check: exact product 1.5 + 3*2^-24, addend -2^-100.
    # The exact sum is strictly less than midpoint, rounding down to 0x3fc00001 (unscaled)
    # and 0x4fc00001 (scaled by 2^32), whereas uncorrected double rounding yields 0x3fc00002 / 0x4fc00002.
    s0_mid = struct.unpack("<f", struct.pack("<I", 0x3f800001))[0]
    s1_mid = struct.unpack("<f", struct.pack("<I", 0x3fc00000))[0]
    s2_mid = struct.unpack("<f", struct.pack("<I", 0x8d800000))[0]
    p_mid = s0_mid * s1_mid
    s_mid = p_mid + s2_mid
    v_mid = s_mid - p_mid
    err_mid = (p_mid - (s_mid - v_mid)) + (s2_mid - v_mid)
    assert err_mid < 0.0
    for opcode in ("v_div_scale_f32", "v_div_scale_f32_e64"):
        for denominator, numerator in (("1.0", "0x71800000"), ("0", "1.0"),
                                       ("0x00800000", "1.0"), ("1.0", "1.0")):
            program = uncompressed_export([
                f"v_mov_b32_e32 v19, {denominator}",
                f"v_mov_b32_e32 v20, {numerator}",
                f"{opcode} v12, vcc_lo, v19, v19, v20",
                "v_cndmask_b32_e64 v13, 0, 1.0, vcc_lo",
            ])
            try:
                translate(program)
            except ValueError as error:
                assert "Unverified v_div_scale_f32" in str(error), error
            else:
                raise AssertionError("Unverified division prescaling was accepted")

    for bad in (
        "v_div_scale_f32 v21, vcc_lo, v19, v20",
        "v_div_scale_f32 s21, vcc_lo, v19, v20, v19",
        "v_div_scale_f32 v21, v22, v19, v20, v19",
        "v_div_fmas_f32 v12, v19, v20, v21",
        "v_div_fmas_f32 v12, v19, v20, v21 clamp clamp",
        "v_div_fmas_f32 v12, v19, v20, v21 op_sel:[1,0,0]",
        "v_div_fmas_f32 v12, v19, v20",
        "v_div_fmas_f32 s12, v19, v20, v21",
        "v_div_fmas_f32 v256, v19, v20, v21",
    ):
        must_reject(translate, with_constants([bad]))

    # 64-bit scalar ALU and BFE controls.
    for op in SCALAR_BFE64_OPS:
        bfe = scalar_bfe64(op)
        translate_int(bfe)
        signed = op == "s_bfe_i64"
        high = "4294967295u" if signed else "0u"
        literal = translate_int(scalar_bfe64(op, "0x80000000"))
        assert f"s_bfe64(uint2(2147483648u, {high})," in literal
        imm_ctrl = translate_int(scalar_bfe64(op, control="0x200020"))
        assert ", 2097184u," in imm_ctrl
        must_reject(translate_int, scalar_bfe64(op, "0x8123456789abcdef"))
        must_reject(translate_int, scalar_bfe64(op, "0x100000000"))
        pos = bfe.index(f"{op} s[44:45], s[44:45], s46")
        for operands in ("s[45:46], s[44:45], s46", "exec, s[44:45], s46",
                         "s[44:45], exec, s46", "s[44:45], s[44:45], s[46:47]"):
            must_reject(translate_int, bfe[:pos] + [f"{op} {operands}"] + bfe[pos+1:])
        wait = bfe.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc") + 1
        for operands in ("s[44:45], 0, 0", "s[50:51], s[44:45], 0", "s[50:51], 0, s46"):
            must_reject(translate_int, bfe[:wait] + [f"{op} {operands}"] + bfe[wait:])
    assert set(SCALAR_BFE64_OPS) <= ACCEPTED
    for op in SCALAR_BITSET64_OPS:
        lowered = translate_int(scalar_bitset64(op))
        assert "_bit = params[1].y & 63u;" in lowered, lowered
        for index in (0, 31, 32, 63, 64):
            assert "salu" in translate_int(scalar_bitset64(op, str(index)))
        for bad in ("v12", "s[46:47]"):
            must_reject(translate_int, scalar_bitset64(op, bad))
        untracked = [line.replace("s[44:45]", "s[60:61]")
                     for line in scalar_bitset64(op)]
        must_reject(translate_int, untracked)
    select64 = scalar_select64()
    translate_int(select64)
    selected = select64.index(S_CSELECT_B64)
    for bad in ("s_cselect_b64 s[45:46], s[44:45], s[46:47]",
                "s_cselect_b64 exec, s[44:45], s[46:47]",
                "s_cselect_b64 s[44:45], exec, s[46:47]"):
        must_reject(translate_int, select64[:selected] + [bad] + select64[selected+1:])
    wait = select64.index("s_buffer_load_dwordx4 s[44:47], s[44:47], 0xc") + 1
    for bad in ("s_cselect_b64 s[44:45], 0, -1", "s_cselect_b64 s[48:49], s[44:45], 0"):
        must_reject(translate_int, select64[:wait] + ["s_cmp_eq_u32 0, 0", bad] + select64[wait:])
    program = scalar64()
    source = translate(program)
    for expected in ("uint2 s_lshl_b64(", "uint2 s_lshr_b64(", "uint2 s_ashr_i64(",
                     "uint2 s_bfm_b64(", "uint s_bfe_u32(", "uint s_bfe_i32(",
                     "reversebits(", "asfloat(salu"):
        assert expected in source, expected
    # Existing programs without shifts/bfm/bfe emit none of the scalar helper functions.
    assert "uint2 s_lshl_b64(" not in translate(assembly(False))
    # Verify 32-bit hex literal sign-extension for s_ashr_i64 vs zero-extension for b64 ops.
    ashr_pos = program.index("s_ashr_i64 s[72:73], s[52:53], s56")
    signed_lit = translate(program[:ashr_pos] + ["s_ashr_i64 s[72:73], 0x80000000, 4"] + program[ashr_pos+1:])
    assert "s_ashr_i64(uint2(2147483648u, 4294967295u), 4u)" in signed_lit
    unsigned_lit = translate(program[:ashr_pos] + ["s_and_b64 s[72:73], s[52:53], 0x80000000"] + program[ashr_pos+1:])
    assert "2147483648u" in unsigned_lit and "0u)" in unsigned_lit
    # Verify all 10 SCC observation bits flip between the two constant sets.
    refs = []
    for values in SCALAR64_SETS:
        constants = bytearray(256)
        struct.pack_into("<12I", constants, 16, *values)
        refs.append(scalar64_reference(program, struct.unpack("<64I", constants)))
    assert (refs[0][1] ^ refs[1][1]) == 0x3ff, (bin(refs[0][1]), bin(refs[1][1]))
    # Negative controls: exec save/restore, #257 ops, out-of-scope 64-bit ops, invalid operands and SCC rules.
    body_start = program.index("s_and_b64 s[72:73], s[52:53], s[54:55]")
    for bad in (
        "s_and_saveexec_b64 s[70:71], s[52:53]",
        "s_or_saveexec_b64 s[70:71], s[52:53]",
        "s_andn2_saveexec_b64 s[70:71], s[52:53]",
        "s_or_b64 exec, exec, s[52:53]",
        "s_and_b64 vcc, s[52:53], s[54:55]",
        "s_cmov_b64 exec, s[52:53]",
        "s_not_b64 vcc, s[52:53]",
        "s_and_b64 s[70:71], exec, s[52:53]",
        "s_xor_b64 s[70:71], vcc, s[52:53]",
        "s_cmov_b64 s[70:71], s[52:53]",  # before any SCC write
        "s_and_b64 s[53:54], s[52:53], s[54:55]",  # unaligned pair
        "s_and_b64 s[106:107], s[52:53], s[54:55]",  # out-of-range SGPR pair
        "s_bfe_u32 s106, s60, 0x80004",  # out-of-range SGPR
        "s_bfe_u32 s70, v12, 0x80004",  # VGPR source
        "s_bfe_u32 s70, m0, 0x80004",  # m0 source
        "s_and_b64 s[70:71], s[28:29], s[52:53]",  # texture descriptor source
        "s_and_b64 s[70:71], s[44:45], s[52:53]",  # CBV descriptor source
        "s_and_b64 s[70:71], s[50:51], s[52:53]",  # saved exec source
        "s_and_b64 s[70:71], s[20:21], s[52:53]",  # texture_ptr source
        "s_and_b64 s[70:71], s[90:91], s[52:53]",  # unset source
        "s_cselect_b64 s[70:71], s[52:53], s[54:55]",  # before any SCC write
        "s_bfm_b32 s70, v12, 8",  # VGPR source
        "s_bfm_b32 s106, 4, 8",  # out-of-range SGPR
        "s_bfm_b32 s[70:71], 4, 8",  # pair destination
        "s_bfm_b32 s70, 4",  # missing operand
        "s_bfm_b32 s70, s90, 8",  # unset source
        "s_bfm_b64 s70, 4, 8",  # single destination
        "s_bfm_b64 s[70:71], v12, 8",  # VGPR source
        "s_bfm_b64 s[70:71], 4",  # missing operand
        "s_bfm_b64 s[106:107], 4, 8",  # out-of-range SGPR pair
        "s_cselect_b32 s70, s56, s57",
    ):
        must_reject(translate, program[:body_start] + [bad] + program[body_start:])
    for supported in (
        "s_add_u32 s70, s56, s57",
        "s_and_b32 s70, s56, s57",
        "s_not_b32 s70, s56",
        "s_bfm_b32 s70, 4, 8",
        "s_cmp_eq_i32 s56, s57",
    ):
        translate(program[:body_start] + [supported] + program[body_start:])
    # s_cmov_b64 rejects unset or non-numeric destination pairs even after SCC is set.
    after_and = body_start + 1
    for bad_dst in ("s[90:91]", "s[44:45]", "s[50:51]", "s[28:29]"):
        must_reject(translate, program[:after_and] + [f"s_cmov_b64 {bad_dst}, s[52:53]"] + program[after_and:])
    # Pending scalar load read and write overlap must be rejected.
    load_wait = program.index("s_waitcnt lgkmcnt(0)", program.index("s_buffer_load_dwordx4 s[60:63], s[44:47], 0x30"))
    for early in ("s_and_b64 s[70:71], s[52:53], s[54:55]", "s_not_b64 s[52:53], 0", "s_bfe_u32 s60, 1, 0x80004"):
        must_reject(translate, program[:load_wait] + [early] + program[load_wait:])
    # s_wqm_b64 and exec-restoring s_and_b64 clear tracked SCC.
    wqm_pos = program.index("s_wqm_b64 exec, exec")
    exec_restore_pos = program.index("s_and_b64 exec, exec, s[50:51]")
    prefix = ["s_mov_b32 s90, 0", "s_mov_b32 s91, 0", "s_not_b64 s[92:93], 0"]
    must_reject(translate, program[:wqm_pos] + prefix + [program[wqm_pos], "s_cmov_b64 s[90:91], s[92:93]"] + program[wqm_pos+1:])
    must_reject(translate, program[:exec_restore_pos] + prefix + [program[exec_restore_pos], "s_cmov_b64 s[90:91], s[92:93]"] + program[exec_restore_pos+1:])
    # A scalar result cannot supply a sampler word, texture address high word, or v_cndmask condition.
    sample_pos = program.index("image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D")
    must_reject(translate, program[:sample_pos] + ["s_bfe_u32 s40, 1, 0x10000"] + program[sample_pos:])
    sel = select(True)
    cnd_pos = sel.index("v_cndmask_b32_e64 v13, v13, s46, s[52:53]")
    must_reject(translate, sel[:cnd_pos] + ["s_not_b64 s[52:53], 0"] + sel[cnd_pos:])

    # Wave32 execution mask lifecycle: save, WQM, restore (via s_and_b64, s_mov_b64, s_and_b32, s_mov_b32)
    # and rejections for mismatched registers, wrong mask state, and SCC invalidation.
    base_wave32 = list(assembly(False))
    save_idx = base_wave32.index("s_mov_b64 s[50:51], exec")
    wqm_idx = base_wave32.index("s_wqm_b64 exec, exec")
    restore_idx = base_wave32.index("s_and_b64 exec, exec, s[50:51]")

    w32_and = list(base_wave32)
    w32_and[save_idx] = "s_mov_b64 s14, exec_lo"
    w32_and[wqm_idx] = "s_wqm_b64 exec_lo, exec_lo"
    w32_and[restore_idx] = "s_and_b64 exec_lo, exec_lo, s14"
    assert translate(w32_and) == translate(base_wave32)

    w32_mov = list(base_wave32)
    w32_mov[save_idx] = "s_mov_b64 s14, exec_lo"
    w32_mov[wqm_idx] = "s_wqm_b64 exec_lo, exec_lo"
    w32_mov[restore_idx] = "s_mov_b64 exec_lo, s14"
    assert translate(w32_mov) == translate(base_wave32)

    w32_b32 = list(base_wave32)
    w32_b32[save_idx] = "s_mov_b32 s14, exec_lo"
    w32_b32[wqm_idx] = "s_wqm_b32 exec_lo, exec_lo"
    w32_b32[restore_idx] = "s_and_b32 exec_lo, exec_lo, s14"
    assert lower(w32_b32, 20, 24, [1, 2, 3, 4], constant_descriptor=64, wave_size=32) == translate(base_wave32)
    must_reject(translate, w32_b32)
    must_reject(lambda p: lower(p, 20, 24, [1, 2, 3, 4], constant_descriptor=64, wave_size=64), w32_b32)

    w32_b32_mov = list(base_wave32)
    w32_b32_mov[save_idx] = "s_mov_b32 s14, exec_lo"
    w32_b32_mov[wqm_idx] = "s_wqm_b32 exec_lo, exec_lo"
    w32_b32_mov[restore_idx] = "s_mov_b32 exec_lo, s14"
    assert lower(w32_b32_mov, 20, 24, [1, 2, 3, 4], constant_descriptor=64, wave_size=32) == translate(base_wave32)
    must_reject(translate, w32_b32_mov)
    must_reject(lambda p: lower(p, 20, 24, [1, 2, 3, 4], constant_descriptor=64, wave_size=64), w32_b32_mov)

    must_reject(lambda p: lower(p, 20, 24, [1, 2, 3, 4], constant_descriptor=64, wave_size=16), base_wave32)
    must_reject(lambda p: lower(p, 20, 24, [1, 2, 3, 4], constant_descriptor=64, wave_size=32.0), w32_b32)
    invalid_save = w32_b32[:save_idx] + ["s_mov_b32 s106, exec_lo"] + w32_b32[save_idx + 1:]
    invalid_save[restore_idx] = "s_and_b32 exec_lo, exec_lo, s106"
    must_reject(lambda p: lower(p, 20, 24, [1, 2, 3, 4], constant_descriptor=64, wave_size=32), invalid_save)
    must_reject(lambda p: lower(p, 20, 24, [1, 2, 3, 4], constant_descriptor=64, wave_size="wave32"), base_wave32)

    w64_mov = list(base_wave32)
    w64_mov[restore_idx] = "s_mov_b64 exec, s[50:51]"
    assert translate(w64_mov) == translate(base_wave32)

    w32_disasm_pair = list(base_wave32)
    w32_disasm_pair[save_idx] = "s_mov_b64 s50, exec_lo"
    w32_disasm_pair[wqm_idx] = "s_wqm_b64 exec_lo, exec_lo"
    w32_disasm_pair[restore_idx] = "s_and_b64 exec, exec, s[50:51]"
    assert translate(w32_disasm_pair) == translate(base_wave32)

    w32_disasm_mov_pair = list(base_wave32)
    w32_disasm_mov_pair[save_idx] = "s_mov_b64 s50, exec_lo"
    w32_disasm_mov_pair[wqm_idx] = "s_wqm_b64 exec_lo, exec_lo"
    w32_disasm_mov_pair[restore_idx] = "s_mov_b64 exec, s[50:51]"
    assert translate(w32_disasm_mov_pair) == translate(base_wave32)

    w32_b32_pair_reject = list(base_wave32)
    w32_b32_pair_reject[save_idx] = "s_mov_b32 s50, exec_lo"
    w32_b32_pair_reject[wqm_idx] = "s_wqm_b64 exec_lo, exec_lo"
    w32_b32_pair_reject[restore_idx] = "s_and_b64 exec, exec, s[50:51]"
    must_reject(translate, w32_b32_pair_reject)

    w32_unaligned = list(base_wave32)
    w32_unaligned[save_idx] = "s_mov_b64 s51, exec_lo"
    must_reject(translate, w32_unaligned)

    w64_odd_pair = list(base_wave32)
    w64_odd_pair[save_idx] = "s_mov_b64 s[49:50], exec"
    must_reject(translate, w64_odd_pair)

    w64_oob_pair = list(base_wave32)
    w64_oob_pair[save_idx] = "s_mov_b64 s[105:106], exec"
    must_reject(translate, w64_oob_pair)

    must_reject(translate, base_wave32[:restore_idx] + ["s_and_b64 exec, exec, s[49:50]"] + base_wave32[restore_idx+1:])
    must_reject(translate, base_wave32[:restore_idx] + ["s_and_b64 exec, exec, s[105:106]"] + base_wave32[restore_idx+1:])

    w32_sym_and = list(base_wave32)
    w32_sym_and[save_idx] = "s_mov_b64 s50, exec_lo"
    w32_sym_and[wqm_idx] = "s_wqm_b64 exec_lo, exec_lo"
    w32_sym_and[restore_idx] = "s_and_b64 exec, exec, s50"
    assert translate(w32_sym_and) == translate(base_wave32)

    w32_sym_mov = list(base_wave32)
    w32_sym_mov[save_idx] = "s_mov_b64 s50, exec_lo"
    w32_sym_mov[wqm_idx] = "s_wqm_b64 exec_lo, exec_lo"
    w32_sym_mov[restore_idx] = "s_mov_b64 exec, s50"
    assert translate(w32_sym_mov) == translate(base_wave32)

    must_reject(translate, w32_and[:restore_idx] + ["s_and_b64 exec_lo, exec_lo, s15"] + w32_and[restore_idx+1:])
    must_reject(translate, w32_mov[:restore_idx] + ["s_mov_b64 exec_lo, s15"] + w32_mov[restore_idx+1:])
    must_reject(translate, ["s_mov_b64 s14, exec_lo", "s_and_b64 exec_lo, exec_lo, s14"] + w32_and[wqm_idx:])
    must_reject(translate, ["s_mov_b64 s14, exec_lo", "s_mov_b64 exec_lo, s14"] + w32_and[wqm_idx:])
    must_reject(translate, w32_and[:restore_idx+1] + ["s_and_b64 exec_lo, exec_lo, s14"] + w32_and[restore_idx+1:])
    must_reject(translate, w32_and[:wqm_idx+1] + ["s_wqm_b64 exec_lo, exec_lo"] + w32_and[wqm_idx+1:])
    prefix_scc = ["s_mov_b32 s90, 0", "s_mov_b32 s91, 0", "s_not_b64 s[92:93], 0"]
    must_reject(translate, w32_and[:wqm_idx] + prefix_scc + [w32_and[wqm_idx], "s_cmov_b64 s[90:91], s[92:93]"] + w32_and[wqm_idx+1:])
    must_reject(translate, w32_and[:restore_idx] + prefix_scc + [w32_and[restore_idx], "s_cmov_b64 s[90:91], s[92:93]"] + w32_and[restore_idx+1:])
    must_reject(translate, w32_mov[:restore_idx] + prefix_scc + [w32_mov[restore_idx], "s_cmov_b64 s[90:91], s[92:93]"] + w32_mov[restore_idx+1:])

    # Full-width restore requires both saved halves intact; clobbering the high SGPR must be rejected.
    clobbered_mov = list(base_wave32)
    clobbered_mov[save_idx] = "s_mov_b64 s50, exec_lo"
    clobbered_mov[wqm_idx] = "s_wqm_b64 exec_lo, exec_lo"
    clobbered_mov.insert(restore_idx, "s_mov_b32 s51, 0")
    clobbered_mov[restore_idx + 1] = "s_mov_b64 exec, s50"
    must_reject(translate, clobbered_mov)

    clobbered_and = list(base_wave32)
    clobbered_and[save_idx] = "s_mov_b64 s50, exec_lo"
    clobbered_and[wqm_idx] = "s_wqm_b64 exec_lo, exec_lo"
    clobbered_and.insert(restore_idx, "s_mov_b32 s51, 0")
    clobbered_and[restore_idx + 1] = "s_and_b64 exec, exec, s50"
    must_reject(translate, clobbered_and)

    clobbered_pair = list(base_wave32)
    clobbered_pair.insert(restore_idx, "s_mov_b32 s51, 0")
    must_reject(translate, clobbered_pair)

    # A b64 expansion followed by a low-only restore without wave32 precondition leaves the upper half unrestored.
    b64_low_restore_and = list(base_wave32)
    b64_low_restore_and[restore_idx] = "s_and_b32 exec_lo, exec_lo, s50"
    must_reject(translate, b64_low_restore_and)

    b64_low_restore_mov = list(base_wave32)
    b64_low_restore_mov[restore_idx] = "s_mov_b32 exec_lo, s50"
    must_reject(translate, b64_low_restore_mov)

    b64_low_save_and = list(base_wave32)
    b64_low_save_and[save_idx] = "s_mov_b32 s50, exec_lo"
    b64_low_save_and[restore_idx] = "s_and_b32 exec_lo, exec_lo, s50"
    must_reject(translate, b64_low_save_and)

    b32_high_restore = list(base_wave32)
    b32_high_restore[save_idx] = "s_mov_b32 s14, exec_lo"
    b32_high_restore[wqm_idx] = "s_wqm_b32 exec_lo, exec_lo"
    b32_high_restore[restore_idx] = "s_and_b64 exec, exec, s14"
    must_reject(translate, b32_high_restore)

    # 32-bit integer comparisons: signed vs unsigned casts, wave64 spelling, inline constants, and rejections.
    i32_source = translate(int_predicate_group("i32", "e32", 0))
    u32_source = translate(int_predicate_group("u32", "e64", 4))
    assert "asint(params[0].w) < asint(alu20)" in i32_source, i32_source
    assert "asuint(params[0].w) >= asuint(alu20)" in u32_source, u32_source
    assert translate(int_predicate_group("i32", "e32", 0, wave64=True)) == i32_source
    assert translate(int_predicate_group("u32", "e64", 4, wave64=True)) == u32_source
    for kind, token, expected in (
        ("i32", "-1", "asint(4294967295u)"),
        ("u32", "-1", "4294967295u"),
        ("i32", "0x80000000", "asint(2147483648u)"),
        ("u32", "0x80000000", "2147483648u"),
        ("i32", "1.0", "asint(1065353216u)"),
        ("u32", "1.0", "1065353216u"),
    ):
        prog = with_constants(["v_mov_b32_e32 v18, s45",
                               f"v_cmp_lt_{kind}_e32 vcc_lo, {token}, v18",
                               "v_cndmask_b32_e32 v12, 0, 1.0, vcc_lo"])
        assert f"bool cmp21 = {expected} < " in translate(prog), (kind, token)
    base_int = int_predicate_group("i32", "e64", 0)
    for bad in (
        "v_cmp_lt_i32_e64 s52, -s44, v18",
        "v_cmp_lt_i32_e64 s52, |s44|, v18",
        "v_cmp_lt_i32_e64 s52, abs(s44), v18",
        "v_cmp_lt_i32_e64 s52, s44, -v18",
        "v_cmp_lt_i32_e64 s52, s44, v18 clamp",
        "v_cmpx_lt_i32_e64 s52, s44, v18",
        "v_cmp_lt_i32_e64 s52, 0x100000000, v18",
    ):
        changed = [line.replace("v_cmp_lt_i32_e64 s52, s44, v18", bad) for line in base_int]
        assert changed != base_int
        must_reject(translate, changed)
    pos = base_int.index("s_waitcnt vmcnt(0)")
    must_reject(translate, base_int[:pos] + ["v_cmp_eq_u32_e32 vcc_lo, v12, v13"] + base_int[pos:])
    load = next(i for i, line in enumerate(base_int) if line.startswith("s_buffer_load_dwordx4"))
    pos = base_int.index("s_waitcnt lgkmcnt(0)", load)
    must_reject(translate, base_int[:pos] + ["v_cmp_eq_i32_e32 vcc_lo, s44, 0"] + base_int[pos:])
    pos = base_int.index("s_wqm_b64 exec, exec") + 1
    must_reject(translate, base_int[:pos] + ["v_cmp_eq_i32_e32 vcc_lo, 0, 0"] + base_int[pos:])
    # Every 32-bit integer comparison form is lowered here too, so tools/form_coverage.py counts all of them.
    for kind in ("i32", "u32"):
        for encoding in ("e32", "e64"):
            for first in (0, 4):
                translate(int_predicate_group(kind, encoding, first))

    # 16-bit integer comparisons: low-half extension, operands, wave64 spelling and rejected neighbours.
    i16_source = translate(int16_group("i16", "e32", 1))
    u16_source = translate(int16_group("u16", "e64", 1))
    assert "(asint(asuint(params[0].w) << 16) >> 16) != (asint(asuint(alu20) << 16) >> 16)" in i16_source, i16_source
    assert "(asint(4294967280u << 16) >> 16) < (asint(asuint(alu20) << 16) >> 16)" in i16_source, i16_source
    assert "(asint(32768u << 16) >> 16) > " in i16_source, i16_source
    assert "(asuint(alu20) & 0xffffu) >= (asuint(params[0].w) & 0xffffu)" in u16_source, u16_source
    assert "(asuint(alu20) & 0xffffu) < (64u & 0xffffu)" in u16_source, u16_source
    assert "(asuint(alu20) & 0xffffu) <= (32768u & 0xffffu)" in u16_source, u16_source
    for kind in ("i16", "u16"):
        for encoding in ("e32", "e64"):
            for group in (0, 1):
                assert (translate(int16_group(kind, encoding, group, wave64=True))
                        == translate(int16_group(kind, encoding, group)))
    for kind, signed in (("i16", True), ("u16", False)):
        for left, right in INT16_PAIRS:
            low = [w & 0xffff for w in (left, right)]
            if signed:
                low = [v - 0x10000 if v & 0x8000 else v for v in low]
            expected = [float(low[0] < low[1]), float(low[0] == low[1]),
                        float(low[0] <= low[1]), float(low[0] > low[1])]
            assert int16_reference(kind, "e32", 0, left, right) == expected
    left, right = INT16_PAIRS[0]
    assert int16_reference("i16", "e32", 0, left, right) != int16_reference("u16", "e32", 0, left, right)
    base16 = int16_group("i16", "e64", 0)
    target = "v_cmp_lt_i16_e64 s52, s44, v18"
    for bad in (
        "v_cmp_lt_i16_e64 s52, -s44, v18",
        "v_cmp_lt_i16_e64 s52, |s44|, v18",
        "v_cmp_lt_i16_e64 s52, s44, v18 clamp",
        "v_cmp_lt_i16_e64 s52, s44, v18 op_sel:[1,0,0]",
        "v_cmp_lt_i16_e64 s52, 1.0, v18",
        "v_cmp_lt_i16_e64 s52, 65, v18",
        "v_cmp_lt_i16_e64 s52, -17, v18",
        "v_cmp_lt_i16_e64 s52, 0x10000, v18",
        "v_cmp_f_i16_e64 s52, s44, v18",
        "v_cmp_t_u16_e64 s52, s44, v18",
        "v_cmpx_lt_i16_e64 s52, s44, v18",
        "v_cmpx_lt_i16_e32 s44, v18",
        "v_cmpx_lt_i32_e32 s44, v18",
        "v_cmp_lt_i64_e64 s52, s[45:46], v[18:19]",
        "v_cmp_lt_i64_e32 vcc_lo, v[18:19], s[44:45]",
        "v_cmp_class_f64_e32 vcc_lo, s[44:45], v18",
        "v_cmp_lt_f64_e64 s52, s[44:45], v[18:19]",
    ):
        changed = [line.replace(target, bad) for line in base16]
        assert changed != base16
        must_reject(translate, changed)
    pos = base16.index("s_waitcnt vmcnt(0)")
    must_reject(translate, base16[:pos] + ["v_cmp_eq_u16_e32 vcc_lo, v12, v13"] + base16[pos:])
    load = next(i for i, line in enumerate(base16) if line.startswith("s_buffer_load_dwordx4"))
    pos = base16.index("s_waitcnt lgkmcnt(0)", load)
    must_reject(translate, base16[:pos] + ["v_cmp_eq_i16_e32 vcc_lo, s44, 0"] + base16[pos:])
    pos = base16.index("s_wqm_b64 exec, exec") + 1
    must_reject(translate, base16[:pos] + ["v_cmp_eq_i16_e32 vcc_lo, 0, 0"] + base16[pos:])

    check_cmp64_controls()

    f16_hlsl = translate_int(uncompressed_export(f16_compares()))
    assert "uint class16(uint bits)" in f16_hlsl and "float widen16(uint word)" in f16_hlsl
    for words in F16_COMPARE_SETS:
        expected_words = f16_compares_reference(words)
        assert all((word & 0x7f800000) == 0 for word in expected_words), expected_words
        assert expected_words != f16_compares_reference((words[0] ^ 0x8000, words[1], words[2] ^ 0x8000, words[3] ^ 0x3ff))
    f16_program = uncompressed_export(f16_compares())
    for bad in (
        "v_cmp_lt_f16_e32 vcc_lo, -s44, v18",
        "v_cmp_lt_f16_e32 vcc_lo, |s44|, v18",
        "v_cmp_lt_f16_e32 vcc_lo, abs(s44), v18",
        "v_cmp_class_f16_e32 vcc_lo, -s44, v18",
        "v_cmp_class_f16_e32 vcc_lo, |s44|, v18",
        "v_cmp_class_f16_e32 vcc_lo, abs(s44), v18",
        "v_cmp_lt_f16_e32 vcc_lo, 0x10000, v18",
        "v_cmp_lt_f16_e64 s52, 0x10000, v18",
        "v_cmp_class_f16_e32 vcc_lo, 0x10000, v18",
        "v_cmp_lt_f16_e64 s52, s44, v18 clamp",
        "v_cmp_class_f16_e64 s52, s44, v18 clamp",
        "v_cmp_lt_f16_e64 s52, s44, v18 op_sel:[1,0,0]",
        "v_cmpx_lt_f16_e32 s44, v18",
        "v_cmpx_lt_f16_e64 s52, s44, v18",
        "v_cmpx_class_f16_e32 s44, v18",
        "v_cmpx_class_f16_e64 s52, s44, v18",
        "v_cmp_class_f16_e64 s52, s44, -s45",
        "v_cmp_class_f16_e64 s52, s44, |s45|",
        "v_cmp_class_f16_e64 s52, s44, 1.0",
    ):
        must_reject(translate_int, uncompressed_export(["v_mov_b32_e32 v18, s45", bad]))
    pos = f16_program.index("s_waitcnt vmcnt(0)")
    must_reject(translate_int, f16_program[:pos] + ["v_cmp_eq_f16_e32 vcc_lo, v12, v13"] + f16_program[pos:])
    load = next(i for i, line in enumerate(f16_program) if line.startswith("s_buffer_load_dwordx4"))
    pos = f16_program.index("s_waitcnt lgkmcnt(0)", load)
    must_reject(translate_int, f16_program[:pos] + ["v_cmp_eq_f16_e32 vcc_lo, s44, v16"] + f16_program[pos:])
    must_reject(translate_int, f16_program[:pos] + ["v_cmp_class_f16_e32 vcc_lo, s44, v16"] + f16_program[pos:])

    f32_hlsl = translate_int(uncompressed_export(f32_compares()))
    assert "uint4 main(" in f32_hlsl
    for words in F32_COMPARE_SETS:
        expected_words = f32_compares_reference(words)
        assert expected_words != f32_compares_reference((words[0], words[1] ^ 0x3f800000, words[2], words[3]))
    f32_program = uncompressed_export(f32_compares())
    for bad in (
        "v_cmp_lt_f32_e64 s52, s44, v18 clamp",
        "v_cmp_lt_f32_e64 s52, s44, v18 op_sel:[1,0,0]",
        "v_cmpx_lt_f32_e32 s44, v18",
        "v_cmpx_lt_f32_e64 s52, s44, v18",
        "v_cmp_lt_f32_sdwa vcc_lo, s44, v18 src0_sel:WORD_0 src1_sel:DWORD",
        "v_cmp_lt_f32_sdwa vcc_lo, s44, v18 src0_sel:DWORD src1_sel:BYTE_2",
        "v_cmp_lt_f32_e32 vcc_lo, s44, v255",
    ):
        must_reject(translate_int, uncompressed_export(["v_mov_b32_e32 v18, s45", bad]))
    pos = f32_program.index("s_waitcnt vmcnt(0)")
    must_reject(translate_int, f32_program[:pos] + ["v_cmp_eq_f32_e32 vcc_lo, v12, v13"] + f32_program[pos:])
    load = next(i for i, line in enumerate(f32_program) if line.startswith("s_buffer_load_dwordx4"))
    pos = f32_program.index("s_waitcnt lgkmcnt(0)", load)
    must_reject(translate_int, f32_program[:pos] + ["v_cmp_eq_f32_e32 vcc_lo, s44, v16"] + f32_program[pos:])

    # 32-bit integer min/max/min3/max3/med3 controls and rejections.
    pair_src = translate(int_minmax_pair("e32"))
    assert "asfloat(min(asint(params[0].w), asint(alu20)))" in pair_src, pair_src
    assert "asfloat(max(asuint(params[0].w), asuint(alu20)))" in pair_src, pair_src
    assert translate(int_minmax_pair("e64")) == pair_src
    tri_i32 = translate(int_minmax_triple("i32"))
    tri_u32 = translate(int_minmax_triple("u32"))
    assert "asfloat(min(min(asint(params[0].w), asint(alu20)), asint(alu21)))" in tri_i32, tri_i32
    assert tri_i32.count("int3 med") == 2 and tri_u32.count("uint3 med") == 2
    for bad_pair in (
        "v_min_i32_e64 v12, -s44, v18",
        "v_min_i32_e64 v12, |s44|, v18",
        "v_min_i32_e64 v12, s44, v18 clamp",
        "v_min_i32_e64 v12, s44",
        "v_min_i32_e64 v12, 0x100000000, v18",
    ):
        changed = [line.replace("v_min_i32_e64 v12, s44, v18", bad_pair) for line in int_minmax_pair("e64")]
        assert changed != int_minmax_pair("e64")
        must_reject(translate, changed)
    for bad_tri in (
        "v_med3_i32 v14, -s44, v18, v19",
        "v_med3_i32 v14, s44, |v18|, v19",
        "v_med3_i32 v14, s44, v18, v19 clamp",
        "v_med3_i32 v14, s44, v18",
    ):
        changed = [line.replace("v_med3_i32 v14, s44, v18, v19", bad_tri) for line in int_minmax_triple("i32")]
        assert changed != int_minmax_triple("i32")
        must_reject(translate, changed)
    base_pair = int_minmax_pair("e32")
    wait = base_pair.index("s_waitcnt vmcnt(0)")
    must_reject(translate, base_pair[:wait] + ["v_min_i32_e32 v12, 0, 1"] + base_pair[wait:])
    must_reject(translate, base_pair[:wait] + ["v_med3_u32 v13, 0, 1, 2"] + base_pair[wait:])
    load = next(i for i, line in enumerate(base_pair) if line.startswith("s_buffer_load_dwordx4"))
    lgkm = base_pair.index("s_waitcnt lgkmcnt(0)", load)
    must_reject(translate, base_pair[:lgkm] + ["v_max_u32_e32 v20, s44, 0"] + base_pair[lgkm:])
    wqm = base_pair.index("s_wqm_b64 exec, exec") + 1
    must_reject(translate, base_pair[:wqm] + ["v_min_u32_e32 v20, 0, 1"] + base_pair[wqm:])

    # v_frexp_mant_f32 and v_ldexp_f32 controls and rejections.
    frexp_body = translate(frexp_mant_values()).split("float4 main", 1)[1]
    assert "frexp_mant32(params[0].w)" in frexp_body, frexp_body
    assert "frexp_mant32((-abs(params[1].x)))" in frexp_body, frexp_body
    assert "saturate(frexp_mant32(params[1].z))" in frexp_body, frexp_body
    frexp_exp_body = translate(frexp_exp_values()).split("float4 main", 1)[1]
    assert "asfloat(frexp_exp32(params[0].w))" in frexp_exp_body, frexp_exp_body
    assert "asfloat(frexp_exp32(params[1].y))" in frexp_exp_body, frexp_exp_body
    must_reject(translate, [
        line + " clamp" if line.startswith("v_frexp_exp_i32_f32_e64 ") else line
        for line in frexp_exp_values()
    ])
    ldexp_body = translate(ldexp_values()).split("float4 main", 1)[1]
    assert "ldexp32(alu20, asint(params[1].x))" in ldexp_body, ldexp_body
    assert "ldexp32((-abs(params[0].w)), asint(2u))" in ldexp_body, ldexp_body
    assert "ldexp32(params[1].y, asint(4294967294u))" in ldexp_body, ldexp_body
    assert "saturate(ldexp32(params[1].z, asint(3u)))" in ldexp_body, ldexp_body
    for bad in (
        "v_ldexp_f32 v12, v18, -s45",
        "v_ldexp_f32 v12, v18, |s45|",
        "v_ldexp_f32 v12, v18, abs(s45)",
        "v_ldexp_f32 v12, v18",
        "v_ldexp_f32 v12, v18, 0x100000000",
    ):
        changed = [line.replace("v_ldexp_f32 v12, v18, s45", bad) for line in ldexp_values()]
        assert changed != ldexp_values()
        must_reject(translate, changed)
    base_ld = ldexp_values()
    wait = base_ld.index("s_waitcnt vmcnt(0)")
    must_reject(translate, base_ld[:wait] + ["v_frexp_mant_f32_e32 v12, 1.0"] + base_ld[wait:])
    must_reject(translate, base_ld[:wait] + ["v_ldexp_f32 v13, 1.0, 2"] + base_ld[wait:])
    load = next(i for i, line in enumerate(base_ld) if line.startswith("s_buffer_load_dwordx4"))
    lgkm = base_ld.index("s_waitcnt lgkmcnt(0)", load)
    must_reject(translate, base_ld[:lgkm] + ["v_ldexp_f32 v20, 1.0, s45"] + base_ld[lgkm:])
    wqm = base_ld.index("s_wqm_b64 exec, exec") + 1
    must_reject(translate, base_ld[:wqm] + ["v_ldexp_f32 v20, 1.0, 2"] + base_ld[wqm:])

    # v_mul_{i32_i24,u32_u24,hi_i32_i24,hi_u32_u24} controls and rejections.
    mul24_hlsl = translate(mul24_values())
    for helper_sig in (
        "uint mul24_u32(uint a, uint b)",
        "uint mul24_i32(uint a, uint b)",
        "uint mul24_hi_u32(uint a, uint b)",
        "uint mul24_hi_i32(uint a, uint b)",
    ):
        assert helper_sig in mul24_hlsl, helper_sig
    assert mul24_hlsl.index("uint mul24_hi_u32") < mul24_hlsl.index("uint mul24_hi_i32")
    # Using only v_mul_hi_i32_i24 must still pull in its mul24_hi_u32 dependency before it.
    hi_only = translate(with_constants(["v_mul_hi_i32_i24_e64 v12, s44, s45", "v_mov_b32_e32 v13, 0",
                                        "v_mov_b32_e32 v14, 0", "v_mov_b32_e32 v15, 0"]))
    assert hi_only.index("uint mul24_hi_u32") < hi_only.index("uint mul24_hi_i32")
    assert "mul24_u32(" not in hi_only and "mul24_i32(" not in hi_only
    base_hlsl = translate(assembly(False))
    assert "mul24_" not in base_hlsl
    mul24_main = mul24_hlsl.split("float4 main", 1)[1]
    for snippet in (
        "asfloat(mul24_i32(asuint(params[0].w), asuint(alu",
        "asfloat(mul24_i32(asuint(params[1].y), asuint(params[1].z)))",
        "asfloat(mul24_u32(asuint(params[1].x), asuint(alu",
        "asfloat(mul24_u32(asuint(params[1].z), asuint(params[0].w)))",
        "asfloat(mul24_hi_i32(asuint(params[0].w), asuint(alu",
        "asfloat(mul24_hi_i32(asuint(params[1].y), asuint(params[1].z)))",
        "asfloat(mul24_hi_u32(asuint(params[1].x), asuint(alu",
        "asfloat(mul24_hi_u32(asuint(params[1].z), asuint(params[0].w)))",
    ):
        assert snippet in mul24_main, snippet
    for words in MUL24_SETS:
        expected = mul24_reference(words)
        for ch in expected:
            assert half_rtz(ch) == ch, (words, expected)
    for mode in ("no_mask_24", "i24_as_u24", "u24_as_i24", "hi_i24_as_u24", "hi_i24_bad_lo0", "hi_u24_as_i24"):
        assert any(mul24_reference(w, mode=mode) != mul24_reference(w) for w in MUL24_SETS), mode
    base_m24 = mul24_values()
    for bad in (
        "v_mul_i32_i24_e64 v21, -s46, s47",
        "v_mul_i32_i24_e64 v21, |s46|, s47",
        "v_mul_i32_i24_e64 v21, abs(s46), s47",
        "v_mul_i32_i24_e64 v21, s46, s47 clamp",
        "v_mul_u32_u24_e64 v23, s47, s44 clamp",
        "v_mul_hi_i32_i24_e64 v25, s46, s47 clamp",
        "v_mul_hi_u32_u24_e64 v27, s47, s44 clamp",
        "v_mul_i32_i24_sdwa v21, s46, s47 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
        "v_mul_i32_i24_e64 v21, s46",
        "v_mul_u32_u24_e32 v256, s45, v29",
        "v_mul_hi_u32_u24_e64 v27, s47, 0x100000000",
    ):
        changed = [line.replace("v_mul_i32_i24_e64 v21, s46, s47", bad) for line in base_m24]
        assert changed != base_m24, bad
        must_reject(translate, changed)
    wait = base_m24.index("s_waitcnt vmcnt(0)")
    must_reject(translate, base_m24[:wait] + ["v_mul_i32_i24_e32 v12, 1, v28"] + base_m24[wait:])
    must_reject(translate, base_m24[:wait] + ["v_mul_hi_u32_u24_e64 v13, 1, 2"] + base_m24[wait:])
    load = next(i for i, line in enumerate(base_m24) if line.startswith("s_buffer_load_dwordx4"))
    lgkm = base_m24.index("s_waitcnt lgkmcnt(0)", load)
    must_reject(translate, base_m24[:lgkm] + ["v_mul_u32_u24_e64 v20, s44, 2"] + base_m24[lgkm:])
    wqm = base_m24.index("s_wqm_b64 exec, exec") + 1
    must_reject(translate, base_m24[:wqm] + ["v_mul_hi_i32_i24_e64 v20, 1, 2"] + base_m24[wqm:])
    check_int16_vop3_controls()
    check_int16_ternary_controls()
    check_f16_arith_controls()


def check_program_extent(directory, library):
    # A conditional jumps over the early exit. The alternate block starts
    # with a two-dword instruction (s_mov_b32 with a literal operand).
    branch = bytes.fromhex("010084bf")
    end = bytes.fromhex("000081bf")
    mov = bytes.fromhex("ff0380be 01000000")
    pad = bytes.fromhex("00009fbf")
    code = branch + end + mov + end
    padding = pad * 5
    base = 288 - 40
    extent = 40 + len(code) + len(padding)
    archive = directory / "bounded.rpf"

    def extract(body, size=extent, limit=base + extent):
        archive.write_bytes(envelope(body, directory_end=(size, limit)))
        return native_program(archive, "fixture", llvm=library)[1]

    jump_over_nop = bytes.fromhex("010082bf 000080bf")
    assert extract(jump_over_nop + code + padding[8:]) == jump_over_nop + code
    must_reject(extract, bytes.fromhex("010082bf 810380be") + code + padding[8:])
    assert extract(code + padding + end) == code  # marker in metadata
    # A terminator in metadata cannot stand in for one missing inside the section.
    must_reject(extract, bytes(len(code)) + padding + end)
    assert extract(code + padding + pad * 4, limit=0xdeadbeefdeadbeef) == code
    must_reject(extract, code + padding, extent, base + extent + 4)
    must_reject(extract, code + padding[:-4])
    must_reject(extract, code + padding, extent, 0xdeadbeefdeadbeef)
    must_reject(extract, code + padding[:4] + bytes(16) + end)
    must_reject(extract, bytes.fromhex("020084bf") + code[4:] + padding)  # target inside literal
    must_reject(extract, bytes.fromhex("000080bf") + code[4:] + padding)  # disconnected block
    must_reject(extract, bytes.fromhex("040084bf") + code[4:] + padding)  # target in padding
    must_reject(extract, branch + bytes.fromhex("ff0380be") + end + bytes.fromhex("000080bf") + end + padding)


def check_f16_literal_fold(directory, assembler, objcopy, library):
    """LLVM 15 prints a 16-bit literal that equals an integer inline constant as that integer, so 0xffff (a NaN) reads as
    -1 and 0x0001 (a subnormal) as 1. The plain binary16 VOP2 lowering refuses every integer except 0 for this reason."""
    source, obj, binary = directory / "fold.s", directory / "fold.o", directory / "fold.bin"
    source.write_text(chr(10).join(("v_add_f16 v1, 0xffff, v3", "v_add_f16 v2, 0x0001, v3", "v_add_f16 v4, 0x0041, v3",
                                 "v_add_f16 v5, 0x0000, v3", "v_add_f16 v6, 0x3c00, v3", "s_endpgm", "")), encoding="ascii")
    subprocess.run([assembler, "-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-filetype=obj", source, "-o", obj], check=True)
    subprocess.run([objcopy, "-O", "binary", "--only-section=.text", obj, binary], check=True)
    assert disassemble(binary.read_bytes(), library)[:5] == [
        "v_add_f16_e32 v1, -1, v3", "v_add_f16_e32 v2, 1, v3", "v_add_f16_e32 v4, 0x41, v3",
        "v_add_f16_e32 v5, 0, v3", "v_add_f16_e32 v6, 1.0, v3"]


def check_binary(directory, assembler, objcopy, library):
    check_program_extent(directory, library)
    check_f16_literal_fold(directory, assembler, objcopy, library)
    fixtures = [(assembly(False), translate), (assembly(True), translate), (curve(), translate), (shaping(), translate), (select(), translate), (absolute_value(), translate),
                (rounding(), translate), (arithmetic(), translate), (with_scalar_constant_reads(), translate), (with_scalar_pair_reads(), translate),
                (with_three_constant_buffers(), translate_three_buffers),
                (truncate_values(), translate), (affine_coordinates(), translate), (ceiling(), translate),
                (scalar64(), translate), (sopk_scalar(), translate), (sopk_cmpk_cmovk(), translate_int)]
    fixtures.extend((align_integer_group(byte), translate_int) for byte in (False, True))
    fixtures.extend((vop3_ternary_int_group(group), translate_int) for group in (1, 2))
    fixtures.append((readfirstlane_program(), translate_int))
    fixtures.append((uncompressed_export(["v_mul_lo_u32 v12, s44, s45"]), translate_int))
    fixtures.append((rgb_sample(), translate))
    fixtures.append((with_constants(TRIG_BODY), translate))
    fixtures.append((with_constants(CLASS_BODY), translate))
    fixtures.append((sext_values(), translate))
    fixtures.append((sop1_scalar_ops(), translate_int))
    fixtures.append((bfm_scalar_ops(), translate_int))
    fixtures.append((brev_scalar_ops(), translate_int))
    fixtures.extend((plain_bitscan_ops(second), translate_int) for second in (False, True))
    fixtures.append((swap_ops(), translate_int))
    fixtures.extend((quadmask_bitreplicate_program(op), translate_int)
                    for op in QUADMASK_BITREPLICATE_INSTRUCTIONS)
    fixtures.extend((int16_vop3_ops(group), translate_int) for group in range(4))
    fixtures.extend((int16_ternary_ops(group), translate_int) for group in range(5))
    fixtures.extend((f16_arith_program(*key), translate_f16) for key in F16_ARITH_PROGRAMS)
    fixtures.extend((scalar_bitset64(op, str(index)), translate_int)
                    for op in SCALAR_BITSET64_OPS for index in (0, 31, 32, 63, 64))
    fixtures.append((scalar_select64(), translate_int))
    for op in SCALAR_BFE64_OPS:
        fixtures.extend((program, translate_int) for program in (
            scalar_bfe64(op), scalar_bfe64(op, "0x80000000"), scalar_bfe64(op, control="0x400000"),
        ))
    fixtures.extend((scalar_pack_values(halves), translate) for halves in PACK_HALVES)
    fixtures.extend((scalar_bit_count_values(width), translate) for width in (32, 64))
    fixtures.extend((scalar_first_bit_values(width), translate) for width in (32, 64))
    fixtures.extend((scalar_carry_chain(subtract), translate_int) for subtract in (False, True))
    fixtures.extend((scalar_leading_bit_values(width), translate) for width in (32, 64))
    fixtures.append((scalar_mul_hi_values(), translate))
    fixtures.append((with_constants(RECIPROCAL_BODY), translate))
    fixtures.append((bitwise_program(), translate))
    fixtures.append((fract_values(), translate))
    fixtures.extend((program(), translate) for program in (frexp_mant_values, frexp_exp_values, ldexp_values))
    fixtures.append((mul24_values(), translate))
    fixtures.append((three_way(), translate))
    fixtures.extend((predicate_group(first), translate) for first in range(0, 16, 4))
    fixtures.extend((int_predicate_group(kind, encoding, first), translate)
                    for kind in ("i32", "u32") for encoding in ("e32", "e64") for first in (0, 4))
    fixtures.append((unsigned_less_than(), translate))
    fixtures.extend((int16_group(kind, encoding, group), translate)
                    for kind in ("i16", "u16") for encoding in ("e32", "e64") for group in (0, 1))
    fixtures.extend((cmp64_group(kind, encoding, group), translate_int)
                    for kind in ("i64", "u64") for encoding in ("e32", "e64") for group in (0, 1))

    fixtures.extend((int_minmax_pair(encoding), translate) for encoding in ("e32", "e64"))
    fixtures.extend((int_minmax_triple(kind), translate) for kind in ("i32", "u32"))
    fixtures.extend((predicate_group(first, sdwa=True), translate) for first in range(0, 16, 4))
    fixtures.extend((cvt_int_fixture(encoding), translate) for encoding in ("e32", "e64"))
    fixtures.append((cvt_int_literals(), translate))
    fixtures.append((with_constants(CUBE_BODY), translate))
    fixtures.append((with_constants(MAD_BODY), translate))
    fixtures.extend((dual_program(case), translate_dual) for case in DUAL_CASES)
    fixtures.append((with_constants(MOVK_BODY), translate))
    fixtures.append((with_constants(HALF_BODY), translate))
    fixtures.append((with_constants(VOP3P_BODY), translate))
    fixtures.append((with_constants(FMA_MIX_BODY), translate))
    fixtures.append((with_constants(NARROW_BODY), translate))
    fixtures.append((mode_program(), translate))
    fixtures.append((dual_program(swap_weights=True), translate_dual))
    fixtures.extend((with_constants(body), translate) for body, *_ in E64_ARITH)
    fixtures.append((with_constants(APPROX_E64_BODY), translate))
    fixtures.extend((predicate_group_e64(first), translate) for first in range(0, 16, 4))
    fixtures.append((scalar(), translate))
    fixtures.append((lod_zero_sample(), translate))
    fixtures.append((with_constants(LITERAL_FMA_BODY), translate))
    fixtures.append((with_constants(MAD_LITERAL_BODY), translate))
    fixtures.append((unpack_bytes(), translate))
    fixtures.append((reciprocal_iflag(), translate))
    fixtures.append((offset_nibbles(), translate))
    fixtures.append((legacy_multiply_add(), translate))
    fixtures.append((uncompressed_export(MAC_LEGACY_BODY), translate))
    fixtures.append((scheduling_hints(), translate))
    fixtures.extend((buffer_color(rg), translate_buffer_color) for rg in (False, True))
    fixtures.extend((unsigned_conversion(quad), translate_buffer_color) for quad in (False, True))
    fixtures.extend((attribute_color(rg), translate_attribute) for rg in (False, True))
    fixtures.extend((bit_program(op), translate) for op in BIT_OPERATIONS)
    fixtures.extend((program, function) for _, program, function, _ in e64_spelling_programs())
    fixtures.append((with_constants(DPP_INT32_BODY), translate))
    fixtures.append((with_constants(DPP_BODY), translate))
    fixtures.extend((with_constants(body), translate) for body, *_ in SDWA_ARITH)
    fixtures.extend((with_constants(body), translate) for _, body, *_ in SDWA_UNARY)
    fixtures.append((with_constants(PK_INT16_BODY), translate))
    fixtures.extend((with_constants(body), translate) for body in (SDWA_PRESERVE_BODY, SDWA_PAD_BODY))
    exp_sync_program = ["s_waitcnt_expcnt null, 0"] + assembly(False)[:-1] + ["s_waitcnt_expcnt null, 0", "s_endpgm"]
    fixtures.append((exp_sync_program, translate))
    fixtures.append((sopk_wait_program(), translate))
    fixtures.append((s_version_program(), translate_int))
    fixtures.append((with_constants(DOT_BODY), translate, "gfx1030"))
    fixtures.append((ds_swizzle_program(), translate_ds_swizzle, "gfx1030", "+wavefrontsize64,-wavefrontsize32"))
    fixtures.append((ds_nop_program(), translate_ds_swizzle, "gfx1030", "+wavefrontsize64,-wavefrontsize32"))
    fixtures.append((uncompressed_export(cvt_int_body(0xFFFFFFFF, 0x80000000)), translate))
    fixtures.append((uncompressed_export(cvt_int_body(0xFFFFFFFF, 0x80000000, e64=True)), translate))
    fixtures.append((uncompressed_export(cvt_int_body(0xFFFFFFFF, 0x80000000, e64=True, clamp=True)), translate))
    fixtures.append((uncompressed_export(cvt_float_body(0x3fc00000, 0xc0300000)), translate_int))
    fixtures.extend((uncompressed_export(nc_integer_group(group)), translate_int) for group in (1, 2, 3))
    fixtures.append((title_program(), translate_title))
    fixtures.append((uncompressed_export(CVT_FLOAT_E64_PLAIN_BODY), translate_int))
    fixtures.append((uncompressed_export(CVT_FLOAT_E64_U32_MOD_BODY), translate_int))
    fixtures.append((uncompressed_export(CVT_FLOAT_E64_I32_MOD_BODY), translate_int))
    fixtures.append((uncompressed_export(cvt_flr_rpi_pair_body(0x3effffff, 0xbf000001, e64=False)), translate_int))
    fixtures.append((uncompressed_export(cvt_flr_rpi_pair_body(0xbeffffff, 0x4b000001, e64=True)), translate_int))
    fixtures.append((uncompressed_export(cvt_flr_rpi_mod_body("flr_i32", CVT_FLR_RPI_MOD_SETS[0])), translate_int))
    fixtures.append((uncompressed_export(cvt_flr_rpi_mod_body("rpi_i32", CVT_FLR_RPI_MOD_SETS[1])), translate_int))
    fixtures.append((uncompressed_export(f16_compares()), translate_int))
    fixtures.extend((uncompressed_export(e64_integer_group(group)), translate_int) for group in E64_INT_BODIES)
    fixtures.extend((uncompressed_export(carry_out_group(op)), translate_int) for op in CARRY_OUT_OPS)
    fixtures.extend((uncompressed_export(carry_in_group(op)), translate_int) for op in CARRY_IN_OPS)
    fixtures.extend((uncompressed_export(sdwa_integer_group(group)), translate_int) for group in SDWA_INT_BODIES)
    fixtures.extend((uncompressed_export(pack_group(group)), translate_int) for group in PACK_BODIES)
    fixtures.extend((uncompressed_export(sad_group(group)), translate_int) for group in SAD_BODIES)
    fixtures.extend((compressed_export(mask), translate_masked) for mask in (0x3, 0xC, 0xF))
    fixtures.extend((uncompressed_export([], mask), translate_masked) for mask in (0x1, 0x8, 0x5, 0xA))
    fixtures.append((masked_region(), translate))
    fixtures.append((masked_complementary(), translate))
    fixtures.append((masked_skip_asm(), translate, "gfx1013", "", masked_skip()))
    fixtures.append((uncompressed_export(flat_scratch_ops_group1()), translate_int))
    fixtures.append((uncompressed_export(flat_scratch_ops_group2()), translate_int))
    for index, entry in enumerate(fixtures):
        instructions, lower_fixture = entry[:2]
        cpu = entry[2] if len(entry) > 2 else "gfx1013"
        features = entry[3] if len(entry) > 3 else ""
        reference = entry[4] if len(entry) > 4 else instructions
        source = directory / "fixture.s"
        obj = directory / "fixture.o"
        binary = directory / "fixture.bin"
        source.write_text("\n".join(instructions) + "\n", encoding="ascii")
        subprocess.run([assembler, "-triple=amdgcn--amdpal", f"-mcpu={cpu}",
                        *([f"-mattr={features}"] if features else []),
                        "-filetype=obj", source, "-o", obj], check=True)
        subprocess.run([objcopy, "-O", "binary", "--only-section=.text", obj, binary], check=True)
        archive = directory / "fixture.rpf"
        archive.write_bytes(envelope(binary.read_bytes()))
        _, native = native_program(archive, "fixture")
        assert native == binary.read_bytes()
        offsets = isinstance(reference[0], tuple)
        decoded = disassemble(native, library, cpu=cpu.encode(), offsets=offsets, features=features.encode())
        if "ds_nop" in instructions:
            assert native.startswith(bytes.fromhex("000050d8 00000000"))
            assert decoded.count("ds_nop") == instructions.count("ds_nop")
        assert lower_fixture(decoded) == lower_fixture(reference)
        if index == 0:
            check_container_selector(directory, native, library)
    check_wave32_mask_binary(directory, assembler, objcopy, library)
    check_exp_en_decoding(directory, assembler, objcopy, library)


def check_wave32_mask_binary(directory, assembler, objcopy, library):
    reference = assembly(False)
    save = reference.index("s_mov_b64 s[50:51], exec")
    wqm = reference.index("s_wqm_b64 exec, exec")
    restore = reference.index("s_and_b64 exec, exec, s[50:51]")
    program = list(reference)
    program[save] = "s_mov_b32 s14, exec_lo"
    program[wqm] = "s_wqm_b32 exec_lo, exec_lo"
    source = directory / "wave32_mask.s"
    obj = directory / "wave32_mask.o"
    binary = directory / "wave32_mask.bin"
    features = "+wavefrontsize32,-wavefrontsize64"
    for operation in ("s_and_b32 exec_lo, exec_lo, s14", "s_mov_b32 exec_lo, s14"):
        program[restore] = operation
        source.write_text("\n".join(program) + "\n", encoding="ascii")
        subprocess.run([assembler, "-triple=amdgcn--amdpal", "-mcpu=gfx1013", f"-mattr={features}",
                        "-filetype=obj", source, "-o", obj], check=True)
        subprocess.run([objcopy, "-O", "binary", "--only-section=.text", obj, binary], check=True)
        decoded = disassemble(binary.read_bytes(), library, features=features.encode("ascii"))
        assert lower(decoded, 20, 24, [1, 2, 3, 4], constant_descriptor=64, wave_size=32) == translate(reference)
        must_reject(lambda p: lower(p, 20, 24, [1, 2, 3, 4], constant_descriptor=64, wave_size=64), decoded)


# LLVM 15 disassembles any raw four-bit compressed-export EN value, but reassembling the printed
# text keeps a packed pair only when its first operand is enabled; the duplicated high-half operand
# never enables its pair. Measured on gfx1013 by patching only the EN nibble of the reference word.
# The source VGPR field of a fully disabled pair re-encodes as zero, so those bits are excluded
# from the preservation check instead of being treated as changed semantics.
EXP_EN_REENCODE = (0x0, 0x3, 0x0, 0x3, 0xC, 0xF, 0xC, 0xF,
                   0x0, 0x3, 0x0, 0x3, 0xC, 0xF, 0xC, 0xF)


def check_exp_en_decoding(directory, assembler, objcopy, library):
    """Issue #361: replace only the four EN bits of the assembled reference compressed export and
    decode all 16 raw values through the normal binary -> disassembly -> lowering path."""
    def assembled(text):
        source = directory / "decoding.s"
        obj = directory / "decoding.o"
        binary = directory / "decoding.bin"
        source.write_text(text, encoding="ascii")
        subprocess.run([assembler, "-triple=amdgcn--amdpal", "-mcpu=gfx1013",
                        "-filetype=obj", source, "-o", obj], check=True)
        subprocess.run([objcopy, "-O", "binary", "--only-section=.text", obj, binary], check=True)
        return binary.read_bytes()

    base = assembled("\n".join(compressed_export(0xF)) + "\n")
    word = assembled("exp mrt0 v16, v16, v17, v17 done compr vm\ns_endpgm\n")[:8]
    offset = base.find(word)
    assert offset >= 0 and base.find(word, offset + 1) < 0, "the reference export occurs once"
    base_lines = disassemble(base, library)
    measured = {}
    for en, reencoded in enumerate(EXP_EN_REENCODE):
        patched = bytearray(base)
        patched[offset] = patched[offset] & 0xF0 | en
        lines = disassemble(bytes(patched), library)
        changed = [a != b for a, b in zip(base_lines, lines)]
        assert len(lines) == len(base_lines) and changed.count(True) == (en != 0xF)
        text = next(line for line in lines if line.startswith("exp "))
        operands = text.split("mrt0", 1)[1].lstrip(", ").split(" done compr vm", 1)[0].split(", ")
        # The printed source registers are unchanged: each enabled channel names its packed source
        # VGPR (v16 for channels 0/1, v17 for channels 2/3) and a disabled channel prints off.
        printed = ["off"] * 4
        for channel, register in enumerate(("v16", "v16", "v17", "v17")):
            if en >> channel & 1:
                printed[channel] = register
        assert operands == printed, (hex(en), text)
        reword = assembled(text + "\ns_endpgm\n")[:8]
        measured[en] = reword[0] & 0xF
        assert measured[en] == reencoded, (hex(en), reword.hex(" "))
        # Every bit outside EN matches the reference word except the source VGPR field of a fully
        # disabled pair, which the assembler encodes as zero.
        assert reword[0] & 0xF0 == word[0] & 0xF0, (hex(en), reword.hex(" "))
        assert reword[1:4] == word[1:4] and reword[6:] == word[6:], (hex(en), reword.hex(" "))
        assert reword[4] == (0x10 if reencoded & 0x3 else 0x0), (hex(en), reword.hex(" "))
        assert reword[5] == (0x11 if reencoded & 0xC else 0x0), (hex(en), reword.hex(" "))
        if en in (0x3, 0xC, 0xF):
            # The normalized pair masks lower from the patched word exactly as from the fixture
            # program, disabled-pair source bits and all, and report their enable mask.
            assert translate_masked(lines) == translate_masked(compressed_export(en))
        else:
            # EN=0x0 round-trips byte-wise but enables no pair, and every non-pair mask is
            # explicitly rejected rather than silently re-encoded to its normalized form.
            try:
                translate_masked(lines)
            except ValueError as error:
                assert "whole packed pairs" in str(error), error
            else:
                raise AssertionError(f"lowering accepted raw compressed EN {en:#04x}")
    # The rule is pair-first-operand presence, not an OR over the pair: 0x2 and 0xa enable only
    # the duplicated high-half operand and lose their pair (OR would mispredict 0x3 and 0xF),
    # while 0x5 enables both first operands and keeps both pairs.
    assert measured[0x2] == 0x0 and measured[0xA] == 0x0 and measured[0x5] == 0xF



def check_compressed_export_draw(directory, compiler, renderer, backend):
    return check_float_export_masks(directory, compiler, renderer, backend, (0x3, 0xC, 0xF),
                                    compressed_export, (0.5, 1.5, -2.5, 3.5), half_rtz)


def check_uncompressed_export_draw(directory, compiler, renderer, backend):
    body = [f"v_sub_f32_e64 v{12 + i}, s{44 + i}, 0" for i in range(4)]
    values = tuple(value + 2**-20 for value in (0.5, 1.5, -2.5, 3.5))
    return check_float_export_masks(directory, compiler, renderer, backend, (0x1, 0x8, 0x5, 0xA, 0xF),
                                    lambda mask: uncompressed_export(body, mask), values, float)


def check_float_export_masks(directory, compiler, renderer, backend, masks, program, values, channel_value):
    sentinel = (-1.0, 2.0, -3.0, 4.0)
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, *values)
    source = directory / "masked-export.hlsl"
    shader = directory / "masked-export.cso"
    for requested in masks:
        hlsl, mask = translate_masked(program(requested))
        assert mask == requested
        source.write_text(hlsl, encoding="ascii")
        compile_shader(compiler, source, shader)
        expected = tuple(channel_value(value) if requested & (1 << ch) else sentinel[ch]
                         for ch, value in enumerate(values))
        good = write_case(directory / "masked.case", constants, expected, write_mask=mask)
        cases = [(good, "pass")]
        if requested != 0xF:
            wrong_mask = write_case(directory / "wrong-mask.case", constants, expected, write_mask=0xF)
            wrong_values = tuple(value + 0.25 if mask & (1 << ch) else value
                                 for ch, value in enumerate(expected))
            wrong = write_case(directory / "wrong-value.case", constants, wrong_values, write_mask=mask)
            cases.extend([(wrong_mask, "mismatch"), (wrong, "mismatch"), (good, "pass")])
        run_pixel_batch(renderer, backend, shader, cases)


def check_ceiling_draws(directory, compiler, renderer, backend):
    instructions = ceiling()
    source = directory / "ceiling.hlsl"
    shader = directory / "ceiling.cso"
    source.write_text(translate(instructions), encoding="ascii")
    compile_shader(compiler, source, shader)
    cases = []
    for idx, (a, b, c, d) in enumerate(((-1.75, 1.75, -1.75, -1.75), (-2.0, 2.0, 0.0, 0.0),
                                        (-0.75, 0.75, -0.75, -0.75), (2.25, -2.75, 2.25, 2.25))):
        constants = bytearray(256)
        struct.pack_into("<4f", constants, 12, a, b, c, d)
        # Python's integer results lose the sign of zero. The readback compares float bits.
        expected = (math.copysign(math.ceil(a), a),
                    math.copysign(math.ceil(-abs(b)), -1), math.ceil(abs(c)),
                    min(math.ceil(abs(d)), 1))
        case = directory / f"ceiling-{idx}.case"
        write_case(case, constants, expected)
        cases.append((case, "pass"))
    run_pixel_batch(renderer, backend, shader, cases)

    floor = [line.replace("v_ceil_f32_e32", "v_floor_f32_e32") for line in instructions]
    source.write_text(translate(floor), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, cases[-1][0]],
                           capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Replacing ceiling with floor escaped detection")


def check_affine_lod_draw(directory: Path, compiler: str, renderer: str, backend: str) -> None:
    program = affine_coordinates()
    coordinate = f"float2(alu{program.index('v_fma_f32 v8, 0.5, v8, 0')}, uv.y)"
    source = translate(program)
    assert source.count(coordinate) == 1
    # Replace values on uncovered lanes with the quad's top-left coordinates before computing LOD.
    poor_gradient = (f"(uv.x + uv.y < 0.96875f ? {coordinate} : "
                     "(((floor(uv * 4.0f) * 2.0f + 0.5f) / 8.0f) + float2(0.5f, 0.0f))"
                     " * float2(params[0].z * 0.5f, 1.0f))")
    hlsl = directory / "quad_lod.hlsl"
    shader = directory / "quad_lod.cso"
    draw = [renderer, backend, "--check-pixel-quad-lod", shader]
    hlsl.write_text(source, encoding="ascii")
    compile_shader(compiler, hlsl, shader)
    subprocess.run(draw, check=True)
    hlsl.write_text(source.replace(coordinate, poor_gradient), encoding="ascii")
    compile_shader(compiler, hlsl, shader)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("The draw accepted a missing helper-lane gradient")
# v_min_f32/v_max_f32 on the least binary32 subnormal keep it under -denorm preserve and flush it
# to zero under -denorm ftz; v_mul_f32 by 1.0 folds to a move, so that channel exports identical
# bits under both modes and cannot detect the mode by itself.
DENORM_BODY = [
    "v_min_f32_e32 v12, s44, s45",
    "v_max_f32_e32 v13, s44, s46",
    "v_mul_f32_e32 v14, s44, 1.0",
]
LEAST_SUBNORMAL = struct.unpack("<f", struct.pack("<I", 1))[0]


def check_denorm_mode_draw(directory, compiler, renderer, backend):
    """Draw one program under both -denorm modes with mode-matched references."""
    source = directory / "denorm_mode.hlsl"
    shader = directory / "denorm_mode.cso"
    preserve_case = directory / "denorm_mode-preserve.case"
    ftz_case = directory / "denorm_mode-ftz.case"
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, LEAST_SUBNORMAL, 1.0, -1.0, 0.0)
    source.write_text(translate(uncompressed_export(DENORM_BODY)), encoding="ascii")
    passthrough = 1.0  # the untouched sample alpha
    preserve = (LEAST_SUBNORMAL, LEAST_SUBNORMAL, LEAST_SUBNORMAL, passthrough)
    ftz = (flush_ftz(LEAST_SUBNORMAL), flush_ftz(LEAST_SUBNORMAL), LEAST_SUBNORMAL, passthrough)
    write_case(preserve_case, constants, preserve)
    write_case(ftz_case, constants, ftz)
    # The min and max channels differ between the modes, so the preserve build must fail the ftz reference.
    compile_shader(compiler, source, shader)
    run_pixel_batch(renderer, backend, shader, [(ftz_case, "mismatch"), (preserve_case, "pass")])
    # The folding channel expects the same raw subnormal under ftz, so this run passing proves the
    # folded move exports identical bits in both modes; the min and max channels expect the flush.
    compile_shader(compiler, source, shader, mode="ftz")
    subprocess.run([renderer, backend, "--check-pixel-case", shader, ftz_case], check=True)


# Packed binary16 operations on subnormal halves. s44 holds the least (0x0001) and largest (0x03ff)
# binary16 subnormals, s45 the least normal 0x0400 and -0, s46 0.5 in both halves and s47 -0x0401 in
# both. The required row is the add: two normal inputs whose sum is the subnormal 0x8001, the case
# DXIL's fadd FTZ text requires to flush. max keeps a subnormal input and the multiply makes the
# subnormal 0x0200 from normal inputs (supplementary rows), and the fma rounds a subnormal and a normal
# result to nearest even. v14 exports 0x80000200, a binary32 subnormal as a word, so the draw reads
# back through the integer target, where no float path can flush the packed bits.
F16_DENORM_BODY = [
    "v_pk_add_f16 v12, s45, s47",
    "v_pk_max_f16 v13, s44, s47",
    "v_pk_mul_f16 v14, s45, s46",
    "v_pk_fma_f16 v15, s44, s46, s44",
]
F16_DENORM_WORDS = (0x03ff0001, 0x80000400, 0x38003800, 0x84018401)


def f16_denorm_words(flush):
    """Reference words for F16_DENORM_BODY, optionally flushing binary16 subnormal inputs and results."""
    def half(word):
        if flush and word & 0x7c00 == 0:
            word &= 0x8000
        return struct.unpack("<e", struct.pack("<H", word))[0]

    def pack(value):
        word, = struct.unpack("<H", struct.pack("<e", value))
        return word & 0x8000 if flush and word & 0x7c00 == 0 else word

    a, b, c, d = F16_DENORM_WORDS
    ops = (lambda x, y, z: x + y, lambda x, y, z: max(x, y), lambda x, y, z: x * y, lambda x, y, z: x * y + z)
    sources = ((b, d, 0), (a, d, 0), (b, c, 0), (a, c, a))
    words = []
    for op, (x, y, z) in zip(ops, sources):
        # Python doubles hold these products and sums exactly, so one rounding to binary16 matches the shader.
        lanes = [pack(op(half(x >> s & 0xffff), half(y >> s & 0xffff), half(z >> s & 0xffff))) for s in (0, 16)]
        words.append(lanes[0] | lanes[1] << 16)
    return tuple(words)


def check_f16_denorm_draw(directory, compiler, renderer, backend):
    """Draw packed binary16 subnormals under both -denorm modes; the f32 flag must not change them."""
    source = directory / "f16_denorm.hlsl"
    shader = directory / "f16_denorm.cso"
    preserve_case = directory / "f16_denorm-preserve.case"
    flushed_case = directory / "f16_denorm-flushed.case"
    constants = bytearray(256)
    struct.pack_into("<4I", constants, 12, *F16_DENORM_WORDS)
    source.write_text(lower(uncompressed_export(F16_DENORM_BODY), 20, 24, [1, 2, 3, 4],
                            constant_descriptor=64, integer_export=True), encoding="ascii")

    def as_floats(words):
        return tuple(struct.unpack("<f", struct.pack("<I", word))[0] for word in words)

    preserve = f16_denorm_words(flush=False)
    write_case(preserve_case, constants, as_floats(preserve), integer_rt=True)
    # A reference that flushes binary16 subnormals differs from the preserve one in every channel,
    # so the ftz build must fail it.
    flushed = f16_denorm_words(flush=True)
    assert all(p != f for p, f in zip(preserve, flushed)), (preserve, flushed)
    write_case(flushed_case, constants, as_floats(flushed), integer_rt=True)
    compile_shader(compiler, source, shader, mode="preserve")
    subprocess.run([renderer, backend, "--check-pixel-case", shader, preserve_case], check=True)
    compile_shader(compiler, source, shader, mode="ftz")
    run_pixel_batch(renderer, backend, shader, [(flushed_case, "mismatch"), (preserve_case, "pass")])


def check_nan_class_draw(directory, compiler, renderer, backend):
    """Check v_add_f32 NaN results by class on masked channels; finite channels stay strict."""
    source = directory / "nan_class.hlsl"
    shader = directory / "nan_class.cso"
    case = directory / "nan_class.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    words = (0x7fc00000, 0x7f800001, 0xffc00003, 0x3f800000)
    values = [struct.unpack("<f", struct.pack("<I", word))[0] for word in words]
    constants = bytearray(256)
    struct.pack_into("<4I", constants, 12, *words)
    program = uncompressed_export([
        "v_add_f32_e32 v12, s44, s47",
        "v_add_f32_e32 v13, s45, s46",
        "v_add_f32_e32 v14, s45, s45",
    ])
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    write_case(case, constants, (*values[:3], 1.0), nan_mask=0b0111)
    subprocess.run(draw, check=True)
    # A masked channel whose shader output is finite must fail on every backend.
    finite = [line.replace("v12, s44, s47", "v12, s47, s47") for line in program]
    source.write_text(translate(finite), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("A finite export satisfied a NaN-class channel")
    # A strict finite channel against a NaN export must also fail on every backend.
    nan_alpha = program[:-2] + ["v_add_f32_e32 v15, s45, s46"] + program[-2:]
    source.write_text(translate(nan_alpha), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("A NaN export satisfied the strict finite channel")


def check_integer_readback_draw(directory, compiler, renderer, backend):
    """Round-trip v_mov_b32 bit copies bit-for-bit through the integer readback target."""
    source = directory / "integer_readback.hlsl"
    shader = directory / "integer_readback.cso"
    pass_case = directory / "integer_readback-pass.case"
    corrupted_case = directory / "integer_readback-corrupted.case"
    words = (0x7fc00001, 0xffc00003, 0x00000001, 0x80000000)
    constants = bytearray(256)
    struct.pack_into("<I", constants, 12, words[0])
    program = uncompressed_export([
        "v_mov_b32_e32 v12, s44",
        f"v_mov_b32_e32 v13, {words[1]:#x}",
        f"v_mov_b32_e32 v14, {words[2]:#x}",
        f"v_mov_b32_e32 v15, {words[3]:#x}",
    ])
    source.write_text(lower(program, 20, 24, [1, 2, 3, 4], constant_descriptor=64,
                            integer_export=True), encoding="ascii")
    compile_shader(compiler, source, shader)
    expected = tuple(struct.unpack("<f", struct.pack("<I", word))[0] for word in words)
    write_case(pass_case, constants, expected, integer_rt=True)
    corrupted = list(expected)
    corrupted[2] = struct.unpack("<f", struct.pack("<I", words[2] ^ 3))[0]
    write_case(corrupted_case, constants, corrupted, integer_rt=True)
    run_pixel_batch(renderer, backend, shader, [(corrupted_case, "mismatch"), (pass_case, "pass")])


def check_subrev_u32_draw(directory, compiler, renderer, backend):
    source = directory / "subrev_u32.hlsl"
    shader = directory / "subrev_u32.cso"
    program = uncompressed_export([
        "v_mov_b32_e32 v18, s45",
        "v_subrev_nc_u32_e32 v12, s44, v18",
        "v_sub_nc_u32_e32 v13, s44, v18",
        "v_mov_b32_e32 v19, s44",
        "v_subrev_nc_u32_e32 v14, v19, v18",
        "v_mov_b32_e32 v15, 0",
    ])
    source.write_text(lower(program, 20, 24, [1, 2, 3, 4], constant_descriptor=64,
                            integer_export=True), encoding="ascii")
    compile_shader(compiler, source, shader)
    cases = []
    # Check normal subtraction and unsigned wrap with exact RGBA32_UINT readback.
    for name, inputs, words, status in (
        ("forward-oracle", (1, 0), (1, 0xffffffff), "mismatch"),
        ("wrap", (1, 0), (0xffffffff, 1), "pass"),
        ("normal", (3, 7), (4, 0xfffffffc), "pass"),
    ):
        constants = bytearray(256)
        struct.pack_into("<2I", constants, 12, *inputs)
        case = directory / f"subrev-u32-{name}.case"
        a, b = inputs
        two_vgprs = (b - a) % (1 << 32)
        case.write_bytes(constants + struct.pack("<4I", *words, two_vgprs, 0) * 64 + b"\x01")
        cases.append((case, status))
    run_pixel_batch(renderer, backend, shader, cases)


def binary32_normal_or_zero(word):
    """True for a normal binary32 or a signed zero: no NaN, infinity or subnormal pattern."""
    exponent = (word >> 23) & 0xff
    return 0 < exponent < 0xff or word & 0x7fffffff == 0


def check_align_draws(directory, compiler, renderer, backend):
    source, shader = directory / "align.hlsl", directory / "align.cso"
    pairs = ((0x12345678, 0x23456789), (0x3f123456, 0x40123456), (0x89abcdef, 0x91234567))
    for byte in (False, True):
        cases = []
        for a, b in pairs:
            for count in (0, 1, 3, 4, 31, 32, 63, 0xffffffff):
                constants = bytearray(256)
                struct.pack_into("<3I", constants, 12, a, b, count)
                words = align_reference(a, b, count, byte)
                assert all(binary32_normal_or_zero(w) for w in (a, b, *words))
                path = directory / f"align-{byte}-{len(cases)}.case"
                path.write_bytes(constants + struct.pack("<4I", *words) * 64 + b"\x01")
                cases.append((path, "pass"))
        wrong = directory / f"align-wrong-{byte}.case"
        data = bytearray(cases[0][0].read_bytes())
        data[256] ^= 1
        wrong.write_bytes(data)
        for swapped in (False, True):
            source.write_text(translate_int(align_integer_group(byte, swapped)), encoding="ascii")
            compile_shader(compiler, source, shader)
            batch = [(path, "mismatch" if swapped else status) for path, status in cases]
            if not swapped:
                batch.insert(1, (wrong, "mismatch"))
            run_pixel_batch(renderer, backend, shader, batch)


def check_vop3_ternary_int_draws(directory, compiler, renderer, backend):
    source, shader = directory / "vop3_int.hlsl", directory / "vop3_int.cso"
    pairs = ((0x12345678, 0x23456789), (0x456789ab, 0x11223344), (0x89abcdef, 0x91234567))
    counts = (0, 1, 2, 3, 4, 8, 16, 24, 31, 32)
    for group in (1, 2):
        cases = []
        for a, b in pairs:
            for c in counts:
                constants = bytearray(256)
                struct.pack_into("<3I", constants, 12, a, b, c)
                words = vop3_ternary_int_reference(group, a, b, c)
                assert all(binary32_normal_or_zero(w) for w in (a, b, *words))
                path = directory / f"vop3_int-{group}-{len(cases)}.case"
                path.write_bytes(constants + struct.pack("<4I", *words) * 64 + b"\x01")
                cases.append((path, "pass"))
        wrong = directory / f"vop3_int-wrong-{group}.case"
        data = bytearray(cases[0][0].read_bytes())
        data[256] ^= 1
        wrong.write_bytes(data)
        for mutated in (False, True):
            source.write_text(translate_int(vop3_ternary_int_group(group, mutated)), encoding="ascii")
            compile_shader(compiler, source, shader)
            batch = [(path, "mismatch" if mutated else status) for path, status in cases]
            if not mutated:
                batch.insert(1, (wrong, "mismatch"))
            run_pixel_batch(renderer, backend, shader, batch)


def check_integer_nc_draws(directory, compiler, renderer, backend):
    """Draw the 32-bit no-carry add/sub forms through the integer readback target and compare the words.

    Group 1 and 2 run on every row of NC_INT_ROWS and group 3 on the third row: 11 draws from 3 compiles.
    The expected words follow the sub and subrev operand order of the translator (see the docs); on a desktop
    GPU the draws establish that the translated shader computes exactly that, not what the console does.
    """
    source = directory / "integer_nc.hlsl"
    shader = directory / "integer_nc.cso"

    def words_as_floats(words):
        return tuple(struct.unpack("<f", struct.pack("<I", word))[0] for word in words)

    def expected_words(group, a, b):
        words = tuple(word & 0xffffffff for word in NC_INT_RESULTS[group](a, b))
        # Integer export carries any pattern, but normal rows keep the draw independent of NaN and denormal handling.
        assert all(binary32_normal_or_zero(word) for word in (a, b, *words)), (group, hex(a), hex(b))
        return words

    for group, rows in ((1, NC_INT_ROWS), (2, NC_INT_ROWS), (3, NC_INT_ROWS[2:3])):
        source.write_text(translate_int(uncompressed_export(nc_integer_group(group))), encoding="ascii")
        compile_shader(compiler, source, shader)
        cases = []
        for idx, (a, b) in enumerate(rows):
            constants = bytearray(256)
            struct.pack_into("<II", constants, 12, a, b)
            words = expected_words(group, a, b)
            case = directory / f"integer_nc-{group}-{idx}.case"
            write_case(case, constants, words_as_floats(words), integer_rt=True)
            cases.append((case, "pass"))
        if group == 1:
            # Negative control: the third row has A - B != B - A, so swapped sub and subrev words must mismatch.
            a, b = NC_INT_ROWS[2]
            constants = bytearray(256)
            struct.pack_into("<II", constants, 12, a, b)
            words = expected_words(1, a, b)
            assert words[1] != words[2]
            swapped = (words[0], words[2], words[1], words[3])
            wrong = directory / "integer_nc-1-swapped.case"
            write_case(wrong, constants, words_as_floats(swapped), integer_rt=True)
            cases.insert(-1, (wrong, "mismatch"))
        if len(cases) > 1:
            run_pixel_batch(renderer, backend, shader, cases)
        else:
            subprocess.run([renderer, backend, "--check-pixel-case", shader, cases[0][0]], check=True)
    return 0


def check_e64_integer_draws(directory, compiler, renderer, backend):
    """Draw the integer e64 forms and v_xnor_b32 through the integer readback target and compare the words.

    Each group of E64_INT_BODIES is compiled once and drawn in one renderer batch: every row of E64_INT_ROWS
    with the right words, and the first row once per E64_INT_WRONG entry with one channel replaced by the word
    a wrong lowering would give, which must report a mismatch. On a desktop GPU the draws establish that the
    translated shader computes the stated operation, not what the console does.
    """
    source = directory / "e64_integer.hlsl"
    shader = directory / "e64_integer.cso"

    def add_case(cases, row, words, status):
        constants = bytearray(256)
        struct.pack_into("<3I", constants, 12, *row)
        case = directory / f"e64_integer-{len(cases)}.case"
        case.write_bytes(bytes(constants) + struct.pack("<4I", *words) * 64 + b"\x01")
        cases.append((case, status))

    for group, rows in E64_INT_ROWS.items():
        source.write_text(translate_int(uncompressed_export(e64_integer_group(group))), encoding="ascii")
        compile_shader(compiler, source, shader)
        cases = []
        for index, row in enumerate(rows):
            words = tuple(word & 0xffffffff for word in E64_INT_RESULTS[group](*row))
            # Integer export carries any pattern, but normal rows keep the draw independent of NaN and denormal handling.
            assert all(binary32_normal_or_zero(word) for word in (*row, *words)), (group, [hex(w) for w in (*row, *words)])
            add_case(cases, row, words, "pass")
            if index:
                continue
            for channel, wrong, reason in E64_INT_WRONG[group]:
                wrong_words = list(words)
                wrong_words[channel] = wrong(*row) & 0xffffffff
                assert wrong_words[channel] != words[channel], (group, reason)
                add_case(cases, row, wrong_words, "mismatch")
        try:
            run_pixel_batch(renderer, backend, shader, cases)
        except subprocess.CalledProcessError as error:
            if error.returncode == 77:
                return 77
            raise
    return 0


def check_sdwa_integer_draws(directory, compiler, renderer, backend):
    """Draw the SDWA forms of the 32-bit integer operations through the integer readback target.

    Each group of SDWA_INT_BODIES is compiled once and drawn in one renderer batch: every row of
    SDWA_INT_ROWS with the right words, then one draw per SDWA_INT_WRONG entry with one channel replaced by
    the word a wrong selection, extension, placement or operation would give, which must report a mismatch.
    On a desktop GPU the draws establish that the translated shader computes the stated words, not what the
    console does.
    """
    source = directory / "sdwa_integer.hlsl"
    shader = directory / "sdwa_integer.cso"
    for group in SDWA_INT_BODIES:
        source.write_text(translate_int(uncompressed_export(sdwa_integer_group(group))), encoding="ascii")
        compile_shader(compiler, source, shader)
        cases = []
        for row, words, status, _ in sdwa_integer_cases(group):
            constants = bytearray(256)
            struct.pack_into("<4I", constants, 12, *row)
            case = directory / f"sdwa_integer-{len(cases)}.case"
            case.write_bytes(bytes(constants) + struct.pack("<4I", *words) * 64 + b"\x01")
            cases.append((case, status))
        try:
            run_pixel_batch(renderer, backend, shader, cases)
        except subprocess.CalledProcessError as error:
            if error.returncode == 77:
                return 77
            raise
    return 0


def check_pack_int16_draws(directory, compiler, renderer, backend):
    """Draw v_cvt_pk_u16_u32 and v_cvt_pk_i16_i32 through the integer readback target and compare the words.

    Each group of PACK_BODIES is compiled once and drawn in one renderer batch: every row of PACK_ROWS with
    the right words, and the first row once per PACK_WRONG entry with one channel replaced by the word a wrong
    lowering would give, which must report a mismatch. On a desktop GPU the draws establish that the
    translated shader computes the stated operation, not what the console does.
    """
    source = directory / "pack_int16.hlsl"
    shader = directory / "pack_int16.cso"

    def add_case(cases, row, words, status):
        constants = bytearray(256)
        struct.pack_into("<2I", constants, 12, *row)
        case = directory / f"pack_int16-{len(cases)}.case"
        case.write_bytes(bytes(constants) + struct.pack("<4I", *words) * 64 + b"\x01")
        cases.append((case, status))

    for group in PACK_BODIES:
        source.write_text(translate_int(uncompressed_export(pack_group(group))), encoding="ascii")
        compile_shader(compiler, source, shader)
        cases = []
        for index, row in enumerate(PACK_ROWS):
            words = tuple(word & 0xffffffff for word in PACK_RESULTS[group](*row))
            # Integer export carries any pattern, but normal rows keep the draw independent of NaN and denormal handling.
            assert all(binary32_normal_or_zero(word) for word in (*row, *words)), (group, [hex(w) for w in (*row, *words)])
            add_case(cases, row, words, "pass")
            if index:
                continue
            seen = set()
            for channel, wrong, reason in PACK_WRONG[group]:
                wrong_words = list(words)
                wrong_words[channel] = wrong(*row) & 0xffffffff
                assert wrong_words[channel] != words[channel], (group, reason)
                assert (channel, wrong_words[channel]) not in seen, (group, reason)
                seen.add((channel, wrong_words[channel]))
                add_case(cases, row, wrong_words, "mismatch")
        try:
            run_pixel_batch(renderer, backend, shader, cases)
        except subprocess.CalledProcessError as error:
            if error.returncode == 77:
                return 77
            raise
    return 0


def check_carry_out_draws(directory, compiler, renderer, backend):
    """Draw v_add_co_u32, v_sub_co_u32 and v_subrev_co_u32 through the integer readback target, bit for bit.

    Each program of carry_out_group() is compiled once and drawn in one renderer batch: every row of
    CARRY_OUT_ROWS with the right words, then one draw per CARRY_OUT_WRONG entry with one channel of its row
    replaced by the word a wrong lowering would give, which must report a mismatch. The carry is observed
    through v_cndmask_b32. On a desktop GPU the draws establish that the translated shader computes the stated
    operation, not what the console does.
    """
    source = directory / "carry_out.hlsl"
    shader = directory / "carry_out.cso"

    def add_case(cases, row, words, status):
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *row)
        case = directory / f"carry_out-{len(cases)}.case"
        case.write_bytes(bytes(constants) + struct.pack("<4I", *words) * 64 + b"\x01")
        cases.append((case, status))

    for op in CARRY_OUT_OPS:
        source.write_text(translate_int(uncompressed_export(carry_out_group(op))), encoding="ascii")
        compile_shader(compiler, source, shader)
        rows = [(a, b, *_CARRY_SELECT) for a, b in CARRY_OUT_ROWS[op]]
        right = [carry_out_words(op, *row) for row in rows]
        cases = []
        for row, words in zip(rows, right):
            # Integer export carries any pattern, but normal rows keep the draw independent of NaN and denormal handling.
            assert all(binary32_normal_or_zero(word) for word in (*row, *words)), (op, [hex(w) for w in (*row, *words)])
            add_case(cases, row, words, "pass")
        # Both carry states are drawn for each mask register, so a constant mask cannot pass.
        for channel in (1, 3):
            assert {words[channel] for words in right} == set(_CARRY_SELECT), (op, channel)
        for index, channel, wrong, reason in CARRY_OUT_WRONG[op]:
            wrong_words = list(right[index])
            wrong_words[channel] = wrong(*rows[index]) & 0xffffffff
            assert wrong_words[channel] != right[index][channel], (op, reason)
            add_case(cases, rows[index], wrong_words, "mismatch")
        try:
            run_pixel_batch(renderer, backend, shader, cases)
        except subprocess.CalledProcessError as error:
            if error.returncode == 77:
                return 77
            raise
    return 0


def check_carry_in_draws(directory, compiler, renderer, backend):
    """Draw v_add_co_ci_u32, v_sub_co_ci_u32 and v_subrev_co_ci_u32 through the integer readback target.

    Each program of carry_in_group() is compiled once and drawn in one renderer batch: every row of
    CARRY_IN_ROWS with the right words, then one draw per CARRY_IN_WRONG entry with one channel of its row
    replaced by the word a wrong lowering would give, which must report a mismatch. The carry-out is observed
    through v_cndmask_b32. On a desktop GPU the draws establish that the translated shader computes the stated
    operation, not what the console does.
    """
    source = directory / "carry_in.hlsl"
    shader = directory / "carry_in.cso"

    def add_case(cases, row, words, status):
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *row)
        case = directory / f"carry_in-{len(cases)}.case"
        case.write_bytes(bytes(constants) + struct.pack("<4I", *words) * 64 + b"\x01")
        cases.append((case, status))

    for op in CARRY_IN_OPS:
        source.write_text(translate_int(uncompressed_export(carry_in_group(op))), encoding="ascii")
        compile_shader(compiler, source, shader)
        rows = CARRY_IN_ROWS[op]
        right = [carry_in_words(op, *row) for row in rows]
        cases = []
        for row, words in zip(rows, right):
            first_in, second_in = carry_in_masks(op, *row)
            raw = (_carry_in(op, row[2], row[3], first_in)[0], _carry_in(op, row[3], row[2], second_in)[0])
            # Integer export carries any pattern, but normal words keep the draw independent of NaN and denormal
            # handling. The only other word is the all-ones result of a borrow taken from the borrow-in alone.
            assert all(binary32_normal_or_zero(word) for word in (*row, *words)), (op, [hex(w) for w in (*row, *words)])
            assert all(binary32_normal_or_zero(word) or word == 0xffffffff for word in raw), (op, [hex(w) for w in raw])
            add_case(cases, row, words, "pass")
        # Each carry-in mask is set and clear on some row, and so is each carry-out, so a constant mask cannot pass.
        assert {carry_in_masks(op, *row) for row in rows} >= {(True, True), (False, False), (True, False), (False, True)}, op
        for channel in (1, 3):
            assert {words[channel] for words in right} == set(_CARRY_IN_SELECT), (op, channel)
        # Row 0 carries or borrows only because of the carry-in, on both channels.
        c, d = rows[0][2], rows[0][3]
        assert not _carry_in(op, c, d, 0)[1] and not _carry_in(op, d, c, 0)[1], op
        assert right[0][1] == right[0][3] == _CARRY_IN_SELECT[1], op
        for index, channel, wrong, reason in CARRY_IN_WRONG[op]:
            wrong_words = list(right[index])
            wrong_words[channel] = wrong(*rows[index]) & 0xffffffff
            assert wrong_words[channel] != right[index][channel], (op, reason)
            add_case(cases, rows[index], wrong_words, "mismatch")
        try:
            run_pixel_batch(renderer, backend, shader, cases)
        except subprocess.CalledProcessError as error:
            if error.returncode == 77:
                return 77
            raise
    return 0


def check_sad_draws(directory, compiler, renderer, backend):
    """Draw v_sad_hi_u8, v_sad_u32 and v_msad_u8 through the integer readback target and compare the words.

    Each group of SAD_BODIES is compiled once and drawn in one renderer batch: every row of SAD_ROWS with the
    right words, and the first row once per SAD_WRONG entry with one channel replaced by the word a wrong
    lowering would give, which must report a mismatch. On a desktop GPU the draws establish that the
    translated shader computes the stated operation, not what the console does.
    """
    source = directory / "sad.hlsl"
    shader = directory / "sad.cso"

    def add_case(cases, row, words, status):
        constants = bytearray(256)
        struct.pack_into("<3I", constants, 12, *row)
        case = directory / f"sad-{len(cases)}.case"
        case.write_bytes(bytes(constants) + struct.pack("<4I", *words) * 64 + b"\x01")
        cases.append((case, status))

    for group, rows in SAD_ROWS.items():
        source.write_text(translate_int(uncompressed_export(sad_group(group))), encoding="ascii")
        compile_shader(compiler, source, shader)
        cases = []
        for index, row in enumerate(rows):
            words = tuple(word & 0xffffffff for word in SAD_RESULTS[group](*row))
            # Integer export carries any pattern, but normal rows keep the draw independent of NaN and denormal handling.
            assert all(binary32_normal_or_zero(word) for word in (*row, *words)), (group, [hex(w) for w in (*row, *words)])
            add_case(cases, row, words, "pass")
            if index:
                continue
            seen = set()
            for channel, wrong, reason in SAD_WRONG[group]:
                wrong_words = list(words)
                wrong_words[channel] = wrong(*row) & 0xffffffff
                assert wrong_words[channel] != words[channel], (group, reason)
                assert (channel, wrong_words[channel]) not in seen, (group, reason)
                seen.add((channel, wrong_words[channel]))
                add_case(cases, row, wrong_words, "mismatch")
        try:
            run_pixel_batch(renderer, backend, shader, cases)
        except subprocess.CalledProcessError as error:
            if error.returncode == 77:
                return 77
            raise
    return 0


# flat_load_dword and global_load_dword through TitleLookup (issue #319). Four sorted title ranges alias four
# buffers. Each range's buffer holds at least as many words as the range, and slot 1 holds more, so a load that
# ignored the end of its range would read a real word instead of TITLE_MISS.
TITLE_RANGES = (  # (begin, end, slot)
    (0x0_0010_0000, 0x0_0010_0102, 1),  # its last DWORD would end two bytes past the range
    (0x0_ffff_f800, 0x1_0000_0800, 2),  # crosses the 32-bit boundary
    (0x2_0000_0000, 0x2_0000_1000, 0),
    (0x2_0000_1000, 0x2_0000_2000, 3),  # adjacent to the previous range, in another slot
)
TITLE_ALIAS_WORDS = (0x480, 0x80, 0x480, 0x480)
# Address A is constant words 3 and 4 (s44, s45), address B words 5 and 6 (s46, s47). The four channels load A,
# A + 2044, B - 2048 and B. Flat offsets are 11-bit unsigned on gfx1013, so the negative offset is a global load.
TITLE_BODY = [
    "v_mov_b32_e32 v20, s44",
    "v_mov_b32_e32 v21, s45",
    "v_mov_b32_e32 v22, s46",
    "v_mov_b32_e32 v23, s47",
    "flat_load_dword v12, v[20:21]",
    "flat_load_dword v13, v[20:21] offset:2044",
    "global_load_dword v14, v[22:23], off offset:-2048",
    "global_load_dword v15, v[22:23], off",
    "s_waitcnt vmcnt(0) lgkmcnt(0)",
]
TITLE_CHANNELS = ((0, 0), (0, 2044), (1, -2048), (1, 0))  # (address A or B, offset)
TITLE_CASES = (  # (A, B)
    (0x2_0000_0000, 0x2_0000_1800),  # first words: an offset inside a range, the first word of the next slot
    (0x0_0010_00fc, 0x2_0000_17fc),  # last whole DWORD of two ranges; A + 2044 is past the first range
    (0x0_ffff_fc00, 0x1_0000_0400),  # A + 2044 carries into the high word, B - 2048 borrows from it
    (0x1_8000_0000, 0x0_0000_0800),  # misses: a hole between ranges, and below the first range
    (0x0_0010_0100, 0x0_0010_0900),  # the DWORD straddling the first range's end, from both addresses
    (0x2_0000_0ffc, 0x0_0010_0800),  # three slots in one draw
)
# Wrong references that the draw must refuse, each with the (case, channel) it changes: the high address word
# dropped, the immediate ignored, the immediate masked to 12 bits instead of sign-extended (no borrow), the carry
# dropped, zero instead of the sentinel, a DWORD accepted when only its first byte is inside the range, the slot
# masked to one bit, and the range count frozen one short.
TITLE_MUTATIONS = {"high": (0, 0), "immediate": (0, 1), "mask": (2, 2), "carry": (2, 1), "sentinel": (3, 0),
                   "straddle": (4, 0), "slot": (0, 2), "count": (0, 2)}


def title_word(slot, index):
    """The word at DWORD index of slot's aliased buffer: distinct per slot and index, and a normal binary32."""
    return 0xa0000000 | slot << 20 | index


def title_memory_bytes():
    """The case trailer after flag bit 2: counts, the ranges as TitleRange records, then each slot's words."""
    slots = pixel_aot.TITLE_ALIAS_SLOTS
    data = bytearray(struct.pack("<2I", len(TITLE_RANGES), slots))
    for begin, end, slot in TITLE_RANGES:
        data += struct.pack("<6I", begin & 0xffffffff, begin >> 32, end & 0xffffffff, end >> 32, slot, 0)
    for slot in range(slots):
        words = [title_word(slot, j) for j in range(TITLE_ALIAS_WORDS[slot])] if slot < len(TITLE_ALIAS_WORDS) else [0]
        data += struct.pack(f"<{1 + len(words)}I", len(words), *words)
    return bytes(data)


def title_reference(base, offset, mutation=None):
    """The DWORD a load at base + offset returns, or TITLE_MISS; mutation names one wrong reference."""
    mask = 0xffffffff
    if mutation == "high":
        base &= mask
    if mutation == "immediate":
        offset = 0
    extended = (offset & 0xfff) if mutation == "mask" else offset & 0xffffffffffffffff
    low = (base & mask) + (extended & mask)
    carry = 0 if mutation == "carry" else low >> 32
    address = (((base >> 32) + (extended >> 32) + carry) & mask) << 32 | (low & mask)
    ranges = TITLE_RANGES[:-1] if mutation == "count" else TITLE_RANGES
    for begin, end, slot in ranges:
        if address % 4 == 0 and begin <= address and address + (1 if mutation == "straddle" else 4) <= end:
            return title_word(slot & 1 if mutation == "slot" else slot, (address - begin) // 4)
    return 0 if mutation == "sentinel" else pixel_aot.TITLE_MISS


def title_words(a, b, mutation=None):
    return tuple(title_reference((a, b)[which], offset, mutation) for which, offset in TITLE_CHANNELS)


def title_program():
    return uncompressed_export(TITLE_BODY)


def translate_title(instructions):
    """translate_int() with the TITLE_RANGES table bound for flat and global loads."""
    return lower(instructions, 20, 24, [1, 2, 3, 4], constant_descriptor=64, integer_export=True,
                 title_ranges=len(TITLE_RANGES), flat_global_only=True)


def title_controls():
    """Check that each wrong reference changes its own case and channel, so its draw control can fail."""
    assert all(a % 4 == 0 and b % 4 == 0 for a, b in TITLE_CASES)
    assert list(TITLE_RANGES) == sorted(TITLE_RANGES) and len({slot for *_, slot in TITLE_RANGES}) == 4
    expected = [title_words(a, b) for a, b in TITLE_CASES]
    hits = [word for words in expected for word in words if word != pixel_aot.TITLE_MISS]
    assert len(hits) == 14 and len({word >> 20 & 0xf for word in hits}) == 4
    assert expected[2] == (title_word(2, 0x100), title_word(2, 0x2ff), title_word(2, 0x100), title_word(2, 0x300))
    assert expected[4] == (pixel_aot.TITLE_MISS,) * 4
    for mutation, (index, channel) in TITLE_MUTATIONS.items():
        assert title_words(*TITLE_CASES[index], mutation)[channel] != expected[index][channel], mutation
    # Without the carry A + 2044 lands in the low 4 GiB, and without the borrow B - 2048 stays above 4 GiB.
    assert title_words(*TITLE_CASES[2], "carry")[1] == title_words(*TITLE_CASES[2], "mask")[2] == pixel_aot.TITLE_MISS


def check_title_load_controls():
    title_controls()
    program = title_program()
    source = translate_title(program)
    assert source.count("bool TitleLookup(") == 1 and source.count("uint title_load_dword(") == 1
    assert "static const uint TITLE_RANGE_COUNT = 4u;" in source
    assert "ByteAddressBuffer TitleAliases[8] : register(t1, space1);" in source
    assert "StructuredBuffer<TitleRange> TitleRanges : register(t0, space1);" in source
    assert "uint2(0x7fcu, 0x0u)" in source and "uint2(0xfffff800u, 0xffffffffu)" in source
    assert source.count("title_load_dword(uint2(") == 4
    assert "TitleLookup" not in translate_int(uncompressed_export(TITLE_BODY[:4]))
    for count in (0, -1, True, 2.0, 1 << 32):
        expect_value_error(lambda p: lower(p, 20, 24, [1, 2, 3, 4], constant_descriptor=64, integer_export=True,
                                           title_ranges=count), program, "Invalid title range count")
    expect_value_error(translate_int, program, "Title memory is not bound")
    def without_flat_binding(instructions):
        return lower(instructions, 20, 24, [1, 2, 3, 4], constant_descriptor=64,
                     integer_export=True, title_ranges=len(TITLE_RANGES))

    expect_value_error(without_flat_binding, program, "Flat loads require the global-only address profile")
    global_program = [line.replace("flat_load_dword", "global_load_dword")
                      .replace(":21]", ":21], off")
                      if line.startswith("flat_load_dword") else line for line in program]
    assert without_flat_binding(global_program).count("title_load_dword(uint2(") == 4
    flat = "flat_load_dword v12, v[20:21]"

    def replaced(line):
        return [line if entry == flat else entry for entry in program]

    # Neighbours that stay unsupported: other widths, sub-DWORD and D16 loads, stores, atomics, scratch, the
    # global saddr and LDS forms, cache-policy bits, and offsets outside each segment's assembler range.
    for neighbour in ("flat_load_dwordx2 v[12:13], v[20:21]", "flat_load_dwordx3 v[12:14], v[20:21]",
                      "flat_load_dwordx4 v[12:15], v[20:21]", "flat_load_ubyte v12, v[20:21]",
                      "flat_load_sbyte v12, v[20:21]", "flat_load_ushort v12, v[20:21]",
                      "flat_load_sshort v12, v[20:21]", "flat_load_short_d16 v12, v[20:21]",
                      "flat_load_short_d16_hi v12, v[20:21]", "flat_store_dword v[20:21], v12",
                      "flat_atomic_add v12, v[20:21], v13 glc", "global_load_dwordx2 v[12:13], v[20:21], off",
                      "global_load_ubyte v12, v[20:21], off", "global_store_dword v[20:21], v12, off",
                      "global_load_dword v12, v20, s[44:45]", "global_load_dword v12, v20, s[44:45] offset:-8",
                      "global_load_dword v[20:21], off lds", "scratch_load_dword v12, v20, off",
                      "scratch_load_dword v12, off, s48", "scratch_store_dword off, v12, s48 offset:16",
                      "flat_load_dword v12, v[20:21] glc", "flat_load_dword v12, v[20:21] offset:4 glc slc dlc",
                      "global_load_dword v12, v[20:21], off slc"):
        expect_value_error(translate_title, replaced(neighbour), "Unsupported instruction")
    for form in ("flat_load_dword v12, v[20:21] offset:2048", "flat_load_dword v12, v[20:21] offset:4088",
                 "flat_load_dword v12, v[20:21] offset:-8", "flat_load_dword v12, v[20:21], off",
                 "global_load_dword v12, v[20:21]", "global_load_dword v12, v[20:21], off offset:2048",
                 "global_load_dword v12, v[20:21], off offset:-2049"):
        expect_value_error(translate_title, replaced(form), "load form")
    must_reject(translate_title, replaced("flat_load_dword v12, v[20:22]"))
    must_reject(translate_title, replaced("flat_load_dword v12, v[24:25]"))  # address never written
    expect_value_error(translate_title, replaced("flat_load_dword v256, v[20:21]"), "Invalid title memory load")
    wqm = program.index("s_wqm_b64 exec, exec") + 1
    expect_value_error(translate_title, program[:wqm] + [flat] + program[wqm:], "Invalid title memory load")
    # A loaded word is ready only after vmcnt covers its load: no wait, an lgkmcnt-only wait, or vmcnt(1) with
    # the newest load still outstanding are refused; a reused destination before the wait is refused too.
    wait = program.index("s_waitcnt vmcnt(0) lgkmcnt(0)")
    for replacement in ([], ["s_waitcnt lgkmcnt(0)"], ["s_waitcnt vmcnt(1)"]):
        must_reject(translate_title, program[:wait] + replacement + program[wait + 1:])
    assert translate_title(program[:wait] + ["s_waitcnt vmcnt(0)"] + program[wait + 1:]) == source
    expect_value_error(translate_title, program[:wait] + ["v_add_f32_e32 v16, v12, v12"] + program[wait:],
                       "Sample used before vmcnt wait")
    expect_value_error(translate_title, program[:wait] + ["flat_load_dword v12, v[22:23]"] + program[wait:],
                       "Invalid title memory load")
    loaded = program[:wait + 1] + ["flat_load_dword v15, v[12:13]", "s_waitcnt vmcnt(0)"] + program[wait + 1:]
    assert translate_title(loaded).count("title_load_dword(uint2(") == 5


def check_title_load_draws(directory, compiler, renderer, backend):
    """Draw TITLE_CASES and one wrong reference per mutation through the integer readback: 22 draws, one batch."""
    source = directory / "title_loads.hlsl"
    shader = directory / "title_loads.cso"
    source.write_text(translate_title(title_program()), encoding="ascii")
    compile_shader(compiler, source, shader)
    memory = title_memory_bytes()

    def case(name, a, b, words):
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, a & 0xffffffff, a >> 32, b & 0xffffffff, b >> 32)
        floats = tuple(struct.unpack("<f", struct.pack("<I", word))[0] for word in words)
        return write_case(directory / f"title_{name}.case", constants, floats, integer_rt=True, title_memory=memory)

    cases = [(case(n, a, b, title_words(a, b)), "pass") for n, (a, b) in enumerate(TITLE_CASES)]
    title_controls()
    # Each expected mismatch is followed by the passing draw of the same addresses.
    for mutation, (index, _) in TITLE_MUTATIONS.items():
        a, b = TITLE_CASES[index]
        cases += [(case(mutation, a, b, title_words(a, b, mutation)), "mismatch"), cases[index]]
    run_pixel_batch(renderer, backend, shader, cases)
    return 0


def check_lshl_add_u32_draw(directory, compiler, renderer, backend):
    source = directory / "lshl_add_u32.hlsl"
    shader = directory / "lshl_add_u32.cso"
    program = uncompressed_export([
        "s_lshl1_add_u32 s60, s44, s45",
        "s_lshl2_add_u32 s61, s44, s45",
        "s_lshl3_add_u32 s62, s44, s45",
        "s_lshl4_add_u32 s63, s44, s45",
        "v_mov_b32_e32 v12, s60",
        "v_mov_b32_e32 v13, s61",
        "v_mov_b32_e32 v14, s62",
        "v_mov_b32_e32 v15, s63",
    ])
    source.write_text(lower(program, 20, 24, [1, 2, 3, 4], constant_descriptor=64,
                            integer_export=True), encoding="ascii")
    compile_shader(compiler, source, shader)
    cases = []
    # Check normal shifted addition, shift-overflow wrap, and add-overflow wrap.
    for name, inputs, words, status in (
        ("oracle-mismatch", (3, 5), (11, 17, 29, 999), "mismatch"),
        ("normal", (3, 5), (11, 17, 29, 53), "pass"),
        ("shift-wrap", (0x80000001, 10), (12, 14, 18, 26), "pass"),
        ("add-wrap", (1, 0xffffffff), (1, 3, 7, 15), "pass"),
    ):
        constants = bytearray(256)
        struct.pack_into("<2I", constants, 12, *inputs)
        case = directory / f"lshl-add-u32-{name}.case"
        case.write_bytes(constants + struct.pack("<4I", *words) * 64 + b"\x01")
        cases.append((case, status))
    run_pixel_batch(renderer, backend, shader, cases)

    # Check SCC carry-out across selective shift-overflow and add-overflow conditions.
    scc_source = directory / "lshl_add_scc.hlsl"
    scc_shader = directory / "lshl_add_scc.cso"
    scc_program = uncompressed_export([
        "s_lshl1_add_u32 s60, s44, s45",
        "s_cselect_b32 s71, 1, 0",
        "s_lshl2_add_u32 s61, s44, s45",
        "s_cselect_b32 s72, 1, 0",
        "s_lshl3_add_u32 s62, s44, s45",
        "s_cselect_b32 s73, 1, 0",
        "s_lshl4_add_u32 s63, s44, s45",
        "s_cselect_b32 s74, 1, 0",
        "v_mov_b32_e32 v12, s71",
        "v_mov_b32_e32 v13, s72",
        "v_mov_b32_e32 v14, s73",
        "v_mov_b32_e32 v15, s74",
    ])
    scc_source.write_text(lower(scc_program, 20, 24, [1, 2, 3, 4], constant_descriptor=64,
                                integer_export=True), encoding="ascii")
    compile_shader(compiler, scc_source, scc_shader)
    scc_cases = []
    for name, inputs, words, status in (
        ("oracle-mismatch", (1, 1), (1, 0, 0, 0), "mismatch"),
        ("none", (1, 1), (0, 0, 0, 0), "pass"),
        ("carry-4", (0x10000000, 0), (0, 0, 0, 1), "pass"),
        ("carry-3-4", (0x20000000, 0), (0, 0, 1, 1), "pass"),
        ("carry-2-3-4", (0x40000000, 0), (0, 1, 1, 1), "pass"),
        ("carry-all-shift", (0x80000000, 0), (1, 1, 1, 1), "pass"),
        ("carry-all-add", (1, 0xfffffffe), (1, 1, 1, 1), "pass"),
    ):
        constants = bytearray(256)
        struct.pack_into("<2I", constants, 12, *inputs)
        case = directory / f"lshl-add-scc-{name}.case"
        case.write_bytes(constants + struct.pack("<4I", *words) * 64 + b"\x01")
        scc_cases.append((case, status))
    run_pixel_batch(renderer, backend, scc_shader, scc_cases)


def check_absdiff_i32_draw(directory, compiler, renderer, backend):
    source = directory / "absdiff_i32.hlsl"
    shader = directory / "absdiff_i32.cso"
    program = uncompressed_export([
        "s_absdiff_i32 s60, s44, s45",
        "s_cselect_b32 s71, 1, 0",
        "v_mov_b32_e32 v12, s60",
        "v_mov_b32_e32 v13, s71",
        "v_mov_b32_e32 v14, 0",
        "v_mov_b32_e32 v15, 0",
    ])
    source.write_text(lower(program, 20, 24, [1, 2, 3, 4], constant_descriptor=64,
                            integer_export=True), encoding="ascii")
    compile_shader(compiler, source, shader)
    cases = []
    # Check normal difference, reversed difference, zero diff (scc=0), negative operands, INT_MIN boundary,
    # and (2^31, 2^32) overflow-wrap cases.
    for name, inputs, words, status in (
        ("oracle-mismatch", (10, 4), (6, 0, 0, 0), "mismatch"),
        ("positive-diff", (10, 4), (6, 1, 0, 0), "pass"),
        ("negative-diff", (4, 10), (6, 1, 0, 0), "pass"),
        ("zero-diff", (7, 7), (0, 0, 0, 0), "pass"),
        ("both-negative", (0xfffffff6, 0xfffffffc), (6, 1, 0, 0), "pass"),
        ("mixed-signs", (0xfffffffb, 8), (13, 1, 0, 0), "pass"),
        ("int-min-zero", (0x80000000, 0), (0x80000000, 1, 0, 0), "pass"),
        ("overflow-wrap-max-min", (0x7fffffff, 0x80000000), (1, 1, 0, 0), "pass"),
        ("overflow-wrap-min-pos", (0x80000000, 1), (0x7fffffff, 1, 0, 0), "pass"),
    ):
        constants = bytearray(256)
        struct.pack_into("<2I", constants, 12, *inputs)
        case = directory / f"absdiff-i32-{name}.case"
        case.write_bytes(constants + struct.pack("<4I", *words) * 64 + b"\x01")
        cases.append((case, status))
    run_pixel_batch(renderer, backend, shader, cases)


def check_readfirstlane_draw(directory, compiler, renderer, backend):
    source = directory / "readfirstlane.hlsl"
    shader = directory / "readfirstlane.cso"
    program = readfirstlane_program()
    source.write_text(translate_int(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    cases = []
    for name, input_val, words, status in (
        ("oracle-mismatch", 42, (42, 47, 0x12345678, 999), "mismatch"),
        ("normal", 42, (42, 47, 0x12345678, 0), "pass"),
        ("zero", 0, (0, 5, 0x12345678, 0), "pass"),
        ("large", 0x7ffffff0, (0x7ffffff0, 0x7ffffff5, 0x12345678, 0), "pass"),
    ):
        constants = bytearray(256)
        struct.pack_into("<I", constants, 12, input_val)
        case = directory / f"readfirstlane-{name}.case"
        case.write_bytes(constants + struct.pack("<4I", *words) * 64 + b"\x01")
        cases.append((case, status))
    run_pixel_batch(renderer, backend, shader, cases)


def check_valu_int32_draw(directory, compiler, renderer, backend):
    """Draw 32-bit vector integer ALU forms through the integer readback target and compare the words."""
    # Program A: mul_lo_u32, add3_u32, lshl_add_u32, lshl_or_b32
    src_a = directory / "valu_int32_a.hlsl"
    cso_a = directory / "valu_int32_a.cso"
    prog_a = uncompressed_export([
        "v_mul_lo_u32 v12, s44, s45",
        "v_add3_u32 v13, s44, s45, s46",
        "v_lshl_add_u32 v14, s44, 2, s45",
        "v_lshl_or_b32 v15, s44, s45, s46",
    ])
    src_a.write_text(lower(prog_a, 20, 24, [1, 2, 3, 4], constant_descriptor=64,
                           integer_export=True), encoding="ascii")
    compile_shader(compiler, src_a, cso_a)

    cases_a = []
    # (s44, s45, s46) -> (v12, v13, v14, v15)
    for name, inputs, words, status in (
        ("oracle-mismatch-mul", (0xffffffff, 0xffffffff, 0), (0x2, 0xfffffffe, 0xfffffffb, 0x80000000), "mismatch"),
        ("oracle-mismatch-lane", (0xffffffff, 0xffffffff, 0), (0x1, 0xfffffffe, 0xfffffffb, 0x12345678), "mismatch"),
        ("wrap-max", (0xffffffff, 0xffffffff, 0), (0x1, 0xfffffffe, 0xfffffffb, 0x80000000), "pass"),
        ("wrap-large-nonzero-high", (0x12345678, 0x9abcdef0, 0), (0x242d2080, 0xacf13568, 0xe38e38d0, 0x56780000), "pass"),
        ("overflow-add", (0x40000000, 0x80000000, 1), (0x0, 0xc0000001, 0x80000000, 0x40000001), "pass"),
    ):
        constants = bytearray(256)
        struct.pack_into("<3I", constants, 12, *inputs)
        case = directory / f"valu-int32-a-{name}.case"
        case.write_bytes(constants + struct.pack("<4I", *words) * 64 + b"\x01")
        cases_a.append((case, status))
    run_pixel_batch(renderer, backend, cso_a, cases_a)

    # Negative control: verify that substituting a wrong expression for v_mul_lo_u32 changes the result.
    mut_src = directory / "valu_int32_a_mut.hlsl"
    mut_cso = directory / "valu_int32_a_mut.cso"
    mut_prog = [line.replace("v_mul_lo_u32 v12, s44, s45", "v_add3_u32 v12, s44, s45, 0") for line in prog_a]
    assert mut_prog != prog_a
    mut_src.write_text(lower(mut_prog, 20, 24, [1, 2, 3, 4], constant_descriptor=64,
                             integer_export=True), encoding="ascii")
    compile_shader(compiler, mut_src, mut_cso)
    wrap_max_case = directory / "valu-int32-a-wrap-max.case"
    wrong = subprocess.run([renderer, backend, "--check-pixel-case", mut_cso, wrap_max_case],
                           capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Replacing v_mul_lo_u32 with addition escaped detection")

    # Program B: and_or_b32, or3_b32, mul_lo_u32, add3_u32
    src_b = directory / "valu_int32_b.hlsl"
    cso_b = directory / "valu_int32_b.cso"
    prog_b = uncompressed_export([
        "v_mul_lo_u32 v12, s44, s45",
        "v_add3_u32 v13, s44, s45, s46",
        "v_and_or_b32 v14, s44, s45, s46",
        "v_or3_b32 v15, s44, s45, s46",
    ])
    src_b.write_text(lower(prog_b, 20, 24, [1, 2, 3, 4], constant_descriptor=64,
                           integer_export=True), encoding="ascii")
    compile_shader(compiler, src_b, cso_b)

    cases_b = []
    for name, inputs, words, status in (
        ("oracle-mismatch", (0x0f, 0xf0, 0xff), (0xe10, 0x1fe, 0xff, 0x0), "mismatch"),
        ("precedence-and-or", (0x0f, 0xf0, 0xff), (0xe10, 0x1fe, 0xff, 0xff), "pass"),
        ("identity-and-or", (0xffffffff, 0x1, 0), (0xffffffff, 0x0, 0x1, 0xffffffff), "pass"),
    ):
        constants = bytearray(256)
        struct.pack_into("<3I", constants, 12, *inputs)
        case = directory / f"valu-int32-b-{name}.case"
        case.write_bytes(constants + struct.pack("<4I", *words) * 64 + b"\x01")
        cases_b.append((case, status))
    run_pixel_batch(renderer, backend, cso_b, cases_b)

    # Program C: shift immediate > 31 on v_lshl_add_u32 (shift 33 yielding 2, verifying & 31u).
    src_c = directory / "valu_int32_shift.hlsl"
    cso_c = directory / "valu_int32_shift.cso"
    prog_c = uncompressed_export([
        "v_lshl_add_u32 v12, s44, 33, s45",
        "v_mov_b32_e32 v13, 0",
        "v_mov_b32_e32 v14, 0",
        "v_mov_b32_e32 v15, 0",
    ])
    src_c.write_text(lower(prog_c, 20, 24, [1, 2, 3, 4], constant_descriptor=64,
                           integer_export=True), encoding="ascii")
    compile_shader(compiler, src_c, cso_c)

    cases_c = []
    # (s44, s45) = (1, 0) -> (1 << (33 & 31)) + 0 = 1 << 1 = 2
    for name, inputs, words, status in (
        ("oracle-mismatch", (1, 0), (999, 0, 0, 0), "mismatch"),
        ("shift-33", (1, 0), (2, 0, 0, 0), "pass"),
    ):
        constants = bytearray(256)
        struct.pack_into("<2I", constants, 12, *inputs)
        case = directory / f"valu-int32-c-{name}.case"
        case.write_bytes(constants + struct.pack("<4I", *words) * 64 + b"\x01")
        cases_c.append((case, status))
    run_pixel_batch(renderer, backend, cso_c, cases_c)


def check_valu_div_fmas_draw(directory, compiler, renderer, backend):
    """Draw v_div_fmas_f32 forms through integer readback target, testing VCC true/false,
    runtime constant inputs, FMA cancellation, pre-scale underflow, and subnormal scale boundary."""
    src = directory / "valu_div_fmas.hlsl"
    cso = directory / "valu_div_fmas.cso"
    prog = uncompressed_export([
        "v_cmp_lt_f32 vcc_lo, 1.0, 0",              # VCC = false
        "v_div_fmas_f32 v12, s44, s45, s46",        # VCC = false, unscaled
        "v_div_fmas_f32 v13, s44, s45, s46 clamp",  # VCC = false, clamped
        "v_cmp_lt_f32 vcc_lo, 0, 1.0",              # VCC = true
        "v_div_fmas_f32 v14, s44, s45, s46",        # VCC = true, scaled
        "v_div_fmas_f32 v15, s44, s45, s46 clamp",  # VCC = true, clamped
    ])
    src.write_text(lower(prog, 20, 24, [1, 2, 3, 4], constant_descriptor=64,
                         integer_export=True), encoding="ascii")
    compile_shader(compiler, src, cso)

    cases = []
    # (name, (s44, s45, s46), (v12, v13, v14, v15), status)
    for name, inputs, words, status in (
        # Oracle mismatch controls:
        ("oracle-mismatch-cancel", (0x3f800001, 0x3f7ffffe, 0xbf800000), (0, 0, 0, 0), "mismatch"),
        ("oracle-mismatch-underflow", (0x00800000, 0x33000000, 0x00000000), (0, 0, 0, 0), "mismatch"),
        ("oracle-mismatch-midpoint", (0x3f800001, 0x3fc00000, 0x8d800000), (0x3fc00002, 0x3f800000, 0x4fc00002, 0x3f800000), "mismatch"),
        # Midpoint single-rounding: exact sum 1.5 + 3*2^-24 - 2^-100 rounds down to 0x3fc00001 / 0x4fc00001
        ("midpoint", (0x3f800001, 0x3fc00000, 0x8d800000), (0x3fc00001, 0x3f800000, 0x4fc00001, 0x3f800000), "pass"),
        # Cancellation: exact product plus addend is -2^-46
        ("cancel", (0x3f800001, 0x3f7ffffe, 0xbf800000), (0xa8800000, 0x00000000, 0xb8800000, 0x00000000), "pass"),
        # An exact product of 2^-150 minus 2^-196 falls short of increasing the odd subnormal.
        ("subnormal-low-word", (0x1a000001, 0x19fffffe, 0x00400001),
         (0x00400001, 0x00400001, 0x10000003, 0x10000003), "pass"),
        ("subnormal-high-word", (0x1a000001, 0x19fffffe, 0x00040001),
         (0x00040001, 0x00040001, 0x0e000030, 0x0e000030), "pass"),
        # Pre-scale underflow: 2^-126 * 2^-25 = 2^-151 before scale, 2^-119 after scale
        ("underflow", (0x00800000, 0x33000000, 0x00000000), (0x00000000, 0x00000000, 0x04000000, 0x04000000), "pass"),
        # Scale boundary: 2^-126 * 0.5 = 2^-127 intermediate subnormal, 2^-95 after scale
        ("boundary", (0x00800000, 0x3f000000, 0x00000000), (0x00400000, 0x00400000, 0x10000000, 0x10000000), "pass"),
        # Normal arithmetic: 2.0 * 0.5 + 1.0 = 2.0
        ("normal", (0x40000000, 0x3f000000, 0x3f800000), (0x40000000, 0x3f800000, 0x50000000, 0x3f800000), "pass"),
    ):
        constants = bytearray(256)
        struct.pack_into("<3I", constants, 12, *inputs)
        case = directory / f"valu-div-fmas-{name}.case"
        case.write_bytes(constants + struct.pack("<4I", *words) * 64 + b"\x01")
        cases.append((case, status))
    run_pixel_batch(renderer, backend, cso, cases)

    # Negative control 1: omit the 2^32 scale on VCC=true.
    mut_src1 = directory / "valu_div_fmas_mut_no_scale.hlsl"
    mut_cso1 = directory / "valu_div_fmas_mut_no_scale.cso"
    hlsl_text = src.read_text(encoding="ascii")
    mut_hlsl1 = hlsl_text.replace("double scale = cond ? 4294967296.0 : 1.0;", "double scale = 1.0;")
    assert mut_hlsl1 != hlsl_text
    mut_src1.write_text(mut_hlsl1, encoding="ascii")
    compile_shader(compiler, mut_src1, mut_cso1)
    underflow_case = directory / "valu-div-fmas-underflow.case"
    wrong1 = subprocess.run([renderer, backend, "--check-pixel-case", mut_cso1, underflow_case],
                            capture_output=True, text=True)
    if wrong1.returncode != 1 or "pixel_mismatch" not in wrong1.stderr:
        raise AssertionError("Omitting 2^32 scale on VCC=true escaped detection")

    # Negative control 2: early float32 rounding / unfused mad.
    mut_src2 = directory / "valu_div_fmas_mut_unfused.hlsl"
    mut_cso2 = directory / "valu_div_fmas_mut_unfused.cso"
    mut_hlsl2 = hlsl_text.replace("return (float)s_sc;",
                                  "return (float)(cond ? (asfloat(0x4f800000u) * mad(s0, s1, s2)) : mad(s0, s1, s2));")
    assert mut_hlsl2 != hlsl_text
    mut_src2.write_text(mut_hlsl2, encoding="ascii")
    compile_shader(compiler, mut_src2, mut_cso2)
    cancel_case = directory / "valu-div-fmas-cancel.case"
    wrong2 = subprocess.run([renderer, backend, "--check-pixel-case", mut_cso2, cancel_case],
                            capture_output=True, text=True)
    if wrong2.returncode != 1 or "pixel_mismatch" not in wrong2.stderr:
        raise AssertionError("Unfused mad / premature float32 rounding escaped detection")

    # Negative control 3: double rounding without 2Sum midpoint correction.
    mut_src3 = directory / "valu_div_fmas_mut_double_round.hlsl"
    mut_cso3 = directory / "valu_div_fmas_mut_double_round.cso"
    mut_hlsl3 = hlsl_text.replace("if (midpoint && err_sc != 0.0)", "if (false)")
    assert mut_hlsl3 != hlsl_text
    mut_src3.write_text(mut_hlsl3, encoding="ascii")
    compile_shader(compiler, mut_src3, mut_cso3)
    for label in ("midpoint", "subnormal-low-word", "subnormal-high-word"):
        midpoint_case = directory / f"valu-div-fmas-{label}.case"
        wrong3 = subprocess.run([renderer, backend, "--check-pixel-case", mut_cso3, midpoint_case],
                                capture_output=True, text=True)
        if wrong3.returncode != 1 or "pixel_mismatch" not in wrong3.stderr:
            raise AssertionError(f"Double-rounding midpoint error escaped detection: {label}")

    # Negative control 4: flush intermediate subnormal to zero before scaling (ftz).
    mut_src4 = directory / "valu_div_fmas_mut_ftz.hlsl"
    mut_cso4 = directory / "valu_div_fmas_mut_ftz.cso"
    mut_hlsl4 = hlsl_text.replace("precise double s = p + (double)s2;",
                                  "precise double s = (abs(p + (double)s2) < asfloat(0x00800000u)) ? 0.0 : (p + (double)s2);")
    assert mut_hlsl4 != hlsl_text
    mut_src4.write_text(mut_hlsl4, encoding="ascii")
    compile_shader(compiler, mut_src4, mut_cso4)
    wrong4 = subprocess.run([renderer, backend, "--check-pixel-case", mut_cso4, underflow_case],
                            capture_output=True, text=True)
    if wrong4.returncode != 1 or "pixel_mismatch" not in wrong4.stderr:
        raise AssertionError("Flushing intermediate subnormal before scale escaped detection")


# v_cvt_f32_u32_e32 and v_cvt_f32_i32_e32 convert one 32-bit word to binary32, reading it unsigned
# or two's-complement. The inputs are shader literals moved through v_mov_b32 bit copies, not
# constant-buffer words (#360): the pairs cover zero and one, the last exactly representable integer
# 2**24, the ties 2**24+1 and 2**25+2, the non-tie neighbors 2**25+1 and 2**25+3 of a midpoint, the
# ties 2**24+3 and 2**25-1 that round up (3 mod 4, so clearing their low bit changes the result), and
# the extreme words 0xFFFFFFFF, 0x80000000 and 0x7FFFFFFF.
CVT_INT_WORDS = (
    (0, 1),
    (2**24, 2**24 + 1),
    (2**25 + 1, 2**25 + 3),
    (2**25 + 2, 0x7FFFFFFF),
    (2**24 + 3, 2**25 - 1),
    (0xFFFFFFFF, 0x80000000),
)


def cvt_int_body(w0, w1, *, swap=False, e64=False, clamp=False):
    """Convert each literal word twice: unsigned into the lower channel of its pair, signed into the
    upper. Swapping the conversions changes every channel whose word has the sign bit set. The e64
    encoding additionally carries the clamp modifier; the default e32 encoding carries neither, so
    asking for clamp without e64 builds the text the translator must refuse."""
    unsigned, signed = ("i32", "u32") if swap else ("u32", "i32")
    encoding = "_e64" if e64 else "_e32"
    suffix = " clamp" if clamp else ""
    body = []
    for dst, word in ((12, w0), (13, w0), (14, w1), (15, w1)):
        op = unsigned if dst % 2 == 0 else signed
        body += [f"v_mov_b32_e32 v{dst}, {word:#x}", f"v_cvt_f32_{op}{encoding} v{dst}, v{dst}{suffix}"]
    return body


def check_cvt_int_f32_draw(directory, compiler, renderer, backend):
    source = directory / "cvt_int_f32.hlsl"
    shader = directory / "cvt_int_f32.cso"
    case = directory / "cvt_int_f32.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    constants = bytes(256)
    # The tie words sit exactly between two representable values and must round to the even one;
    # a half-up or half-away rule would export a different word, so the rows discriminate rounding.
    assert (u32f(2**24 + 1), u32f(2**25 + 2)) == (2.0**24, 2.0**25)
    tie_pair = (2**24 + 3, 2**25 - 1)
    assert tie_pair in CVT_INT_WORDS
    assert (u32f(tie_pair[0]), u32f(tie_pair[1])) == (2.0**24 + 4, 2.0**25)
    assert (u32f(tie_pair[0] & ~1), u32f(tie_pair[1] & ~1)) == (2.0**24 + 2, 2.0**25 - 2)
    for w0, w1 in CVT_INT_WORDS:
        program = uncompressed_export(cvt_int_body(w0, w1))
        source.write_text(translate(program), encoding="ascii")
        compile_shader(compiler, source, shader)
        write_case(case, constants, (u32f(w0), i32f(w0), u32f(w1), i32f(w1)))
        if (w0, w1) == tie_pair:
            # The round-up ties of [2**24, 2**25) must not be read with their low bit cleared:
            # that lands on the representable value 2 below the nearest-even result.
            wrong_case = directory / "cvt_int_f32-tie-wrong.case"
            write_case(wrong_case, constants, (u32f(w0 & ~1), i32f(w0), u32f(w1 & ~1), i32f(w1)))
            run_pixel_batch(renderer, backend, shader, [(wrong_case, "mismatch"), (case, "pass")])
        else:
            subprocess.run(draw, check=True)
    # The last pair's words all have the sign bit set, so swapping the two conversions must be caught.
    w0, w1 = CVT_INT_WORDS[-1]
    assert u32f(w0) != i32f(w0) and u32f(w1) != i32f(w1)
    source.write_text(translate(uncompressed_export(cvt_int_body(w0, w1, swap=True))), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Swapping the unsigned and signed conversions escaped detection")


def check_cvt_int_f32_e64_draw(directory, compiler, renderer, backend):
    source = directory / "cvt_int_f32_e64.hlsl"
    shader = directory / "cvt_int_f32_e64.cso"
    case = directory / "cvt_int_f32_e64.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    constants = bytes(256)
    # The plain e64 encoding performs the same conversion as e32, so the same word pairs and
    # nearest-even references prove the two encodings agree, ties included.
    for w0, w1 in CVT_INT_WORDS:
        program = uncompressed_export(cvt_int_body(w0, w1, e64=True))
        source.write_text(translate(program), encoding="ascii")
        compile_shader(compiler, source, shader)
        write_case(case, constants, (u32f(w0), i32f(w0), u32f(w1), i32f(w1)))
        subprocess.run(draw, check=True)
    # SGPR source row: constant-buffer words with the sign bit set, read directly by the unclamped e64
    # conversions. The words are the normal binary32 patterns of -2.0 and -1.0, so the load carries no
    # NaN or denormal. Reading the unsigned conversion as signed must be caught.
    w0, w1 = 0xC0000000, 0xBF800000
    scalar_constants = bytearray(256)
    struct.pack_into("<2I", scalar_constants, 12, w0, w1)
    scalar_constants = bytes(scalar_constants)
    assert (u32f(w0), u32f(w1)) == (3221225472.0, 3212836864.0)
    assert (i32f(w0), i32f(w1)) == (-1073741824.0, -1082130432.0)
    program = uncompressed_export(["v_cvt_f32_u32_e64 v12, s44", "v_cvt_f32_i32_e64 v13, s44",
                                   "v_cvt_f32_u32_e64 v14, s45", "v_cvt_f32_i32_e64 v15, s45"])
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    sgpr_wrong = directory / "cvt_int_f32_e64-sgpr-wrong.case"
    write_case(sgpr_wrong, scalar_constants, (i32f(w0), i32f(w0), i32f(w1), i32f(w1)))
    write_case(case, scalar_constants, (u32f(w0), i32f(w0), u32f(w1), i32f(w1)))
    run_pixel_batch(renderer, backend, shader, [(sgpr_wrong, "mismatch"), (case, "pass")])
    # Clamp rows: the modifier maps the conversion through min(1.0, max(0.0, .)), computed from
    # the same u32f/i32f references. The words cover the interior 0 and 1, values past 1 (2,
    # 0xFFFFFFFF and 0x7FFFFFFF, which rounds to 2147483648.0 before clamping) and the negative
    # signed words 0xFFFFFFFF and 0x80000000, which clamp to 0.
    for w0, w1 in ((0, 1), (2, 0xFFFFFFFF), (0x80000000, 0x7FFFFFFF)):
        program = uncompressed_export(cvt_int_body(w0, w1, e64=True, clamp=True))
        source.write_text(translate(program), encoding="ascii")
        compile_shader(compiler, source, shader)
        expected = tuple(min(1.0, max(0.0, ref(word)))
                         for word, ref in ((w0, u32f), (w0, i32f), (w1, u32f), (w1, i32f)))
        write_case(case, constants, expected)
        subprocess.run(draw, check=True)
    # Dropping clamp must fail the clamped reference still on disk: without it u32 of 2 exports
    # 2.0 where the reference holds 1.0.
    write_case(case, constants, (1.0, 1.0, 1.0, 0.0))
    source.write_text(translate(uncompressed_export(cvt_int_body(2, 0xFFFFFFFF, e64=True))), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Dropping clamp from the e64 conversion escaped detection")
    # Clamped VGPR row whose positive words have a zero low half: the signed conversion clamps them to
    # 1.0, and any reading that drops bits 16 to 30 gives 0 and clamps to 0.0.
    w0, w1 = 0x00010000, 0x7FFF0000
    program = uncompressed_export(cvt_int_body(w0, w1, e64=True, clamp=True))
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    expected = tuple(min(1.0, max(0.0, ref(word)))
                     for word, ref in ((w0, u32f), (w0, i32f), (w1, u32f), (w1, i32f)))
    assert expected == (1.0, 1.0, 1.0, 1.0)
    masked = tuple(min(1.0, max(0.0, ref(word)))
                   for word, ref in ((w0, u32f), (w0 & 0x8000FFFF, i32f), (w1, u32f), (w1 & 0x8000FFFF, i32f)))
    assert masked == (1.0, 0.0, 1.0, 0.0)
    masked_case = directory / "cvt_int_f32_e64-masked.case"
    write_case(masked_case, constants, masked)
    write_case(case, constants, expected)
    run_pixel_batch(renderer, backend, shader, [(masked_case, "mismatch"), (case, "pass")])
    # The last pair's words all have the sign bit set, so under clamp swapping the two conversions
    # exchanges 1.0 and 0.0 on every channel: u32 clamps to 1 where i32 of the same word clamps to 0.
    w0, w1 = CVT_INT_WORDS[-1]
    program = uncompressed_export(cvt_int_body(w0, w1, e64=True, clamp=True))
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    write_case(case, constants, tuple(min(1.0, max(0.0, ref(word)))
                                      for word, ref in ((w0, u32f), (w0, i32f), (w1, u32f), (w1, i32f))))
    subprocess.run(draw, check=True)
    source.write_text(translate(uncompressed_export(cvt_int_body(w0, w1, e64=True, clamp=True, swap=True))), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Swapping the clamped unsigned and signed conversions escaped detection")


def cvt_u32_ref(word):
    """Reference for v_cvt_u32_f32_e32: the result word for the binary32 bit pattern word."""
    x = struct.unpack("<f", struct.pack("<I", word))[0]
    if math.isnan(x) or x <= 0.0:
        return 0
    if x >= 2.0**32:
        return 0xFFFFFFFF
    return math.trunc(x)


def cvt_i32_ref(word):
    """Reference for v_cvt_i32_f32_e32: the two's-complement result word for the binary32 pattern word."""
    x = struct.unpack("<f", struct.pack("<I", word))[0]
    if math.isnan(x):
        return 0
    if x <= -2.0**31:
        return 0x80000000
    if x >= 2.0**31:
        return 0x7FFFFFFF
    return math.trunc(x) & 0xFFFFFFFF


# (word, u32 result, i32 result), consecutive pairs form the draw rows.
CVT_FLOAT_WORDS = (
    (0x00000000, 0x00000000, 0x00000000),
    (0x80000000, 0x00000000, 0x00000000),
    (0x3fc00000, 0x00000001, 0x00000001),
    (0xc0300000, 0x00000000, 0xfffffffe),
    (0xbf000000, 0x00000000, 0x00000000),
    (0xbf800000, 0x00000000, 0xffffffff),
    (0x7fc00000, 0x00000000, 0x00000000),
    (0xffc00001, 0x00000000, 0x00000000),
    (0x7f800000, 0xffffffff, 0x7fffffff),
    (0xff800000, 0x00000000, 0x80000000),
    (0x4f800000, 0xffffffff, 0x7fffffff),
    (0x4f7fffff, 0xffffff00, 0x7fffffff),
    (0x4f000000, 0x80000000, 0x7fffffff),
    (0x4effffff, 0x7fffff80, 0x7fffff80),
    (0xcf000000, 0x00000000, 0x80000000),
    (0xcf000001, 0x00000000, 0x80000000),
    (0x00000001, 0x00000000, 0x00000000),
    (0x3f7fffff, 0x00000000, 0x00000000),
)


def cvt_float_body(w0, w1, *, swap=False):
    """Convert each literal binary32 word twice: to u32 in the even registers, to i32 in the odd ones.
    Swapping the conversions changes the channels whose u32 and i32 results differ."""
    even, odd = ("i32", "u32") if swap else ("u32", "i32")
    body = []
    for dst, word in ((12, w0), (13, w0), (14, w1), (15, w1)):
        op = even if dst % 2 == 0 else odd
        body += [f"v_mov_b32_e32 v{dst}, {word:#x}", f"v_cvt_{op}_f32_e32 v{dst}, v{dst}"]
    return body


def as_float(word):
    return struct.unpack("<f", struct.pack("<I", word))[0]


def check_cvt_float_int_draw(directory, compiler, renderer, backend):
    assert F32_DENORM_MODE == "preserve"
    source = directory / "cvt_float_int.hlsl"
    shader = directory / "cvt_float_int.cso"
    case = directory / "cvt_float_int.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    constants = bytes(256)

    def write_words(words, path=case):
        # NaN payload guard: the float round trip must keep every expected word.
        for word in words:
            assert struct.unpack("<I", struct.pack("<f", as_float(word)))[0] == word, hex(word)
        write_case(path, constants, tuple(as_float(word) for word in words), integer_rt=True)

    for word, u32, i32 in CVT_FLOAT_WORDS:
        assert (cvt_u32_ref(word), cvt_i32_ref(word)) == (u32, i32), hex(word)
    for (w0, u0, i0), (w1, u1, i1) in zip(CVT_FLOAT_WORDS[0::2], CVT_FLOAT_WORDS[1::2]):
        source.write_text(translate_int(uncompressed_export(cvt_float_body(w0, w1))), encoding="ascii")
        compile_shader(compiler, source, shader)
        write_words((u0, i0, u1, i1))
        subprocess.run(draw, check=True)

    # SGPR source: normal binary32 constants read through the buffer load.
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, 3.75, -3.75, 1e10, -1e10)
    constants = bytes(constants)
    program = uncompressed_export(["v_cvt_u32_f32_e32 v12, s44", "v_cvt_i32_f32_e32 v13, s45",
                                   "v_cvt_u32_f32_e32 v14, s46", "v_cvt_i32_f32_e32 v15, s47"])
    source.write_text(translate_int(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    write_words((3, 0xfffffffd, 0xffffffff, 0x80000000))
    subprocess.run(draw, check=True)

    # Literal source: hex binary32 operands outside the inline-constant set, converted directly. The
    # unsigned values exceed 1.0 and the signed ones are negative below -1.0, so clamping the unsigned
    # operand or taking the magnitude of the signed one changes every channel.
    constants = bytes(256)
    literals = (0x40600000, 0xc0300000, 0x4f7fffff, 0xc0500000)
    program = uncompressed_export([f"v_cvt_u32_f32_e32 v12, {literals[0]:#010x}",
                                   f"v_cvt_i32_f32_e32 v13, {literals[1]:#010x}",
                                   f"v_cvt_u32_f32_e32 v14, {literals[2]:#010x}",
                                   f"v_cvt_i32_f32_e32 v15, {literals[3]:#010x}"])
    source.write_text(translate_int(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    expected = (cvt_u32_ref(literals[0]), cvt_i32_ref(literals[1]),
                cvt_u32_ref(literals[2]), cvt_i32_ref(literals[3]))
    assert expected == (3, 0xfffffffe, 0xffffff00, 0xfffffffd)
    one = 0x3f800000
    mutated = (cvt_u32_ref(one), cvt_i32_ref(literals[1] & 0x7fffffff),
               cvt_u32_ref(one), cvt_i32_ref(literals[3] & 0x7fffffff))
    assert mutated == (1, 2, 1, 3)
    mutated_case = directory / "cvt_float_int-lit-mutated.case"
    write_words(mutated, mutated_case)
    write_words(expected)
    run_pixel_batch(renderer, backend, shader, [(mutated_case, "mismatch"), (case, "pass")])

    # Negative control: swapped conversions against the unswapped expectation. Channels v14 and v15
    # hold -1.0, which converts to 0 unsigned and 0xffffffff signed.
    constants = bytes(256)
    w0, w1 = 0xbf000000, 0xbf800000
    assert cvt_u32_ref(w1) != cvt_i32_ref(w1)
    source.write_text(translate_int(uncompressed_export(cvt_float_body(w0, w1, swap=True))), encoding="ascii")
    compile_shader(compiler, source, shader)
    write_words((0, 0, 0, 0xffffffff))
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Swapping the float-to-word conversions escaped detection")


def apply_f32_mod(word, *, abs_mod=False, neg_mod=False):
    if abs_mod:
        word &= 0x7fffffff
    if neg_mod:
        word ^= 0x80000000
    return word


CVT_FLOAT_E64_PLAIN_BODY = [
    "v_mov_b32_e32 v18, s45",
    "v_cvt_u32_f32_e64 v12, s44",
    "v_cvt_i32_f32_e64 v13, s44",
    "v_cvt_u32_f32_e64 v14, v18",
    "v_cvt_i32_f32_e64 v15, v18",
]

CVT_FLOAT_E64_U32_MOD_BODY = [
    "v_mov_b32_e32 v18, s47",
    "v_cvt_u32_f32_e64 v12, |s44|",
    "v_cvt_u32_f32_e64 v13, -s45",
    "v_cvt_u32_f32_e64 v14, -|s46|",
    "v_cvt_u32_f32_e64 v15, |v18|",
]

CVT_FLOAT_E64_I32_MOD_BODY = [
    "v_mov_b32_e32 v18, s47",
    "v_cvt_i32_f32_e64 v12, |s44|",
    "v_cvt_i32_f32_e64 v13, -s45",
    "v_cvt_i32_f32_e64 v14, -|s46|",
    "v_cvt_i32_f32_e64 v15, -v18",
]

CVT_FLOAT_E64_MOD_SETS = (
    # Negative fractions: dropping any abs or neg modifier changes the output word.
    (0xc0300000, 0xc0300000, 0xc0300000, 0xbfc00000),  # -2.75, -2.75, -2.75, -1.5
    # Positive fractions and 2^31 saturation / near-limit boundaries.
    (0x40200000, 0x4f000000, 0x4f000000, 0x4effffff),  # +2.5, +2^31, +2^31, +(2^31-128)
    # Negative saturation boundaries and unsigned high range after abs/neg.
    (0xcf7fffff, 0xcf000000, 0xcf800000, 0xcf800000),  # -(2^32-256), -2^31, -2^32, -2^32
    # Signed infinities and negative zero.
    (0xff800000, 0xff800000, 0x7f800000, 0x80000000),  # -inf, -inf, +inf, -0.0
    # Quiet/signaling NaNs of both signs and smallest negative subnormal -> all 0.
    (0x7fc00000, 0xffc00001, 0x7f800001, 0x80000001),
)


def check_cvt_float_int_e64_draw(directory, compiler, renderer, backend):
    assert F32_DENORM_MODE == "preserve"
    source = directory / "cvt_float_int_e64.hlsl"
    shader = directory / "cvt_float_int_e64.cso"

    def make_case(name, input_words, expected_words):
        path = directory / f"cvt_e64_{name}.case"
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *input_words)
        for word in expected_words:
            assert struct.unpack("<I", struct.pack("<f", as_float(word)))[0] == word, hex(word)
        write_case(path, bytes(constants), tuple(as_float(word) for word in expected_words), integer_rt=True)
        return path

    # Plain e64 parity on all 18 CVT_FLOAT_WORDS through SGPR (s44) and VGPR (v18).
    source.write_text(translate_int(uncompressed_export(CVT_FLOAT_E64_PLAIN_BODY)), encoding="ascii")
    compile_shader(compiler, source, shader)
    cases = []
    for idx, ((w0, u0, i0), (w1, u1, i1)) in enumerate(zip(CVT_FLOAT_WORDS[0::2], CVT_FLOAT_WORDS[1::2])):
        cases.append((make_case(f"plain_{idx}", (w0, w1, 0, 0), (u0, i0, u1, i1)), "pass"))
    # Bitwise negative control: flipping a single bit in an expected word must report mismatch.
    cases.append((make_case("plain_bitflip", (0x4effffff, 0xc0300000, 0, 0),
                            (0x7fffff81, 0x7fffff80, 0x00000000, 0xfffffffe)), "mismatch"))
    run_pixel_batch(renderer, backend, shader, cases)

    # Swapping u32 and i32 in the plain e64 shader must fail on (-0.5, -1.0).
    swapped = [
        line.replace("v_cvt_u32_f32_e64", "TMP").replace("v_cvt_i32_f32_e64", "v_cvt_u32_f32_e64")
        .replace("TMP", "v_cvt_i32_f32_e64")
        for line in CVT_FLOAT_E64_PLAIN_BODY
    ]
    source.write_text(translate_int(uncompressed_export(swapped)), encoding="ascii")
    compile_shader(compiler, source, shader)
    swap_case = make_case("plain_swap", (0xbf000000, 0xbf800000, 0, 0), (0, 0, 0, 0xffffffff))
    wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, swap_case], capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Swapping u32 and i32 on e64 float-to-word conversions escaped detection")

    # Source abs/neg modifier draws and shader-mutation negative controls for both u32 and i32.
    mod_specs = (
        ("u32", CVT_FLOAT_E64_U32_MOD_BODY, cvt_u32_ref,
         ((True, False), (False, True), (True, True), (True, False)),
         (("|s44|", "s44", "drop abs on u32 SGPR"),
          ("-s45", "s45", "drop neg on u32 SGPR"),
          ("-|s46|", "|s46|", "drop neg on u32 -|SGPR|"),
          ("-|s46|", "-s46", "drop abs on u32 -|SGPR|"),
          ("|v18|", "v18", "drop abs on u32 VGPR"))),
        ("i32", CVT_FLOAT_E64_I32_MOD_BODY, cvt_i32_ref,
         ((True, False), (False, True), (True, True), (False, True)),
         (("|s44|", "s44", "drop abs on i32 SGPR"),
          ("-s45", "s45", "drop neg on i32 SGPR"),
          ("-|s46|", "|s46|", "drop neg on i32 -|SGPR|"),
          ("-|s46|", "-s46", "drop abs on i32 -|SGPR|"),
          ("-v18", "v18", "drop neg on i32 VGPR"))),
    )
    for kind, body, ref_fn, chan_mods, mutations in mod_specs:
        source.write_text(translate_int(uncompressed_export(body)), encoding="ascii")
        compile_shader(compiler, source, shader)
        batch = []
        set0_expected = None
        for idx, words in enumerate(CVT_FLOAT_E64_MOD_SETS):
            expected = tuple(
                ref_fn(apply_f32_mod(w, abs_mod=a, neg_mod=n))
                for w, (a, n) in zip(words, chan_mods)
            )
            if idx == 0:
                set0_expected = expected
            batch.append((make_case(f"{kind}_mod_{idx}", words, expected), "pass"))
        # Bitwise negative control on the saturation boundary set: flipping bit 0 must fail.
        boundary_expected = tuple(
            ref_fn(apply_f32_mod(w, abs_mod=a, neg_mod=n))
            for w, (a, n) in zip(CVT_FLOAT_E64_MOD_SETS[2], chan_mods)
        )
        flipped = (boundary_expected[0] ^ 1, *boundary_expected[1:])
        batch.append((make_case(f"{kind}_bitflip", CVT_FLOAT_E64_MOD_SETS[2], flipped), "mismatch"))
        run_pixel_batch(renderer, backend, shader, batch)

        # Shader-mutation negative controls against set 0 (-2.75, -2.75, -2.75, -1.5).
        set0_case = make_case(f"{kind}_mut_ref", CVT_FLOAT_E64_MOD_SETS[0], set0_expected)
        for old, new, label in mutations:
            mutated = [line.replace(old, new) for line in body]
            assert mutated != body, label
            source.write_text(translate_int(uncompressed_export(mutated)), encoding="ascii")
            compile_shader(compiler, source, shader)
            wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, set0_case],
                                   capture_output=True, text=True)
            if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
                raise AssertionError(f"Modifier mutation ({label}) escaped detection")


def cvt_flr_i32_ref(word):
    """Reference for finite in-range v_cvt_flr_i32_f32: floor(x) in [-2^31, 2^31)."""
    x = struct.unpack("<f", struct.pack("<I", word))[0]
    assert math.isfinite(x) and -2.0**31 <= x < 2.0**31, hex(word)
    return math.floor(x) & 0xFFFFFFFF


def cvt_rpi_i32_ref(word):
    """Reference for finite in-range v_cvt_rpi_i32_f32: floor(x + 0.5) in [-2^31, 2^31)."""
    x = struct.unpack("<f", struct.pack("<I", word))[0]
    assert math.isfinite(x) and -2.0**31 <= x < 2.0**31, hex(word)
    return math.floor(x + 0.5) & 0xFFFFFFFF


def cvt_flr_rpi_pair_body(w0, w1, *, e64):
    """Build a proven-constant SGPR/VGPR shader converting two finite in-range binary32 words."""
    suf = "_e64" if e64 else "_e32"
    return [
        f"s_mov_b32 s44, {w0:#010x}",
        f"s_mov_b32 s45, {w1:#010x}",
        "v_mov_b32_e32 v18, s45",
        f"v_cvt_flr_i32_f32{suf} v12, s44",
        f"v_cvt_rpi_i32_f32{suf} v13, s44",
        f"v_cvt_flr_i32_f32{suf} v14, v18",
        f"v_cvt_rpi_i32_f32{suf} v15, v18",
    ]


def cvt_flr_rpi_mod_body(kind, words):
    """Build a proven-constant e64 modifier shader (|s44|, -s45, -|s46|, -v18)."""
    w0, w1, w2, w3 = words
    return [
        f"s_mov_b32 s44, {w0:#010x}",
        f"s_mov_b32 s45, {w1:#010x}",
        f"s_mov_b32 s46, {w2:#010x}",
        f"s_mov_b32 s47, {w3:#010x}",
        "v_mov_b32_e32 v18, s47",
        f"v_cvt_{kind}_f32_e64 v12, |s44|",
        f"v_cvt_{kind}_f32_e64 v13, -s45",
        f"v_cvt_{kind}_f32_e64 v14, -|s46|",
        f"v_cvt_{kind}_f32_e64 v15, -v18",
    ]


# Proven finite in-range binary32 pairs covering zeros, positive/negative halves and fractions,
# single-ULP half boundaries (0x3effffff = 0.5-2^-25, 0xbf000001 = -0.5-2^-24, 0xbeffffff = -0.5+2^-25),
# large odd integer 2^23+1 (0x4b000001 = 8388609.0), negative min subnormal (0x80000001 -> flr=-1, rpi=0),
# and in-range signed i32 boundaries (2^31 - 128 and -2^31).
CVT_FLR_RPI_PAIRS = (
    (0x00000000, 0x80000000),  # +0.0, -0.0
    (0x3f000000, 0xbf000000),  # +0.5, -0.5
    (0x3fc00000, 0xbfc00000),  # +1.5, -1.5
    (0x40200000, 0xc0200000),  # +2.5, -2.5
    (0x3e800000, 0xbe800000),  # +0.25, -0.25
    (0x3f400000, 0xbf400000),  # +0.75, -0.75
    (0x3effffff, 0xbf000001),  # 0.5 - 2^-25 (rpi=0), -0.5 - 2^-24 (rpi=-1)
    (0xbeffffff, 0x4b000001),  # -0.5 + 2^-25 (rpi=0, flr=-1), 2^23 + 1 = 8388609.0
    (0x80000001, 0x00000001),  # -min_subnormal (flr=-1, rpi=0), +min_subnormal (flr=0, rpi=0)
    (0x4effffff, 0xcf000000),  # 2^31 - 128, -2^31
)

CVT_FLR_RPI_MOD_SETS = (
    (0xc0200000, 0x3fc00000, 0xc0300000, 0xbf400000),  # -2.5, +1.5, -2.75, -0.75
    (0xbf400000, 0x3effffff, 0xbf000001, 0x4b000001),  # -0.75, 0.5-2^-25, -0.5-2^-24, 8388609.0
    (0xc0400000, 0x4f000000, 0x4f000000, 0x80000001),  # |-3.0|=+3, -(+2^31)=-2^31, -|+2^31|=-2^31, -(-min_subnormal)
)


def check_cvt_flr_rpi_draw(directory, compiler, renderer, backend):
    assert F32_DENORM_MODE == "preserve"
    source = directory / "cvt_flr_rpi.hlsl"
    shader = directory / "cvt_flr_rpi.cso"

    def make_case(name, expected_words):
        path = directory / f"cvt_flr_rpi_{name}.case"
        for word in expected_words:
            assert struct.unpack("<I", struct.pack("<f", as_float(word)))[0] == word, hex(word)
        write_case(path, bytes(256), tuple(as_float(word) for word in expected_words), integer_rt=True)
        return path

    # Plain e32 and e64 across all 10 proven-in-range pairs (20 finite binary32 values via SGPR/VGPR).
    for tag, e64 in (("e32", False), ("e64", True)):
        for idx, (w0, w1) in enumerate(CVT_FLR_RPI_PAIRS):
            body = cvt_flr_rpi_pair_body(w0, w1, e64=e64)
            source.write_text(translate_int(uncompressed_export(body)), encoding="ascii")
            compile_shader(compiler, source, shader)
            expected = (cvt_flr_i32_ref(w0), cvt_rpi_i32_ref(w0), cvt_flr_i32_ref(w1), cvt_rpi_i32_ref(w1))
            batch = [(make_case(f"{tag}_{idx}", expected), "pass")]
            if idx == 6:
                bad_exp = (expected[0] ^ 1, *expected[1:])
                batch.append((make_case(f"{tag}_bitflip", bad_exp), "mismatch"))
            run_pixel_batch(renderer, backend, shader, batch)

    # Shader-mutation controls: swapping flr/rpi and replacing flr with truncating i32 must fail.
    base_half = cvt_flr_rpi_pair_body(0x3f000000, 0xbf000000, e64=False)
    ref_case = make_case("swap_ref", (cvt_flr_i32_ref(0x3f000000), cvt_rpi_i32_ref(0x3f000000),
                                      cvt_flr_i32_ref(0xbf000000), cvt_rpi_i32_ref(0xbf000000)))
    for label, mutated in (
        ("swap flr and rpi e32", [line.replace("v_cvt_flr_i32_f32_e32", "TMP")
                                  .replace("v_cvt_rpi_i32_f32_e32", "v_cvt_flr_i32_f32_e32")
                                  .replace("TMP", "v_cvt_rpi_i32_f32_e32") for line in base_half]),
        ("replace flr with trunc i32 e32", [line.replace("v_cvt_flr_i32_f32_e32", "v_cvt_i32_f32_e32")
                                            for line in base_half]),
    ):
        source.write_text(translate_int(uncompressed_export(mutated)), encoding="ascii")
        compile_shader(compiler, source, shader)
        wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, ref_case],
                               capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Mutation ({label}) escaped detection")

    # Source abs/neg modifier draws and modifier-drop mutations for both flr and rpi e64.
    chan_mods = ((True, False), (False, True), (True, True), (False, True))
    mutations = (
        ("|s44|", "s44", "drop abs on SGPR"),
        ("-s45", "s45", "drop neg on SGPR"),
        ("-|s46|", "|s46|", "drop neg on -|SGPR|"),
        ("-|s46|", "-s46", "drop abs on -|SGPR|"),
        ("-v18", "v18", "drop neg on VGPR"),
    )
    for kind, ref_fn in (("flr_i32", cvt_flr_i32_ref), ("rpi_i32", cvt_rpi_i32_ref)):
        set0_body = cvt_flr_rpi_mod_body(kind, CVT_FLR_RPI_MOD_SETS[0])
        set0_expected = None
        for idx, words in enumerate(CVT_FLR_RPI_MOD_SETS):
            body = cvt_flr_rpi_mod_body(kind, words)
            source.write_text(translate_int(uncompressed_export(body)), encoding="ascii")
            compile_shader(compiler, source, shader)
            expected = tuple(ref_fn(apply_f32_mod(w, abs_mod=a, neg_mod=n)) for w, (a, n) in zip(words, chan_mods))
            batch = [(make_case(f"{kind}_mod_{idx}", expected), "pass")]
            if idx == 0:
                set0_expected = expected
                batch.append((make_case(f"{kind}_mod_bitflip", (expected[0] ^ 1, *expected[1:])), "mismatch"))
            run_pixel_batch(renderer, backend, shader, batch)
        set0_case = make_case(f"{kind}_mut_ref", set0_expected)
        for old, new, label in mutations:
            mutated = [line.replace(old, new) for line in set0_body]
            assert mutated != set0_body, label
            source.write_text(translate_int(uncompressed_export(mutated)), encoding="ascii")
            compile_shader(compiler, source, shader)
            wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, set0_case],
                                   capture_output=True, text=True)
            if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
                raise AssertionError(f"{kind} modifier mutation ({label}) escaped detection")


def check_int_minmax_draws(directory, compiler, renderer, backend):
    source = directory / "int_minmax.hlsl"
    shader = directory / "int_minmax.cso"

    def order_key(value, signed):
        return struct.unpack("<i" if signed else "<I", struct.pack("<f", value))[0]

    def pick_pair(a, b):
        return (
            min(a, b, key=lambda v: order_key(v, True)),
            max(a, b, key=lambda v: order_key(v, True)),
            min(a, b, key=lambda v: order_key(v, False)),
            max(a, b, key=lambda v: order_key(v, False)),
        )

    def pick_triple(a, b, c, signed):
        ordered = sorted((a, b, c), key=lambda v: order_key(v, signed))
        tied = sorted((c, b, b), key=lambda v: order_key(v, signed))[1]
        return ordered[0], ordered[2], ordered[1], tied

    # Exact binary16 floats whose 32-bit patterns reverse order across signed, unsigned, and float.
    pair_cases = {}
    for encoding in ("e32", "e64"):
        source.write_text(translate(int_minmax_pair(encoding)), encoding="ascii")
        compile_shader(compiler, source, shader)
        cases = []
        for idx, (a, b) in enumerate(((-2.0, 1.0), (-2.0, -1.0))):
            constants = bytearray(256)
            struct.pack_into("<4f", constants, 12, a, b, 0.0, 0.0)
            case = directory / f"int_minmax-pair-{encoding}-{idx}.case"
            case.write_bytes(constants + struct.pack("<4f", *pick_pair(a, b)) * 64)
            cases.append((case, "pass"))
        run_pixel_batch(renderer, backend, shader, cases)
        pair_cases[encoding] = cases

    swapped = [line.replace("v_min_i32_e32", "v_min_u32_e32") for line in int_minmax_pair("e32")]
    source.write_text(translate(swapped), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, pair_cases["e32"][0][0]],
                           capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Replacing v_min_i32 with v_min_u32 escaped detection")

    triple_cases = {}
    for kind in ("i32", "u32"):
        source.write_text(translate(int_minmax_triple(kind)), encoding="ascii")
        compile_shader(compiler, source, shader)
        cases = []
        for idx, (a, b, c) in enumerate(((-2.0, -1.0, 1.0), (0.5, -0.5, 4.0))):
            constants = bytearray(256)
            struct.pack_into("<4f", constants, 12, a, b, c, 0.0)
            case = directory / f"int_minmax-triple-{kind}-{idx}.case"
            case.write_bytes(constants + struct.pack("<4f", *pick_triple(a, b, c, kind == "i32")) * 64)
            cases.append((case, "pass"))
        run_pixel_batch(renderer, backend, shader, cases)
        triple_cases[kind] = cases

    wrong_med = [line.replace("v_med3_i32", "v_med3_u32") for line in int_minmax_triple("i32")]
    source.write_text(translate(wrong_med), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, triple_cases["i32"][0][0]],
                           capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Replacing v_med3_i32 with v_med3_u32 escaped detection")


def check_cvt_int_draws(directory, compiler, renderer, backend):
    source = directory / "cvt_int.hlsl"
    shader = directory / "cvt_int.cso"
    compile_shader = [compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source]

    # Two distinct constant sets: Set 1 tests negative numbers and values > 1.0;
    # Set 2 tests boundary zero, one, and an exact integer (1000).
    sets = [
        ((-20, 15, -50, 200), [-20.0, 15.0, -50.0, 200.0], [-20.0, 1.0, 0.0, 200.0]),
        ((0, 1, 1000, 0), [0.0, 1.0, 1000.0, 0.0], [0.0, 1.0, 1.0, 0.0]),
    ]
    for _, e32_vals, e64_vals in sets:
        for x in (*e32_vals, *e64_vals):
            assert half_rtz(x) == x, x

    # e32: unclamped conversions of SGPRs and VGPR.
    source.write_text(translate(cvt_int_fixture("e32")), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    e32_cases = []
    for idx, ((c0, c1, c2, c3), expected, _) in enumerate(sets):
        constants = bytearray(256)
        struct.pack_into("<4i", constants, 12, c0, c1, c2, c3)
        case = directory / f"cvt_int-e32-{idx}.case"
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        e32_cases.append((case, "pass"))
    run_pixel_batch(renderer, backend, shader, e32_cases)

    # Negative control: swapping signed v_cvt_f32_i32 with unsigned v_cvt_f32_u32 must fail.
    swapped = [line.replace("v_cvt_f32_i32", "v_cvt_f32_u32") for line in cvt_int_fixture("e32")]
    source.write_text(translate(swapped), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, e32_cases[0][0]],
                           capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Replacing v_cvt_f32_i32 with v_cvt_f32_u32 escaped detection")

    # e64: includes clamp modifier (15.0 clamped -> 1.0; -50.0 clamped -> 0.0).
    source.write_text(translate(cvt_int_fixture("e64")), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    e64_cases = []
    for idx, ((c0, c1, c2, c3), _, expected) in enumerate(sets):
        constants = bytearray(256)
        struct.pack_into("<4i", constants, 12, c0, c1, c2, c3)
        case = directory / f"cvt_int-e64-{idx}.case"
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        e64_cases.append((case, "pass"))
    run_pixel_batch(renderer, backend, shader, e64_cases)

    # Negative control: dropping clamp on e64 must fail.
    unclamped = [line.replace(" clamp", "") for line in cvt_int_fixture("e64")]
    source.write_text(translate(unclamped), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    wrong_clamp = subprocess.run([renderer, backend, "--check-pixel-case", shader, e64_cases[0][0]],
                                 capture_output=True, text=True)
    if wrong_clamp.returncode != 1 or "pixel_mismatch" not in wrong_clamp.stderr:
        raise AssertionError("Dropping clamp on v_cvt_f32_u32_e64/v_cvt_f32_i32_e64 escaped detection")

    # Direct literal operands with and without clamp.
    literal_case = directory / "cvt_int-lit.case"
    draw = [renderer, backend, "--check-pixel-case", shader, literal_case]
    literal_expected = [-7.0, 25.0, 0.0, 1.0]
    source.write_text(translate(cvt_int_literals()), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    literal_case.write_bytes(bytes(256) + struct.pack("<4f", *literal_expected) * 64)
    subprocess.run(draw, check=True)

    # Negative control: swapping signed literal to unsigned must report pixel mismatch.
    mutated_lit = [line.replace("v_cvt_f32_i32_e32 v12, -7", "v_cvt_f32_u32_e32 v12, -7")
                   for line in cvt_int_literals()]
    source.write_text(translate(mutated_lit), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    wrong_lit = subprocess.run(draw, capture_output=True, text=True)
    if wrong_lit.returncode != 1 or "pixel_mismatch" not in wrong_lit.stderr:
        raise AssertionError("Literal signed-to-unsigned conversion mismatch escaped detection")

    # Negative control: dropping clamp on negative literal must report pixel mismatch.
    mutated_clamp = [line.replace("v_cvt_f32_i32_e64 v14, -100 clamp", "v_cvt_f32_i32_e64 v14, -100")
                     for line in cvt_int_literals()]
    source.write_text(translate(mutated_clamp), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    wrong_lit_clamp = subprocess.run(draw, capture_output=True, text=True)
    if wrong_lit_clamp.returncode != 1 or "pixel_mismatch" not in wrong_lit_clamp.stderr:
        raise AssertionError("Literal clamp removal escaped detection")


def check_frexp_ldexp_draws(directory, compiler, renderer, backend):
    source = directory / "frexp_ldexp.hlsl"
    shader = directory / "frexp_ldexp.cso"

    source.write_text(translate(frexp_mant_values()), encoding="ascii")
    compile_shader(compiler, source, shader)
    frexp_cases = []
    for idx, (a, b, c, d) in enumerate(((3.0, 6.0, -0.25, -3.0), (-0.0, 0.0, 0.75, 12.0))):
        constants = bytearray(256)
        struct.pack_into("<4f", constants, 12, a, b, c, d)
        expected = (
            math.frexp(a)[0],
            math.frexp(-abs(b))[0],
            math.frexp(abs(c))[0],
            min(1.0, max(0.0, math.frexp(d)[0])),
        )
        case = directory / f"frexp-{idx}.case"
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        frexp_cases.append((case, "pass"))
    run_pixel_batch(renderer, backend, shader, frexp_cases)

    wrong_frexp = [line.replace("v_frexp_mant_f32_e32", "v_mov_b32_e32") for line in frexp_mant_values()]
    source.write_text(translate(wrong_frexp), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, frexp_cases[0][0]],
                           capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Replacing v_frexp_mant_f32 with v_mov_b32 escaped detection")

    source.write_text(translate(ldexp_values()), encoding="ascii")
    compile_shader(compiler, source, shader)
    ldexp_cases = []
    for idx, (a, exp_i, c, d) in enumerate(((0.75, 3, 6.0, 0.25), (-0.0, 10, -2.0, -0.5), (1.5, -2, 0.5, 0.0625))):
        constants = bytearray(256)
        struct.pack_into("<fiff", constants, 12, a, exp_i, c, d)
        expected = (
            math.ldexp(a, exp_i),
            math.ldexp(-abs(a), 2),
            math.ldexp(c, -2),
            min(1.0, max(0.0, math.ldexp(d, 3))),
        )
        case = directory / f"ldexp-{idx}.case"
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        ldexp_cases.append((case, "pass"))
    run_pixel_batch(renderer, backend, shader, ldexp_cases)

    wrong_ldexp = [line.replace("v_ldexp_f32 v14, s46, -2", "v_ldexp_f32 v14, s46, 2") for line in ldexp_values()]
    source.write_text(translate(wrong_ldexp), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, ldexp_cases[0][0]],
                           capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Inverting the v_ldexp_f32 exponent sign escaped detection")


MUL24_BODY = [
    "v_mov_b32_e32 v28, s45",
    "v_mov_b32_e32 v29, s46",
    "v_mov_b32_e32 v30, s47",
    "v_mov_b32_e32 v31, s44",
    "v_mul_i32_i24_e32 v20, s44, v28",
    "v_mul_i32_i24_e64 v21, s46, s47",
    "v_mul_u32_u24_e32 v22, s45, v29",
    "v_mul_u32_u24_e64 v23, s47, s44",
    "v_mul_hi_i32_i24_e32 v24, s44, v28",
    "v_mul_hi_i32_i24_e64 v25, s46, s47",
    "v_mul_hi_u32_u24_e32 v26, s45, v29",
    "v_mul_hi_u32_u24_e64 v27, s47, s44",
    "v_mov_b32_e32 v16, 0x2468ace1",
    *[
        line
        for reg in ("v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27")
        for line in (
            "v_lshlrev_b32_e32 v17, 5, v16",
            "v_lshrrev_b32_e32 v18, 27, v16",
            "v_or_b32_e32 v16, v17, v18",
            f"v_xor_b32_e32 v16, {reg}, v16",
            "v_add_nc_u32_e32 v16, 0x9e3779b9, v16",
        )
    ],
    "v_cvt_f32_ubyte0_e32 v12, v16",
    "v_cvt_f32_ubyte1_e32 v13, v16",
    "v_cvt_f32_ubyte2_e32 v14, v16",
    "v_cvt_f32_ubyte3_e32 v15, v16",
]

MUL24_SETS = (
    (0xab800000, 0xcd000200, 0xef7fffff, 0x12800000),
    (0xff800000, 0xee800000, 0xddffffff, 0xccffffff),
    (0x7f000000, 0x80ffffff, 0x99000001, 0xaa000000),
    (0x34ffffff, 0x56000001, 0x78123456, 0x9a654321),
    (0xdeadbeef, 0xcafebabe, 0x00abcdef, 0xff112233),
)


def mul24_values():
    return with_constants(MUL24_BODY)


def mul24_reference(words, *, mode="exact"):
    """Independent 48-bit Python integer reference for 24-bit integer multiplies."""
    mask32 = 0xffffffff

    def u24(x):
        return (x & mask32) if mode == "no_mask_24" else (x & 0xffffff)

    def s24(x):
        if mode == "no_mask_24":
            x &= mask32
            return x - 0x100000000 if (x & 0x80000000) else x
        v = x & 0xffffff
        return v - 0x1000000 if (v & 0x800000) else v

    def mul_i24(a, b):
        if mode == "i24_as_u24":
            return (u24(a) * u24(b)) & mask32
        return (s24(a) * s24(b)) & mask32

    def mul_u24(a, b):
        if mode == "u24_as_i24":
            return (s24(a) * s24(b)) & mask32
        return (u24(a) * u24(b)) & mask32

    def mul_hi_i24(a, b):
        if mode == "hi_i24_as_u24":
            return ((u24(a) * u24(b)) >> 32) & mask32
        prod = s24(a) * s24(b)
        if mode == "hi_i24_bad_lo0" and prod < 0 and (prod & mask32) == 0:
            return ((prod >> 32) - 1) & mask32
        return (prod >> 32) & mask32

    def mul_hi_u24(a, b):
        if mode == "hi_u24_as_i24":
            return ((s24(a) * s24(b)) >> 32) & mask32
        return ((u24(a) * u24(b)) >> 32) & mask32

    s44, s45, s46, s47 = words
    results = (
        mul_i24(s44, s45), mul_i24(s46, s47),
        mul_u24(s45, s46), mul_u24(s47, s44),
        mul_hi_i24(s44, s45), mul_hi_i24(s46, s47),
        mul_hi_u24(s45, s46), mul_hi_u24(s47, s44),
    )
    acc = 0x2468ace1
    for r in results:
        acc = (((acc << 5) | (acc >> 27)) & mask32) ^ (r & mask32)
        acc = (acc + 0x9e3779b9) & mask32
    return [float((acc >> (8 * k)) & 0xff) for k in range(4)]


def check_mul24_draws(directory, compiler, renderer, backend):
    source = directory / "mul24.hlsl"
    shader = directory / "mul24.cso"
    source.write_text(translate(mul24_values()), encoding="ascii")
    compile_shader(compiler, source, shader)

    cases = []

    def add_case(words, expected, status):
        path = directory / f"mul24-{len(cases)}.case"
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *words)
        path.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        cases.append((path, status))

    for words in MUL24_SETS:
        add_case(words, mul24_reference(words), "pass")
    for mode in ("no_mask_24", "i24_as_u24", "u24_as_i24", "hi_i24_as_u24", "hi_i24_bad_lo0", "hi_u24_as_i24"):
        words = next(w for w in MUL24_SETS if mul24_reference(w, mode=mode) != mul24_reference(w))
        add_case(words, mul24_reference(words, mode=mode), "mismatch")
    # End batch with a passing reference after negative controls.
    add_case(MUL24_SETS[0], mul24_reference(MUL24_SETS[0]), "pass")

    run_pixel_batch(renderer, backend, shader, cases)

    # Mutated shader control: replacing v_mul_hi_i32_i24_e32 with v_mul_hi_u32_u24_e32 must mismatch.
    wrong_prog = [
        line.replace("v_mul_hi_i32_i24_e32 v24, s44, v28", "v_mul_hi_u32_u24_e32 v24, s44, v28")
        for line in mul24_values()
    ]
    source.write_text(translate(wrong_prog), encoding="ascii")
    compile_shader(compiler, source, shader)
    mut_case = directory / "mul24-mut.case"
    constants = bytearray(256)
    struct.pack_into("<4I", constants, 12, *MUL24_SETS[0])
    mut_case.write_bytes(constants + struct.pack("<4f", *mul24_reference(MUL24_SETS[0])) * 64)
    wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, mut_case], capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Replacing v_mul_hi_i32_i24_e32 with v_mul_hi_u32_u24_e32 escaped detection")


def check_three_way_draw(directory, compiler, renderer, backend):
    source = directory / "three_way.hlsl"
    shader = directory / "three_way.cso"
    source.write_text(translate(three_way()), encoding="ascii")
    compile_shader(compiler, source, shader)
    cases = []
    for idx, constants in enumerate(((.25, .75, .375, .5), (.625, .125, -.3125, .5))):
        case = three_way_case(directory, *constants, name=f"three_way-{idx}.case")
        cases.append((case, "pass"))
    run_pixel_batch(renderer, backend, shader, cases)
    draw = [renderer, backend, "--check-pixel-case", shader, cases[-1][0]]
    without_abs = [line.replace("|s46|", "s46") for line in three_way()]
    source.write_text(translate(without_abs), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("The renderer failed to detect the missing min3 absolute value")
    source.write_text(translate([line.replace("v_med3_f32 v12", "v_min3_f32 v12") for line in three_way()]),
                      encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("A minimum replaced the median without a mismatch")
    return 0


def check_buffer_color_draw(directory, compiler, renderer, backend):
    source, shader = directory / "buffer.hlsl", directory / "buffer.cso"
    for rg in (False, True):
        source.write_text(translate_buffer_color(buffer_color(rg)), encoding="ascii")
        compile_shader(compiler, source, shader)
        if not rg:
            cases = []
            for idx, (red, green) in enumerate(((0.25, 0.75), (-0.5, 0.125))):
                constants = bytearray(256)
                struct.pack_into("<4f", constants, 16, green, 2, 3, red)
                expected = struct.pack("<4f", red, green, green, red) * 64
                case = directory / f"buffer-{idx}.case"
                case.write_bytes(constants + expected)
                cases.append((case, "pass"))
            struct.pack_into("<f", constants, 28, green)
            wrong = directory / "buffer-wrong.case"
            wrong.write_bytes(constants + expected)
            cases.insert(-1, (wrong, "mismatch"))
            run_pixel_batch(renderer, backend, shader, cases)
            continue
        case = directory / "buffer-rg.case"
        draw = [renderer, backend, "--check-pixel-case-rg", shader, case]
        for red, green in ((0.25, 0.75), (-0.5, 0.125)):
            constants = bytearray(256)
            struct.pack_into("<4f", constants, 16, green, 2, 3, red)
            expected = struct.pack("<4f", red, green, -1, -1) * 64
            case.write_bytes(constants + expected)
            result = subprocess.run(draw)
            if result.returncode:
                return result.returncode
        struct.pack_into("<f", constants, 28, green)
        case.write_bytes(constants + expected)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError("Wrong buffer color escaped detection")
# Powers of two, subnormal endpoints, the largest finite value, and special inputs.
FREXP_EXP_CASES = (
    (0x3F800000, 1, 0xBE800000, -1),
    (0x00000001, -148, 0x007FFFFF, -126),
    (0x00800000, -125, 0x7F7FFFFF, 128),
    (0x00000000, 0, 0x80000000, 0),
    (0x7F800000, 0, 0xFF800000, 0),
    (0x7FC12345, 0, 0xFFC12345, 0),
)


def check_frexp_exp_draws(directory, compiler, renderer, backend):
    source = directory / "frexp_exp.hlsl"
    shader = directory / "frexp_exp.cso"
    case = directory / "frexp_exp.case"
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    source.write_text(translate(frexp_exp_values()), encoding="ascii")
    subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source], check=True)

    def set_case(a, exponent_a, b, exponent_b):
        constants = bytearray(256)
        struct.pack_into("<IiIi", constants, 12, a, exponent_a, b, exponent_b)
        case.write_bytes(constants + struct.pack("<4f", 1.0, 1.0, 0.0, 1.0) * 64)

    for values in FREXP_EXP_CASES:
        set_case(*values)
        subprocess.run(draw, check=True)
    a, exponent_a, b, exponent_b = FREXP_EXP_CASES[1]
    set_case(a, exponent_a + 1, b, exponent_b)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("The draw accepted an incorrect subnormal exponent")


def check_unpack_byte_draws(directory, compiler, renderer, backend):
    instructions = unpack_bytes()
    source = directory / "ubyte.hlsl"
    shader = directory / "ubyte.cso"
    source.write_text(translate(instructions), encoding="ascii")
    compile_shader(compiler, source, shader)
    cases = []
    # Every word is a normal binary32 value, so its bits survive the constant buffer, and
    # all four bytes of each word differ, so reading the wrong byte changes the result.
    for idx, words in enumerate(((0x40ff8001, 0x3e7f02c3, 0x41a5005a, 0x00102030 | 0x3f000000),
                                 (0x4280ff7f, 0x3f11fe22, 0x407f8013, 0x00000000 | 0x40c08070))):
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *words)
        values = [float((word >> 8 * n) & 0xff) for n, word in enumerate(words)]
        values[3] = min(values[3], 1.0)
        case = directory / f"ubyte-{idx}.case"
        case.write_bytes(constants + struct.pack("<4f", *values) * 64)
        cases.append((case, "pass"))
    run_pixel_batch(renderer, backend, shader, cases)

    wrong = [line.replace("ubyte1", "ubyte0") for line in instructions]
    source.write_text(translate(wrong), encoding="ascii")
    compile_shader(compiler, source, shader)
    result = subprocess.run([renderer, backend, "--check-pixel-case", shader, cases[-1][0]],
                            capture_output=True, text=True)
    if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
        raise AssertionError("Reading the wrong byte escaped detection")
def check_reciprocal_iflag_draws(directory, compiler, renderer, backend):
    instructions = reciprocal_iflag()
    source = directory / "rcp_iflag.hlsl"
    shader = directory / "rcp_iflag.cso"
    source.write_text(translate(instructions), encoding="ascii")
    compile_shader(compiler, source, shader)
    inf = math.inf
    cases = []
    # Powers of two have exact reciprocals, so no rounding difference can hide a fault.
    # A zero operand gives an infinity of its sign, and clamp turns +inf into 1.
    for idx, ((a, b, c, d), expected) in enumerate((((2.0, 4.0, -0.125, 0.5), (0.5, -0.25, 8.0, 1.0)),
                                                   ((-0.25, 0.5, 16.0, 4.0), (-4.0, -2.0, 0.0625, 0.25)),
                                                   ((0.0, 0.0, -0.0, 0.0), (inf, -inf, inf, 1.0)))):
        constants = bytearray(256)
        struct.pack_into("<4f", constants, 12, a, b, c, d)
        case = directory / f"rcp_iflag-{idx}.case"
        case.write_bytes(constants + struct.pack("<4f", *expected) * 64)
        cases.append((case, "pass"))
    run_pixel_batch(renderer, backend, shader, cases)

    wrong = [line.replace("v_rcp_iflag_f32", "v_sqrt_f32") for line in instructions]
    source.write_text(translate(wrong), encoding="ascii")
    compile_shader(compiler, source, shader)
    result = subprocess.run([renderer, backend, "--check-pixel-case", shader, cases[-1][0]],
                            capture_output=True, text=True)
    if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
        raise AssertionError("Replacing the reciprocal with a square root escaped detection")
def check_offset_nibble_draws(directory, compiler, renderer, backend):
    instructions = offset_nibbles()
    source = directory / "cvt_off.hlsl"
    shader = directory / "cvt_off.cso"
    source.write_text(translate(instructions), encoding="ascii")
    compile_shader(compiler, source, shader)

    def offset(nibble):
        return (nibble - 16 if nibble & 8 else nibble) / 16

    # Each word is a normal binary32 value whose upper bits are set, so a result that used
    # more than bits 3:0 would change. All 16 nibbles appear across the four draws, and
    # the clamped lane sees both signs.
    high = 0x3f5a6b70
    nibbles = list(range(16))
    cases = []
    for start in range(0, 16, 4):
        group = nibbles[start:start + 4]
        words = [high | n for n in group]
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *words)
        values = [offset(n) for n in group]
        values[3] = min(1.0, max(0.0, values[3]))
        # Python's integer results lose the sign of zero; nibble 0 gives +0.0 either way.
        case = directory / f"cvt_off-{start}.case"
        case.write_bytes(constants + struct.pack("<4f", *values) * 64)
        cases.append((case, "pass"))
    run_pixel_batch(renderer, backend, shader, cases)

    # Reading the nibble as unsigned changes every result with bit 3 set.
    unsigned = translate(instructions).replace("asint(", "(")
    source.write_text(unsigned, encoding="ascii")
    compile_shader(compiler, source, shader)
    words = [high | 0x8, high | 0x9, high | 0xf, high | 0xc]
    constants = bytearray(256)
    struct.pack_into("<4I", constants, 12, *words)
    wrong_case = directory / "cvt_off-wrong.case"
    wrong_case.write_bytes(constants + struct.pack("<4f", -0.5, -0.4375, -0.0625, 0.0) * 64)
    result = subprocess.run([renderer, backend, "--check-pixel-case", shader, wrong_case],
                            capture_output=True, text=True)
    if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
        raise AssertionError("Reading the nibble without its sign escaped detection")
def check_trig_draws(directory, compiler, renderer, backend):
    source = directory / "trig.hlsl"
    shader = directory / "trig.cso"
    instructions = with_constants(TRIG_BODY)
    hlsl = translate(instructions)

    cases = []
    for idx, inputs in enumerate(TRIG_CASES):
        constants = bytearray(256)
        struct.pack_into("<4f", constants, 12, *inputs)
        case = directory / f"trig-{idx}.case"
        case.write_bytes(constants + struct.pack("<4f", *trig_reference(inputs)) * 64)
        cases.append(case)

    source.write_text(hlsl, encoding="ascii")
    compile_shader(compiler, source, shader)
    run_pixel_batch(renderer, backend, shader, [(case, "pass") for case in cases])

    def mismatches(text):
        source.write_text(text, encoding="ascii")
        compile_shader(compiler, source, shader)
        statuses, _ = run_pixel_batch_statuses(renderer, backend, shader, cases)
        return [status == "mismatch" for status in statuses]

    # Both controls change at least one quantized channel of every case by many steps.
    swapped = [line.replace("v_sin_f32", "v_tmp").replace("v_cos_f32", "v_sin_f32").replace("v_tmp", "v_cos_f32")
               for line in instructions]
    assert all(trig_reference(inputs, swap=True) != trig_reference(inputs) for inputs in TRIG_CASES)
    reduced_argument = "(turns(value) * 6.28318548f)"
    assert hlsl.count(reduced_argument) == 2, hlsl
    for name, text in (("calls swapped between sine and cosine", translate(swapped)),
                       ("radians instead of turns", hlsl.replace(reduced_argument, "(value)"))):
        if not all(mismatches(text)):
            raise AssertionError(f"Sine and cosine with {name} escaped detection")
    # Without the reduction, a binary32 x * 2 pi of the large inputs moves each case by 20 or more steps.
    # Whether the adapter shows it depends on the precision of its own sin and cos argument handling, so
    # the count is reported rather than required; the reduced draws above must match on every adapter.
    unreduced = mismatches(hlsl.replace(reduced_argument, "(value * 6.28318548f)"))
    print(f"trig_unreduced_mismatches={sum(unreduced)}/{len(unreduced)}")


def check_legacy_multiply_add_draws(directory, compiler, renderer, backend):
    instructions = legacy_multiply_add()
    source = directory / "mad_legacy.hlsl"
    shader = directory / "mad_legacy.cso"
    edge_case = directory / "mad_legacy-edge.case"
    compile_shader = [compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", shader, source]

    edge_program = uncompressed_export([
        "v_mov_b32_e32 v19, s47",
        "v_mad_legacy_f32 v12, s44, s45, 1.0",
        "v_mad_legacy_f32 v13, s46, 1.0, 1.0",
        "v_madak_f32 v14, s44, v19, 0x3f400000",
        "v_mad_legacy_f32 v15, 1.0, |v19|, 0.0",
    ])
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, -0.0, math.inf, math.nan, -1.0)
    write_case(edge_case, constants, (1.0, math.nan, 0.75, 1.0), nan_mask=0b0010)
    source.write_text(translate(edge_program), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    subprocess.run([renderer, backend, "--check-pixel-case", shader, edge_case], check=True)

    source.write_text(translate(instructions), encoding="ascii")
    subprocess.run(compile_shader, check=True)

    def legacy(x, y, z):
        product = 0.0 if x == 0 or y == 0 else x * y
        rounded_product, = struct.unpack("<f", struct.pack("<f", product))
        return rounded_product + z

    inf, nan = math.inf, math.nan
    # The last set detects unintended fusion: (1 + 2^-23) * (1 - 2^-23)
    # rounds to 1 as a float before adding -1, instead of returning -2^-46.
    delta = 2.0**-23
    sets = ((0.0, inf, 1.5, -2.0), (2.0, 3.0, 0.5, 0.25), (0.0, nan, -0.75, 4.0),
            (-1.5, 0.0, 2.0, 0.5), (1 + delta, 1 - delta, -1.0, 0.0))

    def write_legacy_case(path, values, *, fused_first_channel=False):
        a, b, c, d = values
        constants = bytearray(256)
        struct.pack_into("<4f", constants, 12, a, b, c, d)
        expected = (legacy(a, b, c), legacy(b, a, -d), legacy(abs(c), d, a),
                    min(1.0, max(0.0, legacy(a, d, c))))
        if fused_first_channel:
            expected = (a * b + c, *expected[1:])
        assert not any(math.isnan(v) for v in expected), expected
        path.write_bytes(constants + struct.pack("<4f", *expected) * 64)

    cases = []
    for idx, values in enumerate(sets):
        path = directory / f"mad_legacy-{idx}.case"
        write_legacy_case(path, values)
        cases.append((path, "pass"))
    fused_case = directory / "mad_legacy-fused.case"
    write_legacy_case(fused_case, sets[-1], fused_first_channel=True)
    cases.insert(-1, (fused_case, "mismatch"))
    run_pixel_batch(renderer, backend, shader, cases)

    # An ordinary multiply-add gives NaN for 0 * inf on the first set.
    wrong = [line.replace("v_mad_legacy_f32", "v_fma_f32") for line in instructions]
    source.write_text(translate(wrong), encoding="ascii")
    subprocess.run(compile_shader, check=True)
    result = subprocess.run([renderer, backend, "--check-pixel-case", shader, cases[0][0]],
                            capture_output=True, text=True)
    if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
        raise AssertionError("Replacing the legacy multiply-add lost the zero rule without detection")


def check_mac_legacy_draws(directory, compiler, renderer, backend):
    """Draw MAC_LEGACY_BODY on every constant set, then draw wrong lowerings and wrong references.

    The right shader must match mac_legacy_reference on all sets. Each wrong lowering must mismatch on at
    least one set, and a fused reference must mismatch on the last. A renderer that reports an unavailable
    adapter (exit 77) makes this a skip. On a desktop GPU the draws establish what the translated shader
    computes, not what the console does.
    """
    source = directory / "mac_legacy.hlsl"
    shader = directory / "mac_legacy.cso"
    program = uncompressed_export(MAC_LEGACY_BODY)

    def write(path, values, expected):
        constants = bytearray(256)
        struct.pack_into("<4f", constants, 12, *values)
        write_case(path, constants, expected)

    cases = []
    for idx, values in enumerate(MAC_LEGACY_SETS):
        path = directory / f"mac_legacy-{idx}.case"
        write(path, values, mac_legacy_reference(*values))
        cases.append(path)
    fused = directory / "mac_legacy-fused.case"
    write(fused, MAC_LEGACY_SETS[-1], mac_legacy_reference(*MAC_LEGACY_SETS[-1], fused_first_channel=True))
    assert mac_legacy_reference(*MAC_LEGACY_SETS[-1])[0] == 0.0 != mac_legacy_reference(
        *MAC_LEGACY_SETS[-1], fused_first_channel=True)[0]

    def statuses(body_template=None):
        with patch.dict(ALU, {} if body_template is None else {"mac_legacy": (3, body_template)}):
            source.write_text(translate(program), encoding="ascii")
        compile_shader(compiler, source, shader)
        return run_pixel_batch_statuses(renderer, backend, shader, [*cases, fused])[0]

    right = statuses()
    if right != ["pass"] * len(cases) + ["mismatch"]:
        raise AssertionError(f"Unexpected statuses for the legacy multiply-accumulate: {right}")
    legacy = "(({0} == 0 || {1} == 0) ? 0.0f : {0} * {1})"
    # A plain product of 0 and infinity is NaN, so the first set separates the lowering without the zero rule.
    assert math.isnan(MAC_LEGACY_SETS[0][0] * MAC_LEGACY_SETS[0][1])
    mutants = (
        ("legacy zero rule removed", "(({0} * {1}) + {2})", None),
        ("accumulator ignored", f"({legacy} + 0)", lambda x, y, acc: 0.0),
        ("source 1 as accumulator", f"({legacy} + {{1}})", lambda x, y, acc: y),
        ("source 0 as accumulator", f"({legacy} + {{0}})", lambda x, y, acc: x),
    )
    for label, template, accumulate in mutants:
        if accumulate is not None:
            # The Python model of the wrong lowering must differ from the right one, so the fixture can see it.
            assert any(mac_legacy_reference(*values, accumulate=accumulate) != mac_legacy_reference(*values)
                       for values in MAC_LEGACY_SETS if not any(map(math.isnan, values))), label
        if "mismatch" not in statuses(template)[:len(cases)]:
            raise AssertionError(f"The wrong lowering escaped detection: {label}")


def check_rgb_sample_draw(directory: Path, compiler: str, renderer: str, backend: str) -> None:
    hlsl = directory / "rgb-sample.hlsl"
    shader = directory / "rgb-sample.cso"
    hlsl.write_text(translate(rgb_sample()), encoding="ascii")
    compile_shader(compiler, hlsl, shader)
    cases = []
    for alpha, status in ((1.0, "mismatch"), (0.5, "pass")):
        colors = (struct.pack("<4f", half_rtz(red / 255), half_rtz(green / 255), half_rtz(128 / 255), alpha)
                  for green in RAMP for red in RAMP)
        case = directory / f"rgb-sample-{status}.case"
        case.write_bytes(bytes(256) + b"".join(colors))
        cases.append((case, status))
    result = run_pixel_batch(renderer, backend, shader, cases)
    if "pixel_mismatch x=0 y=0" not in result.stderr:
        raise AssertionError("RGB sample with an overwritten alpha register escaped detection")


def check_render(directory, compiler, renderer, backend, *, core_only=False):
    for swap in (False, True):
        source = directory / "fixture.hlsl"
        shader = directory / "fixture.cso"
        source.write_text(translate(assembly(swap)), encoding="ascii")
        compile_shader(compiler, source, shader, ieee=False)
        draw = [renderer, backend, "--check-pixel", shader]
        result = subprocess.run([*draw, "bgra" if swap else "rgba"])
        if result.returncode:
            return result.returncode
        if swap:
            wrong = subprocess.run([*draw, "rgba"], capture_output=True, text=True)
            if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
                raise AssertionError("The renderer failed to detect the changed channel order")
    check_dpp_int32_draw(directory, compiler, renderer, backend)
    check_dpp_draw(directory, compiler, renderer, backend)
    check_vop3p_draw(directory, compiler, renderer, backend)
    check_bitwise_draw(directory, compiler, renderer, backend)
    check_fma_mix_draw(directory, compiler, renderer, backend)
    check_dot_draw(directory, compiler, renderer, backend)
    check_pk_int16_draw(directory, compiler, renderer, backend)
    check_narrow_draw(directory, compiler, renderer, backend)
    check_mode_register_draw(directory, compiler, renderer, backend)
    check_movk_draw(directory, compiler, renderer, backend)
    check_half_draw(directory, compiler, renderer, backend)
    for exp_shader_inst in ("s_waitcnt_expcnt null, 0x3", "s_waitcnt expcnt(0)"):
        exp_program = [exp_shader_inst] + assembly(False)[:-1] + [exp_shader_inst, "s_endpgm"]
        source.write_text(translate(exp_program), encoding="ascii")
        compile_shader(compiler, source, shader, ieee=False)
        subprocess.run([renderer, backend, "--check-pixel", shader, "rgba"], check=True)
    check_rgb_sample_draw(directory, compiler, renderer, backend)
    check_trig_draws(directory, compiler, renderer, backend)
    program = arithmetic()
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    arith_cases = []
    for idx, constants in enumerate(((4.0, -.0625, .125, -.015625), (2.0, -.03125, .0625, -.0078125))):
        arith_cases.append((arithmetic_case(directory, *constants, name=f"arithmetic-{idx}.case"), "pass"))
    run_pixel_batch(renderer, backend, shader, arith_cases)
    swapped = [line.replace("v_subrev_f32", "v_sub_f32") for line in program]
    source.write_text(translate(swapped), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, arith_cases[-1][0]],
                           capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Swapped subtraction operands escaped detection")

    program = with_scalar_constant_reads()
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    scalar_const_cases = []
    for idx, (red, green, blue) in enumerate(((.25, -.5, .75), (-.125, .625, .875))):
        constants = bytearray(256)
        for offset, value in ((0, red), (8, green), (252, blue)):
            struct.pack_into("<f", constants, offset, value)
        case = directory / f"scalar-constants-{idx}.case"
        write_case(case, constants, (red, green, blue, 1))
        scalar_const_cases.append((case, "pass"))
    run_pixel_batch(renderer, backend, shader, scalar_const_cases)
    wrong_word = [line.replace("s62, s[44:47], 0x8", "s62, s[44:47], 0x4") for line in program]
    source.write_text(translate(wrong_word), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, scalar_const_cases[-1][0]],
                           capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Reading the neighboring constant word escaped detection")
    check_three_buffer_draw(directory, compiler, renderer, backend)

    check_affine_lod_draw(directory, compiler, renderer, backend)
    program = with_scalar_pair_reads()
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    pair_cases = []
    for idx, constants in enumerate(((2.0, -.0625, 1.0, .5, .25, .5), (.5, -.03125, .75, 2.0, .125, .25))):
        case = scalar_pair_case(directory, *constants, name=f"scalar-pairs-{idx}.case")
        pair_cases.append((case, "pass"))
    # Words 4, 6 and 63 are the second word of each pair; each must reach its own channel.
    reference = pair_cases[-1][0].read_bytes()
    for offset in (16, 24, 252):
        corrupted = bytearray(reference)
        struct.pack_into("<f", corrupted, offset, 1.0)
        wrong_case = directory / f"scalar-pairs-wrong-{offset}.case"
        wrong_case.write_bytes(corrupted)
        pair_cases.insert(-1, (wrong_case, "mismatch"))
    run_pixel_batch(renderer, backend, shader, pair_cases)
    shifted = [line.replace("s[62:63], s[44:47], 0x14", "s[62:63], s[44:47], 0x10") for line in program]
    source.write_text(translate(shifted), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, pair_cases[-1][0]],
                           capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Reading the pair one word early escaped detection")

    # The NVIDIA driver (32.0.16.2124, RTX 5060 Ti) refuses PSO creation for the curve program
    # under -denorm preserve (E_OUTOFMEMORY). Its rows contain no binary32 subnormals, so the
    # mode cannot change the exported bits; record this fixture's mode as any, not the assumption,
    # and assert that mode independence on every expected word instead of arguing it.
    for program_fn, case_fn, mode in ((curve, curve_case, "any"),
                                      (shaping, shaping_case, F32_DENORM_MODE)):
        source.write_text(translate(program_fn()), encoding="ascii")
        compile_shader(compiler, source, shader, mode=mode)
        cases = []
        for idx, gain in enumerate((.75, 1.125)):
            case = case_fn(directory, gain, name=f"{program_fn.__name__}-{idx}.case")
            if mode != F32_DENORM_MODE:
                words = struct.unpack("<256f", case.read_bytes()[256:1280])
                assert all(flush_ftz(word) == word for word in words), program_fn.__name__
            cases.append((case, "pass"))
        corrupted = bytearray(cases[-1][0].read_bytes())
        struct.pack_into("<f", corrupted, 16, 0.0)
        wrong_case = directory / f"{program_fn.__name__}-stale.case"
        wrong_case.write_bytes(corrupted)
        cases.insert(-1, (wrong_case, "mismatch"))
        run_pixel_batch(renderer, backend, shader, cases)

    if core_only:
        return 0
    check_predicate_and_rounding_draws(directory, compiler, renderer, backend)
    return check_blend_and_scalar_draws(directory, compiler, renderer, backend)


def check_sopk_cmpk_cmovk_draw(directory, compiler, renderer, backend):
    source = directory / "sopk_cmpk_cmovk.hlsl"
    shader = directory / "sopk_cmpk_cmovk.cso"
    program = sopk_cmpk_cmovk()

    def write_words(path, words, expected):
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *words)
        for word in expected:
            assert struct.unpack("<I", struct.pack("<f", as_float(word)))[0] == word, hex(word)
        write_case(path, bytes(constants), tuple(as_float(word) for word in expected), integer_rt=True)

    source.write_text(translate_int(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    cases = []
    for idx, words in enumerate(SOPK_CMPK_WORDS):
        case = directory / f"sopk_cmpk_cmovk-{idx}.case"
        write_words(case, words, sopk_cmpk_cmovk_reference(words))
        cases.append((case, "pass"))
    run_pixel_batch(renderer, backend, shader, cases)

    for old, new, label in (
        ("s_cmpk_lt_i32 s44, 0x8000", "s_cmpk_lt_u32 s44, 0x8000", "signed sopk comparison"),
        ("s_cmovk_i32 s50, 0xfffe", "s_cmovk_i32 s50, 0x7fff", "sopk conditional move"),
    ):
        mutated = [line.replace(old, new) for line in program]
        assert mutated != program, label
        source.write_text(translate_int(mutated), encoding="ascii")
        compile_shader(compiler, source, shader)
        wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, cases[0][0]],
                               capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Mutating {label} escaped detection")


F16_COMPARE_SETS = (
    (0xdead3c00, 0xbeef4000, 0x12343c00, 0x100),
    (0xffffbc00, 0xffff3c00, 0xffffbc00, 0x008),
    (0x00008000, 0x00000000, 0x00008000, 0x020),
    (0x00000000, 0x00008000, 0x00000000, 0x040),
    (0x00000001, 0x00000001, 0x00000001, 0x080),
    (0x00008001, 0x00003800, 0x00008001, 0x010),
    (0x00007c00, 0x0000fc00, 0x0000fc00, 0x004),
    (0x0000fc00, 0x00007c00, 0x00007c00, 0x200),
    (0x00007e00, 0x00003c00, 0x00007c01, 0x001),
    (0x00004000, 0x0000fe00, 0x0000fe00, 0x002),
    (0x0000c400, 0x0000c400, 0x00003118, 0x101),
    (0x00003118, 0x00003119, 0x0000b800, 0x008),
    (0x0000bc00, 0x0000c000, 0x00004400, 0x000),
    (0x0000bc00, 0x00004400, 0x00004000, 0x3ff),
)


def f16_compares() -> list[str]:
    """Build a shader testing all 16 binary16 predicates and v_cmp_class_f16 across e32 and e64."""
    body = [
        "v_mov_b32_e32 v12, 0",
        "v_mov_b32_e32 v13, 0",
        "v_mov_b32_e32 v14, 0",
        "v_mov_b32_e32 v15, 0",
        "v_mov_b32_e32 v16, s45",
        "v_mov_b32_e32 v18, s44",
        "v_mov_b32_e32 v20, s46",
        "v_mov_b32_e32 v21, s47",
    ]

    def emit_flag(acc: str, bit: int, cond: str, insn: str) -> None:
        body.append(insn)
        body.append(f"v_cndmask_b32_e64 v17, 0, 1, {cond}")
        if bit:
            body.append(f"v_lshlrev_b32_e32 v17, {bit}, v17")
        body.append(f"v_or_b32_e32 {acc}, {acc}, v17")

    for bit, pred in enumerate(PREDICATES):
        emit_flag("v12", bit, "vcc_lo", f"v_cmp_{pred}_f16_e32 vcc_lo, s44, v16")
    for bit, insn in (
        (16, "v_cmp_lt_f16_e32 vcc_lo, v18, v16"),
        (17, "v_cmp_eq_f16_e32 vcc_lo, 1.0, v16"),
        (18, "v_cmp_eq_f16_e32 vcc_lo, 1, v16"),
        (19, "v_cmp_lt_f16_e32 vcc_lo, -0.5, v16"),
        (20, "v_cmp_lt_f16_e32 vcc_lo, 0.15915494, v16"),
        (21, "v_cmp_lt_f16_e32 vcc_lo, 0xc000, v16"),
    ):
        emit_flag("v12", bit, "vcc_lo", insn)

    for bit, pred in enumerate(PREDICATES):
        emit_flag("v13", bit, "s52", f"v_cmp_{pred}_f16_e64 s52, -s44, |s45|")
    for bit, cond, insn in (
        (16, "s53", "v_cmp_lt_f16_e64 s53, v18, s45"),
        (17, "vcc_lo", "v_cmp_le_f16_e64 vcc_lo, -|s44|, 2.0"),
        (18, "s52", "v_cmp_gt_f16_e64 s52, s44, -1.0"),
        (19, "s52", "v_cmp_eq_f16_e64 s52, -4.0, v16"),
        (20, "s52", "v_cmp_lt_f16_e64 s52, |v18|, 0x3800"),
    ):
        emit_flag("v13", bit, cond, insn)

    for cls in range(10):
        body.append(f"v_mov_b32_e32 v19, {1 << cls}")
        emit_flag("v14", cls, "vcc_lo", "v_cmp_class_f16_e32 vcc_lo, s46, v19")
    for bit, cond, insn in (
        (10, "vcc_lo", "v_cmp_class_f16_e32 vcc_lo, s46, v21"),
        (11, "s52", "v_cmp_class_f16_e64 s52, s46, s47"),
        (12, "vcc_lo", "v_cmp_class_f16_e64 vcc_lo, -s46, s47"),
        (13, "s53", "v_cmp_class_f16_e64 s53, |s46|, s47"),
        (14, "s52", "v_cmp_class_f16_e64 s52, -|v20|, s47"),
        (15, "s52", "v_cmp_class_f16_e64 s52, -s46, 0x3ff"),
        (16, "s52", "v_cmp_class_f16_e64 s52, 0.5, s47"),
        (17, "s52", "v_cmp_class_f16_e64 s52, -1, 2"),
        (18, "s52", "v_cmp_class_f16_e64 s52, 1, 0x80"),
        (19, "vcc_lo", "v_cmp_class_f16_e32 vcc_lo, 0x7c01, v21"),
        (20, "vcc_lo", "v_cmp_class_f16_e32 vcc_lo, -0.5, v21"),
    ):
        emit_flag("v14", bit, cond, insn)

    for bit, cond, insn in (
        (0, "s52", "v_cmp_eq_f16_e64 s52, s44, v16"),
        (1, "s52", "v_cmp_lt_f16_e64 s52, |v18|, |v16|"),
        (2, "s52", "v_cmp_eq_f16_e64 s52, -v18, -s45"),
        (3, "s52", "v_cmp_o_f16_e64 s52, s44, -|v16|"),
        (4, "s52", "v_cmp_u_f16_e64 s52, s44, s45"),
        (5, "vcc_lo", "v_cmp_neq_f16_e32 vcc_lo, 0.5, v16"),
        (6, "vcc_lo", "v_cmp_ge_f16_e32 vcc_lo, -2.0, v16"),
        (7, "vcc_lo", "v_cmp_gt_f16_e32 vcc_lo, 4.0, v16"),
        (8, "vcc_lo", "v_cmp_le_f16_e32 vcc_lo, -4.0, v16"),
        (9, "vcc_lo", "v_cmp_eq_f16_e32 vcc_lo, 2.0, v16"),
        (10, "vcc_lo", "v_cmp_eq_f16_e32 vcc_lo, 0, v16"),
        (11, "s52", "v_cmp_class_f16_e64 s52, v20, 4"),
        (12, "vcc_lo", "v_cmp_class_f16_e32 vcc_lo, v20, v21"),
        (13, "vcc_lo", "v_cmp_eq_f16_e32 vcc_lo, -1, v16"),
    ):
        emit_flag("v15", bit, cond, insn)

    return body


def f16_compares_reference(words: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Compute the exact 4 uint bitmasks expected from f16_compares()."""
    def half_float(bits: int) -> float:
        return struct.unpack("<e", struct.pack("<H", bits & 0xffff))[0]

    def half_class(bits: int) -> int:
        h = bits & 0xffff
        mag = h & 0x7fff
        neg = h >= 0x8000
        if mag > 0x7c00:
            return 0x2 if (h & 0x0200) else 0x1
        if mag == 0x7c00:
            return 0x4 if neg else 0x200
        if mag >= 0x0400:
            return 0x8 if neg else 0x100
        if mag != 0:
            return 0x10 if neg else 0x80
        return 0x20 if neg else 0x40

    def cmp_pred(pred: str, a: float, b: float) -> bool:
        ordered = (not math.isnan(a)) and (not math.isnan(b))
        table = {
            "f": False,
            "lt": ordered and a < b,
            "eq": ordered and a == b,
            "le": ordered and a <= b,
            "gt": ordered and a > b,
            "lg": ordered and a != b,
            "ge": ordered and a >= b,
            "o": ordered,
            "u": not ordered,
            "nge": not (ordered and a >= b),
            "nlg": not (ordered and a != b),
            "ngt": not (ordered and a > b),
            "nle": not (ordered and a <= b),
            "neq": not (ordered and a == b),
            "nlt": not (ordered and a < b),
            "tru": True,
        }
        return table[pred]

    lhs_h, rhs_h, cls_h = words[0] & 0xffff, words[1] & 0xffff, words[2] & 0xffff
    cls_mask = words[3] & 0xffffffff
    lhs = half_float(lhs_h)
    rhs = half_float(rhs_h)
    sub_min = half_float(0x0001)
    inv2pi = half_float(0x3118)
    nan_val = half_float(0xffff)

    w0 = 0
    for bit, pred in enumerate(PREDICATES):
        if cmp_pred(pred, lhs, rhs):
            w0 |= 1 << bit
    for bit, cond in (
        (16, cmp_pred("lt", lhs, rhs)),
        (17, cmp_pred("eq", 1.0, rhs)),
        (18, cmp_pred("eq", sub_min, rhs)),
        (19, cmp_pred("lt", -0.5, rhs)),
        (20, cmp_pred("lt", inv2pi, rhs)),
        (21, cmp_pred("lt", -2.0, rhs)),
    ):
        if cond:
            w0 |= 1 << bit

    w1 = 0
    for bit, pred in enumerate(PREDICATES):
        if cmp_pred(pred, -lhs, abs(rhs)):
            w1 |= 1 << bit
    for bit, cond in (
        (16, cmp_pred("lt", lhs, rhs)),
        (17, cmp_pred("le", -abs(lhs), 2.0)),
        (18, cmp_pred("gt", lhs, -1.0)),
        (19, cmp_pred("eq", -4.0, rhs)),
        (20, cmp_pred("lt", abs(lhs), 0.5)),
    ):
        if cond:
            w1 |= 1 << bit

    w2 = 0
    for cls in range(10):
        if (half_class(cls_h) & (1 << cls)) != 0:
            w2 |= 1 << cls
    for bit, cond in (
        (10, (half_class(cls_h) & cls_mask) != 0),
        (11, (half_class(cls_h) & cls_mask) != 0),
        (12, (half_class(cls_h ^ 0x8000) & cls_mask) != 0),
        (13, (half_class(cls_h & 0x7fff) & cls_mask) != 0),
        (14, (half_class((cls_h & 0x7fff) ^ 0x8000) & cls_mask) != 0),
        (15, (half_class(cls_h ^ 0x8000) & 0x3ff) != 0),
        (16, (half_class(0x3800) & cls_mask) != 0),
        (17, (half_class(0xffff) & 2) != 0),
        (18, (half_class(0x0001) & 0x80) != 0),
        (19, (half_class(0x7c01) & cls_mask) != 0),
        (20, (half_class(0xb800) & cls_mask) != 0),
    ):
        if cond:
            w2 |= 1 << bit

    w3 = 0
    for bit, cond in (
        (0, cmp_pred("eq", lhs, rhs)),
        (1, cmp_pred("lt", abs(lhs), abs(rhs))),
        (2, cmp_pred("eq", -lhs, -rhs)),
        (3, cmp_pred("o", lhs, -abs(rhs))),
        (4, cmp_pred("u", lhs, rhs)),
        (5, cmp_pred("neq", 0.5, rhs)),
        (6, cmp_pred("ge", -2.0, rhs)),
        (7, cmp_pred("gt", 4.0, rhs)),
        (8, cmp_pred("le", -4.0, rhs)),
        (9, cmp_pred("eq", 2.0, rhs)),
        (10, cmp_pred("eq", 0.0, rhs)),
        (11, (half_class(cls_h) & 4) != 0),
        (12, (half_class(cls_h) & cls_mask) != 0),
        (13, cmp_pred("eq", nan_val, rhs)),
    ):
        if cond:
            w3 |= 1 << bit

    return (w0, w1, w2, w3)


def check_f16_compare_draws(directory, compiler, renderer, backend):
    source = directory / "f16-compare.hlsl"
    shader = directory / "f16-compare.cso"
    program = uncompressed_export(f16_compares())
    source.write_text(translate_int(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    cases = []
    for index, words in enumerate(F16_COMPARE_SETS):
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *words)
        expected = tuple(as_float(w) for w in f16_compares_reference(words))
        path = write_case(directory / f"f16-cmp-{index}.case", constants, expected, integer_rt=True)
        cases.append((path, "pass"))
        if index < 4:
            wrong_words = list(f16_compares_reference(words))
            wrong_words[index] ^= 1 << index
            wrong_path = write_case(
                directory / f"f16-cmp-bad-{index}.case",
                constants,
                tuple(as_float(w) for w in wrong_words),
                integer_rt=True,
            )
            cases.append((wrong_path, "mismatch"))
    run_pixel_batch(renderer, backend, shader, cases)

    mutated_shader = directory / "f16-compare-mutated.cso"
    for label, old, new, case_index in (
        ("f16 e32 predicate", "v_cmp_lt_f16_e32 vcc_lo, s44, v16", "v_cmp_le_f16_e32 vcc_lo, s44, v16", 4),
        ("f16 e64 neg", "v_cmp_lt_f16_e64 s52, -s44, |s45|", "v_cmp_lt_f16_e64 s52, s44, |s45|", 1),
        ("f16 class neg", "v_cmp_class_f16_e64 vcc_lo, -s46, s47", "v_cmp_class_f16_e64 vcc_lo, s46, s47", 2),
    ):
        mutated = [new if line == old else line for line in program]
        assert mutated != program
        source.write_text(translate_int(mutated), encoding="ascii")
        compile_shader(compiler, source, mutated_shader)
        draw = [renderer, backend, "--check-pixel-case", mutated_shader, directory / f"f16-cmp-{case_index}.case"]
        result = subprocess.run(draw, capture_output=True, text=True)
        if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
            raise AssertionError(f"Mutating {label} escaped detection")


F32_COMPARE_SETS = (
    # (lhs, rhs, extra_a, extra_b)
    # 0: standard ordered comparisons
    (0x3f800000, 0x40000000, 0x3f000000, 0x3fc00000),  # 1.0, 2.0, 0.5, 1.5
    # 1: equal values and signed zeros
    (0x00000000, 0x80000000, 0xbf800000, 0x3f800000),  # +0.0, -0.0, -1.0, 1.0
    # 2: reversed order
    (0x40a00000, 0x3f800000, 0x41200000, 0x3f000000),  # 5.0, 1.0, 10.0, 0.5
    # 3: equal positive finite
    (0x40200000, 0x40200000, 0x40400000, 0x40000000),  # 2.5, 2.5, 3.0, 2.0
    # 4: NaN operand
    (0x7fc00000, 0x3f800000, 0x00000000, 0x00000000),  # NaN, 1.0, 0, 0
    # 5: infinite and extreme finite
    (0x7f800000, 0xff800000, 0x7f7fffff, 0xff7fffff),  # +inf, -inf, max_flt, min_flt
    (0x7f800001, 0x40000000, 0x00000000, 0x80000000),  # signaling NaN, 2.0, +0, -0
)


def f32_compares() -> list[str]:
    """Build a shader testing all 16 binary32 predicates across e32, e64, and SDWA."""
    body = [
        "v_mov_b32_e32 v12, 0",
        "v_mov_b32_e32 v13, 0",
        "v_mov_b32_e32 v14, 0",
        "v_mov_b32_e32 v15, 0",
        "v_mov_b32_e32 v16, s45",
        "v_mov_b32_e32 v18, s44",
        "v_mov_b32_e32 v20, s46",
        "v_mov_b32_e32 v21, s47",
    ]

    def emit_flag(acc: str, bit: int, cond: str, insn: str) -> None:
        body.append(insn)
        body.append(f"v_cndmask_b32_e64 v17, 0, 1, {cond}")
        if bit:
            body.append(f"v_lshlrev_b32_e32 v17, {bit}, v17")
        body.append(f"v_or_b32_e32 {acc}, {acc}, v17")

    # Channel 0: all 16 predicates in _e32 encoding (vcc_lo)
    for bit, pred in enumerate(PREDICATES):
        emit_flag("v12", bit, "vcc_lo", f"v_cmp_{pred}_f32_e32 vcc_lo, s44, v16")
    for bit, insn in (
        (16, "v_cmp_lt_f32_e32 vcc_lo, v18, v16"),
        (17, "v_cmp_eq_f32_e32 vcc_lo, 1.0, v16"),
        (18, "v_cmp_eq_f32_e32 vcc_lo, 1, v16"),
        (19, "v_cmp_lt_f32_e32 vcc_lo, -0.5, v16"),
        (20, "v_cmp_lt_f32_e32 vcc_lo, 0.15915494, v16"),
        (21, "v_cmp_lt_f32_e32 vcc_lo, 0x3f800000, v16"),
    ):
        emit_flag("v12", bit, "vcc_lo", insn)

    # Channel 1: all 16 predicates in _e64 encoding (SGPR destination s52) with input modifiers
    for bit, pred in enumerate(PREDICATES):
        emit_flag("v13", bit, "s52", f"v_cmp_{pred}_f32_e64 s52, -s44, |s45|")
    for bit, cond, insn in (
        (16, "s53", "v_cmp_lt_f32_e64 s53, v18, s45"),
        (17, "vcc_lo", "v_cmp_le_f32_e64 vcc_lo, -|s44|, 2.0"),
        (18, "s52", "v_cmp_gt_f32_e64 s52, s44, -1.0"),
        (19, "s52", "v_cmp_eq_f32_e64 s52, -4.0, v16"),
        (20, "s52", "v_cmp_lt_f32_e64 s52, |v18|, 0x3f000000"),
    ):
        emit_flag("v13", bit, cond, insn)

    # Channel 2: all 16 predicates in whole-dword _sdwa encoding (vcc_lo)
    for bit, pred in enumerate(PREDICATES):
        emit_flag("v14", bit, "vcc_lo", f"v_cmp_{pred}_f32_sdwa vcc_lo, s44, v16 src0_sel:DWORD src1_sel:DWORD")

    # Channel 3: heterogeneous combinations across e32 and e64
    for bit, cond, insn in (
        (0, "s52", "v_cmp_eq_f32_e64 s52, s44, v16"),
        (1, "s52", "v_cmp_lt_f32_e64 s52, |v18|, |v16|"),
        (2, "s52", "v_cmp_eq_f32_e64 s52, -v18, -s45"),
        (3, "s52", "v_cmp_o_f32_e64 s52, s44, -|v16|"),
        (4, "s52", "v_cmp_u_f32_e64 s52, s44, s45"),
        (5, "vcc_lo", "v_cmp_neq_f32_e32 vcc_lo, 0.5, v16"),
        (6, "vcc_lo", "v_cmp_ge_f32_e32 vcc_lo, -2.0, v16"),
        (7, "vcc_lo", "v_cmp_gt_f32_e32 vcc_lo, 4.0, v16"),
        (8, "vcc_lo", "v_cmp_le_f32_e32 vcc_lo, -4.0, v16"),
        (9, "vcc_lo", "v_cmp_eq_f32_e32 vcc_lo, 2.0, v16"),
        (10, "vcc_lo", "v_cmp_eq_f32_e32 vcc_lo, 0, v16"),
        (11, "s52", "v_cmp_gt_f32_e64 s52, v20, 0.0"),
        (12, "vcc_lo", "v_cmp_lt_f32_e32 vcc_lo, v20, v21"),
        (13, "vcc_lo", "v_cmp_eq_f32_e32 vcc_lo, -1, v16"),
    ):
        emit_flag("v15", bit, cond, insn)

    return body


def f32_compares_reference(words: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Compute the exact 4 uint bitmasks expected from f32_compares()."""
    def cmp_pred(pred: str, a: float, b: float) -> bool:
        ordered = (not math.isnan(a)) and (not math.isnan(b))
        table = {
            "f": False,
            "lt": ordered and a < b,
            "eq": ordered and a == b,
            "le": ordered and a <= b,
            "gt": ordered and a > b,
            "lg": ordered and a != b,
            "ge": ordered and a >= b,
            "o": ordered,
            "u": not ordered,
            "nge": not (ordered and a >= b),
            "nlg": not (ordered and a != b),
            "ngt": not (ordered and a > b),
            "nle": not (ordered and a <= b),
            "neq": not (ordered and a == b),
            "nlt": not (ordered and a < b),
            "tru": True,
        }
        return table[pred]

    def word_to_float(w: int) -> float:
        return struct.unpack("<f", struct.pack("<I", w & 0xffffffff))[0]

    lhs = word_to_float(words[0])
    rhs = word_to_float(words[1])
    extra_a = word_to_float(words[2])
    extra_b = word_to_float(words[3])

    w0 = 0
    for bit, pred in enumerate(PREDICATES):
        if cmp_pred(pred, lhs, rhs):
            w0 |= 1 << bit
    for bit, cond in (
        (16, cmp_pred("lt", lhs, rhs)),
        (17, cmp_pred("eq", 1.0, rhs)),
        # Integer immediates are interpreted as binary32 bits, not converted.
        (18, cmp_pred("eq", word_to_float(1), rhs)),
        (19, cmp_pred("lt", -0.5, rhs)),
        (20, cmp_pred("lt", word_to_float(0x3e22f983), rhs)),
        (21, cmp_pred("lt", 1.0, rhs)),
    ):
        if cond:
            w0 |= 1 << bit

    w1 = 0
    for bit, pred in enumerate(PREDICATES):
        if cmp_pred(pred, -lhs, abs(rhs)):
            w1 |= 1 << bit
    for bit, cond in (
        (16, cmp_pred("lt", lhs, rhs)),
        (17, cmp_pred("le", -abs(lhs), 2.0)),
        (18, cmp_pred("gt", lhs, -1.0)),
        (19, cmp_pred("eq", -4.0, rhs)),
        (20, cmp_pred("lt", abs(lhs), 0.5)),
    ):
        if cond:
            w1 |= 1 << bit

    w2 = 0
    for bit, pred in enumerate(PREDICATES):
        if cmp_pred(pred, lhs, rhs):
            w2 |= 1 << bit

    w3 = 0
    for bit, cond in (
        (0, cmp_pred("eq", lhs, rhs)),
        (1, cmp_pred("lt", abs(lhs), abs(rhs))),
        (2, cmp_pred("eq", -lhs, -rhs)),
        (3, cmp_pred("o", lhs, -abs(rhs))),
        (4, cmp_pred("u", lhs, rhs)),
        (5, cmp_pred("neq", 0.5, rhs)),
        (6, cmp_pred("ge", -2.0, rhs)),
        (7, cmp_pred("gt", 4.0, rhs)),
        (8, cmp_pred("le", -4.0, rhs)),
        (9, cmp_pred("eq", 2.0, rhs)),
        (10, cmp_pred("eq", 0.0, rhs)),
        (11, cmp_pred("gt", extra_a, 0.0)),
        (12, cmp_pred("lt", extra_a, extra_b)),
        (13, cmp_pred("eq", word_to_float(-1), rhs)),
    ):
        if cond:
            w3 |= 1 << bit

    return (w0, w1, w2, w3)


def check_f32_compare_draws(directory, compiler, renderer, backend):
    source = directory / "f32-compare.hlsl"
    shader = directory / "f32-compare.cso"
    program = uncompressed_export(f32_compares())
    source.write_text(translate_int(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    cases = []
    for index, words in enumerate(F32_COMPARE_SETS):
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *words)
        expected = tuple(as_float(w) for w in f32_compares_reference(words))
        path = write_case(directory / f"f32-cmp-{index}.case", constants, expected, integer_rt=True)
        cases.append((path, "pass"))
        if index < 4:
            wrong_words = list(f32_compares_reference(words))
            wrong_words[index] ^= 1 << index
            wrong_path = write_case(
                directory / f"f32-cmp-bad-{index}.case",
                constants,
                tuple(as_float(w) for w in wrong_words),
                integer_rt=True,
            )
            cases.append((wrong_path, "mismatch"))
    run_pixel_batch(renderer, backend, shader, cases)

    mutated_shader = directory / "f32-compare-mutated.cso"
    for label, old, new, case_index in (
        ("f32 e32 predicate", "v_cmp_lt_f32_e32 vcc_lo, s44, v16", "v_cmp_le_f32_e32 vcc_lo, s44, v16", 3),
        ("f32 e64 neg", "v_cmp_lt_f32_e64 s52, -s44, |s45|", "v_cmp_lt_f32_e64 s52, s44, |s45|", 2),
    ):
        mutated = [new if line == old else line for line in program]
        assert mutated != program
        source.write_text(translate_int(mutated), encoding="ascii")
        compile_shader(compiler, source, mutated_shader)
        draw = [renderer, backend, "--check-pixel-case", mutated_shader, directory / f"f32-cmp-{case_index}.case"]
        result = subprocess.run(draw, capture_output=True, text=True)
        if result.returncode != 1 or "pixel_mismatch" not in result.stderr:
            raise AssertionError(f"Mutating {label} escaped detection")


def check_predicate_and_rounding_draws(directory, compiler, renderer, backend):
    source = directory / "fixture.hlsl"
    shader = directory / "fixture.cso"
    program = truncate_values()
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    trunc_cases = []
    for idx, constants in enumerate(((1.75, -1.75, -2.75, 2.75), (-1.75, 2.75, 3.75, -1.75))):
        trunc_cases.append((truncate_case(directory, *constants, name=f"truncate-{idx}.case"), "pass"))
    run_pixel_batch(renderer, backend, shader, trunc_cases)
    rounded_down = [line.replace("v_trunc_f32", "v_floor_f32") for line in program]
    source.write_text(translate(rounded_down), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, trunc_cases[-1][0]],
                           capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Rounding toward negative infinity escaped detection")

    program = sopk_scalar()
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    sopk_cases = []
    for idx, (red, green) in enumerate(((0.5, 0.25), (0.125, 1.0))):
        sopk_cases.append((sopk_case(directory, red, green, name=f"sopk-{idx}.case"), "pass"))
    run_pixel_batch(renderer, backend, shader, sopk_cases)

    changed = [line.replace("s_mulk_i32 s50, 0x3", "s_mulk_i32 s50, 0x5") for line in program]
    source.write_text(translate(changed), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, sopk_cases[0][0]],
                           capture_output=True, text=True)

    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Changing the sopk multiply escaped detection")

    check_sopk_cmpk_cmovk_draw(directory, compiler, renderer, backend)

    source.write_text(translate(rounding()), encoding="ascii")
    compile_shader(compiler, source, shader)
    # Ties go to even (2.5 to 2, -3.5 to -4), unlike rounding half away from zero or half up.
    round_cases = []
    for idx, constants in enumerate(((4.0, 2.5, 1.25), (8.0, -3.5, -0.75))):
        round_cases.append((rounding_case(directory, *constants, name=f"rounding-{idx}.case"), "pass"))
    run_pixel_batch(renderer, backend, shader, round_cases)
    draw = [renderer, backend, "--check-pixel-case", shader, round_cases[-1][0]]
    for old, new, failure in (("v_rndne_f32_e32", "v_floor_f32_e32", "Flooring instead of rounding"),
                              ("v_ceil_f32_e32", "v_floor_f32_e32", "Flooring instead of ceiling")):
        source.write_text(translate([line.replace(old, new) for line in rounding()]), encoding="ascii")
        compile_shader(compiler, source, shader)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"{failure} escaped detection")
    # Rounding half up gives -3 for the -3.5 tie, so the same draw must catch it.
    source.write_text(translate(rounding()).replace("round(", "floor(0.5f + "), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Rounding ties up instead of to even escaped detection")

    program = absolute_value()
    case = absolute_case(directory)
    draw = [renderer, backend, "--check-pixel-case", shader, case]
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader, ieee=False)
    subprocess.run(draw, check=True)

    without_abs = [line.replace("|v12|", "v12") for line in program]
    source.write_text(translate(without_abs), encoding="ascii")
    compile_shader(compiler, source, shader, ieee=False)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("The renderer failed to detect the missing absolute value")
    check_ceiling_draws(directory, compiler, renderer, backend)
    check_unpack_byte_draws(directory, compiler, renderer, backend)
    check_reciprocal_iflag_draws(directory, compiler, renderer, backend)
    check_offset_nibble_draws(directory, compiler, renderer, backend)
    check_legacy_multiply_add_draws(directory, compiler, renderer, backend)
    source.write_text(translate(select()), encoding="ascii")
    compile_shader(compiler, source, shader)
    select_cases = []
    for idx, constants in enumerate(((.5, .75, .375, .25), (.125, .5, 1.0, 0.0))):
        select_cases.append((select_case(directory, *constants, name=f"select-{idx}.case"), "pass"))
    corrupted = bytearray(select_cases[-1][0].read_bytes())
    struct.pack_into("<f", corrupted, 12, .5)
    wrong_select = directory / "select-stale.case"
    wrong_select.write_bytes(corrupted)
    select_cases.insert(-1, (wrong_select, "mismatch"))
    run_pixel_batch(renderer, backend, shader, select_cases)
    # Binary32 values on either side of a half-ULP midpoint, in both signs.
    values = (1 + 3 / 4096, 1 + 1 / 4096, -1 - 3 / 4096, -1 - 1 / 4096)
    constants = bytearray(256)
    struct.pack_into("<4f", constants, 12, *values)
    program = with_constants([f"v_sub_f32_e64 v{12 + i}, s{44 + i}, 0" for i in range(4)])
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    case = directory / "signed-export.case"
    write_case(case, constants, tuple(map(half_rtz, values)))
    subprocess.run([renderer, backend, "--check-pixel-case", shader, case], check=True)
    check_sext_draw(directory, compiler, renderer, backend)
    check_scalar_pack_draw(directory, compiler, renderer, backend)
    check_scalar_bit_count_draw(directory, compiler, renderer, backend)
    check_scalar_carry_chain_draw(directory, compiler, renderer, backend)
    check_scalar_leading_bit_draw(directory, compiler, renderer, backend)
    check_scalar_mul_hi_draw(directory, compiler, renderer, backend)
    check_literal_fma_draw(directory, compiler, renderer, backend)
    return 0


def check_comparison_draws(directory, compiler, renderer, backend):
    """The comparison and class draws, split from draw-predicate-rounding so each CTest entry stays under its
    timeout on adapters where every renderer launch pays for device creation with the debug layer (issue #448)."""
    check_f16_compare_draws(directory, compiler, renderer, backend)
    check_f32_compare_draws(directory, compiler, renderer, backend)
    check_class_draw(directory, compiler, renderer, backend)
    check_predicate_draws(directory, compiler, renderer, backend)
    check_int_predicate_draws(directory, compiler, renderer, backend)
    check_unsigned_less_than_draws(directory, compiler, renderer, backend)
    check_int16_predicate_draws(directory, compiler, renderer, backend)
    check_sdwa_predicate_draws(directory, compiler, renderer, backend)
    return 0


def check_blend_and_scalar_draws(directory, compiler, renderer, backend):
    source = directory / "fixture.hlsl"
    shader = directory / "fixture.cso"
    capture = directory / "blend.rgba"
    draw = [renderer, backend, "--dual-pixel", shader, capture]
    for case in DUAL_CASES:
        source.write_text(translate_dual(dual_program(case)), encoding="ascii")
        compile_shader(compiler, source, shader)
        subprocess.run(draw, check=True)
    wrong = subprocess.run([renderer, backend, "--dual-pixel-swapped", shader, capture], capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Swapped texture bindings escaped detection")
    source.write_text(translate_dual(dual_program(swap_weights=True)), encoding="ascii")
    compile_shader(compiler, source, shader)
    wrong = subprocess.run(draw, capture_output=True, text=True)
    if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
        raise AssertionError("Swapped blend weights escaped detection")
    check_cube_draw(directory, compiler, renderer, backend)
    check_sdwa_draw(directory, compiler, renderer, backend)
    check_sdwa_arith_draws(directory, compiler, renderer, backend)
    check_fract_draw(directory, compiler, renderer, backend)
    check_reciprocal_draw(directory, compiler, renderer, backend)
    check_int_minmax_draws(directory, compiler, renderer, backend)
    check_cvt_int_draws(directory, compiler, renderer, backend)

    check_frexp_ldexp_draws(directory, compiler, renderer, backend)
    check_mul24_draws(directory, compiler, renderer, backend)
    run_quadmask_bitreplicate(directory, compiler, renderer, backend)
    check_scalar_select64_draw(directory, compiler, renderer, backend)
    check_scalar_bfe64_draw(directory, compiler, renderer, backend)

    program = scalar64()
    source.write_text(translate(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    scalar64_cases = []
    for idx, values in enumerate(SCALAR64_SETS):
        scalar64_cases.append((scalar64_case(directory, values, name=f"scalar64-{idx}.case"), "pass"))
    # Keep the original two sets for the SCC toggle checks and add a draw-only case.
    _, _, *middle, _, alt = SCALAR64_SETS[0]
    a = 0x11234567 << 32 | 0xffffffff
    b = middle[1] << 32 | middle[0]
    # s_not_b64 of a | b sets only bit 63, so its SCC depends on the high word alone.
    assert ~(a | b) & (2**64 - 1) == 1 << 63
    scalar64_cases.append((scalar64_case(directory, (0xffffffff, 0x11234567, *middle, 0x00400018, alt),
                                         name="scalar64-high.case"), "pass"))
    run_pixel_batch(renderer, backend, shader, scalar64_cases)
    # Mutations verified against Set 1 (where negative signed values and SCC=1 paths are active).
    draw = [renderer, backend, "--check-pixel-case", shader, scalar64_cases[0][0]]
    for old, new, label in (
        ("s_ashr_i64 s[72:73], s[52:53], s56", "s_lshr_b64 s[72:73], s[52:53], s56", "signed right shift"),
        ("s_bfe_i32 s72, s60, s61", "s_bfe_u32 s72, s60, s61", "signed bitfield extract"),
        ("s_andn2_b64 s[72:73], s[52:53], s[54:55]", "s_and_b64 s[72:73], s[52:53], s[54:55]", "andn2 inversion"),
        ("s_cmov_b64 s[78:79], -1", "s_cmov_b64 s[78:79], 0", "conditional move"),
        ("s_nand_b64 s[72:73], s[52:53], s[54:55]", "s_and_b64 s[72:73], s[52:53], s[54:55]", "nand inversion"),
        ("s_nor_b64 s[72:73], s[52:53], s[54:55]", "s_or_b64 s[72:73], s[52:53], s[54:55]", "nor inversion"),
        ("s_xnor_b64 s[72:73], s[52:53], s[54:55]", "s_xor_b64 s[72:73], s[52:53], s[54:55]", "xnor inversion"),
        ("s_nor_b64 s[72:73], s[82:83], -1", "s_nor_b64 s[72:73], s[82:83], 0", "nor SCC probe"),
    ):
        mutated = [line.replace(old, new) for line in program]
        assert mutated != program, label
        source.write_text(translate(mutated), encoding="ascii")
        compile_shader(compiler, source, shader)
        wrong = subprocess.run(draw, capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Mutation of {label} escaped detection")
    check_mad_draw(directory, compiler, renderer, backend)
    check_mad_literal_draw(directory, compiler, renderer, backend)
    source.write_text(translate(scalar()), encoding="ascii")
    compile_shader(compiler, source, shader, ieee=False)
    scalar_cases = [(scalar_case(directory, words, name=f"scalar-{idx}.case"), "pass")
                    for idx, words in enumerate(SCALAR_WORDS)]
    run_pixel_batch(renderer, backend, shader, scalar_cases)
    for old, new in SCALAR_MUTATIONS:
        case = scalar_case(directory, scalar_detecting_words(old, new))
        source.write_text(translate([line.replace(old, new) for line in scalar()]), encoding="ascii")
        compile_shader(compiler, source, shader, ieee=False)
        wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, case], capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Replacing {old} with {new} escaped detection")
    check_sop1_scalar_ops_draw(directory, compiler, renderer, backend)
    return 0


def check_sop1_scalar_ops_draw(directory, compiler, renderer, backend):
    source = directory / "sop1_scalar_ops.hlsl"
    shader = directory / "sop1_scalar_ops.cso"
    program = sop1_scalar_ops()

    def write_words(path, words, expected):
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *words)
        for word in expected:
            assert struct.unpack("<I", struct.pack("<f", as_float(word)))[0] == word, hex(word)
        write_case(path, bytes(constants), tuple(as_float(word) for word in expected), integer_rt=True)

    source.write_text(translate_int(program), encoding="ascii")
    compile_shader(compiler, source, shader)
    cases = []
    for idx, words in enumerate(SOP1_WORDS):
        case = directory / f"sop1_scalar_ops-{idx}.case"
        write_words(case, words, sop1_scalar_ops_reference(words))
        cases.append((case, "pass"))
    run_pixel_batch(renderer, backend, shader, cases)

    for old, new, label in (
        ("s_abs_i32 s50, s44", "s_not_b32 s50, s44", "signed scalar abs"),
        ("s_bitset0_b32 s52, s47", "s_bitset1_b32 s52, s47", "scalar bitset clear"),
    ):
        mutated = [line.replace(old, new) for line in program]
        assert mutated != program, label
        source.write_text(translate_int(mutated), encoding="ascii")
        compile_shader(compiler, source, shader)
        wrong = subprocess.run([renderer, backend, "--check-pixel-case", shader, cases[1][0]],
                               capture_output=True, text=True)
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"Mutating {label} escaped detection")

    # Both high bits start set; s45 equals s46, so the taken s_cmov_b32 leaves the value unchanged.
    words = (2, 0xc1400009, 0xc1400009, 31)
    set_bit = 4
    probe = [line.replace("s_bitset1_b32 s52, 31", f"s_bitset1_b32 s52, {set_bit}") for line in program]
    assert probe != program
    source.write_text(translate_int(probe), encoding="ascii")
    compile_shader(compiler, source, shader)
    probe_case = directory / "sop1_scalar_ops-probe.case"
    write_words(probe_case, words, sop1_scalar_ops_reference(words, set_bit))
    subprocess.run([renderer, backend, "--check-pixel-case", shader, probe_case], check=True)


def check_bfm_draw(directory, compiler, renderer, backend):
    """s_bfm_b32 and s_bfm_b64 against the Python reference, then controls that must mismatch."""
    source = directory / "bfm.hlsl"
    shader = directory / "bfm.cso"
    program = bfm_scalar_ops()
    lowered = translate_int(program)

    cases = []
    for idx, words in enumerate(BFM_WORDS):
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *words)
        expected = bfm_scalar_ops_reference(words)
        case = directory / f"bfm-{idx}.case"
        write_case(case, bytes(constants), tuple(as_float(word) for word in expected), integer_rt=True)
        cases.append(case)

    source.write_text(lowered, encoding="ascii")
    compile_shader(compiler, source, shader)
    run_pixel_batch(renderer, backend, shader, [(case, "pass") for case in cases])
    # Controls on the generated HLSL: a narrower count mask and swapped count/offset. Words 1 and 4 distinguish both.
    for old, new, label in (("(1u << (params[0].w & 31u))", "(1u << (params[0].w & 15u))", "b32 count mask"),
                            ("((1u << (params[0].w & 31u)) - 1u) << (params[1].x & 31u)",
                             "((1u << (params[1].x & 31u)) - 1u) << (params[0].w & 31u)", "b32 operand order")):
        mutated = lowered.replace(old, new)
        assert mutated != lowered, label
        source.write_text(mutated, encoding="ascii")
        compile_shader(compiler, source, shader)
        statuses, _ = run_pixel_batch_statuses(renderer, backend, shader, cases)
        if "mismatch" not in statuses:
            raise AssertionError(f"Mutating {label} escaped detection")


def check_brev_draw(directory, compiler, renderer, backend):
    """s_brev_b32 against the Python reference, then controls that must mismatch."""
    source = directory / "brev.hlsl"
    shader = directory / "brev.cso"
    program = brev_scalar_ops()
    lowered = translate_int(program)

    cases = []
    for idx, words in enumerate(BREV_WORDS):
        constants = bytearray(256)
        struct.pack_into("<4I", constants, 12, *words)
        expected = brev_scalar_ops_reference(words)
        case = directory / f"brev-{idx}.case"
        write_case(case, bytes(constants), tuple(as_float(word) for word in expected), integer_rt=True)
        cases.append(case)

    source.write_text(lowered, encoding="ascii")
    compile_shader(compiler, source, shader)
    run_pixel_batch(renderer, backend, shader, [(case, "pass") for case in cases])

    def expect_mismatch(text, label):
        source.write_text(text, encoding="ascii")
        compile_shader(compiler, source, shader)
        statuses, _ = run_pixel_batch_statuses(renderer, backend, shader, cases)
        if "mismatch" not in statuses:
            raise AssertionError(f"Mutating {label} escaped detection")

    # Control (a): a bitwise complement instead of the reversal.
    mutated = lowered.replace("reversebits(", "~(")
    assert mutated != lowered
    expect_mismatch(mutated, "reversebits to complement")
    # Control (b): the program writes SCC. s_not_b32 sets SCC to a nonzero result, so the select lane flips for the
    # word sets whose compare disagrees (set 3 turns SCC on, set 6 turns it off); the destinations stay unchanged
    # here, so only the select channel can reveal it.
    extra = program[:]
    at = extra.index("s_cselect_b32 s52, s47, 0")
    extra.insert(at, "s_not_b32 s54, s46")
    expect_mismatch(translate_int(extra), "SCC written after s_brev_b32")
    # Control (b'): the instruction itself replaced by one that writes SCC.
    swapped = [line.replace("s_brev_b32 s50, s46", "s_not_b32 s50, s46") for line in program]
    assert swapped != program
    expect_mismatch(translate_int(swapped), "s_brev_b32 replaced by s_not_b32")


def check_plain_bitscan_draw(directory, compiler, renderer, backend):
    """Plain bfrev/ffbh/ffbl/xnor against the Python reference, then program-text mutations that must mismatch."""
    source = directory / "plain_bitscan.hlsl"
    shader = directory / "plain_bitscan.cso"
    references = [plain_bitscan_reference(words) for words in PLAIN_BITSCAN_WORDS]
    lanes = {  # per mutation: (fixture, lane, lane value the mutated program produces for the words)
        "ffbh_i32 -> ffbh_u32": (0, 2, lambda w: plain_ffbh_u(w[2])),
        "ffbl_b32 -> ffbh_u32": (1, 0, lambda w: plain_ffbh_u(w[0])),
        "xnor_b32 -> xor_b32": (0, 3, lambda w: (w[3] ^ w[0]) & 0xffffffff),
        "bfrev_b32 -> not_b32": (0, 0, lambda w: ~w[0] & 0xffffffff),
    }
    # Before any GPU run: every mutated lane differs from the reference in at least one case.
    for label, (fixture, lane, value) in lanes.items():
        assert any(value(words) != ref[fixture][lane] for words, ref in zip(PLAIN_BITSCAN_WORDS, references)), label

    def make_cases(fixture):
        cases = []
        for idx, (words, ref) in enumerate(zip(PLAIN_BITSCAN_WORDS, references)):
            constants = bytearray(256)
            struct.pack_into("<4I", constants, 12, *words)
            case = directory / f"plain_bitscan-{fixture}-{idx}.case"
            write_case(case, bytes(constants), tuple(as_float(word) for word in ref[fixture]), integer_rt=True)
            cases.append(case)
        return cases

    cases = [make_cases(0), make_cases(1)]
    programs = [plain_bitscan_ops(), plain_bitscan_ops(True)]
    for fixture, program in enumerate(programs):
        source.write_text(translate_int(program), encoding="ascii")
        compile_shader(compiler, source, shader)
        run_pixel_batch(renderer, backend, shader, [(case, "pass") for case in cases[fixture]])

    for (old, new, label), fixture in ((("v_ffbh_i32 ", "v_ffbh_u32 ", "ffbh_i32 -> ffbh_u32"), 0),
                                       (("v_ffbl_b32_e32", "v_ffbh_u32_e32", "ffbl_b32 -> ffbh_u32"), 1),
                                       (("v_xnor_b32 ", "v_xor_b32 ", "xnor_b32 -> xor_b32"), 0),
                                       (("v_bfrev_b32 ", "v_not_b32 ", "bfrev_b32 -> not_b32"), 0)):
        assert lanes[label][0] == fixture, label
        mutated = [line.replace(old, new) for line in programs[fixture]]
        assert mutated != programs[fixture], label
        source.write_text(translate_int(mutated), encoding="ascii")
        compile_shader(compiler, source, shader)
        statuses, _ = run_pixel_batch_statuses(renderer, backend, shader, cases[fixture])



def check_swap_draw(directory, compiler, renderer, backend):
    """v_swap_b32 chain against the Python reference, then wrong lowerings that must mismatch."""
    source = directory / "swap.hlsl"
    shader = directory / "swap.cso"

    def write_cases(prefix):
        cases = []
        for idx, words in enumerate(SWAP_WORDS):
            constants = bytearray(256)
            struct.pack_into("<4I", constants, 12, *words)
            case = directory / f"{prefix}-{idx}.case"
            write_case(case, bytes(constants), tuple(as_float(word) for word in swap_ops_reference(words)),
                       integer_rt=True)
            cases.append(case)
        return cases

    cases = write_cases("swap")
    source.write_text(translate_int(swap_ops()), encoding="ascii")
    compile_shader(compiler, source, shader)
    run_pixel_batch(renderer, backend, shader, [(case, "pass") for case in cases])
    for label, body in swap_mutants():
        assert any(swap_emulate(body, words) != swap_ops_reference(words) for words in SWAP_WORDS), label
        source.write_text(translate_int(swap_ops(body)), encoding="ascii")
        compile_shader(compiler, source, shader)
        statuses, _ = run_pixel_batch_statuses(renderer, backend, shader, cases)
        if "mismatch" not in statuses:
            raise AssertionError(f"Mutating {label} escaped detection")



def check_int16_vop3_draw(directory, compiler, renderer, backend):
    """The 12 non-packed 16-bit integer VOP3 ops, a low16 chain and 16-bit compares against the Python reference,
    then mutations that must mismatch (one batch per mutation; the statuses are printed) and one that must be refused.

    Only the zero-extended low half of each result is observed. Whether hardware keeps or zeroes the destination's
    high half is not measured here, and the sentinel preloads do not detect a high-half leak.
    """
    check_int16_vop3_mutation_coverage()
    source = directory / "int16_vop3.hlsl"
    shader = directory / "int16_vop3.cso"

    def build(text):
        source.write_text(text, encoding="ascii")
        compile_shader(compiler, source, shader)

    def cases(group, tag):
        paths = []
        for idx, words in enumerate(INT16_VOP3_WORDS):
            constants = bytearray(256)
            struct.pack_into("<4I", constants, 12, *words)
            expected = int16_vop3_reference(group, words)
            path = directory / f"int16_vop3-{tag}-{group}-{idx}.case"
            write_case(path, bytes(constants), tuple(as_float(word) for word in expected), integer_rt=True)
            paths.append(path)
        return paths

    def detected(group, label, text):
        """Build the mutated shader and require at least one mismatching word set; print the statuses."""
        build(text)
        statuses, _ = run_pixel_batch_statuses(renderer, backend, shader, cases(group, "mut"))
        print(f"int16_vop3 mutation {label}: group {group} statuses {statuses}")
        if "mismatch" not in statuses:
            raise AssertionError(f"Mutating {label} escaped detection")

    programs = [int16_vop3_ops(group) for group in range(4)]
    lowered = [translate_int(program) for program in programs]
    for group in range(4):
        build(lowered[group])
        run_pixel_batch(renderer, backend, shader, [(path, "pass") for path in cases(group, "ref")])
    # (a) a source read from bits 31:16, applied to the first source word of every op in group 0.
    mutated = re.sub(r"(uint i16s\d+_0 = )(.+);", r"\1((\2) >> 16u);", lowered[0])
    assert mutated != lowered[0] and mutated.count(">> 16u);") == 4, "source half"
    detected(0, "the source high half", mutated)
    # (b) the 16-bit shift count masked to five bits instead of four.
    mutated = lowered[2].replace("& 15u)", "& 31u)")
    assert mutated != lowered[2] and mutated.count("& 31u)") == 3, "shift mask"
    detected(2, "the 16-bit shift count mask", mutated)
    # (c) signed min as unsigned min.
    mutated_program = [line.replace("v_min_i16", "v_min_u16") for line in programs[1]]
    assert mutated_program != programs[1], "signed min"
    detected(1, "signed min as unsigned min", translate_int(mutated_program))
    # (d) widening the low half with WORD_1 reads the undefined high half and must be refused at lowering.
    refused = [line.replace("src0_sel:WORD_0", "src0_sel:WORD_1") for line in programs[0]]
    assert refused != programs[0]
    expect_value_error(translate_int, refused, "Reads the undefined high half of a 16-bit result")
    # (e) a low16 result read through >> 16u: as the source of the chained op, and as the source of the first compare.
    add_index = programs[3].index("v_add_nc_u16 v24, v18, v19")
    chain_read = f"uint i16s{add_index + 1}_0 = asuint(alu{add_index});"
    assert lowered[3].count(chain_read) == 1, chain_read
    detected(3, "the low16 chain source", lowered[3].replace(
        chain_read, f"uint i16s{add_index + 1}_0 = (asuint(alu{add_index}) >> 16u);"))
    compare_index = programs[3].index("v_cmp_lt_u16 vcc_lo, v24, v21")
    compare_read = f"bool cmp{compare_index} = (asuint(alu{add_index}) & 0xffffu) <"
    assert lowered[3].count(compare_read) == 1, compare_read
    detected(3, "the low16 compare source", lowered[3].replace(
        compare_read, f"bool cmp{compare_index} = ((asuint(alu{add_index}) >> 16u) & 0xffffu) <"))


def check_int16_ternary_draw(directory, compiler, renderer, backend):
    """The 8 three-operand 16-bit integer VOP3 ops and a low16 chain against the Python reference, then mutations that
    must mismatch (one batch per mutation; the statuses are printed).

    Only the zero-extended low half of each result is observed (SDWA WORD_0). Whether hardware keeps or zeroes the
    destination's high half is not measured here, and the sentinel preloads do not detect a high-half leak.
    """
    check_int16_ternary_mutation_coverage()
    source = directory / "int16_ternary.hlsl"
    shader = directory / "int16_ternary.cso"

    def build(text):
        source.write_text(text, encoding="ascii")
        compile_shader(compiler, source, shader)

    def cases(group, tag):
        paths = []
        for idx, words in enumerate(INT16_TERNARY_WORDS):
            constants = bytearray(256)
            struct.pack_into("<4I", constants, 12, *words)
            expected = int16_ternary_reference(group, words)
            path = directory / f"int16_ternary-{tag}-{group}-{idx}.case"
            write_case(path, bytes(constants), tuple(as_float(word) for word in expected), integer_rt=True)
            paths.append(path)
        return paths

    def detected(group, label, text):
        """Build the mutated shader and require at least one mismatching word set; print the statuses."""
        build(text)
        statuses, _ = run_pixel_batch_statuses(renderer, backend, shader, cases(group, "mut"))
        print(f"int16_ternary mutation {label}: group {group} statuses {statuses}")
        if "mismatch" not in statuses:
            raise AssertionError(f"Mutating {label} escaped detection")

    programs = [int16_ternary_ops(group) for group in range(5)]
    lowered = [translate_int(program) for program in programs]
    for group in range(5):
        build(lowered[group])
        run_pixel_batch(renderer, backend, shader, [(path, "pass") for path in cases(group, "ref")])
    # (a) signed treated as unsigned: every sign-extension of a source half becomes a zero-extension.
    for group, label in ((1, "min3/max3 signed as unsigned"), (2, "med3 signed as unsigned"),
                         (4, "med3 chain signed as unsigned")):
        mutated = re.sub(r"\(asint\((i16s\d+_\d) << 16\) >> 16\)", r"(\1 & 0xffffu)", lowered[group])
        assert mutated != lowered[group] and "<< 16) >> 16" not in mutated, label
        detected(group, label, mutated)
    # (b) med3 replaced by min3.
    for group, label in ((2, "med3 replaced by min3"), (4, "chained med3 replaced by min3")):
        mutated_program = [line.replace("v_med3_", "v_min3_") for line in programs[group]]
        assert mutated_program != programs[group], label
        detected(group, label, translate_int(mutated_program))
    # (c) the mad addend dropped: v_mad_* x, a, b, c becomes v_mul_lo_u16 x, a, b.
    mutated_program = [re.sub(r"v_mad_[ui]16 (v\d+, [^,]+, [^,]+), [^,]+$", r"v_mul_lo_u16 \1", line)
                       for line in programs[0]]
    assert sum(line.startswith("v_mul_lo_u16") for line in mutated_program) == 4, "mad addend"
    detected(0, "the mad addend dropped", translate_int(mutated_program))
    # (d) a source read from bits 31:16, applied to the first source word of every op in groups 0 and 1.
    for group in (0, 1):
        mutated = re.sub(r"(uint i16s\d+_0 = )(.+);", r"\1((\2) >> 16u);", lowered[group])
        assert mutated != lowered[group] and mutated.count(">> 16u);") == 4, "source half"
        detected(group, "the source high half", mutated)
    # (e) widening the low half with WORD_1 reads the undefined high half and must be refused at lowering.
    refused = [line.replace("src0_sel:WORD_0", "src0_sel:WORD_1") for line in programs[0]]
    assert refused != programs[0]
    expect_value_error(translate_int, refused, "Reads the undefined high half of a 16-bit result")

FLAT_SCRATCH_WORDS = (
    (0x12345678, 0x9abcdef0, 0x00ff00ff, 0xaa55aa55),
    (0x00000001, 0x80000000, 0x7fffffff, 0xffffffff),
    (0x01020304, 0x05060708, 0x090a0b0c, 0x0d0e0f10),
    (0xdeadbeef, 0xcafebabe, 0xfeedface, 0xbaadf00d),
    (0x00000000, 0x11111111, 0x22222222, 0x33333333),
)


def flat_scratch_ops_group1():
    return [
        "s_mov_b32 s48, 0",
        # 1-dword store & load
        "v_mov_b32_e32 v10, s44",
        "scratch_store_dword off, v10, s48 offset:16",
        "scratch_load_dword v12, off, s48 offset:16",
        # 2-dword store & load
        "v_mov_b32_e32 v10, s45",
        "v_mov_b32_e32 v11, s46",
        "scratch_store_dwordx2 off, v[10:11], s48 offset:32 glc",
        "scratch_load_dwordx2 v[13:14], off, s48 offset:32 slc",
        # 1-dword store & load with saddr offset 64
        "v_mov_b32_e32 v10, s47",
        "scratch_store_dword off, v10, s48 offset:64",
        "scratch_load_dword v15, off, s48 offset:64",
        "s_waitcnt vmcnt(0)",
    ]


def flat_scratch_ops_group2():
    return [
        "s_mov_b32 s48, 0",
        # 3-dword store & load
        "v_mov_b32_e32 v10, s44",
        "v_mov_b32_e32 v11, s45",
        "v_mov_b32_e32 v12, s46",
        "scratch_store_dwordx3 off, v[10:12], s48 offset:16 glc",
        "scratch_load_dwordx3 v[20:22], off, s48 offset:16 dlc",
        # 4-dword store & load
        "v_mov_b32_e32 v13, s47",
        "scratch_store_dwordx4 off, v[10:13], s48 offset:64",
        "scratch_load_dwordx4 v[23:26], off, s48 offset:64",
        "s_waitcnt vmcnt(0)",
        "v_mov_b32_e32 v12, v23",
        "v_mov_b32_e32 v13, v24",
        "v_mov_b32_e32 v14, v21",
        "v_mov_b32_e32 v15, v26",
    ]


def check_flat_scratch_controls():
    """Scratch dword loads and stores validate addressing, alignments, register spans and modifiers."""
    p1 = with_constants([
        "s_mov_b32 s48, 0",
        "scratch_store_dword off, v12, s48 offset:16",
        "scratch_load_dword v13, off, s48 offset:16",
        "s_waitcnt vmcnt(0)",
    ])
    src1 = translate(p1)
    assert "uint scratch[256]" in src1
    assert "scratch[4u] = asuint(" in src1
    assert "precise float alu" in src1

    p2 = with_constants([
        "s_mov_b32 s48, 0",
        "scratch_store_dwordx2 off, v[12:13], s48 offset:32",
        "scratch_load_dwordx2 v[14:15], off, s48 offset:32",
        "s_waitcnt vmcnt(0)",
    ])
    src2 = translate(p2)
    assert "scratch[8u] = asuint(" in src2
    assert "scratch[8u + 1u] = asuint(" in src2

    # Rejection controls:
    # 1. Address bounds and alignment
    expect_value_error(translate, with_constants(["s_mov_b32 s48, 0", "scratch_store_dword off, v12, s48 offset:2048"]), "range")
    expect_value_error(translate, with_constants(["s_mov_b32 s48, 0", "scratch_store_dword off, v12, s48 offset:-2049"]), "range")
    expect_value_error(translate, with_constants(["s_mov_b32 s48, 0", "scratch_store_dword off, v12, s48 offset:5"]), "4-byte aligned")
    expect_value_error(translate, with_constants(["s_mov_b32 s48, 0", "scratch_store_dword off, v12, s48 offset:2"]), "4-byte aligned")
    expect_value_error(translate, with_constants(["s_mov_b32 s48, 0", "scratch_store_dword off, v12, s48 offset:-4"]), "Negative scratch address")
    # 2. Dynamic or unproven access
    expect_value_error(translate, with_constants(["s_mov_b32 s48, 0", "v_mov_b32_e32 v1, 0", "scratch_load_dword v12, v1, off"]), "Dynamic or unproven scratch address")
    expect_value_error(translate, with_constants(["s_mov_b32 s48, 0", "v_mov_b32_e32 v1, 0", "scratch_store_dword v1, v12, off"]), "Dynamic or unproven scratch address")
    expect_value_error(translate, with_constants(["s_mov_b32 s4, s44", "scratch_load_dword v12, off, s4"]), "Dynamic or unproven scratch address")
    expect_value_error(translate, with_constants(["scratch_load_dword v12, off, s60"]), "Uninitialized or pending scalar register")
    # 3. Read without defining store (load before store)
    expect_value_error(translate, with_constants(["s_mov_b32 s48, 0", "scratch_load_dword v12, off, s48 offset:16"]), "unwritten address")
    # 4. Consumer before matching waitcnt
    pending = [
        "s_mov_b32 s48, 0",
        "scratch_store_dword off, v12, s48 offset:16",
        "scratch_load_dword v13, off, s48 offset:16",
    ]
    consumer = ["v_mov_b32_e32 v14, v13"]
    for partial in ([], ["s_waitcnt vmcnt(1)"], ["s_waitcnt_vmcnt null, 1"]):
        expect_value_error(translate, with_constants(pending + partial + consumer), "Sample used before vmcnt wait")
    translate(with_constants(pending + ["s_waitcnt_vmcnt null, 0"] + consumer))
    # 5. Invalid registers, widths, spans, modifiers
    expect_value_error(translate, with_constants(["s_mov_b32 s48, 0", "scratch_store_dword off, v256, s48"]), "range")
    expect_value_error(translate, with_constants(["s_mov_b32 s48, 0", "scratch_load_dword v12, off, s106"]), "Invalid saddr register")
    expect_value_error(translate, with_constants(["s_mov_b32 s48, 0", "scratch_store_dwordx2 off, v[4:6], s48"]), "width")
    expect_value_error(translate, with_constants(["s_mov_b32 s48, 0", "scratch_store_dwordx4 off, v[4:6], s48"]), "width")
    expect_value_error(translate, with_constants(["s_mov_b32 s48, 0", "scratch_store_dwordx2 off, v4, s48"]), "span")
    expect_value_error(translate, with_constants(["s_mov_b32 s48, 0", "scratch_store_dword off, v12, s48 lds"]), "LDS")
    expect_value_error(translate, with_constants(["s_mov_b32 s48, 0", "scratch_store_dword off, v12, s48 clamp"]), "modifier")
    expect_value_error(translate, with_constants(["s_mov_b32 s48, 0", "scratch_load_dword v12, off"]), "3 operands")
    expect_value_error(translate, with_constants(["s_mov_b32 s48, 0", "scratch_load_dword_e32 v12, off, s48"]), "Unsupported instruction")
    expect_value_error(translate, with_constants(["s_mov_b32 s48, 0", "scratch_load_dword_e64 v12, off, s48"]), "Unsupported instruction")


def check_flat_scratch_draws(directory, compiler, renderer, backend):
    """Offscreen readback for scratch 1, 2, 3, 4 dword loads and stores across two fixtures."""
    source = directory / "flat_scratch.hlsl"
    shader = directory / "flat_scratch.cso"

    def make_cases(group):
        cases = []
        for idx, words in enumerate(FLAT_SCRATCH_WORDS):
            constants = bytearray(256)
            struct.pack_into("<4I", constants, 12, *words)
            case_path = directory / f"scratch-{group}-{idx}.case"
            expected_words = words if group == 0 else (words[0], words[1], words[1], words[3])
            expected = tuple(as_float(w) for w in expected_words)
            write_case(case_path, bytes(constants), expected, integer_rt=True)
            cases.append(case_path)
        return cases

    cases = [make_cases(0), make_cases(1)]
    programs = [uncompressed_export(flat_scratch_ops_group1()), uncompressed_export(flat_scratch_ops_group2())]

    for fixture, program in enumerate(programs):
        source.write_text(translate_int(program), encoding="ascii")
        compile_shader(compiler, source, shader)
        run_pixel_batch(renderer, backend, shader, [(case, "pass") for case in cases[fixture]])

    for (old, new, label), fixture in (
        (("v_mov_b32_e32 v10, s44", "v_mov_b32_e32 v10, s45", "v10 data s44->s45"), 0),
        (("v_mov_b32_e32 v11, s46", "v_mov_b32_e32 v11, s45", "v11 data s46->s45"), 0),
        (("scratch_store_dword off, v10, s48 offset:64", "scratch_store_dword off, v11, s48 offset:64", "v10->v11"), 0),
        (("v_mov_b32_e32 v10, s44", "v_mov_b32_e32 v10, s45", "g2 v10 data s44->s45"), 1),
        (("v_mov_b32_e32 v13, s47", "v_mov_b32_e32 v13, s44", "g2 v13 data s47->s44"), 1),
    ):
        mutated = [line.replace(old, new) for line in programs[fixture]]
        assert mutated != programs[fixture], label
        source.write_text(translate_int(mutated), encoding="ascii")
        compile_shader(compiler, source, shader)
        statuses, _ = run_pixel_batch_statuses(renderer, backend, shader, cases[fixture])
        if "mismatch" not in statuses:
            raise AssertionError(f"Mutating {label} escaped detection")


def check_f16_arith_draw(directory, compiler, renderer, backend):
    """The four plain binary16 VOP2 ops against the Python reference: 40 pairs of baked operands, every accepted
    reader and two chains, all through the lossless R32G32B32A32_UINT readback. Then a binary32 -denorm ftz build of the
    programs with subnormal operands or results, which must still pass, and 11 deviations of the lowering that must
    each mismatch (one batch per mutation; the statuses are printed).

    Only the zero-extended low half of each result is observed. Whether hardware keeps or zeroes the destination's
    high half is not measured here; a source read from bits 31:16 is the high-half control.
    """
    check_f16_arith_mutation_coverage()
    source = directory / "f16_arith.hlsl"
    shader = directory / "f16_arith.cso"
    constants = bytes(256)

    def build(text, mode=F32_DENORM_MODE):
        source.write_text(text, encoding="ascii")
        compile_shader(compiler, source, shader, mode=mode)

    def case(kind, index, tag):
        path = directory / f"f16_arith-{tag}-{kind}-{index}.case"
        write_case(path, constants, tuple(as_float(word) for word in f16_arith_expected(kind, index)), integer_rt=True)
        return path

    programs = {key: f16_arith_program(*key) for key in F16_ARITH_PROGRAMS}
    for key in F16_ARITH_PROGRAMS:
        build(translate_f16(programs[key]))
        run_pixel_batch(renderer, backend, shader, [(case(*key, "ref"), "pass")])
    # Binary32 flushing cannot reach binary16 values: every operand, product and sum is a binary32 normal.
    subnormal = [key for key in F16_ARITH_PROGRAMS if any(
        half & 0x7c00 == 0 and half & 0x3ff for half in f16_arith_source_halves(*key))]
    assert subnormal and len(subnormal) < len(F16_ARITH_PROGRAMS), subnormal
    for key in subnormal:
        build(translate_f16(programs[key]), mode="ftz")
        run_pixel_batch(renderer, backend, shader, [(case(*key, "ftz"), "pass")])
    for mutation in F16_ARITH_MUTATIONS:
        key = f16_arith_detecting_program(mutation)
        build(f16_arith_mutated(mutation, programs[key]))
        statuses, _ = run_pixel_batch_statuses(renderer, backend, shader, [case(*key, "mut")])
        print(f"f16_arith mutation {mutation}: program {key} statuses {statuses}")
        if "mismatch" not in statuses:
            raise AssertionError(f"Mutating {mutation} escaped detection")
    # A widening that selects WORD_1 reads the undefined high half and must be refused at lowering.
    refused = [line.replace("src0_sel:WORD_0", "src0_sel:WORD_1") for line in programs["reader", 0]]
    assert refused != programs["reader", 0]
    expect_value_error(translate_f16, refused, "Reads the undefined high half of a 16-bit result")


def check_e64_render(directory, compiler, renderer, backend):
    """The e64 draw cases on their own, so they run as a separate CTest entry per backend. A renderer that reports an
    unavailable adapter (exit 77) makes this a skip, as for the other pixel draws."""
    try:
        check_e64_draws(directory, compiler, renderer, backend)
    except subprocess.CalledProcessError as error:
        if error.returncode == 77:
            return 77
        raise
    check_denorm_mode_draw(directory, compiler, renderer, backend)
    check_f16_denorm_draw(directory, compiler, renderer, backend)
    check_nan_class_draw(directory, compiler, renderer, backend)
    check_integer_readback_draw(directory, compiler, renderer, backend)
    check_subrev_u32_draw(directory, compiler, renderer, backend)
    check_lshl_add_u32_draw(directory, compiler, renderer, backend)
    check_absdiff_i32_draw(directory, compiler, renderer, backend)
    check_readfirstlane_draw(directory, compiler, renderer, backend)
    check_valu_int32_draw(directory, compiler, renderer, backend)
    check_cvt_int_f32_draw(directory, compiler, renderer, backend)
    check_cvt_int_f32_e64_draw(directory, compiler, renderer, backend)
    check_cvt_float_int_draw(directory, compiler, renderer, backend)
    return 0


def main():
    if len(sys.argv) == 1:
        with tempfile.TemporaryDirectory() as tmp:
            check_controls(Path(tmp))
        return 0
    mode, *args = sys.argv[1:]
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        if mode in ("draw-blend-scalar", "draw-predicate-rounding", "draw-integer-nc",
                    "draw-e64-integer", "draw-bfm", "draw-brev", "draw-align",
                    "draw-sdwa-integer", "draw-plain-bitscan", "draw-vop3-ternary-int", "draw-swap",
                    "draw-mac-legacy", "draw-carry-out", "draw-carry-in", "draw-pack-int16", "draw-sad",
                    "draw-int16-vop3", "draw-int16-ternary", "draw-compressed-export", "draw-uncompressed-export",
                    "draw-title-loads", "draw-ds-swizzle", "draw-exec-regions", "draw-div-fmas", "draw-cmp64",
                    "draw-f16-arith", "draw-flat-scratch"):
            try:
                draw = {"draw-blend-scalar": check_blend_and_scalar_draws,
                        "draw-predicate-rounding": check_predicate_and_rounding_draws,
                        "draw-integer-nc": check_integer_nc_draws,
                        "draw-e64-integer": check_e64_integer_draws,
                        "draw-carry-out": check_carry_out_draws,
                        "draw-carry-in": check_carry_in_draws,
                        "draw-sdwa-integer": check_sdwa_integer_draws,
                        "draw-bfm": check_bfm_draw,
                        "draw-brev": check_brev_draw,
                        "draw-align": check_align_draws,
                        "draw-plain-bitscan": check_plain_bitscan_draw,
                        "draw-vop3-ternary-int": check_vop3_ternary_int_draws,
                        "draw-swap": check_swap_draw,
                        "draw-mac-legacy": check_mac_legacy_draws,
                        "draw-pack-int16": check_pack_int16_draws,
                        "draw-sad": check_sad_draws,
                        "draw-int16-vop3": check_int16_vop3_draw,
                        "draw-int16-ternary": check_int16_ternary_draw,
                        "draw-f16-arith": check_f16_arith_draw,
                        "draw-compressed-export": check_compressed_export_draw,
                        "draw-uncompressed-export": check_uncompressed_export_draw,
                        "draw-title-loads": check_title_load_draws,
                        "draw-ds-swizzle": check_ds_swizzle_draw,
                        "draw-exec-regions": check_masked_draw,
                        "draw-div-fmas": check_valu_div_fmas_draw,
                        "draw-cmp64": check_cmp64_draw,
                        "draw-flat-scratch": check_flat_scratch_draws}[mode]
                return draw(directory, *args)
            except subprocess.CalledProcessError as error:
                if error.returncode == 77 and str(error.cmd[0]) == str(args[1]):
                    return 77
                raise
        if mode == "draw-first-bit":
            try:
                return check_scalar_first_bit_draw(directory, *args)
            except subprocess.CalledProcessError as error:
                if error.returncode == 77 and str(error.cmd[0]) == str(args[1]):
                    return 77
                raise
        if mode == "draw-cvt-float-int-e64":
            try:
                return check_cvt_float_int_e64_draw(directory, *args)
            except subprocess.CalledProcessError as error:
                if error.returncode == 77 and str(error.cmd[0]) == str(args[1]):
                    return 77
                raise
        if mode == "draw-cvt-flr-rpi":
            try:
                return check_cvt_flr_rpi_draw(directory, *args)
            except subprocess.CalledProcessError as error:
                if error.returncode == 77 and str(error.cmd[0]) == str(args[1]):
                    return 77
                raise
        if mode == "draw-bitset64":
            try:
                return check_scalar_bitset64_draw(directory, *args)
            except subprocess.CalledProcessError as error:
                if error.returncode == 77 and str(error.cmd[0]) == str(args[1]):
                    return 77
                raise
        if mode == "draw-frexp-exp":
            try:
                return check_frexp_exp_draws(directory, *args)
            except subprocess.CalledProcessError as error:
                if error.returncode == 77 and str(error.cmd[0]) == str(args[1]):
                    return 77
                raise
        if mode == "draw-comparisons":
            try:
                return check_comparison_draws(directory, *args)
            except subprocess.CalledProcessError as error:
                if error.returncode == 77 and str(error.cmd[0]) == str(args[1]):
                    return 77
                raise
        if mode == "draw-e64-spellings":
            return check_e64_spelling_draws(directory, *args)
        if mode == "draw-e64":
            # The controls already run in the binary and draw modes; this mode only draws.
            return check_e64_render(directory, *args)
        if mode == "draw-e64-approx":
            return check_e64_approx_draws(directory, *args)
        if mode == "draw-sdwa-unary":
            return check_sdwa_unary_draws(directory, *args)
        check_controls(directory)
        try:
            if mode == "binary":
                check_binary(directory, *args)
            elif mode == "bits":
                return check_bit_draws(directory, *args)
            elif mode in ("draw", "draw-core"):
                return check_render(directory, *args, core_only=mode == "draw-core")
            elif mode == "three-way":
                return check_three_way_draw(directory, *args)
            elif mode == "buffer-color":
                return check_buffer_color_draw(directory, *args)
            else:
                raise ValueError("Expected binary, draw, draw-e64, three-way or buffer-color mode")
        except subprocess.CalledProcessError as error:
            if error.returncode == 77 and len(args) > 1 and str(error.cmd[0]) == str(args[1]):
                return 77
            raise


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:  # also on skip or failure; tools/form_coverage.py reads the test's status from JUnit
        write_record(os.environ.get("FORM_COVERAGE_RECORD"))
