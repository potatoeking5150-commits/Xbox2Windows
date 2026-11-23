"""Verify partial dmask image_sample (0x1..0xf) and non-zero texture descriptor offsets in pixel_aot."""
from fractions import Fraction
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from pixel_aot_check import RAMP, dual_program, half_rtz, lower, must_reject, single

SAMPLER = (0x92, 0xc00000, 0x14500000, 0)
W_G = struct.unpack("<f", bytes.fromhex("7424373f"))[0]
W_R = struct.unpack("<f", bytes.fromhex("9a99593e"))[0]
W_B = struct.unpack("<f", bytes.fromhex("2aa9933d"))[0]


def fma32(a, b, c):
    return single(float(Fraction(a) * Fraction(b) + Fraction(c)))


def make_header(tex_offset):
    off = "null" if tex_offset == 0 else f"0x{tex_offset:x}"
    return [
        "s_inst_prefetch 0x3",
        "s_mov_b64 s[12:13], exec",
        "s_wqm_b64 exec, exec",
        "s_mov_b32 s2, s9",
        "s_mov_b32 s9, 0",
        "s_mov_b32 m0, s2",
        f"s_load_dwordx8 s[4:11], s[8:9], {off}",
        "v_interp_p1_f32_e32 v2, v0, attr0.x",
        "v_interp_p1_f32_e32 v3, v0, attr0.y",
        "v_interp_p2_f32_e32 v2, v1, attr0.x",
        "v_interp_p2_f32_e32 v3, v1, attr0.y",
        "s_and_b64 exec, exec, s[12:13]",
        "s_movk_i32 s0, 0x92",
        "s_mov_b32 s1, 0xc00000",
        "s_mov_b32 s2, 0x14500000",
        "s_mov_b32 s3, 0",
        "s_waitcnt lgkmcnt(0)",
    ]


def luma_rgb_program(tex_offset=0x20):
    """3-channel RGB (dmask:0x7) Rec.709 luma reduction loaded from non-zero descriptor offset."""
    return [
        *make_header(tex_offset),
        "image_sample v[0:2], v[2:3], s[4:11], s[0:3] dmask:0x7 dim:SQ_RSRC_IMG_2D",
        "s_waitcnt vmcnt(0)",
        "v_mul_legacy_f32_e32 v3, 0x3f372474, v1",
        "v_fmac_f32_e32 v3, 0x3e59999a, v0",
        "v_fmac_f32_e32 v3, 0x3d93a92a, v2",
        "v_cvt_pkrtz_f16_f32_e32 v0, v3, v3",
        "v_cvt_pkrtz_f16_f32_e32 v1, v3, v3",
        "exp mrt0 v0, v0, v1, v1 done compr vm",
        "s_endpgm",
    ]


def single_channel_program(dmask, tex_offset=0x40):
    """1-channel single-VGPR sample (dmask:0x1 -> R, 0x2 -> G, 0x4 -> B, 0x8 -> A)."""
    return [
        *make_header(tex_offset),
        f"image_sample v4, v[2:3], s[4:11], s[0:3] dmask:0x{dmask:x} dim:SQ_RSRC_IMG_2D",
        "s_waitcnt vmcnt(0)",
        "v_cvt_pkrtz_f16_f32_e32 v0, v4, v4",
        "v_cvt_pkrtz_f16_f32_e32 v1, v4, v4",
        "exp mrt0 v0, v0, v1, v1 done compr vm",
        "s_endpgm",
    ]


def sparse_two_channel_program(dmask, tex_offset=0x0):
    """2-channel sparse sample (e.g. dmask:0x5 -> R,B or dmask:0xa -> G,A)."""
    return [
        *make_header(tex_offset),
        f"image_sample v[4:5], v[2:3], s[4:11], s[0:3] dmask:0x{dmask:x} dim:SQ_RSRC_IMG_2D",
        "s_waitcnt vmcnt(0)",
        "v_cvt_pkrtz_f16_f32_e32 v0, v4, v5",
        "v_cvt_pkrtz_f16_f32_e32 v1, v5, v4",
        "exp mrt0 v0, v0, v1, v1 done compr vm",
        "s_endpgm",
    ]


def sparse_rga_program():
    return [
        *make_header(0),
        "v_mov_b32_e32 v7, 0.25",
        "image_sample v[4:6], v[2:3], s[4:11], s[0:3] dmask:0xb dim:SQ_RSRC_IMG_2D",
        "s_waitcnt vmcnt(0)",
        "v_cvt_pkrtz_f16_f32_e32 v0, v4, v5",
        "v_cvt_pkrtz_f16_f32_e32 v1, v6, v7",
        "exp mrt0 v0, v0, v1, v1 done compr vm",
        "s_endpgm",
    ]


def build_case(eval_fn):
    data = bytearray(256)
    for y in RAMP:
        for x in RAMP:
            r, g, b, a = single(x / 255), single(y / 255), single(128 / 255), 1.0
            data += struct.pack("<4f", *eval_fn(r, g, b, a))
    return data


def check_rejections():
    base = luma_rgb_program(0x20)
    # Wrong texture_offset (default 0 does not match 0x20 load)
    must_reject(lambda: lower(base, 8, 9, SAMPLER))
    # Unaligned or negative texture_offset
    must_reject(lambda: lower(base, 8, 9, SAMPLER, texture_offset=16))
    must_reject(lambda: lower(base, 8, 9, SAMPLER, texture_offset=-32))
    # texture_offset without sampler
    must_reject(lambda: lower(base, 8, 9, None, texture_offset=32))
    # Overlapping texture_offset and second_texture_offset
    must_reject(lambda: lower(base, 8, 9, SAMPLER, texture_offset=32, second_texture_offset=32))
    for invalid in (-32, 16):
        must_reject(lambda: lower(base, 8, 9, SAMPLER, texture_offset=32, second_texture_offset=invalid))
    # Overlapping texture_offset and constant_descriptor
    must_reject(lambda: lower(base, 8, 9, SAMPLER, constant_descriptor=0x24, texture_offset=0x20))
    must_reject(lambda: lower(base, 8, 9, SAMPLER, texture_offset=0x20, extra_constants=[(8, 0x24)]))
    # dmask:0x0 rejected
    must_reject(lambda: lower([s.replace("dmask:0x7", "dmask:0x0") for s in base], 8, 9, SAMPLER, texture_offset=0x20))
    # VGPR width mismatch (dmask:0x7 needs 3 VGPRs, not 4 or 2)
    must_reject(lambda: lower([s.replace("v[0:2]", "v[0:3]") for s in base], 8, 9, SAMPLER, texture_offset=0x20))
    must_reject(lambda: lower([s.replace("v[0:2]", "v[0:1]") for s in base], 8, 9, SAMPLER, texture_offset=0x20))
    # Single-channel must use vN (not v[N:N]); multi-channel must use v[a:b] (not vN)
    one = single_channel_program(0x1, 0x40)
    must_reject(lambda: lower([s.replace("image_sample v4,", "image_sample v256,") for s in one],
                              8, 9, SAMPLER, texture_offset=0x40))
    must_reject(lambda: lower([s.replace("image_sample v4,", "image_sample v[4:4],") for s in one], 8, 9, SAMPLER, texture_offset=0x40))
    must_reject(lambda: lower([s.replace("dmask:0x1", "dmask:0x3") for s in one], 8, 9, SAMPLER, texture_offset=0x40))


def check_adjacent_descriptors():
    first = "s_load_dwordx8 s[28:35], s[20:21], null"
    second = "s_load_dwordx8 s[56:63], s[20:21], 0x20"
    program = dual_program()
    program[program.index(first)] = first.replace("null", "0x20")
    program[program.index(second)] = second.replace("0x20", "null")
    source = lower(program, 20, 24, [1, 2, 3, 4], texture_offset=32, second_texture_offset=0)
    assert source.index("= tex.Sample(") < source.index("= tex1.Sample(")
    must_reject(lambda: lower(program, 20, 24, [1, 2, 3, 4], 0x10, texture_offset=32, second_texture_offset=0))


def run(directory, compiler, renderer, backend):
    check_rejections()
    check_adjacent_descriptors()

    def eval_luma(r, g, b, a):
        y1 = fma32(W_R, r, single(W_G * g))
        y2 = fma32(W_B, b, y1)
        yh = half_rtz(y2)
        return (yh, yh, yh, yh)

    cases = [
        ("luma_rgb_0x20", luma_rgb_program(0x20), 0x20, eval_luma),
        ("single_r_0x40", single_channel_program(0x1, 0x40), 0x40, lambda r, g, b, a: (half_rtz(r),) * 4),
        ("single_g_0x40", single_channel_program(0x2, 0x40), 0x40, lambda r, g, b, a: (half_rtz(g),) * 4),
        ("single_a_0x40", single_channel_program(0x8, 0x40), 0x40, lambda r, g, b, a: (half_rtz(a),) * 4),
        ("sparse_rb_0x0", sparse_two_channel_program(0x5, 0x0), 0x0,
         lambda r, g, b, a: (half_rtz(r), half_rtz(b), half_rtz(b), half_rtz(r))),
        ("sparse_ga_0x0", sparse_two_channel_program(0xa, 0x0), 0x0,
         lambda r, g, b, a: (half_rtz(g), half_rtz(a), half_rtz(a), half_rtz(g))),
        ("sparse_rga_0x0", sparse_rga_program(), 0x0,
         lambda r, g, b, a: (half_rtz(r), half_rtz(g), half_rtz(a), 0.25)),
    ]

    for tag, prog, tex_off, eval_fn in cases:
        hlsl = directory / f"{tag}.hlsl"
        cso = directory / f"{tag}.cso"
        case_file = directory / f"{tag}.case"
        hlsl.write_text(lower(prog, 8, 9, SAMPLER, texture_offset=tex_off), encoding="ascii")
        subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", cso, hlsl], check=True)
        case_file.write_bytes(build_case(eval_fn))
        res = subprocess.run([renderer, backend, "--check-pixel-case", cso, case_file])
        if res.returncode != 0:
            return res.returncode

    # Change either the selected channel or the unsampled neighbor.
    clobbered = sparse_rga_program()
    clobbered.insert(clobbered.index("s_waitcnt vmcnt(0)") + 1, "v_mov_b32_e32 v7, v6")
    bad_hlsl = directory / "bad_dmask.hlsl"
    bad_cso = directory / "bad_dmask.cso"
    for program, tag in ((sparse_two_channel_program(0x6), "sparse_rb_0x0"), (clobbered, "sparse_rga_0x0")):
        bad_hlsl.write_text(lower(program, 8, 9, SAMPLER), encoding="ascii")
        subprocess.run([compiler, "-Gis", "-T", "ps_6_0", "-E", "main", "-Fo", bad_cso, bad_hlsl], check=True)
        wrong = subprocess.run(
            [renderer, backend, "--check-pixel-case", bad_cso, directory / f"{tag}.case"],
            capture_output=True, text=True,
        )
        if wrong.returncode != 1 or "pixel_mismatch" not in wrong.stderr:
            raise AssertionError(f"{tag}: changed output escaped detection")

    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        sys.exit(run(Path(tmp), *sys.argv[1:]))
