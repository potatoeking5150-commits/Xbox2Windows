"""Translate a source-built mesh triangle and compare exports and rasterization."""

from pathlib import Path
import json
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from mesh_entry_aot import lower

# Two explicitly modeled wave64s. Only the first three group threads export vertices.
PROGRAM = """
s_mov_b32 s10, s17
s_mov_b32 s11, 0
s_load_dwordx4 s[24:27], s[10:11], 0x20
s_load_dwordx4 s[16:19], s[10:11], 0x60
s_load_dwordx4 s[20:23], s[10:11], 0x80
s_waitcnt lgkmcnt(0)
s_buffer_load_dword s15, s[24:27], null
s_buffer_load_dword s30, s[16:19], 0x4
s_buffer_load_dword s31, s[20:23], 0xc
s_waitcnt lgkmcnt(0)
v_mbcnt_lo_u32_b32 v0, -1, 0
v_mbcnt_hi_u32_b32 v0, -1, v0
s_bfe_u32 s8, s3, 0x40018
v_lshl_add_u32 v0, s8, 6, v0
s_mov_b64 s[22:23], exec
v_cmpx_eq_u32_e32 0, v0
s_cbranch_execz 2
s_mov_b32 s7, s15
s_mov_b32 m0, s7
s_cbranch_execz 1
s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)
v_mov_b32_e32 v9, 0x200400
exp prim, v9, off, off, off done
s_mov_b64 exec, s[22:23]
v_cmpx_gt_u32_e32 3, v0
v_cmp_eq_u32_e64 s[10:11], 1, v0
v_cmp_eq_u32_e64 s[12:13], 2, v0
v_mov_b32_e32 v8, 0x40400000
v_mov_b32_e32 v9, 0xbf800000
v_cndmask_b32_e64 v1, v9, v8, s[10:11]
v_cndmask_b32_e64 v2, v9, v8, s[12:13]
v_mov_b32_e32 v3, s30
v_add_f32_e32 v3, s31, v3
v_mov_b32_e32 v4, 0x3f800000
exp pos0, v1, v2, v3, v4 done
s_endpgm
""".strip().splitlines()
TRIANGLE = [(-1, -1, 0.5, 1), (3, -1, 0.5, 1), (-1, 3, 0.5, 1)]
PIXEL = "float4 main() : SV_Target { return float4(0,1,0,1); }"


def fixture(constants, positions, visible, traced):
    trace = bytearray([0xCD] * 80)
    if traced:
        struct.pack_into("<2I", trace, 0, 3, 1)
        for i, vertex in enumerate(positions):
            struct.pack_into("<4f", trace, 16 + i * 16, *vertex)
        struct.pack_into("<3I", trace, 64, 0, 1, 2)
    pixels = (
        visible
        if isinstance(visible, bytes)
        else bytes([0, 255 if visible else 0, 0, 255]) * 1024
    )
    return constants + trace + pixels


def draw(dxc, renderer, adapter, tmp, source, cases, *, indexed=False, pixel_source=PIXEL):
    pixel = tmp / "pixel.hlsl"
    pixel.write_text(pixel_source)
    ps = tmp / "pixel.dxil"
    subprocess.run([dxc, "-T", "ps_6_0", "-E", "main", "-Fo", ps, pixel], check=True)
    shader = tmp / "mesh.hlsl"
    shader.write_text(source)
    for traced in (False, True):
        ms = tmp / "mesh.dxil"
        flags = ["-DTRACE_EXPORTS"] if traced else []
        subprocess.run(
            [dxc, "-T", "ms_6_5", "-E", "main", *flags, "-Fo", ms, shader], check=True
        )
        for constants, positions, visible in cases:
            path = tmp / "draw.bin"
            data = fixture(constants, positions, visible, traced)
            path.write_bytes(data)
            args = [
                renderer,
                adapter,
                "--mesh-indexed" if indexed else "--mesh-entry",
                ms,
                ps,
                path,
            ]
            result = subprocess.run(args)
            if result.returncode == 77:
                return 77
            result.check_returncode()
            if traced:
                bad = bytearray(data)
                bad[len(constants) + 64] ^= 1
                path.write_bytes(bad)
                result = subprocess.run(args, capture_output=True, text=True)
                if (
                    result.returncode != 1
                    or "Mesh export trace mismatch" not in result.stderr
                ):
                    raise AssertionError(
                        "Unexpected result for a wrong primitive index"
                    )
    return 0


def triangle_inputs():
    inputs = bytearray(1024)
    struct.pack_into("<f", inputs, 4, 0.125)
    struct.pack_into("<f", inputs, 512 + 12, 0.375)
    return inputs


def enumerate_pcs(program):
    # All fixture branch displacements span only scalar four-byte instructions.
    return [(i * 4, line) for i, line in enumerate(program)]


def source():
    # Branches only span scalar instructions of known 4-byte length.
    decoded = enumerate_pcs(PROGRAM)
    for bad in ((2, 1), (3, 0)):
        try:
            lower(decoded, [0x1003], bad)
        except ValueError:
            continue
        raise AssertionError("Output declaration exceeded")
    for change in (
        {"s_waitcnt lgkmcnt(0)": "s_nop 0"},
        {"s_mov_b32 s7, s15": "s_mov_b32 s7, s14"},
        {"s_cbranch_execz 1": "s_branch -1"},
        {
            "s_buffer_load_dword s15, s[24:27], null": "s_buffer_load_dword s15, s[24:27], 0x4"
        },
        {"v_mov_b32_e32 v9, 0x200400": "v_mov_b32_e32 v9, 0x200600"},
        {"exp pos0, v1, v2, v3, v4 done": "exp pos0, v1, v2, v3, v4"},
    ):
        try:
            lower([(pc, change.get(op, op)) for pc, op in decoded], [0x1003], (3, 1))
        except (KeyError, ValueError):
            continue
        raise AssertionError("Invalid mesh path accepted")
    return lower(decoded, [0x1003], (3, 1))


def binary(tmp, assembler, objcopy, library, *, program=None, words=None, header=None):
    from pixel_aot_check import envelope
    from pixel_aot import _read_named_container

    src, obj, raw = (tmp / name for name in ("fixture.s", "fixture.o", "fixture.bin"))
    lines = PROGRAM if program is None else program
    src.write_text(
        "\n".join(
            s.replace("mrtz,", "mrtz")
            .replace("null,", "null")
            .replace("prim,", "prim")
            .replace("pos0,", "pos0")
            for s in lines
        )
        + "\n"
    )
    subprocess.run(
        [
            assembler,
            "-triple=amdgcn--amdpal",
            "-mcpu=gfx1013",
            "-mattr=+wavefrontsize64,-wavefrontsize32",
            "-filetype=obj",
            src,
            "-o",
            obj,
        ],
        check=True,
    )
    subprocess.run(
        [objcopy, "-O", "binary", "--only-section=.text", obj, raw], check=True
    )
    code = raw.read_bytes()
    padded = code + bytes.fromhex("00009fbf") * (4 - (len(code) // 4 + 10) % 4)
    record = envelope(
        padded,
        directory_end=(40 + len(padded), 288 + len(padded)),
        compute_width=128,
        mesh_limits=(3, 1),
    )
    container = tmp / "input.dxbc"
    container.write_bytes(_read_named_container(record, len(b"fixture") + 2))
    translated = tmp / "out.hlsl"
    geometry = tmp / "geometry.json"
    geometry.write_text(json.dumps(words if words is not None else [4099]))
    tool = Path(__file__).resolve().parents[1] / "tools" / "mesh_entry_aot.py"
    command = [
        sys.executable,
        "-B",
        tool,
        container,
        "--llvm",
        library,
        "--output",
        translated,
        *(["--vertex-header", str(header)] if header is not None else []),
        "--geometry-words",
        geometry,
    ]
    subprocess.run(command, check=True)
    expected = (
        source()
        if program is None
        else lower(enumerate_pcs(program), words, (3, 1), header)
    )
    assert translated.read_text() == expected
    if header is not None:
        subprocess.run([*command, "--compute-fallback"], check=True)
        expected = lower(enumerate_pcs(program), words, (3, 1), header, compute_fallback=True)
        assert translated.read_text() == expected


def offset_draw():
    program = PROGRAM.copy()
    program[program.index("s_buffer_load_dword s30, s[16:19], 0x4")] = (
        "s_buffer_load_dword s30, s[16:19], 0x104"
    )
    program[-2:-2] = [
        "s_buffer_load_dword s9, s[24:27], 0x404",
        "v_mov_b32_e32 v10, 0x408",
        "buffer_load_dword v10, v10, s[24:27], 0 offen",
        "s_waitcnt vmcnt(0) lgkmcnt(0)",
        "s_cmp_lt_u32 3, 3",
        "s_cselect_b32 s14, 0x3fa00000, 0",
        "v_add_f32_e32 v3, s9, v3",
        "v_add_f32_e32 v3, v10, v3",
        "v_add_f32_e32 v3, s14, v3",
    ]

    point, matrix, scalar, vector = 0.125, 0.375, 0.0625, 0.125
    inputs = triangle_inputs()
    struct.pack_into("<f", inputs, 4, 0.625)
    struct.pack_into("<f", inputs, 260, point)
    # Geometry words 1 and 2 are distinct decoys for high words 257 and 258.
    words = [4099, *struct.unpack("<2I", struct.pack("<2f", 0.875, 1.125))]
    words += [0] * 254
    words.extend(struct.unpack("<2I", struct.pack("<2f", scalar, vector)))
    # The equal operands contribute zero to the reference depth.
    depth = point + matrix + scalar + vector
    positions = [(x, y, depth, w) for x, y, _, w in TRIANGLE]
    return lower(enumerate_pcs(program), words, (3, 1)), [(inputs, positions, True)]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        if sys.argv[1] == "binary":
            binary(Path(directory), *sys.argv[2:])
            sys.exit(0)
        clipped = triangle_inputs()
        struct.pack_into("<f", clipped, 4, -0.625)  # Plus 0.375 gives Z = -0.25.
        clipped_positions = [(x, y, -0.25, w) for x, y, _, w in TRIANGLE]
        # The triangle runs right then up in render-target coordinates, so it
        # is back-facing with the harness's clockwise front-face convention.
        winding_pixel = (
            "float4 main(bool front : SV_IsFrontFace) : SV_Target "
            "{ return float4(0, front ? 0 : 1, 0, 1); }"
        )
        sys.exit(
            draw(
                *sys.argv[1:],
                Path(directory),
                source(),
                [(triangle_inputs(), TRIANGLE, True), (clipped, clipped_positions, False)],
                pixel_source=winding_pixel,
            )
            or draw(*sys.argv[1:], Path(directory), *offset_draw())
        )
