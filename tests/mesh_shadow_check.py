"""Cast a shadow with translated programs: light-space depth, then a SampleCmpLevelZero receiver and a lit caster.

Pass 1 draws the caster with the translated mesh program into a depth texture, from an oblique directional
light. Pass 2 draws a receiver plane whose translated pixel program compares its light-space depth against
that texture with image_sample_c_lz, then the caster lit with N.L from the #749 pixel program. The expected
image and the expected shadow map come from an exact CPU model of the scene; no D3D12 code is involved.
"""

from fractions import Fraction
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from mesh_entry_check import binary, enumerate_pcs, fixture
from mesh_indexed_check import HEADER, WORDS, inputs, lower
from mesh_lit_check import AMBIENT, LIGHT_OFFSET, export16, pixel_program, unorm8
from mesh_textured_check import mesh_program
from pixel_aot import comparison_sampler_state, disassemble, lower as lower_pixel, native_program
from pixel_aot_check import envelope
from pixel_compare_check import passes, sampler_words

SIDE = 64
REFERENCE_OFFSET = 0x90  # params[9] of the receiver's b0: its light-space depth
# The scene in camera NDC, an orthographic view with depth z. A directional light projects a point
# (x, y, z) to (x + a z, y + b z) with depth z. Every vertex lies on the pixel grid in both views, so pixel
# centres of the receiver sample texel centres of the map, at least half a texel from any texel edge.
LIGHT_SHEAR = (Fraction(-1, 2), Fraction(1, 2))
STALE_SHEAR = (Fraction(1, 2), Fraction(1, 2))  # the previous frame's light, for the stale-map control
CASTER_DEPTH = Fraction(1, 4)
RECEIVER_DEPTH = Fraction(3, 4)
HIDDEN_DEPTH = Fraction(7, 8)  # behind the receiver: in the map, but it must not shadow
CASTER = ((12, 48), (44, 48), (28, 12))  # pixel corners: an upright triangle
HIDDEN = ((40, 4), (62, 4), (62, 20))  # light-space pixel corners of a caster behind the receiver
RECEIVER = ((-1, 1), (3, 1), (-1, -3))  # NDC: covers the whole target
# Caster shading: normal (x - cx, y - cy, 1/2) at NDC (x, y), lit from the upper left like the shadow.
CENTRE = (Fraction(0), Fraction(-1, 8))
# N.L is an odd multiple of 1/128 at every pixel centre, so no exported value is 1/2, where the 8-bit
# conversion would round a half; the ambient's 2^-17 keeps each sum off a binary16 value, as in #749.
NORMAL_Z = Fraction(17, 32)
SHADE_LIGHT = (Fraction(-1, 2), Fraction(1, 2), Fraction(1, 4), AMBIENT)


def ndc(corner):
    px, py = corner
    return Fraction(px, SIDE // 2) - 1, 1 - Fraction(py, SIDE // 2)


def project(point, shear, depth):
    return point[0] + shear[0] * depth, point[1] + shear[1] * depth


def centre(x, y):
    return Fraction(2 * x + 1 - SIDE, SIDE), Fraction(SIDE - 2 * y - 1, SIDE)


def inside(triangle, x, y):
    """Exact coverage of the pixel centre; no centre may lie on an edge, so the fill rule never decides."""
    cx, cy = centre(x, y)
    edges = [(x1 - x0) * (cy - y0) - (y1 - y0) * (cx - x0)
             for (x0, y0), (x1, y1) in zip(triangle, triangle[1:] + triangle[:1])]
    assert not (0 in edges and (min(edges) >= 0 or max(edges) <= 0)), "pixel centre on a triangle edge"
    return min(edges) > 0 or max(edges) < 0


def shadow_map(casters):
    """Light-space depth after a LESS depth test from a cleared 1.0, as D32_FLOAT texels."""
    texels = []
    for y in range(SIDE):
        for x in range(SIDE):
            texels.append(min([depth for triangle, depth in casters if inside(triangle, x, y)] + [Fraction(1)]))
    return texels


def caster_light(shear, corners=CASTER):
    return [project(ndc(corner), shear, CASTER_DEPTH) for corner in corners]


def light_casters(shear, hidden, corners=CASTER):
    casters = [(caster_light(shear, corners), CASTER_DEPTH)]
    if hidden:
        casters.append(([ndc(corner) for corner in HIDDEN], HIDDEN_DEPTH))
    return casters


def exact16(value):
    """A receiver result: compare result times 1/2 plus 1/4, exact in binary32 and in binary16."""
    assert Fraction(struct.unpack("<e", struct.pack("<e", float(value)))[0]) == value
    return value


def scene_image(texels, compare="LESS", *, use_map=True, corners=CASTER):
    """Exact expected colours: the lit caster over the receiver, the receiver lit or in shadow."""
    caster = [ndc(corner) for corner in corners]
    pixels = bytearray()
    shadowed = 0
    for y in range(SIDE):
        for x in range(SIDE):
            cx, cy = centre(x, y)
            if inside(caster, x, y):
                normal = (cx - CENTRE[0], cy - CENTRE[1], NORMAL_Z)
                dot = sum(n * l for n, l in zip(normal, SHADE_LIGHT))
                gray = unorm8(export16(max(dot, 0) + SHADE_LIGHT[3]))
                pixels.extend((gray, gray, gray, 255))
                continue
            # The receiver's map coordinate: u = (x' + 1) / 2, v = (1 - y') / 2 of its light-space position,
            # a texel centre here. CLAMP addressing takes the edge texel outside [0, 1].
            lx, ly = project((cx, cy), LIGHT_SHEAR, RECEIVER_DEPTH)
            su, sv = (lx + 1) / 2 * SIDE, (1 - ly) / 2 * SIDE
            assert su.denominator == 2 and sv.denominator == 2
            tx = min(max(math.floor(su), 0), SIDE - 1)
            ty = min(max(math.floor(sv), 0), SIDE - 1)
            lit = passes(compare, RECEIVER_DEPTH, texels[ty * SIDE + tx]) if use_map else True
            shadowed += not lit
            value = exact16(Fraction(int(lit), 2) + Fraction(1, 4))
            pixels.extend((unorm8(value), unorm8(value), unorm8(exact16(value / 2)), 255))
    return bytes(pixels), shadowed


def coverage_only(texels):
    """The wrong model for a negative control: any texel a caster wrote shadows, whatever its depth."""
    return scene_image([Fraction(0) if texel < 1 else texel for texel in texels])[0]


def draw_inputs(positions, attributes=None, light=None, reference=None):
    data = inputs([(*map(float, point), 1) for point in positions])
    for i, attribute in enumerate(attributes or ()):
        struct.pack_into("<3f", data, 1028 + 64 + i * 12, *map(float, attribute))
    if light:
        struct.pack_into("<4f", data, LIGHT_OFFSET, *map(float, light))
    if reference is not None:
        struct.pack_into("<f", data, REFERENCE_OFFSET, float(reference))
    trace = fixture(b"", [(*map(float, positions[i]), 1.0) for i in (1, 2, 0)], b"", True)
    return bytes(data), trace


def light_draws(shear, hidden):
    return [draw_inputs([(*point, depth) for point in triangle]) for triangle, depth in light_casters(shear, hidden)]


def camera_draws(corners=CASTER):
    receiver_uv = []
    for point in RECEIVER:
        lx, ly = project(point, LIGHT_SHEAR, RECEIVER_DEPTH)
        receiver_uv.append(((lx + 1) / 2, (1 - ly) / 2, 0))
    receiver = draw_inputs([(*point, RECEIVER_DEPTH) for point in RECEIVER], receiver_uv, reference=RECEIVER_DEPTH)
    caster = [ndc(corner) for corner in corners]
    normals = [(x - CENTRE[0], y - CENTRE[1], NORMAL_Z) for x, y in caster]
    lit = draw_inputs([(*point, CASTER_DEPTH) for point in caster], normals, light=SHADE_LIGHT)
    return [receiver, lit]


def case_bytes(compare, image, texels):
    draws = light_draws(STALE_SHEAR, False) + light_draws(LIGHT_SHEAR, True) + camera_draws()
    # The host sampler comes from the decoded descriptor words, as in pixel_compare_check.
    state = comparison_sampler_state(sampler_words(compare, "point"))
    header = struct.pack("<8I", SIDE, 1, 2, state["d3d12_filter"], state["d3d12_comparison_func"], 0, 0, 0)
    depth = b"".join(struct.pack("<f", float(value)) for value in texels)
    return header + b"".join(d for d, _ in draws) + b"".join(t for _, t in draws) + image + depth


def receiver_program(words, *, sample=True):
    """Interpolate the map coordinate, load the light-space depth from b0 and compare it against the map."""
    code = ["s_mov_b32 m0, s24",
            "v_interp_p1_f32_e32 v8, v0, attr0.x", "v_interp_p1_f32_e32 v9, v0, attr0.y",
            "v_interp_p2_f32_e32 v9, v1, attr0.y", "v_interp_p2_f32_e32 v8, v1, attr0.x",
            "s_mov_b32 s21, 0", "s_load_dwordx4 s[44:47], s[20:21], 0x40"]
    code += ["s_load_dwordx8 s[28:35], s[20:21], null"] if sample else []
    code += ["s_waitcnt lgkmcnt(0)", f"s_buffer_load_dword s60, s[44:47], {REFERENCE_OFFSET:#x}",
             "s_waitcnt lgkmcnt(0)"]
    if sample:
        code += [f"s_mov_b32 s{40 + k}, 0x{word:x}" for k, word in enumerate(words)]
        code += ["v_mov_b32_e32 v7, s60",
                 "image_sample_c_lz v12, v[7:9], s[28:35], s[40:43] dmask:0x1 dim:SQ_RSRC_IMG_2D",
                 "s_waitcnt vmcnt(0)"]
    else:
        code += ["v_mov_b32_e32 v12, 1.0"]  # negative control: the map is never read
    return code + ["v_mul_f32_e32 v13, 0.5, v12", "v_add_f32_e32 v13, 0x3e800000, v13",
                   "v_mul_f32_e32 v14, 0.5, v13", "v_mov_b32_e32 v15, 1.0",
                   "v_cvt_pkrtz_f16_f32_e32 v16, v13, v13", "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
                   "exp mrt0 v16, v16, v17, v17 done compr vm", "s_endpgm"]


def caster_program():
    """The #749 N.L program, reading the normal from PARAM0.xyz as this mesh program exports it."""
    order = dict(zip("wzxy", "xyzw"))
    return [line[:-1] + order[line[-1]] if "attr0." in line else line for line in pixel_program()]


def translate_receiver(code, words):
    return lower_pixel(code, 20, 24, words if words else None, 64, texcoord_components=4)


def translate_caster(code):
    return lower_pixel(code, 8, 14, None, 0)


def controls():
    """CPU checks of the scene before any draw."""
    current = shadow_map(light_casters(LIGHT_SHEAR, True))
    base, shadowed = scene_image(current)
    # The hidden caster is in the map, but behind the receiver it casts nothing.
    assert HIDDEN_DEPTH in current and scene_image(shadow_map(light_casters(LIGHT_SHEAR, False)))[0] == base
    assert shadowed > 0 and differences(base, coverage_only(current)) > 0
    words = sampler_words("LESS", "point")
    assert "SampleCmpLevelZero" in translate_receiver(receiver_program(words), words)
    assert "SampleCmpLevelZero" not in translate_receiver(receiver_program(None, sample=False), None)
    return current, base, shadowed


def check_binary(tmp, assembler, objcopy, library):
    binary(tmp, assembler, objcopy, library, program=mesh_program(), words=WORDS, header=HEADER)
    words = sampler_words("LESS", "point")
    for name, code, translate in (("receiver", receiver_program(words), lambda c: translate_receiver(c, words)),
                                  ("caster", caster_program(), translate_caster)):
        source, obj, raw = (tmp / f"{name}{suffix}" for suffix in (".s", ".o", ".bin"))
        source.write_text("\n".join(code) + "\n", encoding="ascii")
        subprocess.run([assembler, "-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-filetype=obj", source, "-o", obj],
                       check=True)
        subprocess.run([objcopy, "-O", "binary", "--only-section=.text", obj, raw], check=True)
        archive = tmp / f"{name}.rpf"
        archive.write_bytes(envelope(raw.read_bytes()))
        _, native = native_program(archive, "fixture")
        assert native == raw.read_bytes()
        assert translate(disassemble(native, library)) == translate(code)
        print(f"{name}: {len(native)} bytes decoded to the same translation")


def compile_stage(compiler, tmp, name, text, profile="ps_6_0", flags=()):
    source, shader = tmp / f"{name}.hlsl", tmp / f"{name}.dxil"
    source.write_text(text, encoding="ascii")
    subprocess.run([compiler, "-T", profile, "-E", "main", *flags, "-Fo", shader, source], check=True,
                   stdout=subprocess.DEVNULL)
    return shader


def draw(renderer, adapter, shaders, path, data, *, stale=False, mismatches=0, map_mismatches=0):
    path.write_bytes(data)
    result = subprocess.run([renderer, adapter, "--mesh-shadow", *shaders, path, *(["stale"] if stale else [])],
                            capture_output=True, text=True)
    print(result.stdout, end="")
    if result.returncode == 77:
        return 77
    lines = result.stdout.splitlines()
    clean = "debug_layer=enabled" not in result.stdout or "debug_errors=0 debug_warnings=0" in lines
    expected = {f"pixels={SIDE * SIDE} mismatches={mismatches}",
                f"shadow_map_texels={SIDE * SIDE} shadow_map_mismatches={map_mismatches}"}
    failing = mismatches or map_mismatches
    message = "Shadow map reference mismatch" if map_mismatches else "Draw reference mismatch"
    if (result.returncode != (1 if failing else 0) or not clean or not expected <= set(lines)
            or (failing and message not in result.stderr.splitlines())):
        raise AssertionError(f"Unexpected shadow draw result\n{result.stdout}\n{result.stderr}")
    return 0


def differences(first, second):
    return sum(first[i:i + 4] != second[i:i + 4] for i in range(0, len(first), 4))


def run(tmp, compiler, renderer, adapter):
    current, base, shadowed = controls()
    mesh = compile_stage(compiler, tmp, "mesh", lower(enumerate_pcs(mesh_program()), WORDS, (3, 1), HEADER),
                         "ms_6_5", ["-DTRACE_EXPORTS"])
    caster = compile_stage(compiler, tmp, "caster", translate_caster(caster_program()))
    shaders = {}
    for compare in ("LESS", "GREATER"):
        words = sampler_words(compare, "point")
        shaders[compare] = compile_stage(compiler, tmp, f"receiver-{compare}",
                                         translate_receiver(receiver_program(words), words))
    path = tmp / "shadow.case"
    if draw(renderer, adapter, (mesh, shaders["LESS"], caster), path, case_bytes("LESS", base, current)) == 77:
        return 77
    print(f"shadowed_pixels={shadowed}")

    # Coverage instead of depth: the caster behind the receiver would shadow the pixels that sample it.
    expected = differences(base, coverage_only(current))
    draw(renderer, adapter, (mesh, shaders["LESS"], caster), path, case_bytes("LESS", coverage_only(current), current),
         mismatches=expected)
    print(f"Expected failure: coverage instead of depth, mismatches={expected}")

    # The light pass is read back too: a map reference without the hidden caster misses its texels.
    without_hidden = shadow_map(light_casters(LIGHT_SHEAR, False))
    expected = sum(a != b for a, b in zip(current, without_hidden))
    draw(renderer, adapter, (mesh, shaders["LESS"], caster), path, case_bytes("LESS", base, without_hidden),
         map_mismatches=expected)
    print(f"Expected failure: shadow map without the hidden caster, shadow_map_mismatches={expected}")

    # LESS and GREATER exchanged: every receiver pixel flips, and GREATER matches its own reference.
    greater = scene_image(current, "GREATER")[0]
    expected = differences(base, greater)
    caster_pixels = sum(inside([ndc(c) for c in CASTER], x, y) for y in range(SIDE) for x in range(SIDE))
    assert expected == SIDE * SIDE - caster_pixels
    draw(renderer, adapter, (mesh, shaders["GREATER"], caster), path, case_bytes("GREATER", greater, current))
    draw(renderer, adapter, (mesh, shaders["GREATER"], caster), path, case_bytes("GREATER", base, current),
         mismatches=expected)
    print(f"Expected failure: GREATER for LESS, mismatches={expected}")

    # Stale map: the camera pass binds the map drawn for the previous light.
    previous = shadow_map(light_casters(STALE_SHEAR, False))
    stale = scene_image(previous)[0]
    expected = differences(base, stale)
    assert expected > 0
    draw(renderer, adapter, (mesh, shaders["LESS"], caster), path, case_bytes("LESS", stale, current), stale=True)
    draw(renderer, adapter, (mesh, shaders["LESS"], caster), path, case_bytes("LESS", base, current), stale=True,
         mismatches=expected)
    print(f"Expected failure: stale shadow map, mismatches={expected}")

    # Ignored map: the receiver never samples it, so no pixel is in shadow.
    ignored_shader = compile_stage(compiler, tmp, "receiver-ignored",
                                   translate_receiver(receiver_program(None, sample=False), None))
    unshadowed = scene_image(current, use_map=False)[0]
    expected = differences(base, unshadowed)
    assert expected == shadowed
    draw(renderer, adapter, (mesh, ignored_shader, caster), path, case_bytes("LESS", unshadowed, current))
    draw(renderer, adapter, (mesh, ignored_shader, caster), path, case_bytes("LESS", base, current),
         mismatches=expected)
    print(f"Expected failure: ignored shadow map, mismatches={expected}")
    return 0


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        if len(sys.argv) > 1 and sys.argv[1] == "binary":
            check_binary(Path(directory), *sys.argv[2:])
            sys.exit(0)
        if len(sys.argv) == 1:
            controls()
            sys.exit(0)
        sys.exit(run(Path(directory), *sys.argv[1:]))
