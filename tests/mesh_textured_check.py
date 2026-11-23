"""Verify indexed PARAM0 UVs and a translated sample in the same D3D12 draw."""

from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from mesh_attribute_check import program
from mesh_entry_check import binary, enumerate_pcs, fixture
from mesh_fallback_check import compile_fallback, draw as draw_fallback, fallback_fixture
from mesh_indexed_check import HEADER, SMALL, WORDS, inputs, lower, small_raster
from pixel_aot import lower as lower_pixel
from pixel_aot_check import compile_shader, lod_zero_sample


PALETTE = ((255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), (255, 255, 0, 255))
TEXTURE_INPUTS = ((PALETTE, False), (PALETTE[1:] + PALETTE[:1], False), (PALETTE, True))
VERTEX_ORDER = (1, 2, 0)
POSITIONS = [SMALL[i] for i in VERTEX_ORDER]
# The export shader does not use the last register of either 512-byte CBV.
PIXEL_CBV_OFFSETS = (31 * 16, 512 + 31 * 16)
PIXEL_CBV_WORDS = (0x1234ABCD, 0x89ABCDEF)


def vertex_attributes(flip):
    return [(1 - u if flip else u, v, 0, 1) for u, v in ((0, 2), (2, 2), (0, 0))]


def texture_case(colors, flip):
    data = inputs(SMALL)
    for i, uv in enumerate(vertex_attributes(flip)):
        struct.pack_into("<3f", data, 1028 + 64 + i * 12, *uv[:3])
    coverage = small_raster()
    pixels = bytearray(coverage)
    for y in range(32):
        for x in range(32):
            offset = (y * 32 + x) * 4
            if coverage[offset + 1]:
                # Edges run from x=4 to 28 and y=5 to 28. The UV range is two units per edge.
                u = 2 * (x + 0.5 - 4) / 24
                v = 2 * (y + 0.5 - 5) / 23
                col = (1 - u if flip else u) >= 0.5
                row = v >= 0.5
                pixels[offset:offset + 4] = bytes(colors[2 * row + col])
    texels = bytes(channel for pixel in colors for channel in pixel)
    return data, bytes(pixels), texels


def mesh_program():
    code = program()
    # PARAM0.x/y are the first two fetched attributes rather than the color fixture's swizzle.
    code[code.index("exp param0 v7, v4, v6, v5 done")] = "exp param0 v5, v6, v7, v4 done"
    return code


def texture_cases():
    return tuple(texture_case(colors, flip) for colors, flip in TEXTURE_INPUTS)


def pixel_source():
    return lower_pixel(lod_zero_sample(), 20, 24, [1, 2, 3, 4], texcoord_components=4)


def fallback_pixel_source():
    source = pixel_source()
    entry = "float4 main(float4 uv : TEXCOORD0) : SV_Target0 {\n"
    assert source.count(entry) == 1, "Expected the translated texture shader entry point"
    a, b = PIXEL_CBV_WORDS
    guard = (f"    if (gate_a != 0x{a:08x}u || gate_b != 0x{b:08x}u)\n"
             "        return float4(0, 0, 0, 1);\n")
    return ("cbuffer GuardA : register(b0) { uint gate_a : packoffset(c31.x); };\n"
            "cbuffer GuardB : register(b1) { uint gate_b : packoffset(c31.x); };\n"
            + source.replace(entry, entry + guard, 1))


def run(tmp, compiler, renderer, adapter):
    code = mesh_program()
    mesh_source, mesh = tmp / "mesh.hlsl", tmp / "mesh.dxil"
    mesh_source.write_text(lower(enumerate_pcs(code), WORDS, (3, 1), HEADER), encoding="ascii")
    pixel_hlsl, pixel = tmp / "pixel.hlsl", tmp / "pixel.dxil"
    pixel_hlsl.write_text(pixel_source(), encoding="ascii")
    compile_shader(compiler, pixel_hlsl, pixel)
    cases = texture_cases()
    for traced in (False, True):
        flags = ["-DTRACE_EXPORTS"] if traced else []
        subprocess.run([compiler, "-T", "ms_6_5", "-E", "main", *flags, "-Fo", mesh, mesh_source], check=True)
        for index, (data, pixels, texels) in enumerate(cases):
            path = tmp / f"case-{index}.bin"
            path.write_bytes(fixture(data, POSITIONS, pixels, traced) + texels)
            result = subprocess.run([renderer, adapter, "--mesh-textured", mesh, pixel, path])
            if result.returncode == 77:
                return 77
            result.check_returncode()
    for index in (1, 2):
        data, _, texels = cases[index]
        baseline = cases[0][1]
        path = tmp / f"stale-{index}.bin"
        path.write_bytes(fixture(data, POSITIONS, baseline, True) + texels)
        result = subprocess.run([renderer, adapter, "--mesh-textured", mesh, pixel, path],
                                capture_output=True, text=True)
        if (result.returncode != 1 or "pixels=1024 mismatches=276" not in result.stdout.splitlines()
                or ("debug_layer=enabled" in result.stdout
                    and "debug_errors=0 debug_warnings=0" not in result.stdout.splitlines())
                or "Draw reference mismatch" not in result.stderr):
            raise AssertionError(f"Unexpected stale-image result\n{result.stdout}\n{result.stderr}")
        print(f"stale_reference={index} mismatches=276")
    return 0


def run_fallback(tmp, compiler, renderer, adapter):
    shaders = compile_fallback(compiler, tmp, mesh_program(), fallback_pixel_source())
    cases = texture_cases()
    labels = (("baseline", None), ("rotated-texture", "stale-texture"), ("flipped-uv", "stale-uv"))
    for (label, stale), (_, flip), (data, pixels, texels) in zip(labels, TEXTURE_INPUTS, cases, strict=True):
        for offset, word in zip(PIXEL_CBV_OFFSETS, PIXEL_CBV_WORDS, strict=True):
            struct.pack_into("<I", data, offset, word)
        attributes = vertex_attributes(flip)
        indexed_attributes = [attributes[i] for i in VERTEX_ORDER]
        draw_fallback(renderer, adapter, tmp, shaders, label,
                      fallback_fixture(data, POSITIONS, pixels, indexed_attributes) + texels, textured=True)
        if stale:
            draw_fallback(renderer, adapter, tmp, shaders, stale,
                          fallback_fixture(data, POSITIONS, cases[0][1], indexed_attributes) + texels,
                          textured=True, bad_pixels=276)
    data, pixels, texels = cases[0]
    attributes = vertex_attributes(False)
    indexed_attributes = [attributes[i] for i in VERTEX_ORDER]
    for binding, offset in enumerate(PIXEL_CBV_OFFSETS):
        stale_data = bytearray(data)
        struct.pack_into("<I", stale_data, offset, 0)
        draw_fallback(renderer, adapter, tmp, shaders, f"stale-pixel-b{binding}",
                      fallback_fixture(stale_data, POSITIONS, pixels, indexed_attributes) + texels,
                      textured=True, bad_pixels=276)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        tmp = Path(directory)
        if sys.argv[1] == "binary":
            binary(tmp, *sys.argv[2:], program=mesh_program(), words=WORDS, header=HEADER)
        elif sys.argv[1] == "fallback":
            run_fallback(tmp, *sys.argv[2:])
        else:
            sys.exit(run(tmp, *sys.argv[1:]))
