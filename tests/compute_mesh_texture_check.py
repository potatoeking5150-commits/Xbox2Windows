"""Verify the GPU-only texture handoff between three translated shader stages."""
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

from mesh_fallback_check import compile_fallback, fallback_fixture
from mesh_textured_check import (
    HEADER, PALETTE, PIXEL_CBV_OFFSETS, PIXEL_CBV_WORDS, POSITIONS, VERTEX_ORDER, WORDS,
    enumerate_pcs, fallback_pixel_source, fixture, lower, mesh_program, pixel_source, texture_case,
    vertex_attributes,
)
from compute_store_aot import lower as lower_compute
from pixel_aot_check import compile_shader


def producer(texels):
    words = struct.unpack("<4I", texels)
    code = ["s_mov_b32 s9, 0", "s_load_dwordx4 s[12:15], s[8:9], null"]
    code.extend(f"v_mov_b32_e32 v{20 + i}, {word:#x}" for i, word in enumerate(words))
    code += ["s_waitcnt lgkmcnt(0)",
             "buffer_store_dwordx2 v[20:21], off, s[12:15], 0",
             "buffer_store_dwordx2 v[22:23], off, s[12:15], 0 offset:256",
             "s_endpgm"]
    return lower_compute(code, 1, table_reg=8, descriptor_offset=0)


def compile_native(tmp, compiler):
    mesh_source, mesh = tmp / "mesh.hlsl", tmp / "mesh.dxil"
    mesh_source.write_text(lower(enumerate_pcs(mesh_program()), WORDS, (3, 1), HEADER), encoding="ascii")
    subprocess.run([compiler, "-T", "ms_6_5", "-E", "main", "-DTRACE_EXPORTS",
                    "-Fo", mesh, mesh_source], check=True)
    pixel_hlsl, pixel = tmp / "pixel.hlsl", tmp / "pixel.dxil"
    pixel_hlsl.write_text(pixel_source(), encoding="ascii")
    compile_shader(compiler, pixel_hlsl, pixel)
    return mesh, pixel


def fallback_reference(data, pixels):
    for offset, word in zip(PIXEL_CBV_OFFSETS, PIXEL_CBV_WORDS, strict=True):
        struct.pack_into("<I", data, offset, word)
    attributes = vertex_attributes(False)
    return fallback_fixture(data, POSITIONS, pixels, [attributes[i] for i in VERTEX_ORDER])


def check(tmp, compiler, renderer, adapter, *, fallback=False):
    if fallback:
        draw_shaders = compile_fallback(compiler, tmp, mesh_program(), fallback_pixel_source())
        operation = "--compute-mesh-texture-fallback"
    else:
        draw_shaders = compile_native(tmp, compiler)
        operation = "--compute-mesh-texture"

    shaders = []
    for i, palette in enumerate((PALETTE, PALETTE[1:] + PALETTE[:1])):
        data, pixels, texels = texture_case(palette, False)
        source, shader, reference = (tmp / f"palette-{i}.{ext}" for ext in ("hlsl", "dxil", "case"))
        source.write_text(producer(texels), encoding="ascii")
        subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", shader, source], check=True)
        # The case file carries geometry and the final reference, with no texture upload payload.
        expected = fallback_reference(data, pixels) if fallback else fixture(data, POSITIONS, pixels, True)
        reference.write_bytes(expected)
        shaders.append(shader)
        command = [renderer, adapter, operation, *draw_shaders, reference, *shaders]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 77:
            print(result.stdout, end="")
            return 77
        verify(result, count=i + 1, selected=i, fallback=fallback)

    stale = subprocess.run([*command, "--stale"], capture_output=True, text=True)
    verify(stale, count=2, selected=0, fallback=fallback, stale=True)
    print("Both GPU-produced palettes matched; the stale SRV differed at 276 pixels.")
    return 0


def verify(result, *, count, selected, fallback, stale=False):
    lines = result.stdout.splitlines()
    expected = (f"texture_producers={count} bound_texture={selected}",
                f"pixels=1024 mismatches={276 if stale else 0}")
    if fallback:
        expected += ("mesh_exports bytes=128 matched=1",)
    if (result.returncode != int(stale) or any(line not in lines for line in expected)
            or (stale and "Draw reference mismatch" not in result.stderr.splitlines())
            or ("debug_layer=enabled" in result.stdout and "debug_errors=0 debug_warnings=0" not in lines)):
        raise AssertionError(f"Unexpected texture handoff result\n{result.stdout}\n{result.stderr}")
    print(result.stdout, end="")


if __name__ == "__main__":
    with TemporaryDirectory() as directory:
        args = sys.argv[1:]
        fallback = args[0] == "fallback"
        sys.exit(check(Path(directory), *(args[1:] if fallback else args), fallback=fallback))
