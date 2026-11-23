"""Compare a translated mesh's compute exports and vertex draw on SM 6.0."""

from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from mesh_attribute_check import gradient, program as attribute_program
from mesh_entry_check import enumerate_pcs, fixture
from mesh_indexed_check import HEADER, MESH, WORDS, cases
from mesh_pixel_pipeline_check import buffer_color_program
from mesh_entry_aot import lower
from pixel_aot_check import attribute_color, translate_attribute, translate_buffer_color


def compile_fallback(dxc, directory, program, pixel_source):
    source = directory / "mesh.hlsl"
    source.write_text(lower(enumerate_pcs(program), WORDS, (3, 1), HEADER, compute_fallback=True))
    ps_source = directory / "pixel.hlsl"
    ps_source.write_text(pixel_source)
    compute, vertex, pixel = (directory / (stage + ".dxil") for stage in ("cs", "vs", "ps"))
    for stage, entry, shader, binary in (
        ("cs", "main", source, compute),
        ("vs", "vertex_main", source, vertex),
        ("ps", "main", ps_source, pixel),
    ):
        subprocess.run([dxc, "-T", stage + "_6_0", "-E", entry, "-Fo", binary, shader], check=True)
    return compute, vertex, pixel


def fallback_fixture(constants, positions, pixels, attributes=()):
    native = fixture(constants, positions, pixels, True)
    end = len(constants) + 80
    extra = b"".join(struct.pack("<4f", *value) for value in attributes).ljust(48, b"\xcd")
    return native[:end] + extra + native[end:]


def indexed_program():
    program = MESH.copy()
    # Give the indexed vertex fetch a depth offset from each compute CBV.
    first_wait = program.index("s_waitcnt lgkmcnt(0)")
    program[first_wait:first_wait] = [
        "s_load_dwordx4 s[40:43], s[10:11], 32",
        "s_load_dwordx4 s[44:47], s[10:11], 64",
    ]
    first_read = program.index("s_buffer_load_dword s15, s[24:27], null")
    program[first_read:first_read] = [
        "s_buffer_load_dword s48, s[40:43], 4",
        "s_buffer_load_dword s49, s[44:47], 12",
    ]
    before_export = program.index("v_mov_b32_e32 v4, 0x3f800000")
    program[before_export:before_export] = [
        "v_add_f32_e32 v3, s48, v3",
        "v_add_f32_e32 v3, s49, v3",
    ]
    return program


def indexed_fixture(case, point_depth=0.0625, matrix_depth=0.125):
    constants, positions, pixels = case
    constants = bytearray(constants)
    struct.pack_into("<f", constants, 4, point_depth)
    struct.pack_into("<f", constants, 512 + 12, matrix_depth)
    struct.pack_into("<4f", constants, 64, 0, 1, 0, 1)
    positions = [(x, y, z + point_depth + matrix_depth, w) for x, y, z, w in positions]
    return fallback_fixture(constants, positions, pixels)


def draw(renderer, adapter, directory, shaders, label, data, *, bad_pixels=None, bad_export=None, textured=False):
    path = directory / (label + ".case")
    path.write_bytes(data)
    print(label, flush=True)
    operation = "--mesh-textured-fallback" if textured else "--mesh-indexed-fallback"
    command = [renderer, adapter, operation, *shaders, path]
    negative = bad_pixels is not None
    result = subprocess.run(command, capture_output=textured or negative, text=True)
    if result.stdout is not None:
        print(result.stdout, end="")
        print(result.stderr, end="")
    if not negative:
        if result.returncode == 77:
            sys.exit(77)
        result.check_returncode()
        if not textured:
            return
    lines = result.stdout.splitlines()
    if (f"backend={adapter} debug_layer=unavailable" not in lines
            and "debug_errors=0 debug_warnings=0" not in lines):
        raise AssertionError(f"{label} did not finish with a clean D3D12 debug layer")
    if not negative:
        if "mesh_exports bytes=128 matched=1" not in lines or "pixels=1024 mismatches=0" not in lines:
            raise AssertionError(f"{label} did not match the export and pixel references")
        return
    error = "Mesh export trace mismatch" if bad_export is not None else "Draw reference mismatch"
    if (
        result.returncode != 1
        or f"pixels=1024 mismatches={bad_pixels}" not in lines
        or f"mesh_exports bytes=128 matched={int(bad_export is None)}" not in lines
        or (bad_export is not None and f"first_mesh_export_mismatch_byte={bad_export}" not in lines)
        or error not in result.stderr
    ):
        raise AssertionError(f"Expected {error} and {bad_pixels} wrong pixels")


def main(dxc, renderer, adapter, directory):
    pixel_source = translate_buffer_color(buffer_color_program())
    program = indexed_program()
    shaders = compile_fallback(dxc, directory, program, pixel_source)
    full_case, partial_case = cases((1, 2, 0))
    partial = indexed_fixture(partial_case)
    for label, data in (
        ("full", indexed_fixture(full_case)),
        ("partial", partial),
        ("partial-change-b0", indexed_fixture(partial_case, point_depth=0.1875)),
        ("partial-change-b1", indexed_fixture(partial_case, matrix_depth=-0.125)),
    ):
        draw(renderer, adapter, directory, shaders, label, data)

    bad = bytearray(partial)
    bad[len(partial_case[0]) + 76] ^= 1
    draw(renderer, adapter, directory, shaders, "wrong-guard", bad, bad_pixels=0, bad_export=76)

    # Change the source primitive from (0,1,2) to (0,1,1). A vertex shader
    # that ignores the generated index buffer incorrectly draws the old triangle.
    collapsed = [op.replace("0x200400", "0x100400") for op in program]
    shaders = compile_fallback(dxc, directory, collapsed, pixel_source)
    draw(renderer, adapter, directory, shaders, "collapsed-index", partial, bad_pixels=276, bad_export=72)

    shaders = compile_fallback(dxc, directory, attribute_program(),
                               translate_attribute(attribute_color(False)))
    for swap in (False, True):
        constants, positions, pixels = gradient(swap)
        # PARAM0 is exported as (blue, alpha, green, red) in vertex order 1,2,0.
        attrs = [(0, 1, 0, 2), (0, 1, 2, 0), (0, 1, 0, 0)]
        if swap:
            attrs[0], attrs[1] = attrs[1], attrs[0]
        draw(renderer, adapter, directory, shaders, f"gradient-swap-{swap}",
             fallback_fixture(constants, positions, pixels, attrs))
    draw(renderer, adapter, directory, shaders, "stale-palette",
         fallback_fixture(constants, positions, gradient(False)[2], attrs), bad_pixels=992)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        main(*sys.argv[1:], Path(tmp))
