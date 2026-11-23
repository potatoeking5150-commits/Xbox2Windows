"""Draw two indexed triangles into one color/depth target and verify both readbacks."""

from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from mesh_attribute_check import program
from mesh_entry_check import enumerate_pcs, fixture
from mesh_fallback_check import compile_fallback, fallback_fixture
from mesh_indexed_check import HEADER, SMALL, WORDS, inputs, lower
from pixel_aot_check import attribute_color, translate_attribute

COLORS = ((255, 0, 0, 255), (0, 0, 255, 255))


def scene(depths, order, fallback=False):
    uploads, traces = [], []
    for i in order:
        points = [(x + i / 4, y, depths[i], w) for x, y, _, w in SMALL]
        data = inputs(points)
        red, green, blue = (channel / 255 for channel in COLORS[i][:3])
        for vertex in range(3):
            struct.pack_into("<3f", data, 1028 + 64 + 12 * vertex, red, green, blue)
        uploads.append(data)
        positions = [points[j] for j in (1, 2, 0)]
        if fallback:
            # PARAM0 is exported in shader order, before the pixel shader swizzle.
            attributes = [(blue, 1, green, red)] * 3
            traces.append(fallback_fixture(b"", positions, b"", attributes))
        else:
            traces.append(fixture(b"", positions, b"", True))

    rgba, depth_bytes = bytearray(), bytearray()
    overlap = 0
    for y in range(32):
        for x in range(32):
            # Integer twice-center coordinates avoid edge rounding in the reference.
            hits = [i for i in range(2) if x >= 4 + 4 * i and y < 28
                    and 23 * (2 * x + 1 - 2 * (4 + 4 * i)) < 24 * (2 * y + 1 - 10)]
            overlap += len(hits) == 2
            if hits:
                winner = min(hits, key=depths.__getitem__)
                rgba.extend(COLORS[winner])
                depth_bytes.extend(struct.pack("<f", depths[winner]))
            else:
                rgba.extend((0, 0, 0, 255))
                depth_bytes.extend(struct.pack("<f", 1))
    return b"".join(uploads + traces), bytes(rgba), bytes(depth_bytes), overlap


def compile_native(compiler, tmp, pixel_source):
    for stage, profile, text, flags in (
        ("mesh", "ms_6_5", lower(enumerate_pcs(program()), WORDS, (3, 1), HEADER), ["-DTRACE_EXPORTS"]),
        ("pixel", "ps_6_0", pixel_source, []),
    ):
        source = tmp / f"{stage}.hlsl"
        source.write_text(text, encoding="ascii")
        subprocess.run([compiler, "-T", profile, "-E", "main", *flags, "-Fo",
                        tmp / f"{stage}.dxil", source], check=True)
    return tmp / "mesh.dxil", tmp / "pixel.dxil"


def run(tmp, compiler, renderer, adapter, *, fallback=False):
    pixel_source = translate_attribute(attribute_color(False))
    if fallback:
        shaders = compile_fallback(compiler, tmp, program(), pixel_source)
        operation = "--mesh-depth-pair-fallback"
        trace_bytes = 256
    else:
        shaders = compile_native(compiler, tmp, pixel_source)
        operation = "--mesh-depth-pair"
        trace_bytes = 160

    path = tmp / "depth-pair.case"
    command = [renderer, adapter, operation, *shaders, path]
    base = scene((0.25, 0.75), (0, 1), fallback)
    reverse = scene((0.25, 0.75), (1, 0), fallback)
    swapped = scene((0.75, 0.25), (0, 1), fallback)
    assert base[1:] == reverse[1:]
    data, rgba, depth, overlap = swapped
    depth_changes = sum(depth[i:i + 4] != base[2][i:i + 4] for i in range(0, len(depth), 4))
    assert overlap == 190 and depth_changes == 172

    for label, content, color_errors, depth_errors, message in (
        ("baseline", b"".join(base[:3]), 0, 0, None),
        ("reverse-order", b"".join(reverse[:3]), 0, 0, None),
        ("swapped-depths", data + rgba + depth, 0, 0, None),
        ("stale-color", data + base[1] + depth, overlap, 0, "Draw reference mismatch"),
        ("stale-depth", data + rgba + base[2], 0, depth_changes, "Depth reference mismatch"),
    ):
        print(label, flush=True)
        path.write_bytes(content)
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout, end="", flush=True)
        print(result.stderr, end="", file=sys.stderr, flush=True)
        if label == "baseline" and result.returncode == 77:
            return 77
        lines = set(result.stdout.splitlines())
        expected = {
            f"mesh_exports bytes={trace_bytes} matched=1",
            f"pixels=1024 mismatches={color_errors}",
            f"depth_pixels=1024 depth_mismatches={depth_errors}",
        }
        if (result.returncode != int(message is not None) or not expected <= lines
                or (message is not None and message not in result.stderr.splitlines())
                or (f"backend={adapter} debug_layer=unavailable" not in lines
                    and "debug_errors=0 debug_warnings=0" not in lines)):
            raise AssertionError(f"Unexpected depth-pair result for {label}")
        if message is not None:
            print(f"Expected failure: {message}, color={color_errors}, depth={depth_errors}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    fallback = args[:1] == ["fallback"]
    if fallback:
        args = args[1:]
    with tempfile.TemporaryDirectory() as directory:
        sys.exit(run(Path(directory), *args, fallback=fallback))
