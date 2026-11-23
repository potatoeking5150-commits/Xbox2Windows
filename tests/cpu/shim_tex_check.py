"""Compare uploaded-texture draws from the linked shim EXE, mapped image and AOT process.

Unlike the compute-written texture of shim_draw_check.py --texture-reuse, these cases
upload texels from the CPU: a vertex-buffer quad samples an 8x8 texture through an
SRV with a point-clamp static sampler, using upload footprints, transitions and a
fence-waited readback. tests/cpu/tex_gen.py owns the texel tables and the CPU rules.

tex     8x8 RGBA8 table, exact rule, exit 42.
bc7     8x8 mode-6 BC7 texture (repository block builder and tiled-to-linear
        conversion), exact rule, exit 42.
scene   one frame with an RGBA8 draw and a BC7 draw, exact rule, exit 42.
texuv   UVs shifted by half a texel; must exit 20 and match its own rule.
texbad  one corrupted texel in the upload; must exit 20 and match its own rule.

Every case must agree byte for byte across the native EXE, pe_map and the AOT build.
Variant frames must differ from the exact rule. Passing frames are saved as BMPs under
X2W_SHIM_DRAW_OUT (default: the current directory) for inspection.
"""
import os
from pathlib import Path
import subprocess
import sys

from shim_startup_check import EXPECTED, SHIM_IMPORTS, check
from tex_gen import tex_frame, texuv_frame, texbad_frame, bc7_frame, scene_frame, write_bmp


SIDE = 64
BYTES = SIDE * SIDE * 4

CASE_RULES = {
    "tex": (42, tex_frame),
    "texuv": (20, texuv_frame),
    "texbad": (20, texbad_frame),
    "bc7": (42, bc7_frame),
    "scene": (42, scene_frame),
}


def compare(mapper, exe, lifted, companion, io, xmem, d3d12):
    map_command = [sys.executable, "-B", Path(__file__).with_name("shim_startup_check.py"), "--map",
                   mapper, exe, companion, io, xmem, d3d12, ""]
    markers = sorted(f"bound {dll}!{symbol}" for dll, symbols in SHIM_IMPORTS.items() for symbol in symbols)
    out_dir = Path(os.environ.get("X2W_SHIM_DRAW_OUT", "."))
    exact = tex_frame()
    for case, (expected_exit, rule) in CASE_RULES.items():
        env = dict(os.environ, X2W_SHIM_DRAW_CASE=case, X2W_SHIM_COMPUTE_CASE="")
        rule_frame = rule()
        frames = {}
        for runner, command in (("native", [exe]), ("pe_map", map_command), ("AOT", [lifted])):
            child = subprocess.run(command, env=env, capture_output=True, timeout=120)
            lines = child.stderr.decode(errors="replace").splitlines()
            if (case == "tex" and runner == "native" and child.returncode == 77
                    and env.get("X2W_SHIM_DRAW_ADAPTER") == "hardware"):
                check("hardware capability diagnostic", lines, ["no D3D12 physical adapter"])
                print(lines[-1])
                return 77
            if runner == "pe_map":
                check("mapper returned", lines[-1:], ["mapped host returned"])
                check("explicit shim bindings", sorted(lines[:len(markers)]), markers)
                check(f"{case} mapped draw trace", lines[len(markers):-1], [])
            else:
                check(f"{case} {runner} diagnostics", lines, [])
            check(f"{case} {runner} startup output", child.stdout[:len(EXPECTED.output)], EXPECTED.output)
            pixels = child.stdout[len(EXPECTED.output):]
            check(f"{case} {runner} exits", child.returncode, expected_exit)
            check(f"{case} {runner} publishes a full frame", len(pixels), BYTES)
            check(f"{case} {runner} matches the CPU rule", pixels, rule_frame)
            frames[runner] = pixels
        if expected_exit != 42 and rule_frame == exact:
            raise AssertionError(f"{case}: the variant rule equals the exact rule, so the control is vacuous")
        check(f"{case} runners agree", (frames["native"], frames["AOT"]), (frames["pe_map"], frames["pe_map"]))
        if expected_exit == 42:
            shot = out_dir / f"shim_{case}_frame.bmp"
            write_bmp(shot, SIDE, rule_frame)
            print(f"{case}: saved {shot}")
        print(f"{case}: exits checked; rgba_bytes={BYTES}")
    return 0


if __name__ == "__main__":
    sys.exit(compare(*map(Path, sys.argv[1:-1])))
