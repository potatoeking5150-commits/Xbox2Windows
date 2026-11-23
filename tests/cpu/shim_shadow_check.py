"""Compare two submitted shadow frames from one native, mapped and translated EXE."""
import os
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mesh_shadow_check as shadow
from shim_startup_check import EXPECTED, SHIM_IMPORTS, check


def build(directory, dxc):
    shadow.compile_stage(dxc, directory, "mesh",
                         shadow.lower(shadow.enumerate_pcs(shadow.mesh_program()), shadow.WORDS, (3, 1), shadow.HEADER),
                         "ms_6_5", ["-DTRACE_EXPORTS"])
    shadow.compile_stage(dxc, directory, "caster", shadow.translate_caster(shadow.caster_program()))
    words = shadow.sampler_words("LESS", "point")
    shadow.compile_stage(dxc, directory, "receiver", shadow.translate_receiver(shadow.receiver_program(words), words))
    moved = tuple((x + 4, y) for x, y in shadow.CASTER)
    lights = [shadow.light_casters(shadow.LIGHT_SHEAR, True, corners) for corners in (shadow.CASTER, moved)]
    maps = [shadow.shadow_map(casters) for casters in lights]
    draws = [shadow.draw_inputs([(*point, depth) for point in triangle])
             for casters in lights for triangle, depth in casters]
    draws += shadow.camera_draws() + shadow.camera_draws(moved)
    state = shadow.comparison_sampler_state(words)
    header = struct.pack("<8I", shadow.SIDE, len(lights[0]), len(lights[1]),
                         state["d3d12_filter"], state["d3d12_comparison_func"], 0, 0, 0)
    (directory / "frames.case").write_bytes(header + b"".join(data for data, _ in draws)
                                            + b"".join(trace for _, trace in draws))
    first = shadow.scene_image(maps[0])[0]
    second = shadow.scene_image(maps[1], corners=moved)[0]
    stale = shadow.scene_image(maps[0], corners=moved)[0]
    depths = [b"".join(struct.pack("<f", float(value)) for value in frame) for frame in maps]
    mismatch = shadow.differences(second, stale)
    if not mismatch or maps[0] == maps[1]:
        raise AssertionError("moving the caster must change depth and reveal the stale binding")
    return first + depths[0], second + depths[1], stale + depths[1], mismatch


def run(directory, dxc, mapper, exe, lifted, companion, io, xmem, d3d12):
    first, second, stale, mismatch = build(directory, dxc)
    mapped = [sys.executable, "-B", Path(__file__).with_name("shim_startup_check.py"), "--map",
              mapper, exe, companion, io, xmem, d3d12, ""]
    markers = sorted(f"bound {dll}!{symbol}" for dll, symbols in SHIM_IMPORTS.items() for symbol in symbols)
    created = "Xbox device created: EXE import returned a desktop device"
    closed = "shadow close surfaces=4 device=1"
    frame_zero = [created, "shadow record frame=0 map=0", "shadow submit frame=0 fence=1", "shadow wait fence=1"]
    frame_one = ["shadow submit frame=1 fence=2", "shadow wait fence=2", closed]
    cases = (
        ("shadow", first + second, 42, [*frame_zero, "shadow record frame=1 map=1", *frame_one]),
        ("shadow-stale", first + stale, 42, [*frame_zero, "shadow record frame=1 map=0", *frame_one]),
        ("shadow-reject", b"", 20, ["Xbox device rejected: no device, released adapter and factory",
                                  "shadow close surfaces=0 device=0"]),
        ("shadow-fail", first, 24, [*frame_zero, "shadow record frame=1 map=1",
                                  "shadow submit frame=2 rejected", closed]),
    )
    for case, expected, status, diagnostics in cases:
        env = dict(os.environ, X2W_SHIM_DRAW_CASE=case, X2W_SHIM_COMPUTE_CASE="", X2W_SHIM_CHAIN_DIR=str(directory))
        for runner, command in (("native", [exe]), ("pe_map", mapped), ("AOT", [lifted])):
            child = subprocess.run(command, env=env, capture_output=True, timeout=120)
            lines = child.stderr.decode(errors="replace").splitlines()
            if child.returncode == 77 and case == "shadow" and runner == "native":
                print("skip:", child.stdout.decode(errors="replace"), lines)
                return 77
            if runner == "pe_map":
                check(f"{case} mapped providers", sorted(lines[:len(markers)]), markers)
                check(f"{case} mapped completion", lines[-1:], ["mapped host returned"])
                lines = lines[len(markers):-1]
            check(f"{case} {runner} ordered events", lines, diagnostics)
            check(f"{case} {runner} exit", child.returncode, status)
            check(f"{case} {runner} startup", child.stdout[:len(EXPECTED.output)], EXPECTED.output)
            payload = child.stdout[len(EXPECTED.output):]
            check(f"{case} {runner} frame payload length", len(payload), len(expected))
            check(f"{case} {runner} readback mismatch count", shadow.differences(payload, expected), 0)
            if case == "shadow-stale":
                check(f"{case} {runner} differences from fresh depth binding",
                      shadow.differences(payload, first + second), mismatch)
            print(f"{case} {runner}: {len(payload)} frame bytes, ordered submissions and cleanup, exit={status}")
    print(f"stale map: {mismatch} second-frame color pixels differ; light depth still matches")
    return 0


if __name__ == "__main__":
    with TemporaryDirectory() as folder:
        sys.exit(run(Path(folder), *map(Path, sys.argv[1:-1])))
