"""Draw each completed archive record through D3D12 and compare the frames across runners.

The GPU codec (archive/texture_gpu.cpp) uploads each record's 4x4 RGBA8 texture,
draws a point-sampled quad on a 16x16 target, waits on the frame's fence and
writes the readback to stdout. One device, queue and pipeline serve all records.

both       two complete records: two frames, fences 1 and 2, exit 0.
truncated  the second record fails in blast: only the first frame is drawn and
           the decoder status 2 reaches the process exit.
first      the first record fails in blast: no device is created, nothing is
           drawn and the process exits 2.
stale      X2W_ARCHIVE_TEXTURE_CASE=stale skips the second upload: the second
           frame repeats the first and differs from its own reference.
fail       X2W_ARCHIVE_TEXTURE_CASE=fail passes a second texture one byte short:
           the upload is refused before any GPU work, the device is still
           released and the consumer status 4 reaches the process exit.
leak       X2W_ARCHIVE_TEXTURE_CASE=leak keeps one extra device object: the
           release reports one remaining device reference.

Every case runs the same reader image natively, under pe_map and as the AOT
build, and must match the independent CPU reference, the upload hashes, the
completed fence values, the device references left after release, and the
exit status. All runs must report the same adapter and driver, which the
native run supplies. X2W_SHIM_DRAW_ADAPTER selects WARP or hardware; a missing
physical adapter exits 77.
"""
import os
from pathlib import Path
import subprocess
import sys

from archive_check import FIRST_RECORD, SECOND_RECORD, check
from archive_texture_check import expected_texture

SIDE = 16
SCALE = SIDE // 4
FIRST = expected_texture(b"AIAIAIAIAIAIA" + b"AIA")
SECOND = expected_texture(b"BIBIBIBIBIBIB" + b"BIB")


def frame(texture):
    """Point sampling at pixel centres maps pixel (x, y) to texel (x / 4, y / 4)."""
    pixels = bytearray()
    for y in range(SIDE):
        for x in range(SIDE):
            texel = ((y // SCALE) * 4 + x // SCALE) * 4
            pixels += texture[texel:texel + 4]
    return bytes(pixels)


def first_difference(actual, expected):
    """Name the first differing pixel instead of printing whole frames."""
    if actual == expected:
        return None
    if len(actual) != len(expected):
        return f"{len(actual)} bytes instead of {len(expected)}"
    index = next(i for i, (x, y) in enumerate(zip(actual, expected)) if x != y) // 4
    frame_index, pixel = divmod(index, SIDE * SIDE)
    y, x = divmod(pixel, SIDE)
    got, want = (data[index * 4:index * 4 + 4].hex() for data in (actual, expected))
    return f"frame {frame_index + 1} pixel ({x}, {y}): {got} instead of {want}"


def upload(texture):
    value = 0xcbf29ce484222325
    for byte in texture:
        value = ((value ^ byte) * 0x100000001b3) & 0xffffffffffffffff
    return f"upload {value:016x}"


def device_line(native):
    """The adapter and driver the native run reports; every other run must match it."""
    child = subprocess.run([native], input=FIRST_RECORD, capture_output=True, timeout=60,
                           env={**os.environ, "X2W_ARCHIVE_TEXTURE_CASE": ""})
    if child.returncode == 77 and os.environ.get("X2W_SHIM_DRAW_ADAPTER") == "hardware":
        print(child.stderr.decode(errors="replace").strip())
        print("capability skip: no D3D12 physical adapter")
        sys.exit(77)
    check("native device probe exit", child.returncode, 0)
    lines = [line for line in child.stderr.decode("utf-8").splitlines() if line.startswith("gpu device ")]
    check("native device lines", len(lines), 1)
    return lines[0]


def compare(mapper, native, lifted, codec, blast):
    device = device_line(native)
    print(device)
    full = ["alloc 8", "input 8", "output 13"]
    partial = ["alloc 8", "input 4", "input 0", "output 1"]
    first = [device, upload(FIRST), "fence 1", "free"]
    second = full + [upload(SECOND), "fence 2", "free"]
    stale = frame(FIRST)
    mismatch = sum(x != y for x, y in zip(stale, frame(SECOND), strict=True))
    check("stale frame bytes differing from the second reference", mismatch, 9 * SCALE * SCALE)
    both = FIRST_RECORD + SECOND_RECORD
    cases = [
        ("both", "", both, frame(FIRST) + frame(SECOND), 0,
         full + first + second + ["gpu release 0", "completed 2"]),
        ("truncated", "", FIRST_RECORD + SECOND_RECORD[:5], frame(FIRST), 2,
         full + first + partial + ["free", "gpu release 0", "completed 1"]),
        ("first", "", FIRST_RECORD[:5], b"", 2, partial + ["free", "completed 0"]),
        ("stale", "stale", both, frame(FIRST) + stale, 0,
         full + first + full + ["upload skipped", "fence 2", "free", "gpu release 0", "completed 2"]),
        ("fail", "fail", both, frame(FIRST), 4,
         full + first + full + ["texture geometry mismatch: rows=4 row_bytes=16 size=63", "free",
                                "gpu release 0", "completed 1"]),
        ("leak", "leak", both, frame(FIRST) + frame(SECOND), 0,
         full + first + second + ["gpu release 1", "completed 2"]),
    ]
    mapped = [sys.executable, "-B", Path(__file__).with_name("archive_check.py"),
              "--mapped", mapper, native, codec, blast]
    for runner, command in (("native", [native]), ("pe_map", mapped), ("AOT", [lifted])):
        for label, case, data, output, status, events in cases:
            env = {**os.environ, "X2W_ARCHIVE_TEXTURE_CASE": case}
            child = subprocess.run(command, input=data, capture_output=True, timeout=60, env=env)
            trace = ("\n".join(events) + "\n").encode("utf-8")
            check(f"{runner} {label} trace and status", (child.stderr, child.returncode), (trace, status))
            check(f"{runner} {label} first differing pixel", first_difference(child.stdout, output), None)
            print(f"{runner} {label}: {len(child.stdout) // (SIDE * SIDE * 4)} frames, exit {status}")


if __name__ == "__main__":
    compare(*map(Path, sys.argv[1:]))
