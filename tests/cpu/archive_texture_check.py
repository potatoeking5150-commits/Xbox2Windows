"""Observe each completed archive record at the synchronous RGBA8 handoff."""
from pathlib import Path
import subprocess
import sys

from archive_check import FIRST_RECORD, SECOND_RECORD, check


def expected_texture(red):
    levels = (0, 85, 170, 255)
    pixels = bytearray()
    for y, blue in enumerate(levels):
        for x, green in enumerate(levels):
            pixels.extend((red[4 * y + x], green, blue, 255))
    return bytes(pixels)


def compare(mapper, native, lifted, codec, blast):
    first = expected_texture(b"AIAIAIAIAIAIA" + b"AIA")
    second = expected_texture(b"BIBIBIBIBIBIB" + b"BIB")
    mapped = [sys.executable, "-B", Path(__file__).with_name("archive_check.py"),
              "--mapped", mapper, native, codec, blast]
    complete = b"alloc 8\ninput 8\noutput 13\nfree\n"
    truncated = b"alloc 8\ninput 4\ninput 0\noutput 1\nfree\n"
    cases = [
        ("both", FIRST_RECORD + SECOND_RECORD, first + second, 0, complete * 2 + b"completed 2\n"),
        ("truncated", FIRST_RECORD + SECOND_RECORD[:5], first, 2, complete + truncated + b"completed 1\n"),
    ]
    for runner, command in (("native", [native]), ("pe_map", mapped), ("AOT", [lifted])):
        for label, data, expected, status, trace in cases:
            child = subprocess.run(command, input=data, capture_output=True, timeout=15)
            check(f"{runner} {label} status/trace", (child.returncode, child.stderr), (status, trace))
            check(f"{runner} {label} RGBA8 bytes", child.stdout, expected)
            if label == "both":
                stale = sum(x != y for x, y in zip(child.stdout[64:], first, strict=True))
                check(f"{runner} stale texture differing bytes", stale, 9)
            print(f"{runner} {label}: {len(child.stdout)} RGBA8 bytes, exit {child.returncode}")


if __name__ == "__main__":
    compare(*map(Path, sys.argv[1:]))
