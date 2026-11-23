"""Compare all 32 result bits of the plain e32 float-to-unsigned pixel conversion."""
from pathlib import Path
import os
import re
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

from ffx_dispatch_aot_check import CONVERSIONS
from pixel_aot_check import (
    assembly, compile_shader, translate, translate_buffer_color, unsigned_conversion, write_record,
)


CASES = [(bits, word) for bits, word, _ in CONVERSIONS] + [
    (0x3ff00000, 1), (0x40700000, 3), (0x3f7fffff, 0), (0x4b812345, 0x0102468a),
    (0x7f800001, 0), (0xff800001, 0), (0xffc12345, 0), (0x4f800001, 0xffffffff),
]


def check_forms():
    lines = unsigned_conversion(True)
    i = lines.index("v_cvt_u32_f32_e32 v32, s4")
    for instruction, diagnostic in (
        ("v_cvt_u32_f32_e32 v32, v3", "Unbound pixel input v3"),
        ("v_cvt_u32_f32_e32 v256, s4", "Invalid unsigned conversion register"),
        ("v_cvt_u32_f32_e64 v32, |s4|", "Invalid conversion"),
        ("v_cvt_u32_f32_e64 v32, s4 clamp", "Unsupported instruction"),
        ("v_cvt_u32_f32_e32 v32, -s4", "Unsupported instruction"),
    ):
        try:
            translate_buffer_color(lines[:i] + [instruction] + lines[i+1:])
        except ValueError as exc:
            if diagnostic not in str(exc):
                raise
        else:
            raise AssertionError(instruction)
    for bad, message in (
        (lines[:5] + lines[6:], "Constant read before load completion"),
        (lines[:6] + ["v_mov_b32_e32 v34, s4"] + lines[6:i] +
         ["v_cvt_u32_f32_e32 v32, v34"] + lines[i+1:], "helper-lane definition"),
    ):
        try:
            translate_buffer_color(bad)
        except ValueError as exc:
            if message not in str(exc):
                raise
        else:
            raise AssertionError(message)
    sampled = assembly(False)
    wait = sampled.index("s_waitcnt vmcnt(0)")
    for dst, src, diagnostic in (("v12", "0x3ff00000", "Write overlaps an outstanding sample"),
                                 ("v32", "v12", "Sample used before vmcnt wait")):
        try:
            translate(sampled[:wait] + [f"v_cvt_u32_f32_e32 {dst}, {src}"] + sampled[wait:])
        except ValueError as exc:
            if diagnostic not in str(exc):
                raise
        else:
            raise AssertionError(diagnostic)


def run(dxc, renderer, adapter, directory):
    check_forms()
    cases = []
    for index, (bits, word) in enumerate(CASES):
        data = bytearray(256)
        struct.pack_into("<I", data, 16, bits)
        color = struct.pack("<4f", *word.to_bytes(4, "little"))
        path = directory / f"{index}.case"
        path.write_bytes(data + color * 64)
        cases.append(path)
    for variant in ("entry", "whole-quad", "wrong-rounding"):
        text = translate_buffer_color(unsigned_conversion(variant != "entry"))
        expected = ["pass"] * len(cases)
        selected = cases
        if variant == "wrong-rounding":
            assert text.count("uint(finite)") == 1
            text = text.replace("uint(finite)", "uint(round(finite))")
            selected = [cases[next(i for i, (bits, _) in enumerate(CASES) if bits == value)]
                        for value in (0x3ff00000, 0x40700000)]
            expected = ["mismatch", "mismatch"]  # 1.875 and 3.75 would become 2 and 4
        source, shader = directory / "shader.hlsl", directory / "shader.dxil"
        source.write_text(text)
        compile_shader(dxc, source, shader)
        result = subprocess.run([renderer, adapter, "--check-pixel-batch", shader, *selected],
                                capture_output=True, text=True, timeout=90)
        if result.returncode == 77 and adapter == "hardware" and variant == "entry":
            print(result.stdout + result.stderr)
            return 77
        statuses = re.findall(r"^case=(\d+) status=(\w+)$", result.stdout, re.M)
        if (statuses != [(str(i), s) for i, s in enumerate(expected)]
                or result.returncode != int(variant == "wrong-rounding")):
            raise AssertionError(result.stdout + result.stderr)
        print(f"{adapter} {variant}: {len(selected)} words; {expected.count('mismatch')} expected mismatches")
    write_record(os.environ.get("FORM_COVERAGE_RECORD"))
    return 0


if __name__ == "__main__":
    with TemporaryDirectory() as folder:
        sys.exit(run(*sys.argv[1:], Path(folder)))
