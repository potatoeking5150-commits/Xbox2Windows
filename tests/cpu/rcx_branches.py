"""Compare native and translated LOOP, LOOPE, LOOPNE, JRCXZ and their ECX (67h) forms."""
import ctypes
from pathlib import Path
import re
import sys

import pefile
from iced_x86 import Code

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from ffx_version_query import INTEGER_HEADER, lower, region

U64 = ctypes.c_uint64
MASK64 = (1 << 64) - 1
MASK32 = (1 << 32) - 1

# Every branch Code the fixture must contain, so a misassembled prefix cannot pass silently.
EXPECTED_CODES = {
    "loop_count": Code.LOOP_REL8_64_RCX, "loopd_count": Code.LOOP_REL8_64_ECX,
    "loope_scan": Code.LOOPE_REL8_64_RCX, "loopne_scan": Code.LOOPNE_REL8_64_RCX,
    "looped_scan": Code.LOOPE_REL8_64_ECX, "loopned_scan": Code.LOOPNE_REL8_64_ECX,
    "jrcxz_probe": Code.JRCXZ_REL8_64, "jecxz_probe": Code.JECXZ_REL8_64,
}
EXPORTS = tuple(EXPECTED_CODES)

# Source mutations of the generated C++. Each control DLL must disagree with the reference somewhere.
CONTROLS = {
    # The ECX decrement keeps the upper half of RCX, and JECXZ tests all of RCX. (A full 64-bit LOOP would
    # run for about 2^47 iterations on the cases with upper-half noise, so the loop test stays on ECX.)
    "noecx": (re.compile(r"rcx = static_cast<uint32_t>\(rcx - 1\);|if \(static_cast<uint32_t>\(rcx\) == 0\)"),
              lambda m: ("rcx = (rcx & ~0xffffffffull) | static_cast<uint32_t>(rcx - 1);"
                         if m.group(0).startswith("rcx =") else "if (rcx == 0)")),
    # LOOPE and LOOPNE ignore ZF and behave like LOOP.
    "nozf": (re.compile(r" && !?zero\)"), ")"),
    # JRCXZ and JECXZ branch on a nonzero count.
    "jrcxz": (re.compile(r"== 0\) goto"), "!= 0) goto"),
}

# These values are the low bytes of every scan needle. The tail gives a
# zero-count-guard mutant both a matching and a differing byte for each needle.
SCAN_VALUES = (7, 3, 9, 0, 45)
SCAN = bytes([7] * 12 + [3] + [7] * 5 + [9, 9, 9, 7, 7, 7] + list(range(40, 80)) + [0] + list(SCAN_VALUES))
PRESETS = (0, 1, 1 << 63, 3 << 62)  # add r8, r8: (ZF, CF) = (1, 0), (0, 0), (1, 1), (0, 1)
UPPER = (0, 0xabcd, MASK32)


def emit(image, output, control="lifted"):
    parts = [INTEGER_HEADER]
    with pefile.PE(str(image), fast_load=True) as pe:
        for name in EXPORTS:
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            codes = {i.code for i in body}
            if EXPECTED_CODES[name] not in codes:
                raise AssertionError(f"{name} does not contain {EXPECTED_CODES[name]}")
            code = lower(body, ["rcx = a; rdx = b; r8 = c; r9 = state;"], {}, {}, {})
            parts.append(f'extern "C" __declspec(dllexport) word {name}(word a, word b, word c, word state) {{\n{code}\n}}\n')
    text = "".join(parts)
    if control != "lifted":
        expression, replacement = CONTROLS[control]
        text, count = expression.subn(replacement, text)
        if not count:
            raise AssertionError(f"control {control} changed nothing")
        print(f"control {control}: {count} replacements")
    output.write_text(text, encoding="ascii")


def preset_flags(preset):
    total = preset + preset
    return int(total & MASK64 == 0), int(total > MASK64)


def reference(name, count, offset, value):
    """(RAX, state[0] flags, state[1] RCX, state[2] pointer offset)."""
    ecx = name in ("loopd_count", "looped_scan", "loopned_scan", "jecxz_probe")

    def low(rcx):
        return rcx & MASK32 if ecx else rcx

    def decrement(rcx):
        return (rcx - 1) & MASK32 if ecx else (rcx - 1) & MASK64  # a 32-bit write zero-extends

    rcx = count
    if name in ("loop_count", "loopd_count"):
        zf, cf = preset_flags(value)
        rax = 0
        while True:
            rax += 1
            rcx = decrement(rcx)
            if low(rcx) == 0:
                break
        return rax, cf << 1 | zf, rcx, 0
    if name in ("jrcxz_probe", "jecxz_probe"):
        zf, cf = preset_flags(value)
        return (1 if low(rcx) == 0 else 2), cf << 1 | zf, rcx, 0
    # Scans: the flags start from add r8, 0 and then come from each cmp.
    zf, cf = int(value == 0), 0
    pointer = offset
    if low(rcx) != 0:
        while True:
            a, b = SCAN[pointer], value & 0xff
            zf, cf = int(a == b), int(a < b)
            pointer += 1
            rcx = decrement(rcx)
            keep = zf if name in ("loope_scan", "looped_scan") else not zf
            if not (low(rcx) != 0 and keep):
                break
    return 0, cf << 1 | zf, rcx, pointer - offset


def cases():
    for name in EXPORTS:
        if name in ("loop_count", "loopd_count"):
            for upper in (UPPER if name == "loopd_count" else (0,)):
                for n in (*range(1, 41), 1000):
                    for preset in PRESETS:
                        yield name, (upper << 32 | n, 0, preset)
        elif name in ("jrcxz_probe", "jecxz_probe"):
            for rcx in (0, 1, 1 << 32, MASK32, MASK64, 0xabcd00000000, 0xabcd00000001):
                for preset in PRESETS:
                    yield name, (rcx, 0, preset)
        else:
            ecx = name.startswith(("looped", "loopned"))
            for upper in (UPPER if ecx else (0,)):
                for n in range(0, 25):
                    for offset in (0, 10, 12, 13, 18, 21, 24):
                        for value in SCAN_VALUES:
                            yield name, (upper << 32 | n, offset, value)


def load_lib(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    for name in EXPORTS:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [U64, ctypes.c_void_p, U64, ctypes.POINTER(U64 * 3)], U64
    return lib


def mismatches(lib, expected):
    buffer = (ctypes.c_uint8 * len(SCAN)).from_buffer_copy(SCAN)
    state = (U64 * 3)()
    found = []
    for name, (count, offset, value), want in expected:
        state[0] = state[1] = state[2] = 0xdeadbeef
        rax = getattr(lib, name)(count, ctypes.addressof(buffer) + offset, value, ctypes.byref(state))
        got = (rax, state[0], state[1], state[2])
        if got != want:
            found.append((name, (count, offset, value), want, got))
    return found


def check(native_path, lifted_path, *control_paths):
    expected = [(name, args, reference(name, *args)) for name, args in cases()]
    for label, path in (("native", native_path), ("lifted", lifted_path)):
        bad = mismatches(load_lib(path), expected)
        if bad:
            name, args, want, got = bad[0]
            raise AssertionError(f"{label}: {len(bad)} mismatches; first {name}{tuple(hex(a) for a in args)} = "
                                 f"{tuple(hex(g) for g in got)}, expected {tuple(hex(w) for w in want)}")
    print(f"{len(expected)} cases over {len(EXPORTS)} exports (RAX, RCX, pointer, ZF/CF) matched the reference "
          "in native and translated DLLs")
    for path in control_paths:
        control = Path(path).stem.rsplit("_", 1)[-1]
        bad = mismatches(load_lib(path), expected)
        print(f"control {control}: {len(bad)}/{len(expected)} mismatches")
        if not bad:
            raise AssertionError(f"control {control} matched every case")


if __name__ == "__main__":
    mode, native, other, *rest = sys.argv[1:]
    if mode == "emit":
        emit(Path(native), Path(other), *rest)
    elif mode == "check":
        check(native, other, *rest)
    else:
        raise ValueError(mode)
