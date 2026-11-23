"""Compare native and translated RDRAND and RDSEED in the opt-in host-entropy profile by their properties.

RDRAND is in rdrand.asm and RDSEED in rdseed.asm, each with its own test and CPUID skip. The values are not
reproducible, so the check tests width, carry and flag rules on every read, counts natural failures without
requiring one, and runs deterministic succeed and fail variants of the translation, which replace the host
read by a fixed result. Variation checks only catch a constant translation; they say nothing about entropy.
"""
import ctypes
from pathlib import Path
import re
import sys

import pefile
from iced_x86 import Decoder, FlowControl, Mnemonic as M

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from ffx_version_query import INTEGER_HEADER, lower, region

U64 = ctypes.c_uint64
ROUNDS = 2000
WIDTHS = (16, 32, 64)
MNEMONIC = {"rdrand": M.RDRAND, "rdseed": M.RDSEED}
CPUID_BIT = {"rdrand": "CPUID.01H:ECX bit 30", "rdseed": "CPUID.(EAX=07H,ECX=0):EBX bit 18"}
PRESETS = (0, 1, 1 << 63, 0xA << 60)  # add r8, r8 before the read: (ZF), (), (ZF, CF, OF), (CF, OF)
PATTERN = {16: 0x5a5a, 32: 0x5a5a5a5a, 64: 0x5a5a5a5a5a5a5a5a}
LITERAL = {16: "0x5a5a", 32: "0x5a5a5a5aul", 64: "0x5a5a5a5a5a5a5a5aull"}
# RFLAGS bits the instruction must clear on the host: OF, SF, ZF, AF, PF.
CLEARED = {"OF": 11, "SF": 7, "ZF": 6, "AF": 4, "PF": 2}

STEP = re.compile(r"_rd(?:rand|seed)(16|32|64)_step\(&rnd\)")


def exports(part):
    return [f"{part}_{bits}" for bits in WIDTHS] + [f"{part}_unused"]


# name: (stub, [(pattern, replacement)], rules). stub None keeps the host read; 1 or 0 replaces it by a read
# that stores PATTERN and reports success or failure; "constant" by a successful read of 0x1234.
VARIANTS = {
    "lifted": (None, [], "natural"),
    "succeed": (1, [], "succeed"),
    "fail": (0, [], "fail"),
    # Must-break controls.
    "constant": ("constant", [], "natural"),
    "nocarry": (1, [(r"carry = rnd_ok != 0; ", "carry = false; ")], "succeed"),
    "noext": (1, [(r"(\w+) = static_cast<uint32_t>\(rnd\);",
                   r"\1 = (\1 & ~0xffffffffull) | static_cast<uint32_t>(rnd);")], "succeed"),
    "wide16": (1, [(r"(\w+) = \(\1 & ~65535ull\) \| static_cast<uint16_t>\(rnd\);",
                    r"\1 = static_cast<uint16_t>(rnd);")], "succeed"),
    "flags": (1, [(r"carry = rnd_ok != 0; zero = false; overflow = false; ", "carry = rnd_ok != 0; ")], "succeed"),
    "nozero": (0, [(r"if \(!rnd_ok\) rnd = 0; ", "")], "fail"),
    # Built only at /O2: without volatile the optimizer removes the read whose result the guest discards.
    "plain": (None, [(r"const volatile int rnd_ok", "const int rnd_ok")], "natural"),
}
MUST_PASS = ("lifted", "succeed", "fail")


def emit(part, image, output, variant="lifted"):
    parts = [INTEGER_HEADER]
    with pefile.PE(str(image), fast_load=True) as pe:
        for name in exports(part):
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            found = [i for i in body if i.mnemonic in MNEMONIC.values()]
            if [i.mnemonic for i in found] != [MNEMONIC[part]]:
                raise AssertionError(f"{name} contains {found}, expected one {part.upper()}")
            code = lower(body, ["rcx = a; rdx = b; r8 = c; r9 = state;"], {}, {}, {}, host_entropy=True)
            parts.append(f'extern "C" __declspec(dllexport) word {name}(word a, word b, word c, word state) {{\n{code}\n}}\n')
    text = "".join(parts)
    stub, edits, _ = VARIANTS[variant]
    if stub is not None:
        if stub == "constant":
            replace = lambda m: "(rnd = 0x1234, 1)"
        else:
            replace = lambda m: f"(rnd = {LITERAL[int(m[1])]}, {stub})"
        text, count = STEP.subn(replace, text)
        if count != len(exports(part)):
            raise AssertionError(f"variant {variant}: replaced {count} reads, expected {len(exports(part))}")
    for pattern, replacement in edits:
        text, count = re.subn(pattern, replacement, text)
        if not count:
            raise AssertionError(f"variant {variant} changed nothing for {pattern}")
    if variant != "lifted":
        print(f"variant {variant}: applied")
    output.write_text(text, encoding="ascii")


def rejected_by_default(part, image):
    """The default profile keeps refusing the instruction."""
    with pefile.PE(str(image), fast_load=True) as pe:
        for name in exports(part):
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            try:
                lower(region(pe, *bounds), ["rcx = a; rdx = b; r8 = c; r9 = state;"], {}, {}, {})
            except ValueError as exc:
                if "unsupported query operation" not in str(exc):
                    raise
            else:
                raise AssertionError(f"{name}: the default profile lowered {part.upper()}")


def load_lib(path, part):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    for name in exports(part):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [U64, U64, U64, ctypes.POINTER(U64)], U64
    return lib


def load_harness(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    for name in ("has_rdrand", "has_rdseed"):
        getattr(lib, name).argtypes, getattr(lib, name).restype = [], ctypes.c_uint32
    for name in ("rdrand_rflags", "rdseed_rflags"):
        getattr(lib, name).argtypes, getattr(lib, name).restype = [ctypes.c_uint32], U64
    return lib


def read(lib, name, preset):
    state = (U64 * 2)(0xdeadbeefdeadbeef, 0xdeadbeefdeadbeef)
    getattr(lib, name)(0, 0, preset, ctypes.cast(state, ctypes.POINTER(U64)))
    flags = state[1]
    return state[0], flags & 0xff, flags >> 8 & 0xff, flags >> 16 & 0xff, flags >> 24


def rule(bits, value, cf, zf, of, rest, rules):
    """The first broken rule of one read, or None."""
    if rest:
        return f"flag bytes above OF are {rest:#x}"
    if zf or of:
        return f"ZF {zf} OF {of} after the read, expected both cleared"
    if bits == 16 and value >> 16 != (1 << 48) - 1:
        return f"16-bit read changed bits 16-63: {value:#x}"
    if bits == 32 and value >> 32:
        return f"32-bit read left the upper half: {value:#x}"
    low = value & ((1 << bits) - 1)
    if cf not in (0, 1):
        return f"CF byte {cf}"
    if rules == "succeed" and (cf != 1 or low != PATTERN[bits]):
        return f"succeed stub gave CF {cf} value {low:#x}, expected CF 1 value {PATTERN[bits]:#x}"
    if rules == "fail" and cf != 0:
        return f"fail stub gave CF {cf}, expected 0"
    if cf == 0 and low != 0:
        return f"failed read left {low:#x}, expected 0 at {bits} bits"
    return None


def violations(lib, part, rules):
    """(first violated property or None, {bits: natural failures}, {bits: successful values})."""
    failures, values = {bits: 0 for bits in WIDTHS}, {bits: [] for bits in WIDTHS}
    for round_ in range(ROUNDS):
        preset = PRESETS[round_ % len(PRESETS)]
        for bits in WIDTHS:
            value, cf, zf, of, rest = read(lib, f"{part}_{bits}", preset)
            problem = rule(bits, value, cf, zf, of, rest, rules)
            if problem:
                return f"{part}_{bits} preset {preset:#x}: {problem}", failures, values
            if cf:
                values[bits].append(value & ((1 << bits) - 1))
            else:
                failures[bits] += 1
        unused = getattr(lib, f"{part}_unused")(0, 0, preset, None)
        if unused != 7:
            return f"{part}_unused returned {unused:#x}, expected 7", failures, values
    return None, failures, values


def variation(part, values):
    """(problem or None, measurable)."""
    for bits in WIDTHS:
        if len(values[bits]) < 2:
            return None, False
        if len(set(values[bits])) < 2:
            return f"{part}_{bits}: all {len(values[bits])} successful reads returned {values[bits][0]:#x}", True
    if not any(v >> 32 for v in values[64]):
        return f"{part}_64: no successful read set a bit of the upper half", True
    return None, True


def reads_in(path, part):
    """The RDRAND or RDSEED instructions reachable from each export of a DLL, following every branch (an /O2
    leaf may keep no unwind record)."""
    with pefile.PE(str(path), fast_load=True) as pe:
        base, image = pe.OPTIONAL_HEADER.ImageBase, pe.get_memory_mapped_image()
        reads = {}
        for name in exports(part):
            pending, seen, found = [base + export_rva(pe, name)], set(), 0
            while pending:
                ip = pending.pop()
                while ip not in seen:
                    seen.add(ip)
                    if not base <= ip < base + len(image):
                        raise AssertionError(f"{Path(path).name} {name}: branch leaves the image at {ip:#x}")
                    instruction = Decoder(64, image[ip - base:ip - base + 15], ip=ip).decode()
                    found += instruction.mnemonic == MNEMONIC[part]
                    flow = instruction.flow_control
                    if flow == FlowControl.RETURN:
                        break
                    if flow in (FlowControl.CONDITIONAL_BRANCH, FlowControl.UNCONDITIONAL_BRANCH):
                        pending.append(instruction.near_branch_target)
                        if flow == FlowControl.UNCONDITIONAL_BRANCH:
                            break
                    elif flow not in (FlowControl.NEXT, FlowControl.CALL):
                        raise AssertionError(f"{Path(path).name} {name}: cannot follow {instruction} at {ip:#x}")
                    ip = instruction.next_ip
            reads[name] = found
    return reads


def check(part, harness_path, native_path, lifted_path, *extra_paths):
    harness = load_harness(harness_path)
    if not getattr(harness, f"has_{part}")():
        print(f"skip: this host has no {part.upper()} ({CPUID_BIT[part]})")
        raise SystemExit(77)
    for selector in (0, 1):
        for _ in range(64):
            flags = getattr(harness, f"{part}_rflags")(selector)
            left = [name for name, bit in CLEARED.items() if flags >> bit & 1]
            if left:
                raise AssertionError(f"native {part.upper()} left {left} set (RFLAGS {flags:#x}, preset {selector})")
    print(f"native {part.upper()} cleared OF, SF, ZF, AF and PF after both presets")
    rejected_by_default(part, native_path)
    print(f"the default profile rejects every {part.upper()}")

    measurable = True
    for label, path in (("native", native_path), ("lifted", lifted_path)):
        problem, failures, values = violations(load_lib(path, part), part, "natural")
        if problem:
            raise AssertionError(f"{label}: {problem}")
        spread, ok = variation(part, values)
        if spread:
            raise AssertionError(f"{label}: {spread}")
        measurable &= ok
        print(f"{label}: {ROUNDS} rounds per width kept the width, carry and flag rules; natural failures "
              f"{failures}")
    skipped = []
    for path in extra_paths:
        stem = Path(path).stem
        if stem.endswith("_o2"):
            variant = stem[len(f"cpu_{part}_"):-len("_o2")]
            reads = reads_in(path, part)
            if variant == "plain":
                if reads[f"{part}_unused"]:
                    raise AssertionError(f"{stem}: the discarded {part.upper()} survived without volatile")
                print(f"{stem}: without volatile, /O2 drops the discarded read and the walk reports it: {reads}")
                continue
            if any(count != 1 for count in reads.values()):
                raise AssertionError(f"{stem}: reads {reads}, expected one per export")
            problem, _, values = violations(load_lib(path, part), part, "natural")
            if problem or variation(part, values)[0]:
                raise AssertionError(f"{stem}: {problem or variation(part, values)[0]}")
            print(f"{stem}: every {part.upper()} is still in the /O2 code, including the discarded one, "
                  "and the rules hold")
            continue
        variant = stem.rsplit("_", 1)[-1]
        rules = VARIANTS[variant][2]
        problem, _, values = violations(load_lib(path, part), part, rules)
        if rules == "natural" and problem is None:
            problem, ok = variation(part, values)
            if not ok:
                skipped.append(f"control {variant} is not measurable: fewer than 2 successful reads at a width")
                continue
        if variant in MUST_PASS:
            if problem:
                raise AssertionError(f"variant {variant}: {problem}")
            print(f"variant {variant}: every read followed the {rules} rules")
        else:
            print(f"control {variant}: {problem}")
            if problem is None:
                raise AssertionError(f"control {variant} kept every property")
    if not measurable:
        skipped.insert(0, "fewer than 2 natural successes at a width, so variation is not measurable")
    if skipped:
        print("skip: " + "; ".join(skipped))
        raise SystemExit(77)


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    if mode == "emit":
        part, image, output, *variant = args
        emit(part, Path(image), Path(output), *variant)
    elif mode == "check":
        check(*args)
    else:
        raise ValueError(mode)
