"""Compare native and translated x64 string instructions (MOVS, STOS, LODS, SCAS, CMPS) with REP prefixes and DF."""
import ctypes
from pathlib import Path
import re
import sys

import pefile
from iced_x86 import Decoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from ffx_version_query import INTEGER_HEADER, lower, region

U64 = ctypes.c_uint64
SUFFIX = {1: "b", 2: "w", 4: "d", 8: "q"}

# (export, kind, width, repeat prefix, DF set around the instruction, STD then CLD before it)
EXPORTS = (
    *((f"once_{kind}{SUFFIX[w]}", kind, w, None, False, False) for kind in ("movs", "stos", "lods", "scas", "cmps") for w in SUFFIX),
    *((f"rep_{kind}{SUFFIX[w]}", kind, w, "rep", False, False) for kind in ("movs", "stos", "lods") for w in SUFFIX),
    *((f"{rep}_{kind}{SUFFIX[w]}", kind, w, rep, False, False)
      for kind in ("scas", "cmps") for rep in ("repe", "repne") for w in SUFFIX),
    ("std_movsw", "movs", 2, None, True, False),
    ("std_rep_movsb", "movs", 1, "rep", True, False),
    ("std_rep_movsq", "movs", 8, "rep", True, False),
    ("std_stosb", "stos", 1, None, True, False),
    ("std_rep_stosw", "stos", 2, "rep", True, False),
    ("std_rep_stosd", "stos", 4, "rep", True, False),
    ("std_lodsb", "lods", 1, None, True, False),
    ("std_lodsq", "lods", 8, None, True, False),
    ("std_cmpsb", "cmps", 1, None, True, False),
    ("std_repne_scasb", "scas", 1, "repne", True, False),
    ("std_repe_scasq", "scas", 8, "repe", True, False),
    ("std_repe_cmpsw", "cmps", 2, "repe", True, False),
    ("std_repne_cmpsd", "cmps", 4, "repne", True, False),
    ("std_cld_rep_movsb", "movs", 1, "rep", False, True),
)

# Source mutations of the generated C++. Each control DLL must disagree with the reference somewhere.
CONTROLS = {
    # A whole-range memmove gives the same bytes only when the ranges do not overlap forward.
    "memmove": (re.compile(r"while \(rcx != 0\) \{ store<(\w+)>\(rdi, load(?:_always)?<\1>\(rsi\)\); rsi \+= step; rdi \+= step; --rcx; \}"),
                r"if (rcx != 0) { std::memmove(reinterpret_cast<void*>(rdi), reinterpret_cast<const void*>(rsi), "
                r"rcx * sizeof(\1)); rsi += rcx * step; rdi += rcx * step; rcx = 0; }"),
    # CMPS subtracts the destination element from the source element.
    "swapped": (re.compile(r"a = load_always<(\w+)>\(rsi\), b = load_always<\1>\(rdi\)"), r"a = load_always<\1>(rdi), b = load_always<\1>(rsi)"),
    # STD walks the pointers downward.
    "forward": (re.compile(r"direction = true;"), "direction = false;"),
}

SIZE = 1024
BOUNDS = (16, SIZE - 16)  # guard bytes on both ends must stay untouched
# (destination, source) offsets: 96 apart they match until the breaks at 520 and 616 (or start on them),
# 1 and 3 apart they overlap within an element, and 704/720 compare inside the 0x5a run.
PAIRS = ((608, 512), (512, 608), (616, 520), (520, 616), (401, 400), (400, 401), (403, 400), (400, 403),
         (720, 704), (704, 720), (760, 800), (300, 900))
COUNTS = (0, 1, 2, 5, 17, 40)
FLAG_SELECTORS = (0, 1, 2)  # cmp state[0], 1: CF only, ZF only, neither


def pattern():
    """A period-16 pattern of distinct bytes, so offsets 16 apart compare equal, with planted breaks
    at 520 and 616 (rel. 8 from 512 and 608), 480 below them, and a 0x5a run over 700..763 broken at 731."""
    base = bytes((n * 37 + 11) & 0xff for n in range(16))
    buf = bytearray(base[n % 16] for n in range(SIZE))
    buf[520], buf[616], buf[480] = 0x00, 0xff, 0x11
    buf[700:764] = b"\x5a" * 64
    buf[731] = 0x3c
    return bytes(buf)


PATTERN = pattern()
# A wrong lowering (the mutated controls) can walk past the 1 KiB buffer, so it sits between 1 KiB pads of
# 0xcc. The comparison covers the pads too, so a stray write is a mismatch instead of heap corruption.
PAD = 1024
PADDED = bytes([0xcc]) * PAD + PATTERN + bytes([0xcc]) * PAD
RAX_VALUES = (0x5a5a5a5a5a5a5a5a, 0x0123456789abcdef, int.from_bytes(PATTERN[1000:1008], "little"))


def emit(image, output, control="lifted"):
    parts = [INTEGER_HEADER]
    with pefile.PE(str(image), fast_load=True) as pe:
        for name, *_ in EXPORTS:
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            code = lower(region(pe, *bounds), ["rcx = a; rdx = b; r8 = c; r9 = state;"], {}, {}, {})
            parts.append(f'extern "C" __declspec(dllexport) word {name}(word a, word b, word c, word state) {{\n{code}\n}}\n')
    text = "".join(parts)
    if control != "lifted":
        expression, replacement = CONTROLS[control]
        text, count = expression.subn(replacement, text)
        if not count:
            raise AssertionError(f"control {control} changed nothing")
        print(f"control {control}: {count} replacements")
    output.write_text(text, encoding="ascii")


def reference(kind, width, repeat, down, buf, rdi, rsi, rcx, rax, flags):
    """Architectural result: one element at a time, the count tested before and ZF after each element."""
    step = -width if down else width
    mask = (1 << (8 * width)) - 1
    zf, cf = flags
    touched = []

    def element(address):
        touched.append(address)
        return int.from_bytes(buf[address:address + width], "little")

    def put(address, value):
        touched.append(address)
        buf[address:address + width] = (value & mask).to_bytes(width, "little")

    def once():
        nonlocal rdi, rsi, rax, zf, cf
        if kind == "movs":
            put(rdi, element(rsi))
            rsi += step
            rdi += step
        elif kind == "stos":
            put(rdi, rax)
            rdi += step
        elif kind == "lods":
            value = element(rsi)
            rax = value if width >= 4 else (rax & ~mask) | value
            rsi += step
        else:
            a = rax & mask if kind == "scas" else element(rsi)
            b = element(rdi)
            zf, cf = int(a == b), int(a < b)
            if kind == "cmps":
                rsi += step
            rdi += step

    if repeat is None:
        once()
    else:
        while rcx:
            once()
            rcx -= 1
            if (repeat == "repe" and not zf) or (repeat == "repne" and zf):
                break
    inside = all(BOUNDS[0] <= t and t + width <= BOUNDS[1] for t in touched)
    return inside, (cf << 1 | zf, rax, rsi, rdi, rcx)


def load_lib(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    for name, *_ in EXPORTS:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [U64, U64, U64, ctypes.POINTER(U64 * 5)], U64
    return lib


def run(lib, name, buf, dst, src, count, selector, rax):
    """Call one export on a fresh copy of the padded pattern; returns (state, memory) with pointers as offsets."""
    ctypes.memmove(buf, PADDED, len(PADDED))
    base = ctypes.addressof(buf) + PAD
    state = (U64 * 5)(selector, rax, 0, 0, 0)
    returned = getattr(lib, name)(base + dst, base + src, count, ctypes.byref(state))
    if returned != state[1]:
        raise AssertionError(f"{name} returned {returned:#x} but stored RAX {state[1]:#x}")
    flags, rax, rsi, rdi, rcx = state
    return (flags, rax, (rsi - base) & 0xffffffffffffffff, (rdi - base) & 0xffffffffffffffff, rcx), bytes(buf)


def cases():
    for name, kind, width, repeat, down, _ in EXPORTS:
        for dst, src in PAIRS:
            for count in COUNTS:
                for selector in FLAG_SELECTORS:
                    for rax in RAX_VALUES:
                        flags = ((0, 1), (1, 0), (0, 0))[selector]
                        buf = bytearray(PATTERN)
                        inside, state = reference(kind, width, repeat, down, buf, dst, src, count, rax, flags)
                        if inside:
                            yield name, (dst, src, count, selector, rax), state, PADDED[:PAD] + buf + PADDED[-PAD:]


def mismatches(lib, expected):
    buf = (ctypes.c_uint8 * len(PADDED))()
    found = []
    for name, args, state, memory in expected:
        got = run(lib, name, buf, *args)
        if got != (state, memory):
            found.append((name, args, state, got[0]))
    return found


def check(native_path, lifted_path, *control_paths):
    expected = list(cases())
    per_export = {name: 0 for name, *_ in EXPORTS}
    for name, *_ in expected:
        per_export[name] += 1
    if min(per_export.values()) == 0:
        raise AssertionError(f"exports without an in-bounds case: {[n for n, c in per_export.items() if not c]}")
    for label, path in (("native", native_path), ("lifted", lifted_path)):
        bad = mismatches(load_lib(path), expected)
        if bad:
            name, args, want, got = bad[0]
            raise AssertionError(f"{label}: {len(bad)} mismatches; first {name}{args} = {got}, expected {want}")
    print(f"{len(expected)} cases over {len(EXPORTS)} exports (memory, RAX, RSI, RDI, RCX, ZF/CF) "
          "matched the reference in native and translated DLLs")
    for path in control_paths:
        control = Path(path).stem.rsplit("_", 1)[-1]
        bad = mismatches(load_lib(path), expected)
        print(f"control {control}: {len(bad)}/{len(expected)} mismatches")
        if not bad:
            raise AssertionError(f"control {control} matched every case")


def check_rejections():
    for code, diagnostic in (
        ("67 f3 a4 c3", "unsupported string address size at 0x1000"),  # rep movs [edi], [esi]
        ("67 ae c3", "unsupported string address size at 0x1000"),     # scas al, [edi]
        ("f2 a4 c3", "unsupported REPNE string prefix at 0x1000"),     # repne movsb
        ("f2 48 ab c3", "unsupported REPNE string prefix at 0x1000"),  # repne stosq
        ("f2 ac c3", "unsupported REPNE string prefix at 0x1000"),     # repne lodsb
        ("64 a4 c3", "unsupported instruction prefix at 0x1000"),      # movs from fs:[rsi]
    ):
        try:
            lower(list(Decoder(64, bytes.fromhex(code), ip=0x1000)), [], {}, {}, {})
        except ValueError as exc:
            if not str(exc).startswith(diagnostic):
                raise AssertionError(f"{code}: unexpected diagnostic {exc}") from exc
        else:
            raise AssertionError(f"accepted {code}")
    # The SSE2 scalar MOVSD and CMPSD share mnemonics with the string forms and stay unsupported here.
    for code in ("f2 0f 10 c1 c3", "f2 0f c2 c1 00 c3"):
        try:
            lower(list(Decoder(64, bytes.fromhex(code), ip=0x1000)), [], {}, {}, {})
        except ValueError as exc:
            if not str(exc).startswith("unsupported query operation at 0x1000"):
                raise AssertionError(f"{code}: unexpected diagnostic {exc}") from exc
        else:
            raise AssertionError(f"accepted {code}")
    print("rejected 67h string addressing, REPNE on MOVS/STOS/LODS, an FS source and the SSE2 MOVSD/CMPSD forms")


if __name__ == "__main__":
    mode, native, other, *rest = sys.argv[1:]
    if mode == "emit":
        emit(Path(native), Path(other), *rest)
    elif mode == "check":
        check(native, other, *rest)
        check_rejections()
    else:
        raise ValueError(mode)
