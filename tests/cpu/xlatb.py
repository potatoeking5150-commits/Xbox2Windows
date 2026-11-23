"""Compare native and translated XLATB, its REX.W spelling and its 67h (EBX-based) form."""
import ctypes
from pathlib import Path
import re
import sys

import pefile
from iced_x86 import Code, Register

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from ffx_version_query import INTEGER_HEADER, lower, region

U64 = ctypes.c_uint64
MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1

# name: (number of XLATB instructions, table base register iced must decode).
EXPORTS = {
    "xlat_basic": (1, Register.RBX), "xlat_rexw": (1, Register.RBX),
    "xlat_addr32": (1, Register.EBX), "xlat_twice": (2, Register.RBX),
}

# Source mutations of the generated C++. Each control DLL must disagree with the reference somewhere.
CONTROLS = {
    # AL is sign-extended instead of zero-extended, so indices 128 to 255 read below the table.
    "signed": (re.compile(r"\+ static_cast<uint8_t>\(rax\)"), "+ static_cast<int8_t>(rax)"),
    # The 67h sum keeps the upper half of RBX. Those reads fault, and a fault counts as a mismatch.
    "noaddr32": (re.compile(r"static_cast<uint32_t>\((rbx \+ static_cast<uint8_t>\(rax\))\)"), r"(\1)"),
    # The loaded byte replaces all of RAX.
    "clear": (re.compile(r"rax = \(rax & ~0xffull\) \| load_always<uint8_t>"), "rax = load_always<uint8_t>"),
}

PAGE = 0x1000
TABLE_AT = 0x400  # the table sits mid-page; the rest of the page holds 0xcc
TABLE = bytes((167 * k + 13) & 0xff for k in range(256))  # a permutation: 167 is odd
RAX_UPPER = (0, 0xdeadbeefcafe00, 0xffffffffffffff00)
RBX_UPPER = (0, 0xdeadbeef, 0xffffffff, 0x80000000)
PRESETS = (0, 1, 1 << 63, 3 << 62)  # add r8, r8: (ZF, CF) = (1, 0), (0, 0), (1, 1), (0, 1)


def emit(image, output, control="lifted"):
    parts = [INTEGER_HEADER]
    with pefile.PE(str(image), fast_load=True) as pe:
        for name, (count, base) in EXPORTS.items():
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            xlat = [i for i in body if i.code == Code.XLAT_M8]
            if len(xlat) != count or any(i.memory_base != base for i in xlat):
                raise AssertionError(f"{name} does not contain {count} XLATB with base {base}")
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


def low_page():
    """Commit a page below 4 GiB so a 32-bit table address can reach it."""
    virtual_alloc = ctypes.windll.kernel32.VirtualAlloc
    virtual_alloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
    virtual_alloc.restype = ctypes.c_void_p
    for hint in range(0x10000000, 0xf0000000, 0x01000000):
        address = virtual_alloc(hint, PAGE, 0x3000, 0x04)  # MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE
        if address and address + PAGE <= 1 << 32:
            return address
    raise RuntimeError("no committable page below 4 GiB")


def reference(name, rax, preset):
    total = preset + preset
    flags = int(total > MASK64) << 1 | int(total & MASK64 == 0)
    al = rax & 0xff
    for _ in range(EXPORTS[name][0]):
        al = TABLE[al]
    return (rax & ~0xff) | al, flags


def cases():
    for name in EXPORTS:
        for upper in (RBX_UPPER if name == "xlat_addr32" else (0,)):
            for noise in RAX_UPPER:
                for al in range(256):
                    for preset in PRESETS:
                        yield name, (upper, noise | al, preset)


def load_lib(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    for name in EXPORTS:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [U64, U64, U64, ctypes.POINTER(U64)], U64
    return lib


def mismatches(lib, expected, page):
    table = page + TABLE_AT
    found = []
    state = U64()
    for name, (upper, rax, preset), want in expected:
        state.value = 0xdeadbeef
        try:
            got = (getattr(lib, name)(upper << 32 | table, rax, preset, ctypes.byref(state)), state.value)
        except OSError as exc:  # ctypes turns an access violation into OSError
            got = (f"fault: {exc}", None)
        if got != want:
            found.append((name, (upper, rax, preset), want, got))
    return found


def check(native_path, lifted_path, *control_paths):
    page = low_page()
    ctypes.memmove(page, b"\xcc" * PAGE, PAGE)
    ctypes.memmove(page + TABLE_AT, TABLE, len(TABLE))
    expected = [(name, args, reference(name, args[1], args[2])) for name, args in cases()]
    for label, path in (("native", native_path), ("lifted", lifted_path)):
        bad = mismatches(load_lib(path), expected, page)
        if bad:
            name, (upper, rax, preset), want, got = bad[0]
            raise AssertionError(f"{label}: {len(bad)} mismatches; first {name}(rbx upper {upper:#x}, rax {rax:#x}, "
                                 f"preset {preset:#x}) = {got}, expected {tuple(hex(w) for w in want)}")
    if ctypes.string_at(page + TABLE_AT, len(TABLE)) != TABLE:
        raise AssertionError("the table changed")
    print(f"{len(expected)} cases over {len(EXPORTS)} exports (RAX and ZF/CF) matched the reference in native and "
          f"translated DLLs; table at {page + TABLE_AT:#x}")
    for path in control_paths:
        control = Path(path).stem.rsplit("_", 1)[-1]
        bad = mismatches(load_lib(path), expected, page)
        faults = sum(1 for *_, got in bad if isinstance(got[0], str))
        print(f"control {control}: {len(bad)}/{len(expected)} mismatches ({faults} faults)")
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
