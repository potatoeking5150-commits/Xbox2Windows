"""Compare the faults of functions that read guest memory with those of their translation by the shared
integer lowering, built by the project's build and with /O2: each read happens once, at its own width,
whether or not its value is used."""
import ctypes
from pathlib import Path
import sys

import pefile
from iced_x86 import Decoder, Mnemonic

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from ffx_version_query import INTEGER_HEADER, lower, region
from x87_lower import X87_HEADER

Word = ctypes.c_uint64
MASK = (1 << 64) - 1
GUARDS = (0x5a5affff12345678, 0xa5a5000087654321)
PAGE = 0x1000
VALUE = 0xfedcba9876543210
ACCESS_VIOLATION = 0xC0000005
# Export: the read it performs, as (width in bytes, the template argument of its volatile read).
READS = {"read_dead": (8, "uint64_t"), "cmp_dead": (4, "uint32_t"), "test_dead": (1, "uint8_t"),
         "cmov_dead": (8, "uint64_t"), "add_in_place": (4, "uint32_t"), "read_live": (8, "uint64_t"),
         "fld_dead": (8, "uint64_t"), "read_vector_dead": (16, "std::array<uint8_t,16>"),
         "copy_bytes": (1, "uint8_t")}
X87 = ("fld_dead",)
INITIAL = ["rcx = a; rdx = b; r8 = reinterpret_cast<word>(trace);"]


def lift_export(pe, name):
    (bounds,) = function_ranges(pe, export_rva(pe, name))
    body = region(pe, *bounds)
    # The translated function uses local variables for RSI/RDI. Their native
    # save and restore are supplied by the host ABI.
    host_ops = {i.ip: ";" for i in body if name == "copy_bytes" and i.mnemonic in
                (Mnemonic.PUSH, Mnemonic.POP)}
    return lower(body, INITIAL, {}, {}, {}, host_ops=host_ops, x87=name in X87)


def emit(image, output):
    parts = [INTEGER_HEADER, X87_HEADER]
    with pefile.PE(str(image), fast_load=True) as pe:
        for name in READS:
            code = lift_export(pe, name)
            parts.append(f'extern "C" __declspec(dllexport) word lifted_{name}(word a, word b, word* trace) {{\n'
                         f'{code}\n}}\n')
    output.write_text("".join(parts), encoding="ascii")


def check_frame_read():
    # NOP; MOV RAX,[RSP+16]; RET. A host replacement can invalidate the
    # private-frame proof even though the decoded instruction was a NOP.
    body = list(Decoder(64, bytes.fromhex("90 48 8b 44 24 10 c3"), ip=0x1000))
    fixed = lower(body, [], {}, {}, {})
    replaced = lower(body, [], {}, {}, {}, host_ops={0x1000: "rsp = rcx;"})
    if "load<" not in fixed or "load_always<" in fixed or replaced.count("load_always<") != 1:
        raise AssertionError("private-frame load escaped its address provenance")


def check_source(image):
    """Each export's one memory operand is read once, at its width, through the volatile load."""
    with pefile.PE(str(image), fast_load=True) as pe:
        for name, (width, ctype) in READS.items():
            body = lift_export(pe, name)
            reads = body.count("load_always<")
            if reads != 1 or f"load_always<{ctype}>(" not in body or "load<" in body:
                raise AssertionError(f"{name}: {reads} volatile read(s), expected one {ctype} read:\n{body}")
            print(f"control {name}: its {width}-byte memory operand is read once through the volatile load")


def reference(name, y, offset):
    """For a readable operand at byte offset of the page: (0, RAX, the three trace words, the page's words)."""
    page = bytearray(VALUE.to_bytes(8, "little") * (PAGE // 8))
    width = READS[name][0]
    value = int.from_bytes(page[offset:offset + width], "little")
    if name == "cmp_dead":
        rax = (y + 1) & MASK
    elif name == "cmov_dead":
        rax = 0
    elif name == "add_in_place":
        total = (value + y) & 0xffffffff
        page[offset:offset + 4] = total.to_bytes(4, "little")
        rax = int(total == 0)
    elif name == "read_live":
        rax = value
    else:
        rax = y
    words = tuple(int.from_bytes(page[k:k + 8], "little") for k in range(0, PAGE, 8))
    return 0, rax, (1, 2, MASK), words


def two_pages(kernel32):
    """A writable page holding VALUE in every word, followed by an inaccessible page."""
    base = kernel32.VirtualAlloc(None, 2 * PAGE, 0x3000, 0x04)  # MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE
    if not base:
        raise ctypes.WinError(ctypes.get_last_error())
    old = ctypes.c_uint32()
    if not kernel32.VirtualProtect(base + PAGE, PAGE, 0x01, ctypes.byref(old)):  # PAGE_NOACCESS
        raise ctypes.WinError(ctypes.get_last_error())
    return base


def check(guard_path, original_path, lifted_path, optimized_path):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.VirtualAlloc.restype = ctypes.c_void_p
    kernel32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
    kernel32.VirtualProtect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_void_p]
    guarded = ctypes.CDLL(guard_path).guarded_call
    guarded.argtypes = [ctypes.c_void_p, Word, Word, ctypes.POINTER(Word), ctypes.POINTER(Word)]
    guarded.restype = ctypes.c_uint32
    base = two_pages(kernel32)
    words = (Word * (PAGE // 8)).from_address(base)
    native = ctypes.CDLL(original_path)
    sides = [("original", lambda name: getattr(native, name))]
    for label, path in (("translation", lifted_path), ("/O2 translation", optimized_path)):
        library = ctypes.CDLL(path)
        sides.append((label, lambda name, library=library: getattr(library, "lifted_" + name)))

    def run(function, pointer, y):
        """A fresh page and fresh trace words, then (exception code or 0, RAX or None, trace, the page's words)."""
        words[:] = [VALUE] * len(words)
        trace = (Word * 5)(GUARDS[0], MASK, MASK, MASK, GUARDS[1])
        result = Word(0)
        code = guarded(ctypes.cast(function, ctypes.c_void_p), pointer, y,
                       ctypes.cast(ctypes.byref(trace, ctypes.sizeof(Word)), ctypes.POINTER(Word)), ctypes.byref(result))
        if (trace[0], trace[4]) != GUARDS:
            raise AssertionError("a trace guard word was overwritten")
        return code, (None if code else result.value), tuple(trace)[1:4], tuple(words)

    def compare(name, pointer, y, case, wanted):
        """The original must give wanted (None: an access violation after the first marker only); both
        translations must give what the original gives."""
        original = run(sides[0][1](name), pointer, y)
        if wanted is None:
            wanted = (ACCESS_VIOLATION, None, (1, MASK, MASK), (VALUE,) * len(words))
        if original != wanted:
            raise AssertionError(f"original {name}, {case}: code {original[0]:#x}, {original[1]}, trace {original[2]}")
        for label, function in sides[1:]:
            got = run(function(name), pointer, y)
            if got != original:
                raise AssertionError(f"{label} {name}, {case}: code {got[0]:#x}, {got[1]}, trace {got[2]}; "
                                     f"original code {original[0]:#x}, {original[1]}, trace {original[2]}")
        outcome = (f"raise {original[0]:#010x} after the first marker and before the second" if original[0]
                   else f"return {original[1]:#x} with trace {tuple(hex(word) for word in original[2])}")
        print(f"{name}, {case}: the original and both translations {outcome}")

    for name in READS:
        if name == "copy_bytes":
            continue
        # The second y makes the low 32 bits of VALUE + y wrap to zero, so add_in_place sets ZF.
        for y in (5, (1 << 32) - (VALUE & 0xffffffff)):
            compare(name, base, y, f"readable operand, y = {y:#x}", reference(name, y, 0))
        compare(name, 0, 5, "null operand", None)
        compare(name, base + PAGE, 5, "operand in an inaccessible page", None)
        # An operand that ends at the inaccessible page returns; one byte later it crosses into the page.
        width = READS[name][0]
        compare(name, base + PAGE - width, 5, f"{width}-byte operand ending at an inaccessible page",
                reference(name, 5, PAGE - width))
        if width > 1:
            compare(name, base + PAGE - width + 1, 5, f"{width}-byte operand crossing into an inaccessible page", None)


    # A zero count may use an inaccessible source. A partial copy makes the
    # first completed write visible before the next read faults.
    unchanged = (VALUE,) * len(words)
    low_byte = (MASK & ~0xff) | (VALUE >> 56)
    compare("copy_bytes", base + PAGE, 0, "REP MOVSB with zero count",
            (0, 0, (1, 2, MASK), unchanged))
    compare("copy_bytes", base + PAGE - 1, 1, "REP MOVSB last readable byte",
            (0, 1, (1, 2, low_byte), unchanged))
    compare("copy_bytes", base + PAGE - 1, 2, "REP MOVSB second iteration faults",
            (ACCESS_VIOLATION, None, (1, MASK, low_byte), unchanged))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "emit":
        emit(Path(sys.argv[2]), Path(sys.argv[3]))
    elif mode == "check":
        check_frame_read()
        check_source(sys.argv[3])
        check(*sys.argv[2:])
    else:
        raise ValueError(mode)
