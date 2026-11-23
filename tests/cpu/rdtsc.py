"""Compare native and translated RDTSC and RDTSCP by the properties a time-stamp read must keep. RDTSC is in
rdtsc.asm and RDTSCP in rdtscp.asm, so each fixture covers only the instruction its own test runs."""
import ctypes
from pathlib import Path
import re
import sys

import pefile
from iced_x86 import Code, Decoder, FlowControl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from cpu_leaf_aot import export_rva
from cpu_ranges import function_ranges
from ffx_version_query import INTEGER_HEADER, lower, region

U64 = ctypes.c_uint64
MASK32 = (1 << 32) - 1
ROUNDS = 2000

# part: {export: the time-stamp instructions it must contain}. CPUID and the serialized reads are in
# rdtsc_harness.asm.
PARTS = {
    "rdtsc": {"rdtsc_op": (Code.RDTSC,), "rdtsc_twice": (Code.RDTSC, Code.RDTSC), "rdtsc_unused": (Code.RDTSC,)},
    "rdtscp": {"rdtscp_op": (Code.RDTSCP,), "rdtscp_unused": (Code.RDTSCP,)},
}
TSC_CODES = (Code.RDTSC, Code.RDTSCP)

# Source mutations of the generated C++. Each control DLL must violate a property somewhere.
CONTROLS = {
    # A zero counter: the translated read no longer lies between two native reads.
    "constant": (re.compile(r"__rdtsc\(\)"), "0ull"),
    # The halves trade places.
    "swap": (re.compile(r"rax = static_cast<uint32_t>\(tsc\); rdx = tsc >> 32;"),
             "rax = tsc >> 32; rdx = static_cast<uint32_t>(tsc);"),
    # RAX keeps the whole counter instead of its zero-extended low half.
    "noext": (re.compile(r"rax = static_cast<uint32_t>\(tsc\);"), "rax = tsc;"),
    # RDTSCP drops TSC_AUX.
    "noaux": (re.compile(r"rcx = aux;"), "rcx = 0;"),
    # Each read keeps the value of its first call, so later calls fall outside the native bracket.
    "stale": (re.compile(r"const volatile uint64_t tsc = __rdtsc\(\);"), "static const uint64_t tsc = __rdtsc();"),
    # The reads clobber ZF and CF.
    "flags": (re.compile(r"rdx = tsc >> 32; \}"), "rdx = tsc >> 32; zero = !zero; carry = !carry; }"),
    # RDTSCP clobbers ZF and CF.
    "pflags": (re.compile(r"rcx = aux; \}"), "rcx = aux; zero = !zero; carry = !carry; }"),
    # Only built at /O2: without volatile the optimizer removes the read whose result the guest overwrites.
    "plain": (re.compile(r"const volatile uint64_t tsc"), "const uint64_t tsc"),
}
PRESETS = (0, 1, 1 << 63, 3 << 62)  # add r8, r8: (ZF, CF) = (1, 0), (0, 0), (1, 1), (0, 1)
MASK64 = (1 << 64) - 1


def flags_of(preset):
    total = preset + preset
    return int(total > MASK64) << 1 | int(total & MASK64 == 0)


def emit(part, image, output, control="lifted"):
    parts = [INTEGER_HEADER]
    with pefile.PE(str(image), fast_load=True) as pe:
        for name, codes in PARTS[part].items():
            (bounds,) = function_ranges(pe, export_rva(pe, name))
            body = region(pe, *bounds)
            found = tuple(i.code for i in body if i.code in TSC_CODES)
            if found != codes:
                raise AssertionError(f"{name} contains {found}, expected {codes}")
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


def pin(harness, native=None):
    """Keep every read on one logical processor, so counters and TSC_AUX are comparable. With native, prefer
    the highest processor of the affinity mask whose TSC_AUX is nonzero, so a translation that drops it shows.
    The thread must report that processor before anything is read there."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetCurrentThread.restype = ctypes.c_void_p
    kernel32.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                                ctypes.POINTER(ctypes.c_size_t)]
    kernel32.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    kernel32.SetThreadAffinityMask.restype = ctypes.c_size_t
    kernel32.GetCurrentProcessorNumber.restype = ctypes.c_uint32
    process, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel32.GetProcessAffinityMask(kernel32.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(system)):
        raise OSError(ctypes.get_last_error(), "GetProcessAffinityMask")
    cpus = [k for k in range(process.value.bit_length() - 1, -1, -1) if process.value >> k & 1]
    for cpu in cpus:
        if not kernel32.SetThreadAffinityMask(kernel32.GetCurrentThread(), 1 << cpu):
            raise OSError(ctypes.get_last_error(), "SetThreadAffinityMask")
        for _ in range(200):
            kernel32.Sleep(1)
            running = kernel32.GetCurrentProcessorNumber()
            if running == cpu:
                break
        else:
            raise AssertionError(f"pinned to logical processor {cpu}, still running on {running}")
        if native is None or call(native, "rdtscp_op")[2]:
            return cpu
    return cpu


def load_lib(path, part):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    for name in PARTS[part]:
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = [U64, U64, U64, ctypes.POINTER(U64)], U64
    return lib


def load_harness(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    lib.has_rdtscp.argtypes, lib.has_rdtscp.restype = [], ctypes.c_uint32
    lib.serial_tsc.argtypes, lib.serial_tsc.restype = [], U64
    return lib


def call(lib, name, preset=0):
    state = (U64 * 5)(*([0xdeadbeefdeadbeef] * 5))
    getattr(lib, name)(0, 0, preset, ctypes.cast(state, ctypes.POINTER(U64)))
    return list(state)


def counter(rax, rdx):
    return rdx << 32 | rax


def violations(harness, lib, part, aux=None):
    """The first violated property of lib's RDTSC or RDTSCP exports, or None. The bracket reads are
    LFENCE-serialized native reads, so neither can move across the translated read; aux is the TSC_AUX of the
    pinned processor."""
    advanced = False
    for round_ in range(ROUNDS):
        preset = PRESETS[round_ % len(PRESETS)]
        if part == "rdtscp":
            before = harness.serial_tsc()
            got = call(lib, "rdtscp_op", preset)
            after = harness.serial_tsc()
            if any(v > MASK32 for v in got[:3]):
                return f"rdtscp_op left upper bits: {[hex(v) for v in got[:3]]}"
            if not before <= counter(*got[:2]) <= after:
                return f"rdtscp_op {counter(*got[:2]):#x} outside the native bracket {before:#x}..{after:#x}"
            if got[2] != aux:
                return f"rdtscp_op ECX {got[2]:#x}, TSC_AUX of the pinned processor {aux:#x}"
            if got[3] != flags_of(preset):
                return f"rdtscp_op changed ZF/CF: preset {preset:#x} gave {got[3]}, expected {flags_of(preset)}"
            unused = lib.rdtscp_unused(0, 0, preset, None)
            if unused != 7:
                return f"rdtscp_unused returned {unused:#x}, expected 7"
            continue
        before = harness.serial_tsc()
        got = call(lib, "rdtsc_op", preset)
        after = harness.serial_tsc()
        if got[0] > MASK32 or got[1] > MASK32:
            return f"rdtsc_op left upper bits: rax {got[0]:#x} rdx {got[1]:#x}"
        if not before <= counter(*got[:2]) <= after:
            return f"rdtsc_op {counter(*got[:2]):#x} outside the native bracket {before:#x}..{after:#x}"
        if got[3] != flags_of(preset):
            return f"rdtsc_op changed ZF/CF: preset {preset:#x} gave {got[3]}, expected {flags_of(preset)}"
        twice = call(lib, "rdtsc_twice", preset)
        if any(v > MASK32 for v in twice[:4]):
            return f"rdtsc_twice left upper bits: {[hex(v) for v in twice[:4]]}"
        first, second = counter(*twice[:2]), counter(*twice[2:4])
        if second < first:
            return f"rdtsc_twice went backwards: {first:#x} then {second:#x}"
        advanced |= second != first
        if twice[4] != flags_of(preset):
            return f"rdtsc_twice changed ZF/CF: preset {preset:#x} gave {twice[4]}, expected {flags_of(preset)}"
        unused = lib.rdtsc_unused(0, 0, preset, None)
        if unused != 7:
            return f"rdtsc_unused returned {unused:#x}, expected 7"
    if part == "rdtsc" and not advanced:
        return f"rdtsc_twice returned the same value for both reads in all {ROUNDS} rounds"
    return None


def tsc_reads(path, part):
    """The time-stamp instructions reachable in each export of a DLL, sorted by address. The walk follows every
    branch from the export to each RET inside the image; an optimized translation may keep no unwind record
    for a leaf, so this does not use function_ranges."""
    with pefile.PE(str(path), fast_load=True) as pe:
        base, image = pe.OPTIONAL_HEADER.ImageBase, pe.get_memory_mapped_image()
        reads = {}
        for name in PARTS[part]:
            pending, seen, found = [base + export_rva(pe, name)], set(), {}
            while pending:
                ip = pending.pop()
                while ip not in seen:
                    seen.add(ip)
                    if not base <= ip < base + len(image):
                        raise AssertionError(f"{Path(path).name} {name}: branch leaves the image at {ip:#x}")
                    instruction = Decoder(64, image[ip - base:ip - base + 15], ip=ip).decode()
                    if instruction.code in TSC_CODES:
                        found[ip] = instruction.code
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
            reads[name] = tuple(found[ip] for ip in sorted(found))
    return reads


def check_optimized(harness, path, part, aux):
    """An /O2 build: the translation keeps every read, the plain control loses the overwritten RDTSC."""
    name, reads = Path(path).name, tsc_reads(path, part)
    if Path(path).stem.endswith("_plain_o2"):
        if reads["rdtsc_unused"]:
            raise AssertionError(f"{name}: the overwritten RDTSC survived without volatile: {reads}")
        print(f"{name}: without volatile, /O2 drops the overwritten RDTSC and the walk reports it: {reads}")
        return
    if {k: sorted(v) for k, v in reads.items()} != {k: sorted(v) for k, v in PARTS[part].items()}:
        raise AssertionError(f"{name}: time-stamp reads {reads}, expected {PARTS[part]}")
    problem = violations(harness, load_lib(path, part), part, aux)
    if problem:
        raise AssertionError(f"{name}: {problem}")
    print(f"{name}: every {part.upper()} is still in the /O2 code, including the overwritten one, "
          f"and the properties hold")


def check(part, harness_path, native_path, lifted_path, *control_paths):
    """part rdtsc runs on every host. part rdtscp exits 77 on a host without RDTSCP, or after the other checks
    when TSC_AUX is 0 on every processor, where the noaux control cannot fail."""
    harness, native = load_harness(harness_path), load_lib(native_path, part)
    aux = None
    if part == "rdtscp":
        if not harness.has_rdtscp():
            print("skip: this host has no RDTSCP (CPUID 80000001h EDX bit 27)")
            raise SystemExit(77)
        cpu = pin(harness, native)
        aux = call(native, "rdtscp_op")[2]
    elif part == "rdtsc":
        cpu = pin(harness)
    else:
        raise ValueError(part)
    for label, lib in (("native", native), ("lifted", load_lib(lifted_path, part))):
        problem = violations(harness, lib, part, aux)
        if problem:
            raise AssertionError(f"{label}: {problem}")
    extra = f", returned TSC_AUX {aux:#x}" if part == "rdtscp" else ", never went backwards and advanced"
    print(f"{ROUNDS} rounds on logical processor {cpu}: translated {part.upper()} lay between two serialized "
          f"native reads, zero-extended its results, kept ZF/CF{extra}")
    skipped = []
    for path in control_paths:
        if Path(path).stem.endswith("_o2"):
            check_optimized(harness, path, part, aux)
            continue
        control = Path(path).stem.rsplit("_", 1)[-1]
        if control == "noaux" and aux == 0:
            skipped.append("control noaux is not distinguishable: TSC_AUX is 0 on every processor")
            continue
        problem = violations(harness, load_lib(path, part), part, aux)
        print(f"control {control}: {problem}")
        if problem is None:
            raise AssertionError(f"control {control} kept every {part.upper()} property")
    if skipped:
        print("skip: " + "; ".join(skipped))
        raise SystemExit(77)


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    if mode == "emit":
        part, image, output, *control = args
        emit(part, Path(image), Path(output), *control)
    elif mode == "check":
        check(*args)
    else:
        raise ValueError(mode)
