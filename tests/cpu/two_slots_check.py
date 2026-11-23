"""Differential checks and rejection controls for two ordinary imports from two providers in one translation.

Four callers are built from two_slots.asm: one linked against the named mix3 provider of cpu_aot_import_tail_jump
(tail_provider.cpp) and the named fold4 provider (slot_provider.cpp), one with the altered mix3 provider, one with
the altered fold4 provider, and one that imports mix3 by ordinal #3 from that test's ordinal provider, bound with the
link name mix3; an altered provider returns one more. Each caller is translated once from the
command line with two --bind groups, mix3 with three arguments and 8 guaranteed bytes and fold4 with four and 2, as
listed in CMakeLists.txt. The altered callers show that each import site reaches the provider its slot names; the
contract checks show that each binding keeps its own argument count and return width, including at the tail jump;
the command-line checks show that the per-binding options cannot be applied to the wrong binding silently."""

import ctypes
from pathlib import Path
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

import pefile
from iced_x86 import Decoder, FlowControl, Mnemonic, OpKind, Register

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
from cpu_switch_aot import Binding, translate
from direct_check import GUARDS, MASK, SENTINELS, input_pairs, observe, register_probe

VARIANTS = ("named", "first_altered", "second_altered", "ordinal")
# The ordinal caller imports mix3 as #3 from cpu_tail_ordinal_provider; its binding names the link name mix3.
MIX3 = {"ordinal": ("#3", "mix3")}
ENTRIES = ("route2", "call_both", "call_second")
# Import calls ("rax = ...") and tail jumps ("return ...") of each import in the translation, and its arguments.
SITES = {"mix3": (2, 0, "rcx, rdx, r8"), "fold4": (3, 1, "rcx, rdx, r8, r9")}
# route2's tail jump through the slot of fold4 makes its return width the 2 bytes that binding states.
WIDTHS = [2, 8, 8]
UNTOUCHED = (MASK, MASK, MASK)


def mix3(first, second, altered):
    first, second = first & MASK, second & MASK
    stored = (first - second + 11) & MASK
    return (stored * 3 + (first >> 5) + altered) & MASK, (first, second, stored)


def fold4(first, second, fourth, altered):
    first, second, fourth = first & MASK, second & MASK, fourth & MASK
    return (((first ^ second) + 0x2d) * 5 + (fourth >> 7) + altered) & 0xffff, (first, second, fourth)


def reference(name, x, y, first, second):
    """RAX and the guarded trace after name(x, y, trace); first and second are 1 for an altered provider."""
    words = UNTOUCHED
    if name == "route2" and x & 3 == 0:
        value, words = mix3(x, y, first)
        rax = (value + 1) ^ x
    elif name == "route2" and x & 3 == 1:
        rax, words = fold4(x, y + 1, x + y + 1, second)
    elif name == "route2" and x & 3 == 2:
        value, words = fold4(x, y, x ^ y, second)
        rax = value + x
    elif name == "route2":
        rax = x + 5
    elif name == "call_both":
        value, _ = mix3(x, y, first)
        value, words = fold4(value, y, y + 1, second)
        rax = value + y
    else:  # call_second
        rax, words = fold4(x, y, y - x, second)
    return rax & MASK, (GUARDS[0], *words, GUARDS[1])


def linear(pe, rva):
    """The instructions from rva up to the first RET, read in address order across jumps."""
    out = []
    for ins in Decoder(64, pe.get_data(rva, 0x100), ip=rva):
        out.append(ins)
        if ins.flow_control == FlowControl.RETURN:
            return out
    raise AssertionError(f"no RET at RVA {rva:#x}")


def layout(caller, providers, mix3_name="mix3"):
    """IAT slots and import sites, after checking each by its encoding: route2 calls FF 15 through the slot of mix3,
    jumps FF 25 through the slot of fold4 and calls E8 a thunk FF 25 through the same slot; call_both calls FF 15
    through both slots in that order; call_second through the slot of fold4."""
    with pefile.PE(str(caller), fast_load=True) as pe:
        pe.parse_data_directories(directories=[0, 1])
        slots = {}
        for desc in pe.DIRECTORY_ENTRY_IMPORT:
            dll = desc.dll.decode("ascii").casefold()
            for n, imp in enumerate(desc.imports):
                name = f"#{imp.ordinal}" if imp.import_by_ordinal else imp.name.decode("ascii")
                slots[dll, name] = desc.struct.FirstThunk + 8 * n
        wanted = {(providers[0].name.casefold(), mix3_name), (providers[1].name.casefold(), "fold4")}
        if set(slots) != wanted:
            raise AssertionError(f"{caller.name}: unexpected import table {slots}")
        slot3, slot4 = slots[providers[0].name.casefold(), mix3_name], slots[providers[1].name.casefold(), "fold4"]
        exports = {e.name.decode("ascii"): e.address for e in pe.DIRECTORY_ENTRY_EXPORT.symbols}
        body = {name: linear(pe, exports[name]) for name in ENTRIES}

        def raw(ins):
            return pe.get_data(ins.ip, ins.len)

        def slot(ins):
            return ins.next_ip + struct.unpack_from("<i", raw(ins), 2)[0]

        def through(name, opcode):
            return [slot(i) for i in body[name] if raw(i)[:2] == opcode and i.len == 6]

        thunk_call, = [i for i in body["route2"] if i.flow_control == FlowControl.CALL]
        thunk = Decoder(64, pe.get_data(thunk_call.near_branch_target, 6), ip=thunk_call.near_branch_target).decode()
        if (through("route2", b"\xff\x15") != [slot3] or through("route2", b"\xff\x25") != [slot4]
                or raw(thunk)[:2] != b"\xff\x25" or thunk.len != 6 or slot(thunk) != slot4
                or thunk.ip in exports.values()):
            raise AssertionError(f"{caller.name}: route2 does not reach mix3 and fold4 through their slots")
        if through("call_both", b"\xff\x15") != [slot3, slot4] or through("call_second", b"\xff\x15") != [slot4]:
            raise AssertionError(f"{caller.name}: call_both or call_second does not call through the expected slots")
        sites = {
            "route2 add": next(i.ip for i in body["route2"] if i.mnemonic == Mnemonic.ADD
                               and i.op0_register == Register.RAX and i.op1_kind != OpKind.REGISTER),
            "route2 movzx": next(i.ip for i in body["route2"] if i.mnemonic == Mnemonic.MOVZX),
            "route2 mix3": [i.ip for i in body["route2"] if raw(i)[:2] == b"\xff\x15"][0],
            "call_both mix3": [i.ip for i in body["call_both"] if raw(i)[:2] == b"\xff\x15"][0],
        }
    print(f"{caller.name}: route2 calls FF 15 through the slot of mix3 at RVA {slot3:#x}, jumps FF 25 through the "
          f"slot of fold4 at RVA {slot4:#x} and calls (E8) its thunk at RVA {thunk.ip:#x}; call_both calls through "
          "both slots in that order; call_second through the slot of fold4")
    return {"mix3": slot3, "fold4": slot4, **sites}


def bindings(providers, mix3_arguments=3, mix3_bytes=8, fold4_arguments=4, fold4_bytes=2, variant="named"):
    symbol, link_name = MIX3.get(variant, ("mix3", None))
    return (Binding(providers[0], symbol, link_name, mix3_arguments, mix3_bytes),
            Binding(providers[1], "fold4", None, fold4_arguments, fold4_bytes))


def check_source(variant, caller, built, providers):
    """The built source is the translation with both bindings: each import declared once with its own number of
    arguments, each call and tail jump passing them, route2's return width the 2 bytes the fold4 binding states.
    The ordinal caller's mix3 is declared and called by its link name."""
    source, functions = translate(caller, list(ENTRIES), binding=bindings(providers, variant=variant))
    if Path(built).read_text(encoding="utf-8") != source:
        raise AssertionError(f"{built} is not the translation of {','.join(ENTRIES)} with both bindings")
    lines = [line.strip() for line in source.splitlines()]
    for symbol, (calls, tails, passed) in SITES.items():
        counts = (lines.count(f"rax = {symbol}({passed});"), lines.count(f"return {symbol}({passed});"))
        params = ", ".join(["word"] * (passed.count(",") + 1))
        declaration = f'extern "C" __declspec(dllimport) word {symbol}({params});'
        if lines.count(declaration) != 1 or counts != (calls, tails):
            raise AssertionError(f"{variant} {symbol}: {counts} calls and tail jumps:\n{source}")
        print(f"{variant}: {symbol} declared once with {params.count('word')} arguments, {calls} x rax = "
              f"{symbol}({passed}); {tails} x return {symbol}({passed});")
    widths = [function.return_width for function in functions]
    if widths != WIDTHS:
        raise AssertionError(f"{variant}: return widths {widths}, expected {WIDTHS}")
    print(f"{variant}: return widths {widths} for {', '.join(ENTRIES)}")
    return source


def check_contracts(caller, providers):
    """Each binding keeps its own return width: the tail jump through the slot of fold4 gives route2 the width the
    fold4 binding states, whatever mix3 states."""
    # mix3's own width is shown by the rejection control that states 2 bytes for it where route2 reads all of RAX.
    for fold4_bytes in (8, 2):
        _, (function,) = translate(caller, ["route2"], binding=bindings(providers, fold4_bytes=fold4_bytes))
        if function.return_width != fold4_bytes:
            raise AssertionError(f"route2 with {fold4_bytes} bytes stated for fold4: width {function.return_width}")
        print(f"route2 with 8 bytes stated for mix3 and {fold4_bytes} for fold4: return width {fold4_bytes}, from "
              "the tail jump through the slot of fold4")


def check_execution(probe_path, builds):
    probe = register_probe(probe_path)
    if observe(probe, ctypes.CDLL(probe_path).clobber_control, 0, 0)[2] != (0, 0):
        raise AssertionError("physical register probe missed the clobber control")
    print("control clobber_control through probe: RBX and RSI sampled as 0 and 0")
    inputs = input_pairs()
    if {x & 3 for x, _ in inputs} != {0, 1, 2, 3}:
        raise AssertionError("the inputs do not reach every case of route2")
    results = {}
    for variant, (caller, lifted_path, _, _, _) in builds.items():
        first, second = int(variant == "first_altered"), int(variant == "second_altered")
        native, lifted = ctypes.CDLL(str(caller)), ctypes.CDLL(str(lifted_path))
        for name in ENTRIES:
            source = ctypes.cast(getattr(native, name), ctypes.c_void_p).value
            translated = ctypes.cast(getattr(lifted, f"lifted_{name}"), ctypes.c_void_p).value
            rows = []
            for x, y in inputs:
                want, trace = reference(name, x, y, first, second)
                original, actual = observe(probe, source, x, y), observe(probe, translated, x, y)
                if original != (want, trace, SENTINELS):
                    raise AssertionError(f"{variant} native {name}({x:#x}, {y:#x}): {original}")
                if actual != (want, trace, SENTINELS):
                    raise AssertionError(f"{variant} lifted_{name}({x:#x}, {y:#x}): {actual}")
                rows.append(actual[0])
            results[variant, name] = rows
            print(f"{variant} {name}: {len(rows)} calls match the reference natively and translated on the whole "
                  "return value, guarded trace (with fold4's fourth argument), RBX and RSI")
    # Each altered provider changes exactly the paths that reach its slot.
    expected = {("first_altered", "route2"): {0}, ("second_altered", "route2"): {1, 2},
                ("first_altered", "call_second"): set(), ("second_altered", "call_second"): {0, 1, 2, 3}}
    for (variant, name), cases in expected.items():
        changed = {x & 3 for (x, _), a, b in zip(inputs, results["named", name], results[variant, name]) if a != b}
        if changed != cases:
            raise AssertionError(f"{variant} {name}: returns differ for cases {sorted(changed)}, expected {sorted(cases)}")
        print(f"{variant} {name}: translated returns differ from the named build for x & 3 in {sorted(changed)}")
    for variant in ("first_altered", "second_altered"):
        changed = sum(a != b for a, b in zip(results["named", "call_both"], results[variant, "call_both"]))
        if not changed:
            raise AssertionError(f"{variant} call_both: translated returns never differ")
        print(f"{variant} call_both: translated returns differ on {changed} of {len(inputs)} calls")
    if any(results["ordinal", name] != results["named", name] for name in ENTRIES):
        raise AssertionError("the ordinal caller's translated returns differ from the named caller's")
    print("ordinal: translated returns equal the named caller's on every path")


def check_rejections(builds, sites):
    """Bindings the translator must reject for the named caller, and for the caller that imports mix3 by ordinal."""
    caller, _, _, first, second = builds["named"]
    ordinal_caller, _, _, ordinal_provider, _ = builds["ordinal"]
    one, two, three = first.name, second.name, ordinal_provider.name
    providers = (first, second)
    controls = [
        ("#3 bound without its link name", "route2", ordinal_caller,
         (Binding(ordinal_provider, "#3", None, 3, 8), Binding(second, "fold4", None, 4, 2)),
         f"import-library link name required for {three}!#3"),
        # The link name of #3 is the C name of the named fold4: both bindings would declare fold4.
        ("#3 bound with the link name fold4", "route2", ordinal_caller,
         (Binding(ordinal_provider, "#3", "fold4", 3, 8), Binding(second, "fold4", None, 4, 2)),
         "both declare the C name fold4"),
    ]
    for label, name, image, bound, reason in controls:
        reject(image, name, bound, label, reason)
    controls = [
        ("RIP-relative jump outside the IAT", "jump_non_import", bindings(providers),
         "is not an ordinary IAT slot"),
        ("only the mix3 binding", "route2", Binding(first, "mix3", None, 3, 8),
         f"call or tail jump through a second IAT slot RVA {sites['fold4']:#x}"),
        ("two bindings of one slot", "route2", (Binding(first, "mix3", None, 3, 8), Binding(first, "mix3", None, 3, 1)),
         f"two bindings of {one}!mix3"),
        ("fold4 bound to the mix3 provider", "route2",
         (Binding(first, "mix3", None, 3, 8), Binding(first, "fold4", None, 4, 2)),
         f"binding {one}!fold4 names no ordinary import of the image"),
        ("fold4 bound as fold3", "route2", (Binding(first, "mix3", None, 3, 8), Binding(second, "fold3", None, 4, 2)),
         f"binding {two}!fold3 names no ordinary import of the image"),
        ("the mix3 binding unused", "call_second", bindings(providers),
         f"binding {one}!mix3 supplied but no call or tail jump uses it"),
        ("full RAX read after the call of mix3, 2 bytes stated for mix3", "route2", bindings(providers, mix3_bytes=2),
         f"undefined register rax at RVA {sites['route2 add']:#x}"),
        ("AX read after the thunk call, 1 byte stated for fold4", "route2", bindings(providers, fold4_bytes=1),
         f"undefined register rax at RVA {sites['route2 movzx']:#x}"),
        ("mix3 bound with four arguments, R9 undefined at its call", "call_both", bindings(providers, mix3_arguments=4),
         f"callee argument r9 is undefined at RVA {sites['call_both mix3']:#x}"),
    ]
    for label, name, bound, reason in controls:
        reject(caller, name, bound, label, reason)
    print(f"{len(controls) + 2} rejection controls passed")


def reject(image, name, bound, label, reason):
    try:
        translate(image, [name], binding=bound)
    except ValueError as exc:
        if reason not in str(exc):
            raise AssertionError(f"{name} ({label}): expected {reason!r}, got {exc}") from exc
        print(f"rejected {name} ({label}): {exc}")
    else:
        raise AssertionError(f"accepted control {name} ({label})")


def check_command_line(builds, sites, built_source):
    """The per-binding options follow the --bind before them. Reversing the two groups gives the same source;
    options swapped between the groups, before the first group, given twice or missing are refused; a single --bind
    still takes its options anywhere."""
    caller, _, _, first, second = builds["named"]
    mix3_group = ["--bind", str(first), "mix3", "--import-arguments", "3", "--return-bytes", "8"]
    fold4_group = ["--bind", str(second), "fold4", "--import-arguments", "4", "--return-bytes", "2"]
    entries = ",".join(ENTRIES)
    with TemporaryDirectory() as folder:
        output = Path(folder) / "lifted.cpp"

        def run(names, options):
            output.unlink(missing_ok=True)
            done = subprocess.run([sys.executable, "-B", str(TOOLS / "cpu_switch_aot.py"), str(caller), names,
                                   str(output), *options], capture_output=True, text=True)
            return done.returncode, (done.stdout + done.stderr).strip().splitlines()[-1:]

        code, _ = run(entries, fold4_group + mix3_group)
        if code or output.read_text(encoding="utf-8") != built_source:
            raise AssertionError("the --bind groups in reverse order did not give the same translation")
        print("reversed --bind groups (fold4 first): the same translation as the CMake build")
        source, _ = translate(caller, ["call_second"], binding=Binding(second, "fold4", None, 4, 2))
        code, _ = run("call_second", ["--return-bytes", "2", "--import-arguments", "4", "--bind", str(second), "fold4"])
        if code or output.read_text(encoding="utf-8") != source:
            raise AssertionError("a single --bind after its options did not translate as before")
        print("a single --bind after its options: translated as before")
        swapped = ["--bind", str(first), "mix3", "--import-arguments", "4", "--return-bytes", "2",
                   "--bind", str(second), "fold4", "--import-arguments", "3", "--return-bytes", "8"]
        # Swapped, mix3 takes four arguments, and R9 is not an argument at its call in route2.
        controls = [
            ("options swapped between the groups", swapped, 1,
             f"callee argument r9 is undefined at RVA {sites['route2 mix3']:#x}"),
            ("an option before the first of two groups", ["--return-bytes", "8", *mix3_group[:5], *fold4_group], 2,
             "--return-bytes comes before the first of several --bind options"),
            ("an option twice in one group", [*mix3_group, "--import-arguments", "4", *fold4_group], 2,
             f"--import-arguments given twice for --bind {first} mix3"),
            ("the second group without --return-bytes", [*mix3_group, *fold4_group[:5]], 2,
             "--bind requires --import-arguments and --return-bytes"),
        ]
        for label, options, wanted, reason in controls:
            code, last = run(entries, options)
            if code != wanted or not last or reason not in last[0] or output.exists():
                raise AssertionError(f"command line ({label}): exit {code}, {last}")
            print(f"refused on the command line ({label}): {last[0]}")
    print(f"{len(controls)} command-line controls passed")


if __name__ == "__main__":
    probe_path, *paths = sys.argv[1:]
    paths = [Path(path) for path in paths]
    # Per variant (named, mix3 altered, fold4 altered, mix3 by ordinal): caller, translated DLL and its source,
    # then the mix3 and the fold4 provider.
    builds = {variant: paths[5 * n:5 * n + 5] for n, variant in enumerate(VARIANTS)}
    layouts, sources = {}, {}
    for variant, (caller, _, source, first, second) in builds.items():
        layouts[variant] = layout(caller, (first, second), MIX3.get(variant, ("mix3",))[0])
        sources[variant] = check_source(variant, caller, source, (first, second))
    check_contracts(builds["named"][0], builds["named"][3:5])
    check_execution(probe_path, builds)
    check_rejections(builds, layouts["named"])
    check_command_line(builds, layouts["named"], sources["named"])
