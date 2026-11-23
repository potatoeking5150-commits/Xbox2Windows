"""Exercise a source-built SPIR-V subgroup compaction dispatch on a GPU against an independent CPU
reference, on both the SPIR-V-to-HLSL bridge and a direct desktop-HLSL control.

Usage: compaction.py <spirv-as> <spirv-val> <spirv-cross> <dxc> <offscreen_check> <hardware|warp>

The fixture is one 64-thread group. It computes a sparse per-lane predicate from a deterministic
64-bit mask, ballots it, derives the first/last active lane and each lane's compaction rank, and
writes every intermediate to its own word. Compacted payloads go to per-wave disjoint ranges keyed
by rank, so the readback does not depend on workgroup or wave scheduling order. The CPU reference is
recomputed with the lane width the shader actually observed, because the width is adapter-dependent
(for example WARP reports four-lane waves where an RTX 4060 reports 32).

Negative controls mutate the predicate bit selection (a wrong ballot bit) and the compaction write
index by one (a wrong write index). The first divergent word is predicted from the fixture layout
and checked on the GPU; a control that does not diverge is a test failure.
"""
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).with_name("compaction.spvasm")
DIRECT_HLSL = Path(__file__).with_name("compaction.hlsl")
SKIP = 77

THREADS = 64
WORD_COUNT = 328
SENTINEL = 0xC5C5C5C5
LANE = 0
WIDTH = 64
BALLOT = 65
BALLOT_SLOTS = 64
FIRST = 129
COUNT = 145
LAST = 161
SELECTED = 177
RANK = 193
COMPACT = 257
GUARD = 321
SUPPORTED_WIDTHS = (4, 8, 16, 32, 64)

# Two deterministic input masks. Every 4-bit nibble of both masks has exactly one bit set, so every
# wave at every supported width has at least one active lane. mask_a sets bit 31 (last lane of a
# 32-lane wave) and clears bit 32; mask_b does the reverse, so the 31/32 ballot-word boundary is
# exercised on both sides of a wave split.
MASKS = {
    "mask_a": (0x84218421, 0x18421842),
    "mask_b": (0x28421842, 0x84218421),
}

# Negative controls: (mutation applied to the fixture constants, predicted first divergent word).
# Flipping the predicate bit selection changes wave 0's ballot word first. Shifting the compaction
# write index by one leaves the first compacted slot untouched first.
CONTROLS = {
    "ballot_bit": ({"pred_flip": 1, "write_bias": 0}, BALLOT),
    "write_index": ({"pred_flip": 0, "write_bias": 1}, COMPACT),
}

PLACEHOLDERS = {
    "mask_lo": ("%mask_lo = OpConstant %uint 0x00000000 ; X2W_MASK_LO",
                "%mask_lo = OpConstant %uint 0x{value:08x} ; X2W_MASK_LO"),
    "mask_hi": ("%mask_hi = OpConstant %uint 0x00000000 ; X2W_MASK_HI",
                "%mask_hi = OpConstant %uint 0x{value:08x} ; X2W_MASK_HI"),
    "pred_flip": ("%pred_flip = OpConstant %uint 0 ; X2W_PRED_FLIP",
                  "%pred_flip = OpConstant %uint {value} ; X2W_PRED_FLIP"),
    "write_bias": ("%write_bias = OpConstant %uint 0 ; X2W_WRITE_BIAS",
                   "%write_bias = OpConstant %uint {value} ; X2W_WRITE_BIAS"),
}


def substitute(assembly, mask, pred_flip, write_bias):
    values = {"mask_lo": mask[0], "mask_hi": mask[1],
              "pred_flip": pred_flip, "write_bias": write_bias}
    result = assembly
    for key, (marker, template) in PLACEHOLDERS.items():
        if marker not in result:
            raise RuntimeError(f"fixture marker missing: {marker}")
        result = result.replace(marker, template.format(value=values[key]))
    return result


def emit_bridge(directory, tools, mask, pred_flip=0, write_bias=0):
    assembler, validator, cross, dxc = tools
    stem = directory / f"compaction-{mask[0]:08x}-{pred_flip}-{write_bias}"
    assembly = substitute(FIXTURE.read_text(encoding="utf-8"), mask, pred_flip, write_bias)
    source = stem.with_suffix(".spvasm")
    source.write_text(assembly, encoding="utf-8")
    module = stem.with_suffix(".spv")
    subprocess.run([assembler, "--target-env", "vulkan1.2", source, "-o", module], check=True)
    subprocess.run([validator, "--target-env", "vulkan1.2", module], check=True)
    hlsl = stem.with_suffix(".hlsl")
    subprocess.run([cross, module, "--hlsl", "--shader-model", "60", "--output", hlsl], check=True)
    # spirv-cross emits no root signature; the fixture contract is a single u0 byte-address UAV.
    text = hlsl.read_text(encoding="utf-8")
    marker = "void main(SPIRV_Cross_Input stage_input)"
    if marker not in text:
        raise RuntimeError("unexpected spirv-cross entry point")
    hlsl.write_text(text.replace(marker, '[RootSignature("UAV(u0)")]\n' + marker), encoding="utf-8")
    compiled = stem.with_suffix(".dxil")
    subprocess.run([dxc, "-T", "cs_6_0", "-E", "main", "-Fo", compiled, hlsl], check=True)
    return compiled


def emit_direct(directory, dxc, mask):
    lo, hi = mask
    compiled = directory / f"compaction-direct-{lo:08x}.dxil"
    subprocess.run([dxc, "-T", "cs_6_0", "-E", "main",
                    f"-DMASK_LO=0x{lo:08x}u", f"-DMASK_HI=0x{hi:08x}u",
                    "-Fo", compiled, DIRECT_HLSL], check=True)
    return compiled


def run(offscreen_check, adapter, dxil):
    result = subprocess.run([offscreen_check, adapter, "--spirv-compaction", dxil],
                            capture_output=True, text=True)
    lines = result.stdout.splitlines()
    for line in lines:
        if line.startswith(("adapter=", "driver=", "backend=", "debug_errors=", "skip:")):
            print(line)
    if result.returncode == SKIP:
        return SKIP, None
    if result.returncode != 0:
        raise RuntimeError(f"offscreen_check failed for {dxil.name}: {result.stderr.strip()}")
    if f"backend={adapter} debug_layer=enabled" in lines and "debug_errors=0 debug_warnings=0" not in lines:
        raise RuntimeError(f"D3D12 debug diagnostics for {dxil.name}: {result.stderr.strip()}")
    observed = {}
    for line in lines:
        match = re.fullmatch(r"observed\[(\d+)\]=([0-9a-f]{8})", line)
        if match:
            observed[int(match.group(1))] = int(match.group(2), 16)
    if sorted(observed) != list(range(WORD_COUNT)):
        raise RuntimeError(f"incomplete readback from {dxil.name}: {len(observed)} words")
    return 0, observed


def predicate_at(local, mask, pred_flip=0):
    lo, hi = mask
    index = local ^ pred_flip
    word = hi if (index >> 5) else lo
    return (word >> (index & 31)) & 1


def reference(mask, width, pred_flip=0, write_bias=0):
    """Independent CPU reference for the whole fixture, computed at the observed lane width."""
    words = [SENTINEL] * WORD_COUNT
    active = [predicate_at(local, mask, pred_flip) for local in range(THREADS)]
    for wave in range(THREADS // width):
        lanes = active[wave * width:(wave + 1) * width]
        ballot = [0, 0, 0, 0]
        for lane, bit in enumerate(lanes):
            if bit:
                ballot[lane >> 5] |= 1 << (lane & 31)
        selected = [lane for lane, bit in enumerate(lanes) if bit]
        if not selected:
            raise RuntimeError("fixture mask left a wave with no active lane")
        rank = [sum(lanes[:lane]) for lane in range(width)]
        base = wave * width
        for local in range(base, base + width):
            lane = local - base
            words[LANE + local] = lane
            words[RANK + local] = rank[lane]
            if lanes[lane]:
                destination = COMPACT + wave * width + rank[lane] + write_bias
                if destination >= WORD_COUNT:
                    raise RuntimeError("compacted write left the fixture's output range")
                words[destination] = 0x80000000 | local
        words[WIDTH] = width
        words[BALLOT + wave * 4 + 0] = ballot[0]
        words[BALLOT + wave * 4 + 1] = ballot[1]
        words[BALLOT + wave * 4 + 2] = ballot[2]
        words[BALLOT + wave * 4 + 3] = ballot[3]
        words[FIRST + wave] = selected[0]
        words[COUNT + wave] = len(selected)
        words[LAST + wave] = selected[-1]
        words[SELECTED + wave] = len(selected) - 1
    return words


def field(index):
    if LANE <= index < LANE + THREADS:
        return "lane"
    if index == WIDTH:
        return "width"
    if BALLOT <= index < BALLOT + BALLOT_SLOTS:
        return "ballot"
    if FIRST <= index < FIRST + 16:
        return "first"
    if COUNT <= index < COUNT + 16:
        return "count"
    if LAST <= index < LAST + 16:
        return "last"
    if SELECTED <= index < SELECTED + 16:
        return "selected_rank"
    if RANK <= index < RANK + THREADS:
        return "rank"
    if COMPACT <= index < COMPACT + THREADS:
        return "compact"
    return "guard"


def mismatches(observed, expected):
    return [index for index in range(WORD_COUNT) if observed[index] != expected[index]]


def report_mismatches(observed, expected, problems):
    first = problems[0]
    print(f"first_mismatch=word={first} field={field(first)} "
          f"cpu={expected[first]:08x} gpu={observed[first]:08x}")
    for extra in problems[:8]:
        print(f"mismatch word={extra} field={field(extra)} "
              f"cpu={expected[extra]:08x} gpu={observed[extra]:08x}")
    return first


def check_source(label, adapter, mask_name, observed, width, mask):
    expected = reference(mask, width)
    problems = mismatches(observed, expected)
    if problems:
        report_mismatches(observed, expected, problems)
    print(f"source={label} adapter={adapter} mask={mask_name} width={width} "
          f"waves={THREADS // width} compared={WORD_COUNT} mismatches={len(problems)}")
    if problems:
        raise RuntimeError(f"{label} readback differs from the CPU reference at word {problems[0]}")


def run_control(adapter, observed, mask, width, name, mutation, predicted):
    mutated = reference(mask, width, **mutation)
    unexpected = mismatches(observed, mutated)
    if unexpected:
        first = report_mismatches(observed, mutated, unexpected)
        raise RuntimeError(f"negative control {name} differs from its mutated reference at word {first}")

    expected = reference(mask, width)
    problems = mismatches(observed, expected)
    if not problems:
        raise RuntimeError(f"negative control {name} produced the reference readback")
    first = report_mismatches(observed, expected, problems)
    print(f"negative_control={name} adapter={adapter} first_mismatch={first} "
          f"predicted={predicted} mismatches={len(problems)} mutated_reference=exact status=caught")
    if first != predicted:
        raise RuntimeError(f"negative control {name}: first mismatch word {first}, predicted {predicted}")


def main(argv):
    if len(argv) != 6:
        sys.exit(__doc__)
    assembler, validator, cross, dxc, offscreen_check, adapter = argv
    tools = (assembler, validator, cross, dxc)
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        for mask_name, mask in MASKS.items():
            bridge = emit_bridge(directory, tools, mask)
            direct = emit_direct(directory, dxc, mask)
            results = {}
            for label, dxil in (("bridge", bridge), ("direct", direct)):
                code, observed = run(offscreen_check, adapter, dxil)
                if code == SKIP:
                    print(f"skip: {adapter} has no capable D3D12 adapter")
                    return SKIP
                width = observed[WIDTH]
                if width == 128:
                    print(f"skip: {adapter} subgroup width {width} exceeds the {THREADS}-thread fixture")
                    return SKIP
                if width not in SUPPORTED_WIDTHS:
                    raise RuntimeError(f"{label} reported an invalid subgroup width: {width}")
                check_source(label, adapter, mask_name, observed, width, mask)
                code, repeat = run(offscreen_check, adapter, dxil)
                if repeat != observed:
                    raise RuntimeError(f"{label} readback changed between identical dispatches")
                print(f"source={label} adapter={adapter} mask={mask_name} repeat=identical")
                results[label] = observed
            if results["bridge"] != results["direct"]:
                first = next(index for index in range(WORD_COUNT)
                             if results["bridge"][index] != results["direct"][index])
                raise RuntimeError(f"bridge and direct HLSL differ at word {first} ({field(first)})")
            print(f"adapter={adapter} mask={mask_name} bridge_equals_direct=true")
            for control_name, (mutation, predicted) in CONTROLS.items():
                control = emit_bridge(directory, tools, mask, **mutation)
                code, observed = run(offscreen_check, adapter, control)
                if code == SKIP:
                    print(f"skip: {adapter} has no capable D3D12 adapter")
                    return SKIP
                run_control(adapter, observed, mask, width, control_name, mutation, predicted)
    print(f"subgroup compaction fixture passed on {adapter}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
