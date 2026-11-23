"""Translate bounded raw-UAV gfx1013 compute programs under explicit bindings."""
import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

from pixel_aot import INTEGER, container_code, disassemble
from resource_inventory import parts


CONSTANT_WORDS = 16
AXES = "xyz"
# Unsigned VOPC compares a lane guard may use: v_cmp_<op>_u32 a, b is (a <op> b) per lane.
GUARD_COMPARES = {"gt": ">", "lt": "<", "ge": ">=", "le": "<="}
# The data ALU subset: the unsigned 32-bit operations the LLVM 15 code generator emits for the
# read-compute-write fixture, on known words only. v_and_b32 and v_lshrrev_b32 (src1 >> src0[4:0])
# reuse the pixel profile's integer templates, which have the same semantics. v_mad_u32_u24 is
# src0[23:0] * src1[23:0] + src2, modulo 2^32. None of them writes vcc, so the guard compare stays valid.
ALU = {
    "v_and_b32_e32": f"({INTEGER['and_b32'][1]})",
    "v_lshrrev_b32_e32": f"({INTEGER['lshrrev_b32'][1]})",
    "v_min_u32_e32": "min({0}, {1})",
    "v_mad_u32_u24": "((({0} & 0xffffffu) * ({1} & 0xffffffu)) + {2})",
}
ALU_SOURCE = r"(s\d+|v\d+|-?\d+|0x[0-9a-f]+)"
# Results are inline HLSL expressions; a longer one is rejected instead of growing without bound.
ALU_EXPRESSION_LIMIT = 1024


def checked_group(group):
    """Return (x, y, z) within the D3D12 cs_6_0 numthreads limits."""
    x, y, z = (group, 1, 1) if isinstance(group, int) else tuple(group)
    if not (all(isinstance(n, int) for n in (x, y, z)) and 1 <= x <= 1024 and 1 <= y <= 1024
            and 1 <= z <= 64 and x * y * z <= 1024):
        raise ValueError("Unsupported thread group")
    return x, y, z


def group_size(psv):
    size = struct.unpack_from("<I", psv)[0] if len(psv) >= 4 else 0
    if not 48 <= size <= len(psv) - 4 or psv[28] != 5:
        raise ValueError("Expected compute PSV runtime info with thread dimensions")
    return checked_group(struct.unpack_from("<3I", psv, 40))


def is_word(value):
    return isinstance(value, int) or (isinstance(value, tuple)
                                      and value[0] in ("constant_word", "thread", "source", "alu"))


def hlsl_word(value):
    if isinstance(value, int):
        return f"0x{value:08x}u"
    if value[0] == "source":
        # The source binding is read-only, so the read may sit where the word is used.
        return f"inputData.Load({value[1]})"
    if value[0] == "alu":
        return value[1]
    return f"tid.{AXES[value[1]]}" if value[0] == "thread" else f"params[{value[1] // 4}].{'xyzw'[value[1] % 4]}"


def alu_word(token, scalar, vector, pending):
    """The known word an ALU source names, or None: a VGPR, a loaded SGPR, an inline constant or a literal."""
    if token[0] == "v":
        value = vector.get(int(token[1:]))
    elif token[0] == "s":
        n = int(token[1:])
        value = scalar.get(n) if n <= 105 and n not in pending else None
    else:
        value = int(token, 0) & 0xffffffff if -(1 << 31) <= int(token, 0) <= 0xffffffff else None
    return value if is_word(value) else None


def vector_operands(line):
    """VGPR numbers named by an instruction, single or as a range."""
    regs = set()
    for single, first, last in re.findall(r"\bv(\d+)\b|\bv\[(\d+):(\d+)\]", line):
        regs.update([int(single)] if single else range(int(first), int(last) + 1))
    return regs


def labelled(decoded):
    """Name each s_cbranch_execz target as a label; decoded is [(byte offset, instruction)].

    Only forward targets at a decoded instruction boundary are named. lower() then checks
    where the label sits.
    """
    starts = {pc for pc, _ in decoded}
    targets, lines = set(), []
    for pc, line in decoded:
        if m := re.fullmatch(r"s_cbranch_execz (-?\d+)", line):
            # A backward target, printed signed or as its 16-bit pattern, is never accepted.
            target = pc + 4 + 4 * int(m[1])
            if target <= pc or target not in starts:
                raise ValueError("Branch outside decoded instruction boundaries")
            targets.add(target)
            line = f"s_cbranch_execz .L{target}"
        lines.append((pc, line))
    return [item for pc, line in lines for item in ([f".L{pc}:"] if pc in targets else []) + [line]]


def lower(instructions, group, *, table_reg, descriptor_offset, group_reg=None, stride=None,
          constant_descriptor=None, group_reg_y=None, group_reg_z=None, source_descriptor=None,
          source_stride=None, cpu="gfx1013", lds_bytes=0):
    """v0, v1 and v2 are local X, Y and Z; group_reg, group_reg_y and group_reg_z are the group IDs.

    gfx10 compute waves receive each local ID in its own VGPR. Y and Z are bound only when that
    group dimension is larger than one. Descriptors name u0 and optionally b0 and a read-only raw
    source t0; source_stride is the t0 byte stride that indexed reads use.
    """
    dims = checked_group(group)
    if cpu not in ("gfx1013", "gfx1030"):
        raise ValueError("Unsupported compute target")
    ds_group = cpu == "gfx1030" and dims == (32, 1, 1)
    if not 0 <= table_reg <= 105 or not 0 <= descriptor_offset < 1 << 20 or descriptor_offset % 4:
        raise ValueError("Invalid table or thread group binding")
    descriptor_bindings = {descriptor_offset: "uav"}
    if constant_descriptor is not None:
        if (not 0 <= constant_descriptor < 1 << 20 or constant_descriptor % 4
                or abs(constant_descriptor - descriptor_offset) < 16):
            raise ValueError("Invalid or overlapping constant descriptor binding")
        descriptor_bindings[constant_descriptor] = "cbv"
    if source_descriptor is not None:
        if (not 0 <= source_descriptor < 1 << 20 or source_descriptor % 4
                or any(abs(source_descriptor - offset) < 16 for offset in descriptor_bindings)):
            raise ValueError("Invalid or overlapping source descriptor binding")
        descriptor_bindings[source_descriptor] = "srv"
    if source_stride is not None and (source_descriptor is None or not 4 <= source_stride <= 0x3fff
                                      or source_stride % 4):
        raise ValueError("Indexed reads require a source binding and aligned stride")
    group_regs = {axis: reg for axis, reg in enumerate((group_reg, group_reg_y, group_reg_z)) if reg is not None}
    if (any(not 0 <= reg <= 105 or reg == table_reg for reg in group_regs.values())
            or len(set(group_regs.values())) != len(group_regs)):
        raise ValueError("Invalid group register")
    if stride is not None and (not group_regs or not 4 <= stride <= 0x3fff or stride % 4):
        raise ValueError("Indexed stores require a group register and aligned stride")
    scalar = {table_reg: "table"}
    for axis, reg in group_regs.items():
        scalar[reg] = ("group", axis)
    vector = {axis: ("local", axis) for axis in range(3) if axis == 0 or dims[axis] > 1}
    pending = set()
    # VGPRs a buffer_load_dword still writes, with the word they receive (None when unknown);
    # they enter vector only at s_waitcnt vmcnt(0).
    loading = {}
    stores = []
    ds_steps = []
    lds_enabled = ds_group and 128 <= lds_bytes <= 65536
    lds_word = None
    ds_wait = False
    written_spans = []
    layout = None
    # One per-lane guard: v_cmp into vcc_lo, s_and_saveexec_b32, the stores under the mask, then
    # s_or_b32 restoring exec or the final exit. An optional s_cbranch_execz must land exactly
    # where the guarded stores end. The guarded stores become an HLSL if.
    vcc = guard = None
    guard_span = None
    if not instructions or instructions[-1] != "s_endpgm":
        raise ValueError("Missing final program exit")

    for line in instructions[:-1]:
        if ds_wait:
            if line != "s_waitcnt lgkmcnt(0)":
                raise ValueError("Expected immediate s_waitcnt lgkmcnt(0) after DS")
            ds_wait = False
        if m := re.fullmatch(r"(\.L\w+):", line):
            if guard is None or guard["label"] != m[1] or guard["at_label"]:
                raise ValueError("Unverified branch target")
            guard["at_label"] = True
            continue
        if (guard is not None and guard["at_label"]
                and not re.fullmatch(r"s_waitcnt_depctr 0x[0-9a-f]+|s_nop \d+|s_or_b32 .*", line)):
            raise ValueError("Lane guard branch does not reach its restore")
        if re.fullmatch(r"s_inst_prefetch (?:0x[0-9a-f]+|\d+)|s_nop \d+|s_waitcnt_depctr 0x[0-9a-f]+", line):
            continue
        if m := re.fullmatch(r"s_clause (0x[0-9a-f]+|\d+)", line):
            if int(m[1], 0) > 0x3f:
                raise ValueError("Invalid s_clause operand")
            continue
        if not line.startswith("s_waitcnt") and vector_operands(line) & loading.keys():
            raise ValueError("Vector load result used or overwritten before s_waitcnt vmcnt(0)")
        if line in ("s_waitcnt lgkmcnt(0)", "s_waitcnt lgkmcnt(0) vmcnt(0)", "s_waitcnt vmcnt(0) lgkmcnt(0)",
                    "s_waitcnt vmcnt(0)"):
            if "lgkmcnt" in line:
                pending.clear()
            if "vmcnt" in line:
                for reg, value in loading.items():
                    if value is None:
                        vector.pop(reg, None)
                    else:
                        vector[reg] = value
                loading.clear()
        elif m := re.fullmatch(r"v_cmp_(gt|lt|ge|le)_u32_e32 vcc_lo, (s\d+|v\d+|-?\d+|0x[0-9a-f]+), v(\d+)", line):
            op, src, right = m[1], m[2], int(m[3])
            if src[0] in "sv":
                n = int(src[1:])
                left = (None if src[0] == "s" and (n > 105 or n in pending) or src[0] == "v" and n > 255
                        else (scalar if src[0] == "s" else vector).get(n))
            else:
                value = int(src, 0)
                left = value & 0xffffffff if -(1 << 31) <= value <= 0xffffffff else None
            # A word read or computed in the program is not a verified guard operand.
            if (right > 255 or not all(is_word(word) and not (isinstance(word, tuple) and word[0] in ("source", "alu"))
                                       for word in (left, vector.get(right)))):
                raise ValueError("Unverified lane guard compare")
            vcc = (op, left, vector[right])
        elif m := re.fullmatch(r"s_and_saveexec_b32 s(\d+), vcc_lo", line):
            saved = int(m[1])
            if guard_span is not None or vcc is None or saved > 105 or saved in pending:
                raise ValueError("Unverified or repeated lane guard")
            guard = dict(label=None, at_label=False, holders=None,
                         scalar=dict(scalar), vector=dict(vector), pending=set(pending), loading=dict(loading))
            scalar[saved] = ("exec",)
            guard_span = [len(stores), None, vcc]
        elif m := re.fullmatch(r"s_cbranch_execz (\.L\w+)", line):
            # Taken only when no lane passed the guard; it must skip exactly the guarded stores.
            if guard is None:
                raise ValueError("Unverified lane guard branch")
            guard["label"] = m[1]
            # A taken branch skips the block: only an SGPR that holds the mask here holds it on both paths.
            holders = {reg for reg, value in scalar.items() if value == ("exec",)}
            guard["holders"] = holders if guard["holders"] is None else guard["holders"] & holders
        elif m := re.fullmatch(r"s_or_b32 exec_lo, (?:exec_lo, s(\d+)|s(\d+), exec_lo)", line):
            # Any SGPR that still holds the saved mask restores it, on the skipped path too.
            reg = int(m[1] or m[2])
            if (guard is None or scalar.get(reg) != ("exec",)
                    or guard["label"] is not None and (not guard["at_label"] or reg not in guard["holders"])):
                raise ValueError("Unverified lane guard restore")
            # Registers written under the guard keep their old values in the lanes that did not
            # pass it, or in the whole wave when the branch skips the block: unknown after the
            # restore. A wait under the guard may have been skipped, so earlier loads stay pending.
            # A load from before the guard keeps its word only if its register still holds that
            # word on the guarded path, waited for or not. Any other loaded register, including a
            # load issued under the guard (it writes only the lanes that passed), is unknown after
            # its wait.
            loading = {reg: guard["loading"].get(reg) if loading.get(reg, vector.get(reg)) == guard["loading"].get(reg)
                       else None for reg in guard["loading"].keys() | loading.keys()}
            for state, before in ((scalar, guard["scalar"]), (vector, guard["vector"])):
                for reg in [reg for reg, value in state.items() if reg not in before or before[reg] != value]:
                    del state[reg]
            pending |= guard["pending"]
            guard = None
            guard_span[1] = len(stores)
        elif m := re.fullmatch(r"s_mov_b32 s(\d+), (s\d+|-?\d+|0x[0-9a-f]+)", line):
            dst, src = int(m[1]), m[2]
            if dst in pending or dst > 105:
                raise ValueError("Invalid scalar destination or pending load")
            if src.startswith("s"):
                n = int(src[1:])
                if n > 105 or n in pending or n not in scalar:
                    raise ValueError("Unknown or pending scalar source")
                scalar[dst] = scalar[n]
            else:
                value = int(src, 0)
                if not -(1 << 31) <= value <= 0xffffffff:
                    raise ValueError("Invalid scalar immediate")
                scalar[dst] = value & 0xffffffff
        elif m := re.fullmatch(r"s_movk_i32 s(\d+), (-?\d+|0x[0-9a-f]+)", line):
            dst = int(m[1])
            value = int(m[2], 0)
            if dst in pending or dst > 105 or not -0x8000 <= value <= 0xffff:
                raise ValueError("Invalid scalar destination or movk immediate")
            signed = value - 0x10000 if value >= 0x8000 else value
            scalar[dst] = signed & 0xffffffff
        elif m := re.fullmatch(r"s_lshl_b32 s(\d+), s(\d+), (\d+)", line):
            dst, group, shift = map(int, m.groups())
            tag = scalar.get(group)
            axis = tag[1] if isinstance(tag, tuple) and tag[0] == "group" else None
            # A group's first dispatch index, group * size, only for a power-of-two size.
            if (dst in pending or dst > 105 or axis is None or dims[axis] & (dims[axis] - 1)
                    or shift != dims[axis].bit_length() - 1):
                raise ValueError("Unverified group base")
            scalar[dst] = ("group_base", axis)
        elif m := re.fullmatch(r"s_load_dwordx([48]) s\[(\d+):(\d+)\], s\[(\d+):(\d+)\], (null|0x[0-9a-f]+)", line):
            # x8 loads two descriptors bound at consecutive 16-byte slots of the table.
            count = int(m[1])
            dest, last, base, high = map(int, m.groups()[1:5])
            offset = 0 if m[6] == "null" else int(m[6], 16)
            regs = range(dest, last + 1)
            slots = range(offset, offset + 4 * count, 16)
            if (dest % 4 or last != dest + count - 1 or last > 105 or base % 2 or high != base + 1
                    or high > 105 or pending.intersection((*regs, base, high))
                    or (scalar.get(base), scalar.get(high)) != ("table", 0)
                    or any(slot not in descriptor_bindings for slot in slots)):
                raise ValueError("Unresolved compute descriptor load")
            for word, reg in enumerate(regs):
                scalar[reg] = (descriptor_bindings[slots[word // 4]], word % 4)
            pending.update(regs)
        elif m := re.fullmatch(r"(?:s_buffer_load_dwordx([24]) s\[(\d+):(\d+)\]|s_buffer_load_dword s(\d+)), "
                               r"s\[(\d+):(\d+)\], (null|0x[0-9a-f]+)", line):
            count = int(m[1] or 1)
            dest, last = (int(m[4]), int(m[4])) if m[4] else (int(m[2]), int(m[3]))
            base, high = int(m[5]), int(m[6])
            offset = 0 if m[7] == "null" else int(m[7], 16)
            regs = range(dest, last + 1)
            if (dest % count or last != dest + count - 1 or last > 105
                    or base % 4 or high != base + 3 or high > 105
                    or offset % 4 or offset // 4 + count > CONSTANT_WORDS
                    or pending.intersection((*regs, *range(base, high + 1)))
                    or any(scalar.get(base + i) != ("cbv", i) for i in range(4))):
                raise ValueError("Unresolved or out-of-range compute constant read")
            for word, reg in enumerate(regs, offset // 4):
                scalar[reg] = ("constant_word", word)
            pending.update(regs)
        elif m := re.fullmatch(r"v_mov_b32(?:_e32|_e64)? v(\d+), (s\d+|v\d+|-?\d+|0x[0-9a-f]+)", line):
            dst, src = int(m[1]), m[2]
            if dst > 255:
                raise ValueError("Invalid vector destination")
            if src.startswith("s"):
                n = int(src[1:])
                val = scalar.get(n)
                if n > 105 or n in pending or not is_word(val):
                    raise ValueError("Unresolved scalar move source")
                vector[dst] = val
            elif src.startswith("v"):
                n = int(src[1:])
                val = vector.get(n)
                if n > 255 or not is_word(val):
                    raise ValueError("Unresolved vector move source")
                vector[dst] = val
            else:
                value = int(src, 0)
                if not -(1 << 31) <= value <= 0xffffffff:
                    raise ValueError("Invalid literal move")
                vector[dst] = value & 0xffffffff
        elif m := re.fullmatch(r"v_lshl_add_u32 v(\d+), s(\d+), (\d+), v(\d+)", line):
            dst, group, shift, local = map(int, m.groups())
            tag = scalar.get(group)
            axis = tag[1] if isinstance(tag, tuple) and tag[0] == "group" else None
            # The dispatch-thread ID is group * size + local only for a power-of-two size.
            if (dst > 255 or group in pending or axis is None or dims[axis] & (dims[axis] - 1)
                    or shift != dims[axis].bit_length() - 1 or vector.get(local) != ("local", axis)):
                raise ValueError("Unverified dispatch index")
            vector[dst] = ("thread", axis)
        elif m := re.fullmatch(r"v_add_lshl_u32 v(\d+), s(\d+), v(\d+), (\d+)", line):
            dst, base, local, shift = map(int, m.groups())
            tag = scalar.get(base)
            # (group base + local ID of the same axis) << shift: the byte offset of a dispatch
            # index, as from v_lshlrev_b32. The store checks bound the shift.
            if (dst > 255 or not (isinstance(tag, tuple) and tag[0] == "group_base")
                    or vector.get(local) != ("local", tag[1])):
                raise ValueError("Unverified dispatch byte offset")
            vector[dst] = ("thread_bytes", tag[1], shift)
        elif (ds_group and (m := re.fullmatch(r"v_add_nc_u32_e32 v(\d+), (\d+), v(\d+)", line))
              and vector.get(int(m[3])) == ("local", 0)):
            dst, rotation, _ = map(int, m.groups())
            if dst > 255 or rotation > 31:
                raise ValueError("Unverified DS lane bias")
            vector[dst] = ("ds_biased_lane", rotation)
        elif (ds_group and (m := re.fullmatch(r"v_and_b32_e32 v(\d+), 31, v(\d+)", line))
              and (tag := vector.get(int(m[2]))) and isinstance(tag, tuple) and tag[0] == "ds_biased_lane"):
            dst = int(m[1])
            if dst > 255:
                raise ValueError("Invalid vector destination")
            vector[dst] = ("ds_lane", tag[1])
        elif m := re.fullmatch(r"v_lshlrev_b32(?:_e32|_e64)? v(\d+), (\d+), v(\d+)", line):
            dst, shift, src = map(int, m.groups())
            tag = vector.get(src)
            if ds_group and dst <= 255 and shift == 2 and isinstance(tag, tuple) and tag[0] == "ds_lane":
                vector[dst] = ("ds_address", tag[1])
                continue
            if lds_enabled and dst <= 255 and shift == 2 and tag == ("local", 0):
                vector[dst] = ("lds_address",)
                continue
            # The element and row checks at the store bound the shift.
            if dst > 255 or not (isinstance(tag, tuple) and tag[0] == "thread"):
                raise ValueError("Unverified dispatch byte offset")
            vector[dst] = ("thread_bytes", tag[1], shift)
        elif m := re.fullmatch(r"ds_(b?permute)_b32 v(\d+), v(\d+), v(\d+)", line):
            op, dst, addr, src = m[1], *map(int, m.groups()[1:])
            address = vector.get(addr)
            value = vector.get(src)
            if (not ds_group or guard_span is not None or max(dst, addr, src) > 255
                    or not isinstance(address, tuple) or address[0] != "ds_address" or not is_word(value)
                    or len(ds_steps) >= 64):
                raise ValueError("Unverified DS lane routing")
            # The bijective address is (local X + bias) mod 32. Scatter reads from its inverse.
            bias = address[1] if op == "bpermute" else (-address[1]) % 32
            slot = f"ds_stage_{len(ds_steps)}"
            result = f"ds_result_{len(ds_steps)}"
            ds_steps.append((slot, f"    {slot}[local.x] = {hlsl_word(value)};\n"
                                   f"    GroupMemoryBarrierWithGroupSync();\n"
                                   f"    uint {result} = {slot}[(local.x + {bias}u) & 31u];"))
            vector[dst] = ("alu", result)
            ds_wait = True
        elif m := re.fullmatch(r"ds_write_b(8|16|32) v(\d+), v(\d+)(?: offset:([0-3]))?", line):
            width, addr, src = map(int, m.groups()[:3])
            offset = int(m[4] or 0)
            value = vector.get(src)
            if (not lds_enabled or guard_span is not None or max(addr, src) > 255
                    or vector.get(addr) != ("lds_address",) or not is_word(value)
                    or offset % (width // 8) or (width < 32 and lds_word is None)):
                raise ValueError("Unverified private LDS write")
            # Every lane owns one disjoint word. Capture values even if their VGPRs are overwritten.
            if width == 32:
                lds_word = value
            else:
                shift = offset * 8
                mask = (1 << width) - 1
                keep = 0xffffffff ^ (mask << shift)
                expression = (f"(({hlsl_word(lds_word)} & 0x{keep:08x}u) | "
                              f"(({hlsl_word(value)} & 0x{mask:x}u) << {shift}u))")
                if len(expression) > ALU_EXPRESSION_LIMIT:
                    raise ValueError("LDS expression exceeds the profile limit")
                lds_word = ("alu", expression)
            ds_wait = True
        elif m := re.fullmatch(r"ds_read_(b32|[ui](?:8|16)) v(\d+), v(\d+)(?: offset:([0-3]))?", line):
            op, dst, addr, offset = m[1], int(m[2]), int(m[3]), int(m[4] or 0)
            width = int(op[1:])
            if (not lds_enabled or guard_span is not None or max(dst, addr) > 255
                    or vector.get(addr) != ("lds_address",) or lds_word is None or offset % (width // 8)):
                raise ValueError("Unverified private LDS read")
            if width == 32:
                vector[dst] = lds_word
            else:
                shifted = f"({hlsl_word(lds_word)} >> {offset * 8}u)"
                if op.startswith("i"):
                    # Shifting to the sign bit discards any higher neighbors before the arithmetic shift.
                    extend = 32 - width
                    expression = f"asuint(asint({shifted} << {extend}u) >> {extend}u)"
                else:
                    expression = f"({shifted} & 0x{(1 << width) - 1:x}u)"
                if len(expression) > ALU_EXPRESSION_LIMIT:
                    raise ValueError("LDS expression exceeds the profile limit")
                vector[dst] = ("alu", expression)
            ds_wait = True
        elif m := re.fullmatch(r"v_lshl_add_u32 v(\d+), v(\d+), (\d+), v(\d+)", line):
            dst, row, shift, column = map(int, m.groups())
            row_tag, column_tag = vector.get(row), vector.get(column)
            # (row index << shift) + column byte offset, from two different dispatch axes.
            # The plane and element checks at the store bound both shifts.
            if (dst > 255 or not (isinstance(row_tag, tuple) and row_tag[0] == "thread")
                    or not (isinstance(column_tag, tuple) and column_tag[0] == "thread_bytes")
                    or row_tag[1] == column_tag[1]):
                raise ValueError("Unverified plane byte offset")
            vector[dst] = ("plane_bytes", row_tag[1], shift, *column_tag[1:])
        elif ((m := re.fullmatch(rf"(v_(?:and_b32|lshrrev_b32|min_u32)_e32) v(\d+), {ALU_SOURCE}, (v\d+)", line))
              or (m := re.fullmatch(rf"(v_mad_u32_u24) v(\d+), {ALU_SOURCE}, {ALU_SOURCE}, {ALU_SOURCE}", line))
              and None not in [alu_word(token, scalar, vector, pending) for token in m.groups()[2:]]):
            # VOP2 takes a VGPR as src1. A v_mad_u32_u24 with an address operand is a row pitch below.
            op, dst = m[1], int(m[2])
            words = [alu_word(token, scalar, vector, pending) for token in m.groups()[2:]]
            if dst > 255 or None in words:
                raise ValueError("Unverified ALU operand")
            expression = ALU[op].format(*map(hlsl_word, words))
            if len(expression) > ALU_EXPRESSION_LIMIT:
                raise ValueError("ALU expression too large")
            vector[dst] = ("alu", expression)
        elif m := re.fullmatch(r"v_mad_u32_u24 v(\d+), (v\d+|-?\d+|0x[0-9a-f]+), (v\d+|-?\d+|0x[0-9a-f]+), v(\d+)", line):
            dst, column = int(m[1]), int(m[4])
            # The literal pitch may be either multiplicand; LLVM 15 places it in src0.
            row, pitch = (m[2], m[3]) if m[2].startswith("v") else (m[3], m[2])
            row_tag = vector.get(int(row[1:])) if row.startswith("v") else None
            column_tag = vector.get(column)
            # (bits [23:0] of the row index) * pitch + column byte offset. The row check at the store
            # bounds the column shift against the pitch.
            if (dst > 255 or pitch.startswith("v") or not 4 <= int(pitch, 0) < 1 << 24 or int(pitch, 0) % 4
                    or not (isinstance(row_tag, tuple) and row_tag[0] == "thread")
                    or not (isinstance(column_tag, tuple) and column_tag[0] == "thread_bytes")
                    or row_tag[1] == column_tag[1]):
                raise ValueError("Unverified row pitch byte offset")
            vector[dst] = ("pitch_bytes", row_tag[1], int(pitch, 0), *column_tag[1:])
        elif m := re.fullmatch(r"buffer_load_dword v(\d+), (off|v\d+), s\[(\d+):(\d+)\], 0( idxen)?( offen)?"
                               r"(?: offset:(\d+))?", line):
            dst, base, high = int(m[1]), int(m[3]), int(m[4])
            offset = int(m[7] or 0)
            if (dst > 255 or high != base + 3 or base % 4 or high > 105
                    or pending.intersection(range(base, high + 1))):
                raise ValueError("Unverified buffer read descriptor")
            kinds = {scalar.get(base + i) for i in range(4)}
            if kinds == {("uav", i) for i in range(4)}:
                # The store target would need ordering against other invocations' stores.
                raise ValueError("Reads from the stored UAV are not supported")
            if kinds != {("srv", i) for i in range(4)}:
                raise ValueError("Unverified buffer read descriptor")
            if offset % 4 or offset > 4095:
                raise ValueError("Unverified read offset")
            tag = vector.get(int(m[2][1:])) if m[2] != "off" else None
            if m[2] == "off" and not m[5] and not m[6]:
                address = f"{offset}u"
            elif (m[6] and not m[5] and isinstance(tag, tuple) and tag[0] == "thread_bytes"
                  and 2 <= tag[2] < 32):
                # A shift below 2 gives unaligned byte offsets, which ByteAddressBuffer.Load cannot express.
                address = f"(tid.{AXES[tag[1]]} << {tag[2]}u) + {offset}u"
            elif (m[5] and not m[6] and source_stride is not None and isinstance(tag, tuple)
                  and tag[0] == "thread"):
                # idxen: index * stride + instruction offset, inside one source element.
                if offset + 4 > source_stride:
                    raise ValueError("Read leaves its source element")
                address = f"tid.{AXES[tag[1]]} * {source_stride}u + {offset}u"
            else:
                raise ValueError("Unsupported read address")
            vector.pop(dst, None)
            loading[dst] = ("source", address)
        elif m := re.fullmatch(r"buffer_store_dword(?:x([234]))? (v\d+|v\[\d+:\d+\]), (off|v\d+|v\[\d+:\d+\]), "
                               r"s\[(\d+):(\d+)\], 0( idxen)?( offen)?(?: offset:(\d+))?", line):
            count = int(m[1] or 1)
            first, *last = map(int, re.findall(r"\d+", m[2]))
            base, high = int(m[4]), int(m[5])
            offset = int(m[8] or 0)
            if (last != ([] if count == 1 else [first + count - 1])
                    or high != base + 3 or base % 4 or offset % 4 or offset > 4095
                    or pending.intersection(range(base, high + 1))
                    or any(scalar.get(base + i) != ("uav", i) for i in range(4))):
                raise ValueError("Unverified raw UAV store")
            address = f"{offset}u"
            addr_regs = list(map(int, re.findall(r"\d+", m[3])))
            tags = [vector.get(reg) for reg in addr_regs]
            if (m[3] == "off" and not m[6] and not m[7] and dims == (1, 1, 1) and stride is None):
                store_layout = ("fixed",)
                covered = set()
            elif (stride is not None and m[6] and not m[7] and len(addr_regs) == 1
                    and isinstance(tags[0], tuple) and tags[0][0] == "thread" and offset + 4 * count <= stride):
                # One index axis: every other group dimension must be one, so no two
                # invocations of a group share an address.
                index = tags[0][1]
                store_layout = ("index", index)
                covered = {index}
                address = f"tid.{AXES[index]} * {stride}u + {offset}u"
            elif (stride is not None and m[6] and m[7] and len(addr_regs) == 2 and addr_regs[1] == addr_regs[0] + 1
                    and all(isinstance(tag, tuple) for tag in tags)
                    and tags[0][0] == "thread" and tags[1][0] == "thread_bytes" and tags[0][1] != tags[1][1]):
                # idxen offen: address = index * stride + byte offset + instruction offset.
                (_, index), (_, column, shift) = tags
                if (offset + 4 * count > 1 << shift or dims[column] << shift > stride):
                    raise ValueError("Store leaves its element or row")
                store_layout = ("index_offset", index, column, shift)
                covered = {index, column}
                address = f"tid.{AXES[index]} * {stride}u + (tid.{AXES[column]} << {shift}u) + {offset}u"
            elif (stride is not None and m[6] and m[7] and len(addr_regs) == 2 and addr_regs[1] == addr_regs[0] + 1
                    and all(isinstance(tag, tuple) for tag in tags)
                    and tags[0][0] == "thread" and tags[1][0] == "plane_bytes"
                    and tags[0][1] not in (tags[1][1], tags[1][3])):
                # Three axes: index * stride + (row << row_shift) + (column << shift) + instruction offset.
                (_, index), (_, row, row_shift, column, shift) = tags
                if (offset + 4 * count > 1 << shift or dims[column] << shift > 1 << row_shift
                        or dims[row] << row_shift > stride):
                    raise ValueError("Store leaves its element, row or plane")
                store_layout = ("index_plane", index, row, row_shift, column, shift)
                covered = {index, row, column}
                address = (f"tid.{AXES[index]} * {stride}u + (tid.{AXES[row]} << {row_shift}u)"
                           f" + (tid.{AXES[column]} << {shift}u) + {offset}u")
            elif (m[7] and not m[6] and len(addr_regs) == 1
                    and isinstance(tags[0], tuple) and tags[0][0] == "pitch_bytes"):
                # offen only: (row & 0xffffff) * pitch + (column << shift) + instruction offset.
                # Without idxen the descriptor stride is not used, so the stride binding is ignored.
                _, row, pitch, column, shift = tags[0]
                # A group's rows start on a 24-bit-aligned power-of-two boundary, so masking keeps them
                # consecutive; their pitches must not wrap the 32-bit address onto each other.
                if (offset + 4 * count > 1 << shift or dims[column] << shift > pitch
                        or dims[row] * pitch > 1 << 32):
                    raise ValueError("Store leaves its element or row")
                store_layout = ("pitch", row, pitch, column, shift)
                covered = {row, column}
                address = f"(tid.{AXES[row]} & 0xffffffu) * {pitch}u + (tid.{AXES[column]} << {shift}u) + {offset}u"
            else:
                raise ValueError("Unsupported store address")
            if any(dims[axis] > 1 for axis in range(3) if axis not in covered):
                raise ValueError("Store address does not separate every invocation of the group")
            if layout not in (None, store_layout):
                raise ValueError("Stores use different address layouts")
            layout = store_layout
            span_end = offset + 4 * count
            if any(lo < span_end and offset < hi for lo, hi in written_spans):
                raise ValueError("Overlapping raw UAV stores")
            written_spans.append((offset, span_end))
            values = [vector.get(reg) for reg in range(first, first + count)]
            if not all(is_word(value) for value in values):
                raise ValueError("Store data is not a known word")
            words = ", ".join(map(hlsl_word, values))
            arg = words if count == 1 else f"uint{count}({words})"
            stores.append(f"    outputData.Store{count if count > 1 else ''}({address}, {arg});")
        else:
            raise ValueError(f"Unsupported compute instruction: {line}")
    if ds_wait:
        raise ValueError("Expected immediate s_waitcnt lgkmcnt(0) after DS")
    if not stores:
        raise ValueError("Expected at least one observable store")
    if guard is not None and guard["label"] is not None and not guard["at_label"]:
        raise ValueError("Unverified branch target")
    if guard_span is not None:
        # Without a restore the mask holds until the exit.
        start, end = guard_span[0], len(stores) if guard_span[1] is None else guard_span[1]
        if start == end:
            raise ValueError("Expected a store under the lane guard")
        op, left, right = guard_span[2]
        stores[start:end] = [f"    if ({hlsl_word(left)} {GUARD_COMPARES[op]} {hlsl_word(right)}) {{",
                             *("    " + store for store in stores[start:end]), "    }"]
    body = "\n".join(stores)
    cbuffer = (f"cbuffer Constants : register(b0) {{ uint4 params[{CONSTANT_WORDS // 4}]; }};\n"
               if constant_descriptor is not None else "")
    if source_descriptor is not None:
        cbuffer += "ByteAddressBuffer inputData : register(t0);\n"
    local = ""
    if ds_steps:
        cbuffer += "".join(f"groupshared uint {slot}[32];\n" for slot, _ in ds_steps)
        body = "\n".join([*(step for _, step in ds_steps), body])
        local = ", uint3 local : SV_GroupThreadID"
    return (f"{cbuffer}RWByteAddressBuffer outputData : register(u0);\n[numthreads({dims[0]},{dims[1]},{dims[2]})]\n"
            f"void main(uint3 tid : SV_DispatchThreadID{local}) {{\n{body}\n}}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("container", type=Path, help="Local extracted DXBC container")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llvm", required=True)
    parser.add_argument("--cpu", choices=("gfx1013", "gfx1030"), default="gfx1013")
    parser.add_argument("--lds-bytes", type=int, default=0, help="Known native group LDS allocation")
    parser.add_argument("--table-reg", type=int, required=True)
    parser.add_argument("--descriptor-offset", type=lambda n: int(n, 0), required=True)
    parser.add_argument("--constant-descriptor", type=lambda n: int(n, 0), help="Table byte offset bound to b0")
    parser.add_argument("--group-reg", type=int, help="SGPR holding the group X ID")
    parser.add_argument("--group-reg-y", type=int, help="SGPR holding the group Y ID")
    parser.add_argument("--group-reg-z", type=int, help="SGPR holding the group Z ID")
    parser.add_argument("--stride", type=int, help="Indexed raw UAV byte stride")
    parser.add_argument("--source-descriptor", type=lambda n: int(n, 0),
                        help="Table byte offset bound to stable read-only t0, with no overlap with u0")
    parser.add_argument("--source-stride", type=int, help="Indexed t0 byte stride")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".hlsl" or args.output.resolve() == args.container.resolve():
        parser.error("Choose an HLSL destination distinct from the input")
    blob = args.container.read_bytes()
    chunks = parts(blob)
    group = group_size(chunks.get(b"PSV0", b""))
    code = container_code(chunks.get(b"DXIL", b""), stage=5, llvm=args.llvm)
    features = b"+wavefrontsize32,-wavefrontsize64" if args.cpu == "gfx1030" else b""
    decoded = disassemble(code, args.llvm, offsets=True, cpu=args.cpu.encode(), features=features)
    source = lower(labelled(decoded), group, table_reg=args.table_reg,
                   descriptor_offset=args.descriptor_offset, group_reg=args.group_reg, stride=args.stride,
                   constant_descriptor=args.constant_descriptor, group_reg_y=args.group_reg_y,
                   group_reg_z=args.group_reg_z, source_descriptor=args.source_descriptor,
                   source_stride=args.source_stride, cpu=args.cpu, lds_bytes=args.lds_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source, encoding="ascii")
    print(json.dumps({"container_sha256": hashlib.sha256(blob).hexdigest(),
                      "native_sha256": hashlib.sha256(code).hexdigest(), "numthreads": list(group)}))


if __name__ == "__main__":
    main()
