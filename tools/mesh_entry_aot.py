"""Specialize a bounded, two-wave gfx1013 mesh entry for desktop D3D12.

Geometry words and dispatch ABI are static inputs. Position and transform buffers
remain GPU inputs. Only forward control flow decided by static values is accepted.
"""

import argparse
import json
import re
import struct
from pathlib import Path

from pixel_aot import container_code, disassemble
from resource_inventory import parts

U32 = (1 << 32) - 1
WAVE = (1 << 64) - 1


def span(token):
    m = re.fullmatch(r"[sv]\[(\d+):(\d+)\]", token)
    if not m or int(m[2]) < int(m[1]):
        raise ValueError("Invalid register span")
    return range(int(m[1]), int(m[2]) + 1)


# Inline raw descriptor constructed from the two bound entry SGPRs, a byte
# stride, identity component selects, and the observed raw-buffer flags.
VERTEX_DESCRIPTOR = [0x20000, 0x10000, U32, 0x11014FAC]
VERTEX_BYTES = 512


class Wave:
    def __init__(self, wave, geometry, vertex_header=None):
        self.wave = wave
        self.geometry = geometry
        self.s = {3: wave << 24, 17: 0x10000, 20: 0, 21: 0}
        self.vertex_header = vertex_header
        if vertex_header is not None:
            self.s.update({18: VERTEX_DESCRIPTOR[0], 19: 0})
        self.v = [{5: 0} for _ in range(64)]
        self.exec = WAVE
        self.vcc = None
        self.scc = None
        self.pending = set()
        self.pending_v = set()
        self.barriers = []
        self.counts = []
        self.positions = {}
        self.attributes = {}
        self.indices = {}
        self.m0 = None

    def scalar(self, t):
        if t.startswith("s["):
            regs = span(t)
            if len(regs) != 2:
                raise ValueError("Expected a scalar pair")
            lo, hi = (self.scalar(f"s{i}") for i in regs)
            return lo | hi << 32
        if re.fullmatch(r"s\d+", t):
            i = int(t[1:])
            if i in self.pending:
                raise ValueError("Scalar load used before wait")
            return self.s[i]
        if t == "exec":
            return self.exec
        if t == "vcc":
            if self.vcc is None:
                raise ValueError("Undefined wave predicate")
            return self.vcc
        if re.fullmatch(r"-?\d+\.\d+", t):
            return struct.unpack("<I", struct.pack("<f", float(t)))[0]
        return int(t, 0)

    def put(self, dest, value, wide=False):
        if dest == "exec":
            self.exec = value & WAVE
        elif dest == "vcc":
            self.vcc = value & WAVE
        elif dest == "m0":
            self.m0 = value & U32
        else:
            if dest.startswith("s["):
                rr = span(dest)
                if len(rr) != 2:
                    raise ValueError("Expected a pair destination")
                start = rr.start
                wide = True
            elif re.fullmatch(r"s\d+", dest):
                start = int(dest[1:])
            else:
                raise ValueError("Unsupported scalar destination")
            regs = range(start, start + (2 if wide else 1))
            if self.pending.intersection(regs):
                raise ValueError("Pending scalar destination")
            for j, i in enumerate(regs):
                self.s[i] = (value >> (32 * j)) & U32

    def lane_value(self, t, lane):
        if re.fullmatch(r"v\d+", t):
            if (lane, t) in self.pending_v:
                raise ValueError("Vector load used before wait")
            return self.v[lane][int(t[1:])]
        return self.scalar(t)

    def load(self, op, args):
        dest, base, offset = args
        regs = span(dest) if "[" in dest else [int(dest[1:])]
        rr = span(base)
        off = 0 if offset == "null" else self.scalar(offset)
        count = int(op.split("dwordx")[1]) if "dwordx" in op else 1
        if off % 4 or count != len(regs) or self.pending.intersection(regs):
            raise ValueError("Invalid scalar load")
        if op == "s_load_dwordx4":
            roles = (
                {32: 32, 96: 96, 128: 128}
                if self.vertex_header is None
                else {0: 32, 32: 96, 64: 128}
            )
            if self.scalar(base) != 0x10000 or off not in roles:
                raise ValueError("Unknown descriptor table or offset")
            data = [(roles[off], j) for j in range(4)]
        else:
            desc = [self.scalar(f"s{i}") for i in rr]
            if self.vertex_header is not None and desc == VERTEX_DESCRIPTOR:
                if off != 0 or count != 1:
                    raise ValueError(
                        "Only the specialized header word is available to scalar loads"
                    )
                self.s[regs[0]] = self.vertex_header
                self.pending.update(regs)
                return
            kind = desc[0][0] if isinstance(desc[0], tuple) else None
            if desc != [(kind, j) for j in range(4)]:
                raise ValueError("Unbound buffer")
            first = off // 4
            if kind == 32:
                data = self.geometry[first : first + count]
            elif kind in (96, 128) and first + count <= 128:
                name = "pointData" if kind == 96 else "matrixData"
                data = [
                    f"{name}[{i // 4}].{'xyzw'[i % 4]}"
                    for i in range(first, first + count)
                ]
            else:
                raise ValueError("Unsupported uniform read")
            if len(data) != count:
                raise ValueError("Geometry read outside binding")
        self.s.update(zip(regs, data))
        self.pending.update(regs)

    def scalar_op(self, op, args):
        d, *src = args
        if op in (
            "s_load_dwordx4",
            "s_buffer_load_dword",
            "s_buffer_load_dwordx2",
            "s_buffer_load_dwordx4",
            "s_buffer_load_dwordx16",
        ):
            self.load(op, args)
            return
        if op.startswith("s_cmp_"):
            a, b = map(self.scalar, args)
            if not isinstance(a, int) or not isinstance(b, int):
                raise ValueError("Dynamic scalar predicate")
            a, b = a & U32, b & U32
            self.scc = {
                "s_cmp_eq_u32": a == b,
                "s_cmp_lt_u32": a < b,
                "s_cmp_gt_u32": a > b,
            }[op]
            return
        if op == "s_and_saveexec_b64":
            a = self.scalar(src[0])
            old = self.exec
            self.exec &= a
            self.put(d, old, True)
            self.scc = self.exec != 0
            return
        vals = [self.scalar(t) for t in src]
        if any(type(v) is not int for v in vals):
            raise ValueError("Scalar ALU requires concrete operands")
        a = vals[0]
        b = vals[-1]
        if op in ("s_mov_b32", "s_mov_b64"):
            v = a
        elif op == "s_movk_i32":
            v = ((a & 0xFFFF) ^ 0x8000) - 0x8000
        elif op in ("s_cselect_b32", "s_cselect_b64"):
            if self.scc is None:
                raise ValueError("Undefined scalar predicate")
            v = a if self.scc else b
        elif op == "s_bfe_u32":
            v = ((a & U32) >> (b & 31)) & ((1 << ((b >> 16) & 127)) - 1)
        elif op == "s_lshl_b32":
            v = a << (b & 31)
        elif op == "s_lshr_b32":
            v = (a & U32) >> (b & 31)
        elif op == "s_add_i32":
            v = a + b
        elif op == "s_and_b32" or op == "s_and_b64":
            v = a & b
        elif op in ("s_or_b32", "s_or_b64"):
            v = a | b
        elif op == "s_xor_b64":
            v = a ^ b
        elif op == "s_andn2_b64":
            v = a & ~b
        else:
            raise ValueError(f"Unsupported scalar instruction {op}")
        wide = op.endswith("_b64")
        self.put(d, v, wide)
        if not op.startswith(("s_mov", "s_cselect")):
            self.scc = (
                bool((~(a ^ b) & (a ^ v)) & (1 << 31))
                if op == "s_add_i32"
                else bool(v & (WAVE if wide else U32))
            )

    def vector_op(self, op, args, expressions):
        d, *src = args
        live = [i for i in range(64) if self.exec & (1 << i)]
        if op == "v_readfirstlane_b32":
            if not live:
                raise ValueError("Undefined first live lane")
            self.put(d, self.lane_value(src[0], live[0]))
            return
        if re.fullmatch(r"v_cmpx?_(?:gt|eq|le)_u32_e(?:32|64)", op):
            writes_exec = op.startswith("v_cmpx_")
            a, b = args if writes_exec else src
            mask = 0
            for lane in live:
                x, y = self.lane_value(a, lane), self.lane_value(b, lane)
                which = op.split("_")[2]
                if not isinstance(x, int) or not isinstance(y, int):
                    raise ValueError("Dynamic branch predicate")
                x, y = x & U32, y & U32
                if {"gt": x > y, "eq": x == y, "le": x <= y}[which]:
                    mask |= 1 << lane
            if writes_exec:
                self.exec = mask
            else:
                self.put(d, mask, True)
            return
        float_ops = {
            "v_mul_legacy_f32_e32",
            "v_mul_legacy_f32_e64",
            "v_fmac_f32_e32",
            "v_fmac_f32_e64",
            "v_add_f32_e32",
            "v_fma_f32",
        }
        for lane in live:
            vals = [self.lane_value(t, lane) for t in src]
            if (
                op not in float_ops
                and op
                not in ("v_mov_b32_e32", "v_cndmask_b32_e64", "v_cndmask_b32_e32")
                and any(type(value) is not int for value in vals)
            ):
                raise ValueError("Vector integer ALU requires concrete operands")
            a, b = vals[:2] if len(vals) >= 2 else (vals[0], 0)
            if op == "v_mbcnt_lo_u32_b32":
                v = ((a & U32) & ((1 << min(lane, 32)) - 1)).bit_count() + b
            elif op == "v_mbcnt_hi_u32_b32":
                v = ((a & U32) & ((1 << max(lane - 32, 0)) - 1)).bit_count() + b
            elif op == "v_mov_b32_e32":
                v = a
            elif op in ("v_lshl_add_u32", "v_or3_b32"):
                v = (
                    ((a << (b & 31)) + vals[2])
                    if op == "v_lshl_add_u32"
                    else a | b | vals[2]
                )
            elif op == "v_add_nc_u32_e32":
                v = a + b
            elif op == "v_mad_u32_u24":
                v = (a & 0xFFFFFF) * (b & 0xFFFFFF) + vals[2]
            elif op == "v_lshrrev_b32_e32":
                v = (b & U32) >> (a & 31)
            elif op == "v_lshlrev_b32_e32":
                v = b << (a & 31)
            elif op == "v_and_b32_e32":
                v = a & b
            elif op == "v_bfe_u32":
                v = ((a & U32) >> (b & 31)) & ((1 << (vals[2] & 31)) - 1)
            elif op in ("v_cndmask_b32_e64", "v_cndmask_b32_e32"):
                mask_token = src[2]
                mask = self.scalar(mask_token)
                v = b if mask & (1 << lane) else a
            elif op in float_ops:

                def fp(x):
                    return x if isinstance(x, str) else f"asfloat(0x{x & U32:08x}u)"

                x, y = fp(a), fp(b)
                if op.startswith("v_mul_legacy_f32_"):
                    expr = f"(({x} == 0 || {y} == 0) ? 0.0f : {x} * {y})"
                elif op == "v_fma_f32":
                    expr = f"mad({x}, {y}, {fp(vals[2])})"
                elif op.startswith("v_fmac_f32_"):
                    expr = f"mad({x}, {y}, {fp(self.lane_value(d, lane))})"
                else:
                    expr = f"({x} + {y})"
                if expr not in expressions:
                    expressions[expr] = f"f{len(expressions)}"
                v = expressions[expr]
            else:
                raise ValueError(f"Unsupported lane instruction {op}")
            if (lane, d) in self.pending_v:
                raise ValueError("Pending vector destination")
            self.v[lane][int(d[1:])] = v & U32 if isinstance(v, int) else v

    def run(self, decoded, expressions):
        jumps = {offset: i for i, (offset, _) in enumerate(decoded)}
        pc = 0
        while pc < len(decoded):
            offset, line = decoded[pc]
            pc += 1
            op, _, tail = line.partition(" ")
            args = tail.split(", ")
            if op == "s_endpgm":
                return
            if op in ("s_inst_prefetch", "s_nop"):
                continue
            if op == "s_barrier":
                self.barriers.append(offset)
            elif op == "s_waitcnt":
                if not re.fullmatch(
                    r"(?:vmcnt\(0\)|expcnt\(0\)|lgkmcnt\(0\))(?: (?:vmcnt\(0\)|expcnt\(0\)|lgkmcnt\(0\)))*",
                    tail,
                ):
                    raise ValueError("Unsupported wait")
                if "lgkmcnt" in tail:
                    self.pending.clear()
                if "vmcnt" in tail:
                    self.pending_v.clear()
            elif op == "s_branch" or op.startswith("s_cbranch_"):
                if (
                    op == "s_cbranch_scc0"
                    and self.scc is None
                    or op == "s_cbranch_vccnz"
                    and self.vcc is None
                ):
                    raise ValueError("Undefined branch predicate")
                take = {
                    "s_branch": True,
                    "s_cbranch_scc0": not self.scc,
                    "s_cbranch_execz": not self.exec,
                    "s_cbranch_vccnz": bool(self.vcc),
                }[op]
                target = jumps[offset + 4 + 4 * int(tail)]
                if target < pc:
                    raise ValueError("Expected a forward mesh branch")
                if take:
                    pc = target
            elif op == "s_sendmsg" and tail == "sendmsg(MSG_GS_ALLOC_REQ)":
                if self.exec != 1 or self.wave != 0 or self.m0 is None:
                    raise ValueError("Unverified allocation")
                self.counts.append((self.m0 & 0xFFF, self.m0 >> 12))
            elif op == "buffer_load_dwordx3":
                match = re.fullmatch(
                    r"(v\[\d+:\d+\]), (v\d+), (s\[\d+:\d+\]), 0 offen", tail
                )
                if (
                    self.vertex_header is None
                    or not match
                    or len(span(match[1])) != 3
                    or [self.scalar(f"s{i}") for i in span(match[3])]
                    != VERTEX_DESCRIPTOR
                ):
                    raise ValueError("Unknown three-word vertex fetch")
                for lane in range(64):
                    if self.exec & (1 << lane):
                        address = self.lane_value(match[2], lane)
                        if (
                            type(address) != int
                            or address % 4
                            or not 4 <= address <= VERTEX_BYTES - 12
                        ):
                            raise ValueError(
                                "Three-word vertex fetch is outside the bound payload"
                            )
                        for j, reg in enumerate(span(match[1])):
                            key = (lane, f"v{reg}")
                            if key in self.pending_v:
                                raise ValueError(
                                    "Vertex load overwrites an outstanding result"
                                )
                            self.v[lane][reg] = (
                                f"asfloat(vertexData.Load({address + 4 * j}))"
                            )
                            self.pending_v.add(key)
            elif op == "buffer_load_dword":
                match = re.fullmatch(r"(v\d+), (v\d+), (s\[\d+:\d+\]), 0 offen", tail)
                if not match or [self.scalar(f"s{i}") for i in span(match[3])] != [
                    (32, j) for j in range(4)
                ]:
                    raise ValueError("Unknown geometry fetch")
                for lane in range(64):
                    if self.exec & (1 << lane):
                        addr = self.lane_value(match[2], lane)
                        if (lane, match[1]) in self.pending_v:
                            raise ValueError("Vector load overwrites pending result")
                        if addr % 4 or not 0 <= addr // 4 < len(self.geometry):
                            raise ValueError("Geometry address out of range")
                        self.v[lane][int(match[1][1:])] = self.geometry[addr // 4]
                        self.pending_v.add((lane, match[1]))
            elif op == "exp":
                m = re.fullmatch(r"(prim|pos0|param0),? (.+?)( done)?", tail)
                if not m:
                    raise ValueError("Unsupported mesh export")
                if not m[3] and m[1] != "param0":
                    raise ValueError("Expected completed primitive or position export")
                operands = m[2].split(", ")
                for lane in range(64):
                    if not self.exec & (1 << lane):
                        continue
                    i = self.wave * 64 + lane
                    if m[1] == "prim" and operands[1:] == ["off"] * 3:
                        packed = self.lane_value(operands[0], lane)
                        if packed & 0xC0000000:
                            raise ValueError("Unsupported primitive flags")
                        if i in self.indices:
                            raise ValueError("Duplicate primitive")
                        self.indices[i] = [(packed >> b) & 1023 for b in (0, 10, 20)]
                    elif m[1] == "pos0" and len(operands) == 4:
                        if i in self.positions:
                            raise ValueError("Duplicate position")
                        self.positions[i] = [self.lane_value(t, lane) for t in operands]
                    elif m[1] == "param0" and len(operands) == 4 and all(re.fullmatch(r"v\d+", t) for t in operands):
                        if i in self.attributes:
                            raise ValueError("Duplicate vertex attribute")
                        self.attributes[i] = [self.lane_value(t, lane) for t in operands]
                    else:
                        raise ValueError("Unsupported live export")
            elif op.startswith("s_"):
                self.scalar_op(op, args)
            elif op.startswith("v_"):
                self.vector_op(op, args, expressions)
            else:
                raise ValueError(f"Unsupported mesh instruction {op}")
        raise ValueError("No program exit")


def float4(values):
    components = ", ".join(v if isinstance(v, str) else f"asfloat(0x{v:08x}u)" for v in values)
    return f"float4({components})"


def emit_compute_draw(bindings, expressions, positions, indices, attributes):
    # Match the native trace layout, with optional PARAM0 data after its 80 bytes.
    # The vertex shader follows the generated triangle indices using SV_VertexID.
    lines = [
        *bindings,
        "RWByteAddressBuffer geometryOut : register(u0);",
        "ByteAddressBuffer geometryIn : register(t1);",
        "[numthreads(3,1,1)] void main(uint id:SV_GroupIndex) {",
        *(f"  precise float {name} = {expr};" for expr, name in expressions.items()),
    ]
    for i, values in positions.items():
        lines.append(f"  if (id == {i}) geometryOut.Store4({16 + i * 16}, asuint({float4(values)}));")
    for i, values in attributes.items():
        lines.append(f"  if (id == {i}) geometryOut.Store4({80 + i * 16}, asuint({float4(values)}));")
    lines.extend([
        "  if (!id) {",
        "    geometryOut.Store2(0, uint2(3,1));",
        f"    geometryOut.Store3(64, uint3({','.join(map(str, indices[0]))}));",
        "  }",
        "}",
        # The translated pixel input expects TEXCOORD0 in register zero.
        "struct Vertex {" + ("float4 attr:TEXCOORD0;" if attributes else "") + "float4 position:SV_Position;};",
        "Vertex vertex_main(uint id:SV_VertexID) {",
        "  uint index = geometryIn.Load(64 + id * 4);",
        "  Vertex v;",
        "  v.position = asfloat(geometryIn.Load4(16 + index * 16));",
    ])
    if attributes:
        lines.append("  v.attr = asfloat(geometryIn.Load4(80 + index * 16));")
    lines.extend(["  return v;", "}"])
    return "\n".join(lines) + "\n"


def lower(decoded, geometry, limits, vertex_header=None, *, compute_fallback=False):
    if vertex_header is not None and (
        type(vertex_header) != int or not 0 <= vertex_header <= U32
    ):
        raise ValueError("Invalid specialized vertex header")
    if (
        not isinstance(geometry, list)
        or not geometry
        or any(type(n) is not int or not 0 <= n <= U32 for n in geometry)
    ):
        raise ValueError("Geometry binding requires unsigned words")
    waves = [Wave(i, geometry, vertex_header) for i in range(2)]
    expressions = {}
    for wave in waves:
        wave.run(decoded, expressions)
    if (
        waves[0].barriers != waves[1].barriers
        or len(waves[0].counts) != 1
        or waves[1].counts
    ):
        raise ValueError("Inconsistent mesh synchronization or allocation")
    nv, np = waves[0].counts[0]
    pos = waves[0].positions | waves[1].positions
    attributes = waves[0].attributes | waves[1].attributes
    if attributes and attributes.keys() != pos.keys():
        raise ValueError("Incomplete vertex attribute")
    idx = waves[0].indices | waves[1].indices
    if (nv, np) != (3, 1):
        raise ValueError("Expected a single three-vertex triangle")
    if (
        not 0 < nv <= limits[0]
        or not 0 < np <= limits[1]
        or set(pos) != set(range(nv))
        or set(idx) != set(range(np))
        or any(max(tri) >= nv for tri in idx.values())
    ):
        raise ValueError("Conflicting mesh counts, vertices or primitive indices")
    bindings = [
        "cbuffer Points : register(b0) { float4 pointData[32]; };",
        "cbuffer Matrices : register(b1) { float4 matrixData[32]; };",
    ]
    if vertex_header is not None:
        bindings.insert(0, "ByteAddressBuffer vertexData : register(t0);")
    if compute_fallback:
        return emit_compute_draw(bindings, expressions, pos, idx, attributes)
    lines = [
        *bindings,
        "#ifdef TRACE_EXPORTS",
        "RWByteAddressBuffer trace : register(u0);",
        "#endif",
        "struct Vertex {float4 position:SV_Position;" + ("float4 attr:TEXCOORD0;" if attributes else "") + "};",
        '[outputtopology("triangle")]',
        f"[numthreads(128,1,1)] void main(uint id:SV_GroupIndex, out vertices Vertex v[{nv}], out indices uint3 p[{np}]) {{",
        f"  SetMeshOutputCounts({nv},{np});",
    ]
    for expr, name in expressions.items():
        lines.append(f"  precise float {name} = {expr};")
    for i, values in pos.items():
        lines.extend(
            [
                f"  if(id=={i}) {{ float4 q={float4(values)}; v[{i}].position=q;",
                "#ifdef TRACE_EXPORTS",
                f"    trace.Store4({16 + i * 16},asuint(q));",
                "#endif",
                "  }",
            ]
        )
        if attributes:
            lines.append(f"  if(id=={i}) v[{i}].attr={float4(attributes[i])};")
    for i, tri in idx.items():
        lines.extend(
            [
                f"  if(id=={i}) {{ uint3 t=uint3({','.join(map(str, tri))}); p[{i}]=t;",
                "#ifdef TRACE_EXPORTS",
                f"    trace.Store3({16 + nv * 16 + i * 12},t);",
                "#endif",
                "  }",
            ]
        )
    lines.extend(
        [
            "#ifdef TRACE_EXPORTS",
            f"  if (!id) trace.Store2(0,uint2({nv},{np}));",
            "#endif",
            "}",
        ]
    )
    return "\n".join(lines) + "\n"


def translate(blob, geometry, library, vertex_header=None, *, compute_fallback=False):
    chunks = parts(blob)
    p = chunks[b"PSV0"]
    if (
        len(p) < 52
        or not 48 <= struct.unpack_from("<I", p)[0] <= len(p) - 4
        or p[28] != 13
        or p[31] != 2
        or struct.unpack_from("<3I", p, 40) != (128, 1, 1)
    ):
        raise ValueError("Expected triangle mesh, 128 threads")
    limits = struct.unpack_from("<HH", p, 16)
    if any(not 0 < n <= 256 for n in limits):
        raise ValueError("Invalid mesh declaration")
    code = container_code(chunks[b"DXIL"], stage=13)
    return lower(
        disassemble(
            code, library, offsets=True, features=b"+wavefrontsize64,-wavefrontsize32"
        ),
        geometry,
        limits,
        vertex_header,
        compute_fallback=compute_fallback,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("container", type=Path)
    p.add_argument(
        "--vertex-header",
        type=lambda s: int(s, 0),
        help="Specialize the indexed geometry ABI using this first word of the 512-byte vertex SRV",
    )
    p.add_argument("--geometry-words", required=True, type=Path)
    p.add_argument("--llvm", required=True)
    p.add_argument(
        "--compute-fallback",
        action="store_true",
        help="Emit SM 6.0 compute (main) and vertex (vertex_main) entry points",
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.suffix.lower() != ".hlsl" or a.output.resolve() in (
        a.container.resolve(),
        a.geometry_words.resolve(),
    ):
        p.error("Expected a separate HLSL output")
    src = translate(
        a.container.read_bytes(),
        json.loads(a.geometry_words.read_text()),
        a.llvm,
        a.vertex_header,
        compute_fallback=a.compute_fallback,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(src, encoding="ascii")
