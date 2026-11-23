"""Automated checks with source-owned inputs for RDNA 2 disassembly, extraction, and scanning tools."""
import argparse
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from disasm_rdna2 import disasm_rdna2, decode_instruction, disassemble_binary
from extract_shaders import extract_shaders, extract_native_rdna2_code, sanitize_filename
from pixel_aot import VOP3_INT16, VOP3_INT16_TERNARY, fma_mix_operand, lower, native_program, parse_fma_mix
from scan_unsupported_shaders import (
    analyze_shader_instructions,
    is_instruction_supported,
    is_opcode_recognized,
    classify_isa_category,
    generate_reports,
    scan_directory_or_files,
    detect_shader_stage,
    generate_progress_dashboard,
    disassemble_binary_shader,
    make_failed_shader_record,
    KNOWN_RDNA2_BASE_OPCODES,
)
try:
    from pixel_aot import disassemble as llvm_disassemble
except ImportError:
    llvm_disassemble = None


def resolve_llvm_lib(name_or_path: str) -> str:
    """Resolve an LLVM shared library name or path for ctypes.CDLL loading."""
    p = Path(name_or_path)
    if p.is_file():
        return str(p.resolve())
    import ctypes.util
    found = ctypes.util.find_library(name_or_path)
    if found and Path(found).is_file():
        return str(Path(found).resolve())
    for directory in [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib",
        "/usr/local/lib",
    ]:
        candidate = Path(directory) / name_or_path
        if candidate.is_file():
            return str(candidate.resolve())
    return name_or_path


def resolve_tool(env_var: str, default_names: list[str]) -> str | None:
    """Resolve a command-line tool path via environment variable or system search."""
    val = os.environ.get(env_var)
    if val:
        w = shutil.which(val)
        if w:
            return w
        p = Path(val)
        if p.is_file():
            return str(p.resolve())
    for name in default_names:
        w = shutil.which(name)
        if w:
            return w
    return None


def assemble_with_llvm_mc(instructions: list[str], assembler: str, objcopy: str) -> bytes:
    """Assemble AMDGPU instructions for gfx1013 using llvm-mc and extract .text bytes."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        src = tmp_dir / "fixture.s"
        obj = tmp_dir / "fixture.o"
        binary = tmp_dir / "fixture.bin"
        src.write_text("\n".join(instructions) + "\n", encoding="ascii")
        subprocess.run(
            [assembler, "-triple=amdgcn--amdpal", "-mcpu=gfx1013", "-filetype=obj", str(src), "-o", str(obj)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [objcopy, "-O", "binary", "--only-section=.text", str(obj), str(binary)],
            check=True,
            capture_output=True,
        )
        return binary.read_bytes()


def envelope(code: bytes, name: bytes = b"fixture_ps") -> bytes:
    """Create a source-owned DXBC container wrapping native GFX1013 machine code."""
    program_start = 256
    image = bytearray(program_start) + code
    image[:4] = b"FSSM"
    struct.pack_into("<I", image, 192, program_start - 40)
    native = b"XBCH" + bytes(12) + struct.pack("<II", len(image), 0) + zlib.compress(image)
    program = (bytes(24) + b"XDAT" + struct.pack("<II", 1, 0)
               + struct.pack("<III", 13, 48, len(native)) + native)
    blob = (b"DXBC" + bytes(16) + struct.pack("<4I", 1, 44 + len(program), 1, 36)
            + b"DXIL" + struct.pack("<I", len(program)) + program)
    encoder = zlib.compressobj(wbits=-15)
    payload = encoder.compress(blob) + encoder.flush()
    return (bytes([len(name) + 1]) + name + b"\0\0"
            + struct.pack("<4I", len(payload), 0, len(payload), len(blob)) + payload)


class TestRdna2Disasm(unittest.TestCase):
    """Test pure-Python RDNA 2 disassembler and verify against golden vectors and LLVM."""

    def test_golden_vectors(self):
        vectors = [
            # (hex_bytes, expected_opcode, expected_mnemonic)
            ("000080bf", "s_nop", "s_nop 0"),
            ("000081bf", "s_endpgm", "s_endpgm"),
            ("000082bf", "s_branch", "s_branch label_0x4"),
            ("7f008cbf", "s_waitcnt", "s_waitcnt vmcnt(15) lgkmcnt(0)"),
            ("000382be", "s_mov_b32", "s_mov_b32 s2, s0"),
            ("000482be", "s_mov_b64", "s_mov_b64 s[2:3], s[0:1]"),
            ("000882be", "s_not_b64", "s_not_b64 s[2:3], s[0:1]"),
            ("003c82be", "s_and_saveexec_b32", "s_and_saveexec_b32 s2, s0"),
            ("02048087", "s_and_b64", "s_and_b64 s[0:1], s[2:3], s[4:5]"),
            ("0204808f", "s_lshl_b64", "s_lshl_b64 s[0:1], s[2:3], s4"),
            ("7e04b2be", "s_mov_b64", "s_mov_b64 s[50:51], exec"),
            ("7e0afebe", "s_wqm_b64", "s_wqm_b64 exec, exec"),
            ("7e32fe87", "s_and_b64", "s_and_b64 exec, exec, s[50:51]"),
            ("00008080", "s_sub_u32", "s_sub_u32 s0, s0, s0"),
            # GFX10 scalar operand codes: 108-123 are ttmp0-15, 125 is null (SOP2 ssrc1)
            ("007d8080", "s_sub_u32", "s_sub_u32 s0, s0, null"),
            ("006c8080", "s_sub_u32", "s_sub_u32 s0, s0, ttmp0"),
            # GFX10 / RDNA 2 opcode numbers, encodings as in LLVM's gfx10 MC tests
            ("01020080", "s_add_u32", "s_add_u32 s0, s1, s2"),
            ("01028080", "s_sub_u32", "s_sub_u32 s0, s1, s2"),
            ("01020081", "s_add_i32", "s_add_i32 s0, s1, s2"),
            ("01028081", "s_sub_i32", "s_sub_i32 s0, s1, s2"),
            ("01020083", "s_min_i32", "s_min_i32 s0, s1, s2"),
            ("01028083", "s_min_u32", "s_min_u32 s0, s1, s2"),
            ("01020084", "s_max_i32", "s_max_i32 s0, s1, s2"),
            ("01028084", "s_max_u32", "s_max_u32 s0, s1, s2"),
            ("01020085", "s_cselect_b32", "s_cselect_b32 s0, s1, s2"),
            ("01020087", "s_and_b32", "s_and_b32 s0, s1, s2"),
            ("01020088", "s_or_b32", "s_or_b32 s0, s1, s2"),
            ("01020089", "s_xor_b32", "s_xor_b32 s0, s1, s2"),
            ("0102008a", "s_andn2_b32", "s_andn2_b32 s0, s1, s2"),
            ("0102008f", "s_lshl_b32", "s_lshl_b32 s0, s1, s2"),
            ("01020090", "s_lshr_b32", "s_lshr_b32 s0, s1, s2"),
            ("01020091", "s_ashr_i32", "s_ashr_i32 s0, s1, s2"),
            ("01020093", "s_mul_i32", "s_mul_i32 s0, s1, s2"),
            ("341281b1", "s_cmpk_eq_i32", "s_cmpk_eq_i32 s1, 0x1234"),
            # SOPC: [31:23] == 0b101111110. These encodings were previously
            # consumed by the SOPK prefix check (e.g. 000100bf as s_sopk_1e).
            ("000100bf", "s_cmp_eq_i32", "s_cmp_eq_i32 s0, s1"),
            ("000106bf", "s_cmp_eq_u32", "s_cmp_eq_u32 s0, s1"),
            ("020302bf", "s_cmp_gt_i32", "s_cmp_gt_i32 s2, s3"),
            ("007d00bf", "s_cmp_eq_i32", "s_cmp_eq_i32 s0, null"),
            ("00010cbf", "s_bitcmp0_b32", "s_bitcmp0_b32 s0, s1"),
            ("000212bf", "s_cmp_eq_u64", "s_cmp_eq_u64 s[0:1], s[2:3]"),
            ("04060ebf", "s_bitcmp0_b64", "s_bitcmp0_b64 s[4:5], s6"),
            ("000080bb", "s_waitcnt_vscnt", "s_waitcnt_vscnt s0, 0x0"),
            ("0000fdbc", "s_waitcnt_expcnt", "s_waitcnt_expcnt null, 0x0"),
            ("0300fdbc", "s_waitcnt_expcnt", "s_waitcnt_expcnt null, 0x3"),
            ("0100a0bf", "s_inst_prefetch", "s_inst_prefetch 0x1"),
            ("0255027e", "v_rcp_f32_e32", "v_rcp_f32_e32 v1, v2"),
            ("010d007e", "v_cvt_f32_u32_e32", "v_cvt_f32_u32_e32 v0, v1"),
            ("0000027e", "v_nop", "v_nop"),
            # VOP1 9-bit SRC0: 0..255 scalar/constant, 256..511 VGPR
            ("0103007e", "v_mov_b32_e32", "v_mov_b32_e32 v0, v1"),
            ("0102007e", "v_mov_b32_e32", "v_mov_b32_e32 v0, s1"),
            ("f202047e", "v_mov_b32_e32", "v_mov_b32_e32 v2, 1.0"),
            ("f002067e", "v_mov_b32_e32", "v_mov_b32_e32 v3, 0.5"),
            ("8002087e", "v_mov_b32_e32", "v_mov_b32_e32 v4, 0"),
            # Special source operands 235-239 and 251-254, and the 1/(2*pi) inline
            # constant 248. Encodings and spellings from llvm-mc -mcpu=gfx1013.
            ("eb02007e", "v_mov_b32_e32", "v_mov_b32_e32 v0, src_shared_base"),
            ("ec02007e", "v_mov_b32_e32", "v_mov_b32_e32 v0, src_shared_limit"),
            ("ed02007e", "v_mov_b32_e32", "v_mov_b32_e32 v0, src_private_base"),
            ("ee02007e", "v_mov_b32_e32", "v_mov_b32_e32 v0, src_private_limit"),
            ("ef02007e", "v_mov_b32_e32", "v_mov_b32_e32 v0, src_pops_exiting_wave_id"),
            ("fb02007e", "v_mov_b32_e32", "v_mov_b32_e32 v0, src_vccz"),
            ("fc02007e", "v_mov_b32_e32", "v_mov_b32_e32 v0, src_execz"),
            ("fd02007e", "v_mov_b32_e32", "v_mov_b32_e32 v0, src_scc"),
            ("fe02007e", "v_mov_b32_e32", "v_mov_b32_e32 v0, src_lds_direct"),
            ("f8040606", "v_add_f32_e32", "v_add_f32_e32 v3, 0.15915494, v2"),
            ("fd0382be", "s_mov_b32", "s_mov_b32 s2, src_scc"),
            ("eb0482be", "s_mov_b64", "s_mov_b64 s[2:3], src_shared_base"),
            ("f80482be", "s_mov_b64", "s_mov_b64 s[2:3], 0.15915494309189532"),
            ("000064d5f8040200", "v_add_f64", "v_add_f64 v[0:1], 0.15915494309189532, v[2:3]"),
            ("000064d5ed040200", "v_add_f64", "v_add_f64 v[0:1], src_private_base, v[2:3]"),
            ("0105007e", "v_readfirstlane_b32", "v_readfirstlane_b32 s0, v1"),
            ("0c05587e", "v_readfirstlane_b32", "v_readfirstlane_b32 s44, v12"),
            # VOP2 9-bit SRC0 (VGPR vs SGPR vs inline constant) with GFX10/RDNA 2 opcodes
            ("01050602", "v_cndmask_b32_e32", "v_cndmask_b32_e32 v3, v1, v2, vcc_lo"),
            ("01050606", "v_add_f32_e32", "v_add_f32_e32 v3, v1, v2"),
            ("01040606", "v_add_f32_e32", "v_add_f32_e32 v3, s1, v2"),
            ("f0040606", "v_add_f32_e32", "v_add_f32_e32 v3, 0.5, v2"),
            ("01050610", "v_mul_f32_e32", "v_mul_f32_e32 v3, v1, v2"),
            ("0105062c", "v_lshrrev_b32_e32", "v_lshrrev_b32_e32 v3, v1, v2"),
            ("01050630", "v_ashrrev_i32_e32", "v_ashrrev_i32_e32 v3, v1, v2"),
            ("01050634", "v_lshlrev_b32_e32", "v_lshlrev_b32_e32 v3, v1, v2"),
            ("01050636", "v_and_b32_e32", "v_and_b32_e32 v3, v1, v2"),
            ("01050638", "v_or_b32_e32", "v_or_b32_e32 v3, v1, v2"),
            ("0105063a", "v_xor_b32_e32", "v_xor_b32_e32 v3, v1, v2"),
            ("0105064a", "v_add_nc_u32_e32", "v_add_nc_u32_e32 v3, v1, v2"),
            ("0105063e", "v_mac_f32_e32", "v_mac_f32_e32 v3, v1, v2"),
            ("01050656", "v_fmac_f32_e32", "v_fmac_f32_e32 v3, v1, v2"),
            # VOP1 rounding and byte conversion ops (GFX10 / RDNA 2)
            ("01230a7e", "v_cvt_f32_ubyte0_e32", "v_cvt_f32_ubyte0_e32 v5, v1"),
            ("01250a7e", "v_cvt_f32_ubyte1_e32", "v_cvt_f32_ubyte1_e32 v5, v1"),
            ("01270a7e", "v_cvt_f32_ubyte2_e32", "v_cvt_f32_ubyte2_e32 v5, v1"),
            ("01290a7e", "v_cvt_f32_ubyte3_e32", "v_cvt_f32_ubyte3_e32 v5, v1"),
            ("01410a7e", "v_fract_f32_e32", "v_fract_f32_e32 v5, v1"),
            ("01430a7e", "v_trunc_f32_e32", "v_trunc_f32_e32 v5, v1"),
            ("01450a7e", "v_ceil_f32_e32", "v_ceil_f32_e32 v5, v1"),
            ("01470a7e", "v_rndne_f32_e32", "v_rndne_f32_e32 v5, v1"),
            ("01490a7e", "v_floor_f32_e32", "v_floor_f32_e32 v5, v1"),
            # VOP2 mandatory literal FMA forms
            ("01070a5841312111", "v_fmamk_f32", "v_fmamk_f32 v5, v1, 0x11213141, v3"),
            ("01050a5a41312111", "v_fmaak_f32", "v_fmaak_f32 v5, v1, v2, 0x11213141"),
            # Non-fused literal forms (encodings from LLVM 15 llvm-mc -mcpu=gfx1013 -show-encoding)
            ("01070a4041312111", "v_madmk_f32", "v_madmk_f32 v5, v1, 0x11213141, v3"),
            ("01050a4241312111", "v_madak_f32", "v_madak_f32 v5, v1, v2, 0x11213141"),
            # VINTRP: attr is [15:10], chan is [9:8]
            ("000004c8", "v_interp_p1_f32_e32", "v_interp_p1_f32_e32 v1, v0, attr0.x"),
            ("000108c8", "v_interp_p1_f32_e32", "v_interp_p1_f32_e32 v2, v0, attr0.y"),
            ("00020cc8", "v_interp_p1_f32_e32", "v_interp_p1_f32_e32 v3, v0, attr0.z"),
            ("000310c8", "v_interp_p1_f32_e32", "v_interp_p1_f32_e32 v4, v0, attr0.w"),
            ("01050dc8", "v_interp_p2_f32_e32", "v_interp_p2_f32_e32 v3, v1, attr1.y"),
            # EXP: en=[3:0], target=[9:4], compr=10, done=11, vm=12
            ("000000f800000000", "exp", "exp mrt0 off, off, off, off"),
            ("010800f804000000", "exp", "exp mrt0 v4, off, off, off done"),
            ("8f0800f804030201", "exp", "exp mrtz v4, v3, v2, v1 done"),
            ("cf0000f804030201", "exp", "exp pos0 v4, v3, v2, v1"),
            ("0f1c00f810110000", "exp", "exp mrt0 v16, v16, v17, v17 done compr vm"),
            # SMEM: register widths per op and soffset vs immediate offset
            ("01050cf4000000fa", "s_load_dwordx8", "s_load_dwordx8 s[20:27], s[2:3], 0x0"),
            ("01050cf445230100", "s_load_dwordx8", "s_load_dwordx8 s[20:27], s[2:3], s0 offset:0x12345"),
            ("420120f400000000", "s_buffer_load_dword", "s_buffer_load_dword s5, s[4:7], s0"),
            ("020528f4200000fa", "s_buffer_load_dwordx4", "s_buffer_load_dwordx4 s[20:23], s[4:7], 0x20"),
            # MIMG: contiguous and NSA encodings
            ("001f00f000000000", "image_load", "image_load v[0:3], v0, s[0:7] dmask:0xf dim:SQ_RSRC_IMG_1D unorm"),
            ("080100f002000000", "image_load", "image_load v0, v[2:3], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D"),
            ("000780f020402103", "image_sample", "image_sample v[64:66], v32, s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_1D"),
            ("220790f02040210310080000", "image_sample_l", "image_sample_l v[64:66], [v32, v16, v8], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_1D_ARRAY"),
            ("220794f02040210310080000", "image_sample_b", "image_sample_b v[64:66], [v32, v16, v8], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_1D_ARRAY"),
            ("22079cf02040210310000000", "image_sample_lz", "image_sample_lz v[64:66], [v32, v16], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_1D_ARRAY"),
            # GFX10 VOPC vector comparisons
            ("0105027c", "v_cmp_lt_f32_e32", "v_cmp_lt_f32_e32 vcc_lo, v1, v2"),
            ("0104027c", "v_cmp_lt_f32_e32", "v_cmp_lt_f32_e32 vcc_lo, s1, v2"),
            ("2510027c", "v_cmp_lt_f32_e32", "v_cmp_lt_f32_e32 vcc_lo, s37, v8"),
            ("ff04027c0000803f", "v_cmp_lt_f32_e32", "v_cmp_lt_f32_e32 vcc_lo, 0x3f800000, v2"),
            ("0105227c", "v_cmpx_lt_f32_e32", "v_cmpx_lt_f32_e32 v1, v2"),
            # GFX10 VOP3A and VOP3B 64-bit vector ALU instructions
            ("000041d501050e04", "v_mad_f32", "v_mad_f32 v0, v1, v2, v3"),
            ("00004bd501050e04", "v_fma_f32", "v_fma_f32 v0, v1, v2, v3"),
            ("0c8008d50ce50100", "v_mul_f32_e64", "v_mul_f32_e64 v12, v12, 1.0 clamp"),
            ("000001d42c240260", "v_cmp_lt_f32_e64", "v_cmp_lt_f32_e64 s0, -s44, -v18"),
            ("0e0001d580e40100", "v_cndmask_b32_e64", "v_cndmask_b32_e64 v14, 0, 1.0, s0"),
            ("01000fd702070200", "v_add_co_u32", "v_add_co_u32 v1, s0, v2, v3"),
            ("05004bd501ff09040000803f", "v_fma_f32", "v_fma_f32 v5, v1, 0x3f800000, v2"),
            ("030103d501050208", "v_add_f32_e64", "v_add_f32_e64 v3, |v1|, v2 mul:2"),
            ("340889d42c240200", "v_cmp_lt_i16_e64", "v_cmp_lt_i16_e64 s52, s44, v18 op_sel:[1,0,0]"),
            ("fe0064d501050202", "v_add_f64", "v_add_f64 v[254:255], v[1:2], v[2:3]"),
            ("0200ffd683080200", "v_lshlrev_b64", "v_lshlrev_b64 v[2:3], 3, v[4:5]"),
            ("000076d502061204", "v_mad_u64_u32", "v_mad_u64_u32 v[0:1], s0, s2, v3, v[4:5]"),
            ("000090d502010000", "v_cvt_f64_f32_e64", "v_cvt_f64_f32_e64 v[0:1], v2"),
            ("00008fd502010000", "v_cvt_f32_f64_e64", "v_cvt_f32_f64_e64 v0, v[2:3]"),
            ("000080d502091a04", "v_nop", "v_nop"),
            ("0000abd502091a04", "v_rcp_iflag_f32_e64", "v_rcp_iflag_f32_e64 v0, v2"),
            ("0000afd502091a04", "v_rcp_f64_e64", "v_rcp_f64_e64 v[0:1], v[2:3]"),
            ("0000b1d502091a04", "v_rsq_f64_e64", "v_rsq_f64_e64 v[0:1], v[2:3]"),
            ("0000b4d502091a04", "v_sqrt_f64_e64", "v_sqrt_f64_e64 v[0:1], v[2:3]"),
            ("0000b9d502091a04", "v_ffbh_u32_e64", "v_ffbh_u32_e64 v0, v2"),
            ("0000bad502091a04", "v_ffbl_b32_e64", "v_ffbl_b32_e64 v0, v2"),
            ("0000bbd502091a04", "v_ffbh_i32_e64", "v_ffbh_i32_e64 v0, v2"),
            ("000040d502091a04", "v_mad_legacy_f32", "v_mad_legacy_f32 v0, v2, v4, v6"),
            ("000050d502091a04", "v_mullit_f32", "v_mullit_f32 v0, v2, v4, v6"),
            ("010028d502060000", "v_add_co_ci_u32_e64", "v_add_co_ci_u32_e64 v1, s0, s2, s3, s0"),
            # GFX10 VOP3P packed float16, integer16, dot-products, and mixed-precision FMA
            ("04400acc03030118", "v_pk_add_u16", "v_pk_add_u16 v4, v3, 1"),
            ("04000acc03030118", "v_pk_add_u16", "v_pk_add_u16 v4, v3, 1"),
            ("1a8012cc18e10118", "v_pk_max_f16", "v_pk_max_f16 v26, v24, 0.5 clamp"),
            ("0c4013cc12275a1c", "v_dot2_f32_f16", "v_dot2_f32_f16 v12, v18, v19, v22"),
            ("0c0020cc2c265204", "v_fma_mix_f32", "v_fma_mix_f32 v12, s44, v19, v20"),
            # MUBUF: untyped, format, d16, atomics, cache invalidation, and buffer modifiers
            ("000030e000050203", "buffer_load_dword", "buffer_load_dword v5, off, s[8:11], s3"),
            ("40d030e000054203", "buffer_load_dword", "buffer_load_dword v5, v0, s[8:11], s3 offen offset:64 glc slc dlc"),
            ("ff3f30e000050280", "buffer_load_dword", "buffer_load_dword v5, v[0:1], s[8:11], 0 idxen offen offset:4095"),
            ("000031e000000203", "buffer_load_dword", "buffer_load_dword off, s[8:11], s3 lds"),
            ("000030e00005827d", "buffer_load_dword", "buffer_load_dword v5, off, s[8:11], null tfe"),
            ("000030e000051b03", "buffer_load_dword", "buffer_load_dword v5, off, ttmp[0:3], s3"),
            ("102038e00008037c", "buffer_load_dwordx4", "buffer_load_dwordx4 v[8:11], v0, s[12:15], m0 idxen offset:16"),
            ("00000ce200050280", "buffer_load_format_d16_xyzw", "buffer_load_format_d16_xyzw v[5:6], off, s[8:11], 0"),
            ("0c1070e000050203", "buffer_store_dword", "buffer_store_dword v5, v0, s[8:11], s3 offen offset:12"),
            ("001078e004000280", "buffer_store_dwordx4", "buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen"),
            ("000010e200050203", "buffer_store_format_d16_x", "buffer_store_format_d16_x v5, off, s[8:11], s3"),
            ("0050c8e000050203", "buffer_atomic_add", "buffer_atomic_add v5, v0, s[8:11], s3 offen glc"),
            ("0010c4e000050203", "buffer_atomic_cmpswap", "buffer_atomic_cmpswap v[5:6], v0, s[8:11], s3 offen"),
            ("005044e100050280", "buffer_atomic_cmpswap_x2", "buffer_atomic_cmpswap_x2 v[5:8], v0, s[8:11], 0 offen glc"),
            ("0000c4e100000000", "buffer_gl0_inv", "buffer_gl0_inv"),
            ("0000c8e100000000", "buffer_gl1_inv", "buffer_gl1_inv"),
            # MTBUF: typed buffer loads/stores and format field decoding
            ("0000b0e800050203", "tbuffer_load_format_x", "tbuffer_load_format_x v5, off, s[8:11], s3 format:[BUF_FMT_32_FLOAT]"),
            ("40d06bea00054203", "tbuffer_load_format_xyzw", "tbuffer_load_format_xyzw v[5:8], v0, s[8:11], s3 format:[BUF_FMT_32_32_32_32_FLOAT] offen offset:64 glc slc dlc"),
            ("0030ede800052280", "tbuffer_store_format_d16_xy", "tbuffer_store_format_d16_xy v5, v[0:1], s[8:11], 0 format:[BUF_FMT_16_16_FLOAT] idxen offen"),
            ("000008e800050203", "tbuffer_load_format_x", "tbuffer_load_format_x v5, off, s[8:11], s3"),
            ("0000f8eb00050203", "tbuffer_load_format_x", "tbuffer_load_format_x v5, off, s[8:11], s3 format:127"),
            # FLAT, GLOBAL, and SCRATCH memory instructions
            ("000030dc02007d04", "flat_load_dword", "flat_load_dword v4, v[2:3]"),
            ("ff173bdc02007d04", "flat_load_dwordx4", "flat_load_dwordx4 v[4:7], v[2:3] offset:2047 glc slc dlc"),
            ("100088dc02007d04", "flat_load_sbyte_d16", "flat_load_sbyte_d16 v4, v[2:3] offset:16"),
            ("100070dc02087d00", "flat_store_dword", "flat_store_dword v[2:3], v8 offset:16"),
            ("00007cdc02087d00", "flat_store_dwordx3", "flat_store_dwordx3 v[2:3], v[8:10]"),
            ("0000c8dc02087d00", "flat_atomic_add", "flat_atomic_add v[2:3], v8"),
            ("0000c9dc02087d04", "flat_atomic_add", "flat_atomic_add v4, v[2:3], v8 glc"),
            ("2000c5dc02087d04", "flat_atomic_cmpswap", "flat_atomic_cmpswap v4, v[2:3], v[8:9] offset:32 glc"),
            ("000045dd02087d04", "flat_atomic_cmpswap_x2", "flat_atomic_cmpswap_x2 v[4:5], v[2:3], v[8:11] glc"),
            ("008030dc02007d04", "global_load_dword", "global_load_dword v4, v[2:3], off"),
            ("f09f33dc01000204", "global_load_dword", "global_load_dword v4, v1, s[2:3] offset:-16 glc slc dlc"),
            ("00a030dc02007d00", "global_load_dword", "global_load_dword v[2:3], off lds"),
            ("00a030dc01000200", "global_load_dword", "global_load_dword v1, s[2:3] lds"),
            ("ff8770dc02087d00", "global_store_dword", "global_store_dword v[2:3], v8, off offset:2047"),
            ("008864dc01080200", "global_store_byte_d16_hi", "global_store_byte_d16_hi v1, v8, s[2:3] offset:-2048"),
            ("0080c8dc01080200", "global_atomic_add", "global_atomic_add v1, v8, s[2:3]"),
            ("0080c9dc01080204", "global_atomic_add", "global_atomic_add v4, v1, v8, s[2:3] glc"),
            ("008045dd02087d04", "global_atomic_cmpswap_x2", "global_atomic_cmpswap_x2 v[4:5], v[2:3], v[8:11], off glc"),
            ("104030dc01007d04", "scratch_load_dword", "scratch_load_dword v4, v1, off offset:16"),
            ("f05f33dc00000204", "scratch_load_dword", "scratch_load_dword v4, off, s2 offset:-16 glc slc dlc"),
            ("106030dc00000200", "scratch_load_dword", "scratch_load_dword off, s2 offset:16 lds"),
            ("004020dc00007c04", "scratch_load_ubyte", "scratch_load_ubyte v4, off, m0"),
            ("104070dc01087d00", "scratch_store_dword", "scratch_store_dword v1, v8, off offset:16"),
            ("e04f74dc00080200", "scratch_store_dwordx2", "scratch_store_dwordx2 off, v[8:9], s2 offset:-32"),
            ("000050d800000000", "ds_nop", "ds_nop"),
            ("0000d8d802000018", "ds_read_b32", "ds_read_b32 v24, v2"),
            ("4000d8d802000018", "ds_read_b32", "ds_read_b32 v24, v2 offset:64"),
            ("0400e8d802000018", "ds_read_u8", "ds_read_u8 v24, v2 offset:4"),
            ("1000d8d902000018", "ds_read_b64", "ds_read_b64 v[24:25], v2 offset:16"),
            ("0000fcdb02000018", "ds_read_b128", "ds_read_b128 v[24:27], v2"),
            ("0408dcd802000018", "ds_read2_b32", "ds_read2_b32 v[24:25], v2 offset0:4 offset1:8"),
            ("0102e0d902000018", "ds_read2st64_b64", "ds_read2st64_b64 v[24:27], v2 offset0:1 offset1:2"),
            ("000034d802080000", "ds_write_b32", "ds_write_b32 v2, v8"),
            ("800034d802080000", "ds_write_b32", "ds_write_b32 v2, v8 offset:128"),
            ("040838d802081000", "ds_write2_b32", "ds_write2_b32 v2, v8, v16 offset0:4 offset1:8"),
            ("20007cdb02080000", "ds_write_b128", "ds_write_b128 v2, v[8:11] offset:32"),
            ("080000d802080000", "ds_add_u32", "ds_add_u32 v2, v8 offset:8"),
            ("080082d802080018", "ds_add_rtn_u32", "ds_add_rtn_u32 v24, v2, v8 offset:8 gds"),
            ("100030d802081000", "ds_mskor_b32", "ds_mskor_b32 v2, v8, v16 offset:16"),
            ("0102b8d802081018", "ds_wrxchg2_rtn_b32", "ds_wrxchg2_rtn_b32 v[24:25], v2, v8, v16 offset0:1 offset1:2"),
            ("0000d4d802000018", "ds_swizzle_b32", "ds_swizzle_b32 v24, v2"),
            ("1b80d4d802000018", "ds_swizzle_b32", "ds_swizzle_b32 v24, v2 offset:swizzle(QUAD_PERM,3,2,1,0)"),
            ("1f40d4d802000018", "ds_swizzle_b32", "ds_swizzle_b32 v24, v2 offset:swizzle(SWAP,16)"),
            ("1f1cd4d802000018", "ds_swizzle_b32", "ds_swizzle_b32 v24, v2 offset:swizzle(REVERSE,8)"),
            ("5c00d4d802000018", "ds_swizzle_b32", "ds_swizzle_b32 v24, v2 offset:swizzle(BROADCAST,4,2)"),
            ("0103d4d802000018", "ds_swizzle_b32", 'ds_swizzle_b32 v24, v2 offset:swizzle(BITMASK_PERM,"1100p")'),
            ("0c00fad800000018", "ds_append", "ds_append v24 offset:12 gds"),
            ("0000f4d800000018", "ds_consume", "ds_consume v24"),
            ("0400fed802000018", "ds_ordered_count", "ds_ordered_count v24, v2 offset:4 gds"),
            ("000062d800000000", "ds_gws_sema_release_all", "ds_gws_sema_release_all gds"),
            ("080076d802000000", "ds_gws_barrier", "ds_gws_barrier v2 offset:8 gds"),
            ("0400c8da02080018", "ds_permute_b32", "ds_permute_b32 v24, v2, v8 offset:4"),
            ("0000ccda02080018", "ds_bpermute_b32", "ds_bpermute_b32 v24, v2, v8"),
            ("1000c0da00080000", "ds_write_addtid_b32", "ds_write_addtid_b32 v8 offset:16"),
            ("1000c4da00000018", "ds_read_addtid_b32", "ds_read_addtid_b32 v24 offset:16"),
            ("200054da02000000", "ds_add_src2_f32", "ds_add_src2_f32 v2 offset:32"),
            ("0000f8d902080018", "ds_condxchg32_rtn_b64", "ds_condxchg32_rtn_b64 v[24:25], v2, v[8:9]"),
        ]
        for hex_str, expected_op, expected_asm in vectors:
            raw = bytes.fromhex(hex_str)
            insns = disasm_rdna2(raw)
            self.assertEqual(len(insns), 1, f"Failed for {hex_str}")
            self.assertEqual(insns[0]["opcode"], expected_op, f"Mismatched opcode for {hex_str}")
            self.assertEqual(insns[0]["assembly"], expected_asm, f"Mismatched assembly for {hex_str}")

    def test_compressed_export_keeps_unpaired_mask(self):
        base = bytes.fromhex("0f1c00f810110000")
        operands = {
            0x2: "off, off, off, off",
            0xa: "off, off, off, off",
            0x5: "v16, v16, v17, v17",
        }
        for mask, registers in operands.items():
            with self.subTest(mask=mask):
                raw = bytes([(base[0] & 0xf0) | mask]) + base[1:]
                records = disasm_rdna2(raw)
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["assembly"],
                                 f"exp mrt0 {registers} done compr vm en:{mask:#x}")

    def test_waitcnt_uses_gfx10_counter_fields(self):
        # GFX10 s_waitcnt simm16: vmcnt [3:0] and [15:14], expcnt [6:4], lgkmcnt [13:8].
        vectors = [
            # Printed by LLVM 15.0.7 (MC/Disassembler/AMDGPU/gfx10_dasm_all.txt).
            ("00008cbf", "s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)"),
            ("34128cbf", "s_waitcnt vmcnt(4) expcnt(3) lgkmcnt(18)"),
            ("51c18cbf", "s_waitcnt vmcnt(49) expcnt(5) lgkmcnt(1)"),
            # Encoded by LLVM 15.0.7 (MC/AMDGPU/gfx10_asm_sop.s); bit 7 is unused.
            ("d1c18cbf", "s_waitcnt vmcnt(49) expcnt(5) lgkmcnt(1)"),
            # One counter each; the others at their maximum are omitted.
            ("703f8cbf", "s_waitcnt vmcnt(0)"),
            ("707f8cbf", "s_waitcnt vmcnt(16)"),
            ("7f3f8cbf", "s_waitcnt vmcnt(15)"),
            ("0fff8cbf", "s_waitcnt expcnt(0)"),
            ("7fc08cbf", "s_waitcnt lgkmcnt(0)"),
            ("7fff8cbf", "s_waitcnt vmcnt(63) expcnt(7) lgkmcnt(63)"),
        ]
        for hex_str, expected in vectors:
            with self.subTest(encoding=hex_str):
                records = disasm_rdna2(bytes.fromhex(hex_str + "000081bf"))
                self.assertEqual([(r["opcode"], r["length"]) for r in records],
                                 [("s_waitcnt", 4), ("s_endpgm", 4)])
                self.assertEqual(records[0]["assembly"], expected)
                self.assertTrue(is_instruction_supported(expected))

    def test_literal_operands_keep_stream_aligned(self):
        # SOP1, SOP2, VOP1, VOP2 (src0 == 255) and v_fmaak_f32 each consume a trailing 32-bit literal.
        stream = bytes.fromhex(
            "ff0382be 78563412"  # s_mov_b32 s2, 0x12345678
            "02ff0380 00010000"  # s_add_u32 s3, s2, 0x100
            "ff02087e 89674523"  # v_mov_b32_e32 v4, 0x23456789
            "ff040606 0000803f"  # v_add_f32_e32 v3, 0x3f800000, v2
            "01050a5a 41312111"  # v_fmaak_f32 v5, v1, v2, 0x11213141
            "ff04027c 0000803f"  # v_cmp_lt_f32_e32 vcc_lo, 0x3f800000, v2
            "05004bd5 01ff0904 0000803f"  # v_fma_f32 v5, v1, 0x3f800000, v2
            "000081bf"           # s_endpgm
        )
        insns = disasm_rdna2(stream)
        self.assertEqual(
            [(i["opcode"], i["length"], i["assembly"]) for i in insns],
            [
                ("s_mov_b32", 8, "s_mov_b32 s2, 0x12345678"),
                ("s_add_u32", 8, "s_add_u32 s3, s2, 0x100"),
                ("v_mov_b32_e32", 8, "v_mov_b32_e32 v4, 0x23456789"),
                ("v_add_f32_e32", 8, "v_add_f32_e32 v3, 0x3f800000, v2"),
                ("v_fmaak_f32", 8, "v_fmaak_f32 v5, v1, v2, 0x11213141"),
                ("v_cmp_lt_f32_e32", 8, "v_cmp_lt_f32_e32 vcc_lo, 0x3f800000, v2"),
                ("v_fma_f32", 12, "v_fma_f32 v5, v1, 0x3f800000, v2"),
                ("s_endpgm", 4, "s_endpgm"),
            ],
        )
        # Truncated literals and truncated NSA MIMG must report *_truncated instead of crashing.
        for bad_hex, expected_op in (
            ("ff0382be", "sop1_truncated"),
            ("02ff0380", "sop2_truncated"),
            ("ff02087e", "vop1_truncated"),
            ("01050a5a", "vop2_truncated"),
            ("ff04027c", "vopc_truncated"),
            ("05004bd501ff0904", "vop3_truncated"),
            ("00004bd5", "vop3_truncated"),
            ("220790f020402103", "mimg_truncated"),
        ):
            insns = disasm_rdna2(bytes.fromhex(bad_hex))
            self.assertEqual(len(insns), 1)
            self.assertEqual(insns[0]["opcode"], expected_op)
            self.assertIn("<truncated>", insns[0]["assembly"])

    def test_vop1_readfirstlane_b32_destination_is_an_sgpr(self):
        # Issue #746: in VOP1 the vdst field of v_readfirstlane_b32 names an SGPR
        # even though the mnemonic starts with "v_". LLVM for gfx1013 (triple
        # amdgcn--amdpal) assembles `v_readfirstlane_b32 s2, v0` to this encoding
        # and disassembles those bytes back to `v_readfirstlane_b32 s2, v0`.
        record = disasm_rdna2(bytes.fromhex("0005047e"))[0]
        self.assertEqual(record["opcode"], "v_readfirstlane_b32")
        self.assertEqual(record["length"], 4)
        self.assertEqual(record["assembly"], "v_readfirstlane_b32 s2, v0")

        # The plain register form is the only shape the target assembles:
        # `v_readfirstlane_b32 v2, v0` and `v_readfirstlane_b32 s2, s0` are both
        # rejected with "invalid operand for instruction", so the source stays a
        # VGPR and only the destination class is corrected.

        # Keep a following s_endpgm in both controls.  The VOP1 control checks
        # that the 32-bit instruction still ends at +4.
        control = bytes.fromhex(
            "0005047e"  # v_readfirstlane_b32 s2, v0
            "000081bf"  # s_endpgm
        )
        self.assertEqual(
            [(i["offset"], i["length"], i["opcode"], i["assembly"]) for i in disasm_rdna2(control)],
            [
                (0, 4, "v_readfirstlane_b32", "v_readfirstlane_b32 s2, v0"),
                (4, 4, "s_endpgm", "s_endpgm"),
            ],
        )

        # An intervening 32-bit instruction is an alignment control as well.
        stream = bytes.fromhex(
            "0005047e"  # v_readfirstlane_b32 s2, v0
            "0103007e"  # v_mov_b32_e32 v0, v1
            "000081bf"  # s_endpgm
        )
        self.assertEqual(
            [(i["offset"], i["length"], i["opcode"], i["assembly"]) for i in disasm_rdna2(stream)],
            [
                (0, 4, "v_readfirstlane_b32", "v_readfirstlane_b32 s2, v0"),
                (4, 4, "v_mov_b32_e32", "v_mov_b32_e32 v0, v1"),
                (8, 4, "s_endpgm", "s_endpgm"),
            ],
        )

        # Negative control: vdst == 2 in an ordinary VOP1 is still a VGPR.
        self.assertEqual(
            disasm_rdna2(bytes.fromhex("0103047e"))[0]["assembly"],
            "v_mov_b32_e32 v2, v1",
        )

        # VOP3A control: gfx1013 does not assemble a VOP3A shape of this
        # operation (`v_readfirstlane_b32_e64 s2, v0` fails with "e64 variant of
        # this instruction is not supported") and llvm-mc reports the issue's
        # bytes 020082d500010000 as an invalid instruction encoding.  The record
        # keeps a distinct unsupported-encoding diagnostic, consumes exactly
        # eight bytes, is neither recognized nor supported by the scanner (so a
        # lowerer does not receive an instruction), and the following s_endpgm
        # still decodes at +8.
        vop3a = disasm_rdna2(bytes.fromhex(
            "020082d500010000"  # proposed VOP3A word from issue #746 (invalid on gfx1013)
            "000081bf"          # s_endpgm
        ))
        self.assertEqual(len(vop3a), 2)
        self.assertEqual((vop3a[0]["offset"], vop3a[0]["length"]), (0, 8))
        self.assertEqual(vop3a[0]["opcode"], "v_readfirstlane_b32_unsupported")
        self.assertEqual(vop3a[0]["assembly"],
                         "v_readfirstlane_b32 <unsupported VOP3A encoding>")
        self.assertNotEqual(vop3a[0]["opcode"], "v_readfirstlane_b32")
        self.assertFalse(is_opcode_recognized(vop3a[0]["opcode"]))
        self.assertFalse(is_instruction_supported(vop3a[0]["assembly"]))
        self.assertEqual(
            (vop3a[1]["offset"], vop3a[1]["length"], vop3a[1]["opcode"]),
            (8, 4, "s_endpgm"),
        )

    def test_sopc_literal_keeps_the_following_instruction_aligned(self):
        # src1 == literal (255) makes the SOPC encoding 8 bytes. Without
        # consuming the literal dword, the following s_endpgm would be misread.
        raw = bytes.fromhex("00ff06bf 78563412 000081bf")
        insns = disasm_rdna2(raw)
        self.assertEqual(
            [(i["opcode"], i["length"], i["assembly"]) for i in insns],
            [
                ("s_cmp_eq_u32", 8, "s_cmp_eq_u32 s0, 0x12345678"),
                ("s_endpgm", 4, "s_endpgm"),
            ],
        )

    def test_second_dword_forms_keep_the_stream_aligned(self):
        # Independent GFX10 encodings from LLVM MC 15.0.7 test/MC/AMDGPU (gfx10_asm_vop1.s,
        # gfx10_asm_vop2.s, gfx10_asm_dpp8.s, gfx10_asm_sop.s). Each is one 8-byte instruction.
        endpgm = bytes.fromhex("000081bf")
        cases = (
            ("f9160a7e01060600", "vop1_sdwa", "v_cvt_f32_f16_sdwa <unsupported SDWA>"),
            ("fa020a7e01e40000", "vop1_dpp16", "v_mov_b32_dpp <unsupported DPP16>"),
            ("e9020a7e0188c6fa", "vop1_dpp8", "v_mov_b32_dpp <unsupported DPP8>"),
            ("ea020a7e0188c6fa", "vop1_dpp8", "v_mov_b32_dpp <unsupported DPP8>"),  # fi:1
            ("f9040a0601060606", "vop2_sdwa", "v_add_f32_sdwa <unsupported SDWA>"),
            ("fa040a1001e40000", "vop2_dpp16", "v_mul_f32_dpp <unsupported DPP16>"),
            ("e9040a060188c6fa", "vop2_dpp8", "v_add_f32_dpp <unsupported DPP8>"),
            ("01070a6e21110000", "v_fmamk_f16", "v_fmamk_f16 v5, v1, 0x1121, v3"),
            ("01050a7021110000", "v_fmaak_f16", "v_fmaak_f16 v5, v1, v2, 0x1121"),
            ("341280ba41312111", "s_setreg_imm32_b32", "s_setreg_imm32_b32 0x1234, 0x11213141"),
        )
        for raw, opcode, assembly in cases:
            with self.subTest(raw=raw):
                insns = disasm_rdna2(bytes.fromhex(raw) + endpgm)
                self.assertEqual([(i["offset"], i["length"], i["opcode"]) for i in insns],
                                 [(0, 8, opcode), (8, 4, "s_endpgm")])
                self.assertEqual(insns[0]["assembly"], assembly)
                truncated = disasm_rdna2(bytes.fromhex(raw)[:4])
                self.assertEqual(len(truncated), 1)
                self.assertEqual(truncated[0]["length"], 4)
                self.assertTrue(truncated[0]["opcode"].endswith("_truncated"), truncated[0]["opcode"])

        # Control: SRC0 codes next to the extension codes name an operand and stay one dword.
        for src0 in (232, 235, 248, 251, 256):
            with self.subTest(src0=src0):
                insns = disasm_rdna2(struct.pack("<I", 0x7e0a0200 | src0) + endpgm)
                self.assertEqual([(i["length"], i["opcode"]) for i in insns], [(4, "v_mov_b32_e32"), (4, "s_endpgm")])

    def test_sopc_opcode_table_synthetic_forms(self):
        from disasm_rdna2 import SOPC_OPS
        for op, (expected_name, _) in SOPC_OPS.items():
            w0 = 0xbf000000 | (op << 16) | (2 << 8) | 0
            raw = struct.pack("<I", w0)
            insns = disasm_rdna2(raw)
            self.assertEqual(len(insns), 1, f"Failed decoding SOPC op 0x{op:02x}")
            self.assertEqual(insns[0]["opcode"], expected_name)
            self.assertEqual(insns[0]["length"], 4)

        w0_literal = 0xbf000000 | (0x06 << 16) | (255 << 8) | 0
        truncated_raw = struct.pack("<I", w0_literal)
        insns = disasm_rdna2(truncated_raw)
        self.assertEqual(len(insns), 1)
        self.assertEqual(insns[0]["opcode"], "sopc_truncated")
        self.assertIn("<truncated>", insns[0]["assembly"])

    def test_vopc_opcode_table_synthetic_forms(self):
        from disasm_rdna2 import VOPC_OPS
        for op, (expected_name, _) in VOPC_OPS.items():
            wide = expected_name.endswith(("_f64_e32", "_i64_e32", "_u64_e32"))
            src0 = 0 if wide else 1
            w0 = 0x7c000000 | (op << 17) | (2 << 9) | src0
            insns = disasm_rdna2(struct.pack("<I", w0))
            self.assertEqual(len(insns), 1, f"Failed decoding VOPC op 0x{op:02x}")
            self.assertEqual(insns[0]["opcode"], expected_name)
            self.assertEqual(insns[0]["length"], 4)
            lhs = "s[0:1]" if wide else "s1"
            rhs = "v[2:3]" if wide and "_class_" not in expected_name else "v2"
            dest = "" if expected_name.startswith("v_cmpx_") else "vcc_lo, "
            self.assertEqual(insns[0]["assembly"], f"{expected_name} {dest}{lhs}, {rhs}")

        # Independent GFX10 encodings from LLVM MC's gfx10_asm_vopc/vopcx fixtures.
        for raw, expected in (
            ("6404427c", "v_cmp_lt_f64_e32 vcc_lo, s[100:101], v[2:3]"),
            ("01fd437c", "v_cmp_lt_f64_e32 vcc_lo, v[1:2], v[254:255]"),
        ):
            self.assertEqual(disasm_rdna2(bytes.fromhex(raw))[0]["assembly"], expected)
        sdwa = bytes.fromhex("f904247c01001606")
        insns = disasm_rdna2(sdwa + bytes.fromhex("000081bf"))
        self.assertEqual([i["opcode"] for i in insns], ["vopc_sdwa", "s_endpgm"])
        self.assertEqual(insns[0]["length"], 8)
        self.assertEqual(disasm_rdna2(sdwa[:4])[0]["opcode"], "vopc_truncated")

        # Truncated VOPC literal
        w0_lit = 0x7c000000 | (0x01 << 17) | (2 << 9) | 255
        insns = disasm_rdna2(struct.pack("<I", w0_lit))
        self.assertEqual(len(insns), 1)
        self.assertEqual(insns[0]["opcode"], "vopc_truncated")
        self.assertIn("<truncated>", insns[0]["assembly"])

        # Literal source in VOPC
        raw_lit = struct.pack("<2I", w0_lit, 0x3f800000)
        insns = disasm_rdna2(raw_lit)
        self.assertEqual(len(insns), 1)
        self.assertEqual(insns[0]["opcode"], "v_cmp_lt_f32_e32")
        self.assertEqual(insns[0]["length"], 8)
        self.assertEqual(insns[0]["assembly"], "v_cmp_lt_f32_e32 vcc_lo, 0x3f800000, v2")

        # Scanner recognizes VOPC and assigns correct category
        self.assertTrue(is_opcode_recognized("v_cmp_lt_f32_e32"))
        self.assertTrue(is_opcode_recognized("v_cmpx_gt_i32_e32"))
        self.assertEqual(classify_isa_category("v_cmp_lt_f32_e32"), "VOPC (Vector Compare)")
        self.assertEqual(classify_isa_category("v_cmpx_gt_i32_e32"), "VOPC (Vector Compare)")

    def test_sop2_absdiff_opcode_table_synthetic_forms(self):
        from disasm_rdna2 import SOP2_OPS
        expected_name, _ = SOP2_OPS[0x2c]
        self.assertEqual(expected_name, "s_absdiff_i32")
        w0 = 0x80000000 | (0x2c << 23) | (2 << 16) | (1 << 8) | 0
        raw = struct.pack("<I", w0)
        insns = disasm_rdna2(raw)
        self.assertEqual(len(insns), 1, "Failed decoding SOP2 op 0x2c")
        self.assertEqual(insns[0]["opcode"], "s_absdiff_i32")
        self.assertEqual(insns[0]["length"], 4)
        self.assertEqual(insns[0]["assembly"], "s_absdiff_i32 s2, s0, s1")

        self.assertTrue(is_opcode_recognized("s_absdiff_i32"))
        self.assertEqual(classify_isa_category("s_absdiff_i32"), "SALU (Scalar ALU)")

    def test_sop2_lshl_add_opcode_table_synthetic_forms(self):
        from disasm_rdna2 import SOP2_OPS
        target_ops = {0x2e, 0x2f, 0x30, 0x31}
        for op in target_ops:
            expected_name, _ = SOP2_OPS[op]
            w0 = 0x80000000 | (op << 23) | (2 << 16) | (1 << 8) | 0
            raw = struct.pack("<I", w0)
            insns = disasm_rdna2(raw)
            self.assertEqual(len(insns), 1, f"Failed decoding SOP2 op 0x{op:02x}")
            self.assertEqual(insns[0]["opcode"], expected_name)
            self.assertEqual(insns[0]["length"], 4)

    def test_vop2_integer_opcode_table_synthetic_forms(self):
        from disasm_rdna2 import VOP2_OPS
        target_ops = {0x16, 0x18, 0x1a, 0x1b, 0x1c, 0x1d, 0x25, 0x26, 0x27}
        for op in target_ops:
            expected_name, _ = VOP2_OPS[op]
            w0 = (op << 25) | (2 << 17) | (1 << 9) | 0
            raw = struct.pack("<I", w0)
            insns = disasm_rdna2(raw)
            self.assertEqual(len(insns), 1, f"Failed decoding VOP2 op 0x{op:02x}")
            self.assertEqual(insns[0]["opcode"], expected_name)
            self.assertEqual(insns[0]["length"], 4)

        truncated_raw = b"\x16\x00"
        insns = disasm_rdna2(truncated_raw)
        self.assertEqual(len(insns), 1)
        self.assertEqual(insns[0]["opcode"], "data")
        self.assertEqual(insns[0]["length"], 2)

    def test_vop3_opcode_table_synthetic_forms(self):
        from disasm_rdna2 import VOP3_NATIVE_OPS
        for op, (expected_name, _, arity, is_vop3b, dst_width, src_widths) in VOP3_NATIVE_OPS.items():
            if is_vop3b:
                w0 = 0xd4000000 | (op << 16) | (0 << 8) | 1  # sdst=s0, vdst=v1
            else:
                w0 = 0xd4000000 | (op << 16) | 1  # vdst=v1
            w1 = 258 | (259 << 9)  # src0=v2, src1=v3
            if arity >= 3:
                w1 |= (260 << 18)  # src2=v4
            raw = struct.pack("<2I", w0, w1)
            insns = disasm_rdna2(raw)
            self.assertEqual(len(insns), 1, f"Failed decoding VOP3 op 0x{op:03x}")
            self.assertEqual(insns[0]["opcode"], expected_name)
            self.assertEqual(insns[0]["length"], 8)
            expected_dst = "v[1:2]" if dst_width == 2 else "v1"
            expected_srcs = []
            raw_regs = [2, 3, 4]
            for i in range(arity):
                w = src_widths[i]
                r = raw_regs[i]
                expected_srcs.append(f"v[{r}:{r+1}]" if w == 2 else f"v{r}")
            srcs_str = ", ".join(expected_srcs)
            if is_vop3b:
                self.assertEqual(insns[0]["assembly"], f"{expected_name} {expected_dst}, s0, {srcs_str}")
            else:
                self.assertEqual(insns[0]["assembly"], f"{expected_name} {expected_dst}, {srcs_str}")

        # Scanner recognizes all native VOP3 opcodes
        self.assertTrue(is_opcode_recognized("v_fma_f32"))
        self.assertTrue(is_opcode_recognized("v_add_co_u32"))
        self.assertTrue(is_opcode_recognized("v_cndmask_b32_e64"))
        self.assertEqual(classify_isa_category("v_fma_f32"), "VOP / VALU (Vector ALU - VOP1/VOP2/VOP3)")

    def test_vop3_distinguishes_non_default_modifiers(self):
        # Base instruction: v_add_f32_e64 v3, v1, v2
        w0_base = 0xd4000000 | (0x103 << 16) | 3
        w1_base = 257 | (258 << 9)

        base_asm = disasm_rdna2(struct.pack("<2I", w0_base, w1_base))[0]["assembly"]
        self.assertEqual(base_asm, "v_add_f32_e64 v3, v1, v2")

        # abs on src0 (w0[8] = 1)
        abs_asm = disasm_rdna2(struct.pack("<2I", w0_base | (1 << 8), w1_base))[0]["assembly"]
        self.assertEqual(abs_asm, "v_add_f32_e64 v3, |v1|, v2")

        # neg on src0 (w1[29] = 1)
        neg_asm = disasm_rdna2(struct.pack("<2I", w0_base, w1_base | (1 << 29)))[0]["assembly"]
        self.assertEqual(neg_asm, "v_add_f32_e64 v3, -v1, v2")

        # abs and neg on src0
        abs_neg_asm = disasm_rdna2(struct.pack("<2I", w0_base | (1 << 8), w1_base | (1 << 29)))[0]["assembly"]
        self.assertEqual(abs_neg_asm, "v_add_f32_e64 v3, -|v1|, v2")

        # clamp (w0[15] = 1)
        clamp_asm = disasm_rdna2(struct.pack("<2I", w0_base | (1 << 15), w1_base))[0]["assembly"]
        self.assertEqual(clamp_asm, "v_add_f32_e64 v3, v1, v2 clamp")

        # omod mul:2 (w1[27:28] = 1)
        omod_asm = disasm_rdna2(struct.pack("<2I", w0_base, w1_base | (1 << 27)))[0]["assembly"]
        self.assertEqual(omod_asm, "v_add_f32_e64 v3, v1, v2 mul:2")

        # op_sel (w0[11] = 1)
        opsel_asm = disasm_rdna2(struct.pack("<2I", w0_base | (1 << 11), w1_base))[0]["assembly"]
        self.assertEqual(opsel_asm, "v_add_f32_e64 v3, v1, v2 op_sel:[1,0,0]")

        variants = [base_asm, abs_asm, neg_asm, abs_neg_asm, clamp_asm, omod_asm, opsel_asm]
        self.assertEqual(len(set(variants)), len(variants), "VOP3 modifiers produced colliding assemblies")

    def test_vop3_operand_widths_and_stream_alignment(self):
        # Verify opcode-specific operand widths from LLVM GFX10 MC test cases,
        # mixed-width forms, conversions, and byte-stream alignment with trailing s_endpgm.
        stream = bytes.fromhex(
            "fe0064d501050202"  # v_add_f64 v[254:255], v[1:2], v[2:3]
            "0200ffd683080200"  # v_lshlrev_b64 v[2:3], 3, v[4:5]
            "020068d5040d0000"  # v_ldexp_f64 v[2:3], v[4:5], s6
            "000076d502061204"  # v_mad_u64_u32 v[0:1], s0, v2, v3, v[4:5]
            "00026ed502091a04"  # v_div_scale_f64 v[0:1], s2, v[2:3], v[4:5], v[6:7]
            "000090d502010000"  # v_cvt_f64_f32_e64 v[0:1], v2
            "00008fd502010000"  # v_cvt_f32_f64_e64 v0, v[2:3]
            "000021d402090200"  # v_cmp_lt_f64_e64 s0, v[2:3], v[4:5]
            "000080d502091a04"  # v_nop (8-byte VOP3 slot)
            "0000abd502091a04"  # v_rcp_iflag_f32_e64 v0, v2
            "0000afd502091a04"  # v_rcp_f64_e64 v[0:1], v[2:3]
            "000040d502091a04"  # v_mad_legacy_f32 v0, v2, v4, v6
            "000050d502091a04"  # v_mullit_f32 v0, v2, v4, v6
            "000072d502091a04"  # v_vop3_172 (undecoded special form consumes 8 bytes)
            "05004bd501ff09040000803f"  # v_fma_f32 v5, v1, 0x3f800000, v2 (12-byte literal)
            "000081bf"          # s_endpgm
        )
        insns = disasm_rdna2(stream)
        self.assertEqual(
            [(i["opcode"], i["length"], i["assembly"]) for i in insns],
            [
                ("v_add_f64", 8, "v_add_f64 v[254:255], v[1:2], v[2:3]"),
                ("v_lshlrev_b64", 8, "v_lshlrev_b64 v[2:3], 3, v[4:5]"),
                ("v_ldexp_f64", 8, "v_ldexp_f64 v[2:3], v[4:5], s6"),
                ("v_mad_u64_u32", 8, "v_mad_u64_u32 v[0:1], s0, s2, v3, v[4:5]"),
                ("v_div_scale_f64", 8, "v_div_scale_f64 v[0:1], s2, v[2:3], v[4:5], v[6:7]"),
                ("v_cvt_f64_f32_e64", 8, "v_cvt_f64_f32_e64 v[0:1], v2"),
                ("v_cvt_f32_f64_e64", 8, "v_cvt_f32_f64_e64 v0, v[2:3]"),
                ("v_cmp_lt_f64_e64", 8, "v_cmp_lt_f64_e64 s0, v[2:3], v[4:5]"),
                ("v_nop", 8, "v_nop"),
                ("v_rcp_iflag_f32_e64", 8, "v_rcp_iflag_f32_e64 v0, v2"),
                ("v_rcp_f64_e64", 8, "v_rcp_f64_e64 v[0:1], v[2:3]"),
                ("v_mad_legacy_f32", 8, "v_mad_legacy_f32 v0, v2, v4, v6"),
                ("v_mullit_f32", 8, "v_mullit_f32 v0, v2, v4, v6"),
                ("v_vop3_172", 8, "v_vop3_172 v0, v2, v4, v6"),
                ("v_fma_f32", 12, "v_fma_f32 v5, v1, 0x3f800000, v2"),
                ("s_endpgm", 4, "s_endpgm"),
            ],
        )

    def test_vop3p_opcode_table_synthetic_forms(self):
        from disasm_rdna2 import VOP3P_OPS, VOP3P_3SRC
        for op, (expected_name, _) in VOP3P_OPS.items():
            is_3src = op in VOP3P_3SRC
            w0 = 0xcc000000 | (op << 16) | 1  # vdst = 1
            w1 = (256 + 2) | ((256 + 3) << 9)  # v2, v3
            if is_3src:
                w1 |= ((256 + 4) << 18)  # v4
            if not expected_name.startswith("v_fma_mix"):
                w1 |= (1 << 27) | (1 << 28)
                if is_3src:
                    w0 |= (1 << 14)
            raw = struct.pack("<2I", w0, w1)
            insns = disasm_rdna2(raw)
            self.assertEqual(len(insns), 1, f"Failed decoding VOP3P op 0x{op:02x}")
            self.assertEqual(insns[0]["opcode"], expected_name)
            self.assertEqual(insns[0]["length"], 8)
            if is_3src:
                self.assertEqual(insns[0]["assembly"], f"{expected_name} v1, v2, v3, v4")
            else:
                self.assertEqual(insns[0]["assembly"], f"{expected_name} v1, v2, v3")

        # Truncated VOP3P word
        truncated_raw = struct.pack("<I", 0xcc000000)
        insns = disasm_rdna2(truncated_raw)
        self.assertEqual(len(insns), 1)
        self.assertEqual(insns[0]["opcode"], "vop3p_truncated")
        self.assertIn("<truncated>", insns[0]["assembly"])

        # Literal source in VOP3P
        w0_lit = 0xcc000000 | (0x11 << 16) | 25  # v_pk_min_f16 v25
        w1_lit = (256 + 24) | (255 << 9) | (1 << 27) | (1 << 28)  # v24, literal
        raw_lit = struct.pack("<3I", w0_lit, w1_lit, 0x40007e00)
        insns = disasm_rdna2(raw_lit)
        self.assertEqual(len(insns), 1)
        self.assertEqual(insns[0]["opcode"], "v_pk_min_f16")
        self.assertEqual(insns[0]["length"], 12)
        self.assertEqual(insns[0]["assembly"], "v_pk_min_f16 v25, v24, 0x40007e00")

        # Truncated literal
        raw_lit_trunc = struct.pack("<2I", w0_lit, w1_lit)
        insns = disasm_rdna2(raw_lit_trunc)
        self.assertEqual(len(insns), 1)
        self.assertEqual(insns[0]["opcode"], "vop3p_truncated")
        self.assertIn("<truncated>", insns[0]["assembly"])

    def test_unsupported_vop3p_keeps_literal_stream_aligned(self):
        # Packed add with a source-0 literal and a reserved source-2 selection bit.
        first = 0xcc0a0004 | (1 << 13)
        second = 255 | (129 << 9) | (3 << 27)
        stream = struct.pack("<4I", first, second, 0x3f800000, 0xbf810000)
        decoded = disasm_rdna2(stream)
        self.assertEqual([(row["opcode"], row["length"]) for row in decoded],
                         [("v_pk_add_u16_unsupported", 12), ("s_endpgm", 4)])

    def test_vop3p_distinguishes_non_default_modifiers_and_selections(self):
        """VOP3P instructions must distinguish non-default op_sel, neg, and clamp from default forms."""
        # Base 2-source instruction: v_pk_add_u16 v4, v3, 1 (default op_sel_hi:[1,1])
        w0_base = 0xcc000000 | (0x0a << 16) | 4
        w1_base = 259 | (129 << 9) | (1 << 27) | (1 << 28)

        default_insn = disasm_rdna2(struct.pack("<2I", w0_base, w1_base))[0]["assembly"]
        self.assertEqual(default_insn, "v_pk_add_u16 v4, v3, 1")

        # Modified op_sel on src0 (w0[11] = 1) -> op_sel:[1,0]
        op_sel_insn = disasm_rdna2(struct.pack("<2I", w0_base | (1 << 11), w1_base))[0]["assembly"]
        self.assertEqual(op_sel_insn, "v_pk_add_u16 v4, v3, 1 op_sel:[1,0]")

        # Modified op_sel_hi on src0 (w1[27] = 0) -> op_sel_hi:[0,1]
        op_sel_hi_insn = disasm_rdna2(struct.pack("<2I", w0_base, w1_base & ~(1 << 27)))[0]["assembly"]
        self.assertEqual(op_sel_hi_insn, "v_pk_add_u16 v4, v3, 1 op_sel_hi:[0,1]")

        # Modified neg_lo on src0 (w1[29] = 1) -> neg_lo:[1,0]
        neg_lo_insn = disasm_rdna2(struct.pack("<2I", w0_base, w1_base | (1 << 29)))[0]["assembly"]
        self.assertEqual(neg_lo_insn, "v_pk_add_u16 v4, v3, 1 neg_lo:[1,0]")

        # Modified neg_hi on src0 (w0[8] = 1) -> neg_hi:[1,0]
        neg_hi_insn = disasm_rdna2(struct.pack("<2I", w0_base | (1 << 8), w1_base))[0]["assembly"]
        self.assertEqual(neg_hi_insn, "v_pk_add_u16 v4, v3, 1 neg_hi:[1,0]")

        # Clamp (w0[15] = 1)
        clamp_insn = disasm_rdna2(struct.pack("<2I", w0_base | (1 << 15), w1_base))[0]["assembly"]
        self.assertEqual(clamp_insn, "v_pk_add_u16 v4, v3, 1 clamp")

        # All variants have identical register operands but distinct assemblies
        disasms = [default_insn, op_sel_insn, op_sel_hi_insn, neg_lo_insn, neg_hi_insn, clamp_insn]
        self.assertEqual(len(set(disasms)), len(disasms), "VOP3P modifier variants collided")

        # 3-source instruction: v_pk_fma_f16 v1, v2, v3, v4
        w0_3src = 0xcc000000 | (0x0e << 16) | (1 << 14) | 1  # op_sel_hi[2] = 1
        w1_3src = (256 + 2) | ((256 + 3) << 9) | ((256 + 4) << 18) | (1 << 27) | (1 << 28)
        default_3src = disasm_rdna2(struct.pack("<2I", w0_3src, w1_3src))[0]["assembly"]
        self.assertEqual(default_3src, "v_pk_fma_f16 v1, v2, v3, v4")

        # Modified op_sel_hi on src2 (w0[14] = 0) -> op_sel_hi:[1,1,0]
        op_sel_hi_3src = disasm_rdna2(struct.pack("<2I", w0_3src & ~(1 << 14), w1_3src))[0]["assembly"]
        self.assertEqual(op_sel_hi_3src, "v_pk_fma_f16 v1, v2, v3, v4 op_sel_hi:[1,1,0]")

        # Modified neg_hi on src2 (w0[10] = 1) -> neg_hi:[0,0,1]
        neg_hi_3src = disasm_rdna2(struct.pack("<2I", w0_3src | (1 << 10), w1_3src))[0]["assembly"]
        self.assertEqual(neg_hi_3src, "v_pk_fma_f16 v1, v2, v3, v4 neg_hi:[0,0,1]")

        # Modified neg_lo on src2 (w1[31] = 1) -> neg_lo:[0,0,1]
        neg_lo_3src = disasm_rdna2(struct.pack("<2I", w0_3src, w1_3src | (1 << 31)))[0]["assembly"]
        self.assertEqual(neg_lo_3src, "v_pk_fma_f16 v1, v2, v3, v4 neg_lo:[0,0,1]")

        # Both 0 and 1 in bit 14 (LLVM MC default op_sel_hi = 0b111) decode to default 2-source form
        self.assertEqual(disasm_rdna2(struct.pack("<2I", w0_base | (1 << 14), w1_base))[0]["assembly"], "v_pk_add_u16 v4, v3, 1")

        # 2-source encoding with reserved bit set (e.g. w1[31], w0[10], w0[13], or src2 != 0) is flagged as unsupported
        for reserved_w0, reserved_w1 in [
            (w0_base, w1_base | (1 << 31)),          # neg_lo[2]
            (w0_base | (1 << 10), w1_base),          # neg_hi[2]
            (w0_base | (1 << 13), w1_base),          # op_sel[2]
            (w0_base, w1_base | (256 << 18)),        # src2 = v0
        ]:
            reserved_raw = struct.pack("<2I", reserved_w0, reserved_w1)
            res_rec = disasm_rdna2(reserved_raw)[0]
            self.assertEqual(res_rec["opcode"], "v_pk_add_u16_unsupported")
            self.assertIn("<unsupported_reserved_bits>", res_rec["assembly"])

    def test_vop3p_fma_mix_inline_abs_neg_modifiers(self):
        """v_fma_mix* must decode w0[10:8] as abs and w1[31:29] as neg inline, not neg_lo/neg_hi."""
        # Issue reproduction encoding: v_fma_mix_f32 v12, |s44|, -v19, v20 op_sel:[1,1,0] op_sel_hi:[1,0,0]
        repro = disasm_rdna2(bytes.fromhex("0c1920cc2c26524c"))
        self.assertEqual(len(repro), 1)
        self.assertEqual(
            repro[0]["assembly"],
            "v_fma_mix_f32 v12, |s44|, -v19, v20 op_sel:[1,1,0] op_sel_hi:[1,0,0]",
        )
        operands, op_sel, op_sel_hi, clamp = parse_fma_mix(repro[0]["assembly"].split(" ", 2)[2])
        self.assertEqual((operands, op_sel, op_sel_hi, clamp), (["|s44|", "-v19", "v20"], [1, 1, 0], [1, 0, 0], False))
        scalar = {"s44": "params[0].x"}
        vector = {"v19": "alu_19", "v20": "alu_20"}
        lowered = [
            fma_mix_operand(tok, bool(f16), bool(hi), scalar, vector, set(), set())
            for tok, f16, hi in zip(operands, op_sel, op_sel_hi)
        ]
        self.assertEqual(lowered, ["abs(widen16(asuint(params[0].x) >> 16))", "(-widen16(asuint(alu_19)))", "alu_20"])

        def encode_fma_mix(op, vdst, src0, src1, src2, abs_bits=(0, 0, 0), neg_bits=(0, 0, 0),
                           sel=(0, 0, 0), sel_hi=(0, 0, 0), use_clamp=False, literal=None):
            w0 = (
                0xcc000000
                | (op << 16)
                | (int(use_clamp) << 15)
                | (sel_hi[2] << 14)
                | (sel[2] << 13)
                | (sel[1] << 12)
                | (sel[0] << 11)
                | (abs_bits[2] << 10)
                | (abs_bits[1] << 9)
                | (abs_bits[0] << 8)
                | vdst
            )
            w1 = (
                (neg_bits[2] << 31)
                | (neg_bits[1] << 30)
                | (neg_bits[0] << 29)
                | (sel_hi[1] << 28)
                | (sel_hi[0] << 27)
                | (src2 << 18)
                | (src1 << 9)
                | src0
            )
            return struct.pack("<3I", w0, w1, literal) if literal is not None else struct.pack("<2I", w0, w1)

        opcodes = (
            (0x20, "v_fma_mix_f32"),
            (0x21, "v_fma_mixlo_f16"),
            (0x22, "v_fma_mixhi_f16"),
        )
        cases = (
            # (abs_bits, neg_bits, sel, sel_hi, use_clamp, expected_ops, expected_suffix)
            ((1, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), False, "|v2|, v3, v4", ""),
            ((0, 1, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), False, "v2, |v3|, v4", ""),
            ((0, 0, 1), (0, 0, 0), (0, 0, 0), (0, 0, 0), False, "v2, v3, |v4|", ""),
            ((0, 0, 0), (1, 0, 0), (0, 0, 0), (0, 0, 0), False, "-v2, v3, v4", ""),
            ((0, 0, 0), (0, 1, 0), (0, 0, 0), (0, 0, 0), False, "v2, -v3, v4", ""),
            ((0, 0, 0), (0, 0, 1), (0, 0, 0), (0, 0, 0), False, "v2, v3, -v4", ""),
            ((1, 1, 1), (1, 0, 1), (1, 1, 1), (1, 0, 1), True,
             "-|v2|, |v3|, -|v4|", " op_sel:[1,1,1] op_sel_hi:[1,0,1] clamp"),
        )
        vec = {"v2": "alu_2", "v3": "alu_3", "v4": "alu_4"}
        for op, name in opcodes:
            for abs_b, neg_b, sel, sel_hi, use_clamp, expected_ops, suffix in cases:
                with self.subTest(op=name, abs_b=abs_b, neg_b=neg_b, clamp=use_clamp):
                    raw = encode_fma_mix(op, 1, 258, 259, 260, abs_b, neg_b, sel, sel_hi, use_clamp)
                    insns = disasm_rdna2(raw)
                    self.assertEqual(len(insns), 1)
                    asm = insns[0]["assembly"]
                    self.assertEqual(asm, f"{name} v1, {expected_ops}{suffix}")
                    self.assertNotIn("neg_lo", asm)
                    self.assertNotIn("neg_hi", asm)
                    self.assertTrue(is_instruction_supported(asm))
                    parsed_ops, parsed_sel, parsed_hi, parsed_clamp = parse_fma_mix(asm.split(" ", 2)[2])
                    self.assertEqual((parsed_sel, parsed_hi, parsed_clamp), (list(sel), list(sel_hi), use_clamp))
                    for tok, f16, hi in zip(parsed_ops, parsed_sel, parsed_hi):
                        fma_mix_operand(tok, bool(f16), bool(hi), {}, vec, set(), set())

        # Literal operand with combined -|literal| modifiers must consume trailing DWORD and round-trip.
        lit_raw = encode_fma_mix(0x20, 1, 255, 259, 260, abs_bits=(1, 0, 0), neg_bits=(1, 0, 0), literal=0x3f800000)
        lit_insns = disasm_rdna2(lit_raw)
        self.assertEqual(len(lit_insns), 1)
        self.assertEqual(lit_insns[0]["length"], 12)
        self.assertEqual(lit_insns[0]["assembly"], "v_fma_mix_f32 v1, -|0x3f800000|, v3, v4")
        lit_ops, lit_sel, lit_hi, lit_clamp = parse_fma_mix(lit_insns[0]["assembly"].split(" ", 2)[2])
        self.assertEqual((lit_ops, lit_sel, lit_hi, lit_clamp), (["-|0x3f800000|", "v3", "v4"], [0, 0, 0], [0, 0, 0], False))

        # Standard VOP3P (e.g. v_dot2_f32_f16) retains neg_lo/neg_hi suffix modifiers.
        dot_raw = encode_fma_mix(0x13, 12, 274, 275, 278, abs_bits=(1, 0, 0), neg_bits=(1, 0, 0), sel_hi=(1, 1, 1))
        self.assertEqual(
            disasm_rdna2(dot_raw)[0]["assembly"],
            "v_dot2_f32_f16 v12, v18, v19, v22 neg_lo:[1,0,0] neg_hi:[1,0,0]",
        )

        # Negative controls: parse_fma_mix rejects neg_lo/neg_hi and f32 high-half selection; truncated literal fails safely.
        for bad_tail in (
            "s44, v19, v20 neg_lo:[0,1,0]",
            "s44, v19, v20 neg_hi:[1,0,0]",
            "v2, v3, v4 op_sel:[0,0,0] op_sel_hi:[1,0,0]",
        ):
            with self.assertRaises(ValueError):
                parse_fma_mix(bad_tail)

        trunc_raw = encode_fma_mix(0x20, 1, 255, 259, 260, abs_bits=(1, 0, 0), neg_bits=(1, 0, 0))
        trunc_insns = disasm_rdna2(trunc_raw)
        self.assertEqual(len(trunc_insns), 1)
        self.assertEqual(trunc_insns[0]["opcode"], "vop3p_truncated")
        self.assertEqual(trunc_insns[0]["length"], 8)
        self.assertIn("<truncated>", trunc_insns[0]["assembly"])

        # End-to-end disasm_rdna2 -> pixel_aot.lower check on a minimal pixel shader with asymmetric
        # v_fma_mix* modifiers and packed v_pk_* forms: swapping abs/neg across sources must change HLSL.
        def encode_vop3p_2src(op, vdst, src0, src1, sel=(0, 0), sel_hi=(1, 1), neg_lo=(0, 0), neg_hi=(0, 0)):
            w0 = (
                0xcc000000
                | (op << 16)
                | (1 << 14)
                | (sel[1] << 12)
                | (sel[0] << 11)
                | (neg_hi[1] << 9)
                | (neg_hi[0] << 8)
                | vdst
            )
            w1 = (
                (neg_lo[1] << 30)
                | (neg_lo[0] << 29)
                | (sel_hi[1] << 28)
                | (sel_hi[0] << 27)
                | (src1 << 9)
                | src0
            )
            return struct.pack("<2I", w0, w1)

        def build_fma_mix_shader(swap_modifiers=False):
            abs_b = (0, 1, 0) if swap_modifiers else (1, 0, 0)
            neg_b = (1, 0, 0) if swap_modifiers else (0, 1, 0)
            return b"".join([
                struct.pack("<I", 0xbe800300 | (21 << 16) | 0x80),                    # s_mov_b32 s21, 0
                struct.pack("<2I", 0xf4000000 | (0x02 << 18) | (28 << 6) | 10, 125 << 25),  # s_load_dwordx4 s[28:31], s[20:21], 0x0
                bytes.fromhex("7fc08cbf"),                                            # s_waitcnt lgkmcnt(0)
                struct.pack("<2I", 0xf4000000 | (0x0a << 18) | (28 << 6) | 14, 125 << 25),  # s_buffer_load_dwordx4 s[28:31], s[28:31], 0x0
                bytes.fromhex("7fc08cbf"),                                            # s_waitcnt lgkmcnt(0)
                struct.pack("<I", 0x7e000000 | (8 << 17) | (0x01 << 9) | 29),         # v_mov_b32_e32 v8, s29
                struct.pack("<I", 0x7e000000 | (9 << 17) | (0x01 << 9) | 30),         # v_mov_b32_e32 v9, s30
                struct.pack("<I", 0x7e000000 | (10 << 17) | (0x01 << 9) | 31),        # v_mov_b32_e32 v10, s31
                encode_fma_mix(0x20, 12, 28, 256 + 8, 256 + 9, abs_bits=abs_b, neg_bits=neg_b,
                               sel=(1, 1, 0), sel_hi=(1, 0, 0)),
                encode_fma_mix(0x22, 10, 256 + 8, 256 + 12, 256 + 9, abs_bits=(1, 0, 0), neg_bits=(1, 0, 0),
                               sel=(1, 0, 0), sel_hi=(1, 0, 0), use_clamp=True),
                struct.pack("<I", 0x7e000000 | (13 << 17) | (0x0b << 9) | (256 + 10)),  # v_cvt_f32_f16_e32 v13, v10
                encode_vop3p_2src(0x0f, 11, 28, 256 + 8, sel=(1, 0), sel_hi=(0, 1), neg_lo=(0, 1)),
                encode_vop3p_2src(0x10, 11, 256 + 11, 256 + 9, neg_hi=(1, 0)),
                encode_fma_mix(0x20, 14, 256 + 11, 0xf2, 0x80, sel=(1, 0, 0), sel_hi=(1, 0, 0)),
                struct.pack("<I", 0x7e000000 | (15 << 17) | (0x01 << 9) | 0xf2),      # v_mov_b32_e32 v15, 1.0
                struct.pack("<I", (0x2f << 25) | (16 << 17) | (13 << 9) | (256 + 12)),# v_cvt_pkrtz_f16_f32_e32 v16, v12, v13
                struct.pack("<I", (0x2f << 25) | (17 << 17) | (15 << 9) | (256 + 14)),# v_cvt_pkrtz_f16_f32_e32 v17, v14, v15
                bytes.fromhex("0f1c00f810110000"),                                    # exp mrt0 v16, v16, v17, v17 done compr vm
                bytes.fromhex("000081bf"),                                            # s_endpgm
            ])

        shader_asm = [rec["assembly"] for rec in disasm_rdna2(build_fma_mix_shader(False))]
        swapped_asm = [rec["assembly"] for rec in disasm_rdna2(build_fma_mix_shader(True))]
        self.assertIn("v_fma_mix_f32 v12, |s28|, -v8, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0]", shader_asm)
        self.assertIn("v_fma_mix_f32 v12, -s28, |v8|, v9 op_sel:[1,1,0] op_sel_hi:[1,0,0]", swapped_asm)
        self.assertIn("v_fma_mixhi_f16 v10, -|v8|, v12, v9 op_sel:[1,0,0] op_sel_hi:[1,0,0] clamp", shader_asm)
        self.assertIn("v_pk_add_f16 v11, s28, v8 op_sel:[1,0] op_sel_hi:[0,1] neg_lo:[0,1]", shader_asm)
        self.assertIn("v_pk_mul_f16 v11, v11, v9 neg_hi:[1,0]", shader_asm)

        hlsl = lower(shader_asm, 20, 24, None, constant_descriptor=0)
        swapped_hlsl = lower(swapped_asm, 20, 24, None, constant_descriptor=0)
        self.assertNotEqual(hlsl, swapped_hlsl)
        self.assertIn("mad(abs(widen16(asuint(params[0].x) >> 16)), (-widen16(asuint(alu5))), alu6)", hlsl)
        self.assertIn("mad((-widen16(asuint(params[0].x) >> 16)), abs(widen16(asuint(alu5))), alu6)", swapped_hlsl)
        self.assertIn("saturate(mad((-abs(widen16(asuint(alu5) >> 16))), alu8, alu6))", hlsl)

        bad_shader_asm = [
            "v_fma_mix_f32 v12, s28, v8, v9 op_sel:[1,1,0] neg_lo:[0,1,0]"
            if line.startswith("v_fma_mix_f32 v12,") else line
            for line in shader_asm
        ]
        with self.assertRaises(ValueError):
            lower(bad_shader_asm, 20, 24, None, constant_descriptor=0)

    def test_mubuf_opcode_table_synthetic_forms(self):
        """Every GFX1013 MUBUF opcode must decode with exact width, scanner recognition, and 8-byte alignment."""
        from disasm_rdna2 import MUBUF_OPS

        self.assertEqual(len(MUBUF_OPS), 72)
        for op, (expected_name, width, _) in MUBUF_OPS.items():
            with self.subTest(op=op, name=expected_name):
                w0 = 0xe0000000 | (op << 18)
                w1 = 0 if width == 0 else ((3 << 24) | (2 << 16) | (8 << 8))
                raw = struct.pack("<2I", w0, w1)
                insns = disasm_rdna2(raw)
                self.assertEqual(len(insns), 1)
                self.assertEqual(insns[0]["opcode"], expected_name)
                self.assertEqual(insns[0]["length"], 8)
                if width == 0:
                    self.assertEqual(insns[0]["assembly"], expected_name)
                else:
                    vdata = "v8" if width == 1 else f"v[8:{8 + width - 1}]"
                    self.assertEqual(insns[0]["assembly"], f"{expected_name} {vdata}, off, s[8:11], s3")
                self.assertTrue(is_opcode_recognized(expected_name))
                self.assertEqual(classify_isa_category(expected_name), "VMEM (Vector Memory / Buffer)")

        # Truncated MUBUF header reports mubuf_truncated instead of crashing or misdecoding.
        trunc = disasm_rdna2(struct.pack("<I", 0xe0300000))
        self.assertEqual(len(trunc), 1)
        self.assertEqual(trunc[0]["opcode"], "mubuf_truncated")
        self.assertIn("<truncated>", trunc[0]["assembly"])

        # Unknown MUBUF opcodes keep 8-byte alignment so subsequent instructions decode cleanly.
        stream = struct.pack("<3I", 0xe0000000 | (0x10 << 18), (3 << 24) | (2 << 16) | (8 << 8), 0xbf810000)
        decoded = disasm_rdna2(stream)
        self.assertEqual([(r["opcode"], r["length"]) for r in decoded], [("buffer_mubuf_10", 8), ("s_endpgm", 4)])

    def test_mtbuf_opcode_table_synthetic_forms(self):
        """Every GFX1013 MTBUF opcode and format identifier must decode accurately and stay aligned."""
        from disasm_rdna2 import MTBUF_FORMATS, MTBUF_OPS

        self.assertEqual(len(MTBUF_OPS), 16)
        for op, (expected_name, width, _) in MTBUF_OPS.items():
            with self.subTest(op=op, name=expected_name):
                op_lo = op & 0x7
                op_hi = (op >> 3) & 1
                w0 = 0xe8000000 | (22 << 19) | (op_lo << 16)
                w1 = (3 << 24) | (op_hi << 21) | (2 << 16) | (8 << 8)
                insns = disasm_rdna2(struct.pack("<2I", w0, w1))
                self.assertEqual(len(insns), 1)
                self.assertEqual(insns[0]["opcode"], expected_name)
                self.assertEqual(insns[0]["length"], 8)
                vdata = "v8" if width == 1 else f"v[8:{8 + width - 1}]"
                self.assertEqual(
                    insns[0]["assembly"],
                    f"{expected_name} {vdata}, off, s[8:11], s3 format:[BUF_FMT_32_FLOAT]",
                )
                self.assertTrue(is_opcode_recognized(expected_name))
                self.assertEqual(classify_isa_category(expected_name), "VMEM (Vector Memory / Buffer)")

        # Format 1 is LLVM's omitted default; named formats print in brackets; >=78 print numerically.
        for fmt, expected_suffix in [
            (0, " format:[BUF_FMT_INVALID]"),
            (1, ""),
            *[(code, f" format:[{name}]") for code, name in MTBUF_FORMATS.items() if code != 0],
            (78, " format:78"),
            (127, " format:127"),
        ]:
            w0 = 0xe8000000 | (fmt << 19)
            w1 = (3 << 24) | (2 << 16) | (5 << 8)
            asm = disasm_rdna2(struct.pack("<2I", w0, w1))[0]["assembly"]
            self.assertEqual(asm, f"tbuffer_load_format_x v5, off, s[8:11], s3{expected_suffix}")

        # Truncated MTBUF reports mtbuf_truncated and full instructions keep stream alignment.
        trunc = disasm_rdna2(struct.pack("<I", 0xe8b00000))
        self.assertEqual(len(trunc), 1)
        self.assertEqual(trunc[0]["opcode"], "mtbuf_truncated")
        self.assertIn("<truncated>", trunc[0]["assembly"])

        aligned = disasm_rdna2(struct.pack("<3I", 0xe8b00000, (3 << 24) | (2 << 16) | (5 << 8), 0xbf810000))
        self.assertEqual([(r["opcode"], r["length"]) for r in aligned], [("tbuffer_load_format_x", 8), ("s_endpgm", 4)])

    def test_mubuf_mtbuf_distinguishes_modifiers(self):
        """Buffer addressing, cache, LDS, TFE, and format modifiers must produce distinct disassembly."""
        base_w0 = 0xe0000000 | (0x0c << 18)  # buffer_load_dword
        base_w1 = (3 << 24) | (2 << 16) | (5 << 8) | 1
        variants = {
            "base": disasm_rdna2(struct.pack("<2I", base_w0, base_w1))[0]["assembly"],
            "offen": disasm_rdna2(struct.pack("<2I", base_w0 | (1 << 12), base_w1))[0]["assembly"],
            "idxen": disasm_rdna2(struct.pack("<2I", base_w0 | (1 << 13), base_w1))[0]["assembly"],
            "idxen+offen": disasm_rdna2(struct.pack("<2I", base_w0 | (1 << 12) | (1 << 13), base_w1))[0]["assembly"],
            "offset": disasm_rdna2(struct.pack("<2I", base_w0 | 64, base_w1))[0]["assembly"],
            "glc": disasm_rdna2(struct.pack("<2I", base_w0 | (1 << 14), base_w1))[0]["assembly"],
            "slc": disasm_rdna2(struct.pack("<2I", base_w0, base_w1 | (1 << 22)))[0]["assembly"],
            "dlc": disasm_rdna2(struct.pack("<2I", base_w0 | (1 << 15), base_w1))[0]["assembly"],
            "lds": disasm_rdna2(struct.pack("<2I", base_w0 | (1 << 16), base_w1))[0]["assembly"],
            "tfe": disasm_rdna2(struct.pack("<2I", base_w0, base_w1 | (1 << 23)))[0]["assembly"],
            "soffset_m0": disasm_rdna2(struct.pack("<2I", base_w0, (124 << 24) | (2 << 16) | (5 << 8) | 1))[0]["assembly"],
        }
        self.assertEqual(len(set(variants.values())), len(variants))
        self.assertEqual(variants["idxen+offen"], "buffer_load_dword v5, v[1:2], s[8:11], s3 idxen offen")
        self.assertEqual(variants["lds"], "buffer_load_dword off, s[8:11], s3 lds")
        self.assertEqual(variants["tfe"], "buffer_load_dword v5, off, s[8:11], s3 tfe")
        self.assertEqual(variants["soffset_m0"], "buffer_load_dword v5, off, s[8:11], m0")

        # Atomics suppress tfe, matching LLVM AMDGPU disassembly.
        atomic_asm = disasm_rdna2(struct.pack("<2I", 0xe0000000 | (0x32 << 18), base_w1 | (1 << 23)))[0]["assembly"]
        self.assertEqual(atomic_asm, "buffer_atomic_add v5, off, s[8:11], s3")

        # Trap-handler temporary buffer descriptors (srsrc idx 27..30 -> ttmp[0:3]..ttmp[12:15]).
        for srsrc_idx, t in ((27, 0), (28, 4), (29, 8), (30, 12)):
            ttmp_w1 = (3 << 24) | (srsrc_idx << 16) | (5 << 8)
            ttmp_asm = disasm_rdna2(struct.pack("<2I", base_w0, ttmp_w1))[0]["assembly"]
            self.assertEqual(ttmp_asm, f"buffer_load_dword v5, off, ttmp[{t}:{t + 3}], s3")

    def test_flat_global_scratch_opcode_tables_synthetic_forms(self):
        """All 130 GFX1013 FLAT, GLOBAL, and SCRATCH forms must decode cleanly and keep stream alignment."""
        from disasm_rdna2 import FLAT_OPS, GLOBAL_OPS, SCRATCH_OPS

        self.assertEqual(len(FLAT_OPS), 54)
        self.assertEqual(len(GLOBAL_OPS), 54)
        self.assertEqual(len(SCRATCH_OPS), 22)

        def fmt_v(reg: int, width: int) -> str:
            return f"v{reg}" if width == 1 else f"v[{reg}:{reg + width - 1}]"

        for seg, ops_table in ((0, FLAT_OPS), (2, GLOBAL_OPS), (1, SCRATCH_OPS)):
            for op, (expected_name, kind, vdst_w, vdata_w, _) in ops_table.items():
                with self.subTest(seg=seg, op=op, name=expected_name):
                    w0 = 0xdc000000 | (op << 18) | (seg << 14)
                    w1 = (4 << 24) | (0x7d << 16) | (8 << 8) | 2
                    insns = disasm_rdna2(struct.pack("<2I", w0, w1))
                    self.assertEqual(len(insns), 1)
                    self.assertEqual(insns[0]["opcode"], expected_name)
                    self.assertEqual(insns[0]["length"], 8)

                    vaddr = "v[2:3]" if seg in (0, 2) else "v2"
                    if kind == "load":
                        expected_ops = [fmt_v(4, vdst_w), vaddr]
                    elif kind == "store":
                        expected_ops = [vaddr, fmt_v(8, vdata_w)]
                    else:
                        expected_ops = [vaddr, fmt_v(8, vdata_w)]
                    if seg in (1, 2):
                        expected_ops.append("off")
                    self.assertEqual(insns[0]["assembly"], f"{expected_name} {', '.join(expected_ops)}")
                    self.assertTrue(is_opcode_recognized(expected_name))
                    self.assertEqual(classify_isa_category(expected_name), "VMEM (Vector Memory / Buffer)")

        # Truncated FLAT header reports flat_truncated instead of crashing.
        trunc = disasm_rdna2(struct.pack("<I", 0xdc300000))
        self.assertEqual(len(trunc), 1)
        self.assertEqual(trunc[0]["opcode"], "flat_truncated")
        self.assertIn("<truncated>", trunc[0]["assembly"])

        # Reserved segment=3 and unsupported scratch atomic keep 8-byte alignment before s_endpgm.
        for w0 in (0xdc30c000, 0xdcc84000):
            stream = struct.pack("<3I", w0, (0x7d << 16) | 2, 0xbf810000)
            decoded = disasm_rdna2(stream)
            self.assertEqual([r["length"] for r in decoded], [8, 4])
            self.assertEqual(decoded[1]["opcode"], "s_endpgm")

    def test_flat_global_scratch_distinguishes_modifiers_and_addressing(self):
        """Atomic glc return-value form, signed vs unsigned offsets, saddr, and LDS must stay distinct."""
        # Atomic glc=0 vs glc=1 controls whether vdst is emitted.
        for seg, base_w0, expected_no_rtn, expected_rtn in [
            (0, 0xdc000000 | (0x32 << 18), "flat_atomic_add v[2:3], v8", "flat_atomic_add v4, v[2:3], v8 glc"),
            (2, 0xdc000000 | (0x32 << 18) | (2 << 14), "global_atomic_add v[2:3], v8, off", "global_atomic_add v4, v[2:3], v8, off glc"),
        ]:
            w1 = (4 << 24) | (0x7d << 16) | (8 << 8) | 2
            no_rtn = disasm_rdna2(struct.pack("<2I", base_w0, w1))[0]["assembly"]
            rtn = disasm_rdna2(struct.pack("<2I", base_w0 | (1 << 16), w1))[0]["assembly"]
            self.assertEqual(no_rtn, expected_no_rtn)
            self.assertEqual(rtn, expected_rtn)

        # FLAT offset is 12-bit unsigned while GLOBAL/SCRATCH offsets are 12-bit signed.
        w1_off = (4 << 24) | (0x7d << 16) | 2
        self.assertEqual(disasm_rdna2(struct.pack("<2I", 0xdc300800, w1_off))[0]["assembly"], "flat_load_dword v4, v[2:3] offset:2048")
        self.assertEqual(disasm_rdna2(struct.pack("<2I", 0xdc308800, w1_off))[0]["assembly"], "global_load_dword v4, v[2:3], off offset:-2048")
        self.assertEqual(disasm_rdna2(struct.pack("<2I", 0xdc304fff, w1_off))[0]["assembly"], "scratch_load_dword v4, v2, off offset:-1")

        # Global saddr forms: off vs s[2:3] vs vcc vs ttmp[0:1], plus lds/slc/dlc/glc.
        global_variants = {
            "off": disasm_rdna2(struct.pack("<2I", 0xdc308000, (4 << 24) | (0x7d << 16) | 2))[0]["assembly"],
            "s2": disasm_rdna2(struct.pack("<2I", 0xdc308000, (4 << 24) | (2 << 16) | 1))[0]["assembly"],
            "vcc": disasm_rdna2(struct.pack("<2I", 0xdc308000, (4 << 24) | (106 << 16) | 1))[0]["assembly"],
            "ttmp0": disasm_rdna2(struct.pack("<2I", 0xdc308000, (4 << 24) | (108 << 16) | 1))[0]["assembly"],
            "lds": disasm_rdna2(struct.pack("<2I", 0xdc30a000, (0 << 24) | (2 << 16) | 1))[0]["assembly"],
            "slc_dlc": disasm_rdna2(struct.pack("<2I", 0xdc329010, (4 << 24) | (2 << 16) | 1))[0]["assembly"],
        }
        self.assertEqual(len(set(global_variants.values())), len(global_variants))
        self.assertEqual(global_variants["vcc"], "global_load_dword v4, v1, vcc")
        self.assertEqual(global_variants["ttmp0"], "global_load_dword v4, v1, ttmp[0:1]")
        self.assertEqual(global_variants["lds"], "global_load_dword v1, s[2:3] lds")
        self.assertEqual(global_variants["slc_dlc"], "global_load_dword v4, v1, s[2:3] offset:16 slc dlc")

        # Scratch saddr vs vaddr addressing.
        self.assertEqual(
            disasm_rdna2(struct.pack("<2I", 0xdc304000, (4 << 24) | (2 << 16) | 1))[0]["assembly"],
            "scratch_load_dword v4, off, s2",
        )
        self.assertEqual(
            disasm_rdna2(struct.pack("<2I", 0xdc306010, (0 << 24) | (2 << 16) | 0))[0]["assembly"],
            "scratch_load_dword off, s2 offset:16 lds",
        )

    def test_ds_opcode_table_synthetic_forms(self):
        """All 154 GFX1013 DS opcodes must decode cleanly and invalid variants keep 8-byte alignment."""
        from disasm_rdna2 import DS_OPS

        self.assertEqual(len(DS_OPS), 154)

        def fmt_v(reg: int, width: int) -> str:
            return f"v{reg}" if width == 1 else f"v[{reg}:{reg + width - 1}]"

        for op, (expected_name, vdst_w, has_addr, data0_w, data1_w, offset_mode, gds_mode) in DS_OPS.items():
            with self.subTest(op=op, name=expected_name):
                gds = 1 if gds_mode == "gds_only" else 0
                w0 = 0xd8000000 | (op << 18) | (gds << 17)
                w1 = (24 << 24) | (16 << 16) | (8 << 8) | 2
                insns = disasm_rdna2(struct.pack("<2I", w0, w1))
                self.assertEqual(len(insns), 1)
                self.assertEqual(insns[0]["opcode"], expected_name)
                self.assertEqual(insns[0]["length"], 8)

                ops = []
                if vdst_w:
                    ops.append(fmt_v(24, vdst_w))
                if has_addr:
                    ops.append("v2")
                if data0_w:
                    ops.append(fmt_v(8, data0_w))
                if data1_w:
                    ops.append(fmt_v(16, data1_w))
                expected_asm = expected_name
                if ops:
                    expected_asm += f" {', '.join(ops)}"
                if gds:
                    expected_asm += " gds"
                self.assertEqual(insns[0]["assembly"], expected_asm)
                self.assertTrue(is_opcode_recognized(expected_name))
                self.assertEqual(classify_isa_category(expected_name), "LDS (Local Data Share)")

        # Truncated DS header reports ds_truncated instead of crashing.
        trunc = disasm_rdna2(struct.pack("<I", 0xd8000000 | (0x36 << 18)))
        self.assertEqual(len(trunc), 1)
        self.assertEqual(trunc[0]["opcode"], "ds_truncated")
        self.assertIn("<truncated>", trunc[0]["assembly"])

        # Unknown DS opcode, gds_only with gds=0, no_gds with gds=1, and ds_nop with non-zero offset
        # must all consume 8 bytes so following instructions stay aligned.
        invalid_w0s = [
            0xd8000000 | (0x16 << 18),
            0xd8000000 | (0x3f << 18),
            0xd8000000 | (0xb2 << 18) | (1 << 17),
            0xd8000000 | (0x14 << 18) | 4,
        ]
        for bad_w0 in invalid_w0s:
            stream = struct.pack("<3I", bad_w0, 0, 0xbf810000)
            decoded = disasm_rdna2(stream)
            self.assertEqual([r["length"] for r in decoded], [8, 4])
            self.assertEqual(decoded[1]["opcode"], "s_endpgm")

    def test_ds_distinguishes_offsets_swizzle_and_gds(self):
        """DS single offsets, dual offset0/offset1 pairs, swizzle patterns, and gds must stay distinct."""
        w1_read = (24 << 24) | 2
        self.assertEqual(
            disasm_rdna2(struct.pack("<2I", 0xd8000000 | (0x36 << 18) | 8721, w1_read))[0]["assembly"],
            "ds_read_b32 v24, v2 offset:8721",
        )
        self.assertEqual(
            disasm_rdna2(struct.pack("<2I", 0xd8000000 | (0x36 << 18) | (1 << 17) | 64, w1_read))[0]["assembly"],
            "ds_read_b32 v24, v2 offset:64 gds",
        )

        # Dual offsets emit non-zero offset0 / offset1 independently.
        dual_base = 0xd8000000 | (0x37 << 18)
        self.assertEqual(
            disasm_rdna2(struct.pack("<2I", dual_base, w1_read))[0]["assembly"],
            "ds_read2_b32 v[24:25], v2",
        )
        self.assertEqual(
            disasm_rdna2(struct.pack("<2I", dual_base | 17, w1_read))[0]["assembly"],
            "ds_read2_b32 v[24:25], v2 offset0:17",
        )
        self.assertEqual(
            disasm_rdna2(struct.pack("<2I", dual_base | (34 << 8), w1_read))[0]["assembly"],
            "ds_read2_b32 v[24:25], v2 offset1:34",
        )
        self.assertEqual(
            disasm_rdna2(struct.pack("<2I", dual_base | (34 << 8) | 17, w1_read))[0]["assembly"],
            "ds_read2_b32 v[24:25], v2 offset0:17 offset1:34",
        )

        # Swizzle modes: omitted zero, QUAD_PERM, SWAP, REVERSE, BROADCAST, BITMASK_PERM, decimal fallback.
        swz_base = 0xd8000000 | (0x35 << 18)
        for imm, suffix in (
            (0x0000, ""),
            (0x80e4, " offset:swizzle(QUAD_PERM,0,1,2,3)"),
            (0x041f, " offset:swizzle(SWAP,1)"),
            (0x1c1f, " offset:swizzle(REVERSE,8)"),
            (0x0058, " offset:swizzle(BROADCAST,8,2)"),
            (0x1234, ' offset:swizzle(BITMASK_PERM,"10i01")'),
            (0xc0e4, " offset:49380"),
        ):
            self.assertEqual(
                disasm_rdna2(struct.pack("<2I", swz_base | imm, w1_read))[0]["assembly"],
                f"ds_swizzle_b32 v24, v2{suffix}",
            )

    def test_cli_hex_entry_point(self):
        script = Path(__file__).resolve().parents[1] / "tools" / "disasm_rdna2.py"
        res = subprocess.run(
            [sys.executable, "-B", str(script), "--hex", "000080bf 000081bf"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("s_nop 0", res.stdout)
        self.assertIn("s_endpgm", res.stdout)

    def test_llvm_comparison(self):
        llvm_lib_env = os.environ.get("LLVM_LIB")
        if not llvm_lib_env:
            self.skipTest("LLVM_LIB environment variable not set; verified against golden vectors")

        if llvm_disassemble is None:
            self.fail("pixel_aot.disassemble helper could not be imported")

        resolved_lib = resolve_llvm_lib(llvm_lib_env)

        fixture_instructions = [
            "s_nop 0",
            "s_waitcnt lgkmcnt(0)",
            "s_waitcnt_expcnt null, 0",
            "v_mov_b32_e32 v0, v1",
            "v_mov_b32_e32 v0, s1",
            "v_mov_b32_e32 v2, 1.0",
            "v_mov_b32_e32 v3, 0.5",
            "v_mov_b32_e32 v4, 0",
            "v_mov_b32_e32 v4, src_shared_base",
            "v_mov_b32_e32 v4, src_pops_exiting_wave_id",
            "v_mov_b32_e32 v4, src_scc",
            "v_mov_b32_e32 v4, src_lds_direct",
            "v_add_f32_e32 v3, 0.15915494, v2",
            "v_add_f64 v[0:1], 0.15915494309189532, v[2:3]",
            # Issue #746: VOP1 destination is an SGPR; the LLVM 15 gfx1013 CI job
            # assembles and disassembles this line in test_llvm_comparison.
            "v_readfirstlane_b32 s2, v0",
            "v_cndmask_b32_e32 v3, v1, v2, vcc_lo",
            "v_add_f32_e32 v3, v1, v2",
            "v_sub_f32_e32 v3, v1, v2",
            "v_subrev_f32_e32 v3, v1, v2",
            "v_mul_legacy_f32_e32 v3, v1, v2",
            "v_mul_f32_e32 v3, v1, v2",
            "v_min_f32_e32 v3, v1, v2",
            "v_max_f32_e32 v3, v1, v2",
            "v_min_i32_e32 v3, v1, v2",
            "v_max_i32_e32 v3, v1, v2",
            "v_min_u32_e32 v3, v1, v2",
            "v_max_u32_e32 v3, v1, v2",
            "v_lshrrev_b32_e32 v3, v1, v2",
            "v_ashrrev_i32_e32 v3, v1, v2",
            "v_lshlrev_b32_e32 v3, v1, v2",
            "v_and_b32_e32 v3, v1, v2",
            "v_or_b32_e32 v3, v1, v2",
            "v_xor_b32_e32 v3, v1, v2",
            "v_add_nc_u32_e32 v3, v1, v2",
            "v_sub_nc_u32_e32 v3, v1, v2",
            "v_subrev_nc_u32_e32 v3, v1, v2",
            "v_fmac_f32_e32 v3, v1, v2",
            "v_mac_f32_e32 v3, v1, v2",
            "v_cvt_pkrtz_f16_f32_e32 v3, v1, v2",
            "v_cvt_f32_ubyte0_e32 v5, v1",
            "v_cvt_f32_ubyte1_e32 v5, v1",
            "v_cvt_f32_ubyte2_e32 v5, v1",
            "v_cvt_f32_ubyte3_e32 v5, v1",
            "v_fract_f32_e32 v5, v1",
            "v_trunc_f32_e32 v5, v1",
            "v_ceil_f32_e32 v5, v1",
            "v_rndne_f32_e32 v5, v1",
            "v_floor_f32_e32 v5, v1",
            "v_mov_b32_e32 v4, 0x12345678",
            "v_fmamk_f32 v5, v1, 0x11213141, v3",
            "v_fmaak_f32 v5, v1, v2, 0x11213141",
            "v_interp_p1_f32_e32 v1, v0, attr0.x",
            "v_interp_p2_f32_e32 v2, v1, attr1.y",
            "s_mov_b32 s2, 0x12345678",
            "s_and_saveexec_b32 s2, s0",
            "s_add_u32 s3, s2, 0x100",
            "s_load_dwordx8 s[20:27], s[2:3], 0x0",
            "s_buffer_load_dword s5, s[4:7], s0",
            "s_buffer_load_dwordx4 s[20:23], s[4:7], 0x20",
            "image_load v0, v[2:3], s[0:7] dmask:0x1 dim:SQ_RSRC_IMG_2D",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "image_sample_lz v[64:66], [v32, v16], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_1D_ARRAY",
            "image_sample_l v[64:66], [v32, v16, v8], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_1D_ARRAY",
            "image_sample_b v[64:66], [v32, v16, v8], s[4:11], s[100:103] dmask:0x7 dim:SQ_RSRC_IMG_1D_ARRAY",
            "exp mrt0 v4, off, off, off done",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_cmp_eq_i32 s0, s1",
            "s_cmp_eq_u32 s0, s1",
            "s_cmp_gt_i32 s2, s3",
            "s_cmp_eq_u64 s[0:1], s[2:3]",
            "s_bitcmp0_b32 s0, s1",
            "s_bitcmp0_b64 s[4:5], s6",
            "v_pk_add_u16 v4, v3, 1",
            "v_pk_add_f16 v22, s44, v19 op_sel:[1,0] op_sel_hi:[0,1] neg_lo:[0,1]",
            "v_pk_mul_f16 v23, v22, v20 neg_hi:[1,0]",
            "v_fma_mix_f32 v12, |s44|, -v19, v20 op_sel:[1,1,0] op_sel_hi:[1,0,0]",
            "v_fma_mixhi_f16 v22, -|v19|, v23, v20 op_sel:[1,0,0] op_sel_hi:[1,0,0] clamp",
            "v_cmp_lt_f32_e32 vcc_lo, v1, v2",
            "v_cmp_eq_u32_e32 vcc_lo, s1, v2",
            "v_cmpx_gt_i32_e32 v1, v2",
            "buffer_load_dword v5, off, s[8:11], s3",
            "buffer_load_dword v5, v[0:1], s[8:11], 0 idxen offen offset:4095 glc slc dlc",
            "buffer_load_dword off, s[8:11], s3 lds",
            "buffer_load_dword v5, off, s[8:11], null tfe",
            "buffer_load_dword v5, off, ttmp[0:3], s3",
            "buffer_load_format_d16_xyzw v[5:6], off, s[8:11], 0",
            "buffer_store_dwordx4 v[0:3], v4, s[8:11], 0 offen",
            "buffer_atomic_add v5, v0, s[8:11], s3 offen glc",
            "buffer_atomic_cmpswap_x2 v[5:8], v0, s[8:11], 0 offen glc",
            "buffer_gl0_inv",
            "buffer_gl1_inv",
            "tbuffer_load_format_x v5, off, s[8:11], s3 format:[BUF_FMT_32_FLOAT]",
            "tbuffer_load_format_xyzw v[5:8], v0, s[8:11], s3 format:[BUF_FMT_32_32_32_32_FLOAT] offen offset:64 glc slc dlc",
            "tbuffer_store_format_d16_xy v5, v[0:1], s[8:11], 0 format:[BUF_FMT_16_16_FLOAT] idxen offen",
            "tbuffer_load_format_x v5, off, s[8:11], s3",
            "tbuffer_load_format_x v5, off, s[8:11], s3 format:127",
            "flat_load_dword v4, v[2:3]",
            "flat_load_dwordx4 v[4:7], v[2:3] offset:2047 glc slc dlc",
            "flat_store_dword v[2:3], v8 offset:16",
            "flat_atomic_add v[2:3], v8",
            "flat_atomic_cmpswap_x2 v[4:5], v[2:3], v[8:11] glc",
            "global_load_dword v4, v[2:3], off",
            "global_load_dword v4, v1, s[2:3] offset:-16 glc slc dlc",
            "global_load_dword v1, s[2:3] lds",
            "global_store_byte_d16_hi v1, v8, s[2:3] offset:-2048",
            "global_atomic_add v1, v8, s[2:3]",
            "global_atomic_add v4, v1, v8, s[2:3] glc",
            "scratch_load_dword v4, v1, off offset:16",
            "scratch_load_dword v4, off, s2 offset:-16 glc slc dlc",
            "scratch_load_dword off, s2 offset:16 lds",
            "scratch_store_dwordx2 off, v[8:9], s2 offset:-32",
            "ds_nop",
            "ds_read_b32 v24, v2",
            "ds_read_b32 v24, v2 offset:64",
            "ds_read_b64 v[24:25], v2 offset:16",
            "ds_read_b128 v[24:27], v2",
            "ds_read2_b32 v[24:25], v2 offset0:4 offset1:8",
            "ds_read2st64_b64 v[24:27], v2 offset0:1 offset1:2",
            "ds_write_b32 v2, v8 offset:128",
            "ds_write2_b32 v2, v8, v16 offset0:4 offset1:8",
            "ds_add_rtn_u32 v24, v2, v8 offset:8 gds",
            "ds_mskor_b32 v2, v8, v16 offset:16",
            "ds_wrxchg2_rtn_b32 v[24:25], v2, v8, v16 offset0:1 offset1:2",
            "ds_swizzle_b32 v24, v2",
            "ds_swizzle_b32 v24, v2 offset:swizzle(QUAD_PERM,3,2,1,0)",
            "ds_swizzle_b32 v24, v2 offset:swizzle(SWAP,16)",
            "ds_swizzle_b32 v24, v2 offset:swizzle(REVERSE,8)",
            "ds_swizzle_b32 v24, v2 offset:swizzle(BROADCAST,4,2)",
            'ds_swizzle_b32 v24, v2 offset:swizzle(BITMASK_PERM,"1100p")',
            "ds_append v24 offset:12 gds",
            "ds_consume v24",
            "ds_ordered_count v24, v2 offset:4 gds",
            "ds_gws_sema_release_all gds",
            "ds_gws_barrier v2 offset:8 gds",
            "ds_permute_b32 v24, v2, v8 offset:4",
            "ds_bpermute_b32 v24, v2, v8",
            "ds_write_addtid_b32 v8 offset:16",
            "ds_read_addtid_b32 v24 offset:16",
            "ds_add_src2_f32 v2 offset:32",
            "ds_condxchg32_rtn_b64 v[24:25], v2, v[8:9]",
            "v_mad_f32 v0, v1, v2, v3",
            "v_fma_f32 v0, v1, v2, v3",
            "v_add_f32_e64 v3, v1, v2",
            "v_cndmask_b32_e64 v14, 0, 1.0, vcc_lo",
            "v_add_co_u32 v1, vcc_lo, v2, v3",
            "s_endpgm",
        ]

        assembler = resolve_tool("LLVM_MC", ["llvm-mc-15", "llvm-mc"])
        objcopy = resolve_tool("LLVM_OBJCOPY", ["llvm-objcopy-15", "llvm-objcopy"])

        if not assembler or not objcopy:
            self.fail(
                f"Designated LLVM comparison job requires llvm-mc and llvm-objcopy "
                f"(assembler={assembler}, objcopy={objcopy})"
            )

        try:
            emitted_bytes = assemble_with_llvm_mc(fixture_instructions, assembler, objcopy)
        except Exception as e:
            self.fail(f"Failed to assemble comparison fixture with llvm-mc: {e}")

        self.assertGreater(len(emitted_bytes), 0, "llvm-mc emitted empty machine code")
        self.assertEqual(len(emitted_bytes) % 4, 0, "Emitted bytes length must be multiple of 4")

        try:
            llvm_lines = llvm_disassemble(emitted_bytes, resolved_lib)
        except Exception as e:
            self.fail(
                f"Designated LLVM comparison failed to load or run with '{llvm_lib_env}' "
                f"(resolved: '{resolved_lib}'): {e}"
            )

        rdna2_insns = disasm_rdna2(emitted_bytes)
        self.assertEqual(
            len(llvm_lines), len(fixture_instructions),
            f"LLVM disassembled instruction count mismatch: {llvm_lines}"
        )
        self.assertEqual(
            len(rdna2_insns), len(fixture_instructions),
            f"RDNA 2 disassembled instruction count mismatch: {rdna2_insns}"
        )

        for i, (orig_inst, llvm_line, rdna2_rec) in enumerate(zip(fixture_instructions, llvm_lines, rdna2_insns)):
            orig_op = orig_inst.split()[0].lower()
            llvm_op = llvm_line.split()[0].lower()
            rdna2_op = rdna2_rec["opcode"].lower()
            self.assertEqual(
                llvm_op, orig_op,
                f"Instruction {i}: LLVM opcode '{llvm_op}' does not match original '{orig_op}'"
            )
            self.assertEqual(
                rdna2_op, orig_op,
                f"Instruction {i}: disasm_rdna2 opcode '{rdna2_op}' does not match original '{orig_op}'"
            )
            self.assertEqual(
                rdna2_op, llvm_op,
                f"Instruction {i}: disasm_rdna2 opcode '{rdna2_op}' does not match LLVM opcode '{llvm_op}'"
            )
            if orig_op.startswith(("v_", "exp", "image_", "buffer_", "tbuffer_", "flat_", "global_", "scratch_", "ds_")):
                self.assertEqual(
                    " ".join(rdna2_rec["assembly"].lower().split()),
                    " ".join(llvm_line.lower().split()),
                    f"Instruction {i}: disasm_rdna2 '{rdna2_rec['assembly']}' does not match LLVM '{llvm_line}'"
                )


class TestRdna2Extractor(unittest.TestCase):
    """Test shader extraction from RPF archives against known source-owned fixtures."""

    def test_extract_bytes_match_fixture(self):
        known_code = bytes.fromhex(
            "000080bf"  # s_nop 0
            "7f008cbf"  # s_waitcnt
            "000081bf"  # s_endpgm
        )
        record = envelope(known_code, b"PS_SourceOwned_Fixture")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = tmp_path / "synthetic_shaders.rpf"
            archive.write_bytes(b"RPF_MAGIC_HEADER_PAD\0" + record)

            output_dir = tmp_path / "extracted"
            out_path, manifest_file = extract_shaders(archive, output_dir, target_stages={"pixel"})

            extracted_file = out_path / "pixel" / "PS_SourceOwned_Fixture.bin"
            self.assertTrue(extracted_file.is_file(), "Extracted binary file not found")
            extracted_bytes = extracted_file.read_bytes()

            # Byte-for-byte exact check against source-owned input
            self.assertEqual(extracted_bytes, known_code)

            # Check manifest.json metadata
            self.assertTrue(manifest_file.is_file(), "Manifest not generated")
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(manifest["total_extracted"], 1)
            self.assertEqual(manifest["shaders"][0]["name"], "PS_SourceOwned_Fixture")
            self.assertEqual(manifest["shaders"][0]["native_size"], len(known_code))
            self.assertEqual(manifest["shaders"][0]["stage"], "pixel")

    def test_duplicate_record_names(self):
        """Verify that duplicate record names produce distinct files without overwriting."""
        code_1 = bytes.fromhex("000080bf 000081bf")  # s_nop 0, s_endpgm
        code_2 = bytes.fromhex("000382be 000081bf")  # s_mov_b32, s_endpgm
        record_1 = envelope(code_1, b"PS_DuplicateName")
        record_2 = envelope(code_2, b"PS_DuplicateName")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = tmp_path / "dups.rpf"
            archive.write_bytes(b"RPF\0" + record_1 + record_2)

            output_dir = tmp_path / "extracted_dups"
            out_path, manifest_file = extract_shaders(archive, output_dir, target_stages={"pixel"})

            file_1 = out_path / "pixel" / "PS_DuplicateName.bin"
            file_2 = out_path / "pixel" / "PS_DuplicateName_1.bin"

            self.assertTrue(file_1.is_file(), "First duplicate shader file not found")
            self.assertTrue(file_2.is_file(), "Second duplicate disambiguated file not found")

            self.assertEqual(file_1.read_bytes(), code_1)
            self.assertEqual(file_2.read_bytes(), code_2)

            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(manifest["total_extracted"], 2)
            self.assertEqual(manifest["shaders"][0]["native_file"], "pixel/PS_DuplicateName.bin")
            self.assertEqual(manifest["shaders"][1]["native_file"], "pixel/PS_DuplicateName_1.bin")
            self.assertNotEqual(manifest["shaders"][0]["sha256"], manifest["shaders"][1]["sha256"])

    def test_cli_extraction(self):
        """Invoke extractor via its actual CLI entry point on synthetic fixture."""
        known_code = bytes.fromhex("000080bf 000081bf")
        record = envelope(known_code, b"PS_CliTest")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = tmp_path / "cli_shaders.rpf"
            archive.write_bytes(b"RPF_MAGIC\0" + record)
            output_dir = tmp_path / "cli_out"
            script = Path(__file__).resolve().parents[1] / "tools" / "extract_shaders.py"
            res = subprocess.run(
                [sys.executable, "-B", str(script), str(archive), "--output-dir", str(output_dir), "--stage", "pixel"],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(res.returncode, 0)
            self.assertTrue((output_dir / "pixel" / "PS_CliTest.bin").is_file())
            self.assertEqual((output_dir / "pixel" / "PS_CliTest.bin").read_bytes(), known_code)
            self.assertTrue((output_dir / "manifest.json").is_file())

    def test_cli_limit_must_be_positive(self):
        """A limit below one is rejected instead of extracting a record anyway."""
        records = [envelope(bytes.fromhex("000080bf 000081bf"), b"PS_First"),
                   envelope(bytes.fromhex("000382be 000081bf"), b"PS_Second")]
        script = Path(__file__).resolve().parents[1] / "tools" / "extract_shaders.py"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = tmp_path / "limit.rpf"
            archive.write_bytes(b"RPF\0" + b"".join(records))
            for limit in ("0", "-1"):
                output_dir = tmp_path / f"limit_{limit}"
                res = subprocess.run(
                    [sys.executable, "-B", str(script), str(archive), "--output-dir", str(output_dir),
                     "--stage", "pixel", "--limit", limit],
                    capture_output=True, text=True,
                )
                self.assertEqual(res.returncode, 2, res.stdout + res.stderr)
                self.assertIn("--limit", res.stderr)
                self.assertFalse(output_dir.exists(), "no output may be written for a rejected limit")
            output_dir = tmp_path / "limit_1"
            subprocess.run(
                [sys.executable, "-B", str(script), str(archive), "--output-dir", str(output_dir),
                 "--stage", "pixel", "--limit", "1"],
                capture_output=True, text=True, check=True,
            )
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["total_extracted"], 1)

    def test_extracted_names_round_trip_length_prefixed_records(self) -> None:
        code = bytes.fromhex("000080bf 000081bf")
        names = (b"PS_A", b"PS_" + b"S" * 28, b"!" + b"A" * 32,
                 b"PS_" + b"X" * 122, b"~" + b"B" * 125, b"~" + b"C" * 126)
        legacy_name = b"PS_Legacy"
        legacy = envelope(code, legacy_name)[1:]
        truncated = envelope(code, b"PS_Truncated_Length_Prefix_Fixture")[:-1]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            archive = work / "source_built.rpf"
            archive.write_bytes(b"RPF\0" + b"".join(envelope(code, name) for name in names) + legacy + truncated)
            output, manifest_path = extract_shaders(archive, work / "shaders", target_stages={"pixel"},
                                                    length_prefixed_names=True)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["total_extracted"], len(names) + 1)
            for source_name, record in zip((*names, legacy_name), manifest["shaders"], strict=True):
                name = source_name.decode("ascii")
                with self.subTest(length=len(source_name)):
                    self.assertEqual(record["name"], name)
                    self.assertEqual(record["native_file"], f"pixel/{name}.bin")
                    self.assertEqual((output / record["native_file"]).read_bytes(), code)
                    if source_name != legacy_name:
                        self.assertEqual(native_program(archive, record["name"])[1], code)

    def test_ambiguous_length_prefix_requires_explicit_mode(self) -> None:
        code = bytes.fromhex("000080bf 000081bf")
        legacy_name = b"PS_" + b"L" * 77
        prefixed_name = legacy_name[1:]
        data = envelope(code, legacy_name)[1:]
        self.assertEqual(data, envelope(code, prefixed_name))
        script = Path(__file__).resolve().parents[1] / "tools" / "extract_shaders.py"

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            archive = work / "ambiguous.rpf"
            archive.write_bytes(b"RPF\0" + data)
            for expected_name, options in ((legacy_name, []), (prefixed_name, ["--length-prefixed-names"])):
                name = expected_name.decode("ascii")
                with self.subTest(options=options):
                    output = work / name
                    subprocess.run([sys.executable, "-B", str(script), str(archive), "--output-dir", str(output),
                                    *options], capture_output=True, check=True)
                    manifest_path = output / "manifest.json"
                    record, = json.loads(manifest_path.read_text(encoding="utf-8"))["shaders"]
                    self.assertEqual(record["name"], name)
                    self.assertEqual(record["native_file"], f"pixel/{name}.bin")
                    self.assertEqual((output / record["native_file"]).read_bytes(), code)

    def test_stage_filtering(self):
        known_code = bytes.fromhex("000080bf 000081bf")
        record = envelope(known_code, b"PS_PixelOnly")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = tmp_path / "test.rpf"
            archive.write_bytes(b"HEADER\0" + record)

            output_dir = tmp_path / "extracted_compute"
            out_path, manifest_file = extract_shaders(archive, output_dir, target_stages={"compute"})
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(manifest["total_extracted"], 0)

    def test_false_positive_header_does_not_skip_following_record(self):
        """A candidate header whose deflate payload fails must not skip a valid record inside its span."""
        known_code = bytes.fromhex("000080bf 000081bf")
        valid_record = envelope(known_code, b"PS_AfterFalsePositive")
        bogus_stored = len(valid_record) + 16
        false_header = b"FAKE_HDR\0\0" + struct.pack("<4I", bogus_stored, 0, bogus_stored, 256)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = tmp_path / "false_positive.rpf"
            archive.write_bytes(false_header + b"\x00\x01\x02\x03" + valid_record + bytes(16))

            output_dir = tmp_path / "extracted_fp"
            out_path, manifest_file = extract_shaders(archive, output_dir, target_stages={"pixel"})
            extracted_file = out_path / "pixel" / "PS_AfterFalsePositive.bin"
            self.assertTrue(extracted_file.is_file(), "Valid record after false-positive header was skipped")
            self.assertEqual(extracted_file.read_bytes(), known_code)
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(manifest["total_extracted"], 1)

    def test_oversized_claimed_span_with_unused_data_does_not_skip_following_record(self):
        """Cursor advancement must stop at consumed deflate bytes when unused_data holds another record."""
        code_1 = bytes.fromhex("000080bf 000081bf")
        code_2 = bytes.fromhex("000082be 000081bf")
        record_1 = envelope(code_1, b"PS_OversizedSpan")
        record_2 = envelope(code_2, b"PS_InsideUnusedTail")
        name_end = record_1.index(b"PS_OversizedSpan") + len(b"PS_OversizedSpan")
        stored_1, _, _, decoded_1 = struct.unpack_from("<4I", record_1, name_end + 2)
        payload_1 = record_1[name_end + 18:name_end + 18 + stored_1]
        oversized_stored = stored_1 + len(record_2)
        oversized_record_1 = (record_1[:name_end + 2]
                              + struct.pack("<4I", oversized_stored, 0, oversized_stored, decoded_1)
                              + payload_1)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = tmp_path / "oversized_span.rpf"
            archive.write_bytes(b"RPF\0" + oversized_record_1 + record_2)

            output_dir = tmp_path / "extracted_oversized"
            out_path, manifest_file = extract_shaders(archive, output_dir, target_stages={"pixel"})
            file_1 = out_path / "pixel" / "PS_OversizedSpan.bin"
            file_2 = out_path / "pixel" / "PS_InsideUnusedTail.bin"
            self.assertTrue(file_1.is_file(), "First shader in oversized span not extracted")
            self.assertTrue(file_2.is_file(), "Second shader inside unused_data tail was skipped")
            self.assertEqual(file_1.read_bytes(), code_1)
            self.assertEqual(file_2.read_bytes(), code_2)
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(manifest["total_extracted"], 2)

    def test_disambiguated_stem_does_not_collide_with_existing_or_case_variant(self):
        """Disambiguated stems must not overwrite explicit _1 names or case variants."""
        codes = [
            bytes.fromhex("000080bf 000081bf"),
            bytes.fromhex("000082be 000081bf"),
            bytes.fromhex("000182be 000081bf"),
            bytes.fromhex("0000027e 000081bf"),
        ]
        records = [
            envelope(codes[0], b"PS_Dup"),
            envelope(codes[1], b"PS_Dup"),
            envelope(codes[2], b"PS_Dup_1"),
            envelope(codes[3], b"ps_dup"),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = tmp_path / "collisions.rpf"
            archive.write_bytes(b"RPF\0" + b"".join(records))

            output_dir = tmp_path / "extracted_collisions"
            out_path, manifest_file = extract_shaders(archive, output_dir, target_stages={"pixel"})
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(manifest["total_extracted"], 4)
            paths = [entry["native_file"] for entry in manifest["shaders"]]
            self.assertEqual(len({p.casefold() for p in paths}), 4, f"Colliding output paths: {paths}")
            for entry, expected in zip(manifest["shaders"], codes):
                self.assertEqual((out_path / entry["native_file"]).read_bytes(), expected)

    def test_overlong_record_names_get_bounded_distinct_files(self):
        """A record name near the one-byte length limit must not make the output name unwritable."""
        codes = [bytes.fromhex("000080bf 000081bf"), bytes.fromhex("000382be 000081bf")]
        names = [b"PS_" + b"a" * 250 + b"1", b"PS_" + b"a" * 250 + b"2"]
        records = [envelope(code, name) for code, name in zip(codes, names)]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = tmp_path / "long.rpf"
            archive.write_bytes(b"RPF\0" + b"".join(records))

            out_path, manifest_file = extract_shaders(archive, tmp_path / "extracted_long", target_stages={"pixel"})
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(manifest["total_extracted"], 2)
            self.assertEqual([entry["name"] for entry in manifest["shaders"]],
                             [name.decode("ascii") for name in names])
            paths = [entry["native_file"] for entry in manifest["shaders"]]
            self.assertEqual(len({p.casefold() for p in paths}), 2, f"Colliding output paths: {paths}")
            for path in paths:
                self.assertLessEqual(len(Path(path).name.encode("utf-8")), 160, path)
            for path, expected in zip(paths, codes):
                self.assertEqual((out_path / path).read_bytes(), expected)
        # Truncation counts bytes and never splits a multi-byte character.
        for name in ("é" * 200, "a€" * 100):
            stem = sanitize_filename(name)
            self.assertTrue(name.startswith(stem) and 0 < len(stem.encode("utf-8")) <= 128, stem)

    def test_empty_archive_produces_empty_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = tmp_path / "empty.rpf"
            archive.write_bytes(b"")

            output_dir = tmp_path / "extracted_empty"
            _, manifest_file = extract_shaders(archive, output_dir, target_stages={"all"})
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            self.assertEqual(manifest["total_extracted"], 0)
            self.assertEqual(manifest["shaders"], [])

    def test_invalid_xbch_offset_or_size_rejected(self):
        known_code = bytes.fromhex("000080bf 000081bf")
        program_start = 256
        image = bytearray(program_start) + known_code
        image[:4] = b"FSSM"
        struct.pack_into("<I", image, 192, program_start - 40)
        compressed = zlib.compress(image)

        wrong_size_native = b"XBCH" + bytes(12) + struct.pack("<II", len(image) + 4, 0) + compressed
        wrong_size_program = (bytes(24) + b"XDAT" + struct.pack("<II", 1, 0)
                              + struct.pack("<III", 13, 48, len(wrong_size_native)) + wrong_size_native)
        self.assertIsNone(extract_native_rdna2_code(wrong_size_program))

        valid_native = b"XBCH" + bytes(12) + struct.pack("<II", len(image), 0) + compressed
        overlapping_program = (b"XBCH" + bytes(20) + b"XDAT" + struct.pack("<II", 1, 0)
                               + struct.pack("<III", 13, 0, len(valid_native)) + valid_native)
        self.assertIsNone(extract_native_rdna2_code(overlapping_program))


class TestRdna2Scanner(unittest.TestCase):
    """Test scanner instruction recognition vs full shader translator acceptance."""

    def test_accepted_full_shader(self):
        """Canonical straight-line pixel shader accepted by pixel_aot translator."""
        canonical_shader = [
            "s_mov_b64 s[50:51], exec",
            "s_wqm_b64 exec, exec",
            "s_mov_b32 m0, s24",
            "v_interp_p1_f32_e32 v8, v0, attr0.x",
            "v_interp_p1_f32_e32 v9, v0, attr0.y",
            "v_interp_p2_f32_e32 v9, v1, attr0.y",
            "v_interp_p2_f32_e32 v8, v1, attr0.x",
            "s_and_b64 exec, exec, s[50:51]",
            "s_mov_b32 s21, 0",
            "s_load_dwordx8 s[28:35], s[20:21], null",
            "s_mov_b32 s40, 1",
            "s_mov_b32 s41, 2",
            "s_mov_b32 s42, 3",
            "s_mov_b32 s43, 4",
            "s_waitcnt lgkmcnt(0)",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_endpgm",
        ]
        res = analyze_shader_instructions("canonical_ps", canonical_shader)
        self.assertTrue(res["opcodes_recognized"])
        self.assertTrue(res["translator_accepted"])
        self.assertEqual(res["compatibility_category"], "accepted")
        self.assertTrue(res["is_fully_supported"])

    def test_accepted_full_shader_wave32(self):
        """Wave32 pixel shader accepted by pixel_aot translator and recognized by scanner."""
        wave32_shader = [
            "s_mov_b64 s14, exec_lo",
            "s_wqm_b64 exec_lo, exec_lo",
            "s_mov_b32 m0, s24",
            "v_interp_p1_f32_e32 v8, v0, attr0.x",
            "v_interp_p1_f32_e32 v9, v0, attr0.y",
            "v_interp_p2_f32_e32 v9, v1, attr0.y",
            "v_interp_p2_f32_e32 v8, v1, attr0.x",
            "s_and_b64 exec_lo, exec_lo, s14",
            "s_mov_b32 s21, 0",
            "s_load_dwordx8 s[28:35], s[20:21], null",
            "s_mov_b32 s40, 1",
            "s_mov_b32 s41, 2",
            "s_mov_b32 s42, 3",
            "s_mov_b32 s43, 4",
            "s_waitcnt lgkmcnt(0)",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_endpgm",
        ]
        res = analyze_shader_instructions("canonical_ps_wave32", wave32_shader)
        self.assertTrue(res["opcodes_recognized"])
        self.assertTrue(res["translator_accepted"])
        self.assertEqual(res["compatibility_category"], "accepted")
        self.assertTrue(res["is_fully_supported"])

    def test_accepted_full_shader_wave32_direct_restore(self):
        """Wave32 pixel shader with direct exec_lo restore via s_mov_b64."""
        wave32_shader = [
            "s_mov_b64 s14, exec_lo",
            "s_wqm_b64 exec_lo, exec_lo",
            "s_mov_b32 m0, s24",
            "v_interp_p1_f32_e32 v8, v0, attr0.x",
            "v_interp_p1_f32_e32 v9, v0, attr0.y",
            "v_interp_p2_f32_e32 v9, v1, attr0.y",
            "v_interp_p2_f32_e32 v8, v1, attr0.x",
            "s_mov_b64 exec_lo, s14",
            "s_mov_b32 s21, 0",
            "s_load_dwordx8 s[28:35], s[20:21], null",
            "s_mov_b32 s40, 1",
            "s_mov_b32 s41, 2",
            "s_mov_b32 s42, 3",
            "s_mov_b32 s43, 4",
            "s_waitcnt lgkmcnt(0)",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_endpgm",
        ]
        res = analyze_shader_instructions("canonical_ps_wave32_direct", wave32_shader)
        self.assertTrue(res["opcodes_recognized"])
        self.assertTrue(res["translator_accepted"])
        self.assertEqual(res["compatibility_category"], "accepted")
        self.assertTrue(res["is_fully_supported"])

    def test_accepted_full_shader_wave32_b32_lifecycle(self):
        """Native Wave32 pixel shader with 32-bit save, WQM and restore operations."""
        wave32_shader = [
            "s_mov_b32 s14, exec_lo",
            "s_wqm_b32 exec_lo, exec_lo",
            "s_mov_b32 m0, s24",
            "v_interp_p1_f32_e32 v8, v0, attr0.x",
            "v_interp_p1_f32_e32 v9, v0, attr0.y",
            "v_interp_p2_f32_e32 v9, v1, attr0.y",
            "v_interp_p2_f32_e32 v8, v1, attr0.x",
            "s_and_b32 exec_lo, exec_lo, s14",
            "s_mov_b32 s21, 0",
            "s_load_dwordx8 s[28:35], s[20:21], null",
            "s_mov_b32 s40, 1",
            "s_mov_b32 s41, 2",
            "s_mov_b32 s42, 3",
            "s_mov_b32 s43, 4",
            "s_waitcnt lgkmcnt(0)",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_endpgm",
        ]
        res = analyze_shader_instructions("canonical_ps_wave32_b32", wave32_shader, wave_size=32)
        self.assertTrue(res["opcodes_recognized"])
        self.assertTrue(res["translator_accepted"])
        self.assertEqual(res["compatibility_category"], "accepted")
        self.assertTrue(res["is_fully_supported"])

        # Low-only lifecycle rejected without validated wave32 profile
        res_default = analyze_shader_instructions("canonical_ps_wave32_b32", wave32_shader)
        self.assertFalse(res_default["translator_accepted"])
        self.assertIn("requires a validated wave32 guest profile", res_default["lower_error"])

        res_w64 = analyze_shader_instructions("canonical_ps_wave32_b32", wave32_shader, wave_size=64)
        self.assertFalse(res_w64["translator_accepted"])
        self.assertIn("requires a validated wave32 guest profile", res_w64["lower_error"])

        res_invalid = analyze_shader_instructions("canonical_ps_wave32_b32", wave32_shader, wave_size=16)
        self.assertFalse(res_invalid["translator_accepted"])
        self.assertIn("Invalid guest wave size: 16", res_invalid["lower_error"])

    def test_exec_mask_scanner_translator_agreement(self):
        """Scanner instruction support matches pixel_aot translator acceptance for exec mask ops."""
        from pixel_aot import lower
        from pixel_aot_check import assembly
        from pixel_cb_read import scalar_writes

        base = list(assembly(False))
        save_idx = base.index("s_mov_b64 s[50:51], exec")
        wqm_idx = base.index("s_wqm_b64 exec, exec")
        restore_idx = base.index("s_and_b64 exec, exec, s[50:51]")

        def translator_accepts_save(line):
            shader = list(base)
            shader[save_idx] = line
            wave_size = None
            if line.startswith("s_mov_b32"):
                shader[wqm_idx] = "s_wqm_b32 exec_lo, exec_lo"
                shader[restore_idx] = "s_and_b32 exec_lo, exec_lo, s50"
                wave_size = 32
            elif "s50" in line:
                shader[wqm_idx] = "s_wqm_b64 exec_lo, exec_lo"
                shader[restore_idx] = "s_and_b64 exec_lo, exec_lo, s50"
            try:
                lower(shader, 20, 24, [1, 2, 3, 4], wave_size=wave_size)
                return True
            except Exception:
                return False

        def translator_accepts_restore(line):
            shader = list(base)
            wave_size = None
            if line.startswith(("s_mov_b32", "s_and_b32")):
                shader[save_idx] = "s_mov_b32 s50, exec_lo"
                shader[wqm_idx] = "s_wqm_b32 exec_lo, exec_lo"
                wave_size = 32
            elif "50" in line or "exec_lo" in line or "s14" in line:
                shader[save_idx] = "s_mov_b64 s50, exec_lo" if "50" in line else "s_mov_b64 s14, exec_lo"
                shader[wqm_idx] = "s_wqm_b64 exec_lo, exec_lo"
            shader[restore_idx] = line
            try:
                lower(shader, 20, 24, [1, 2, 3, 4], wave_size=wave_size)
                return True
            except Exception:
                return False

        save_cases = [
            ("s_mov_b64 s[50:51], exec", True),
            ("s_mov_b64 s[50:51], exec_lo", True),
            ("s_mov_b64 s50, exec_lo", True),
            ("s_mov_b64 s50, exec", True),
            ("s_mov_b32 s50, exec_lo", True),
            ("s_mov_b32 s50, exec", True),
            ("s_mov_b32 s[50:51], exec", False),
            ("s_mov_b32 s[50:51], exec_lo", False),
            ("s_mov_b32 s50, exec_hi", False),
        ]

        restore_cases = [
            ("s_mov_b64 exec, s[50:51]", True),
            ("s_mov_b64 exec_lo, s[50:51]", True),
            ("s_mov_b64 exec, s50", True),
            ("s_mov_b64 exec_lo, s50", True),
            ("s_mov_b32 exec, s50", True),
            ("s_mov_b32 exec_lo, s50", True),
            ("s_mov_b32 exec, s[50:51]", False),
            ("s_mov_b32 exec_lo, s[50:51]", False),
            ("s_and_b64 exec, exec, s[50:51]", True),
            ("s_and_b64 exec_lo, exec_lo, s[50:51]", True),
            ("s_and_b64 exec, exec, s50", True),
            ("s_and_b64 exec_lo, exec_lo, s50", True),
            ("s_and_b32 exec, exec, s50", True),
            ("s_and_b32 exec_lo, exec_lo, s50", True),
            ("s_and_b32 exec, exec, s[50:51]", False),
            ("s_and_b32 exec_lo, exec_lo, s[50:51]", False),
            ("s_and_b64 exec, exec, s14", True),
        ]

        for line, expected in save_cases:
            self.assertEqual(is_instruction_supported(line), expected, f"scanner mismatch on {line}")
            self.assertEqual(translator_accepts_save(line), expected, f"translator mismatch on save {line}")

        for line, expected in restore_cases:
            self.assertEqual(is_instruction_supported(line), expected, f"scanner mismatch on {line}")
            self.assertEqual(translator_accepts_restore(line), expected, f"translator mismatch on restore {line}")

        self.assertEqual(scalar_writes("s_mov_b64 s50, exec_lo"), range(50, 52))
        self.assertEqual(scalar_writes("s_mov_b64 s[50:51], exec"), range(50, 52))
        self.assertEqual(scalar_writes("s_mov_b32 s50, exec_lo"), range(50, 51))
        self.assertEqual(scalar_writes("s_mov_b64 exec, s[50:51]"), ())
        self.assertEqual(scalar_writes("s_mov_b32 exec_lo, s50"), ())

        # Mixed-width stream: 64-bit WQM expansion followed by 32-bit low-only restore must be rejected
        mixed_shader = list(base)
        mixed_shader[save_idx] = "s_mov_b64 s50, exec_lo"
        mixed_shader[wqm_idx] = "s_wqm_b64 exec_lo, exec_lo"
        mixed_shader[restore_idx] = "s_and_b32 exec_lo, exec_lo, s50"
        with self.assertRaises(ValueError):
            lower(mixed_shader, 20, 24, [1, 2, 3, 4])

        # Clobbered high SGPR: full-width restore must verify both halves intact
        clobbered_shader = list(base)
        clobbered_shader[save_idx] = "s_mov_b64 s50, exec_lo"
        clobbered_shader[wqm_idx] = "s_wqm_b64 exec_lo, exec_lo"
        clobbered_shader.insert(restore_idx, "s_mov_b32 s51, 0")
        clobbered_shader[restore_idx + 1] = "s_mov_b64 exec, s50"
        with self.assertRaises(ValueError):
            lower(clobbered_shader, 20, 24, [1, 2, 3, 4])

        # Wave32 32-bit execution mask stream rejected without validated wave32 profile
        b32_shader = list(base)
        b32_shader[save_idx] = "s_mov_b32 s50, exec_lo"
        b32_shader[wqm_idx] = "s_wqm_b32 exec_lo, exec_lo"
        b32_shader[restore_idx] = "s_and_b32 exec_lo, exec_lo, s50"
        with self.assertRaises(ValueError):
            lower(b32_shader, 20, 24, [1, 2, 3, 4])
        with self.assertRaises(ValueError):
            lower(b32_shader, 20, 24, [1, 2, 3, 4], wave_size=64)
        with self.assertRaises(ValueError):
            lower(b32_shader, 20, 24, [1, 2, 3, 4], wave_size=16)
        with self.assertRaises(ValueError):
            lower(b32_shader, 20, 24, [1, 2, 3, 4], wave_size="wave32")
        with self.assertRaises(TypeError):
            lower(b32_shader, 20, 24, [1, 2, 3, 4], wave_profile="wave32")
        self.assertIsNotNone(lower(b32_shader, 20, 24, [1, 2, 3, 4], wave_size=32))

    def test_rounding_instructions_accepted(self):
        """v_ceil_f32 and v_rndne_f32 are known RDNA 2 opcodes that pixel_aot lowers."""
        canonical = [
            "s_mov_b64 s[50:51], exec",
            "s_wqm_b64 exec, exec",
            "s_mov_b32 m0, s24",
            "v_interp_p1_f32_e32 v8, v0, attr0.x",
            "v_interp_p1_f32_e32 v9, v0, attr0.y",
            "v_interp_p2_f32_e32 v9, v1, attr0.y",
            "v_interp_p2_f32_e32 v8, v1, attr0.x",
            "s_and_b64 exec, exec, s[50:51]",
            "s_mov_b32 s21, 0",
            "s_load_dwordx8 s[28:35], s[20:21], null",
            "s_mov_b32 s40, 1",
            "s_mov_b32 s41, 2",
            "s_mov_b32 s42, 3",
            "s_mov_b32 s43, 4",
            "s_waitcnt lgkmcnt(0)",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "v_ceil_f32_e32 v12, v12",
            "v_rndne_f32_e64 v13, v13",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_endpgm",
        ]
        res = analyze_shader_instructions("rounding_ps", canonical)
        self.assertTrue(res["opcodes_recognized"])
        self.assertTrue(res["translator_accepted"])
        self.assertEqual(res["compatibility_category"], "accepted")
        self.assertTrue(res["is_fully_supported"])
    def test_trig_instructions_accepted(self):
        """v_sin_f32 and v_cos_f32 are RDNA 2 opcodes that pixel_aot lowers in both encodings."""
        trig_shader = [
            "s_mov_b64 s[50:51], exec",
            "s_wqm_b64 exec, exec",
            "s_mov_b32 m0, s24",
            "v_interp_p1_f32_e32 v8, v0, attr0.x",
            "v_interp_p1_f32_e32 v9, v0, attr0.y",
            "v_interp_p2_f32_e32 v9, v1, attr0.y",
            "v_interp_p2_f32_e32 v8, v1, attr0.x",
            "s_and_b64 exec, exec, s[50:51]",
            "s_mov_b32 s21, 0",
            "s_load_dwordx8 s[28:35], s[20:21], null",
            "s_mov_b32 s40, 1",
            "s_mov_b32 s41, 2",
            "s_mov_b32 s42, 3",
            "s_mov_b32 s43, 4",
            "s_waitcnt lgkmcnt(0)",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "v_sin_f32_e32 v12, v12",
            "v_cos_f32_e64 v13, -|v13| clamp",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_endpgm",
        ]
        res = analyze_shader_instructions("trig_ps", trig_shader)
        self.assertTrue(res["opcodes_recognized"])
        self.assertTrue(res["translator_accepted"])
        self.assertEqual(res["compatibility_category"], "accepted")
        self.assertTrue(res["is_fully_supported"])

    def test_opcodes_recognized_but_translator_rejected(self):
        """Shader with valid RDNA 2 opcodes whose structure violates pixel_aot constraints."""
        valid_rdna2_but_rejected = [
            "s_mov_b64 s[50:51], exec",
            "s_branch 0",  # s_branch is a valid RDNA 2 opcode recognized by scanner, but rejected by translator
            "s_mov_b32 m0, s24",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_endpgm",
        ]
        res = analyze_shader_instructions("branched_ps", valid_rdna2_but_rejected)
        # All opcodes are valid RDNA 2 instructions
        self.assertTrue(res["opcodes_recognized"])
        # But translator rejects the non-straight-line structure
        self.assertFalse(res["translator_accepted"])
        self.assertEqual(res["compatibility_category"], "opcode_recognized_translator_rejected")
        self.assertFalse(res["is_fully_supported"])
        self.assertIsNotNone(res["lower_error"])

    def test_unsupported_opcode_shader(self):
        """Shader containing opcodes unknown or unsupported by the ISA scanner."""
        unknown_opcode_shader = [
            "s_mov_b64 s[50:51], exec",
            "v_imaginary_opcode_xyz v0, v1, v2",
            "s_endpgm",
        ]
        res = analyze_shader_instructions("unknown_ps", unknown_opcode_shader)
        self.assertFalse(res["opcodes_recognized"])
        self.assertFalse(res["translator_accepted"])
        self.assertEqual(res["compatibility_category"], "unsupported_opcodes")
        self.assertIn("v_imaginary_opcode_xyz", res["unsupported_opcodes"])

    def test_vopc_classification(self):
        """Vector compare opcodes with type suffixes must classify as VOPC, not VALU."""
        for opcode in ("v_cmp_lt_f32", "v_cmp_eq_u32", "v_cmp_gt_i32", "v_cmp_le_f16"):
            self.assertEqual(classify_isa_category(opcode), "VOPC (Vector Compare)", opcode)
        for pred in ("f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "tru"):
            self.assertEqual(classify_isa_category(f"v_cmp_{pred}_f32"), "VOPC (Vector Compare)")
        self.assertEqual(classify_isa_category("v_add_f32"), "VOP / VALU (Vector ALU - VOP1/VOP2/VOP3)")

    def test_vopc_e32_destination_must_be_vcc(self):
        """An explicit _e32 compare may only write vcc; suffix-less and _e64 may name an SGPR."""
        families = [f"v_cmp_{op}_{kind}" for kind in ("f16", "i32", "u32", "i16", "u16")
                    for op in ("lt", "eq")] + ["v_cmp_class_f32", "v_cmp_class_f16"]
        f32_predicates = [f"v_cmp_{op}_f32" for op in ("f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "tru")]
        for name in set(families) | set(f32_predicates):
            for dst in ("s52", "s[52:53]"):
                self.assertFalse(is_instruction_supported(f"{name}_e32 {dst}, s44, v18"), (name, dst))
                self.assertTrue(is_instruction_supported(f"{name} {dst}, s44, v18"), (name, dst))
                self.assertTrue(is_instruction_supported(f"{name}_e64 {dst}, s44, v18"), (name, dst))
            for dst in ("vcc", "vcc_lo"):
                self.assertTrue(is_instruction_supported(f"{name}_e32 {dst}, s44, v18"), (name, dst))

    def test_f32_vector_compares_recognition(self):
        """All 16 binary32 vector compare predicates must be recognized in base opcodes and instruction forms."""
        for op in ("f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq", "nlt", "tru"):
            base = f"v_cmp_{op}_f32"
            self.assertTrue(is_opcode_recognized(base), f"Base opcode not recognized: {base}")
            self.assertTrue(is_opcode_recognized(f"{base}_e32"), f"e32 not recognized: {base}_e32")
            self.assertTrue(is_opcode_recognized(f"{base}_e64"), f"e64 not recognized: {base}_e64")
            self.assertTrue(is_instruction_supported(f"{base}_e32 vcc_lo, s44, v18"))
            self.assertTrue(is_instruction_supported(f"{base}_e64 s52, -s44, |v18|"))
            self.assertTrue(is_instruction_supported(f"{base}_sdwa vcc_lo, s44, v18 src0_sel:DWORD src1_sel:DWORD"))

    def test_vop3p_classification(self):
        """Packed and mixed-precision opcodes must classify as VOP3P."""
        for opcode in (
            "v_pk_add_f16", "v_pk_add_u16", "v_pk_sub_i16", "v_pk_fma_f16",
            "v_dot2_f32_f16", "v_dot4_i32_i8", "v_dot8_u32_u4",
            "v_fma_mix_f32", "v_fma_mixlo_f16", "v_fma_mixhi_f16",
        ):
            self.assertEqual(
                classify_isa_category(opcode),
                "VOP3P (Vector Packed / Mixed-Precision)",
                opcode,
            )

    def test_sopk_scalar_is_arithmetic_supported(self):
        """The scanner accepts s_addk_i32 and s_mulk_i32 with a scalar destination and a hex or negative immediate, and rejects a vector destination."""
        for line in ("s_addk_i32 s50, 0xfffe", "s_mulk_i32 s50, 0x3",
                     "s_addk_i32 s50, -2"):
            self.assertTrue(is_instruction_supported(line), line)
        self.assertFalse(is_instruction_supported("s_mulk_i32 v50, 0x3"))

    def test_literal_mad_forms_take_four_operands(self):
        """v_madak_f32 and v_madmk_f32 are supported with four operands; e64 and a missing operand are not."""
        for line in ("v_madak_f32 v5, v1, v2, 0x11213141", "v_madmk_f32 v5, v1, 0x11213141, v3"):
            self.assertTrue(is_instruction_supported(line), line)
        for line in ("v_madak_f32_e64 v5, v1, v2, 0x11213141", "v_madmk_f32 v5, v1, 0x11213141",
                     "v_madak_f32_dpp v5, v1, v2, 0x11213141 quad_perm:[0,1,2,3]",
                     "v_madak_f32 v5, v1, v2, v3", "v_madmk_f32 v5, v1, v2, v3"):
            self.assertFalse(is_instruction_supported(line), line)

    def test_non_packed_int16_vop3_forms(self):
        """The 12 non-packed 16-bit integer VOP3 forms take two integer sources; modifiers and wide operands are not supported."""
        names = ("add_nc_u16", "add_nc_i16", "sub_nc_u16", "sub_nc_i16", "min_u16", "max_u16", "min_i16",
                 "max_i16", "mul_lo_u16", "lshlrev_b16", "lshrrev_b16", "ashrrev_i16")
        self.assertEqual(set(names), set(VOP3_INT16))
        for name in names:
            line = f"v_{name} v5, v1, v2"
            self.assertTrue(is_instruction_supported(line), line)
            self.assertTrue(is_opcode_recognized(f"v_{name}"), name)
        for line in ("v_add_nc_u16_e64 v5, v1, v2", "v_sub_nc_i16 v5, v1, 0x1234",
                     "v_add_nc_i16 v5, v1, 4", "v_add_nc_u16 v5, v1, -16", "v_add_nc_u16 v5, v1, 64",
                     "v_add_nc_u16 v5, v1, 0x00001234", "v_add_nc_u16 v255, s105, v2"):
            self.assertTrue(is_instruction_supported(line), line)
        for line in ("v_add_nc_u16 v5, v1, v2 op_sel:[0,0,1]", "v_add_nc_u16 v5, v1, v2 clamp",
                     "v_add_nc_u16 v5, v1, 1.0", "v_add_nc_u16 v5, v1, 0x10000", "v_add_nc_u16 v5, v1, 65",
                     "v_add_nc_u16 v5, v1, -17", "v_add_nc_u16_dpp v5, v1, v2 quad_perm:[0,1,2,3]",
                     "v_add_nc_u16_sdwa v5, v1, v2 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0",
                     "v_add_nc_u16 v5, -v1, v2", "v_add_nc_u16 v256, v1, v2", "v_add_nc_u16 v5, v256, v2",
                     "v_add_nc_u16 v5, s106, v2"):
            self.assertFalse(is_instruction_supported(line), line)

    def test_non_packed_int16_ternary_forms(self):
        """The 8 three-operand 16-bit integer VOP3 forms take three integer sources; clamp, op_sel and wide operands are not supported."""
        names = ("mad_u16", "mad_i16", "min3_i16", "min3_u16", "max3_i16", "max3_u16", "med3_i16", "med3_u16")
        self.assertEqual(set(names), set(VOP3_INT16_TERNARY))
        for name in names:
            line = f"v_{name} v5, v1, v2, v3"
            self.assertTrue(is_instruction_supported(line), line)
            self.assertTrue(is_opcode_recognized(f"v_{name}"), name)
            for bad in (f"v_{name} v5, v1, v2", f"v_{name} v5, v1, v2, v3 clamp",
                        f"v_{name} v5, v1, v2, v3 op_sel:[0,0,0,1]", f"v_{name} v5, -v1, v2, v3",
                        f"v_{name} v5, v1, 1.0, v3", f"v_{name} v5, v1, v2, 0x10000",
                        f"v_{name}_dpp v5, v1, v2, v3 quad_perm:[0,1,2,3]", f"v_{name} v256, v1, v2, v3",
                        f"v_{name} v5, v1, v2, s106"):
                self.assertFalse(is_instruction_supported(bad), bad)
        for line in ("v_mad_u16_e64 v5, v1, v2, v3", "v_max3_i16 v5, v1, v2, 4", "v_med3_u16 v5, v1, -16, 64",
                     "v_mad_i16 v5, v1, 0x0012, s105"):
            self.assertTrue(is_instruction_supported(line), line)
        for line in ("v_mad_u32_u16 v5, v1, v2, v3", "v_med3_f16 v5, v1, v2, v3"):
            self.assertFalse(is_instruction_supported(line), line)

    def test_float_to_word_conversions_e32_and_e64(self):
        """v_cvt_u32_f32 and v_cvt_i32_f32 accept plain e32 and e64 with optional source abs/neg;
        v_cvt_flr_i32_f32 and v_cvt_rpi_i32_f32 stay out of generic scanner support until arbitrary runtime
        exceptional inputs are backed by a native RDNA reference."""
        for line in ("v_cvt_u32_f32_e32 v12, v13", "v_cvt_i32_f32_e32 v12, s44",
                     "v_cvt_u32_f32_e32 v12, 0x40400000", "v_cvt_u32_f32_e64 v12, v13",
                     "v_cvt_i32_f32_e64 v12, s44", "v_cvt_u32_f32_e64 v12, |s44|",
                     "v_cvt_i32_f32_e64 v12, -s44", "v_cvt_i32_f32_e64 v12, -|v13|",
                     "v_cvt_u32_f32_e64 v12, abs(v13)", "v_cvt_i32_f32_e64 v12, neg(0x40400000)",
                     "v_cvt_u32_f32_e64 v12, |0xc0300000|"):
            self.assertTrue(is_instruction_supported(line), line)
        for line in ("v_cvt_u32_f32_e64 v12, s44 clamp", "v_cvt_i32_f32_e64 v12, s44 omod:2",
                     "v_cvt_i32_f32_e32 v12, 1", "v_cvt_u32_f32_e32 v12, 0.5",
                     "v_cvt_u32_f32_e64 v12, 0.5", "v_cvt_i32_f32_e64 v12, -1",
                     "v_cvt_i32_f32_e64 v12, -0x40400000", "v_cvt_u32_f32_e64 v12, |-s44|",
                     "v_cvt_u32_f32_e32  v12, v13", "v_cvt_u32_f32_e64  v12, v13",
                     "v_cvt_flr_i32_f32_e32 v12, v13", "v_cvt_rpi_i32_f32_e64 v12, -v13"):
            self.assertFalse(is_instruction_supported(line), line)

    def test_disasm_cvt_flr_and_rpi_e32_and_e64(self):
        """tools/disasm_rdna2.py decodes v_cvt_rpi_i32_f32 and v_cvt_flr_i32_f32 in VOP1 and VOP3."""
        cases = (
            (bytes([0x00, 0x19, 0x02, 0x7e]), "v_cvt_rpi_i32_f32_e32", "v_cvt_rpi_i32_f32_e32 v1, v0"),
            (bytes([0x00, 0x1b, 0x02, 0x7e]), "v_cvt_flr_i32_f32_e32", "v_cvt_flr_i32_f32_e32 v1, v0"),
            (bytes([0x01, 0x00, 0x8d, 0xd5, 0x00, 0x01, 0x01, 0x02]), "v_cvt_flr_i32_f32_e64", "v_cvt_flr_i32_f32_e64 v1, v0"),
            (bytes([0x01, 0x01, 0x8c, 0xd5, 0x00, 0x01, 0x01, 0x22]), "v_cvt_rpi_i32_f32_e64", "v_cvt_rpi_i32_f32_e64 v1, -|v0|"),
        )
        for raw, expected_op, expected_asm in cases:
            length, op_name, asm, _ = decode_instruction(raw, 0)
            self.assertEqual(length, len(raw))
            self.assertEqual(op_name, expected_op)
            self.assertEqual(asm, expected_asm)

    def test_inline_slash_comments_and_blank_lines(self):
        """Inline // comments must be stripped and blank/comment lines must not inflate counts."""
        self.assertTrue(is_instruction_supported("s_endpgm // shader exit"))
        res = analyze_shader_instructions(
            "commented_ps",
            [
                "",
                "; header comment",
                "// llvm-style comment",
                "s_endpgm // shader exit",
                "v_imaginary_opcode_xyz v0, v1 // unsupported",
            ],
        )
        self.assertEqual(res["total_instructions"], 2)
        self.assertEqual(res["supported_instructions"], 1)
        self.assertEqual(res["unsupported_count"], 1)
        self.assertEqual(res["support_percentage"], 50.0)
        self.assertEqual(res["unsupported_details"][0]["raw"], "v_imaginary_opcode_xyz v0, v1")

    def test_cli_scanner(self):
        """Invoke scanner via its actual CLI entry point on synthetic assembly file."""
        canonical_shader = "\n".join([
            "s_mov_b64 s[50:51], exec",
            "s_wqm_b64 exec, exec",
            "s_mov_b32 m0, s24",
            "v_interp_p1_f32_e32 v8, v0, attr0.x",
            "v_interp_p1_f32_e32 v9, v0, attr0.y",
            "v_interp_p2_f32_e32 v9, v1, attr0.y",
            "v_interp_p2_f32_e32 v8, v1, attr0.x",
            "s_and_b64 exec, exec, s[50:51]",
            "s_mov_b32 s21, 0",
            "s_load_dwordx8 s[28:35], s[20:21], null",
            "s_mov_b32 s40, 1",
            "s_mov_b32 s41, 2",
            "s_mov_b32 s42, 3",
            "s_mov_b32 s43, 4",
            "s_waitcnt lgkmcnt(0)",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_endpgm",
        ])
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            shader_file = tmp_path / "canonical.s"
            shader_file.write_text(".text\n.globl pixel\n.type pixel,@function\npixel:\n"
                                   + canonical_shader + "\n.size pixel, .-pixel\n", encoding="utf-8")
            out_dir = tmp_path / "scan_reports"
            script = Path(__file__).resolve().parents[1] / "tools" / "scan_unsupported_shaders.py"

            # The scanner prints block characters when its stdout encoding allows them, so both
            # sides use UTF-8 whatever the caller's PYTHONIOENCODING and code page (#844).
            def scan():
                return subprocess.run(
                    [sys.executable, "-B", str(script), str(shader_file), "--output-dir", str(out_dir)],
                    capture_output=True,
                    encoding="utf-8",
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                    check=True,
                )

            res = scan()
            self.assertEqual(res.returncode, 0)
            self.assertIn("\N{FULL BLOCK}", res.stdout)  # the progress bar's UTF-8 form, as in #844
            self.assertIn("SCAN RESULTS SUMMARY", res.stdout)
            self.assertIn("Full Shaders Accepted by Translator:       1", res.stdout)
            self.assertTrue((out_dir / "shader_scan_report.json").is_file())
            self.assertTrue((out_dir / "shader_scan_report.md").is_file())
            with shader_file.open("a", encoding="utf-8") as stream:
                stream.write(".long 0\n")
            result = scan()
            self.assertIn("Full Shaders Accepted by Translator:       0", result.stdout)

    def test_report_generation(self):
        canonical_shader = [
            "s_mov_b64 s[50:51], exec",
            "s_wqm_b64 exec, exec",
            "s_mov_b32 m0, s24",
            "v_interp_p1_f32_e32 v8, v0, attr0.x",
            "v_interp_p1_f32_e32 v9, v0, attr0.y",
            "v_interp_p2_f32_e32 v9, v1, attr0.y",
            "v_interp_p2_f32_e32 v8, v1, attr0.x",
            "s_and_b64 exec, exec, s[50:51]",
            "s_mov_b32 s21, 0",
            "s_load_dwordx8 s[28:35], s[20:21], null",
            "s_mov_b32 s40, 1",
            "s_mov_b32 s41, 2",
            "s_mov_b32 s42, 3",
            "s_mov_b32 s43, 4",
            "s_waitcnt lgkmcnt(0)",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_endpgm",
        ]
        rejected_shader = [
            "s_mov_b64 s[50:51], exec",
            "s_branch 0",
            "s_endpgm",
        ]
        results = [
            analyze_shader_instructions("accepted_ps", canonical_shader),
            analyze_shader_instructions("rejected_ps", rejected_shader),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            json_file, md_file = generate_reports(results, tmp_path)
            self.assertTrue(json_file.is_file())
            self.assertTrue(md_file.is_file())

            data = json.loads(json_file.read_text(encoding="utf-8"))
            summary = data["summary"]
            self.assertEqual(summary["total_shaders_scanned"], 2)
            self.assertEqual(summary["translator_accepted"], 1)
            self.assertEqual(summary["opcodes_recognized_translator_rejected"], 1)

    def test_detect_shader_stage(self):
        """Verify pipeline stage detection from file names and relative paths."""
        self.assertEqual(detect_shader_stage("shaders/pixel/ps_draw.bin"), "pixel")
        self.assertEqual(detect_shader_stage("cs_dispatch.bin"), "compute")
        self.assertEqual(detect_shader_stage("shaders/vertex/vs_main.s"), "vertex")
        self.assertEqual(detect_shader_stage("unknown_kernel.bin"), "other")

    def test_binary_shader_scanning(self):
        """Verify scanning raw .bin files directly with disasm_rdna2 without LLVM."""
        raw_code = bytes.fromhex(
            "000080bf"  # s_nop 0
            "7f008cbf"  # s_waitcnt
            "000081bf"  # s_endpgm
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            bin_file = tmp_path / "ps_test.bin"
            bin_file.write_bytes(raw_code)
            results = scan_directory_or_files([bin_file])
            self.assertEqual(len(results), 1)
            res = results[0]
            self.assertEqual(res["shader_name"], "ps_test.bin")
            self.assertEqual(res["stage"], "pixel")
            self.assertEqual(res["total_instructions"], 3)
            self.assertTrue(res["opcodes_recognized"])

    def test_disassemble_binary_shader_error(self):
        """Verify disassemble_binary_shader raises on invalid or empty input."""
        with self.assertRaises(ValueError):
            disassemble_binary_shader(b"")

    def test_mixed_binary_shader_scanning(self):
        """Verify that failed/invalid binary inputs are retained in results and denominator."""
        valid_code = bytes.fromhex(
            "000080bf"  # s_nop 0
            "000081bf"  # s_endpgm
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            good_file = tmp_path / "ps_good.bin"
            good_file.write_bytes(valid_code)
            bad_file = tmp_path / "ps_bad.bin"
            bad_file.write_bytes(b"")  # Empty binary triggers decoder failure

            results = scan_directory_or_files([tmp_path])
            # Denominator must account for BOTH inputs
            self.assertEqual(len(results), 2)

            by_name = {r["shader_name"]: r for r in results}
            good_res = by_name.get("ps_good.bin")
            bad_res = by_name.get("ps_bad.bin")

            self.assertIsNotNone(good_res)
            self.assertEqual(good_res["total_instructions"], 2)
            self.assertEqual(good_res["supported_instructions"], 2)
            self.assertTrue(good_res["opcodes_recognized"])

            self.assertIsNotNone(bad_res)
            self.assertEqual(bad_res["total_instructions"], 0)
            self.assertFalse(bad_res["is_fully_supported"])
            self.assertFalse(bad_res["translator_accepted"])
            self.assertEqual(bad_res["lower_status"], "decode_failed")
            self.assertIn("empty", bad_res["lower_error"].lower())

            # Verify that dashboard accounts for both inputs and clearly labels instruction vs file metrics
            out_dir = tmp_path / "out"
            json_file, md_file = generate_progress_dashboard(results, out_dir)
            data = json.loads(json_file.read_text(encoding="utf-8"))

            # File-level denominator tested separately: two inputs, one decode failure
            self.assertEqual(data["overall"]["total_shaders"], 2)
            self.assertEqual(data["overall"]["completely_unsupported"], 1)
            self.assertEqual(data["overall"]["fully_accepted"], 0)

            # Instruction metric: every decoded instruction in valid_code is supported,
            # so instruction coverage legitimately reaches 100% while file denominator retains failure.
            self.assertEqual(data["overall"]["total_instructions"], 2)
            self.assertEqual(data["overall"]["supported_instructions"], 2)
            self.assertEqual(data["overall"]["instruction_coverage_percentage"], 100.0)
            self.assertEqual(data["overall"]["coverage_percentage"], 100.0)

            md_text = md_file.read_text(encoding="utf-8")
            self.assertIn("| **Total Shaders Analyzed** | 2 | 100.0% |", md_text)
            self.assertIn("| **Unsupported / Blocked** | 1 | 50.0% |", md_text)
            self.assertIn("| **Total Decoded Instructions** | 2 | 100.0% |", md_text)
            self.assertIn("Instruction Coverage:", md_text)

    def test_dashboard_report_generation(self):
        """Verify generation of PROGRESS.md and progress.json."""
        canonical_shader = [
            "s_mov_b64 s[50:51], exec",
            "s_wqm_b64 exec, exec",
            "s_mov_b32 m0, s24",
            "v_interp_p1_f32_e32 v8, v0, attr0.x",
            "v_interp_p1_f32_e32 v9, v0, attr0.y",
            "v_interp_p2_f32_e32 v9, v1, attr0.y",
            "v_interp_p2_f32_e32 v8, v1, attr0.x",
            "s_and_b64 exec, exec, s[50:51]",
            "s_mov_b32 s21, 0",
            "s_load_dwordx8 s[28:35], s[20:21], null",
            "s_mov_b32 s40, 1",
            "s_mov_b32 s41, 2",
            "s_mov_b32 s42, 3",
            "s_mov_b32 s43, 4",
            "s_waitcnt lgkmcnt(0)",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_endpgm",
        ]
        results = [analyze_shader_instructions("ps_canonical", canonical_shader)]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            json_file, md_file = generate_progress_dashboard(results, tmp_path)
            self.assertTrue(json_file.is_file())
            self.assertTrue(md_file.is_file())
            md_content = md_file.read_text(encoding="utf-8")
            self.assertIn("Instruction Coverage:", md_content)
            self.assertIn("Pipeline Stage Breakdown", md_content)
            json_content = json.loads(json_file.read_text(encoding="utf-8"))
            self.assertIn("overall", json_content)
            self.assertIn("stages", json_content)
            self.assertIn("missing_opcodes_ranked", json_content)

    def test_supported_shader_instructions(self):
        """Verify scanner patterns recognize subrev, mov_b32, waitcnt_expcnt as supported."""
        self.assertTrue(is_instruction_supported("s_waitcnt_expcnt null, 0x0"))
        self.assertTrue(is_instruction_supported("s_waitcnt_expcnt null, 0x3"))
        self.assertTrue(is_instruction_supported("s_waitcnt_expcnt null, 0"))
        self.assertTrue(is_instruction_supported("s_waitcnt_expcnt null, 7"))
        self.assertTrue(is_instruction_supported("s_waitcnt_expcnt s0, 0x0"))
        self.assertTrue(is_instruction_supported("s_waitcnt_expcnt 0x3"))
        self.assertTrue(is_instruction_supported("s_waitcnt_expcnt 0"))
        self.assertTrue(is_instruction_supported("s_waitcnt expcnt(0)"))
        self.assertTrue(is_instruction_supported("s_waitcnt expcnt(7)"))
        self.assertTrue(is_instruction_supported("s_waitcnt lgkmcnt(0)"))
        self.assertTrue(is_instruction_supported("s_waitcnt vmcnt(0)"))
        self.assertTrue(is_instruction_supported("s_waitcnt vmcnt(63)"))
        self.assertTrue(is_instruction_supported("s_waitcnt vmcnt(0) lgkmcnt(0)"))
        self.assertTrue(is_instruction_supported("s_waitcnt lgkmcnt(0) vmcnt(0)"))
        self.assertTrue(is_instruction_supported("v_nop"))
        self.assertFalse(is_instruction_supported("v_nop 0"))
        self.assertFalse(is_instruction_supported("v_nop v0"))
        self.assertTrue(is_instruction_supported("v_nop_e64"))
        self.assertFalse(is_instruction_supported("v_nop_e64 0"))
        # The VOP3 interpolation spelling: only the operand shape of the e32 form.
        for phase, source in ((1, "v0"), (2, "v1")):
            line = f"v_interp_p{phase}_f32_e64 v8, {source}, attr0.x"
            self.assertTrue(is_instruction_supported(line), line)
            self.assertTrue(is_instruction_supported(line.replace("_e64", "_e32")), line)
            for wrong in (line.replace(source, f"-{source}"), line.replace(source, f"|{source}|"), line + " clamp",
                          line + " mul:2", line.replace("attr0", "attr1"), line.replace(source, "s0")):
                self.assertFalse(is_instruction_supported(wrong), wrong)
        self.assertTrue(is_instruction_supported("v_subrev_f32_e32 v1, v2, v3"))
        self.assertTrue(is_instruction_supported("v_subrev_f32_e64 v1, v2, v3 clamp"))
        # v_mac_legacy_f32 has an e32 and an e64 form (the latter with clamp) and no DPP form in LLVM 15. The
        # accumulator is the destination, so only two sources are written. v_fmac_legacy_f32 does not exist on gfx1013.
        for line in ("v_mac_legacy_f32_e32 v1, v2, v3", "v_mac_legacy_f32_e64 v1, v2, v3",
                     "v_mac_legacy_f32_e64 v1, -s2, |v3| clamp"):
            self.assertTrue(is_instruction_supported(line), line)
        self.assertTrue(is_opcode_recognized("v_mac_legacy_f32_e64"))
        for line in ("v_mac_legacy_f32_dpp v1, v2, v3 quad_perm:[0,1,2,3] row_mask:0xf bank_mask:0xf",
                     "v_mac_legacy_f32_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
                     "v_fmac_legacy_f32 v1, v2, v3", "v_fmac_legacy_f32_e32 v1, v2, v3"):
            self.assertFalse(is_instruction_supported(line), line)
        # The 32-bit no-carry family. The INTEGER_SOURCE class is coarser than the translator, so this
        # pins the spellings and the refused modifiers, not the operand ranges.
        for line in (
            "v_add_nc_u32_e32 v1, v2, v3", "v_sub_nc_u32 v1, s2, 64", "v_subrev_nc_u32_e32 v1, v2, v3",
            "v_add_nc_u32_e64 v1, s2, v3", "v_sub_nc_u32_e64 v1, s2, 64", "v_subrev_nc_u32_e64 v1, v2, s3",
            "v_add_nc_i32 v1, s2, v3", "v_sub_nc_i32 v1, 64, v3",
        ):
            self.assertTrue(is_instruction_supported(line), line)
        for line in (
            "v_add_nc_u32_e64 v1, s2, v3 clamp", "v_subrev_nc_u32_e64 v1, s2, v3 clamp",
            "v_add_nc_i32 v1, s2, v3 clamp", "v_sub_nc_i32 v1, s2, v3 clamp",
            "v_add_nc_i32_e64 v1, s2, v3", "v_sub_nc_i32_e64 v1, s2, v3",
            "v_subrev_nc_i32 v1, s2, v3",
            "v_subrev_nc_u32_dpp v1, v2, v3 row_shl:1 row_mask:0xf bank_mask:0xf",
            "v_subrev_nc_u32_sdwa v1, v2, v3 src0_sel:DWORD src1_sel:DWORD",
        ):
            self.assertFalse(is_instruction_supported(line), line)
        # SDWA forms of the 32-bit integer operations: registers, sext(), all fields, no UNUSED_SEXT.
        sdwa = "dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD"
        for line in (
            f"v_and_b32_sdwa v1, v2, v3 {sdwa}", f"v_xnor_b32_sdwa v1, s2, sext(v3) {sdwa}",
            "v_subrev_nc_u32_sdwa v1, sext(s2), v3 dst_sel:BYTE_3 dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:BYTE_0",
            "v_not_b32_sdwa v1, sext(v2) dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:BYTE_2",
            f"v_ashrrev_i32_sdwa v1, v2, v3 {sdwa}", f"v_max_i32_sdwa v1, v2, v3 {sdwa}", f"v_min_u32_sdwa v1, v2, v3 {sdwa}",
        ):
            self.assertTrue(is_instruction_supported(line), line)
        for line in (
            f"v_and_b32_sdwa v1, v2, v3 {sdwa}".replace("UNUSED_PAD", "UNUSED_SEXT"),
            f"v_and_b32_sdwa v1, v2, v3 clamp {sdwa}", f"v_and_b32_sdwa v1, 64, v3 {sdwa}",
            f"v_and_b32_sdwa v1, -v2, v3 {sdwa}", f"v_and_b32_sdwa v1, v2 {sdwa}",
            "v_and_b32_sdwa v1, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD",
            f"v_not_b32_sdwa v1, v2 {sdwa}", f"v_not_b32_sdwa v1, v2, v3 {sdwa}",
            f"v_bfi_b32_sdwa v1, v2, v3 {sdwa}", f"v_add_nc_i32_sdwa v1, v2, v3 {sdwa}",
            f"v_mul_u32_u24_sdwa v1, v2, v3 {sdwa}",
        ):
            self.assertFalse(is_instruction_supported(line), line)
        for opcode in ("v_add_nc_i32", "v_sub_nc_i32", "v_subrev_nc_u32_e64"):
            self.assertTrue(is_opcode_recognized(opcode), opcode)
        self.assertFalse(is_opcode_recognized("v_subrev_nc_i32"))
        # SDWA binary32 arithmetic: whole-dword sources only; a byte or word source, UNUSED_SEXT and
        # operations outside pixel_aot.SDWA_ALU stay unsupported.
        for op in ("exp", "log", "rcp", "rcp_iflag", "rsq", "sqrt", "sin", "cos", "rndne", "frexp_mant", "floor"):
            line = f"v_{op}_f32_sdwa v1, -|s2| dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD"
            self.assertTrue(is_instruction_supported(line), line)
            self.assertTrue(is_instruction_supported(line.replace("dst_sel:DWORD", "clamp dst_sel:BYTE_2")), line)
            self.assertFalse(is_instruction_supported(line.replace("src0_sel:DWORD", "src0_sel:WORD_1")), line)
            self.assertFalse(is_instruction_supported(line.replace("src0_sel:DWORD", "src0_sel:BYTE_0")), line)
            self.assertFalse(is_instruction_supported(line.replace("UNUSED_PAD", "UNUSED_SEXT")), line)
        self.assertTrue(is_instruction_supported(
            "v_add_f32_sdwa v1, s2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD"))
        self.assertFalse(is_instruction_supported(
            "v_add_f32_sdwa v1, s2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0"))
        self.assertFalse(is_instruction_supported(
            "v_mad_f32_sdwa v1, s2, v3, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD"))
        self.assertTrue(is_opcode_recognized("v_mov_b32_sdwa"))
        self.assertTrue(is_opcode_recognized("v_cvt_f32_f16_sdwa"))
        self.assertTrue(is_opcode_recognized("v_pk_add_f16"))
        self.assertTrue(is_opcode_recognized("v_dot4_i32_i8"))
        self.assertTrue(is_instruction_supported("v_cvt_f32_u32_e32 v1, v2"))
        self.assertTrue(is_instruction_supported("v_cvt_f32_u32_e64 v1, v2 clamp"))
        self.assertTrue(is_instruction_supported("v_cvt_f32_i32_e32 v1, v2"))
        self.assertTrue(is_instruction_supported("v_cvt_f32_i32_e64 v1, v2 clamp"))
        self.assertTrue(is_instruction_supported("s_nand_b32 s1, s2, s3"))
        self.assertTrue(is_instruction_supported("s_nor_b32 s1, s2, s3"))
        self.assertTrue(is_instruction_supported("s_xnor_b32 s1, s2, s3"))
        self.assertTrue(is_instruction_supported("s_lshl1_add_u32 s1, s2, s3"))
        self.assertTrue(is_instruction_supported("s_lshl2_add_u32 s1, s2, s3"))
        self.assertTrue(is_instruction_supported("s_lshl3_add_u32 s1, s2, s3"))
        self.assertTrue(is_instruction_supported("s_lshl4_add_u32 s1, s2, s3"))
        self.assertTrue(is_instruction_supported("s_bfm_b32 s1, s2, s3"))
        self.assertTrue(is_instruction_supported("s_bfm_b32 s1, 4, 8"))
        for line in (
            "s_bfe_u64 s[0:1], s[2:3], s4",
            "s_bfe_u64 s[0:1], s[2:3], 0x400000",
            "s_bfe_u64 s[0:1], 0x80000000, 32",
            "s_bfe_i64 s[0:1], s[2:3], s4",
            "s_bfe_i64 s[0:1], s[2:3], 0x20001f",
            "s_bfm_b64 s[0:1], s2, s3",
            "s_bfm_b64 s[0:1], 32, 16",
        ):
            self.assertTrue(is_instruction_supported(line), line)
        for line in (
            "s_bfe_u64 s0, s[2:3], s4",
            "s_bfe_i64 s0, s[2:3], s4",
            "s_bfe_u64 s[0:1], s2, s4",
            "s_bfe_i64 s[0:1], s2, s4",
            "s_bfe_u64 s[0:1], s[2:3]",
            "s_bfe_i64 s[0:1], s[2:3]",
            "s_bfe_u64 s[0:1], 0x8123456789abcdef, s4",
            "s_bfe_i64 s[0:1], 0x8123456789abcdef, s4",
            "s_bfm_b64 s0, s2, s3",
            "s_bfm_b64 s[0:1], s2",
            "s_bfm_b64 s[0:1], s[2:3], s4",
        ):
            self.assertFalse(is_instruction_supported(line), line)
        for opcode in ("s_bfe_u64", "s_bfe_i64", "s_bfm_b64"):
            self.assertTrue(is_opcode_recognized(opcode), opcode)
        # 64-bit integer comparisons: register pairs or an inline integer; VOPC writes VCC and takes a VGPR pair
        # as src1. Literals, floats, modifiers, SGPR src1 or an SGPR destination on e32, and exec writers are refused.
        for op in ("f", "lt", "eq", "le", "gt", "ne", "ge", "t"):
            for kind in ("i64", "u64"):
                for line in (f"v_cmp_{op}_{kind}_e32 vcc_lo, v[18:19], v[20:21]",
                             f"v_cmp_{op}_{kind} vcc_lo, s[44:45], v[20:21]",
                             f"v_cmp_{op}_{kind}_e32 vcc_lo, -1, v[20:21]",
                             f"v_cmp_{op}_{kind}_e64 s52, v[18:19], v[19:20]",
                             f"v_cmp_{op}_{kind}_e64 vcc_lo, s[44:45], s[46:47]",
                             f"v_cmp_{op}_{kind}_e64 s[52:53], 64, v[20:21]",
                             # Without a suffix LLVM 15 picks e64 when the operands need it.
                             f"v_cmp_{op}_{kind} s[52:53], v[18:19], v[20:21]",
                             f"v_cmp_{op}_{kind} vcc_lo, v[18:19], s[44:45]"):
                    self.assertTrue(is_instruction_supported(line), line)
        for line in (
            "v_cmp_lt_i64_e32 vcc_lo, v[18:19], s[44:45]",
            "v_cmp_lt_i64_e32 vcc_lo, v[18:19], 64",
            "v_cmp_lt_i64_e32 s52, v[18:19], v[20:21]",
            "v_cmp_lt_u64_e64 s52, 0x12345678, v[20:21]",
            "v_cmp_lt_u64_e64 s52, 1.0, v[20:21]",
            "v_cmp_lt_u64_e64 s52, v[18:19], v[20:21] clamp",
            "v_cmp_lt_u64_e64 s52, -v[18:19], v[20:21]",
            "v_cmp_lt_u64_e64 s52, v18, v[20:21]",
            "v_cmp_lt_u64_e64 exec, v[18:19], v[20:21]",
            "v_cmpx_lt_i64_e32 v[18:19], v[20:21]",
            "v_cmpx_lt_u64_e64 v[18:19], v[20:21]",
            "v_cmp_lt_f64_e64 s52, v[18:19], v[20:21]",
            "v_cmp_ngt_u64_e64 s52, v[18:19], v[20:21]",
            "v_cmp_lt_u64_e64 s52, 65, v[20:21]",
            "v_cmp_lt_u64_e64 s52, v[18:19], -17",
            "v_cmp_lt_u64_e64 s52, v[255:256], v[20:21]",
            "v_cmp_lt_u64_e64 s52, s[45:46], v[20:21]",
            "v_cmp_lt_i64_e32 vcc_lo, s[45:46], v[20:21]",
            "v_cmp_lt_u64_e64 s52, v[18:20], v[20:21]",
        ):
            self.assertFalse(is_instruction_supported(line), line)
        for line in (
            "v_readfirstlane_b32 s0, v1",
            "v_readfirstlane_b32 s44, v12",
            "v_readfirstlane_b32 s10, 0x1234",
            "v_readfirstlane_b32 s10, 42",
        ):
            self.assertTrue(is_instruction_supported(line), line)
        for line in (
            "v_readfirstlane_b32 v0, v1",
            "v_readfirstlane_b32 s0, v1, v2",
            "v_readfirstlane_b32 s0",
            "v_readfirstlane_b32",
        ):
            self.assertFalse(is_instruction_supported(line), line)
        self.assertTrue(is_opcode_recognized("v_readfirstlane_b32"))
        for line in (
            "v_alignbit_b32 v1, v2, v3, v4",
            "v_alignbit_b32_e64 v1, v2, v3, 4",
            "v_alignbit_b32 v1, s2, v3, 31",
            "v_alignbyte_b32 v1, v2, v3, v4",
            "v_alignbyte_b32_e64 v1, s2, v3, 2",
            "v_alignbyte_b32 v1, v2, s3, 0",
        ):
            self.assertTrue(is_instruction_supported(line), line)
        for line in (
            "v_alignbit_b32_e32 v1, v2, v3, 4",
            "v_alignbyte_b32_e32 v1, v2, v3, 2",
            "v_alignbit_b32 v1, v2, v3",
            "v_alignbyte_b32 v1, v2, v3",
            "v_alignbit_b32 v1, v2, v3, 4, 5",
            "v_alignbyte_b32 v1, v2, v3, 1, 2",
        ):
            self.assertFalse(is_instruction_supported(line), line)
        for opcode in ("v_alignbit_b32", "v_alignbyte_b32"):
            self.assertTrue(is_opcode_recognized(opcode), opcode)
            self.assertTrue(is_opcode_recognized(opcode), opcode)
        for line in (
            "v_add_co_u32 v1, vcc_lo, s2, v3",
            "v_sub_co_u32 v1, s[60:61], 64, v3",
            "v_subrev_co_u32 v1, vcc, v2, 0x12345678",
        ):
            self.assertTrue(is_instruction_supported(line), line)
        for line in (
            "v_add_co_u32_e64 v1, vcc, v2, v3",
            "v_sub_co_u32_e32 v1, vcc, v2, v3",
            "v_subrev_co_u32 v1, exec, v2, v3",
            "v_add_co_u32 v1, v4, v2, v3",
            "v_add_co_u32 v1, vcc, v2, v3 clamp",
            "v_sub_co_u32 v1, vcc, -v2, v3",
            "v_add_co_u32 v1, vcc, v2",
            "v_add_co_ci_u32_e64 v1, vcc, v2, v3, vcc, vcc",
        ):
            self.assertFalse(is_instruction_supported(line), line)
        for opcode in ("v_add_co_u32", "v_sub_co_u32", "v_subrev_co_u32"):
            self.assertTrue(is_opcode_recognized(opcode), opcode)
        for line in (
            "v_add_co_ci_u32_e32 v1, vcc_lo, s2, v3, vcc_lo",
            "v_sub_co_ci_u32_e32 v1, vcc, 0x12345678, v3, vcc",
            "v_subrev_co_ci_u32_e32 v1, vcc_lo, -16, v3, vcc_lo",
            "v_add_co_ci_u32_e64 v1, vcc, v2, v3, vcc",
            "v_sub_co_ci_u32_e64 v1, s5, s2, 7, s6",
            "v_subrev_co_ci_u32_e64 v1, s[4:5], v2, 2.0, s[6:7]",
        ):
            self.assertTrue(is_instruction_supported(line), line)
        for line in (
            "v_add_co_ci_u32 v1, vcc_lo, v2, v3, vcc_lo",
            "v_add_co_ci_u32_e32 v1, vcc_lo, v2, s3, vcc_lo",
            "v_sub_co_ci_u32_e32 v1, s5, v2, v3, vcc_lo",
            "v_subrev_co_ci_u32_e32 v1, vcc_lo, v2, v3, s6",
            "v_add_co_ci_u32_e64 v1, s5, v2, v3, s6 clamp",
            "v_sub_co_ci_u32_e64 v1, s5, -v2, v3, s6",
            "v_subrev_co_ci_u32_e64 v1, exec_lo, v2, v3, s6",
            "v_add_co_ci_u32_e64 v1, s5, v2, v3, v6",
            "v_add_co_ci_u32_e64 v1, s5, v2, v3",
            "v_add_co_ci_u32_dpp v1, vcc_lo, v2, v3, vcc_lo quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
            "v_add_co_ci_u32_sdwa v1, vcc_lo, v2, v3, vcc_lo dst_sel:DWORD dst_unused:UNUSED_PAD "
            "src0_sel:DWORD src1_sel:DWORD",
        ):
            self.assertFalse(is_instruction_supported(line), line)
        for opcode in ("v_add_co_ci_u32", "v_sub_co_ci_u32", "v_subrev_co_ci_u32"):
            self.assertTrue(is_opcode_recognized(opcode), opcode)
        for line in (
            "v_cvt_pk_u16_u32 v1, v2, v3",
            "v_cvt_pk_u16_u32 v1, 0x12345, v3",
            "v_cvt_pk_i16_i32 v1, s2, v3",
            "v_cvt_pk_i16_i32_e64 v1, -16, v3",
        ):
            self.assertTrue(is_instruction_supported(line), line)
        for line in (
            "v_cvt_pk_u16_u32_e32 v1, v2, v3",
            "v_cvt_pk_i16_i32_e32 v1, v2, v3",
            "v_cvt_pk_u16_u32 v1, v2",
            "v_cvt_pk_i16_i32 v1, v2, v3, v4",
            "v_cvt_pk_i16_i32 v1, v2, v3 clamp",
            "v_cvt_pknorm_i16_f32 v1, v2, v3",
            "v_cvt_pknorm_u16_f32 v1, v2, v3",
            "v_cvt_pk_u8_f32 v1, v2, v3, v4",
        ):
            self.assertFalse(is_instruction_supported(line), line)
        for opcode in ("v_cvt_pk_u16_u32", "v_cvt_pk_i16_i32"):
            self.assertTrue(is_opcode_recognized(opcode), opcode)
        for line in (
            "v_sad_hi_u8 v1, v2, v3, v4",
            "v_sad_u32 v1, s2, v3, 64",
            "v_sad_u32_e64 v1, v2, v3, v4",
            "v_msad_u8 v1, v2, 0x7f00ff01, s4",
        ):
            self.assertTrue(is_instruction_supported(line), line)
        for line in (
            "v_sad_hi_u8_e32 v1, v2, v3, v4",
            "v_sad_u32 v1, v2, v3",
            "v_msad_u8 v1, v2, v3, v4, v5",
            "v_msad_u8 v1, v2, v3, v4 clamp",
            "v_sad_u8 v1, v2, v3, v4",
            "v_sad_u16 v1, v2, v3, v4",
            "v_mqsad_pk_u16_u8 v[1:2], v[3:4], v5, v[6:7]",
        ):
            self.assertFalse(is_instruction_supported(line), line)
        for opcode in ("v_sad_hi_u8", "v_sad_u32", "v_msad_u8"):
            self.assertTrue(is_opcode_recognized(opcode), opcode)
        for line in (
            "v_add3_u32 v1, v2, v3, v4",
            "v_add3_u32_e64 v1, s2, v3, 4",
            "v_or3_b32 v1, v2, v3, v4",
            "v_xor3_b32 v1, v2, v3, v4",
            "v_and_or_b32 v1, v2, v3, v4",
            "v_xad_u32 v1, v2, v3, v4",
            "v_lshl_or_b32 v1, v2, v3, v4",
            "v_lshl_add_u32 v1, v2, v3, v4",
            "v_add_lshl_u32 v1, v2, v3, v4",
            "v_add_lshl_u32_e64 v1, s2, v3, 31",
        ):
            self.assertTrue(is_instruction_supported(line), line)
        for line in (
            "v_add3_u32_e32 v1, v2, v3, 4",
            "v_or3_b32_e32 v1, v2, v3, 4",
            "v_xor3_b32_e32 v1, v2, v3, 4",
            "v_and_or_b32_e32 v1, v2, v3, 4",
            "v_xad_u32_e32 v1, v2, v3, 4",
            "v_lshl_or_b32_e32 v1, v2, v3, 4",
            "v_lshl_add_u32_e32 v1, v2, v3, 4",
            "v_add_lshl_u32_e32 v1, v2, v3, 4",
            "v_add3_u32 v1, v2, v3",
            "v_add3_u32 v1, v2, v3, 4, 5",
        ):
            self.assertFalse(is_instruction_supported(line), line)
        for opcode in (
            "v_add3_u32", "v_or3_b32", "v_xor3_b32", "v_and_or_b32",
            "v_xad_u32", "v_lshl_or_b32", "v_lshl_add_u32", "v_add_lshl_u32",
        ):
            self.assertTrue(is_opcode_recognized(opcode), opcode)
        for unsupported in ("v_cvt_f32_i32 v1, v2",
                            "v_cvt_f32_i32_e32 v1, v2 clamp",
                            "v_cvt_f32_u32_e64 v1, -v2",
                            "v_cvt_f32_u32_e64 v1, v2 omod:2",
                            "v_cvt_f32_i32_e32 v1, 1.5",
                            "s_nand_b32 s1, s2",
                            "s_nor_b32 s1, s2",
                            "s_xnor_b32 s1, s2",
                            "s_nand_b32 v1, s2, s3",
                            "s_bfm_b32 s1, s2",
                            "s_bfm_b32 v1, s2, s3",
                            "s_bfm_b32 s1, v2, s3",
                            "s_bfm_b32 s[2:3], s2, s3"):
            self.assertFalse(is_instruction_supported(unsupported), unsupported)

        for line in ("v_swap_b32 v1, v2", "v_swap_b32_e32 v1, v2", "v_swap_b32 v1, v1", "v_swap_b32 v255, v254"):
            self.assertTrue(is_instruction_supported(line), line)
        for line in ("v_swap_b32 v1, s2",
                     "v_swap_b32 s1, v2",
                     "v_swap_b32_e64 v1, v2",
                     "v_swap_b32_dpp v1, v2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf",
                     "v_swap_b32_sdwa v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD",
                     "v_swap_b32 v1, v2 clamp",
                     "v_swap_b32 v1, -v2",
                     "v_swap_b32 v1",
                     "v_swaprel_b32 v1, v2"):
            self.assertFalse(is_instruction_supported(line), line)
        self.assertTrue(is_opcode_recognized("v_swap_b32"))

        # Verify unsupported operands or spellings are rejected by scanner
        self.assertFalse(is_opcode_recognized("v_imaginary_opcode_sdwa"))
        self.assertFalse(is_instruction_supported("s_waitcnt_expcnt null, 0x8"))
        self.assertFalse(is_instruction_supported("s_waitcnt_expcnt null, 8"))
        self.assertFalse(is_instruction_supported("s_waitcnt_expcnt 0x8"))
        self.assertFalse(is_instruction_supported("s_waitcnt_expcnt 8"))
        self.assertFalse(is_instruction_supported("s_waitcnt_expcnt invalid"))
        self.assertFalse(is_instruction_supported("s_waitcnt_expcnt null, invalid"))
        self.assertFalse(is_instruction_supported("s_waitcnt expcnt(8)"))
        self.assertFalse(is_instruction_supported("s_waitcnt vmcnt(64)"))
        self.assertFalse(is_instruction_supported("s_waitcnt lgkmcnt(64)"))
        self.assertTrue(is_instruction_supported("s_waitcnt vmcnt(7) lgkmcnt(0)"))
        self.assertTrue(is_instruction_supported("s_waitcnt vmcnt(7) lgkmcnt(63)"))
        self.assertTrue(is_instruction_supported("s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)"))
        self.assertTrue(is_instruction_supported("s_waitcnt 0"))
        self.assertTrue(is_instruction_supported("s_waitcnt 0x0"))
        self.assertFalse(is_instruction_supported("s_waitcnt foo"))
        self.assertFalse(is_instruction_supported("s_waitcnt vmcnt(0) lgkmcnt(0) vmcnt(1)"))
        self.assertFalse(is_instruction_supported("s_waitcnt lgkmcnt(0)lgkmcnt(1)"))
        self.assertFalse(is_instruction_supported("s_waitcnt expcnt(0) expcnt(1)"))

    def test_sopk_waitcnt_forms(self):
        """null with a count of 0..63 is supported for the three SOPK wait forms; an SGPR or larger count is not."""
        for op in ("s_waitcnt_vmcnt", "s_waitcnt_lgkmcnt", "s_waitcnt_vscnt"):
            for count in ("0x0", "0x3", "0x3f", "0", "1", "9", "10", "63"):
                self.assertTrue(is_instruction_supported(f"{op} null, {count}"), (op, count))
            for unsupported in ("s5, 0x1", "s0, 0x0", "vcc_lo, 0x0", "0x1", "null", "null, 0x40", "null, 64",
                                "null, 0xffff", "null, 0x400", "null, -1", "null, invalid", "null,"):
                self.assertFalse(is_instruction_supported(f"{op} {unsupported}"), (op, unsupported))
        self.assertFalse(is_instruction_supported("s_waitcnt_depctr null, 0x0"))
        decoded = disassemble_binary_shader(bytes.fromhex("00007dbc" "00007dbd" "0000fdbb" "000081bf"))
        self.assertEqual(decoded, ["s_waitcnt_vmcnt null, 0x0", "s_waitcnt_lgkmcnt null, 0x0",
                                   "s_waitcnt_vscnt null, 0x0", "s_endpgm"])
        result = analyze_shader_instructions("sopk_wait", decoded)
        self.assertEqual((result["supported_instructions"], result["unsupported_count"]), (4, 0))
        result = analyze_shader_instructions("sopk_wait_sgpr", ["s_waitcnt_vmcnt s5, 0x1", "s_endpgm"])
        self.assertEqual(result["unsupported_count"], 1)

    def test_sopk_s_version(self):
        """s_version with a 16-bit immediate is supported and known; larger, negative and register operands are not."""
        for operand in ("0x0", "0x1004", "0xffff", "0x00ff", "0", "1", "4100", "9999", "10000", "65535", "59999",
                        "65000", "65529"):
            self.assertTrue(is_instruction_supported(f"s_version {operand}"), operand)
        for operand in ("0x10000", "0xfffff", "65536", "65540", "99999", "100000", "4294967295", "-1", "-0x1",
                        "s0", "null", "v0", "0, 0", "0x1004 0x1", "007", "0x", "1.0", "abc", ""):
            self.assertFalse(is_instruction_supported(f"s_version {operand}".rstrip()), operand)
        self.assertIn("s_version", KNOWN_RDNA2_BASE_OPCODES)
        decoded = disassemble_binary_shader(bytes.fromhex("041080b0" "000080b0" "ffff80b0" "000081bf"))
        self.assertEqual(decoded, ["s_version 0x1004", "s_version 0x0", "s_version 0xffff", "s_endpgm"])
        result = analyze_shader_instructions("s_version", decoded)
        self.assertEqual((result["supported_instructions"], result["unsupported_count"]), (4, 0))
        result = analyze_shader_instructions("s_version_wide", ["s_version 0x10000", "s_endpgm"])
        self.assertEqual(result["unsupported_count"], 1)

    def test_gfx1013_waitcnt_expcnt_decode_and_lower(self):
        """Assemble/decode GFX1013 s_waitcnt_expcnt binary bytes and lower with pixel_aot."""
        from pixel_aot import lower
        code_bytes = bytes.fromhex(
            "0000fdbc"  # s_waitcnt_expcnt null, 0x0
            "000080bf"  # s_nop 0
            "0300fdbc"  # s_waitcnt_expcnt null, 0x3
            "000081bf"  # s_endpgm
        )
        decoded = disassemble_binary_shader(code_bytes)
        self.assertEqual(decoded[0], "s_waitcnt_expcnt null, 0x0")
        self.assertEqual(decoded[1], "s_nop 0")
        self.assertEqual(decoded[2], "s_waitcnt_expcnt null, 0x3")
        self.assertEqual(decoded[3], "s_endpgm")

        res = analyze_shader_instructions("expcnt_test", decoded)
        self.assertTrue(res["opcodes_recognized"])
        self.assertEqual(res["supported_instructions"], 4)
        self.assertEqual(res["unsupported_count"], 0)
        self.assertEqual(res["support_percentage"], 100.0)

        full_shader = ["s_waitcnt_expcnt null, 0x0"] + [
            "s_mov_b64 s[50:51], exec",
            "s_wqm_b64 exec, exec",
            "s_mov_b32 m0, s24",
            "v_interp_p1_f32_e32 v8, v0, attr0.x",
            "v_interp_p1_f32_e32 v9, v0, attr0.y",
            "v_interp_p2_f32_e32 v9, v1, attr0.y",
            "v_interp_p2_f32_e32 v8, v1, attr0.x",
            "s_and_b64 exec, exec, s[50:51]",
            "s_mov_b32 s21, 0",
            "s_load_dwordx8 s[28:35], s[20:21], null",
            "s_mov_b32 s40, 1",
            "s_mov_b32 s41, 2",
            "s_mov_b32 s42, 3",
            "s_mov_b32 s43, 4",
            "s_waitcnt lgkmcnt(0)",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_waitcnt_expcnt null, 0x3",
            "s_endpgm",
        ]
        res_full = analyze_shader_instructions("canonical_ps", full_shader)
        self.assertTrue(res_full["is_fully_supported"])
        self.assertTrue(res_full["translator_accepted"])
        hlsl = lower(full_shader, 20, 24, [1, 2, 3, 4])
        self.assertIn("return float4(", hlsl)

    def test_gfx1013_v_nop_decode_and_lower(self):
        """Assemble/decode GFX1013 v_nop binary bytes and lower with pixel_aot."""
        from pixel_aot import lower
        code_bytes = bytes.fromhex(
            "0000007e"  # v_nop
            "000080bf"  # s_nop 0
            "000081bf"  # s_endpgm
        )
        decoded = disassemble_binary_shader(code_bytes)
        self.assertEqual(decoded[0], "v_nop")
        self.assertEqual(decoded[1], "s_nop 0")
        self.assertEqual(decoded[2], "s_endpgm")

        res = analyze_shader_instructions("v_nop_test", decoded)
        self.assertTrue(res["opcodes_recognized"])
        self.assertEqual(res["supported_instructions"], 3)
        self.assertEqual(res["unsupported_count"], 0)
        self.assertEqual(res["support_percentage"], 100.0)

        full_shader = ["v_nop"] + [
            "s_mov_b64 s[50:51], exec",
            "s_wqm_b64 exec, exec",
            "s_mov_b32 m0, s24",
            "v_interp_p1_f32_e32 v8, v0, attr0.x",
            "v_interp_p1_f32_e32 v9, v0, attr0.y",
            "v_interp_p2_f32_e32 v9, v1, attr0.y",
            "v_interp_p2_f32_e32 v8, v1, attr0.x",
            "s_and_b64 exec, exec, s[50:51]",
            "v_nop",
            "s_mov_b32 s21, 0",
            "s_load_dwordx8 s[28:35], s[20:21], null",
            "s_mov_b32 s40, 1",
            "s_mov_b32 s41, 2",
            "s_mov_b32 s42, 3",
            "s_mov_b32 s43, 4",
            "s_waitcnt lgkmcnt(0)",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_endpgm",
        ]
        res_full = analyze_shader_instructions("canonical_ps_with_v_nop", full_shader)
        self.assertTrue(res_full["is_fully_supported"])
        self.assertTrue(res_full["translator_accepted"])
        hlsl = lower(full_shader, 20, 24, [1, 2, 3, 4])
        self.assertIn("return float4(", hlsl)

        # Negative control: v_nop after color export must be rejected
        with_v_nop_after_exp = full_shader[:-1] + ["v_nop", "s_endpgm"]
        with self.assertRaises(ValueError):
            lower(with_v_nop_after_exp, 20, 24, [1, 2, 3, 4])

    def test_valu_3operand_float_math_scanner(self):
        """Test scanning v_div_fmas_f32 in scan_unsupported_shaders and rejecting v_fma_f16 / v_div_scale_f32."""
        valid_insns = [
            "v_div_fmas_f32 v0, v1, v2, v3",
            "v_div_fmas_f32_e64 v0, v1, v2, v3",
            "v_div_fmas_f32 v0, v1, v2, v3 clamp",
        ]
        for insn in valid_insns:
            self.assertTrue(is_instruction_supported(insn), f"Expected supported: {insn}")

        invalid_insns = [
            "v_fma_f16 v0, v1, v2, v3",
            "v_fma_f16_e64 v0, v1, v2, v3",
            "v_fma_f16 v0, v1, v2, v3 clamp",
            "v_fma_f16 v0, v1, v2, v3 op_sel:[1,0,1]",
            "v_fma_f16 v0, v1, v2, v3 op_sel:[1,0,1] clamp",
            "v_fma_f16 v0, v1, v2, v3 clamp op_sel:[1,0,1]",
            "v_fma_f16 v0, -v1, |v2|, 0.5 op_sel:[0,1,0]",
            "v_div_scale_f32 v0, vcc_lo, v1, v2, v3",
            "v_div_scale_f32_e64 v0, vcc, v1, v2, v3",
            "v_div_scale_f32 v0, s52, v1, v2, v3",
            "v_fma_f16 v0, v1, v2",
            "v_fma_f16 v0, v1, v2, v3 omod:2",
            "v_div_scale_f32 v0, v1, v2, v3",
            "v_div_fmas_f32 v0, v1, v2",
            "v_div_fmas_f32 v0, v1, v2, v3 omod:2",
        ]
        for insn in invalid_insns:
            self.assertFalse(is_instruction_supported(insn), f"Expected unsupported: {insn}")

    def test_valu_3operand_float_math_lower(self):
        """Test lowering v_div_fmas_f32 and rejecting unverified v_fma_f16 and v_div_scale_f32."""
        div_shader = [
            "s_mov_b64 s[50:51], exec",
            "s_wqm_b64 exec, exec",
            "s_mov_b32 m0, s24",
            "v_interp_p1_f32_e32 v8, v0, attr0.x",
            "v_interp_p1_f32_e32 v9, v0, attr0.y",
            "v_interp_p2_f32_e32 v9, v1, attr0.y",
            "v_interp_p2_f32_e32 v8, v1, attr0.x",
            "s_and_b64 exec, exec, s[50:51]",
            "s_mov_b32 s21, 0",
            "s_load_dwordx8 s[28:35], s[20:21], null",
            "s_mov_b32 s40, 1",
            "s_mov_b32 s41, 2",
            "s_mov_b32 s42, 3",
            "s_mov_b32 s43, 4",
            "s_waitcnt lgkmcnt(0)",
            "image_sample v[12:15], v[8:9], s[28:35], s[40:43] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_waitcnt vmcnt(0)",
            "v_cmp_lt_f32 vcc_lo, 0, 1.0",
            "v_div_fmas_f32 v14, 1.0, 2.0, 3.0",
            "v_cvt_pkrtz_f16_f32_e32 v16, v12, v13",
            "v_cvt_pkrtz_f16_f32_e32 v17, v14, v15",
            "exp mrt0 v16, v16, v17, v17 done compr vm",
            "s_endpgm",
        ]
        hlsl = lower(div_shader, 20, 24, [1, 2, 3, 4])
        self.assertIn("div_fmas32(", hlsl)
        self.assertIn("4294967296.0", hlsl)

        # Negative control: v_fma_f16 must be rejected
        with self.assertRaises(ValueError):
            lower(["v_fma_f16 v22, v19, v20, v21", *div_shader], 20, 24, [1, 2, 3, 4])

        # Negative control: v_div_scale_f32 must be rejected
        with self.assertRaises(ValueError):
            lower(["v_div_scale_f32 v12, vcc_lo, v19, v19, v20", *div_shader], 20, 24, [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
