#!/usr/bin/env python3
"""
AMD RDNA 2 (GFX1013 / Scarlett) Shader Disassembler & Inspector.

Decodes raw binary machine code (.bin) from Xbox Scarlett shaders,
prints human-readable assembly with byte offsets, raw hex words, register
tracking, and explanations mapped to the AMD RDNA 2 ISA manual.
"""

import argparse
from collections import Counter
import os
from pathlib import Path
import struct
import sys

# Register decode helpers

# Source operand codes that name hardware values rather than SGPRs (LLVM spellings).
SPECIAL_SRC_OPERANDS = {
    235: "src_shared_base",
    236: "src_shared_limit",
    237: "src_private_base",
    238: "src_private_limit",
    239: "src_pops_exiting_wave_id",
    251: "src_vccz",
    252: "src_execz",
    253: "src_scc",
    254: "src_lds_direct",
}


def decode_s_reg(reg_id: int) -> str:
    """Decode scalar register or scalar constant code."""
    if reg_id <= 105:
        return f"s{reg_id}"
    if reg_id == 106:
        return "vcc_lo"
    if reg_id == 107:
        return "vcc_hi"
    if 108 <= reg_id <= 123:
        return f"ttmp{reg_id - 108}"
    if reg_id == 124:
        return "m0"
    if reg_id == 125:
        return "null"
    if reg_id == 126:
        return "exec_lo"
    if reg_id == 127:
        return "exec_hi"
    if 128 <= reg_id <= 192:
        return str(reg_id - 128)
    if 193 <= reg_id <= 208:
        return str(-(reg_id - 192))
    if reg_id == 240:
        return "0.5"
    if reg_id == 241:
        return "-0.5"
    if reg_id == 242:
        return "1.0"
    if reg_id == 243:
        return "-1.0"
    if reg_id == 244:
        return "2.0"
    if reg_id == 245:
        return "-2.0"
    if reg_id == 246:
        return "4.0"
    if reg_id == 247:
        return "-4.0"
    if reg_id == 248:
        return "0.15915494"
    if reg_id in SPECIAL_SRC_OPERANDS:
        return SPECIAL_SRC_OPERANDS[reg_id]
    if reg_id == 255:
        return "literal"
    return f"s{reg_id}"


def decode_s_reg64(reg_id: int) -> str:
    """Decode a 64-bit scalar operand as an SGPR pair or a named 64-bit special."""
    if reg_id <= 105 and reg_id % 2 == 0:
        return f"s[{reg_id}:{reg_id + 1}]"
    if reg_id == 106:
        return "vcc"
    if 108 <= reg_id <= 122 and (reg_id - 108) % 2 == 0:
        n = reg_id - 108
        return f"ttmp[{n}:{n + 1}]"
    if reg_id == 125:
        return "null"
    if reg_id == 126:
        return "exec"
    if reg_id == 248:
        return "0.15915494309189532"
    return decode_s_reg(reg_id)


def decode_v_reg(reg_id: int) -> str:
    """Decode a 9-bit VOP SRC0 operand: 0..255 scalar/constant, 256..511 VGPR."""
    if 256 <= reg_id <= 511:
        return f"v{reg_id - 256}"
    return decode_s_reg(reg_id)


def decode_v_reg64(reg_id: int) -> str:
    """Decode a 64-bit VOP operand as a VGPR pair, SGPR pair, or constant."""
    if 256 <= reg_id <= 511:
        vreg = reg_id - 256
        return f"v[{vreg}:{vreg + 1}]" if vreg < 255 else "<invalid VGPR pair>"
    return decode_s_reg64(reg_id)


def _format_operand_modifiers(val: str, is_abs: int, is_neg: int) -> str:
    """Wrap a disassembled operand with inline abs (`|...|`) and neg (`-`) modifiers."""
    if is_abs and is_neg:
        return f"-|{val}|"
    if is_abs:
        return f"|{val}|"
    if is_neg:
        return f"-{val}"
    return val


# Instruction Opcode Tables
SOPP_OPS = {
    0x00: ("s_nop", "No operation (wait cycles)"),
    0x01: ("s_endpgm", "Terminate shader execution"),
    0x02: ("s_branch", "Unconditional branch"),
    0x04: ("s_cbranch_scc0", "Branch if SCC == 0"),
    0x05: ("s_cbranch_scc1", "Branch if SCC == 1"),
    0x06: ("s_cbranch_vccz", "Branch if VCC is zero"),
    0x07: ("s_cbranch_vccnz", "Branch if VCC is non-zero"),
    0x08: ("s_cbranch_execz", "Branch if EXEC is zero"),
    0x09: ("s_cbranch_execnz", "Branch if EXEC is non-zero"),
    0x0a: ("s_barrier", "Synchronize wavefronts in workgroup"),
    0x0c: ("s_waitcnt", "Wait for memory operations"),
    0x20: ("s_inst_prefetch", "Instruction cache prefetch"),
    0x21: ("s_clause", "Begin hard clause"),
}

SOP1_OPS = {
    0x03: ("s_mov_b32", "Scalar move 32-bit"),
    0x04: ("s_mov_b64", "Scalar move 64-bit"),
    0x05: ("s_cmov_b32", "Scalar conditional move 32-bit"),
    0x06: ("s_cmov_b64", "Scalar conditional move 64-bit"),
    0x07: ("s_not_b32", "Scalar bitwise NOT 32-bit"),
    0x08: ("s_not_b64", "Scalar bitwise NOT 64-bit"),
    0x09: ("s_wqm_b32", "Scalar whole-quad-mode 32-bit"),
    0x0a: ("s_wqm_b64", "Scalar whole-quad-mode 64-bit"),
    0x20: ("s_setpc_b64", "Set program counter 64-bit"),
    0x21: ("s_swappc_b64", "Swap program counter 64-bit"),
    0x3c: ("s_and_saveexec_b32", "Save execution mask and bitwise AND with scalar source"),
}

SOP2_OPS = {
    0x00: ("s_add_u32", "Scalar add unsigned 32-bit"),
    0x01: ("s_sub_u32", "Scalar sub unsigned 32-bit"),
    0x02: ("s_add_i32", "Scalar add signed 32-bit"),
    0x03: ("s_sub_i32", "Scalar sub signed 32-bit"),
    0x06: ("s_min_i32", "Scalar minimum signed 32-bit"),
    0x07: ("s_min_u32", "Scalar minimum unsigned 32-bit"),
    0x08: ("s_max_i32", "Scalar maximum signed 32-bit"),
    0x09: ("s_max_u32", "Scalar maximum unsigned 32-bit"),
    0x0a: ("s_cselect_b32", "Scalar conditional select 32-bit"),
    0x0b: ("s_cselect_b64", "Scalar conditional select 64-bit"),
    0x0e: ("s_and_b32", "Scalar bitwise AND 32-bit"),
    0x0f: ("s_and_b64", "Scalar bitwise AND 64-bit"),
    0x10: ("s_or_b32", "Scalar bitwise OR 32-bit"),
    0x11: ("s_or_b64", "Scalar bitwise OR 64-bit"),
    0x12: ("s_xor_b32", "Scalar bitwise XOR 32-bit"),
    0x13: ("s_xor_b64", "Scalar bitwise XOR 64-bit"),
    0x14: ("s_andn2_b32", "Scalar AND with NOT 32-bit"),
    0x15: ("s_andn2_b64", "Scalar AND with NOT 64-bit"),
    0x1e: ("s_lshl_b32", "Scalar logical shift left 32-bit"),
    0x1f: ("s_lshl_b64", "Scalar logical shift left 64-bit"),
    0x20: ("s_lshr_b32", "Scalar logical shift right 32-bit"),
    0x21: ("s_lshr_b64", "Scalar logical shift right 64-bit"),
    0x22: ("s_ashr_i32", "Scalar arithmetic shift right 32-bit"),
    0x26: ("s_mul_i32", "Scalar multiply signed 32-bit"),
    0x2c: ("s_absdiff_i32", "Scalar absolute difference signed 32-bit"),
    0x2e: ("s_lshl1_add_u32", "Scalar shift left 1 and add 32-bit"),
    0x2f: ("s_lshl2_add_u32", "Scalar shift left 2 and add 32-bit"),
    0x30: ("s_lshl3_add_u32", "Scalar shift left 3 and add 32-bit"),
    0x31: ("s_lshl4_add_u32", "Scalar shift left 4 and add 32-bit"),
}

SOPK_OPS = {
    0x00: ("s_movk_i32", "Scalar move 16-bit sign-extended immediate"),
    0x01: ("s_version", "Version marker with a 16-bit immediate, no destination"),
    0x02: ("s_cmovk_i32", "Scalar conditional move immediate"),
    0x03: ("s_cmpk_eq_i32", "Scalar compare equal immediate"),
    0x04: ("s_cmpk_lg_i32", "Scalar compare not-equal immediate"),
    0x05: ("s_cmpk_gt_i32", "Scalar compare greater immediate"),
    0x0f: ("s_addk_i32", "Scalar add immediate"),
    0x10: ("s_mulk_i32", "Scalar multiply immediate"),
    0x15: ("s_setreg_imm32_b32", "Write a hardware register field from a 32-bit literal"),
    0x17: ("s_waitcnt_vscnt", "Wait for vector store counter"),
    0x18: ("s_waitcnt_vmcnt", "Wait for vector memory counter"),
    0x19: ("s_waitcnt_expcnt", "Wait for export counter"),
    0x1a: ("s_waitcnt_lgkmcnt", "Wait for LDS/GDS/K/M counter"),
}

SOPC_OPS = {
    # GFX10 / RDNA 2 opcode numbers, as in LLVM SOPC_Real_gfx10 definitions
    0x00: ("s_cmp_eq_i32", "Scalar compare equal signed 32-bit"),
    0x01: ("s_cmp_lg_i32", "Scalar compare not-equal signed 32-bit"),
    0x02: ("s_cmp_gt_i32", "Scalar compare greater signed 32-bit"),
    0x03: ("s_cmp_ge_i32", "Scalar compare greater-or-equal signed 32-bit"),
    0x04: ("s_cmp_lt_i32", "Scalar compare less signed 32-bit"),
    0x05: ("s_cmp_le_i32", "Scalar compare less-or-equal signed 32-bit"),
    0x06: ("s_cmp_eq_u32", "Scalar compare equal unsigned 32-bit"),
    0x07: ("s_cmp_lg_u32", "Scalar compare not-equal unsigned 32-bit"),
    0x08: ("s_cmp_gt_u32", "Scalar compare greater unsigned 32-bit"),
    0x09: ("s_cmp_ge_u32", "Scalar compare greater-or-equal unsigned 32-bit"),
    0x0a: ("s_cmp_lt_u32", "Scalar compare less unsigned 32-bit"),
    0x0b: ("s_cmp_le_u32", "Scalar compare less-or-equal unsigned 32-bit"),
    0x0c: ("s_bitcmp0_b32", "Scalar test that bit src1 of src0 is zero"),
    0x0d: ("s_bitcmp1_b32", "Scalar test that bit src1 of src0 is one"),
    0x0e: ("s_bitcmp0_b64", "Scalar test that bit src1 of 64-bit src0 is zero"),
    0x0f: ("s_bitcmp1_b64", "Scalar test that bit src1 of 64-bit src0 is one"),
    0x12: ("s_cmp_eq_u64", "Scalar compare equal unsigned 64-bit"),
    0x13: ("s_cmp_lg_u64", "Scalar compare not-equal unsigned 64-bit"),
}

SMEM_OPS = {
    0x00: "s_load_dword",
    0x01: "s_load_dwordx2",
    0x02: "s_load_dwordx4",
    0x03: "s_load_dwordx8",
    0x04: "s_load_dwordx16",
    0x08: "s_buffer_load_dword",
    0x09: "s_buffer_load_dwordx2",
    0x0a: "s_buffer_load_dwordx4",
    0x0b: "s_buffer_load_dwordx8",
    0x0c: "s_buffer_load_dwordx16",
}

SMEM_WIDTHS = {
    0x00: (1, 2), 0x01: (2, 2), 0x02: (4, 2), 0x03: (8, 2), 0x04: (16, 2),
    0x08: (1, 4), 0x09: (2, 4), 0x0a: (4, 4), 0x0b: (8, 4), 0x0c: (16, 4),
}

MIMG_OPS = {
    0x00: ("image_load", False, 0),
    0x01: ("image_load_mip", False, 1),
    0x08: ("image_store", False, 0),
    0x09: ("image_store_mip", False, 1),
    0x0e: ("image_get_resinfo", False, 0),
    0x20: ("image_sample", True, 0),
    0x21: ("image_sample_cl", True, 1),
    0x22: ("image_sample_d", True, 0),
    0x24: ("image_sample_l", True, 1),
    0x25: ("image_sample_b", True, 1),
    0x26: ("image_sample_b_cl", True, 2),
    0x27: ("image_sample_lz", True, 0),
    0x28: ("image_sample_c", True, 1),
    0x2c: ("image_sample_c_l", True, 2),
    0x2d: ("image_sample_c_b", True, 2),
    0x2f: ("image_sample_c_lz", True, 1),
    0x30: ("image_sample_o", True, 1),
    0x34: ("image_sample_l_o", True, 2),
    0x35: ("image_sample_b_o", True, 2),
    0x37: ("image_sample_lz_o", True, 1),
    0x40: ("image_gather4", True, 0),
    0x44: ("image_gather4_l", True, 1),
    0x47: ("image_gather4_lz", True, 0),
    0x60: ("image_get_lod", True, 0),
}

MIMG_DIMS = {
    0: ("SQ_RSRC_IMG_1D", 1, 2),
    1: ("SQ_RSRC_IMG_2D", 2, 4),
    2: ("SQ_RSRC_IMG_3D", 3, 6),
    3: ("SQ_RSRC_IMG_CUBE", 3, 4),
    4: ("SQ_RSRC_IMG_1D_ARRAY", 2, 2),
    5: ("SQ_RSRC_IMG_2D_ARRAY", 3, 4),
    6: ("SQ_RSRC_IMG_2D_MSAA", 3, 4),
    7: ("SQ_RSRC_IMG_2D_MSAA_ARRAY", 4, 4),
}

VOP1_OPS = {
    0x00: ("v_nop", "Vector no-op"),
    0x01: ("v_mov_b32_e32", "Vector copy 32-bit"),
    0x02: ("v_readfirstlane_b32", "Read first active lane into SGPR"),
    0x03: ("v_cvt_i32_f64_e32", "Convert float64 to int32"),
    0x04: ("v_cvt_f64_i32_e32", "Convert int32 to float64"),
    0x05: ("v_cvt_f32_i32_e32", "Convert int32 to float32"),
    0x06: ("v_cvt_f32_u32_e32", "Convert uint32 to float32"),
    0x07: ("v_cvt_u32_f32_e32", "Convert float32 to uint32"),
    0x08: ("v_cvt_i32_f32_e32", "Convert float32 to int32"),
    0x0a: ("v_cvt_f16_f32_e32", "Convert float32 to float16"),
    0x0b: ("v_cvt_f32_f16_e32", "Convert float16 to float32"),
    0x0c: ("v_cvt_rpi_i32_f32_e32", "Convert float32 to int32 (round to +inf)"),
    0x0d: ("v_cvt_flr_i32_f32_e32", "Convert float32 to int32 (floor)"),
    0x0f: ("v_cvt_f32_f64_e32", "Convert float64 to float32"),
    0x10: ("v_cvt_f64_f32_e32", "Convert float32 to float64"),
    0x11: ("v_cvt_f32_ubyte0_e32", "Convert unsigned byte 0 to float32"),
    0x12: ("v_cvt_f32_ubyte1_e32", "Convert unsigned byte 1 to float32"),
    0x13: ("v_cvt_f32_ubyte2_e32", "Convert unsigned byte 2 to float32"),
    0x14: ("v_cvt_f32_ubyte3_e32", "Convert unsigned byte 3 to float32"),
    0x15: ("v_cvt_u32_f64_e32", "Convert float64 to uint32"),
    0x16: ("v_cvt_f64_u32_e32", "Convert uint32 to float64"),
    0x20: ("v_fract_f32_e32", "Fractional part float32"),
    0x21: ("v_trunc_f32_e32", "Truncate toward zero float32"),
    0x22: ("v_ceil_f32_e32", "Ceiling float32"),
    0x23: ("v_rndne_f32_e32", "Round to nearest even float32"),
    0x24: ("v_floor_f32_e32", "Floor float32"),
    0x25: ("v_exp_f32_e32", "Exponent 2^x float32"),
    0x27: ("v_log_f32_e32", "Logarithm log2(x) float32"),
    0x2a: ("v_rcp_f32_e32", "Reciprocal 1/x float32"),
    0x2b: ("v_rcp_iflag_f32_e32", "Reciprocal integer flag float32"),
    0x2e: ("v_rsq_f32_e32", "Reciprocal square root 1/sqrt(x)"),
    0x2f: ("v_rcp_f64_e32", "Reciprocal 1/x float64"),
    0x31: ("v_rsq_f64_e32", "Reciprocal square root 1/sqrt(x) float64"),
    0x33: ("v_sqrt_f32_e32", "Square root sqrt(x) float32"),
    0x34: ("v_sqrt_f64_e32", "Square root sqrt(x) float64"),
    0x35: ("v_sin_f32_e32", "Sine float32"),
    0x36: ("v_cos_f32_e32", "Cosine float32"),
    0x37: ("v_not_b32_e32", "Vector bitwise NOT"),
    0x38: ("v_bfrev_b32_e32", "Vector bit-reverse"),
    0x39: ("v_ffbh_u32_e32", "Find first bit high uint32"),
    0x3a: ("v_ffbl_b32_e32", "Find first bit low uint32"),
    0x3b: ("v_ffbh_i32_e32", "Find first bit high int32"),
}

VOPC_OPS = {
    0x00: ("v_cmp_f_f32_e32", "Vector compare v_cmp_f_f32"),
    0x01: ("v_cmp_lt_f32_e32", "Vector compare v_cmp_lt_f32"),
    0x02: ("v_cmp_eq_f32_e32", "Vector compare v_cmp_eq_f32"),
    0x03: ("v_cmp_le_f32_e32", "Vector compare v_cmp_le_f32"),
    0x04: ("v_cmp_gt_f32_e32", "Vector compare v_cmp_gt_f32"),
    0x05: ("v_cmp_lg_f32_e32", "Vector compare v_cmp_lg_f32"),
    0x06: ("v_cmp_ge_f32_e32", "Vector compare v_cmp_ge_f32"),
    0x07: ("v_cmp_o_f32_e32", "Vector compare v_cmp_o_f32"),
    0x08: ("v_cmp_u_f32_e32", "Vector compare v_cmp_u_f32"),
    0x09: ("v_cmp_nge_f32_e32", "Vector compare v_cmp_nge_f32"),
    0x0a: ("v_cmp_nlg_f32_e32", "Vector compare v_cmp_nlg_f32"),
    0x0b: ("v_cmp_ngt_f32_e32", "Vector compare v_cmp_ngt_f32"),
    0x0c: ("v_cmp_nle_f32_e32", "Vector compare v_cmp_nle_f32"),
    0x0d: ("v_cmp_neq_f32_e32", "Vector compare v_cmp_neq_f32"),
    0x0e: ("v_cmp_nlt_f32_e32", "Vector compare v_cmp_nlt_f32"),
    0x0f: ("v_cmp_tru_f32_e32", "Vector compare v_cmp_tru_f32"),
    0x10: ("v_cmpx_f_f32_e32", "Vector compare v_cmpx_f_f32"),
    0x11: ("v_cmpx_lt_f32_e32", "Vector compare v_cmpx_lt_f32"),
    0x12: ("v_cmpx_eq_f32_e32", "Vector compare v_cmpx_eq_f32"),
    0x13: ("v_cmpx_le_f32_e32", "Vector compare v_cmpx_le_f32"),
    0x14: ("v_cmpx_gt_f32_e32", "Vector compare v_cmpx_gt_f32"),
    0x15: ("v_cmpx_lg_f32_e32", "Vector compare v_cmpx_lg_f32"),
    0x16: ("v_cmpx_ge_f32_e32", "Vector compare v_cmpx_ge_f32"),
    0x17: ("v_cmpx_o_f32_e32", "Vector compare v_cmpx_o_f32"),
    0x18: ("v_cmpx_u_f32_e32", "Vector compare v_cmpx_u_f32"),
    0x19: ("v_cmpx_nge_f32_e32", "Vector compare v_cmpx_nge_f32"),
    0x1a: ("v_cmpx_nlg_f32_e32", "Vector compare v_cmpx_nlg_f32"),
    0x1b: ("v_cmpx_ngt_f32_e32", "Vector compare v_cmpx_ngt_f32"),
    0x1c: ("v_cmpx_nle_f32_e32", "Vector compare v_cmpx_nle_f32"),
    0x1d: ("v_cmpx_neq_f32_e32", "Vector compare v_cmpx_neq_f32"),
    0x1e: ("v_cmpx_nlt_f32_e32", "Vector compare v_cmpx_nlt_f32"),
    0x1f: ("v_cmpx_tru_f32_e32", "Vector compare v_cmpx_tru_f32"),
    0x20: ("v_cmp_f_f64_e32", "Vector compare v_cmp_f_f64"),
    0x21: ("v_cmp_lt_f64_e32", "Vector compare v_cmp_lt_f64"),
    0x22: ("v_cmp_eq_f64_e32", "Vector compare v_cmp_eq_f64"),
    0x23: ("v_cmp_le_f64_e32", "Vector compare v_cmp_le_f64"),
    0x24: ("v_cmp_gt_f64_e32", "Vector compare v_cmp_gt_f64"),
    0x25: ("v_cmp_lg_f64_e32", "Vector compare v_cmp_lg_f64"),
    0x26: ("v_cmp_ge_f64_e32", "Vector compare v_cmp_ge_f64"),
    0x27: ("v_cmp_o_f64_e32", "Vector compare v_cmp_o_f64"),
    0x28: ("v_cmp_u_f64_e32", "Vector compare v_cmp_u_f64"),
    0x29: ("v_cmp_nge_f64_e32", "Vector compare v_cmp_nge_f64"),
    0x2a: ("v_cmp_nlg_f64_e32", "Vector compare v_cmp_nlg_f64"),
    0x2b: ("v_cmp_ngt_f64_e32", "Vector compare v_cmp_ngt_f64"),
    0x2c: ("v_cmp_nle_f64_e32", "Vector compare v_cmp_nle_f64"),
    0x2d: ("v_cmp_neq_f64_e32", "Vector compare v_cmp_neq_f64"),
    0x2e: ("v_cmp_nlt_f64_e32", "Vector compare v_cmp_nlt_f64"),
    0x2f: ("v_cmp_tru_f64_e32", "Vector compare v_cmp_tru_f64"),
    0x30: ("v_cmpx_f_f64_e32", "Vector compare v_cmpx_f_f64"),
    0x31: ("v_cmpx_lt_f64_e32", "Vector compare v_cmpx_lt_f64"),
    0x32: ("v_cmpx_eq_f64_e32", "Vector compare v_cmpx_eq_f64"),
    0x33: ("v_cmpx_le_f64_e32", "Vector compare v_cmpx_le_f64"),
    0x34: ("v_cmpx_gt_f64_e32", "Vector compare v_cmpx_gt_f64"),
    0x35: ("v_cmpx_lg_f64_e32", "Vector compare v_cmpx_lg_f64"),
    0x36: ("v_cmpx_ge_f64_e32", "Vector compare v_cmpx_ge_f64"),
    0x37: ("v_cmpx_o_f64_e32", "Vector compare v_cmpx_o_f64"),
    0x38: ("v_cmpx_u_f64_e32", "Vector compare v_cmpx_u_f64"),
    0x39: ("v_cmpx_nge_f64_e32", "Vector compare v_cmpx_nge_f64"),
    0x3a: ("v_cmpx_nlg_f64_e32", "Vector compare v_cmpx_nlg_f64"),
    0x3b: ("v_cmpx_ngt_f64_e32", "Vector compare v_cmpx_ngt_f64"),
    0x3c: ("v_cmpx_nle_f64_e32", "Vector compare v_cmpx_nle_f64"),
    0x3d: ("v_cmpx_neq_f64_e32", "Vector compare v_cmpx_neq_f64"),
    0x3e: ("v_cmpx_nlt_f64_e32", "Vector compare v_cmpx_nlt_f64"),
    0x3f: ("v_cmpx_tru_f64_e32", "Vector compare v_cmpx_tru_f64"),
    0x80: ("v_cmp_f_i32_e32", "Vector compare v_cmp_f_i32"),
    0x81: ("v_cmp_lt_i32_e32", "Vector compare v_cmp_lt_i32"),
    0x82: ("v_cmp_eq_i32_e32", "Vector compare v_cmp_eq_i32"),
    0x83: ("v_cmp_le_i32_e32", "Vector compare v_cmp_le_i32"),
    0x84: ("v_cmp_gt_i32_e32", "Vector compare v_cmp_gt_i32"),
    0x85: ("v_cmp_ne_i32_e32", "Vector compare v_cmp_ne_i32"),
    0x86: ("v_cmp_ge_i32_e32", "Vector compare v_cmp_ge_i32"),
    0x87: ("v_cmp_t_i32_e32", "Vector compare v_cmp_t_i32"),
    0x88: ("v_cmp_class_f32_e32", "Vector compare v_cmp_class_f32"),
    0x89: ("v_cmp_lt_i16_e32", "Vector compare v_cmp_lt_i16"),
    0x8a: ("v_cmp_eq_i16_e32", "Vector compare v_cmp_eq_i16"),
    0x8b: ("v_cmp_le_i16_e32", "Vector compare v_cmp_le_i16"),
    0x8c: ("v_cmp_gt_i16_e32", "Vector compare v_cmp_gt_i16"),
    0x8d: ("v_cmp_ne_i16_e32", "Vector compare v_cmp_ne_i16"),
    0x8e: ("v_cmp_ge_i16_e32", "Vector compare v_cmp_ge_i16"),
    0x8f: ("v_cmp_class_f16_e32", "Vector compare v_cmp_class_f16"),
    0x90: ("v_cmpx_f_i32_e32", "Vector compare v_cmpx_f_i32"),
    0x91: ("v_cmpx_lt_i32_e32", "Vector compare v_cmpx_lt_i32"),
    0x92: ("v_cmpx_eq_i32_e32", "Vector compare v_cmpx_eq_i32"),
    0x93: ("v_cmpx_le_i32_e32", "Vector compare v_cmpx_le_i32"),
    0x94: ("v_cmpx_gt_i32_e32", "Vector compare v_cmpx_gt_i32"),
    0x95: ("v_cmpx_ne_i32_e32", "Vector compare v_cmpx_ne_i32"),
    0x96: ("v_cmpx_ge_i32_e32", "Vector compare v_cmpx_ge_i32"),
    0x97: ("v_cmpx_t_i32_e32", "Vector compare v_cmpx_t_i32"),
    0x98: ("v_cmpx_class_f32_e32", "Vector compare v_cmpx_class_f32"),
    0x99: ("v_cmpx_lt_i16_e32", "Vector compare v_cmpx_lt_i16"),
    0x9a: ("v_cmpx_eq_i16_e32", "Vector compare v_cmpx_eq_i16"),
    0x9b: ("v_cmpx_le_i16_e32", "Vector compare v_cmpx_le_i16"),
    0x9c: ("v_cmpx_gt_i16_e32", "Vector compare v_cmpx_gt_i16"),
    0x9d: ("v_cmpx_ne_i16_e32", "Vector compare v_cmpx_ne_i16"),
    0x9e: ("v_cmpx_ge_i16_e32", "Vector compare v_cmpx_ge_i16"),
    0x9f: ("v_cmpx_class_f16_e32", "Vector compare v_cmpx_class_f16"),
    0xa0: ("v_cmp_f_i64_e32", "Vector compare v_cmp_f_i64"),
    0xa1: ("v_cmp_lt_i64_e32", "Vector compare v_cmp_lt_i64"),
    0xa2: ("v_cmp_eq_i64_e32", "Vector compare v_cmp_eq_i64"),
    0xa3: ("v_cmp_le_i64_e32", "Vector compare v_cmp_le_i64"),
    0xa4: ("v_cmp_gt_i64_e32", "Vector compare v_cmp_gt_i64"),
    0xa5: ("v_cmp_ne_i64_e32", "Vector compare v_cmp_ne_i64"),
    0xa6: ("v_cmp_ge_i64_e32", "Vector compare v_cmp_ge_i64"),
    0xa7: ("v_cmp_t_i64_e32", "Vector compare v_cmp_t_i64"),
    0xa8: ("v_cmp_class_f64_e32", "Vector compare v_cmp_class_f64"),
    0xa9: ("v_cmp_lt_u16_e32", "Vector compare v_cmp_lt_u16"),
    0xaa: ("v_cmp_eq_u16_e32", "Vector compare v_cmp_eq_u16"),
    0xab: ("v_cmp_le_u16_e32", "Vector compare v_cmp_le_u16"),
    0xac: ("v_cmp_gt_u16_e32", "Vector compare v_cmp_gt_u16"),
    0xad: ("v_cmp_ne_u16_e32", "Vector compare v_cmp_ne_u16"),
    0xae: ("v_cmp_ge_u16_e32", "Vector compare v_cmp_ge_u16"),
    0xb0: ("v_cmpx_f_i64_e32", "Vector compare v_cmpx_f_i64"),
    0xb1: ("v_cmpx_lt_i64_e32", "Vector compare v_cmpx_lt_i64"),
    0xb2: ("v_cmpx_eq_i64_e32", "Vector compare v_cmpx_eq_i64"),
    0xb3: ("v_cmpx_le_i64_e32", "Vector compare v_cmpx_le_i64"),
    0xb4: ("v_cmpx_gt_i64_e32", "Vector compare v_cmpx_gt_i64"),
    0xb5: ("v_cmpx_ne_i64_e32", "Vector compare v_cmpx_ne_i64"),
    0xb6: ("v_cmpx_ge_i64_e32", "Vector compare v_cmpx_ge_i64"),
    0xb7: ("v_cmpx_t_i64_e32", "Vector compare v_cmpx_t_i64"),
    0xb8: ("v_cmpx_class_f64_e32", "Vector compare v_cmpx_class_f64"),
    0xb9: ("v_cmpx_lt_u16_e32", "Vector compare v_cmpx_lt_u16"),
    0xba: ("v_cmpx_eq_u16_e32", "Vector compare v_cmpx_eq_u16"),
    0xbb: ("v_cmpx_le_u16_e32", "Vector compare v_cmpx_le_u16"),
    0xbc: ("v_cmpx_gt_u16_e32", "Vector compare v_cmpx_gt_u16"),
    0xbd: ("v_cmpx_ne_u16_e32", "Vector compare v_cmpx_ne_u16"),
    0xbe: ("v_cmpx_ge_u16_e32", "Vector compare v_cmpx_ge_u16"),
    0xc0: ("v_cmp_f_u32_e32", "Vector compare v_cmp_f_u32"),
    0xc1: ("v_cmp_lt_u32_e32", "Vector compare v_cmp_lt_u32"),
    0xc2: ("v_cmp_eq_u32_e32", "Vector compare v_cmp_eq_u32"),
    0xc3: ("v_cmp_le_u32_e32", "Vector compare v_cmp_le_u32"),
    0xc4: ("v_cmp_gt_u32_e32", "Vector compare v_cmp_gt_u32"),
    0xc5: ("v_cmp_ne_u32_e32", "Vector compare v_cmp_ne_u32"),
    0xc6: ("v_cmp_ge_u32_e32", "Vector compare v_cmp_ge_u32"),
    0xc7: ("v_cmp_t_u32_e32", "Vector compare v_cmp_t_u32"),
    0xc8: ("v_cmp_f_f16_e32", "Vector compare v_cmp_f_f16"),
    0xc9: ("v_cmp_lt_f16_e32", "Vector compare v_cmp_lt_f16"),
    0xca: ("v_cmp_eq_f16_e32", "Vector compare v_cmp_eq_f16"),
    0xcb: ("v_cmp_le_f16_e32", "Vector compare v_cmp_le_f16"),
    0xcc: ("v_cmp_gt_f16_e32", "Vector compare v_cmp_gt_f16"),
    0xcd: ("v_cmp_lg_f16_e32", "Vector compare v_cmp_lg_f16"),
    0xce: ("v_cmp_ge_f16_e32", "Vector compare v_cmp_ge_f16"),
    0xcf: ("v_cmp_o_f16_e32", "Vector compare v_cmp_o_f16"),
    0xd0: ("v_cmpx_f_u32_e32", "Vector compare v_cmpx_f_u32"),
    0xd1: ("v_cmpx_lt_u32_e32", "Vector compare v_cmpx_lt_u32"),
    0xd2: ("v_cmpx_eq_u32_e32", "Vector compare v_cmpx_eq_u32"),
    0xd3: ("v_cmpx_le_u32_e32", "Vector compare v_cmpx_le_u32"),
    0xd4: ("v_cmpx_gt_u32_e32", "Vector compare v_cmpx_gt_u32"),
    0xd5: ("v_cmpx_ne_u32_e32", "Vector compare v_cmpx_ne_u32"),
    0xd6: ("v_cmpx_ge_u32_e32", "Vector compare v_cmpx_ge_u32"),
    0xd7: ("v_cmpx_t_u32_e32", "Vector compare v_cmpx_t_u32"),
    0xd8: ("v_cmpx_f_f16_e32", "Vector compare v_cmpx_f_f16"),
    0xd9: ("v_cmpx_lt_f16_e32", "Vector compare v_cmpx_lt_f16"),
    0xda: ("v_cmpx_eq_f16_e32", "Vector compare v_cmpx_eq_f16"),
    0xdb: ("v_cmpx_le_f16_e32", "Vector compare v_cmpx_le_f16"),
    0xdc: ("v_cmpx_gt_f16_e32", "Vector compare v_cmpx_gt_f16"),
    0xdd: ("v_cmpx_lg_f16_e32", "Vector compare v_cmpx_lg_f16"),
    0xde: ("v_cmpx_ge_f16_e32", "Vector compare v_cmpx_ge_f16"),
    0xdf: ("v_cmpx_o_f16_e32", "Vector compare v_cmpx_o_f16"),
    0xe0: ("v_cmp_f_u64_e32", "Vector compare v_cmp_f_u64"),
    0xe1: ("v_cmp_lt_u64_e32", "Vector compare v_cmp_lt_u64"),
    0xe2: ("v_cmp_eq_u64_e32", "Vector compare v_cmp_eq_u64"),
    0xe3: ("v_cmp_le_u64_e32", "Vector compare v_cmp_le_u64"),
    0xe4: ("v_cmp_gt_u64_e32", "Vector compare v_cmp_gt_u64"),
    0xe5: ("v_cmp_ne_u64_e32", "Vector compare v_cmp_ne_u64"),
    0xe6: ("v_cmp_ge_u64_e32", "Vector compare v_cmp_ge_u64"),
    0xe7: ("v_cmp_t_u64_e32", "Vector compare v_cmp_t_u64"),
    0xe8: ("v_cmp_u_f16_e32", "Vector compare v_cmp_u_f16"),
    0xe9: ("v_cmp_nge_f16_e32", "Vector compare v_cmp_nge_f16"),
    0xea: ("v_cmp_nlg_f16_e32", "Vector compare v_cmp_nlg_f16"),
    0xeb: ("v_cmp_ngt_f16_e32", "Vector compare v_cmp_ngt_f16"),
    0xec: ("v_cmp_nle_f16_e32", "Vector compare v_cmp_nle_f16"),
    0xed: ("v_cmp_neq_f16_e32", "Vector compare v_cmp_neq_f16"),
    0xee: ("v_cmp_nlt_f16_e32", "Vector compare v_cmp_nlt_f16"),
    0xef: ("v_cmp_tru_f16_e32", "Vector compare v_cmp_tru_f16"),
    0xf0: ("v_cmpx_f_u64_e32", "Vector compare v_cmpx_f_u64"),
    0xf1: ("v_cmpx_lt_u64_e32", "Vector compare v_cmpx_lt_u64"),
    0xf2: ("v_cmpx_eq_u64_e32", "Vector compare v_cmpx_eq_u64"),
    0xf3: ("v_cmpx_le_u64_e32", "Vector compare v_cmpx_le_u64"),
    0xf4: ("v_cmpx_gt_u64_e32", "Vector compare v_cmpx_gt_u64"),
    0xf5: ("v_cmpx_ne_u64_e32", "Vector compare v_cmpx_ne_u64"),
    0xf6: ("v_cmpx_ge_u64_e32", "Vector compare v_cmpx_ge_u64"),
    0xf7: ("v_cmpx_t_u64_e32", "Vector compare v_cmpx_t_u64"),
    0xf8: ("v_cmpx_u_f16_e32", "Vector compare v_cmpx_u_f16"),
    0xf9: ("v_cmpx_nge_f16_e32", "Vector compare v_cmpx_nge_f16"),
    0xfa: ("v_cmpx_nlg_f16_e32", "Vector compare v_cmpx_nlg_f16"),
    0xfb: ("v_cmpx_ngt_f16_e32", "Vector compare v_cmpx_ngt_f16"),
    0xfc: ("v_cmpx_nle_f16_e32", "Vector compare v_cmpx_nle_f16"),
    0xfd: ("v_cmpx_neq_f16_e32", "Vector compare v_cmpx_neq_f16"),
    0xfe: ("v_cmpx_nlt_f16_e32", "Vector compare v_cmpx_nlt_f16"),
    0xff: ("v_cmpx_tru_f16_e32", "Vector compare v_cmpx_tru_f16"),
}


VOP2_OPS = {
    0x01: ("v_cndmask_b32_e32", "Vector conditional mask select"),
    0x03: ("v_add_f32_e32", "Vector add float32"),
    0x04: ("v_sub_f32_e32", "Vector subtract float32"),
    0x05: ("v_subrev_f32_e32", "Vector reverse subtract float32"),
    0x07: ("v_mul_legacy_f32_e32", "Vector multiply zero-preserving float32"),
    0x08: ("v_mul_f32_e32", "Vector multiply float32"),
    0x0f: ("v_min_f32_e32", "Vector minimum float32"),
    0x10: ("v_max_f32_e32", "Vector maximum float32"),
    0x11: ("v_min_i32_e32", "Vector minimum int32"),
    0x12: ("v_max_i32_e32", "Vector maximum int32"),
    0x13: ("v_min_u32_e32", "Vector minimum uint32"),
    0x14: ("v_max_u32_e32", "Vector maximum uint32"),
    0x16: ("v_lshrrev_b32_e32", "Vector shift right logical"),
    0x18: ("v_ashrrev_i32_e32", "Vector shift right arithmetic"),
    0x1a: ("v_lshlrev_b32_e32", "Vector shift left logical"),
    0x1b: ("v_and_b32_e32", "Vector bitwise AND"),
    0x1c: ("v_or_b32_e32", "Vector bitwise OR"),
    0x1d: ("v_xor_b32_e32", "Vector bitwise XOR"),
    0x1f: ("v_mac_f32_e32", "Vector multiply-accumulate float32"),
    0x20: ("v_madmk_f32", "Vector multiply-add (non-fused) with literal multiplier"),
    0x21: ("v_madak_f32", "Vector multiply-add (non-fused) with literal addend"),
    0x25: ("v_add_nc_u32_e32", "Vector add uint32 without carry"),
    0x26: ("v_sub_nc_u32_e32", "Vector sub uint32 without carry"),
    0x27: ("v_subrev_nc_u32_e32", "Vector reverse sub uint32 without carry"),
    0x2b: ("v_fmac_f32_e32", "Vector fused multiply-accumulate float32"),
    0x2c: ("v_fmamk_f32", "Vector fused multiply-add with literal multiplier"),
    0x2d: ("v_fmaak_f32", "Vector fused multiply-add with literal addend"),
    0x2f: ("v_cvt_pkrtz_f16_f32_e32", "Convert two float32 to packed float16 (round to zero)"),
    0x37: ("v_fmamk_f16", "Vector fused multiply-add float16 with literal multiplier"),
    0x38: ("v_fmaak_f16", "Vector fused multiply-add float16 with literal addend"),
}

VOP3_NATIVE_OPS = {
    0x128: ("v_add_co_ci_u32_e64", "V_ADD_CO_CI_U32 (VOP3B)", 3, True, 1, (1, 1, 1)),
    0x129: ("v_sub_co_ci_u32_e64", "V_SUB_CO_CI_U32 (VOP3B)", 3, True, 1, (1, 1, 1)),
    0x12a: ("v_subrev_co_ci_u32_e64", "V_SUBREV_CO_CI_U32 (VOP3B)", 3, True, 1, (1, 1, 1)),
    0x140: ("v_mad_legacy_f32", "V_MAD_LEGACY_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x141: ("v_mad_f32", "V_MAD_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x142: ("v_mad_i32_i24", "V_MAD_I32_I24 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x143: ("v_mad_u32_u24", "V_MAD_U32_U24 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x144: ("v_cubeid_f32", "V_CUBEID_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x145: ("v_cubesc_f32", "V_CUBESC_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x146: ("v_cubetc_f32", "V_CUBETC_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x147: ("v_cubema_f32", "V_CUBEMA_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x148: ("v_bfe_u32", "V_BFE_U32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x149: ("v_bfe_i32", "V_BFE_I32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x14a: ("v_bfi_b32", "V_BFI_B32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x14b: ("v_fma_f32", "V_FMA_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x14c: ("v_fma_f64", "V_FMA_F64 (VOP3A)", 3, False, 2, (2, 2, 2)),
    0x14d: ("v_lerp_u8", "V_LERP_U8 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x14e: ("v_alignbit_b32", "V_ALIGNBIT_B32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x14f: ("v_alignbyte_b32", "V_ALIGNBYTE_B32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x150: ("v_mullit_f32", "V_MULLIT_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x151: ("v_min3_f32", "V_MIN3_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x152: ("v_min3_i32", "V_MIN3_I32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x153: ("v_min3_u32", "V_MIN3_U32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x154: ("v_max3_f32", "V_MAX3_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x155: ("v_max3_i32", "V_MAX3_I32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x156: ("v_max3_u32", "V_MAX3_U32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x157: ("v_med3_f32", "V_MED3_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x158: ("v_med3_i32", "V_MED3_I32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x159: ("v_med3_u32", "V_MED3_U32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x15a: ("v_sad_u8", "V_SAD_U8 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x15b: ("v_sad_hi_u8", "V_SAD_HI_U8 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x15c: ("v_sad_u16", "V_SAD_U16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x15d: ("v_sad_u32", "V_SAD_U32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x15e: ("v_cvt_pk_u8_f32", "V_CVT_PK_U8_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x15f: ("v_div_fixup_f32", "V_DIV_FIXUP_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x160: ("v_div_fixup_f64", "V_DIV_FIXUP_F64 (VOP3A)", 3, False, 2, (2, 2, 2)),
    0x164: ("v_add_f64", "V_ADD_F64 (VOP3A)", 2, False, 2, (2, 2)),
    0x165: ("v_mul_f64", "V_MUL_F64 (VOP3A)", 2, False, 2, (2, 2)),
    0x166: ("v_min_f64", "V_MIN_F64 (VOP3A)", 2, False, 2, (2, 2)),
    0x167: ("v_max_f64", "V_MAX_F64 (VOP3A)", 2, False, 2, (2, 2)),
    0x168: ("v_ldexp_f64", "V_LDEXP_F64 (VOP3A)", 2, False, 2, (2, 1)),
    0x169: ("v_mul_lo_u32", "V_MUL_LO_U32 (VOP3A)", 2, False, 1, (1, 1)),
    0x16a: ("v_mul_hi_u32", "V_MUL_HI_U32 (VOP3A)", 2, False, 1, (1, 1)),
    0x16c: ("v_mul_hi_i32", "V_MUL_HI_I32 (VOP3A)", 2, False, 1, (1, 1)),
    0x16d: ("v_div_scale_f32", "V_DIV_SCALE_F32 (VOP3B)", 3, True, 1, (1, 1, 1)),
    0x16e: ("v_div_scale_f64", "V_DIV_SCALE_F64 (VOP3B)", 3, True, 2, (2, 2, 2)),
    0x16f: ("v_div_fmas_f32", "V_DIV_FMAS_F32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x170: ("v_div_fmas_f64", "V_DIV_FMAS_F64 (VOP3A)", 3, False, 2, (2, 2, 2)),
    0x171: ("v_msad_u8", "V_MSAD_U8 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x174: ("v_trig_preop_f64", "V_TRIG_PREOP_F64 (VOP3A)", 2, False, 2, (2, 1)),
    0x176: ("v_mad_u64_u32", "V_MAD_U64_U32 (VOP3B)", 3, True, 2, (1, 1, 2)),
    0x177: ("v_mad_i64_i32", "V_MAD_I64_I32 (VOP3B)", 3, True, 2, (1, 1, 2)),
    0x178: ("v_xor3_b32", "V_XOR3_B32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x2ff: ("v_lshlrev_b64", "V_LSHLREV_B64 (VOP3A)", 2, False, 2, (1, 2)),
    0x300: ("v_lshrrev_b64", "V_LSHRREV_B64 (VOP3A)", 2, False, 2, (1, 2)),
    0x301: ("v_ashrrev_i64", "V_ASHRREV_I64 (VOP3A)", 2, False, 2, (1, 2)),
    0x303: ("v_add_nc_u16", "V_ADD_NC_U16 (VOP3A)", 2, False, 1, (1, 1)),
    0x304: ("v_sub_nc_u16", "V_SUB_NC_U16 (VOP3A)", 2, False, 1, (1, 1)),
    0x305: ("v_mul_lo_u16", "V_MUL_LO_U16 (VOP3A)", 2, False, 1, (1, 1)),
    0x307: ("v_lshrrev_b16", "V_LSHRREV_B16 (VOP3A)", 2, False, 1, (1, 1)),
    0x308: ("v_ashrrev_i16", "V_ASHRREV_I16 (VOP3A)", 2, False, 1, (1, 1)),
    0x309: ("v_max_u16", "V_MAX_U16 (VOP3A)", 2, False, 1, (1, 1)),
    0x30a: ("v_max_i16", "V_MAX_I16 (VOP3A)", 2, False, 1, (1, 1)),
    0x30b: ("v_min_u16", "V_MIN_U16 (VOP3A)", 2, False, 1, (1, 1)),
    0x30c: ("v_min_i16", "V_MIN_I16 (VOP3A)", 2, False, 1, (1, 1)),
    0x30d: ("v_add_nc_i16", "V_ADD_NC_I16 (VOP3A)", 2, False, 1, (1, 1)),
    0x30e: ("v_sub_nc_i16", "V_SUB_NC_I16 (VOP3A)", 2, False, 1, (1, 1)),
    0x30f: ("v_add_co_u32", "V_ADD_CO_U32 (VOP3B)", 2, True, 1, (1, 1)),
    0x310: ("v_sub_co_u32", "V_SUB_CO_U32 (VOP3B)", 2, True, 1, (1, 1)),
    0x311: ("v_pack_b32_f16", "V_PACK_B32_F16 (VOP3A)", 2, False, 1, (1, 1)),
    0x312: ("v_cvt_pknorm_i16_f16", "V_CVT_PKNORM_I16_F16 (VOP3A)", 2, False, 1, (1, 1)),
    0x313: ("v_cvt_pknorm_u16_f16", "V_CVT_PKNORM_U16_F16 (VOP3A)", 2, False, 1, (1, 1)),
    0x314: ("v_lshlrev_b16", "V_LSHLREV_B16 (VOP3A)", 2, False, 1, (1, 1)),
    0x319: ("v_subrev_co_u32", "V_SUBREV_CO_U32 (VOP3B)", 2, True, 1, (1, 1)),
    0x340: ("v_mad_u16", "V_MAD_U16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x344: ("v_perm_b32", "V_PERM_B32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x345: ("v_xad_u32", "V_XAD_U32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x346: ("v_lshl_add_u32", "V_LSHL_ADD_U32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x347: ("v_add_lshl_u32", "V_ADD_LSHL_U32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x34b: ("v_fma_f16", "V_FMA_F16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x351: ("v_min3_f16", "V_MIN3_F16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x352: ("v_min3_i16", "V_MIN3_I16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x353: ("v_min3_u16", "V_MIN3_U16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x354: ("v_max3_f16", "V_MAX3_F16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x355: ("v_max3_i16", "V_MAX3_I16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x356: ("v_max3_u16", "V_MAX3_U16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x357: ("v_med3_f16", "V_MED3_F16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x358: ("v_med3_i16", "V_MED3_I16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x359: ("v_med3_u16", "V_MED3_U16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x35e: ("v_mad_i16", "V_MAD_I16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x35f: ("v_div_fixup_f16", "V_DIV_FIXUP_F16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x362: ("v_ldexp_f32", "V_LDEXP_F32 (VOP3A)", 2, False, 1, (1, 1)),
    0x363: ("v_bfm_b32", "V_BFM_B32 (VOP3A)", 2, False, 1, (1, 1)),
    0x364: ("v_bcnt_u32_b32", "V_BCNT_U32_B32 (VOP3A)", 2, False, 1, (1, 1)),
    0x365: ("v_mbcnt_lo_u32_b32", "V_MBCNT_LO_U32_B32 (VOP3A)", 2, False, 1, (1, 1)),
    0x366: ("v_mbcnt_hi_u32_b32", "V_MBCNT_HI_U32_B32 (VOP3A)", 2, False, 1, (1, 1)),
    0x368: ("v_cvt_pknorm_i16_f32", "V_CVT_PKNORM_I16_F32 (VOP3A)", 2, False, 1, (1, 1)),
    0x369: ("v_cvt_pknorm_u16_f32", "V_CVT_PKNORM_U16_F32 (VOP3A)", 2, False, 1, (1, 1)),
    0x36a: ("v_cvt_pk_u16_u32", "V_CVT_PK_U16_U32 (VOP3A)", 2, False, 1, (1, 1)),
    0x36b: ("v_cvt_pk_i16_i32", "V_CVT_PK_I16_I32 (VOP3A)", 2, False, 1, (1, 1)),
    0x36d: ("v_add3_u32", "V_ADD3_U32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x36f: ("v_lshl_or_b32", "V_LSHL_OR_B32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x371: ("v_and_or_b32", "V_AND_OR_B32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x372: ("v_or3_b32", "V_OR3_B32 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x373: ("v_mad_u32_u16", "V_MAD_U32_U16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x375: ("v_mad_i32_i16", "V_MAD_I32_I16 (VOP3A)", 3, False, 1, (1, 1, 1)),
    0x376: ("v_sub_nc_i32", "V_SUB_NC_I32 (VOP3A)", 2, False, 1, (1, 1)),
    0x37f: ("v_add_nc_i32", "V_ADD_NC_I32 (VOP3A)", 2, False, 1, (1, 1)),
}


VOP3P_OPS = {
    0x00: ("v_pk_mad_i16", "Multiply-add signed 16-bit packed"),
    0x01: ("v_pk_mul_lo_u16", "Multiply low unsigned 16-bit packed"),
    0x02: ("v_pk_add_i16", "Add signed 16-bit packed"),
    0x03: ("v_pk_sub_i16", "Subtract signed 16-bit packed"),
    0x04: ("v_pk_lshlrev_b16", "Logical shift left reverse 16-bit packed"),
    0x05: ("v_pk_lshrrev_b16", "Logical shift right reverse 16-bit packed"),
    0x06: ("v_pk_ashrrev_i16", "Arithmetic shift right reverse 16-bit packed"),
    0x07: ("v_pk_max_i16", "Maximum signed 16-bit packed"),
    0x08: ("v_pk_min_i16", "Minimum signed 16-bit packed"),
    0x09: ("v_pk_mad_u16", "Multiply-add unsigned 16-bit packed"),
    0x0a: ("v_pk_add_u16", "Add unsigned 16-bit packed"),
    0x0b: ("v_pk_sub_u16", "Subtract unsigned 16-bit packed"),
    0x0c: ("v_pk_max_u16", "Maximum unsigned 16-bit packed"),
    0x0d: ("v_pk_min_u16", "Minimum unsigned 16-bit packed"),
    0x0e: ("v_pk_fma_f16", "Fused multiply-add float16 packed"),
    0x0f: ("v_pk_add_f16", "Add float16 packed"),
    0x10: ("v_pk_mul_f16", "Multiply float16 packed"),
    0x11: ("v_pk_min_f16", "Minimum float16 packed"),
    0x12: ("v_pk_max_f16", "Maximum float16 packed"),
    0x13: ("v_dot2_f32_f16", "Dot product 2-element float16 accumulate to float32"),
    0x14: ("v_dot2_i32_i16", "Dot product 2-element signed 16-bit accumulate to int32"),
    0x15: ("v_dot2_u32_u16", "Dot product 2-element unsigned 16-bit accumulate to uint32"),
    0x16: ("v_dot4_i32_i8", "Dot product 4-element signed 8-bit accumulate to int32"),
    0x17: ("v_dot4_u32_u8", "Dot product 4-element unsigned 8-bit accumulate to uint32"),
    0x18: ("v_dot8_i32_i4", "Dot product 8-element signed 4-bit accumulate to int32"),
    0x19: ("v_dot8_u32_u4", "Dot product 8-element unsigned 4-bit accumulate to uint32"),
    0x20: ("v_fma_mix_f32", "Fused multiply-add mixed precision to float32"),
    0x21: ("v_fma_mixlo_f16", "Fused multiply-add mixed precision to float16 low"),
    0x22: ("v_fma_mixhi_f16", "Fused multiply-add mixed precision to float16 high"),
}

VOP3P_3SRC = {
    0x00, 0x09, 0x0e, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x20, 0x21, 0x22,
}

EXP_TARGETS = {
    0: "mrt0", 1: "mrt1", 2: "mrt2", 3: "mrt3",
    4: "mrt4", 5: "mrt5", 6: "mrt6", 7: "mrt7",
    8: "mrtz", 9: "null",
    12: "pos0", 13: "pos1", 14: "pos2", 15: "pos3",
    **{32 + i: f"param{i}" for i in range(32)},
}

# GFX1013 / RDNA 2 MUBUF opcodes: op -> (mnemonic, vdata_width, comment)
MUBUF_OPS = {
    0x00: ("buffer_load_format_x", 1, "Untyped/typed buffer format load 1 dword"),
    0x01: ("buffer_load_format_xy", 2, "Untyped/typed buffer format load 2 dwords"),
    0x02: ("buffer_load_format_xyz", 3, "Untyped/typed buffer format load 3 dwords"),
    0x03: ("buffer_load_format_xyzw", 4, "Untyped/typed buffer format load 4 dwords"),
    0x04: ("buffer_store_format_x", 1, "Buffer format store 1 dword"),
    0x05: ("buffer_store_format_xy", 2, "Buffer format store 2 dwords"),
    0x06: ("buffer_store_format_xyz", 3, "Buffer format store 3 dwords"),
    0x07: ("buffer_store_format_xyzw", 4, "Buffer format store 4 dwords"),
    0x08: ("buffer_load_ubyte", 1, "Buffer load unsigned byte"),
    0x09: ("buffer_load_sbyte", 1, "Buffer load signed byte"),
    0x0a: ("buffer_load_ushort", 1, "Buffer load unsigned 16-bit"),
    0x0b: ("buffer_load_sshort", 1, "Buffer load signed 16-bit"),
    0x0c: ("buffer_load_dword", 1, "Buffer load 1 dword"),
    0x0d: ("buffer_load_dwordx2", 2, "Buffer load 2 dwords"),
    0x0e: ("buffer_load_dwordx4", 4, "Buffer load 4 dwords"),
    0x0f: ("buffer_load_dwordx3", 3, "Buffer load 3 dwords"),
    0x18: ("buffer_store_byte", 1, "Buffer store byte"),
    0x19: ("buffer_store_byte_d16_hi", 1, "Buffer store byte from high 16-bit lane"),
    0x1a: ("buffer_store_short", 1, "Buffer store 16-bit short"),
    0x1b: ("buffer_store_short_d16_hi", 1, "Buffer store 16-bit short from high lane"),
    0x1c: ("buffer_store_dword", 1, "Buffer store 1 dword"),
    0x1d: ("buffer_store_dwordx2", 2, "Buffer store 2 dwords"),
    0x1e: ("buffer_store_dwordx4", 4, "Buffer store 4 dwords"),
    0x1f: ("buffer_store_dwordx3", 3, "Buffer store 3 dwords"),
    0x20: ("buffer_load_ubyte_d16", 1, "Buffer load unsigned byte to low 16 bits"),
    0x21: ("buffer_load_ubyte_d16_hi", 1, "Buffer load unsigned byte to high 16 bits"),
    0x22: ("buffer_load_sbyte_d16", 1, "Buffer load signed byte to low 16 bits"),
    0x23: ("buffer_load_sbyte_d16_hi", 1, "Buffer load signed byte to high 16 bits"),
    0x24: ("buffer_load_short_d16", 1, "Buffer load 16-bit short to low 16 bits"),
    0x25: ("buffer_load_short_d16_hi", 1, "Buffer load 16-bit short to high 16 bits"),
    0x30: ("buffer_atomic_swap", 1, "Buffer atomic swap 32-bit"),
    0x31: ("buffer_atomic_cmpswap", 2, "Buffer atomic compare-and-swap 32-bit"),
    0x32: ("buffer_atomic_add", 1, "Buffer atomic add 32-bit"),
    0x33: ("buffer_atomic_sub", 1, "Buffer atomic subtract 32-bit"),
    0x35: ("buffer_atomic_smin", 1, "Buffer atomic signed min 32-bit"),
    0x36: ("buffer_atomic_umin", 1, "Buffer atomic unsigned min 32-bit"),
    0x37: ("buffer_atomic_smax", 1, "Buffer atomic signed max 32-bit"),
    0x38: ("buffer_atomic_umax", 1, "Buffer atomic unsigned max 32-bit"),
    0x39: ("buffer_atomic_and", 1, "Buffer atomic bitwise AND 32-bit"),
    0x3a: ("buffer_atomic_or", 1, "Buffer atomic bitwise OR 32-bit"),
    0x3b: ("buffer_atomic_xor", 1, "Buffer atomic bitwise XOR 32-bit"),
    0x3c: ("buffer_atomic_inc", 1, "Buffer atomic increment 32-bit"),
    0x3d: ("buffer_atomic_dec", 1, "Buffer atomic decrement 32-bit"),
    0x3e: ("buffer_atomic_fcmpswap", 2, "Buffer atomic float32 compare-and-swap"),
    0x3f: ("buffer_atomic_fmin", 1, "Buffer atomic float32 min"),
    0x40: ("buffer_atomic_fmax", 1, "Buffer atomic float32 max"),
    0x50: ("buffer_atomic_swap_x2", 2, "Buffer atomic swap 64-bit"),
    0x51: ("buffer_atomic_cmpswap_x2", 4, "Buffer atomic compare-and-swap 64-bit"),
    0x52: ("buffer_atomic_add_x2", 2, "Buffer atomic add 64-bit"),
    0x53: ("buffer_atomic_sub_x2", 2, "Buffer atomic subtract 64-bit"),
    0x55: ("buffer_atomic_smin_x2", 2, "Buffer atomic signed min 64-bit"),
    0x56: ("buffer_atomic_umin_x2", 2, "Buffer atomic unsigned min 64-bit"),
    0x57: ("buffer_atomic_smax_x2", 2, "Buffer atomic signed max 64-bit"),
    0x58: ("buffer_atomic_umax_x2", 2, "Buffer atomic unsigned max 64-bit"),
    0x59: ("buffer_atomic_and_x2", 2, "Buffer atomic bitwise AND 64-bit"),
    0x5a: ("buffer_atomic_or_x2", 2, "Buffer atomic bitwise OR 64-bit"),
    0x5b: ("buffer_atomic_xor_x2", 2, "Buffer atomic bitwise XOR 64-bit"),
    0x5c: ("buffer_atomic_inc_x2", 2, "Buffer atomic increment 64-bit"),
    0x5d: ("buffer_atomic_dec_x2", 2, "Buffer atomic decrement 64-bit"),
    0x5e: ("buffer_atomic_fcmpswap_x2", 4, "Buffer atomic float64 compare-and-swap"),
    0x5f: ("buffer_atomic_fmin_x2", 2, "Buffer atomic float64 min"),
    0x60: ("buffer_atomic_fmax_x2", 2, "Buffer atomic float64 max"),
    0x71: ("buffer_gl0_inv", 0, "Invalidate GL0 vector cache"),
    0x72: ("buffer_gl1_inv", 0, "Invalidate GL1 vector cache"),
    0x80: ("buffer_load_format_d16_x", 1, "Buffer format load 1 packed 16-bit channel"),
    0x81: ("buffer_load_format_d16_xy", 1, "Buffer format load 2 packed 16-bit channels"),
    0x82: ("buffer_load_format_d16_xyz", 2, "Buffer format load 3 packed 16-bit channels"),
    0x83: ("buffer_load_format_d16_xyzw", 2, "Buffer format load 4 packed 16-bit channels"),
    0x84: ("buffer_store_format_d16_x", 1, "Buffer format store 1 packed 16-bit channel"),
    0x85: ("buffer_store_format_d16_xy", 1, "Buffer format store 2 packed 16-bit channels"),
    0x86: ("buffer_store_format_d16_xyz", 2, "Buffer format store 3 packed 16-bit channels"),
    0x87: ("buffer_store_format_d16_xyzw", 2, "Buffer format store 4 packed 16-bit channels"),
}

# GFX1013 / RDNA 2 MTBUF opcodes: op -> (mnemonic, vdata_width, comment)
MTBUF_OPS = {
    0x00: ("tbuffer_load_format_x", 1, "Typed buffer load 1 channel"),
    0x01: ("tbuffer_load_format_xy", 2, "Typed buffer load 2 channels"),
    0x02: ("tbuffer_load_format_xyz", 3, "Typed buffer load 3 channels"),
    0x03: ("tbuffer_load_format_xyzw", 4, "Typed buffer load 4 channels"),
    0x04: ("tbuffer_store_format_x", 1, "Typed buffer store 1 channel"),
    0x05: ("tbuffer_store_format_xy", 2, "Typed buffer store 2 channels"),
    0x06: ("tbuffer_store_format_xyz", 3, "Typed buffer store 3 channels"),
    0x07: ("tbuffer_store_format_xyzw", 4, "Typed buffer store 4 channels"),
    0x08: ("tbuffer_load_format_d16_x", 1, "Typed buffer load 1 packed 16-bit channel"),
    0x09: ("tbuffer_load_format_d16_xy", 1, "Typed buffer load 2 packed 16-bit channels"),
    0x0a: ("tbuffer_load_format_d16_xyz", 2, "Typed buffer load 3 packed 16-bit channels"),
    0x0b: ("tbuffer_load_format_d16_xyzw", 2, "Typed buffer load 4 packed 16-bit channels"),
    0x0c: ("tbuffer_store_format_d16_x", 1, "Typed buffer store 1 packed 16-bit channel"),
    0x0d: ("tbuffer_store_format_d16_xy", 1, "Typed buffer store 2 packed 16-bit channels"),
    0x0e: ("tbuffer_store_format_d16_xyz", 2, "Typed buffer store 3 packed 16-bit channels"),
    0x0f: ("tbuffer_store_format_d16_xyzw", 2, "Typed buffer store 4 packed 16-bit channels"),
}

# GFX1013 MTBUF 7-bit format identifiers; format 1 (BUF_FMT_8_UNORM) is LLVM's omitted default.
MTBUF_FORMATS = {
    0: "BUF_FMT_INVALID",
    2: "BUF_FMT_8_SNORM",
    3: "BUF_FMT_8_USCALED",
    4: "BUF_FMT_8_SSCALED",
    5: "BUF_FMT_8_UINT",
    6: "BUF_FMT_8_SINT",
    7: "BUF_FMT_16_UNORM",
    8: "BUF_FMT_16_SNORM",
    9: "BUF_FMT_16_USCALED",
    10: "BUF_FMT_16_SSCALED",
    11: "BUF_FMT_16_UINT",
    12: "BUF_FMT_16_SINT",
    13: "BUF_FMT_16_FLOAT",
    14: "BUF_FMT_8_8_UNORM",
    15: "BUF_FMT_8_8_SNORM",
    16: "BUF_FMT_8_8_USCALED",
    17: "BUF_FMT_8_8_SSCALED",
    18: "BUF_FMT_8_8_UINT",
    19: "BUF_FMT_8_8_SINT",
    20: "BUF_FMT_32_UINT",
    21: "BUF_FMT_32_SINT",
    22: "BUF_FMT_32_FLOAT",
    23: "BUF_FMT_16_16_UNORM",
    24: "BUF_FMT_16_16_SNORM",
    25: "BUF_FMT_16_16_USCALED",
    26: "BUF_FMT_16_16_SSCALED",
    27: "BUF_FMT_16_16_UINT",
    28: "BUF_FMT_16_16_SINT",
    29: "BUF_FMT_16_16_FLOAT",
    30: "BUF_FMT_10_11_11_UNORM",
    31: "BUF_FMT_10_11_11_SNORM",
    32: "BUF_FMT_10_11_11_USCALED",
    33: "BUF_FMT_10_11_11_SSCALED",
    34: "BUF_FMT_10_11_11_UINT",
    35: "BUF_FMT_10_11_11_SINT",
    36: "BUF_FMT_10_11_11_FLOAT",
    37: "BUF_FMT_11_11_10_UNORM",
    38: "BUF_FMT_11_11_10_SNORM",
    39: "BUF_FMT_11_11_10_USCALED",
    40: "BUF_FMT_11_11_10_SSCALED",
    41: "BUF_FMT_11_11_10_UINT",
    42: "BUF_FMT_11_11_10_SINT",
    43: "BUF_FMT_11_11_10_FLOAT",
    44: "BUF_FMT_10_10_10_2_UNORM",
    45: "BUF_FMT_10_10_10_2_SNORM",
    46: "BUF_FMT_10_10_10_2_USCALED",
    47: "BUF_FMT_10_10_10_2_SSCALED",
    48: "BUF_FMT_10_10_10_2_UINT",
    49: "BUF_FMT_10_10_10_2_SINT",
    50: "BUF_FMT_2_10_10_10_UNORM",
    51: "BUF_FMT_2_10_10_10_SNORM",
    52: "BUF_FMT_2_10_10_10_USCALED",
    53: "BUF_FMT_2_10_10_10_SSCALED",
    54: "BUF_FMT_2_10_10_10_UINT",
    55: "BUF_FMT_2_10_10_10_SINT",
    56: "BUF_FMT_8_8_8_8_UNORM",
    57: "BUF_FMT_8_8_8_8_SNORM",
    58: "BUF_FMT_8_8_8_8_USCALED",
    59: "BUF_FMT_8_8_8_8_SSCALED",
    60: "BUF_FMT_8_8_8_8_UINT",
    61: "BUF_FMT_8_8_8_8_SINT",
    62: "BUF_FMT_32_32_UINT",
    63: "BUF_FMT_32_32_SINT",
    64: "BUF_FMT_32_32_FLOAT",
    65: "BUF_FMT_16_16_16_16_UNORM",
    66: "BUF_FMT_16_16_16_16_SNORM",
    67: "BUF_FMT_16_16_16_16_USCALED",
    68: "BUF_FMT_16_16_16_16_SSCALED",
    69: "BUF_FMT_16_16_16_16_UINT",
    70: "BUF_FMT_16_16_16_16_SINT",
    71: "BUF_FMT_16_16_16_16_FLOAT",
    72: "BUF_FMT_32_32_32_UINT",
    73: "BUF_FMT_32_32_32_SINT",
    74: "BUF_FMT_32_32_32_FLOAT",
    75: "BUF_FMT_32_32_32_32_UINT",
    76: "BUF_FMT_32_32_32_32_SINT",
    77: "BUF_FMT_32_32_32_32_FLOAT",
}


# VOP1 and VOP2 SRC0 codes that announce a second instruction dword instead of naming an operand:
# DPP8 (233, or 234 with FI set), SDWA (249) and DPP16 (250). The dword holds the real SRC0 and
# the lane or sub-dword controls.
VOP_EXTENSION_FORMS = {233: "DPP8", 234: "DPP8", 249: "SDWA", 250: "DPP16"}


def _vop_extension(data: bytes, offset: int, encoding: str, op_name: str, src0: int) -> tuple[int, str, str, str]:
    """Consume a DPP8, DPP16 or SDWA instruction as one 8-byte record; its operands are not decoded."""
    form = VOP_EXTENSION_FORMS[src0]
    if offset + 8 > len(data):
        return (4, f"{encoding}_truncated", f"{op_name} <truncated>", f"Incomplete {encoding.upper()} {form}")
    base = op_name[:-4] if op_name.endswith("_e32") else op_name
    mnemonic = f"{base}_sdwa" if form == "SDWA" else f"{base}_dpp"
    return (8, f"{encoding}_{form.lower()}", f"{mnemonic} <unsupported {form}>",
            f"{encoding.upper()} {form} is not decoded")


def _format_vgpr_range(reg: int, width: int) -> str:
    """Format a VGPR operand covering `width` consecutive 32-bit registers."""
    return f"v{reg}" if width == 1 else f"v[{reg}:{reg + width - 1}]"


def _format_srsrc4(srsrc: int) -> str:
    """Format a 4-SGPR buffer resource descriptor (s[n:n+3] or ttmp[n:n+3])."""
    if 108 <= srsrc <= 120:
        t = srsrc - 108
        return f"ttmp[{t}:{t + 3}]"
    return f"s[{srsrc}:{srsrc + 3}]"


def _format_buffer_operands_and_mods(
    w0: int,
    w1: int,
    vdata_width: int,
    *,
    format_mod: str | None = None,
    allow_lds: bool = False,
    allow_tfe: bool = True,
) -> tuple[list[str], list[str]]:
    """Decode operands and modifiers shared by MUBUF and MTBUF encodings."""
    imm_offset = w0 & 0xfff
    offen = (w0 >> 12) & 1
    idxen = (w0 >> 13) & 1
    glc = (w0 >> 14) & 1
    dlc = (w0 >> 15) & 1
    lds = ((w0 >> 16) & 1) if allow_lds else 0

    vaddr = w1 & 0xff
    vdata = (w1 >> 8) & 0xff
    srsrc = ((w1 >> 16) & 0x1f) * 4
    slc = (w1 >> 22) & 1
    tfe = (w1 >> 23) & 1
    soffset = (w1 >> 24) & 0xff

    vaddr_str = f"v[{vaddr}:{vaddr + 1}]" if (idxen and offen) else (f"v{vaddr}" if (idxen or offen) else "off")
    srsrc_str = _format_srsrc4(srsrc)
    soffset_str = decode_s_reg(soffset)

    if lds:
        operands = [vaddr_str, srsrc_str, soffset_str]
    else:
        operands = [_format_vgpr_range(vdata, vdata_width), vaddr_str, srsrc_str, soffset_str]

    mods = []
    if format_mod is not None:
        mods.append(format_mod)
    if idxen:
        mods.append("idxen")
    if offen:
        mods.append("offen")
    if imm_offset:
        mods.append(f"offset:{imm_offset}")
    if glc:
        mods.append("glc")
    if slc:
        mods.append("slc")
    if dlc:
        mods.append("dlc")
    if lds:
        mods.append("lds")
    if allow_tfe and tfe:
        mods.append("tfe")
    return operands, mods

# GFX1013 / RDNA 2 FLAT/GLOBAL/SCRATCH base opcodes:
# op -> (suffix, kind, vdst_width, vdata_width, description)
FLAT_BASE_OPS = {
    0x08: ("load_ubyte", "load", 1, 0, "load unsigned byte"),
    0x09: ("load_sbyte", "load", 1, 0, "load signed byte"),
    0x0a: ("load_ushort", "load", 1, 0, "load unsigned 16-bit"),
    0x0b: ("load_sshort", "load", 1, 0, "load signed 16-bit"),
    0x0c: ("load_dword", "load", 1, 0, "load 1 dword"),
    0x0d: ("load_dwordx2", "load", 2, 0, "load 2 dwords"),
    0x0e: ("load_dwordx4", "load", 4, 0, "load 4 dwords"),
    0x0f: ("load_dwordx3", "load", 3, 0, "load 3 dwords"),
    0x18: ("store_byte", "store", 0, 1, "store byte"),
    0x19: ("store_byte_d16_hi", "store", 0, 1, "store byte from high 16-bit lane"),
    0x1a: ("store_short", "store", 0, 1, "store 16-bit short"),
    0x1b: ("store_short_d16_hi", "store", 0, 1, "store 16-bit short from high lane"),
    0x1c: ("store_dword", "store", 0, 1, "store 1 dword"),
    0x1d: ("store_dwordx2", "store", 0, 2, "store 2 dwords"),
    0x1e: ("store_dwordx4", "store", 0, 4, "store 4 dwords"),
    0x1f: ("store_dwordx3", "store", 0, 3, "store 3 dwords"),
    0x20: ("load_ubyte_d16", "load", 1, 0, "load unsigned byte to low 16 bits"),
    0x21: ("load_ubyte_d16_hi", "load", 1, 0, "load unsigned byte to high 16 bits"),
    0x22: ("load_sbyte_d16", "load", 1, 0, "load signed byte to low 16 bits"),
    0x23: ("load_sbyte_d16_hi", "load", 1, 0, "load signed byte to high 16 bits"),
    0x24: ("load_short_d16", "load", 1, 0, "load 16-bit short to low 16 bits"),
    0x25: ("load_short_d16_hi", "load", 1, 0, "load 16-bit short to high 16 bits"),
    0x30: ("atomic_swap", "atomic", 1, 1, "atomic swap 32-bit"),
    0x31: ("atomic_cmpswap", "atomic", 1, 2, "atomic compare-and-swap 32-bit"),
    0x32: ("atomic_add", "atomic", 1, 1, "atomic add 32-bit"),
    0x33: ("atomic_sub", "atomic", 1, 1, "atomic subtract 32-bit"),
    0x35: ("atomic_smin", "atomic", 1, 1, "atomic signed min 32-bit"),
    0x36: ("atomic_umin", "atomic", 1, 1, "atomic unsigned min 32-bit"),
    0x37: ("atomic_smax", "atomic", 1, 1, "atomic signed max 32-bit"),
    0x38: ("atomic_umax", "atomic", 1, 1, "atomic unsigned max 32-bit"),
    0x39: ("atomic_and", "atomic", 1, 1, "atomic bitwise AND 32-bit"),
    0x3a: ("atomic_or", "atomic", 1, 1, "atomic bitwise OR 32-bit"),
    0x3b: ("atomic_xor", "atomic", 1, 1, "atomic bitwise XOR 32-bit"),
    0x3c: ("atomic_inc", "atomic", 1, 1, "atomic increment 32-bit"),
    0x3d: ("atomic_dec", "atomic", 1, 1, "atomic decrement 32-bit"),
    0x3e: ("atomic_fcmpswap", "atomic", 1, 2, "atomic float32 compare-and-swap"),
    0x3f: ("atomic_fmin", "atomic", 1, 1, "atomic float32 min"),
    0x40: ("atomic_fmax", "atomic", 1, 1, "atomic float32 max"),
    0x50: ("atomic_swap_x2", "atomic", 2, 2, "atomic swap 64-bit"),
    0x51: ("atomic_cmpswap_x2", "atomic", 2, 4, "atomic compare-and-swap 64-bit"),
    0x52: ("atomic_add_x2", "atomic", 2, 2, "atomic add 64-bit"),
    0x53: ("atomic_sub_x2", "atomic", 2, 2, "atomic subtract 64-bit"),
    0x55: ("atomic_smin_x2", "atomic", 2, 2, "atomic signed min 64-bit"),
    0x56: ("atomic_umin_x2", "atomic", 2, 2, "atomic unsigned min 64-bit"),
    0x57: ("atomic_smax_x2", "atomic", 2, 2, "atomic signed max 64-bit"),
    0x58: ("atomic_umax_x2", "atomic", 2, 2, "atomic unsigned max 64-bit"),
    0x59: ("atomic_and_x2", "atomic", 2, 2, "atomic bitwise AND 64-bit"),
    0x5a: ("atomic_or_x2", "atomic", 2, 2, "atomic bitwise OR 64-bit"),
    0x5b: ("atomic_xor_x2", "atomic", 2, 2, "atomic bitwise XOR 64-bit"),
    0x5c: ("atomic_inc_x2", "atomic", 2, 2, "atomic increment 64-bit"),
    0x5d: ("atomic_dec_x2", "atomic", 2, 2, "atomic decrement 64-bit"),
    0x5e: ("atomic_fcmpswap_x2", "atomic", 2, 4, "atomic float64 compare-and-swap"),
    0x5f: ("atomic_fmin_x2", "atomic", 2, 2, "atomic float64 min"),
    0x60: ("atomic_fmax_x2", "atomic", 2, 2, "atomic float64 max"),
}

FLAT_OPS = {
    op: (f"flat_{suffix}", kind, vdst_width, vdata_width, f"Flat memory {desc}")
    for op, (suffix, kind, vdst_width, vdata_width, desc) in FLAT_BASE_OPS.items()
}
GLOBAL_OPS = {
    op: (f"global_{suffix}", kind, vdst_width, vdata_width, f"Global memory {desc}")
    for op, (suffix, kind, vdst_width, vdata_width, desc) in FLAT_BASE_OPS.items()
}
SCRATCH_OPS = {
    op: (f"scratch_{suffix}", kind, vdst_width, vdata_width, f"Scratch memory {desc}")
    for op, (suffix, kind, vdst_width, vdata_width, desc) in FLAT_BASE_OPS.items()
    if kind in ("load", "store")
}
FLAT_SEGMENT_TABLES = {
    0: ("flat", FLAT_OPS),
    1: ("scratch", SCRATCH_OPS),
    2: ("global", GLOBAL_OPS),
}


def _decode_flat_family(w0: int, w1: int) -> tuple[int, str, str, str]:
    """Decode one 64-bit FLAT, SCRATCH, or GLOBAL memory instruction."""
    op = (w0 >> 18) & 0x7f
    slc = (w0 >> 17) & 1
    glc = (w0 >> 16) & 1
    seg = (w0 >> 14) & 0x3
    lds = (w0 >> 13) & 1
    dlc = (w0 >> 12) & 1
    raw_off = w0 & 0xfff

    addr = w1 & 0xff
    data_reg = (w1 >> 8) & 0xff
    saddr = (w1 >> 16) & 0x7f
    vdst = (w1 >> 24) & 0xff

    if seg not in FLAT_SEGMENT_TABLES:
        return (8, f"flat_seg{seg}_{op:02x}", f"flat_seg{seg}_{op:02x}", f"Unknown FLAT segment {seg}")

    seg_name, ops_table = FLAT_SEGMENT_TABLES[seg]
    if op not in ops_table:
        return (8, f"{seg_name}_op_{op:02x}", f"{seg_name}_op_{op:02x}", f"Unknown {seg_name.upper()} opcode")

    op_name, kind, vdst_width, vdata_width, comment = ops_table[op]
    if seg == 0:
        # FLAT uses a 12-bit unsigned offset field in GFX10 machine encoding.
        imm_off = raw_off
        vaddr_str = _format_vgpr_range(addr, 2)
        saddr_str = None
    else:
        # GLOBAL and SCRATCH use a 12-bit signed offset.
        imm_off = raw_off - 0x1000 if raw_off >= 0x800 else raw_off
        if seg == 2:
            if saddr == 0x7d:
                vaddr_str = _format_vgpr_range(addr, 2)
                saddr_str = "off"
            else:
                vaddr_str = f"v{addr}"
                saddr_str = decode_s_reg64(saddr)
        else:
            if saddr == 0x7d:
                vaddr_str = f"v{addr}"
                saddr_str = "off"
            else:
                vaddr_str = "off"
                saddr_str = decode_s_reg(saddr)

    operands = []
    if kind == "load":
        if not (lds and seg in (1, 2)):
            operands.append(_format_vgpr_range(vdst, vdst_width))
        operands.append(vaddr_str)
    elif kind == "store":
        operands.append(vaddr_str)
        operands.append(_format_vgpr_range(data_reg, vdata_width))
    else:
        if glc:
            operands.append(_format_vgpr_range(vdst, vdst_width))
        operands.append(vaddr_str)
        operands.append(_format_vgpr_range(data_reg, vdata_width))

    if saddr_str is not None:
        operands.append(saddr_str)

    mods = []
    if imm_off != 0:
        mods.append(f"offset:{imm_off}")
    if glc:
        mods.append("glc")
    if slc:
        mods.append("slc")
    if dlc:
        mods.append("dlc")
    if lds and seg in (1, 2) and kind == "load":
        mods.append("lds")

    asm = f"{op_name} {', '.join(operands)}"
    if mods:
        asm += f" {' '.join(mods)}"
    return (8, op_name, asm, comment)


# GFX1013 DS (LDS / GDS) opcode table:
# opcode -> (mnemonic, vdst_width, has_addr, data0_width, data1_width, offset_mode, gds_mode)
# where offset_mode is 'none' | 'single' | 'dual' | 'swizzle' and gds_mode is 'optional' | 'gds_only' | 'no_gds'.
DS_OPS = {
    0x00: ("ds_add_u32", 0, True, 1, 0, "single", "optional"),
    0x01: ("ds_sub_u32", 0, True, 1, 0, "single", "optional"),
    0x02: ("ds_rsub_u32", 0, True, 1, 0, "single", "optional"),
    0x03: ("ds_inc_u32", 0, True, 1, 0, "single", "optional"),
    0x04: ("ds_dec_u32", 0, True, 1, 0, "single", "optional"),
    0x05: ("ds_min_i32", 0, True, 1, 0, "single", "optional"),
    0x06: ("ds_max_i32", 0, True, 1, 0, "single", "optional"),
    0x07: ("ds_min_u32", 0, True, 1, 0, "single", "optional"),
    0x08: ("ds_max_u32", 0, True, 1, 0, "single", "optional"),
    0x09: ("ds_and_b32", 0, True, 1, 0, "single", "optional"),
    0x0a: ("ds_or_b32", 0, True, 1, 0, "single", "optional"),
    0x0b: ("ds_xor_b32", 0, True, 1, 0, "single", "optional"),
    0x0c: ("ds_mskor_b32", 0, True, 1, 1, "single", "optional"),
    0x0d: ("ds_write_b32", 0, True, 1, 0, "single", "optional"),
    0x0e: ("ds_write2_b32", 0, True, 1, 1, "dual", "optional"),
    0x0f: ("ds_write2st64_b32", 0, True, 1, 1, "dual", "optional"),
    0x10: ("ds_cmpst_b32", 0, True, 1, 1, "single", "optional"),
    0x11: ("ds_cmpst_f32", 0, True, 1, 1, "single", "optional"),
    0x12: ("ds_min_f32", 0, True, 1, 0, "single", "optional"),
    0x13: ("ds_max_f32", 0, True, 1, 0, "single", "optional"),
    0x14: ("ds_nop", 0, False, 0, 0, "none", "no_gds"),
    0x15: ("ds_add_f32", 0, True, 1, 0, "single", "optional"),
    0x18: ("ds_gws_sema_release_all", 0, False, 0, 0, "single", "gds_only"),
    0x19: ("ds_gws_init", 0, True, 0, 0, "single", "gds_only"),
    0x1a: ("ds_gws_sema_v", 0, False, 0, 0, "single", "gds_only"),
    0x1b: ("ds_gws_sema_br", 0, True, 0, 0, "single", "gds_only"),
    0x1c: ("ds_gws_sema_p", 0, False, 0, 0, "single", "gds_only"),
    0x1d: ("ds_gws_barrier", 0, True, 0, 0, "single", "gds_only"),
    0x1e: ("ds_write_b8", 0, True, 1, 0, "single", "optional"),
    0x1f: ("ds_write_b16", 0, True, 1, 0, "single", "optional"),
    0x20: ("ds_add_rtn_u32", 1, True, 1, 0, "single", "optional"),
    0x21: ("ds_sub_rtn_u32", 1, True, 1, 0, "single", "optional"),
    0x22: ("ds_rsub_rtn_u32", 1, True, 1, 0, "single", "optional"),
    0x23: ("ds_inc_rtn_u32", 1, True, 1, 0, "single", "optional"),
    0x24: ("ds_dec_rtn_u32", 1, True, 1, 0, "single", "optional"),
    0x25: ("ds_min_rtn_i32", 1, True, 1, 0, "single", "optional"),
    0x26: ("ds_max_rtn_i32", 1, True, 1, 0, "single", "optional"),
    0x27: ("ds_min_rtn_u32", 1, True, 1, 0, "single", "optional"),
    0x28: ("ds_max_rtn_u32", 1, True, 1, 0, "single", "optional"),
    0x29: ("ds_and_rtn_b32", 1, True, 1, 0, "single", "optional"),
    0x2a: ("ds_or_rtn_b32", 1, True, 1, 0, "single", "optional"),
    0x2b: ("ds_xor_rtn_b32", 1, True, 1, 0, "single", "optional"),
    0x2c: ("ds_mskor_rtn_b32", 1, True, 1, 1, "single", "optional"),
    0x2d: ("ds_wrxchg_rtn_b32", 1, True, 1, 0, "single", "optional"),
    0x2e: ("ds_wrxchg2_rtn_b32", 2, True, 1, 1, "dual", "optional"),
    0x2f: ("ds_wrxchg2st64_rtn_b32", 2, True, 1, 1, "dual", "optional"),
    0x30: ("ds_cmpst_rtn_b32", 1, True, 1, 1, "single", "optional"),
    0x31: ("ds_cmpst_rtn_f32", 1, True, 1, 1, "single", "optional"),
    0x32: ("ds_min_rtn_f32", 1, True, 1, 0, "single", "optional"),
    0x33: ("ds_max_rtn_f32", 1, True, 1, 0, "single", "optional"),
    0x34: ("ds_wrap_rtn_b32", 1, True, 1, 1, "single", "optional"),
    0x35: ("ds_swizzle_b32", 1, True, 0, 0, "swizzle", "optional"),
    0x36: ("ds_read_b32", 1, True, 0, 0, "single", "optional"),
    0x37: ("ds_read2_b32", 2, True, 0, 0, "dual", "optional"),
    0x38: ("ds_read2st64_b32", 2, True, 0, 0, "dual", "optional"),
    0x39: ("ds_read_i8", 1, True, 0, 0, "single", "optional"),
    0x3a: ("ds_read_u8", 1, True, 0, 0, "single", "optional"),
    0x3b: ("ds_read_i16", 1, True, 0, 0, "single", "optional"),
    0x3c: ("ds_read_u16", 1, True, 0, 0, "single", "optional"),
    0x3d: ("ds_consume", 1, False, 0, 0, "single", "optional"),
    0x3e: ("ds_append", 1, False, 0, 0, "single", "optional"),
    0x3f: ("ds_ordered_count", 1, True, 0, 0, "single", "gds_only"),
    0x40: ("ds_add_u64", 0, True, 2, 0, "single", "optional"),
    0x41: ("ds_sub_u64", 0, True, 2, 0, "single", "optional"),
    0x42: ("ds_rsub_u64", 0, True, 2, 0, "single", "optional"),
    0x43: ("ds_inc_u64", 0, True, 2, 0, "single", "optional"),
    0x44: ("ds_dec_u64", 0, True, 2, 0, "single", "optional"),
    0x45: ("ds_min_i64", 0, True, 2, 0, "single", "optional"),
    0x46: ("ds_max_i64", 0, True, 2, 0, "single", "optional"),
    0x47: ("ds_min_u64", 0, True, 2, 0, "single", "optional"),
    0x48: ("ds_max_u64", 0, True, 2, 0, "single", "optional"),
    0x49: ("ds_and_b64", 0, True, 2, 0, "single", "optional"),
    0x4a: ("ds_or_b64", 0, True, 2, 0, "single", "optional"),
    0x4b: ("ds_xor_b64", 0, True, 2, 0, "single", "optional"),
    0x4c: ("ds_mskor_b64", 0, True, 2, 2, "single", "optional"),
    0x4d: ("ds_write_b64", 0, True, 2, 0, "single", "optional"),
    0x4e: ("ds_write2_b64", 0, True, 2, 2, "dual", "optional"),
    0x4f: ("ds_write2st64_b64", 0, True, 2, 2, "dual", "optional"),
    0x50: ("ds_cmpst_b64", 0, True, 2, 2, "single", "optional"),
    0x51: ("ds_cmpst_f64", 0, True, 2, 2, "single", "optional"),
    0x52: ("ds_min_f64", 0, True, 2, 0, "single", "optional"),
    0x53: ("ds_max_f64", 0, True, 2, 0, "single", "optional"),
    0x55: ("ds_add_rtn_f32", 1, True, 1, 0, "single", "optional"),
    0x60: ("ds_add_rtn_u64", 2, True, 2, 0, "single", "optional"),
    0x61: ("ds_sub_rtn_u64", 2, True, 2, 0, "single", "optional"),
    0x62: ("ds_rsub_rtn_u64", 2, True, 2, 0, "single", "optional"),
    0x63: ("ds_inc_rtn_u64", 2, True, 2, 0, "single", "optional"),
    0x64: ("ds_dec_rtn_u64", 2, True, 2, 0, "single", "optional"),
    0x65: ("ds_min_rtn_i64", 2, True, 2, 0, "single", "optional"),
    0x66: ("ds_max_rtn_i64", 2, True, 2, 0, "single", "optional"),
    0x67: ("ds_min_rtn_u64", 2, True, 2, 0, "single", "optional"),
    0x68: ("ds_max_rtn_u64", 2, True, 2, 0, "single", "optional"),
    0x69: ("ds_and_rtn_b64", 2, True, 2, 0, "single", "optional"),
    0x6a: ("ds_or_rtn_b64", 2, True, 2, 0, "single", "optional"),
    0x6b: ("ds_xor_rtn_b64", 2, True, 2, 0, "single", "optional"),
    0x6c: ("ds_mskor_rtn_b64", 2, True, 2, 2, "single", "optional"),
    0x6d: ("ds_wrxchg_rtn_b64", 2, True, 2, 0, "single", "optional"),
    0x6e: ("ds_wrxchg2_rtn_b64", 4, True, 2, 2, "dual", "optional"),
    0x6f: ("ds_wrxchg2st64_rtn_b64", 4, True, 2, 2, "dual", "optional"),
    0x70: ("ds_cmpst_rtn_b64", 2, True, 2, 2, "single", "optional"),
    0x71: ("ds_cmpst_rtn_f64", 2, True, 2, 2, "single", "optional"),
    0x72: ("ds_min_rtn_f64", 2, True, 2, 0, "single", "optional"),
    0x73: ("ds_max_rtn_f64", 2, True, 2, 0, "single", "optional"),
    0x76: ("ds_read_b64", 2, True, 0, 0, "single", "optional"),
    0x77: ("ds_read2_b64", 4, True, 0, 0, "dual", "optional"),
    0x78: ("ds_read2st64_b64", 4, True, 0, 0, "dual", "optional"),
    0x7e: ("ds_condxchg32_rtn_b64", 2, True, 2, 0, "single", "optional"),
    0x80: ("ds_add_src2_u32", 0, True, 0, 0, "single", "optional"),
    0x81: ("ds_sub_src2_u32", 0, True, 0, 0, "single", "optional"),
    0x82: ("ds_rsub_src2_u32", 0, True, 0, 0, "single", "optional"),
    0x83: ("ds_inc_src2_u32", 0, True, 0, 0, "single", "optional"),
    0x84: ("ds_dec_src2_u32", 0, True, 0, 0, "single", "optional"),
    0x85: ("ds_min_src2_i32", 0, True, 0, 0, "single", "optional"),
    0x86: ("ds_max_src2_i32", 0, True, 0, 0, "single", "optional"),
    0x87: ("ds_min_src2_u32", 0, True, 0, 0, "single", "optional"),
    0x88: ("ds_max_src2_u32", 0, True, 0, 0, "single", "optional"),
    0x89: ("ds_and_src2_b32", 0, True, 0, 0, "single", "optional"),
    0x8a: ("ds_or_src2_b32", 0, True, 0, 0, "single", "optional"),
    0x8b: ("ds_xor_src2_b32", 0, True, 0, 0, "single", "optional"),
    0x8d: ("ds_write_src2_b32", 0, True, 0, 0, "single", "optional"),
    0x92: ("ds_min_src2_f32", 0, True, 0, 0, "single", "optional"),
    0x93: ("ds_max_src2_f32", 0, True, 0, 0, "single", "optional"),
    0x95: ("ds_add_src2_f32", 0, True, 0, 0, "single", "optional"),
    0xa0: ("ds_write_b8_d16_hi", 0, True, 1, 0, "single", "optional"),
    0xa1: ("ds_write_b16_d16_hi", 0, True, 1, 0, "single", "optional"),
    0xa2: ("ds_read_u8_d16", 1, True, 0, 0, "single", "optional"),
    0xa3: ("ds_read_u8_d16_hi", 1, True, 0, 0, "single", "optional"),
    0xa4: ("ds_read_i8_d16", 1, True, 0, 0, "single", "optional"),
    0xa5: ("ds_read_i8_d16_hi", 1, True, 0, 0, "single", "optional"),
    0xa6: ("ds_read_u16_d16", 1, True, 0, 0, "single", "optional"),
    0xa7: ("ds_read_u16_d16_hi", 1, True, 0, 0, "single", "optional"),
    0xb0: ("ds_write_addtid_b32", 0, False, 1, 0, "single", "optional"),
    0xb1: ("ds_read_addtid_b32", 1, False, 0, 0, "single", "optional"),
    0xb2: ("ds_permute_b32", 1, True, 1, 0, "single", "no_gds"),
    0xb3: ("ds_bpermute_b32", 1, True, 1, 0, "single", "no_gds"),
    0xc0: ("ds_add_src2_u64", 0, True, 0, 0, "single", "optional"),
    0xc1: ("ds_sub_src2_u64", 0, True, 0, 0, "single", "optional"),
    0xc2: ("ds_rsub_src2_u64", 0, True, 0, 0, "single", "optional"),
    0xc3: ("ds_inc_src2_u64", 0, True, 0, 0, "single", "optional"),
    0xc4: ("ds_dec_src2_u64", 0, True, 0, 0, "single", "optional"),
    0xc5: ("ds_min_src2_i64", 0, True, 0, 0, "single", "optional"),
    0xc6: ("ds_max_src2_i64", 0, True, 0, 0, "single", "optional"),
    0xc7: ("ds_min_src2_u64", 0, True, 0, 0, "single", "optional"),
    0xc8: ("ds_max_src2_u64", 0, True, 0, 0, "single", "optional"),
    0xc9: ("ds_and_src2_b64", 0, True, 0, 0, "single", "optional"),
    0xca: ("ds_or_src2_b64", 0, True, 0, 0, "single", "optional"),
    0xcb: ("ds_xor_src2_b64", 0, True, 0, 0, "single", "optional"),
    0xcd: ("ds_write_src2_b64", 0, True, 0, 0, "single", "optional"),
    0xd2: ("ds_min_src2_f64", 0, True, 0, 0, "single", "optional"),
    0xd3: ("ds_max_src2_f64", 0, True, 0, 0, "single", "optional"),
    0xde: ("ds_write_b96", 0, True, 3, 0, "single", "optional"),
    0xdf: ("ds_write_b128", 0, True, 4, 0, "single", "optional"),
    0xfe: ("ds_read_b96", 3, True, 0, 0, "single", "optional"),
    0xff: ("ds_read_b128", 4, True, 0, 0, "single", "optional"),
}


def _format_ds_swizzle(imm: int) -> str | None:
    """Format the 16-bit offset field of ds_swizzle_b32 like LLVM AMDGPUInstPrinter."""
    if imm == 0:
        return None
    if imm & 0x8000:
        if (imm & 0xff00) == 0x8000:
            lanes = ",".join(str((imm >> (2 * i)) & 3) for i in range(4))
            return f"offset:swizzle(QUAD_PERM,{lanes})"
        return f"offset:{imm}"
    and_mask = imm & 0x1f
    or_mask = (imm >> 5) & 0x1f
    xor_mask = (imm >> 10) & 0x1f
    if and_mask == 0x1f and or_mask == 0:
        if xor_mask in (1, 2, 4, 8, 16):
            return f"offset:swizzle(SWAP,{xor_mask})"
        if xor_mask in (3, 7, 15, 31):
            return f"offset:swizzle(REVERSE,{xor_mask + 1})"
    broadcast_groups = {0x1e: 2, 0x1c: 4, 0x18: 8, 0x10: 16, 0x00: 32}
    if xor_mask == 0 and and_mask in broadcast_groups and (or_mask & and_mask) == 0:
        return f"offset:swizzle(BROADCAST,{broadcast_groups[and_mask]},{or_mask})"
    chars = []
    for bit in range(4, -1, -1):
        ab = (and_mask >> bit) & 1
        ob = (or_mask >> bit) & 1
        xb = (xor_mask >> bit) & 1
        if ab == 0 or ob == 1:
            chars.append(str(ob ^ xb))
        else:
            chars.append("i" if xb else "p")
    return f'offset:swizzle(BITMASK_PERM,"{"".join(chars)}")'


def _decode_ds(w0: int, w1: int) -> tuple[int, str, str, str]:
    """Decode one 64-bit DS (Local / Global Data Share) instruction."""
    op = (w0 >> 18) & 0xff
    gds = (w0 >> 17) & 1
    offset0 = w0 & 0xff
    offset1 = (w0 >> 8) & 0xff
    offset16 = w0 & 0xffff

    addr = w1 & 0xff
    data0 = (w1 >> 8) & 0xff
    data1 = (w1 >> 16) & 0xff
    vdst = (w1 >> 24) & 0xff

    entry = DS_OPS.get(op)
    if entry is None:
        return (8, f"ds_op_{op:02x}", f"ds_op_{op:02x}", f"Unknown DS opcode 0x{op:02x}")

    op_name, vdst_w, has_addr, data0_w, data1_w, offset_mode, gds_mode = entry
    if (gds_mode == "gds_only" and not gds) or (gds_mode == "no_gds" and gds):
        return (8, f"ds_op_{op:02x}", f"ds_op_{op:02x}", f"Unsupported DS gds={gds} for {op_name}")
    if offset_mode == "none" and offset16 != 0:
        return (8, f"ds_op_{op:02x}", f"ds_op_{op:02x}", f"Unsupported DS offset={offset16} for {op_name}")

    operands = []
    if vdst_w:
        operands.append(_format_vgpr_range(vdst, vdst_w))
    if has_addr:
        operands.append(f"v{addr}")
    if data0_w:
        operands.append(_format_vgpr_range(data0, data0_w))
    if data1_w:
        operands.append(_format_vgpr_range(data1, data1_w))

    mods = []
    if offset_mode == "single":
        if offset16 != 0:
            mods.append(f"offset:{offset16}")
    elif offset_mode == "dual":
        if offset0 != 0:
            mods.append(f"offset0:{offset0}")
        if offset1 != 0:
            mods.append(f"offset1:{offset1}")
    elif offset_mode == "swizzle":
        swizzle = _format_ds_swizzle(offset16)
        if swizzle is not None:
            mods.append(swizzle)

    if gds:
        mods.append("gds")

    asm = op_name
    if operands:
        asm += f" {', '.join(operands)}"
    if mods:
        asm += f" {' '.join(mods)}"
    comment = "GDS data share" if gds else "LDS data share"
    return (8, op_name, asm, comment)


def decode_instruction(data: bytes, offset: int) -> tuple[int, str, str, str]:
    """
    Decode one RDNA 2 instruction at given offset.
    Returns: (instruction_byte_length, opcode_name, assembly_string, comment)
    """
    if offset + 4 > len(data):
        return (len(data) - offset, "data", f".byte {data[offset:].hex()}", "Trailing bytes")

    w0, = struct.unpack_from("<I", data, offset)

    # 1. SOPP: [31:23] == 0b101111111 (0xbf800000)
    if (w0 & 0xff800000) == 0xbf800000:
        op = (w0 >> 16) & 0x7f
        simm = w0 & 0xffff
        signed_simm = simm if simm < 0x8000 else simm - 0x10000
        op_info = SOPP_OPS.get(op, (f"s_sopp_{op:02x}", "Unknown SOPP"))
        op_name, comment = op_info

        if op_name == "s_endpgm":
            asm = "s_endpgm"
        elif op_name == "s_nop":
            asm = f"s_nop {simm}"
        elif op_name == "s_waitcnt":
            # GFX10 split: vmcnt is simm16[3:0] plus [15:14], expcnt [6:4], lgkmcnt [13:8].
            # Like LLVM, omit counters at their "no wait" maximum unless all three are.
            counters = (("vmcnt", (simm & 0xf) | ((simm >> 10) & 0x30), 0x3f),
                        ("expcnt", (simm >> 4) & 0x7, 0x7),
                        ("lgkmcnt", (simm >> 8) & 0x3f, 0x3f))
            waits = [f"{name}({value})" for name, value, limit in counters if value != limit]
            asm = "s_waitcnt " + " ".join(waits or (f"{name}({value})" for name, value, _ in counters))
        elif op_name.startswith("s_cbranch") or op_name == "s_branch":
            target = offset + 4 + signed_simm * 4
            asm = f"{op_name} label_0x{target:x}"
        else:
            asm = f"{op_name} 0x{simm:x}"
        return (4, op_name, asm, comment)

    # 2. SOP1: [31:23] == 0b101111101 (0xbe800000)
    if (w0 & 0xff800000) == 0xbe800000:
        sdst = (w0 >> 16) & 0x7f
        op = (w0 >> 8) & 0xff
        ssrc0 = w0 & 0xff
        op_info = SOP1_OPS.get(op, (f"s_sop1_{op:02x}", "Unknown SOP1"))
        op_name, comment = op_info
        length = 4
        is_64 = op_name.endswith(("_b64", "_u64", "_i64"))
        if ssrc0 == 255:
            if offset + 8 > len(data):
                return (4, "sop1_truncated", f"{op_name} <truncated>", "Incomplete SOP1 literal")
            literal, = struct.unpack_from("<I", data, offset + 4)
            src0_str = f"0x{literal:x}"
            length = 8
        else:
            src0_str = decode_s_reg64(ssrc0) if is_64 else decode_s_reg(ssrc0)
        sdst_str = decode_s_reg64(sdst) if is_64 else decode_s_reg(sdst)
        asm = f"{op_name} {sdst_str}, {src0_str}"
        return (length, op_name, asm, comment)

    # SOPC: [31:23] == 0b101111110 (0xbf000000)
    # Must run before the SOPK prefix check, which would otherwise consume these words.
    if (w0 & 0xff800000) == 0xbf000000:
        op = (w0 >> 16) & 0x7f
        ssrc1 = (w0 >> 8) & 0xff
        ssrc0 = w0 & 0xff
        op_info = SOPC_OPS.get(op, (f"s_sopc_{op:02x}", "Unknown SOPC"))
        op_name, comment = op_info
        length = 4
        literal = None
        if ssrc0 == 255 or ssrc1 == 255:
            if offset + 8 > len(data):
                return (4, "sopc_truncated", f"{op_name} <truncated>", "Incomplete SOPC literal")
            literal, = struct.unpack_from("<I", data, offset + 4)
            length = 8

        def sopc_operand(reg_id: int, wide: bool) -> str:
            if reg_id == 255 and literal is not None:
                return f"0x{literal:x}"
            if wide:
                return decode_s_reg64(reg_id)
            return decode_s_reg(reg_id)

        src0_wide = op_name in {"s_cmp_eq_u64", "s_cmp_lg_u64", "s_bitcmp0_b64", "s_bitcmp1_b64"}
        src1_wide = op_name in {"s_cmp_eq_u64", "s_cmp_lg_u64"}
        asm = f"{op_name} {sopc_operand(ssrc0, src0_wide)}, {sopc_operand(ssrc1, src1_wide)}"
        return (length, op_name, asm, comment)

    # 3. SOPK: [31:28] == 0b1011 (0xb0000000)
    if (w0 & 0xf0000000) == 0xb0000000 and (w0 & 0x0f800000) not in (0x0f800000, 0x0e800000, 0x0f000000):
        op = (w0 >> 23) & 0x1f
        sdst = (w0 >> 16) & 0x7f
        simm = w0 & 0xffff
        signed_simm = simm if simm < 0x8000 else simm - 0x10000
        op_info = SOPK_OPS.get(op, (f"s_sopk_{op:02x}", "Unknown SOPK"))
        op_name, comment = op_info
        if op_name == "s_setreg_imm32_b32":
            # The only SOPK form followed by a literal dword; SDST is unused.
            if offset + 8 > len(data):
                return (4, "sopk_truncated", f"{op_name} <truncated>", "Incomplete SOPK literal")
            literal, = struct.unpack_from("<I", data, offset + 4)
            return (8, op_name, f"{op_name} 0x{simm:x}, 0x{literal:x}", comment)
        if op_name == "s_version":
            # No destination operand; the SDST field is not printed.
            return (4, op_name, f"{op_name} 0x{simm:x}", comment)
        asm = f"{op_name} {decode_s_reg(sdst)}, 0x{simm:x}"
        return (4, op_name, asm, comment)

    # 4. SOP2: [31:30] == 0b10 (0x80000000)
    if (w0 & 0xc0000000) == 0x80000000 and (w0 & 0x30000000) != 0x30000000:
        op = (w0 >> 23) & 0x7f
        sdst = (w0 >> 16) & 0x7f
        ssrc1 = (w0 >> 8) & 0xff
        ssrc0 = w0 & 0xff
        op_info = SOP2_OPS.get(op, (f"s_sop2_{op:02x}", "Unknown SOP2"))
        op_name, comment = op_info
        length = 4
        literal = None
        if ssrc0 == 255 or ssrc1 == 255:
            if offset + 8 > len(data):
                return (4, "sop2_truncated", f"{op_name} <truncated>", "Incomplete SOP2 literal")
            literal, = struct.unpack_from("<I", data, offset + 4)
            length = 8
        is_64 = op_name.endswith(("_b64", "_u64", "_i64"))
        src0_wide = is_64 and not op_name.startswith("s_bfm_")
        src0_str = f"0x{literal:x}" if ssrc0 == 255 and literal is not None else (decode_s_reg64(ssrc0) if src0_wide else decode_s_reg(ssrc0))
        src1_wide = is_64 and not op_name.startswith(("s_lshl_", "s_lshr_", "s_ashr_", "s_bfe_", "s_bfm_"))
        src1_str = f"0x{literal:x}" if ssrc1 == 255 and literal is not None else (decode_s_reg64(ssrc1) if src1_wide else decode_s_reg(ssrc1))
        sdst_str = decode_s_reg64(sdst) if is_64 else decode_s_reg(sdst)
        asm = f"{op_name} {sdst_str}, {src0_str}, {src1_str}"
        return (length, op_name, asm, comment)

    # 5. VOP1: [31:25] == 0b0111111 (0x7e000000)
    if (w0 & 0xfe000000) == 0x7e000000:
        vdst = (w0 >> 17) & 0xff
        op = (w0 >> 9) & 0xff
        src0 = w0 & 0x1ff
        op_info = VOP1_OPS.get(op, (f"v_vop1_{op:02x}", "Unknown VOP1"))
        op_name, comment = op_info
        if src0 in VOP_EXTENSION_FORMS:
            return _vop_extension(data, offset, "vop1", op_name, src0)
        length = 4
        wide_dst = op_name.startswith(("v_cvt_f64_i32", "v_cvt_f64_u32", "v_cvt_f64_f32")) or op_name.endswith(("_f64_e32", "_f64"))
        wide_src0 = op_name.startswith(("v_cvt_i32_f64", "v_cvt_u32_f64", "v_cvt_f32_f64", "v_frexp_exp_i32_f64")) or op_name.endswith(("_f64_e32", "_f64"))
        if src0 == 255:
            if offset + 8 > len(data):
                return (4, "vop1_truncated", f"{op_name} <truncated>", "Incomplete VOP1 literal")
            literal, = struct.unpack_from("<I", data, offset + 4)
            src0_str = f"0x{literal:x}"
            length = 8
        elif wide_src0:
            src0_str = decode_v_reg64(src0)
        else:
            src0_str = decode_v_reg(src0)
        dst_str = f"v[{vdst}:{vdst + 1}]" if wide_dst and vdst < 255 else (f"<invalid VGPR pair>" if wide_dst else f"v{vdst}")
        if op_name == "v_readfirstlane_b32":
            # This VOP1 instruction writes an SGPR: its vdst field is an SGPR
            # number even though the mnemonic starts with "v_". LLVM 21 for
            # gfx1013 prints s2 for the encoding 0005047e.
            dst_str = decode_s_reg(vdst)
        if op_name == "v_nop":
            asm = "v_nop"
        else:
            asm = f"{op_name} {dst_str}, {src0_str}"
        return (length, op_name, asm, comment)

    # 6. VOPC: [31:25] == 0b0111110 (0x7c000000)
    if (w0 & 0xfe000000) == 0x7c000000:
        op = (w0 >> 17) & 0xff
        vsrc1 = (w0 >> 9) & 0xff
        src0 = w0 & 0x1ff
        op_info = VOPC_OPS.get(op, (f"v_vopc_{op:02x}", "Unknown VOPC"))
        op_name, comment = op_info
        length = 4
        literal = None
        if src0 in (249, 255):
            extension = "SDWA" if src0 == 249 else "literal"
            if offset + 8 > len(data):
                return (4, "vopc_truncated", f"{op_name} <truncated>", f"Incomplete VOPC {extension}")
            if src0 == 249:
                return (8, "vopc_sdwa", f"{op_name} <unsupported SDWA>", "VOPC SDWA is not decoded")
            literal, = struct.unpack_from("<I", data, offset + 4)
            length = 8
        wide0 = op_name.endswith(("_f64_e32", "_i64_e32", "_u64_e32"))
        wide1 = wide0 and "_class_" not in op_name
        if literal is not None:
            src0_str = f"0x{literal:x}"
        elif wide0:
            src0_str = decode_v_reg64(src0)
        else:
            src0_str = decode_v_reg(src0)
        if wide1:
            src1_str = f"v[{vsrc1}:{vsrc1 + 1}]" if vsrc1 < 255 else "<invalid VGPR pair>"
        else:
            src1_str = f"v{vsrc1}"
        if op_name.startswith("v_cmpx_"):
            asm = f"{op_name} {src0_str}, {src1_str}"
        else:
            asm = f"{op_name} vcc_lo, {src0_str}, {src1_str}"
        return (length, op_name, asm, comment)

    # 7. VOP2: [31] == 0
    if (w0 & 0x80000000) == 0:
        op = (w0 >> 25) & 0x3f
        vdst = (w0 >> 17) & 0xff
        vsrc1 = (w0 >> 9) & 0xff
        src0 = w0 & 0x1ff
        op_info = VOP2_OPS.get(op, (f"v_vop2_{op:02x}", "Unknown VOP2"))
        op_name, comment = op_info
        if src0 in VOP_EXTENSION_FORMS:
            return _vop_extension(data, offset, "vop2", op_name, src0)
        has_literal = src0 == 255 or op in (0x20, 0x21, 0x2c, 0x2d, 0x37, 0x38)
        length = 4
        literal = None
        if has_literal:
            if offset + 8 > len(data):
                return (4, "vop2_truncated", f"{op_name} <truncated>", "Incomplete VOP2 literal")
            literal, = struct.unpack_from("<I", data, offset + 4)
            length = 8
        src0_str = f"0x{literal:x}" if src0 == 255 and literal is not None else decode_v_reg(src0)
        if op_name == "v_cndmask_b32_e32":
            asm = f"{op_name} v{vdst}, {src0_str}, v{vsrc1}, vcc_lo"
        elif op in (0x20, 0x2c, 0x37):
            asm = f"{op_name} v{vdst}, {src0_str}, 0x{literal:x}, v{vsrc1}"
        elif op in (0x21, 0x2d, 0x38):
            asm = f"{op_name} v{vdst}, {src0_str}, v{vsrc1}, 0x{literal:x}"
        else:
            asm = f"{op_name} v{vdst}, {src0_str}, v{vsrc1}"
        return (length, op_name, asm, comment)

    # 8. VINTRP: [31:26] == 0b110010 (0xc8000000)
    if (w0 & 0xfc000000) == 0xc8000000:
        vdst = (w0 >> 18) & 0xff
        op = (w0 >> 16) & 0x3
        attr = (w0 >> 10) & 0x3f
        chan = "xyzw"[(w0 >> 8) & 0x3]
        vsrc = w0 & 0xff
        op_names = {0: "v_interp_p1_f32_e32", 1: "v_interp_p2_f32_e32", 2: "v_interp_mov_f32_e32"}
        op_name = op_names.get(op, f"v_interp_{op}")
        asm = f"{op_name} v{vdst}, v{vsrc}, attr{attr}.{chan}"
        comment = f"Parameter interpolation stage {op+1} for attribute {attr}.{chan}"
        return (4, op_name, asm, comment)

    # 9. EXP: 64-bit [31:26] == 0b111110 (0xf8000000)
    if (w0 & 0xfc000000) == 0xf8000000:
        if offset + 8 > len(data):
            return (4, "exp_truncated", "exp <truncated>", "Incomplete 64-bit export instruction")
        w1, = struct.unpack_from("<I", data, offset + 4)
        en = w0 & 0xf
        target = (w0 >> 4) & 0x3f
        compr = (w0 >> 10) & 1
        done = (w0 >> 11) & 1
        vm = (w0 >> 12) & 1
        vs = [(w1 >> (8 * k)) & 0xff for k in range(4)]
        if compr:
            lo = f"v{vs[0]}" if en & 0x1 else "off"
            hi = f"v{vs[1]}" if en & 0x4 else "off"
            regs = [lo, lo, hi, hi]
        else:
            regs = [f"v{vs[k]}" if (en & (1 << k)) else "off" for k in range(4)]
        target_name = EXP_TARGETS.get(target, f"target_{target}")
        flags = []
        if done:
            flags.append("done")
        if compr:
            flags.append("compr")
        if vm:
            flags.append("vm")
        # The printed compressed operands use the first bit of each pair.
        # Keep a non-paired raw mask visible so it cannot masquerade as a supported export.
        if compr and en not in (0x0, 0x3, 0xc, 0xf):
            flags.append(f"en:{en:#x}")
        flags_str = " ".join(flags)
        asm = f"exp {target_name} {', '.join(regs)} {flags_str}".strip()
        comment = f"Export output pixels/attributes to {target_name.upper()}"
        return (8, "exp", asm, comment)

    # 10. SMEM: 64-bit [31:26] == 0b111101 (0xf4000000)
    if (w0 & 0xfc000000) == 0xf4000000:
        if offset + 8 > len(data):
            return (4, "smem_truncated", "smem <truncated>", "Incomplete 64-bit SMEM instruction")
        w1, = struct.unpack_from("<I", data, offset + 4)
        op = (w0 >> 18) & 0xff
        sdata = (w0 >> 6) & 0x7f
        sbase = (w0 & 0x3f) * 2
        op_name = SMEM_OPS.get(op, f"s_smem_{op:02x}")
        data_width, base_width = SMEM_WIDTHS.get(op, (4, 2))
        sdata_str = decode_s_reg(sdata) if data_width == 1 else f"s[{sdata}:{sdata + data_width - 1}]"
        if base_width == 2 and sbase == 106:
            sbase_str = "vcc"
        else:
            sbase_str = f"s[{sbase}:{sbase + base_width - 1}]"
        soffset = (w1 >> 25) & 0x7f
        imm_offset = w1 & 0x1fffff
        if soffset == 125:
            off_str = f"0x{imm_offset:x}"
        else:
            off_str = decode_s_reg(soffset)
            if imm_offset:
                off_str += f" offset:0x{imm_offset:x}"
        mods = []
        if (w0 >> 16) & 1:
            mods.append("glc")
        if (w0 >> 14) & 1:
            mods.append("dlc")
        mod_str = (" " + " ".join(mods)) if mods else ""
        asm = f"{op_name} {sdata_str}, {sbase_str}, {off_str}{mod_str}"
        comment = "Scalar memory load from constant buffer/resource descriptor"
        return (8, op_name, asm, comment)

    # 11. MIMG: 64-bit + optional NSA dwords [31:26] == 0b111100 (0xf0000000)
    if (w0 & 0xfc000000) == 0xf0000000:
        nsize = (w0 >> 1) & 0x3
        length = 8 + 4 * nsize
        if offset + length > len(data):
            return (len(data) - offset, "mimg_truncated", "mimg <truncated>", "Incomplete MIMG instruction")
        w1, = struct.unpack_from("<I", data, offset + 4)
        op = (w0 >> 18) & 0x7f
        op_name, has_sampler, extra_addrs = MIMG_OPS.get(op, (f"image_mimg_{op:02x}", False, 0))
        dmask = (w0 >> 8) & 0xf
        dim_code = (w0 >> 3) & 0x7
        dim_name, base_addrs, deriv_addrs = MIMG_DIMS[dim_code]
        addr_count = base_addrs + extra_addrs + (deriv_addrs if op == 0x22 else 0)
        d16 = (w1 >> 31) & 1
        n_data = max(1, bin(dmask).count("1"))
        if d16:
            n_data = (n_data + 1) // 2
        vaddr0 = w1 & 0xff
        vdata = (w1 >> 8) & 0xff
        srsrc = ((w1 >> 16) & 0x1f) * 4
        ssamp = ((w1 >> 21) & 0x1f) * 4
        r128 = (w0 >> 15) & 1
        vdata_str = f"v{vdata}" if n_data == 1 else f"v[{vdata}:{vdata + n_data - 1}]"
        if nsize > 0:
            extra_bytes = list(data[offset + 8:offset + length])
            addrs = [vaddr0] + extra_bytes[:max(0, addr_count - 1)]
            vaddr_str = "[" + ", ".join(f"v{r}" for r in addrs) + "]"
        else:
            vaddr_str = f"v{vaddr0}" if addr_count == 1 else f"v[{vaddr0}:{vaddr0 + addr_count - 1}]"
        srsrc_width = 4 if r128 else 8
        srsrc_str = f"s[{srsrc}:{srsrc + srsrc_width - 1}]"
        operands = [vdata_str, vaddr_str, srsrc_str]
        if has_sampler:
            operands.append(f"s[{ssamp}:{ssamp + 3}]")
        mods = [f"dmask:0x{dmask:x}", f"dim:{dim_name}"]
        if (w0 >> 12) & 1:
            mods.append("unorm")
        if (w0 >> 13) & 1:
            mods.append("glc")
        if (w0 >> 25) & 1:
            mods.append("slc")
        if r128:
            mods.append("r128")
        if (w0 >> 14) & 1:
            mods.append("lwe")
        if (w0 >> 7) & 1:
            mods.append("dlc")
        if d16:
            mods.append("d16")
        asm = f"{op_name} {', '.join(operands)} {' '.join(mods)}"
        comment = "Image memory load/store/sample operation"
        return (length, op_name, asm, comment)

    # 12. VOP3P: 64-bit [31:23] == 0b110011000 (0xcc000000)
    if (w0 & 0xff800000) == 0xcc000000:
        if offset + 8 > len(data):
            return (4, "vop3p_truncated", "vop3p <truncated>", "Incomplete 64-bit VOP3P instruction")
        w1, = struct.unpack_from("<I", data, offset + 4)
        op = (w0 >> 16) & 0x7f
        clamp = (w0 >> 15) & 1
        vdst = w0 & 0xff
        src0 = w1 & 0x1ff
        src1 = (w1 >> 9) & 0x1ff
        src2 = (w1 >> 18) & 0x1ff

        op_info = VOP3P_OPS.get(op, (f"v_vop3p_{op:02x}", "Unknown VOP3P"))
        op_name, comment = op_info
        is_3src = op in VOP3P_3SRC
        arity = 3 if is_3src else 2

        length = 8
        literal = None
        if src0 == 255 or src1 == 255 or (is_3src and src2 == 255):
            if offset + 12 > len(data):
                return (8, "vop3p_truncated", f"{op_name} <truncated>", "Incomplete VOP3P literal")
            literal, = struct.unpack_from("<I", data, offset + 8)
            length = 12

        # Check for unmodeled / reserved bits in 2-source VOP3P encodings.
        # Note: bit 14 of w0 is op_sel_hi[2], which LLVM's MC assembler defaults to 1 (op_sel_hi = 0b111)
        # for all packed VOP3P instructions regardless of arity, so bit 14 is ignored on 2-source forms.
        if not is_3src and (src2 != 0 or ((w0 >> 10) & 1) or ((w0 >> 13) & 1) or ((w1 >> 31) & 1)):
            return (length, f"{op_name}_unsupported", f"{op_name} <unsupported_reserved_bits>", "Reserved bitfields set in 2-source VOP3P encoding")

        is_fma_mix = op_name.startswith("v_fma_mix")

        def vop3p_operand(reg_id: int, is_abs: int = 0, is_neg: int = 0) -> str:
            if reg_id == 255 and literal is not None:
                val = f"0x{literal:x}"
            else:
                val = decode_v_reg(reg_id)
            return _format_operand_modifiers(val, is_abs, is_neg)

        op_sel = [(w0 >> 11) & 1, (w0 >> 12) & 1, (w0 >> 13) & 1][:arity]
        op_sel_hi = [(w1 >> 27) & 1, (w1 >> 28) & 1, (w0 >> 14) & 1][:arity]
        neg_lo = [(w1 >> 29) & 1, (w1 >> 30) & 1, (w1 >> 31) & 1][:arity]
        neg_hi = [(w0 >> 8) & 1, (w0 >> 9) & 1, (w0 >> 10) & 1][:arity]

        if is_fma_mix:
            # Mixed-precision FMA repurposes VOP3P w0[10:8] as abs and w1[31:29] as neg.
            abs_bits = neg_hi
            neg_bits = neg_lo
            neg_lo = []
            neg_hi = []
        else:
            abs_bits = [0] * arity
            neg_bits = [0] * arity

        raw_srcs = [src0, src1, src2][:arity]
        src_strs = [vop3p_operand(raw_srcs[i], abs_bits[i], neg_bits[i]) for i in range(arity)]
        asm = f"{op_name} v{vdst}, {', '.join(src_strs)}"

        # Architectural defaults:
        # Standard packed instructions (v_pk_*, v_dot*): default op_sel is all 0, op_sel_hi is all 1, neg_lo/neg_hi all 0.
        # Mixed-precision FMA (v_fma_mix*): default op_sel and op_sel_hi are all 0 (f32, low half).
        default_op_sel = [0] * arity
        default_op_sel_hi = [0 if is_fma_mix else 1] * arity
        default_neg_lo = [0] * len(neg_lo)
        default_neg_hi = [0] * len(neg_hi)

        if op_sel != default_op_sel:
            asm += f" op_sel:[{','.join(str(b) for b in op_sel)}]"
        if op_sel_hi != default_op_sel_hi:
            asm += f" op_sel_hi:[{','.join(str(b) for b in op_sel_hi)}]"
        if neg_lo != default_neg_lo:
            asm += f" neg_lo:[{','.join(str(b) for b in neg_lo)}]"
        if neg_hi != default_neg_hi:
            asm += f" neg_hi:[{','.join(str(b) for b in neg_hi)}]"

        if clamp:
            asm += " clamp"

        return (length, op_name, asm, comment)

    # 12. MUBUF: 64-bit [31:26] == 0b111000 (0xe0000000)
    if (w0 & 0xfc000000) == 0xe0000000:
        if offset + 8 > len(data):
            return (4, "mubuf_truncated", "mubuf <truncated>", "Incomplete 64-bit MUBUF instruction")
        w1, = struct.unpack_from("<I", data, offset + 4)
        op = (w0 >> 18) & 0xff
        op_name, vdata_width, comment = MUBUF_OPS.get(
            op, (f"buffer_mubuf_{op:02x}", 1, "Unknown MUBUF")
        )
        if vdata_width == 0:
            return (8, op_name, op_name, comment)
        allow_lds = op_name.startswith("buffer_load_") and "_d16" not in op_name
        allow_tfe = not op_name.startswith("buffer_atomic_")
        operands, mods = _format_buffer_operands_and_mods(
            w0, w1, vdata_width, allow_lds=allow_lds, allow_tfe=allow_tfe
        )
        asm = f"{op_name} {', '.join(operands)}"
        if mods:
            asm += f" {' '.join(mods)}"
        return (8, op_name, asm, comment)

    # 13. MTBUF: 64-bit [31:26] == 0b111010 (0xe8000000)
    if (w0 & 0xfc000000) == 0xe8000000:
        if offset + 8 > len(data):
            return (4, "mtbuf_truncated", "mtbuf <truncated>", "Incomplete 64-bit MTBUF instruction")
        w1, = struct.unpack_from("<I", data, offset + 4)
        fmt = (w0 >> 19) & 0x7f
        op = ((w0 >> 16) & 0x7) | (((w1 >> 21) & 1) << 3)
        op_name, vdata_width, comment = MTBUF_OPS.get(
            op, (f"tbuffer_mtbuf_{op:02x}", 1, "Unknown MTBUF")
        )
        if fmt == 1:
            format_mod = None
        elif fmt in MTBUF_FORMATS:
            format_mod = f"format:[{MTBUF_FORMATS[fmt]}]"
        else:
            format_mod = f"format:{fmt}"
        operands, mods = _format_buffer_operands_and_mods(
            w0, w1, vdata_width, format_mod=format_mod
        )
        asm = f"{op_name} {', '.join(operands)}"
        if mods:
            asm += f" {' '.join(mods)}"
        return (8, op_name, asm, comment)

    # 14. FLAT / SCRATCH / GLOBAL: 64-bit [31:26] == 0b110111 (0xdc000000)
    if (w0 & 0xfc000000) == 0xdc000000:
        if offset + 8 > len(data):
            return (4, "flat_truncated", "flat <truncated>", "Incomplete 64-bit FLAT/GLOBAL/SCRATCH instruction")
        w1, = struct.unpack_from("<I", data, offset + 4)
        return _decode_flat_family(w0, w1)

    # DS: 64-bit [31:26] == 0b110110 (0xd8000000)
    if (w0 & 0xfc000000) == 0xd8000000:
        if offset + 8 > len(data):
            return (4, "ds_truncated", "ds <truncated>", "Incomplete 64-bit DS instruction")
        w1, = struct.unpack_from("<I", data, offset + 4)
        return _decode_ds(w0, w1)

    # 15. VOP3A / VOP3B: 64-bit [31:26] == 0b110101 (0xd4000000)
    if (w0 & 0xfc000000) == 0xd4000000:
        if offset + 8 > len(data):
            return (len(data) - offset, "vop3_truncated", "vop3 <truncated>", "Incomplete 64-bit VOP3 instruction")
        w1, = struct.unpack_from("<I", data, offset + 4)
        op = (w0 >> 16) & 0x3ff
        clamp = (w0 >> 15) & 1
        vdst = w0 & 0xff
        src0 = w1 & 0x1ff
        src1 = (w1 >> 9) & 0x1ff
        src2 = (w1 >> 18) & 0x1ff
        omod = (w1 >> 27) & 0x3
        neg = [(w1 >> 29) & 1, (w1 >> 30) & 1, (w1 >> 31) & 1]

        is_vopc = False
        is_vop3b = False
        if op < 0x100:
            is_vopc = True
            base_info = VOPC_OPS.get(op, (f"v_vopc_{op:02x}", "Unknown VOPC compare"))
            op_name = base_info[0].replace("_e32", "_e64")
            comment = f"{base_info[1]} (VOP3A)"
            arity = 2
            wide0 = op_name.endswith(("_f64_e64", "_i64_e64", "_u64_e64"))
            wide1 = wide0 and "_class_" not in op_name
            dst_width = 1
            src_widths = (2 if wide0 else 1, 2 if wide1 else 1)
        elif 0x100 <= op < 0x140:
            vop2_op = op - 0x100
            dst_width = 1
            if op in (0x128, 0x129, 0x12a):
                is_vop3b = True
                carry_ops = {
                    0x128: ("v_add_co_ci_u32_e64", "Vector add with carry-in and carry-out uint32"),
                    0x129: ("v_sub_co_ci_u32_e64", "Vector sub with carry-in and carry-out uint32"),
                    0x12a: ("v_subrev_co_ci_u32_e64", "Vector reverse sub with carry-in and carry-out uint32"),
                }
                op_name, comment = carry_ops[op]
                comment += " (VOP3B)"
                arity = 3
                src_widths = (1, 1, 1)
            elif vop2_op == 0x01:
                op_name = "v_cndmask_b32_e64"
                comment = "Vector conditional mask select (VOP3A)"
                arity = 3
                src_widths = (1, 1, 1)
            else:
                base_info = VOP2_OPS.get(vop2_op, (f"v_vop2_{vop2_op:02x}", "Unknown VOP2"))
                op_name = base_info[0].replace("_e32", "_e64")
                comment = f"{base_info[1]} (VOP3A)"
                arity = 2
                src_widths = (1, 1)
        elif 0x180 <= op < 0x200:
            vop1_op = op - 0x180
            base_info = VOP1_OPS.get(vop1_op, (f"v_vop1_{vop1_op:02x}", "Unknown VOP1"))
            op_name = base_info[0].replace("_e32", "_e64")
            comment = f"{base_info[1]} (VOP3A)"
            if op_name == "v_nop":
                arity = 0
                dst_width = 0
                src_widths = ()
            else:
                arity = 1
                if op_name.startswith(("v_cvt_f64_i32", "v_cvt_f64_u32", "v_cvt_f64_f32")):
                    dst_width = 2
                    src_widths = (1,)
                elif op_name.startswith(("v_cvt_i32_f64", "v_cvt_u32_f64", "v_cvt_f32_f64", "v_frexp_exp_i32_f64")):
                    dst_width = 1
                    src_widths = (2,)
                elif op_name.endswith(("_f64_e64", "_f64")):
                    dst_width = 2
                    src_widths = (2,)
                else:
                    dst_width = 1
                    src_widths = (1,)
        elif 0x270 <= op < 0x274:
            vintrp_names = {0: "v_interp_p1_f32_e64", 1: "v_interp_p2_f32_e64", 2: "v_interp_mov_f32_e64"}
            vop_int = op - 0x270
            op_name = vintrp_names.get(vop_int, f"v_interp_{vop_int}_e64")
            comment = f"Parameter interpolation stage {vop_int + 1} (VOP3A)"
            arity = 2
            dst_width = 1
            src_widths = (1, 1)
        elif op in VOP3_NATIVE_OPS:
            op_name, comment, arity, is_vop3b, dst_width, src_widths = VOP3_NATIVE_OPS[op]
        else:
            op_name = f"v_vop3_{op:03x}"
            comment = f"Unknown VOP3 opcode 0x{op:03x}"
            arity = 3
            dst_width = 1
            src_widths = (1, 1, 1)

        if op_name == "v_readfirstlane_b32":
            # gfx1013 assembles this operation only as the 32-bit VOP1: llvm-mc
            # rejects the e64 spelling and reports this VOP3A word as an invalid
            # instruction encoding.  Report a distinct unsupported-encoding
            # record instead of a lowerable instruction, and keep the eight-byte
            # record boundary so the next instruction starts at +8.
            return (8, "v_readfirstlane_b32_unsupported",
                    "v_readfirstlane_b32 <unsupported VOP3A encoding>",
                    "VOP3A encoding of v_readfirstlane_b32 is not available on gfx1013")

        if op_name == "v_nop":
            return (8, "v_nop", "v_nop", "Vector no-op (VOP3A)")

        has_literal = (arity >= 1 and src0 == 255) or (arity >= 2 and src1 == 255) or (arity >= 3 and src2 == 255)
        length = 8
        literal = None
        if has_literal:
            if offset + 12 > len(data):
                return (8, "vop3_truncated", f"{op_name} <truncated>", "Incomplete VOP3 literal")
            literal, = struct.unpack_from("<I", data, offset + 8)
            length = 12

        if is_vop3b:
            sdst = (w0 >> 8) & 0x7f
            abs_bits = [0, 0, 0]
            op_sel = [0, 0, 0, 0]
        else:
            sdst = None
            abs_bits = [(w0 >> 8) & 1, (w0 >> 9) & 1, (w0 >> 10) & 1]
            op_sel = [(w0 >> 11) & 1, (w0 >> 12) & 1, (w0 >> 13) & 1, (w0 >> 14) & 1]

        def vop3_operand(reg_id: int, is_abs: int, is_neg: int, width: int = 1) -> str:
            if reg_id == 255 and literal is not None:
                val = f"0x{literal:x}"
            elif width == 2:
                val = decode_v_reg64(reg_id)
            else:
                val = decode_v_reg(reg_id)
            return _format_operand_modifiers(val, is_abs, is_neg)

        raw_srcs = [src0, src1, src2][:arity]
        src_strs = [vop3_operand(raw_srcs[i], abs_bits[i], neg[i], src_widths[i]) for i in range(arity)]

        if is_vopc:
            dst_str = decode_s_reg(vdst)
            asm = f"{op_name} {dst_str}, {', '.join(src_strs)}"
        elif is_vop3b:
            sdst_str = decode_s_reg(sdst)
            dst_str = f"v[{vdst}:{vdst + 1}]" if dst_width == 2 and vdst < 255 else (f"<invalid VGPR pair>" if dst_width == 2 else f"v{vdst}")
            asm = f"{op_name} {dst_str}, {sdst_str}, {', '.join(src_strs)}"
        else:
            dst_str = f"v[{vdst}:{vdst + 1}]" if dst_width == 2 and vdst < 255 else (f"<invalid VGPR pair>" if dst_width == 2 else f"v{vdst}")
            asm = f"{op_name} {dst_str}, {', '.join(src_strs)}"

        if any(op_sel):
            if op_sel[3]:
                asm += f" op_sel:[{','.join(str(b) for b in op_sel[:4])}]"
            else:
                asm += f" op_sel:[{','.join(str(b) for b in op_sel[:3])}]"

        if clamp:
            asm += " clamp"

        if omod == 1:
            asm += " mul:2"
        elif omod == 2:
            asm += " mul:4"
        elif omod == 3:
            asm += " div:2"

        return (length, op_name, asm, comment)

    # Default fallback: unknown 32-bit word
    return (4, f"word_0x{w0:08x}", f".long 0x{w0:08x}", "Unrecognized RDNA 2 instruction encoding")


def disassemble_binary(data: bytes, shader_name: str = "shader") -> list[dict]:
    """Disassemble raw binary RDNA 2 shader into structured instruction records."""
    instructions = []
    offset = 0
    while offset < len(data):
        length, op_name, asm, comment = decode_instruction(data, offset)
        hex_bytes = data[offset:offset + length].hex()
        instructions.append({
            "offset": offset,
            "length": length,
            "hex": hex_bytes,
            "opcode": op_name,
            "assembly": asm,
            "comment": comment,
        })
        offset += length
        if op_name == "s_endpgm":
            # Some shaders have 4 trailing zero padding bytes after endpgm
            break
    return instructions


# Alias for library usage
disasm_rdna2 = disassemble_binary


def print_disassembly(shader_name: str, instructions: list[dict], show_comments: bool = True):
    """Print beautifully formatted disassembly with offsets and annotations."""
    print("=" * 80)
    print(f" RDNA 2 DISASSEMBLY: {shader_name} ")
    print("=" * 80)
    print(f"{'Offset':<8} {'Raw Hex':<18} {'Assembly':<42} {'Notes'}")
    print("-" * 80)

    for item in instructions:
        off_str = f"+0x{item['offset']:04x}"
        hex_str = item["hex"]
        asm_str = item["assembly"]
        comment_str = f"// {item['comment']}" if (show_comments and item["comment"]) else ""
        print(f"{off_str:<8} {hex_str:<18} {asm_str:<42} {comment_str}")

    print("=" * 80)
    total_bytes = sum(i["length"] for i in instructions)
    opcodes = Counter(i["opcode"] for i in instructions)
    print(f"Summary: {len(instructions)} instructions | {total_bytes} bytes | Top opcodes: {', '.join(f'{k}:{v}' for k, v in opcodes.most_common(5))}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("targets", nargs="*", type=Path, default=[], help="Path to shader binary .bin file(s) or folder")
    parser.add_argument("--hex", default=None, help="Direct hex string to decode (e.g. '0300a0bf8002047e')")
    parser.add_argument("--output", type=Path, default=None, help="Save disassembly listing to output file")

    args = parser.parse_args()

    # Hex decode mode
    if args.hex:
        raw_bytes = bytes.fromhex(args.hex.replace(" ", ""))
        insns = disassemble_binary(raw_bytes, "hex_input")
        print_disassembly("hex_input", insns)
        return 0

    if not args.targets:
        parser.error("Specify targets or provide hex via --hex")

    all_listings = []
    for target in args.targets:
        if target.is_file():
            data = target.read_bytes()
            insns = disassemble_binary(data, target.name)
            print_disassembly(target.name, insns)
            all_listings.append((target.name, insns))
        elif target.is_dir():
            files = sorted(target.glob("*.bin"))
            print(f"[*] Found {len(files)} shader binary files in {target.name}...")
            for f in files[:20]:  # Limit preview to 20 when folder passed
                data = f.read_bytes()
                insns = disassemble_binary(data, f.name)
                print_disassembly(f.name, insns)
                all_listings.append((f.name, insns))

    if args.output and all_listings:
        with args.output.open("w", encoding="utf-8") as out_f:
            for name, insns in all_listings:
                out_f.write(f"; Disassembly: {name}\n")
                for item in insns:
                    out_f.write(f"    {item['assembly']:<40} ; +0x{item['offset']:04x} // {item['comment']}\n")
                out_f.write("\n")
        print(f"[+] Disassembly saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
