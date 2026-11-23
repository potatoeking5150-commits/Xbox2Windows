"""Check tools/form_coverage.py: its reference lists (denominators), one fixture-to-instruction example,
how skipped and failed tests change the count, and that only accepted shader programs count.
Needs requirements-dev.txt (iced-x86); no Windows."""
from datetime import datetime
import os
from pathlib import Path
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import form_coverage as fc
import form_reference as fr

ENCODINGS = {"DPP16", "DPP8", "DS", "EXP", "FLAT", "MIMG", "MTBUF", "MUBUF", "SDWA", "SMEM", "SOP1", "SOP2",
             "SOPC", "SOPK", "SOPP", "VINTRP", "VOP1", "VOP2", "VOP3", "VOP3P", "VOPC"}


def check_denominators():
    for name in ("x64_zen2.tsv", "gfx1013.tsv", "rdna2.tsv"):
        lines = (fc.REFERENCE / name).read_text(encoding="utf-8").splitlines()
        count, source = int(lines[0].split()[1]), lines[0]
        assert "Source:" in source and len(lines) - 1 == count == len(set(lines[1:])), name
    # x64: the committed list is exactly what iced-x86 yields for Zen 2 user mode today.
    assert fc.load_reference("x64_zen2.tsv") == fr.x64_forms()
    # gfx: every form sits under a real encoding with a name that encoding allows.
    for name in ("gfx1013.tsv", "rdna2.tsv"):
        reference = fc.load_reference(name)
        assert {group for group, _ in reference} == ENCODINGS, name
        assert ("SOPP", "s_endpgm") in reference and ("EXP", "exp") in reference
        assert ("SOPK", "s_addk_i32") in reference and ("SOPK", "s_mulk_i32") in reference
        assert fc.gfx_forms({"s_addk_i32", "s_mulk_i32"}, reference) == {
            ("SOPK", "s_addk_i32"), ("SOPK", "s_mulk_i32")}
    gfx1013, rdna2 = fc.load_reference("gfx1013.tsv"), fc.load_reference("rdna2.tsv")
    assert ("SMEM", "s_atomic_add") in gfx1013 and ("SMEM", "s_atomic_add") not in rdna2
    # The LLVM syntax-doc parser: short headings, operand-less instructions, and the trailing index.
    with tempfile.TemporaryDirectory() as tmp:
        doc = Path(tmp) / "sample.rst"
        doc.write_text("DS\n--\n\n.. parsed-literal::\n\n    ds_nop\n    ds_add_u32     vaddr, vdata\n\n"
                       "SOPP\n----\n\n    s_endpgm\n\n.. toctree::\n    :hidden:\n\n    gfx10_vdst_123\n")
        assert fr.llvm_forms(doc) == {("DS", "ds_nop"), ("DS", "ds_add_u32"), ("SOPP", "s_endpgm")}
        doc.write_text("DPP8\n----\n\n    ds_nop\n")
        try:
            fr.llvm_forms(doc)
        except AssertionError:
            pass
        else:
            raise AssertionError("a form filed under the wrong encoding was accepted")


# Small fixtures written here, so later edits to the repository's own tests/cpu/*.asm cannot change the
# expected counts. They use the repository's fixture names, so the real CPU_FIXTURES mapping applies.
BRANCH = """branch PROC
    mov eax, ecx
    cmp ecx, edx
IFDEF INVERT_BRANCH
    jae short high
ELSE
    jb short low
ENDIF
    add eax, 1
    jmp short done
low:
    sub eax, 1
high:
done:
    ret
branch ENDP
END
"""
DIRECT = "direct PROC\n    sub rsp, 28h\n    call helper\n    add rsp, 28h\n    ret\ndirect ENDP\nEND\n"
SWITCH = """switch PROC
    cmp ecx, 3
    ja short default
    lea rdx, table
    movsxd rax, DWORD PTR [rdx+rcx*4]
    add rax, rdx
    jmp rax
default:
    neg eax
    ret
table DD 0
switch ENDP
END
"""


def sample_repo(tmp, cpu, scripts=None):
    """A repository root holding only the given tests/cpu/<name>.asm and tests/<name>.py sources."""
    root = Path(tmp)
    (root / "tests" / "cpu").mkdir(parents=True, exist_ok=True)
    for name, text in cpu.items():
        (root / "tests" / "cpu" / f"{name}.asm").write_text(text, encoding="utf-8")
    for name, text in (scripts or {}).items():
        (root / "tests" / name).write_text(text, encoding="utf-8")
    return root


def check_example():
    """branch.asm -> eight x64 forms, counted only while cpu_leaf_aot_branches passes."""
    branch = fc.masm_mnemonics(BRANCH)
    assert branch == {"mov", "cmp", "jb", "jae", "add", "jmp", "sub", "ret"}, branch  # jae: INVERT_BRANCH build
    reference = fc.load_reference("x64_zen2.tsv")
    assert fc.x64_forms(branch, reference) == {("Integer", m) for m in branch}

    with tempfile.TemporaryDirectory() as tmp:
        root = sample_repo(tmp, {"branch": BRANCH, "direct": DIRECT})
        passed = fc.measure({"cpu_leaf_aot_branches": "pass"}, root)["x64_zen2"]
        skipped = fc.measure({"cpu_leaf_aot_branches": "skip"}, root)["x64_zen2"]
        failed = fc.measure({"cpu_leaf_aot_branches": "fail"}, root)["x64_zen2"]
        assert passed["covered"] == 8 and passed["groups"]["Integer"]["covered"] == 8
        assert skipped["covered"] == 0 and skipped["blocked_by_skip"] == sorted(f"Integer {m}" for m in branch)
        assert failed["covered"] == 0 and failed["blocked_by_skip"] == []
        # A skip never adds coverage, even when another passing fixture already covers the same form.
        both = fc.measure({"cpu_leaf_aot_branches": "skip", "cpu_aot_direct_call": "pass"}, root)["x64_zen2"]
        direct = fc.masm_mnemonics(DIRECT)
        assert direct == {"sub", "call", "add", "ret"}, direct
        assert both["covered"] == len(fc.x64_forms(direct, reference)) == 4
        assert both["blocked_by_skip"] == sorted(f"Integer {m}" for m in branch - direct)

    # gfx1013: _e32 selects the plain encoding, _e64 the VOP3 form; any passing adapter counts.
    gfx = fc.load_reference("gfx1013.tsv")
    assert fc.gfx_forms({"v_interp_p1_f32_e32", "v_fmac_f32_e64", "exp"}, gfx) == {
        ("VINTRP", "v_interp_p1_f32"), ("VOP3", "v_fmac_f32_e64"), ("EXP", "exp")}
    assert fc.fixture_status({"pixel_aot_warp": "pass", "pixel_aot_hardware": "skip"},
                             ("pixel_aot_warp", "pixel_aot_hardware")) == "pass"
    assert fc.fixture_status({"pixel_aot_hardware": "skip"}, ("pixel_aot_warp", "pixel_aot_hardware")) == "skip"


def check_jump_tables():
    """switch.asm (#86): a skipped cpu_aot_jump_tables run contributes zero covered forms."""
    switch = fc.masm_mnemonics(SWITCH)
    assert switch == {"cmp", "ja", "lea", "movsxd", "add", "jmp", "neg", "ret"}, sorted(switch)
    with tempfile.TemporaryDirectory() as tmp:
        root = sample_repo(tmp, {"switch": SWITCH, "branch": BRANCH})
        passed = fc.measure({"cpu_aot_jump_tables": "pass"}, root)["x64_zen2"]
        skipped = fc.measure({"cpu_aot_jump_tables": "skip"}, root)["x64_zen2"]
        assert passed["covered"] == len(switch) == 8
        assert skipped["covered"] == 0 and skipped["blocked_by_skip"] == sorted(f"Integer {m}" for m in switch)
        # Next to a passing fixture, the skipped run still adds nothing: only branch.asm's forms count.
        mixed = fc.measure({"cpu_aot_jump_tables": "skip", "cpu_leaf_aot_branches": "pass"}, root)["x64_zen2"]
        assert mixed["covered"] == len(fc.masm_mnemonics(BRANCH)) == 8
        assert mixed["blocked_by_skip"] == sorted(f"Integer {m}" for m in switch - fc.masm_mnemonics(BRANCH))


def check_scalar_float():
    """scalar_float.asm: the first SSE and SSE2 forms, counted only while cpu_aot_scalar_float passes."""
    root = Path(fc.__file__).resolve().parents[1]
    forms = fc.masm_mnemonics((root / "tests" / "cpu" / "scalar_float.asm").read_text(encoding="utf-8"))
    reference = fc.load_reference("x64_zen2.tsv")
    groups = {}
    for group, form in fc.x64_forms(forms, reference):
        groups.setdefault(group, set()).add(form)
    assert not [form for form in forms if not fc.x64_forms({form}, reference)]
    assert groups["SSE"] == {"movss", "movaps", "addss", "subss", "mulss", "divss", "cvtsi2ss", "cvttss2si",
                             "ucomiss", "comiss", "xorps"}, sorted(groups["SSE"])
    assert groups["SSE2"] == {"addsd", "subsd", "mulsd", "divsd", "cvtsi2sd", "cvttsd2si", "cvtss2sd", "cvtsd2ss",
                              "ucomisd", "comisd", "xorpd"}, sorted(groups["SSE2"])
    # The reference files a mnemonic under its oldest encoding, so these three sit outside both groups.
    assert groups["MMX"] == {"movd", "movq", "pxor"} and "movsd" in groups["Integer"]
    with tempfile.TemporaryDirectory() as tmp:
        sample = sample_repo(tmp, {"scalar_float": "; coverage: source cpu_aot_scalar_float\nf PROC\n    xorps xmm0, xmm0\n    cvtsi2sd xmm0, ecx\n"
                                                   "    ucomisd xmm0, xmm1\n    jp short l\nl:  ret\nf ENDP\nEND\n"})
        passed = fc.measure({"cpu_aot_scalar_float": "pass"}, sample)["x64_zen2"]
        skipped = fc.measure({"cpu_aot_scalar_float": "skip"}, sample)["x64_zen2"]
        failed = fc.measure({"cpu_aot_jump_tables": "pass", "cpu_aot_scalar_float": "fail"}, sample)["x64_zen2"]
        assert passed["covered"] == 5 and passed["groups"]["SSE"]["covered"] == 1
        assert passed["groups"]["SSE2"]["covered"] == 2 and passed["groups"]["Integer"]["covered"] == 2
        assert skipped["covered"] == 0 and skipped["blocked_by_skip"] == [
            "Integer jp", "Integer ret", "SSE xorps", "SSE2 cvtsi2sd", "SSE2 ucomisd"]
        assert failed["covered"] == 0 and failed["blocked_by_skip"] == []
    # In the real repository the fixture is classified and counts with its own test only.
    real = fc.measure({"cpu_aot_scalar_float": "pass"})
    assert real["fixtures"]["scalar_float.asm"] == "pass" and "scalar_float.asm" not in real["unclassified"]
    assert [real["x64_zen2"]["groups"][group]["covered"] for group in ("SSE", "SSE2")] == [11, 11]
    assert fc.measure({"cpu_aot_jump_tables": "pass"})["x64_zen2"]["groups"]["SSE"]["covered"] == 0


def check_unclassified():
    """A CPU fixture missing from CPU_FIXTURES is reported, never counted, never fatal. Shader text in a
    test file counts for nothing: only translation records do."""
    with tempfile.TemporaryDirectory() as tmp:
        root = sample_repo(tmp, {"branch": BRANCH, "added_later": "f PROC\n    xor eax, eax\n    ret\nf ENDP\nEND\n"},
                           {"added_later_check.py": 'P = "s_mov_b32 s0, 0\\ns_endpgm"\n'})
        every_test_passed = {"cpu_leaf_aot_branches": "pass", "added_later": "pass", "added_later_check": "pass"}
        m = fc.measure(every_test_passed, root)
        assert m["unclassified"] == ["added_later.asm"], m["unclassified"]
        assert m["x64_zen2"]["covered"] == 8  # branch.asm only; xor stays uncovered
        assert ("Integer", "xor") in fc.x64_forms({"xor"}, fc.load_reference("x64_zen2.tsv"))
        assert m["gfx1013"]["covered"] == m["gfx1013"]["known_in_fixtures"] == 0
        report = fc.report(m)
        assert "Not counted until classified: added_later.asm" in report
        assert "Shader translation records from this run: none" in report
        assert "Not counted" not in fc.report(fc.measure(every_test_passed, sample_repo(Path(tmp) / "b", {"branch": BRANCH})))


def check_local_cpu_registration():
    with tempfile.TemporaryDirectory() as tmp:
        root = sample_repo(tmp, {
            "local": "; coverage: source cpu_local\nxor eax, eax\n",
            "macro_case": "; coverage: record cpu_macro\nsub eax, eax\n",
            "register_probe": "; coverage: source cpu_local\nadd eax, eax\n",
        })
        for state in ("pass", "skip", "fail", "absent"):
            results = {} if state == "absent" else {"cpu_local": state, "cpu_macro": state}
            measured = fc.measure(results, root, {"cpu_macro": {"adc"}})
            assert measured["fixtures"]["register_probe.asm"] == "not translated"
            assert measured["fixtures"]["local.asm"] == measured["fixtures"]["macro_case.asm"] == state
            assert measured["x64_zen2"]["covered"] == (2 if state == "pass" else 0)
            assert measured["shader_records"] == []
        assert fc.measure({"cpu_macro": "pass"}, root)["x64_zen2"]["covered"] == 0


def check_shader_records():
    """gfx forms come only from programs the translator accepted, recorded by the same run. The review's
    example: a program that reaches only must_reject (s_nop in place of a wait below, s_branch in the real
    controls) must stay out of covered."""
    import pixel_aot_check as pixel  # tests/ is on sys.path when this file runs
    gfx = fc.load_reference("gfx1013.tsv")
    assert ("SOPP", "s_nop") in gfx  # so counting s_nop would change the result
    pixel.ACCEPTED.clear()
    program = pixel.assembly(False)
    pixel.must_reject(pixel.translate, [line.replace("s_waitcnt lgkmcnt(0)", "s_nop 0") for line in program])
    assert not pixel.ACCEPTED  # the rejected program left no trace
    pixel.translate(program)
    accepted = {line.split()[0] for line in program}
    assert pixel.ACCEPTED == accepted and "s_nop" not in accepted
    with tempfile.TemporaryDirectory() as tmp:
        root = sample_repo(Path(tmp) / "repo", {})
        record = Path(tmp) / "records" / "pixel_aot_warp.txt"
        pixel.write_record(record)
        now = time.time()
        records = fc.read_records(record.parent, (now - 60, now + 60))
        assert records == {"pixel_aot_warp": accepted}
        expected = len(fc.gfx_forms(accepted, gfx))
        passed = fc.measure({"pixel_aot_warp": "pass"}, root, records)["gfx1013"]
        skipped = fc.measure({"pixel_aot_warp": "skip"}, root, records)["gfx1013"]
        failed = fc.measure({"pixel_aot_warp": "fail"}, root, records)["gfx1013"]
        assert passed["covered"] == expected > 0
        assert skipped["covered"] == 0 and len(skipped["blocked_by_skip"]) == expected
        assert failed["covered"] == 0 and failed["blocked_by_skip"] == []
        # Negative control: had the rejected program been recorded, s_nop would be covered.
        leaked = fc.measure({"pixel_aot_warp": "pass"}, root, {"pixel_aot_warp": accepted | {"s_nop"}})["gfx1013"]
        assert leaked["covered"] == expected + 1
        # A record written outside the JUnit run comes from another run, so it is ignored.
        os.utime(record, (now - 3600, now - 3600))
        assert fc.read_records(record.parent, (now - 60, now + 60)) == {}
    # The same holds in the real controls. The scheduling-hint fixtures translate programs with s_nop
    # and the hints, so those forms are recorded; s_branch reaches only must_reject and stays out.
    assert ("SOPP", "s_branch") in gfx
    pixel.ACCEPTED.clear()
    with tempfile.TemporaryDirectory() as tmp:
        pixel.check_controls(Path(tmp))
    assert {"s_nop", "s_clause", "s_setprio", "s_sleep", "image_sample"} <= pixel.ACCEPTED
    assert "s_branch" not in pixel.ACCEPTED and "s_code_end" not in pixel.ACCEPTED
    # The mode-register fixture records the immediate write; the register forms reach only rejections.
    assert ("SOPK", "s_setreg_imm32_b32") in gfx and "s_setreg_imm32_b32" in pixel.ACCEPTED
    assert not {"s_setreg_b32", "s_getreg_b32"} & pixel.ACCEPTED
    # The trigonometry fixture covers both encodings of v_sin_f32 and v_cos_f32 once its test passes.
    trig = {("VOP1", "v_sin_f32"), ("VOP1", "v_cos_f32"), ("VOP3", "v_sin_f32_e64"), ("VOP3", "v_cos_f32_e64")}
    assert trig <= gfx and trig <= fc.gfx_forms(pixel.ACCEPTED, gfx)
    with tempfile.TemporaryDirectory() as tmp:
        root = sample_repo(Path(tmp) / "repo", {})
        words = {"v_sin_f32_e32", "v_sin_f32_e64", "v_cos_f32_e32", "v_cos_f32_e64"}
        assert words <= pixel.ACCEPTED
        covered = fc.measure({"pixel_aot_warp": "pass"}, root, {"pixel_aot_warp": set(pixel.ACCEPTED)})
        without = fc.measure({"pixel_aot_warp": "pass"}, root, {"pixel_aot_warp": pixel.ACCEPTED - words})
        assert covered["gfx1013"]["covered"] == without["gfx1013"]["covered"] + len(trig)
    # Count the supported 64-bit SOP1/SOP2 and BFE forms in both gfx1013 and rdna2.
    masked_forms = {("SOP1", "s_and_saveexec_b64"), ("SOPP", "s_cbranch_execz")}
    assert masked_forms <= fc.gfx_forms(pixel.ACCEPTED, gfx)
    scalar64_forms = {
        ("SOP1", "s_cmov_b64"), ("SOP1", "s_not_b64"), ("SOP1", "s_brev_b64"),
        ("SOP2", "s_and_b64"), ("SOP2", "s_or_b64"), ("SOP2", "s_xor_b64"),
        ("SOP2", "s_andn2_b64"), ("SOP2", "s_orn2_b64"),
        ("SOP2", "s_nand_b64"), ("SOP2", "s_nor_b64"), ("SOP2", "s_xnor_b64"),
        ("SOP2", "s_lshl_b64"), ("SOP2", "s_lshr_b64"), ("SOP2", "s_ashr_i64"),
        ("SOP2", "s_bfm_b32"), ("SOP2", "s_bfm_b64"), ("SOP2", "s_bfe_u32"), ("SOP2", "s_bfe_i32"),
        ("SOP2", "s_cselect_b64"),
        ("SOP2", "s_bfe_u64"), ("SOP2", "s_bfe_i64"),
    }
    sopc_forms = {
        ("SOPC", "s_cmp_eq_i32"), ("SOPC", "s_cmp_lg_i32"),
        ("SOPC", "s_cmp_gt_i32"), ("SOPC", "s_cmp_ge_i32"),
        ("SOPC", "s_cmp_lt_i32"), ("SOPC", "s_cmp_le_i32"),
        ("SOPC", "s_cmp_eq_u32"), ("SOPC", "s_cmp_lg_u32"),
        ("SOPC", "s_cmp_gt_u32"), ("SOPC", "s_cmp_ge_u32"),
        ("SOPC", "s_cmp_lt_u32"), ("SOPC", "s_cmp_le_u32"),
        ("SOPC", "s_cmp_eq_u64"), ("SOPC", "s_cmp_lg_u64"),
        ("SOPC", "s_bitcmp0_b32"), ("SOPC", "s_bitcmp1_b32"),
        ("SOPC", "s_bitcmp0_b64"), ("SOPC", "s_bitcmp1_b64"),
    }
    vop2_forms = {
        ("VOP2", "v_lshrrev_b32"), ("VOP2", "v_ashrrev_i32"), ("VOP2", "v_lshlrev_b32"),
        ("VOP2", "v_and_b32"), ("VOP2", "v_or_b32"), ("VOP2", "v_xor_b32"),
        ("VOP2", "v_add_nc_u32"), ("VOP2", "v_sub_nc_u32"), ("VOP2", "v_subrev_nc_u32"),
        ("VOP2", "v_xnor_b32"),
    }
    e64_integer_forms = {("VOP3", f"{form}_e64") for form in (
        "v_and_b32", "v_or_b32", "v_xor_b32", "v_xnor_b32", "v_not_b32",
        "v_lshlrev_b32", "v_lshrrev_b32", "v_ashrrev_i32", "v_cvt_f32_ubyte2")}
    sdwa_integer_forms = {("SDWA", f"{form}_sdwa") for form in (
        "v_and_b32", "v_or_b32", "v_xor_b32", "v_xnor_b32", "v_not_b32",
        "v_lshlrev_b32", "v_lshrrev_b32", "v_ashrrev_i32",
        "v_add_nc_u32", "v_sub_nc_u32", "v_subrev_nc_u32",
        "v_min_i32", "v_max_i32", "v_min_u32", "v_max_u32")}
    assert len(sdwa_integer_forms) == 15
    # gfx10 encodes the carry-out add and subtracts as VOP3 only, under their plain names.
    carry_out_forms = {("VOP3", "v_add_co_u32"), ("VOP3", "v_sub_co_u32"), ("VOP3", "v_subrev_co_u32")}
    # The carry-in add and subtracts in both the VOP2 (e32) and the VOP3 (e64) encoding.
    carry_in_forms = {(group, f"v_{op}_co_ci_u32{suffix}") for op in ("add", "sub", "subrev")
                      for group, suffix in (("VOP2", ""), ("VOP3", "_e64"))}
    sad_forms = {("VOP3", "v_sad_hi_u8"), ("VOP3", "v_sad_u32"), ("VOP3", "v_msad_u8")}
    mul24_forms = {
        ("VOP2", "v_mul_i32_i24"), ("VOP3", "v_mul_i32_i24_e64"),
        ("VOP2", "v_mul_u32_u24"), ("VOP3", "v_mul_u32_u24_e64"),
        ("VOP2", "v_mul_hi_i32_i24"), ("VOP3", "v_mul_hi_i32_i24_e64"),
        ("VOP2", "v_mul_hi_u32_u24"), ("VOP3", "v_mul_hi_u32_u24_e64"),
    }
    pack_int16_forms = {("VOP3", "v_cvt_pk_u16_u32"), ("VOP3", "v_cvt_pk_i16_i32")}
    sop2_lshl_add_forms = {
        ("SOP2", "s_lshl1_add_u32"), ("SOP2", "s_lshl2_add_u32"),
        ("SOP2", "s_lshl3_add_u32"), ("SOP2", "s_lshl4_add_u32"),
    }
    sop2_absdiff_forms = {
        ("SOP2", "s_absdiff_i32"),
    }
    rejected_forms = {
        "s_or_saveexec_b64", "s_andn2_saveexec_b64", "s_and_saveexec_b32",
        "v_mul_i32_i24_sdwa",
        "v_cvt_pk_u8_f32", "v_cvt_pknorm_i16_f32", "v_cvt_pknorm_u16_f32",
        "v_cvt_pknorm_i16_f16", "v_cvt_pknorm_u16_f16",
        "s_lshl0_add_u32", "s_lshl5_add_u32",
        "s_cbranch_execnz", "s_cbranch_vccz", "s_cbranch_vccnz",
        "s_cbranch_scc0", "s_cbranch_scc1",
        "s_absdiff_i64", "s_absdiff_u32",
        "v_sad_u8", "v_sad_u16", "v_qsad_pk_u16_u8", "v_mqsad_pk_u16_u8", "v_mqsad_u32_u8",
    }
    assert not (pixel.ACCEPTED & rejected_forms), pixel.ACCEPTED & rejected_forms
    for name in ("gfx1013.tsv", "rdna2.tsv"):
        covered = fc.gfx_forms(pixel.ACCEPTED, fc.load_reference(name))
        assert scalar64_forms <= covered, (name, scalar64_forms - covered)
        assert sopc_forms <= covered, (name, sopc_forms - covered)
        assert vop2_forms <= covered, (name, vop2_forms - covered)
        assert e64_integer_forms <= covered, (name, e64_integer_forms - covered)
        assert sdwa_integer_forms <= covered, (name, sdwa_integer_forms - covered)
        # The VOP3 spellings of the interpolation pair and of v_nop.
        e64_spelling_forms = {("VOP3", "v_interp_p1_f32_e64"), ("VOP3", "v_interp_p2_f32_e64"), ("VOP3", "v_nop_e64")}
        assert e64_spelling_forms <= covered, (name, e64_spelling_forms - covered)
        assert carry_out_forms <= covered, (name, carry_out_forms - covered)
        assert carry_in_forms <= covered, (name, carry_in_forms - covered)
        assert sad_forms <= covered, (name, sad_forms - covered)
        assert mul24_forms <= covered, (name, mul24_forms - covered)
        assert sop2_lshl_add_forms <= covered, (name, sop2_lshl_add_forms - covered)
        assert sop2_absdiff_forms <= covered, (name, sop2_absdiff_forms - covered)
        assert pack_int16_forms <= covered, (name, pack_int16_forms - covered)
        # v_mac_f32 is in the gfx1013 list only.
        approx_e64_forms = {("VOP3", "v_exp_f32_e64"), ("VOP3", "v_log_f32_e64")}
        approx_e64_forms |= {("VOP3", "v_mac_f32_e64")} if name == "gfx1013.tsv" else set()
        assert approx_e64_forms <= covered, (name, approx_e64_forms - covered)
        # SDWA forms of the unary binary32 operations, whole-dword sources only.
        sdwa_unary_forms = {("SDWA", f"v_{op}_f32_sdwa") for op in (
            "exp", "log", "rcp", "rcp_iflag", "rsq", "sqrt", "sin", "cos", "rndne", "frexp_mant")}
        assert sdwa_unary_forms <= covered, (name, sdwa_unary_forms - covered)
    # Integer and binary16 comparisons: e32 counts as VOPC and e64 as VOP3, for every 16-, 32- and 64-bit
    # predicate the translator lowers; the rejected exec-writing neighbours stay out.
    narrow_int = {f"v_cmp_{op}_{kind}" for op in ("eq", "ne", "lt", "le", "gt", "ge")
                  for kind in ("i16", "u16", "i32", "u32")}
    narrow_int |= {f"v_cmp_{op}_{kind}" for op in ("f", "t") for kind in ("i32", "u32")}
    ops64 = ("f", "lt", "eq", "le", "gt", "ne", "ge", "t")
    int64 = {f"v_cmp_{op}_{kind}" for op in ops64 for kind in ("i64", "u64")}
    assert len(int64) == len(ops64) * 2, sorted(int64)
    int_compares = narrow_int | int64
    f16_compares = {f"v_cmp_{op}_f16" for op in pixel.PREDICATES} | {"v_cmp_class_f16"}
    f32_compares = {f"v_cmp_{op}_f32" for op in pixel.PREDICATES} | {"v_cmp_class_f32"}
    all_compares = int_compares | f16_compares | f32_compares
    compare_forms = ({("VOPC", form) for form in all_compares}
                     | {("VOP3", f"{form}_e64") for form in all_compares})
    # Each name is counted once as VOPC and once as VOP3; the four sets must not share a name.
    assert len(compare_forms) == 2 * (len(narrow_int) + len(int64) + len(f16_compares) + len(f32_compares)), len(compare_forms)
    compare_neighbours = {"v_cmpx_lt_i16", "v_cmpx_lt_i32", "v_cmpx_lt_i64", "v_cmpx_lt_u64",
                          "v_cmpx_lt_f32", "v_cmp_class_f64", "v_cmp_lt_f64"}
    for name in ("gfx1013.tsv", "rdna2.tsv"):
        reference = fc.load_reference(name)
        assert compare_forms | {("VOPC", n) for n in compare_neighbours} <= reference, name
        covered = fc.gfx_forms(pixel.ACCEPTED, reference)
        assert compare_forms <= covered, (name, sorted(compare_forms - covered))
        assert not {form for _, form in covered} & (compare_neighbours | {f"{n}_e64" for n in compare_neighbours})
    # The scalar ALU fixture counts its SOP1, SOP2, SOPK and SOPC forms; the neighbours the controls
    # reject (s_addc_u32, s_cmp_lt_u64, s_getreg_b32) stay out.
    scalar = {"s_movk_i32", "s_cmovk_i32", "s_not_b32", "s_add_u32", "s_sub_u32", "s_add_i32", "s_sub_i32",
              "s_mul_i32", "s_min_u32", "s_max_u32", "s_min_i32", "s_max_i32", "s_and_b32", "s_or_b32",
              "s_xor_b32", "s_andn2_b32", "s_orn2_b32", "s_lshl_b32", "s_lshr_b32", "s_ashr_i32",
              "s_cselect_b32", *(f"s_cmp_{r}_{t}32" for r in ("eq", "lg", "gt", "ge", "lt", "le") for t in "iu"),
              *(f"s_cmpk_{r}_{t}32" for r in ("eq", "lg", "gt", "ge", "lt", "le") for t in "iu")}
    for name in ("gfx1013.tsv", "rdna2.tsv"):
        forms = fc.gfx_forms(pixel.ACCEPTED, fc.load_reference(name))
        assert {form for _, form in forms} >= scalar, sorted(scalar - {form for _, form in forms})
        assert not {"s_addc_u32", "s_cmp_lt_u64", "s_getreg_b32"} & {f for _, f in forms}
        assert {group for group, form in forms if form in scalar} == {"SOP1", "SOP2", "SOPC", "SOPK"}
    # The rounding fixture covers both encodings of v_ceil_f32 and v_rndne_f32 once its test passes.
    rounding = {("VOP1", "v_ceil_f32"), ("VOP1", "v_rndne_f32"),
                ("VOP3", "v_ceil_f32_e64"), ("VOP3", "v_rndne_f32_e64")}
    assert rounding <= gfx and rounding <= fc.gfx_forms(pixel.ACCEPTED, gfx)
    with tempfile.TemporaryDirectory() as tmp:
        root = sample_repo(Path(tmp) / "repo", {})
        words = {"v_ceil_f32_e32", "v_ceil_f32_e64", "v_rndne_f32_e32", "v_rndne_f32_e64"}
        assert words <= pixel.ACCEPTED
        covered = fc.measure({"pixel_aot_warp": "pass"}, root, {"pixel_aot_warp": set(pixel.ACCEPTED)})
        without = fc.measure({"pixel_aot_warp": "pass"}, root, {"pixel_aot_warp": pixel.ACCEPTED - words})
        assert covered["gfx1013"]["covered"] == without["gfx1013"]["covered"] + len(rounding)
    # Float-to-word conversions (truncate, floor, round-to-+inf) cover VOP1 e32 and VOP3 e64;
    # DPP/SDWA variants stay excluded.
    cvt_f32_int = {
        ("VOP1", "v_cvt_u32_f32"), ("VOP1", "v_cvt_i32_f32"),
        ("VOP1", "v_cvt_flr_i32_f32"), ("VOP1", "v_cvt_rpi_i32_f32"),
        ("VOP3", "v_cvt_u32_f32_e64"), ("VOP3", "v_cvt_i32_f32_e64"),
        ("VOP3", "v_cvt_flr_i32_f32_e64"), ("VOP3", "v_cvt_rpi_i32_f32_e64"),
    }
    cvt_f32_excluded = {"v_cvt_flr_i32_f32_dpp", "v_cvt_flr_i32_f32_sdwa",
                        "v_cvt_rpi_i32_f32_dpp", "v_cvt_rpi_i32_f32_sdwa"}
    for name in ("gfx1013.tsv", "rdna2.tsv"):
        reference = fc.load_reference(name)
        assert cvt_f32_int <= reference, name
        forms = fc.gfx_forms(pixel.ACCEPTED, reference)
        assert cvt_f32_int <= forms, (name, sorted(cvt_f32_int - forms))
        assert not ({form for _, form in forms} & cvt_f32_excluded), name


def check_scalar_brev():
    """s_brev_b32 counts as covered SOP1 in both references once the lowering accepts it."""
    import pixel_aot_check as pixel  # tests/ is on sys.path when this file runs
    saved = set(pixel.ACCEPTED)
    pixel.ACCEPTED.clear()
    pixel.translate_int(pixel.brev_scalar_ops())
    assert "s_brev_b32" in pixel.ACCEPTED
    for name in ("gfx1013.tsv", "rdna2.tsv"):
        reference = fc.load_reference(name)
        assert ("SOP1", "s_brev_b32") in reference, name
        assert ("SOP1", "s_brev_b32") in fc.gfx_forms(pixel.ACCEPTED, reference), name
        # Negative control: without the accepted program the form is not covered.
        assert ("SOP1", "s_brev_b32") not in fc.gfx_forms(pixel.ACCEPTED - {"s_brev_b32"}, reference), name
    pixel.ACCEPTED.clear()
    pixel.ACCEPTED.update(saved)


def check_plain_bitscan_forms():
    """The two plain bitscan fixtures cover VOP1 v_bfrev_b32, v_ffbh_u32, v_ffbh_i32, v_ffbl_b32 and VOP2
    v_xnor_b32 in both reference lists; a form neither fixture uses stays uncovered."""
    import pixel_aot_check as pixel
    wanted = {("VOP1", "v_bfrev_b32"), ("VOP1", "v_ffbh_u32"), ("VOP1", "v_ffbh_i32"), ("VOP1", "v_ffbl_b32"),
              ("VOP2", "v_xnor_b32")}
    e64 = {("VOP3", "v_bfrev_b32_e64"), ("VOP3", "v_ffbh_u32_e64"), ("VOP3", "v_xnor_b32_e64")}
    untouched = {("VOP3", "v_ffbl_b32_e64"), ("VOP3", "v_ffbh_i32_e64")}
    for name in ("gfx1013.tsv", "rdna2.tsv"):
        reference = fc.load_reference(name)
        assert wanted | e64 | untouched <= reference, name
        pixel.ACCEPTED.clear()
        assert not wanted & fc.gfx_forms(pixel.ACCEPTED, reference)  # nothing lowered, nothing covered
        for second in (False, True):
            pixel.translate_int(pixel.plain_bitscan_ops(second))
        covered = fc.gfx_forms(pixel.ACCEPTED, reference)
        assert wanted | e64 <= covered, (name, sorted((wanted | e64) - covered))
        assert not untouched & covered, name
    pixel.ACCEPTED.clear()



def check_int16_vop3_forms():
    """The 12 non-packed 16-bit integer VOP3 forms count as covered once the translator accepts their fixtures.
    The unsuffixed spellings match the reference; the _e64 control does not count (gfx_forms keeps _e64 names), and
    the nearby VOP3 forms that stay rejected (f16 ops) stay uncovered."""
    import pixel_aot
    import pixel_aot_check as pixel  # tests/ is on sys.path when this file runs
    saved = set(pixel.ACCEPTED)
    pixel.ACCEPTED.clear()
    try:
        for group in range(3):
            pixel.translate_int(pixel.int16_vop3_ops(group))
        forms = {("VOP3", f"v_{name}") for name in pixel_aot.VOP3_INT16}
        assert len(forms) == 12
        neighbours = {("VOP3", "v_min_f16_e64"), ("VOP3", "v_cvt_f16_u16_e64")}
        for name in ("gfx1013.tsv", "rdna2.tsv"):
            reference = fc.load_reference(name)
            assert forms <= reference, (name, forms - reference)
            assert neighbours <= reference, (name, neighbours - reference)
            covered = fc.gfx_forms(pixel.ACCEPTED, reference)
            assert forms <= covered, (name, forms - covered)
            assert not neighbours & covered, (name, neighbours & covered)
        # The suffixed control spelling is a different word, so it adds no reference form.
        reference = fc.load_reference("gfx1013.tsv")
        assert fc.gfx_forms({"v_add_nc_u16_e64"}, reference) == set()
    finally:
        pixel.ACCEPTED.clear()
        pixel.ACCEPTED.update(saved)
def check_f16_arith_forms():
    """The eight plain binary16 VOP2 forms (e32 and e64 spellings of add, sub, subrev and mul) count as covered once the
    translator accepts their fixtures; min, max, fmac, ldexp and the DPP and SDWA spellings stay uncovered."""
    import pixel_aot_check as pixel
    saved = set(pixel.ACCEPTED)
    pixel.ACCEPTED.clear()
    try:
        for key in pixel.F16_ARITH_PROGRAMS:
            pixel.translate_f16(pixel.f16_arith_program(*key))
        ops = ("add", "sub", "subrev", "mul")
        forms = {("VOP2", f"v_{op}_f16") for op in ops} | {("VOP3", f"v_{op}_f16_e64") for op in ops}
        neighbours = {("VOP2", "v_min_f16"), ("VOP2", "v_max_f16"), ("VOP2", "v_fmac_f16"), ("VOP2", "v_ldexp_f16"),
                      ("VOP3", "v_min_f16_e64"), ("SDWA", "v_add_f16_sdwa"), ("DPP16", "v_add_f16_dpp"),
                      ("DPP8", "v_add_f16_dpp")}
        for name in ("gfx1013.tsv", "rdna2.tsv"):
            reference = fc.load_reference(name)
            assert forms <= reference, (name, forms - reference)
            assert neighbours <= reference, (name, neighbours - reference)
            covered = fc.gfx_forms(pixel.ACCEPTED, reference)
            assert forms <= covered, (name, forms - covered)
            assert not neighbours & covered, (name, neighbours & covered)
        # Without the binary16 fixtures none of the eight forms is covered.
        pixel.ACCEPTED.clear()
        pixel.translate_int(pixel.int16_vop3_ops(0))
        assert not forms & fc.gfx_forms(pixel.ACCEPTED, fc.load_reference("gfx1013.tsv"))
    finally:
        pixel.ACCEPTED.clear()
        pixel.ACCEPTED.update(saved)


def check_int16_ternary_forms():
    """The 8 three-operand 16-bit integer VOP3 forms count as covered once the translator accepts their fixtures; the
    nearby VOP3 forms that stay rejected (v_mad_u32_u16, v_mad_i32_i16, v_sad_u16, f16 ops) stay uncovered."""
    import pixel_aot
    import pixel_aot_check as pixel
    saved = set(pixel.ACCEPTED)
    pixel.ACCEPTED.clear()
    try:
        for group in range(3):
            pixel.translate_int(pixel.int16_ternary_ops(group))
        forms = {("VOP3", f"v_{name}") for name in pixel_aot.VOP3_INT16_TERNARY}
        assert len(forms) == 8
        neighbours = {("VOP3", "v_mad_u32_u16"), ("VOP3", "v_mad_i32_i16"), ("VOP3", "v_sad_u16"),
                      ("VOP3", "v_min_f16_e64")}
        for name in ("gfx1013.tsv", "rdna2.tsv"):
            reference = fc.load_reference(name)
            assert forms <= reference, (name, forms - reference)
            assert neighbours <= reference, (name, neighbours - reference)
            covered = fc.gfx_forms(pixel.ACCEPTED, reference)
            assert forms <= covered, (name, forms - covered)
            assert not neighbours & covered, (name, neighbours & covered)
        # With only the unrelated two-operand fixtures accepted the ternary forms are not covered.
        pixel.ACCEPTED.clear()
        pixel.translate_int(pixel.int16_vop3_ops(0))
        reference = fc.load_reference("gfx1013.tsv")
        assert not forms & fc.gfx_forms(pixel.ACCEPTED, reference)
    finally:
        pixel.ACCEPTED.clear()
        pixel.ACCEPTED.update(saved)


def check_sopk_wait_forms():
    """The SOPK wait forms count once a program using them lowers; a refused SGPR operand records nothing."""
    import pixel_aot_check as pixel
    forms = {("SOPK", "s_waitcnt_vmcnt"), ("SOPK", "s_waitcnt_lgkmcnt"), ("SOPK", "s_waitcnt_vscnt")}
    program = pixel.sopk_wait_program()
    pixel.ACCEPTED.clear()
    pixel.must_reject(pixel.translate, [line.replace("null", "s5") for line in program])
    assert not pixel.ACCEPTED
    pixel.translate(program)
    for name in ("gfx1013.tsv", "rdna2.tsv"):
        reference = fc.load_reference(name)
        assert forms <= reference and ("SOPP", "s_waitcnt_depctr") in reference, name
        covered = fc.gfx_forms(pixel.ACCEPTED, reference)
        assert forms <= covered, (name, forms - covered)
        assert ("SOPP", "s_waitcnt_depctr") not in covered and ("SOPK", "s_waitcnt_expcnt") not in covered, name


def check_s_version_form():
    """s_version counts once a program using it lowers; a refused operand records nothing."""
    import pixel_aot_check as pixel
    form = ("SOPK", "s_version")
    program = pixel.s_version_program()
    pixel.ACCEPTED.clear()
    pixel.must_reject(pixel.translate_int, [line.replace("0x1004", "0x10000") for line in program])
    assert "s_version" not in pixel.ACCEPTED
    pixel.translate_int(program)
    for name in ("gfx1013.tsv", "rdna2.tsv"):
        reference = fc.load_reference(name)
        assert form in reference, name
        assert form in fc.gfx_forms(pixel.ACCEPTED, reference), name
    pixel.ACCEPTED.clear()


def check_parsing():
    with tempfile.TemporaryDirectory() as tmp:
        junit = Path(tmp) / "results.xml"
        junit.write_text(
            '<testsuite time="32" timestamp="2026-11-21T21:05:21">'
            '<testcase name="a" status="run"/>'
            '<testcase name="b" status="fail"/>'
            '<testcase name="c" status="notrun"/>'
            '<testcase name="d" status="run"><failure message="mismatch"/></testcase>'
            '<testcase name="e" status="run"><error message="crash"/></testcase>'
            '<testcase name="f" status="run"><skipped/></testcase>'
            '</testsuite>',
            encoding="utf-8",
        )
        results, window = fc.read_results(junit)
        assert results == {"a": "pass", "b": "fail", "c": "skip", "d": "fail", "e": "fail", "f": "skip"}, results
        start = datetime(2026, 11, 21, 21, 5, 21).timestamp()  # CTest writes local time
        assert window == (start - 1, start + 34), window
    asm = ("emit MACRO dst\nLOCAL skip\n    lock cmpxchg [rdx], eax\n    rep movsb\nskip:\nENDM\n"
           "x PROC FRAME\n    .endprolog\nIFDEF A\n    jae short l\nELSE\n    jnz short l\nENDIF\n"
           "l:  mov DWORD PTR [rdx], eax ; c\ng:: ud2\n    emit rax\nvalue DQ 0\n    dd 1\n    ret\nx ENDP\nEND\n")
    assert fc.masm_mnemonics(asm) == {"cmpxchg", "movsb", "jae", "jne", "mov", "ud2", "ret"}, fc.masm_mnemonics(asm)
    with tempfile.TemporaryDirectory() as tmp:
        isolated = fc.measure({}, sample_repo(tmp, {"branch": asm}))["x64_zen2"]
        assert isolated["fixture_forms_not_in_reference"] == [], isolated["fixture_forms_not_in_reference"]
    # The real repository measures without error, with no macro names or LOCAL leaking into unmatched forms.
    full = fc.measure({})
    assert "not a measure of port completion" in fc.report(full)
    assert full["x64_zen2"]["covered"] == full["gfx1013"]["covered"] == 0
    unmatched = full["x64_zen2"]["fixture_forms_not_in_reference"]
    assert {"delta", "local", "plane_request"}.isdisjoint(unmatched), unmatched


def check_junit_errors():
    """An unusable JUnit file is reported as a one-line diagnostic with exit status 2, not a traceback.
    The record window needs one CTest run, so a merged <testsuites> file is refused rather than guessed."""
    import subprocess
    tool = Path(fc.__file__)
    valid = '<testsuite time="1" timestamp="2026-11-21T21:05:21"><testcase name="a" status="run"/></testsuite>'
    cases = {
        "missing.xml": (None, "No such file"),
        "malformed.xml": ("not xml", "syntax error: line 1, column 0"),
        "merged.xml": (f"<testsuites>{valid}</testsuites>", "expected one CTest <testsuite> root, found <testsuites>"),
        "no_timestamp.xml": ('<testsuite time="1"><testcase name="a" status="run"/></testsuite>',
                             "timestamp"),
        "bad_time.xml": ('<testsuite time="soon" timestamp="2026-11-21T21:05:21"/>', "time"),
        "valid.xml": (valid, None),
    }
    cases.update({
        f"time_{value}.xml": (valid.replace('time="1"', f'time="{value}"'), "finite and nonnegative")
        for value in ("nan", "inf", "-1")
    })
    with tempfile.TemporaryDirectory() as tmp:
        for name, (text, expected) in cases.items():
            junit = Path(tmp) / name
            if text is not None:
                junit.write_text(text, encoding="utf-8")
            if expected and text is not None and name != "malformed.xml":
                try:
                    fc.read_results(junit)
                except ValueError as error:
                    assert expected in str(error), (name, error)
                else:
                    raise AssertionError(f"{name}: read_results accepted an unusable JUnit file")
            run = subprocess.run([sys.executable, "-B", str(tool), str(junit), "--records", tmp],
                                 capture_output=True, text=True)
            if expected is None:
                assert run.returncode == 0, (name, run.returncode, run.stderr)
                assert "Tests: 1 passed, 0 failed, 0 skipped." in run.stdout, run.stdout
                continue
            assert run.returncode == 2, (name, run.returncode, run.stderr)
            assert "Traceback" not in run.stderr, (name, run.stderr)
            assert run.stderr.startswith(f"form_coverage: {junit}: "), (name, run.stderr)
            assert expected in run.stderr, (name, run.stderr)


def check_gaps():
    reference = {("VOP1", "v_a"), ("VOP3", "v_a_e64"), ("VOP3", "v_b_e64"), ("VOP1", "v_c")}
    fixtures = {"passing": ({"v_a_e32"}, "pass"),
                "skipping": ({"v_a_e32", "v_a_e64"}, "skip"),
                "failing": ({"v_a_e64", "v_b_e64"}, "fail")}
    measured = fc.summarize(reference, fixtures, fc.gfx_forms, include_gaps=True)
    assert measured.pop("gaps") == [
        ("VOP1", "v_c", "unobserved"),
        ("VOP3", "v_a_e64", "skipped"),
        ("VOP3", "v_b_e64", "observed_without_pass"),
    ]
    assert measured == fc.summarize(reference, fixtures, fc.gfx_forms)


def check_dashboard():
    """tools/coverage_svg.py draws one square per reference form, coloured by its measured state."""
    import coverage_svg as svg
    with tempfile.TemporaryDirectory() as tmp:
        root = sample_repo(tmp, {"branch": BRANCH, "switch": SWITCH})
        m = fc.measure({"cpu_leaf_aot_branches": "pass", "cpu_aot_jump_tables": "skip"}, root)
    drawn = svg.render(m, "a test run")
    assert drawn == svg.render(m, "a test run") and fc.measure({})["note"] == svg.NOTE in drawn
    assert "%" not in drawn  # counts only: coverage is not a completion percentage
    squares = {svg.COVERED: 0, svg.HELD: 0, svg.OPEN: 0}
    for g in ET.fromstring(drawn).iter("{http://www.w3.org/2000/svg}g"):
        if g.get("fill") in squares:
            squares[g.get("fill")] += len(g)
    held = fc.masm_mnemonics(SWITCH) - fc.masm_mnemonics(BRANCH)
    total = sum(m[key]["total"] for key, _, _ in svg.AREAS)
    assert squares == {svg.COVERED: 8, svg.HELD: len(held), svg.OPEN: total - 8 - len(held)}, squares
    # Every group, down to the one-form EXP box, keeps its squares and label inside its own box, and is
    # labelled there or in the panel's "Too small to label" line.
    root = ET.fromstring(drawn)
    ns = "{http://www.w3.org/2000/svg}"
    boxes = [box for box in root.iter(f"{ns}svg") if box is not root]
    groups = [(name, g) for key, _, _ in svg.AREAS for name, g in m[key]["groups"].items() if g["total"]]
    assert len(boxes) == len(groups), (len(boxes), len(groups))
    labels = []
    for box in boxes:
        w, h = float(box.get("width")), float(box.get("height"))
        for cell in box.iter(f"{ns}rect"):
            right = float(cell.get("x", 0)) + float(cell.get("width"))
            bottom = float(cell.get("y", 0)) + float(cell.get("height"))
            assert right <= w + 0.01 and bottom <= h + 0.01, (box.attrib, cell.attrib)
        for label in box.iter(f"{ns}text"):
            assert len(label.text) * svg.LABEL_CHAR + 4 <= w and float(label.get("y")) <= h, (box.attrib, label.text)
            labels.append(label.text)
    footnotes = "".join(t.text for t in root.iter(f"{ns}text") if t.text.startswith("Too small to label: "))
    for name, g in groups:
        entry = f"{name} {g['covered']}/{g['total']}"
        assert entry in labels or entry in footnotes, entry
    assert "EXP 0/1" in footnotes  # the smallest group cannot hold its label



def check_cpu_records():
    forms = {prefix + cc for prefix in ('j', 'set', 'cmov') for cc in ('o', 'no', 'p', 'np')}
    test = 'cpu_aot_flag_conditions'
    with tempfile.TemporaryDirectory() as tmp:
        # The macro body must not supply guessed forms, even when the test passes.
        root = sample_repo(tmp, {'flag_conditions': 'set&cc al\noperation eax, edx\n'})
        missing = fc.measure({test: 'pass'}, root)
        assert missing['x64_zen2']['covered'] == 0
        passed = fc.measure({test: 'pass'}, root, {test: forms})
        assert passed['x64_zen2']['covered'] == 12
        assert passed['x64_zen2']['fixture_forms_not_in_reference'] == []
        assert passed['shader_records'] == []
        assert passed['gfx1013']['covered'] == passed['gfx1030']['covered'] == 0
        assert passed['unclassified'] == []
        for state in ('fail', 'skip', 'absent'):
            results = {} if state == 'absent' else {test: state}
            report = fc.measure(results, root, {test: forms})
            assert report['x64_zen2']['covered'] == 0
            assert len(report['x64_zen2']['blocked_by_skip']) == (12 if state == 'skip' else 0)
def check_vector_bit_forms():
    import pixel_aot_check as pixel
    recorded = set(pixel.ACCEPTED)
    pixel.check_bit_controls()
    assert pixel.ACCEPTED == recorded, 'acceptance controls leaked forms into unrelated GPU records'
    unary = {'v_bfrev_b32', 'v_ffbh_u32', 'v_ffbh_i32', 'v_ffbl_b32'}
    vop3 = {'v_bfe_u32', 'v_bfe_i32', 'v_bfm_b32', 'v_bcnt_u32_b32'}
    expected = {('VOP1', op) for op in unary} | {('VOP3', op + '_e64') for op in unary}
    expected |= {('VOP3', op) for op in vop3}
    # Count actual accepted source programs, including both encodings of unary operations.
    forms = set()
    for op in pixel.BIT_OPERATIONS:
        program = pixel.bit_program(op)
        pixel.translate(program)
        forms.update(line.split()[0] for line in program)
    for name in ('gfx1013.tsv', 'rdna2.tsv'):
        selected = fc.gfx_forms(forms, fc.load_reference(name))
        assert expected <= selected and len(expected) == 12
        assert not any(group in {'SDWA', 'DPP8', 'DPP16'} for group, _ in selected)
    with tempfile.TemporaryDirectory() as tmp:
        root = sample_repo(tmp, {})
        for state, count in (('pass', 12), ('fail', 0), ('skip', 0)):
            m = fc.measure({'pixel_aot_bits_warp': state}, root,
                           {'pixel_aot_bits_warp': set(pixel.BIT_OPERATIONS)})
            assert m['gfx1013']['covered'] == m['gfx1030']['covered'] == count


def check_avx2_regs():
    """avx2_regs.asm covers all 28 AVX2 forms in docs/coverage/x64_zen2.tsv group AVX2."""
    expected_avx2 = {
        "vbroadcasti128", "vextracti128", "vinserti128", "vperm2i128",
        "vpbroadcastb", "vpbroadcastw", "vpbroadcastd", "vpbroadcastq",
        "vpermd", "vpermpd", "vpermps", "vpermq",
        "vpblendd",
        "vpsllvd", "vpsllvq", "vpsravd", "vpsrlvd", "vpsrlvq",
        "vgatherdpd", "vgatherdps", "vgatherqpd", "vgatherqps",
        "vpgatherdd", "vpgatherdq", "vpgatherqd", "vpgatherqq",
        "vpmaskmovd", "vpmaskmovq",
    }
    reference = fc.load_reference("x64_zen2.tsv")
    all_avx2 = {form for group, form in reference if group == "AVX2"}
    assert expected_avx2 == all_avx2 and len(all_avx2) == 28

    asm_text = (fc.ROOT / "tests" / "cpu" / "avx2_regs.asm").read_text(encoding="utf-8")
    found = fc.masm_mnemonics(asm_text)
    assert expected_avx2 <= found, expected_avx2 - found

    passed = fc.measure({"cpu_aot_avx2_regs": "pass"})
    assert passed["fixtures"]["avx2_regs.asm"] == "pass"
    assert "avx2_regs.asm" not in passed["unclassified"]
    avx2_group = passed["x64_zen2"]["groups"]["AVX2"]
    assert avx2_group == {"covered": 28, "total": 28}, avx2_group
    with tempfile.TemporaryDirectory() as tmp:
        root = sample_repo(tmp, {"avx2_regs": asm_text})
        only_pass = fc.measure({"cpu_aot_avx2_regs": "pass"}, root)["x64_zen2"]
        only_skip = fc.measure({"cpu_aot_avx2_regs": "skip"}, root)["x64_zen2"]
        only_fail = fc.measure({"cpu_aot_avx2_regs": "fail"}, root)["x64_zen2"]
        assert only_pass["groups"]["AVX2"] == {"covered": 28, "total": 28}
        assert only_pass["fixture_forms_not_in_reference"] == []
        assert only_skip["groups"]["AVX2"] == {"covered": 0, "total": 28}
        for form in expected_avx2:
            assert f"AVX2 {form}" in only_skip["blocked_by_skip"]
        assert only_fail["groups"]["AVX2"] == {"covered": 0, "total": 28}
        assert only_fail["blocked_by_skip"] == []

def check_avx_int_regs():
    """avx_int_regs.asm covers 90 surviving AVX packed-integer forms in docs/coverage/x64_zen2.tsv group AVX."""
    expected_avx_int = {
        "vmpsadbw",
        "vpabsb", "vpabsd", "vpabsw",
        "vpackssdw", "vpacksswb", "vpackusdw",
        "vpaddb", "vpaddq", "vpaddsb", "vpaddusb", "vpaddusw",
        "vpalignr",
        "vpblendvb", "vpblendw",
        "vpcmpeqb", "vpcmpeqd", "vpcmpeqq", "vpcmpeqw",
        "vpcmpgtb", "vpcmpgtd", "vpcmpgtq", "vpcmpgtw",
        "vpextrb", "vpextrd", "vpextrq", "vpextrw",
        "vphaddd", "vphaddsw", "vphaddw", "vphminposuw", "vphsubd", "vphsubsw", "vphsubw",
        "vpinsrb", "vpinsrd", "vpinsrq", "vpinsrw",
        "vpmaddwd",
        "vpmaxsb", "vpmaxsd", "vpmaxub", "vpmaxud", "vpmaxuw",
        "vpminsb", "vpminsd", "vpminub", "vpminud", "vpminuw",
        "vpmovmskb",
        "vpmovsxbd", "vpmovsxbq", "vpmovsxbw", "vpmovsxdq", "vpmovsxwd", "vpmovsxwq",
        "vpmovzxbd", "vpmovzxbq", "vpmovzxbw", "vpmovzxdq", "vpmovzxwd", "vpmovzxwq",
        "vpmuldq", "vpmulhrsw", "vpmulhuw", "vpmulld", "vpmuludq",
        "vpsadbw",
        "vpshufhw", "vpshuflw",
        "vpsignb", "vpsignd", "vpsignw",
        "vpslld", "vpsllq", "vpsrld", "vpsrlq",
        "vpsubb", "vpsubd", "vpsubq", "vpsubsb", "vpsubsw", "vpsubusb", "vpsubusw",
        "vpunpckhdq", "vpunpckhqdq", "vpunpckldq", "vpunpcklqdq",
        "vzeroall", "vzeroupper",
    }
    carved_avx_int = {
        "vpackuswb", "vpaddd", "vpaddw", "vpand", "vpor", "vpxor", "vpavgw",
        "vpmaxsw", "vpminsw", "vpmulhw", "vpsllw", "vpsraw", "vpsrlw", "vpsubw",
        "vpunpckhbw", "vpunpckhwd", "vpunpcklbw", "vpunpcklwd", "vpavgb",
        "vpshufb", "vpmaddubsw", "vpaddsw", "vpmullw", "vpandn", "vpsrad",
        "vpsrldq", "vpslldq", "vpshufd",
    }
    assert len(expected_avx_int) == 90 and len(carved_avx_int) == 28
    assert expected_avx_int.isdisjoint(carved_avx_int)
    reference = fc.load_reference("x64_zen2.tsv")
    all_avx = {form for group, form in reference if group == "AVX"}
    assert expected_avx_int <= all_avx, expected_avx_int - all_avx

    asm_text = (fc.ROOT / "tests" / "cpu" / "avx_int_regs.asm").read_text(encoding="utf-8")
    found = fc.masm_mnemonics(asm_text)
    assert expected_avx_int <= found, expected_avx_int - found
    assert carved_avx_int.isdisjoint(found), carved_avx_int & found
    # avx_int_regs.asm covers the 90 surviving AVX packed-integer forms plus transport vmovdqu (91 AVX total).
    expected_avx_total = expected_avx_int | {"vmovdqu"}
    assert (found & all_avx) == expected_avx_total, (found & all_avx) ^ expected_avx_total

    passed = fc.measure({"cpu_aot_avx_int_regs": "pass"})
    assert passed["fixtures"]["avx_int_regs.asm"] == "pass"
    assert "avx_int_regs.asm" not in passed["unclassified"]
    assert passed["x64_zen2"]["groups"]["AVX"]["covered"] == len(expected_avx_total), passed["x64_zen2"]["groups"]["AVX"]
    with tempfile.TemporaryDirectory() as tmp:
        root = sample_repo(tmp, {"avx_int_regs": asm_text})
        only_pass = fc.measure({"cpu_aot_avx_int_regs": "pass"}, root)["x64_zen2"]
        only_skip = fc.measure({"cpu_aot_avx_int_regs": "skip"}, root)["x64_zen2"]
        only_fail = fc.measure({"cpu_aot_avx_int_regs": "fail"}, root)["x64_zen2"]
        assert only_pass["groups"]["AVX"]["covered"] == len(expected_avx_total)
        assert only_pass["fixture_forms_not_in_reference"] == []
        assert only_skip["groups"]["AVX"]["covered"] == 0
        for form in expected_avx_int:
            assert f"AVX {form}" in only_skip["blocked_by_skip"]
        assert only_fail["groups"]["AVX"]["covered"] == 0
        assert only_fail["blocked_by_skip"] == []


def check_avx_move_regs():
    """avx_move_regs.asm covers 50 surviving AVX move/shuffle/blend/logical forms in docs/coverage/x64_zen2.tsv group AVX."""
    expected_avx_move = {
        "vandnpd", "vandnps", "vandpd", "vandps",
        "vblendpd", "vblendps", "vblendvpd", "vblendvps",
        "vbroadcastf128", "vbroadcastsd", "vbroadcastss",
        "vextractf128", "vextractps", "vinsertf128", "vinsertps",
        "vlddqu", "vmaskmovpd", "vmaskmovps",
        "vmovapd", "vmovddup",
        "vmovhlps", "vmovhpd", "vmovhps", "vmovlhps", "vmovlpd", "vmovlps",
        "vmovmskpd", "vmovmskps",
        "vmovntdq", "vmovntdqa", "vmovntpd", "vmovntps",
        "vmovsd", "vmovshdup", "vmovsldup", "vmovss", "vmovupd",
        "vorpd", "vorps",
        "vperm2f128", "vpermilpd", "vpermilps",
        "vshufpd", "vshufps",
        "vunpckhpd", "vunpckhps", "vunpcklpd", "vunpcklps",
        "vxorpd", "vxorps",
    }
    carved_avx_move_rejected = {"vmovdqa", "vmovups", "vmovaps", "vmovq", "vmovd"}
    assert len(expected_avx_move) == 50
    assert expected_avx_move.isdisjoint(carved_avx_move_rejected | {"vmovdqu"})
    reference = fc.load_reference("x64_zen2.tsv")
    all_avx = {form for group, form in reference if group == "AVX"}
    assert expected_avx_move <= all_avx, expected_avx_move - all_avx

    asm_text = (fc.ROOT / "tests" / "cpu" / "avx_move_regs.asm").read_text(encoding="utf-8")
    found = fc.masm_mnemonics(asm_text)
    assert expected_avx_move <= found, expected_avx_move - found
    assert carved_avx_move_rejected.isdisjoint(found), carved_avx_move_rejected & found
    # 50 surviving AVX move/shuffle/blend/logical forms plus vmovdqu and vzeroupper ABI glue (52 AVX total).
    expected_avx_total = expected_avx_move | {"vmovdqu", "vzeroupper"}
    assert (found & all_avx) == expected_avx_total, (found & all_avx) ^ expected_avx_total

    passed = fc.measure({"cpu_aot_avx_move_regs": "pass"})
    assert passed["fixtures"]["avx_move_regs.asm"] == "pass"
    assert "avx_move_regs.asm" not in passed["unclassified"]
    avx_group = passed["x64_zen2"]["groups"]["AVX"]
    assert avx_group["covered"] == len(expected_avx_total), avx_group
    with tempfile.TemporaryDirectory() as tmp:
        root = sample_repo(tmp, {"avx_move_regs": asm_text})
        only_pass = fc.measure({"cpu_aot_avx_move_regs": "pass"}, root)["x64_zen2"]
        only_skip = fc.measure({"cpu_aot_avx_move_regs": "skip"}, root)["x64_zen2"]
        only_fail = fc.measure({"cpu_aot_avx_move_regs": "fail"}, root)["x64_zen2"]
        assert only_pass["groups"]["AVX"]["covered"] == len(expected_avx_total)
        assert only_pass["fixture_forms_not_in_reference"] == []
        assert only_skip["groups"]["AVX"]["covered"] == 0
        for form in expected_avx_move:
            assert f"AVX {form}" in only_skip["blocked_by_skip"]
        assert only_fail["groups"]["AVX"]["covered"] == 0
        assert only_fail["blocked_by_skip"] == []


def check_sse2_regs():
    """sse2_regs.asm covers 23 SSE2 forms in docs/coverage/x64_zen2.tsv group SSE2."""
    expected_sse2 = {
        "cmppd",
        "cvtdq2pd", "cvtdq2ps", "cvtpd2dq", "cvtpd2ps", "cvtpi2pd",
        "cvtps2dq", "cvtps2pd", "cvtsd2si",
        "cvttpd2dq", "cvttps2dq",
        "lfence", "mfence",
        "maxsd", "minsd", "sqrtsd",
        "movapd", "movdqa", "movhpd", "movlpd",
        "movntdq", "movnti", "movntpd",
    }
    rejected_sse2 = {
        # ABI-infeasible task forms: MMX register operands or implicit RDI addressing.
        "cvtpd2pi", "cvttpd2pi", "movdq2q", "movq2dq", "maskmovdqu",
        # Reserved for sibling translators.
        "addpd", "addsd", "andnpd", "andpd", "comisd",
        "cvtsi2sd", "cvttsd2si", "cvtss2sd", "cvtsd2ss",
        "divpd", "divsd", "maxpd", "minpd", "movdqu", "movmskpd", "movupd",
        "mulpd", "mulsd", "orpd",
        "pmuludq", "pshufd", "pshufhw", "pshuflw", "pslldq", "psrldq",
        "psubq", "punpckhqdq", "punpcklqdq",
        "shufpd", "sqrtpd", "subpd", "subsd",
        "ucomisd", "unpckhpd", "unpcklpd", "xorpd",
    }
    assert len(expected_sse2) == 23 and len(rejected_sse2) == 41
    reference = fc.load_reference("x64_zen2.tsv")
    all_sse2 = {form for group, form in reference if group == "SSE2"}
    assert expected_sse2 | rejected_sse2 == all_sse2 and len(all_sse2) == 64

    asm_text = (fc.ROOT / "tests" / "cpu" / "sse2_regs.asm").read_text(encoding="utf-8")
    found = fc.masm_mnemonics(asm_text)
    assert expected_sse2 <= found, expected_sse2 - found
    assert rejected_sse2.isdisjoint(found), rejected_sse2 & found
    # Exactly the 23 implemented SSE2 forms appear; rejected and reserved forms stay out.
    assert (found & all_sse2) == expected_sse2, (found & all_sse2) ^ expected_sse2

    passed = fc.measure({"cpu_aot_sse2_regs": "pass"})
    assert passed["fixtures"]["sse2_regs.asm"] == "pass"
    assert "sse2_regs.asm" not in passed["unclassified"]
    assert passed["x64_zen2"]["groups"]["SSE2"]["covered"] == len(expected_sse2)
    with tempfile.TemporaryDirectory() as tmp:
        root = sample_repo(tmp, {"sse2_regs": asm_text})
        only_pass = fc.measure({"cpu_aot_sse2_regs": "pass"}, root)["x64_zen2"]
        only_skip = fc.measure({"cpu_aot_sse2_regs": "skip"}, root)["x64_zen2"]
        only_fail = fc.measure({"cpu_aot_sse2_regs": "fail"}, root)["x64_zen2"]
        assert only_pass["groups"]["SSE2"]["covered"] == len(expected_sse2)
        assert only_pass["fixture_forms_not_in_reference"] == []
        assert only_skip["groups"]["SSE2"]["covered"] == 0
        for form in expected_sse2:
            assert f"SSE2 {form}" in only_skip["blocked_by_skip"]
        assert only_fail["groups"]["SSE2"]["covered"] == 0
        assert only_fail["blocked_by_skip"] == []


def check_fma3_leaf():
    """fma3_leaf.asm covers all 60 FMA forms in docs/coverage/x64_zen2.tsv group FMA."""
    reference = fc.load_reference("x64_zen2.tsv")
    all_fma = {form for group, form in reference if group == "FMA"}
    assert len(all_fma) == 60, len(all_fma)

    asm_text = (fc.ROOT / "tests" / "cpu" / "fma3_leaf.asm").read_text(encoding="utf-8")
    found = fc.masm_mnemonics(asm_text)
    assert all_fma <= found, all_fma - found

    passed = fc.measure({"cpu_aot_fma3": "pass"})
    assert passed["fixtures"]["fma3_leaf.asm"] == "pass"
    assert "fma3_leaf.asm" not in passed["unclassified"]
    assert passed["x64_zen2"]["groups"]["FMA"] == {"covered": 60, "total": 60}
    with tempfile.TemporaryDirectory() as tmp:
        root = sample_repo(tmp, {"fma3_leaf": asm_text})
        only_pass = fc.measure({"cpu_aot_fma3": "pass"}, root)["x64_zen2"]
        only_skip = fc.measure({"cpu_aot_fma3": "skip"}, root)["x64_zen2"]
        only_fail = fc.measure({"cpu_aot_fma3": "fail"}, root)["x64_zen2"]
        assert only_pass["groups"]["FMA"] == {"covered": 60, "total": 60}
        assert only_pass["fixture_forms_not_in_reference"] == []
        assert only_skip["groups"]["FMA"] == {"covered": 0, "total": 60}
        for form in all_fma:
            assert f"FMA {form}" in only_skip["blocked_by_skip"]
        assert only_fail["groups"]["FMA"] == {"covered": 0, "total": 60}
        assert only_fail["blocked_by_skip"] == []


def check_swap_coverage():
    """v_swap_b32 is VOP1 in both references and counts as covered only through the accepted spelling.
    The _e64, DPP and SDWA spellings reach only rejections, and the neighbouring v_swaprel_b32 stays uncovered."""
    import pixel_aot_check as pixel
    pixel.ACCEPTED.clear()
    with tempfile.TemporaryDirectory() as tmp:
        pixel.check_controls(Path(tmp))
    assert "v_swap_b32" in pixel.ACCEPTED
    assert not {"v_swap_b32_e64", "v_swap_b32_dpp", "v_swap_b32_sdwa", "v_swaprel_b32"} & pixel.ACCEPTED
    for name in ("gfx1013.tsv", "rdna2.tsv"):
        reference = fc.load_reference(name)
        assert ("VOP1", "v_swap_b32") in reference, name
        covered = fc.gfx_forms(pixel.ACCEPTED, reference)
        assert ("VOP1", "v_swap_b32") in covered, name
        assert ("VOP1", "v_swaprel_b32") not in covered, name
        # Control: without the accepted word the form is not covered.
        assert ("VOP1", "v_swap_b32") not in fc.gfx_forms(pixel.ACCEPTED - {"v_swap_b32"}, reference), name


if __name__ == "__main__":
    check_denominators()
    check_example()
    check_jump_tables()
    check_scalar_float()
    check_unclassified()
    check_local_cpu_registration()
    check_cpu_records()
    check_shader_records()
    check_vector_bit_forms()
    check_scalar_brev()
    check_plain_bitscan_forms()
    check_int16_vop3_forms()
    check_int16_ternary_forms()
    check_f16_arith_forms()
    check_swap_coverage()
    check_sopk_wait_forms()
    check_s_version_form()
    check_parsing()
    check_junit_errors()
    check_gaps()
    check_dashboard()
    check_avx2_regs()
    check_avx_int_regs()
    check_avx_move_regs()
    check_sse2_regs()
    check_fma3_leaf()
    print("form coverage checks passed")
