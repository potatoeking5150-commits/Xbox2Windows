"""Measure DXC wall time on generated mixed-sampler compute shaders.

The generated sources follow the shape of the source-built sampler fixture
(tests/spirv_window/sampler.py through the named SPIR-V bridge): one Texture2D,
one SamplerState[2] view, one SamplerComparisonState[2] view and one raw
RWByteAddressBuffer. The declarations and the two control stores are copied
verbatim, so words 0-3 of the UAV are identical in every variant and the
existing `offscreen_check --spirv-sampler-views` oracle applies to any variant
that compiles. That oracle is not run here; see docs/experiments.md.

Families, each changing one property:
  control          the emitted fixture source unchanged (two stores)
  samples_N        N additional independent SampleLevel stores, flat control flow
  branches_B       B additional if/else regions with observable stores
  samplebranch_N   one SampleLevel inside each of N divergent regions

The tool needs only the Python standard library and an external DXC executable.
It runs a fixed command (target cs_6_6, entry main, native 16-bit types allowed,
default optimization), records HLSL bytes, median wall time and peak process
working set, and classifies each result as DXIL, timeout or diagnostic. With
--ftime-report it also records the dominant LLVM/DXIL passes and their share of
the pass time. --out writes a compact CSV, or JSON when the path ends in .json.

Limits:
- The generated sources keep the fixture's declarations and control stores but
  not everything the bridge can emit (loops, address windows, further
  resources). They are a synthetic family for timing the compiler, not a
  replay of the archive.
- The archive-replay attribution stays open: this tool does not read the
  archive and cannot say which real record, with which HLSL size and pass
  totals, hits the timeout.
- No GPU is involved. Compiling a CSO and dispatching it are separate checks;
  a fast compile does not establish equivalent GPU behavior.
- The timeout is local policy supplied with --timeout; the tool has no built-in
  limit or machine threshold. It is not used by CI. Wall times are
  machine-dependent, so compare them on one machine and report the
  adapter/toolchain with the record.
"""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import threading
import time

MIB = 1048576.0
DEFAULT_TARGET = "cs_6_6"
DEFAULT_ENTRY = "main"
DEFAULT_VARIANTS = "control,samples_1024,branches_1024,samplebranch_4096"
MAX_SIZE = 65536
PASS_LIMIT = 5

# Copied from the HLSL that tests/spirv_window/sampler.py emits through
# SPIR-V Cross, with LF line ends. The blank line after the declarations is
# part of the emitted fixture source.
FIXTURE_HEADER = """RWByteAddressBuffer _4 : register(u0, space0);
Texture2D<float4> _2 : register(t0, space0);
SamplerComparisonState _6[2] : register(s1, space3);
SamplerState _3[2] : register(s1, space0);

static uint3 gl_WorkGroupID;
struct SPIRV_Cross_Input
{
    uint3 gl_WorkGroupID : SV_GroupID;
};

"""

# The original samples/branches family generator omitted that one blank line.
# Keeping the difference reproduces the byte sizes recorded for #775; the
# declarations and the two control stores are still identical everywhere.
FAMILY_HEADER = FIXTURE_HEADER[:-1]

CONTROL_STORES = """    _4.Store(gl_WorkGroupID.x * 4 + 0, asuint(_2.SampleLevel(_3[gl_WorkGroupID.x], float2(1.25f, 0.5f), 0.0f).x));
    _4.Store((gl_WorkGroupID.x + 2u) * 4 + 0, asuint(_2.SampleCmpLevelZero(_6[gl_WorkGroupID.x], float2(1.25f, 0.5f), 0.5f)));
"""

EPILOGUE = """
[numthreads(1, 1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_WorkGroupID = stage_input.gl_WorkGroupID;
    comp_main();
}
"""

SAMPLE_SIZES = (1, 4, 16, 64, 256, 1024, 4096, 16384)
BRANCH_SIZES = (1, 4, 16, 64, 256, 1024, 4096, 16384)
SAMPLEBRANCH_SIZES = (64, 256, 1024, 2048, 4096, 6144, 8192, 12288, 16384)

VARIANT_RE = re.compile(r"(samples|branches|samplebranch)_([1-9][0-9]*)")
FTIME_RE = re.compile(
    r"^\s*;\s+([\d.]+)\s+\(\s*[\d.]+%\)\s+([\d.]+)\s+\(\s*[\d.]+%\)\s+"
    r"([\d.]+)\s+\(\s*[\d.]+%\)\s+([\d.]+)\s+\(\s*[\d.]+%\)\s+(.+?)\s*$")

CSV_FIELDS = (
    "variant", "family", "size", "hlsl_bytes", "dxil_bytes", "reps",
    "wall_median_s", "wall_min_s", "wall_max_s", "peak_median_mib", "outcome",
    "exit_codes", "first_diagnostic", "pass_total_s", "top_passes",
)


def _wrap(header, body):
    return header + "void comp_main()\n{\n" + body + "}\n" + EPILOGUE


def control_source():
    """The emitted fixture source, unchanged."""
    return _wrap(FIXTURE_HEADER, CONTROL_STORES)


def samples_source(n):
    """One ordinary SampleLevel store per repetition, appended to the control stores."""
    lines = []
    for k in range(n):
        uv_x = 0.25 + (k % 64) * 0.0078125
        uv_y = 0.5 + (k // 64) * 0.0078125
        offset = 16 + 4 * k  # bytes; the control stores own bytes 0..15
        lines.append(
            f"    _4.Store({offset}u, asuint(_2.SampleLevel(_3[gl_WorkGroupID.x], "
            f"float2({uv_x}f, {uv_y}f), 0.0f).x));\n")
    return _wrap(FAMILY_HEADER, CONTROL_STORES + "".join(lines))


def branches_source(b):
    """A chain of if/else regions with observable stores; sample count fixed."""
    lines = []
    for k in range(b):
        cond = (k % 7) + 1
        low = 65536 + 8 * k
        lines.append(
            f"    if (((gl_WorkGroupID.x + {k}u) & {cond}u) != 0u)\n"
            f"    {{\n"
            f"        _4.Store({low}u, {k}u);\n"
            f"    }}\n"
            f"    else\n"
            f"    {{\n"
            f"        _4.Store({low + 4}u, {k + 1}u);\n"
            f"    }}\n")
    return _wrap(FAMILY_HEADER, "".join(lines) + CONTROL_STORES)


def samplebranch_source(n):
    """One sample inside each divergent region, as in the blocked CS records."""
    lines = []
    for k in range(n):
        low = 65536 + 8 * k
        sample = (f"_2.SampleLevel(_3[gl_WorkGroupID.x], "
                  f"float2(0.25f, 0.5f + {k % 64}.0f * 0.0078125f), 0.0f)")
        lines.append(
            f"    if (((gl_WorkGroupID.x + {k}u) & 1u) != 0u)\n"
            f"    {{\n"
            f"        _4.Store({low}u, asuint({sample}.x));\n"
            f"    }}\n"
            f"    else\n"
            f"    {{\n"
            f"        _4.Store({low + 4}u, {k}u);\n"
            f"    }}\n")
    return _wrap(FIXTURE_HEADER, "".join(lines) + CONTROL_STORES)


def variant_info(name):
    """Return (family, size, source) for a generated variant name."""
    if name == "control":
        return "control", 0, control_source()
    match = VARIANT_RE.fullmatch(name)
    if not match:
        raise ValueError(
            f"unknown variant {name!r}; use control, samples_N, branches_N or samplebranch_N")
    family, size = match.group(1), int(match.group(2))
    if size > MAX_SIZE:
        raise ValueError(f"variant size {size} exceeds the {MAX_SIZE} guard")
    source = {"samples": samples_source, "branches": branches_source,
              "samplebranch": samplebranch_source}[family](size)
    return family, size, source


def standard_variants():
    """Names of the recorded family steps, in report order."""
    return (["control"]
            + [f"samples_{size}" for size in SAMPLE_SIZES]
            + [f"branches_{size}" for size in BRANCH_SIZES]
            + [f"samplebranch_{size}" for size in SAMPLEBRANCH_SIZES])


if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    def open_process(pid):
        return kernel32.OpenProcess(0x0400 | 0x0010, False, pid) or None  # query + read

    def peak_working_set(handle):
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return None
        return counters.PeakWorkingSetSize

    def close_process(handle):
        kernel32.CloseHandle(handle)
else:
    open_process = peak_working_set = close_process = None


def _first_diagnostic(text):
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:400]
    return ""


def _classify(exit_code, timed_out, cso):
    if timed_out:
        return "timeout"
    if exit_code is None:
        return "killed"
    if exit_code < 0 or exit_code >= 0xC0000000:  # signal on POSIX, NTSTATUS on Windows
        return "crash"
    if exit_code != 0:
        return "diagnostic"
    if cso.exists() and cso.stat().st_size > 4:
        return "dxil"
    return "empty"


def run_once(dxc, hlsl, cso, flags, timeout_s, sample_interval=0.005):
    """Run DXC once and return one measurement. Compiler failures are recorded, not raised."""
    if cso.exists():
        cso.unlink()
    args = [str(dxc), *flags, "-Fo", str(cso), str(hlsl)]
    start = time.perf_counter()
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    handle = open_process(proc.pid) if open_process else None
    peak = 0
    stop = threading.Event()

    def monitor():
        nonlocal peak
        while not stop.is_set():
            if handle:
                value = peak_working_set(handle)
                if value and value > peak:
                    peak = value
            stop.wait(sample_interval)

    thread = threading.Thread(target=monitor, daemon=True) if handle else None
    if thread:
        thread.start()
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        stdout, stderr = proc.communicate()
    elapsed = time.perf_counter() - start
    stop.set()
    if thread:
        thread.join(timeout=1.0)
    if handle:
        value = peak_working_set(handle)
        if value and value > peak:
            peak = value
        close_process(handle)
    out_text = (stdout or b"").decode("utf-8", "replace")
    err_text = (stderr or b"").decode("utf-8", "replace")
    exit_code = proc.returncode
    return {
        "args": args,
        "wall_s": elapsed,
        "peak_bytes": peak or None,
        "exit_code": exit_code,
        "outcome": _classify(exit_code, timed_out, cso),
        "cso_bytes": cso.stat().st_size if cso.exists() else 0,
        "first_diagnostic": _first_diagnostic(err_text),
        "output": out_text + err_text,
    }


def parse_ftime(text):
    """Return (per-pass rows, Total row) from a DXC -ftime-report dump."""
    rows = []
    total = None
    for line in text.splitlines():
        match = FTIME_RE.match(line)
        if not match:
            continue
        _user, _system, combined, wall, name = match.groups()
        row = {"name": name, "combined_s": float(combined), "wall_s": float(wall)}
        if name == "Total":
            total = row
        else:
            rows.append(row)
    return rows, total


def dominant_passes(rows, total, limit=PASS_LIMIT):
    """Aggregate repeated pass names and list the largest by combined time."""
    aggregated = {}
    for row in rows:
        entry = aggregated.setdefault(
            row["name"], {"name": row["name"], "instances": 0, "combined_s": 0.0, "wall_s": 0.0})
        entry["instances"] += 1
        entry["combined_s"] += row["combined_s"]
        entry["wall_s"] += row["wall_s"]
    passes = sorted(aggregated.values(), key=lambda entry: entry["combined_s"], reverse=True)
    for entry in passes:
        if total and total["combined_s"]:
            entry["share_percent"] = 100.0 * entry["combined_s"] / total["combined_s"]
        else:
            entry["share_percent"] = None
        entry["combined_s"] = round(entry["combined_s"], 3)
        entry["wall_s"] = round(entry["wall_s"], 3)
        if entry["share_percent"] is not None:
            entry["share_percent"] = round(entry["share_percent"], 1)
    return passes[:limit]


def top_passes_text(passes):
    return "; ".join(
        f"{entry['name']}={entry['combined_s']:.3f}s"
        + (f" ({entry['share_percent']:.1f}%)" if entry["share_percent"] is not None else "")
        for entry in passes)


def measure_variant(name, dxc, work, flags, repeat, timeout_s, ftime):
    family, size, source = variant_info(name)
    data = source.encode("utf-8")
    hlsl = work / f"{name}.hlsl"
    hlsl.write_bytes(data)
    cso = work / f"{name}.cso"
    runs = []
    for index in range(repeat):
        run = run_once(dxc, hlsl, cso, flags, timeout_s)
        runs.append(run)
        peak = run["peak_bytes"]
        memory = f"{peak / MIB:.1f} MiB" if peak else "memory n/a"
        print(f"  rep {index + 1}/{repeat}: {run['wall_s']:.3f} s  {memory}  "
              f"{run['outcome']} (exit {run['exit_code']})", flush=True)
        if run["outcome"] != "dxil":
            break
    walls = [run["wall_s"] for run in runs]
    peaks = [run["peak_bytes"] for run in runs if run["peak_bytes"]]
    outcomes = sorted({run["outcome"] for run in runs})
    passes = []
    pass_total = None
    if ftime:
        with_report = [run for run in runs if run["outcome"] == "dxil" or run["outcome"] == "diagnostic"]
        if with_report:
            rows, total = parse_ftime(with_report[-1]["output"])
            if total:
                passes = dominant_passes(rows, total)
                pass_total = round(total["combined_s"], 3)
    record = {
        "variant": name,
        "family": family,
        "size": size,
        "hlsl_bytes": len(data),
        "dxil_bytes": next((run["cso_bytes"] for run in reversed(runs) if run["cso_bytes"]), 0),
        "reps": len(runs),
        "wall_median_s": round(statistics.median(walls), 3),
        "wall_min_s": round(min(walls), 3),
        "wall_max_s": round(max(walls), 3),
        "peak_median_mib": round(statistics.median(peaks) / MIB, 1) if peaks else None,
        "peak_max_mib": round(max(peaks) / MIB, 1) if peaks else None,
        "outcome": "/".join(outcomes),
        "exit_codes": [run["exit_code"] for run in runs],
        "first_diagnostic": next(
            (run["first_diagnostic"] for run in reversed(runs) if run["first_diagnostic"]), ""),
        "pass_total_s": pass_total,
        "passes": passes,
        "command": " ".join([f'"{dxc}"', *flags, "-Fo", str(cso), str(hlsl)]),
        "runs": [{"wall_s": round(run["wall_s"], 3),
                  "peak_mib": round(run["peak_bytes"] / MIB, 1) if run["peak_bytes"] else None,
                  "exit_code": run["exit_code"],
                  "outcome": run["outcome"]} for run in runs],
    }
    return record


def dxc_version(dxc):
    try:
        result = subprocess.run([str(dxc), "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    lines = [line.strip() for line in (result.stdout + result.stderr).splitlines() if line.strip()]
    return lines[0] if lines else ""


def csv_row(record):
    return {
        "variant": record["variant"],
        "family": record["family"],
        "size": record["size"],
        "hlsl_bytes": record["hlsl_bytes"],
        "dxil_bytes": record["dxil_bytes"],
        "reps": record["reps"],
        "wall_median_s": f"{record['wall_median_s']:.3f}",
        "wall_min_s": f"{record['wall_min_s']:.3f}",
        "wall_max_s": f"{record['wall_max_s']:.3f}",
        "peak_median_mib": "" if record["peak_median_mib"] is None else f"{record['peak_median_mib']:.1f}",
        "outcome": record["outcome"],
        "exit_codes": ";".join(str(code) for code in record["exit_codes"]),
        "first_diagnostic": record["first_diagnostic"],
        "pass_total_s": "" if record["pass_total_s"] is None else f"{record['pass_total_s']:.3f}",
        "top_passes": top_passes_text(record["passes"]),
    }


def print_table(records):
    print()
    print(f"{'variant':<22}{'HLSL B':>10}{'reps':>6}{'median s':>11}"
          f"{'peak MiB':>10}  outcome")
    for record in records:
        peak = "-" if record["peak_median_mib"] is None else f"{record['peak_median_mib']:.1f}"
        print(f"{record['variant']:<22}{record['hlsl_bytes']:>10,}{record['reps']:>6}"
              f"{record['wall_median_s']:>11.3f}{peak:>10}  {record['outcome']}")
        if record["passes"]:
            print(f"    passes: {top_passes_text(record['passes'])}")


def write_report(path, payload, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps({**payload, "variants": records}, indent=1) + "\n",
                        encoding="utf-8")
    else:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(csv_row(record) for record in records)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dxc", type=Path,
                        help="DXC executable to run (required unless --list-variants)")
    parser.add_argument("--target", default=DEFAULT_TARGET,
                        help=f"shader target passed to DXC (default: {DEFAULT_TARGET})")
    parser.add_argument("--entry", default=DEFAULT_ENTRY,
                        help=f"entry point passed to DXC (default: {DEFAULT_ENTRY})")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS,
                        help="comma-separated variant names; control, samples_N, branches_N "
                             f"or samplebranch_N (default: {DEFAULT_VARIANTS})")
    parser.add_argument("--repeat", type=int, default=1,
                        help="measured DXC runs per variant; the median is recorded "
                             "(default: 1)")
    parser.add_argument("--timeout", type=float, metavar="SECONDS",
                        help="wall timeout for one DXC run, for example 90; a local policy "
                             "value supplied by the caller (required unless --list-variants)")
    parser.add_argument("--extra-arg", action="append", default=[], metavar="ARG",
                        help="additional DXC argument, repeatable; use --extra-arg=-Od for "
                             "values that begin with '-'")
    parser.add_argument("--ftime-report", action="store_true",
                        help="add -ftime-report and record the dominant passes and shares")
    parser.add_argument("--out", type=Path,
                        help="write the record here; CSV, or JSON when the name ends in .json")
    parser.add_argument("--work-dir", type=Path,
                        help="keep the generated HLSL and DXIL in this directory")
    parser.add_argument("--list-variants", action="store_true",
                        help="list the generated variant names and source sizes without compiling")
    args = parser.parse_args(argv)

    if args.list_variants:
        print(f"{'variant':<22}{'family':<12}{'size':>8}{'HLSL B':>10}")
        for name in standard_variants():
            family, size, source = variant_info(name)
            print(f"{name:<22}{family:<12}{size:>8}{len(source.encode('utf-8')):>10,}")
        return 0

    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.timeout is None:
        parser.error("--timeout is required; pass the local policy value, for example 90")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.dxc is None:
        parser.error("--dxc is required unless --list-variants is given")
    if not args.dxc.is_file():
        parser.error(f"DXC executable not found: {args.dxc}")
    try:
        names = []
        for name in args.variants.split(","):
            name = name.strip()
            if name and name not in names:
                names.append(name)
        if not names:
            raise ValueError("--variants selects nothing")
        for name in names:
            variant_info(name)
    except ValueError as exc:
        parser.error(str(exc))

    flags = ["-T", args.target, "-E", args.entry, "-enable-16bit-types", *args.extra_arg]
    if args.ftime_report:
        flags.append("-ftime-report")
    print(f"DXC: {args.dxc}")

    temporary = None
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        work = args.work_dir
    else:
        temporary = tempfile.TemporaryDirectory(prefix="dxc-compile-time-")
        work = Path(temporary.name)

    records = []
    try:
        for name in names:
            print(f"{name}:")
            records.append(measure_variant(name, args.dxc, work, flags, args.repeat,
                                           args.timeout, args.ftime_report))
    finally:
        if temporary:
            temporary.cleanup()

    print_table(records)
    payload = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dxc": str(args.dxc),
        "dxc_version": dxc_version(args.dxc),
        "target": args.target,
        "entry": args.entry,
        "flags": flags,
        "repeat": args.repeat,
        "timeout_s": args.timeout,
        "ftime_report": args.ftime_report,
    }
    if args.out:
        write_report(args.out, payload, records)
        print(f"\nwrote {args.out}")
    else:
        print("\nNo --out given; nothing written. Use --out measurements.csv or .json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
