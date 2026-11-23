"""Compare implicit TLS values seen in TLS callbacks on the main and worker threads.

The tls_value fixture keeps one __declspec(thread) int (7) and reports it in
the sink's reserved slot from both its TLS callback (caller 1) and its entry
point (caller 2). Each case runs in a fresh child process: load or map the
image, read the value on the main thread, run one worker thread whose start
routine returns its own TLS value, join it with a timeout, release, and report
the sink events with raw thread IDs translated to semantic identities
("main"/"worker") so two processes can be compared without comparing raw OS
thread IDs.

Expected (Refs #165): under both loaders implicit TLS is initialized and the
callbacks run on both threads with value 7. pe_map serves the data template
through a carrier DLL that Windows loads (tools/pe_map.cpp). The empty-template
control (tls_route) must map under both loaders. A load failure, a worker that
times out, a missing callback, or an unexpected value fails the test.
"""

import argparse
from collections import Counter
import ctypes
import json
from pathlib import Path
import subprocess
import sys

from native_map_check import Mapper
from native_tls_check import Call, load_sink

TLS, ENTRY = 1, 2
ATTACH, DETACH, THREAD_ATTACH, THREAD_DETACH = 1, 0, 2, 3
TLS_VALUE = 7
WAIT_MS = 10000  # shorter than the harness timeout so the exit path is observed
TIMEOUT = 60


def translate(calls, module, main_id, worker_id):
    """(caller, reason, semantic thread, value) for this module's events only."""
    out = []
    for call in calls:
        if call.module != module:
            continue
        if call.thread == main_id:
            who = "main"
        elif call.thread == worker_id:
            who = "worker"
        else:
            continue  # unrelated pool threads; same rule as native_tls_check
        out.append([call.caller, call.reason, who, call.reserved])
    return out


def run(target, image, sink_path, mapper_path):
    """Child process: one load, one main read, one worker probe, release."""
    sink = load_sink(sink_path)
    sink.sink_clear()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentThreadId.argtypes = []
    kernel.GetCurrentThreadId.restype = ctypes.c_uint32
    main_id = kernel.GetCurrentThreadId()
    worker_id = ctypes.c_uint32(0)

    module, status, detail = None, None, None
    if target == "native":
        kernel.LoadLibraryW.argtypes = [ctypes.c_wchar_p]
        kernel.LoadLibraryW.restype = ctypes.c_void_p
        kernel.FreeLibrary.argtypes = [ctypes.c_void_p]
        module = kernel.LoadLibraryW(str(image))
        if not module:
            print(json.dumps({"failure": ctypes.get_last_error(), "events": [], "main": None}))
            return
        release = lambda: kernel.FreeLibrary(module)  # noqa: E731
    else:
        mapper = Mapper(mapper_path)
        # The fixture links the static C runtime, whose imports are bound to the host's system DLLs.
        bindings = {Path(sink_path).name.casefold(): sink, "kernel32.dll": kernel}
        module, status, detail, _ = mapper(Path(image), bindings)
        if not module:
            print(json.dumps({"failure": [status, detail], "events": [], "main": None}))
            return
        release = lambda: mapper.release(module)  # noqa: E731

    # Resolve the probe exports from the loaded/mapped image itself.
    if target == "native":
        kernel.GetProcAddress.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        kernel.GetProcAddress.restype = ctypes.c_void_p
        read_addr = kernel.GetProcAddress(module, b"tls_value_read")
        probe_addr = kernel.GetProcAddress(module, b"tls_value_probe")
    else:
        mapper = Mapper(mapper_path)
        read_addr = mapper.export(module, b"tls_value_read")
        probe_addr = mapper.export(module, b"tls_value_probe")
    if not read_addr or not probe_addr:
        print(json.dumps({"failure": "missing export", "events": [], "main": None}))
        release()
        return
    main_value = ctypes.CFUNCTYPE(ctypes.c_int)(read_addr)()

    kernel.CreateThread.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                                    ctypes.c_void_p, ctypes.c_uint32,
                                    ctypes.POINTER(ctypes.c_uint32)]
    kernel.CreateThread.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.WaitForSingleObject.restype = ctypes.c_uint32
    kernel.GetExitCodeThread.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    kernel.GetExitCodeThread.restype = ctypes.c_int
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.CreateThread(None, 0, probe_addr, None, 0, ctypes.byref(worker_id))
    if not handle:
        print(json.dumps({"failure": "create-thread", "events": [], "main": main_value}))
        release()
        return
    waited = kernel.WaitForSingleObject(handle, WAIT_MS)
    worker_value = ctypes.c_uint32(0xDEAD)
    ok = waited == 0 and bool(kernel.GetExitCodeThread(handle, ctypes.byref(worker_value)))
    kernel.CloseHandle(handle)
    if not ok:
        # The worker may still be executing code from this module.
        print(json.dumps({"failure": "worker-wait", "events": [], "main": main_value,
                          "worker": None}))
        return
    buffer = (Call * 64)()
    count = sink.sink_read(buffer, len(buffer))
    events = translate(buffer[:count], module, main_id, worker_id.value)
    sink.sink_clear()
    release()
    # Detach events stay in the sink after release (the sink outlives the image).
    buffer = (Call * 64)()
    count = sink.sink_read(buffer, len(buffer))
    detach = translate(buffer[:count], module, main_id, worker_id.value)
    print(json.dumps({"failure": None,
                      "events": events, "detach": detach,
                      "main": main_value, "worker": worker_value.value}))


def run_control(target, image, sink_path, mapper_path):
    """Child process for the empty-template control: load, observe attach, release.

    The control fixture (tls_route) exports no tls_value_* symbols, so the
    value-probe run() above cannot be reused here. Success is a clean load
    with the TLS callback and entry point each observed once with
    DLL_PROCESS_ATTACH; any load failure is reported, never masked."""
    sink = load_sink(sink_path)
    sink.sink_clear()
    module = None
    if target == "native":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.LoadLibraryW.argtypes = [ctypes.c_wchar_p]
        kernel.LoadLibraryW.restype = ctypes.c_void_p
        kernel.FreeLibrary.argtypes = [ctypes.c_void_p]
        module = kernel.LoadLibraryW(str(image))
        if not module:
            print(json.dumps({"failure": ctypes.get_last_error()}))
            return
        release = lambda: kernel.FreeLibrary(module)  # noqa: E731
    else:
        mapper = Mapper(mapper_path)
        module, status, detail, _ = mapper(Path(image), {Path(sink_path).name.casefold(): sink})
        if not module:
            print(json.dumps({"failure": [status, detail]}))
            return
        release = lambda: mapper.release(module)  # noqa: E731
    buffer = (Call * 64)()
    count = sink.sink_read(buffer, len(buffer))
    attach = [[c.caller, c.reason, "module" if c.module == module else hex(c.module), c.reserved]
              for c in buffer[:count] if c.reason == ATTACH]
    release()
    print(json.dumps({"failure": None, "attach": attach}))


def child(target, image, sink, mapper, control=False):
    args = ["--run-control" if control else "--run", target,
            str(image), str(sink), str(mapper)]
    result = subprocess.run([sys.executable, "-B", __file__] + args,
                            capture_output=True, text=True, timeout=TIMEOUT)
    if result.returncode:
        raise AssertionError(f"{target} {Path(image).name}: child exited "
                             f"{result.returncode:#x}: {result.stderr}")
    return json.loads(result.stdout)


def check(value_image, control_image, sink, mapper):
    # Control first: the empty-template fixture must load under both loaders,
    # proving the harness itself can map and observe thread notifications.
    for target in ("native", "mapped"):
        observed = child(target, control_image, sink, mapper, control=True)
        if observed["failure"] is not None:
            raise AssertionError(f"control {target}: failed to load: {observed}")
        callers = [c for c, _, _, _ in observed["attach"]]
        if TLS not in callers or ENTRY not in callers:
            raise AssertionError(f"control {target}: missing attach callback: {observed}")
    print("control: empty-template TLS fixture loads under Windows and pe_map")

    expected_events = Counter(
        (caller, reason, thread, TLS_VALUE)
        for caller in (TLS, ENTRY)
        for reason, thread in ((ATTACH, "main"), (THREAD_ATTACH, "worker"),
                               (THREAD_DETACH, "worker"))
    )
    expected_detach = Counter((caller, DETACH, "main", TLS_VALUE) for caller in (TLS, ENTRY))
    for target in ("native", "mapped"):
        seen = child(target, value_image, sink, mapper)
        print(f"{target}: main={seen.get('main')} worker={seen.get('worker')} "
              f"events={seen['events']} detach={seen.get('detach')}")
        # Every callback and entry-point call on both threads must observe value 7.
        if seen["failure"] is not None:
            raise AssertionError(f"{target} failed: {seen}")
        if seen["main"] != TLS_VALUE or seen["worker"] != TLS_VALUE:
            raise AssertionError(f"{target} thread-local values differ from {TLS_VALUE}: {seen}")
        for observed, expected in ((seen["events"], expected_events),
                                   (seen["detach"], expected_detach)):
            if Counter(map(tuple, observed)) != expected:
                raise AssertionError(f"{target}: expected {expected}, got {observed}")

    print(f"TLS callbacks and entry points on the main and worker threads all read {TLS_VALUE} "
          f"under Windows and under pe_map")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--run", "--run-control"):
        (run_control if sys.argv[1] == "--run-control" else run)(*sys.argv[2:6])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        for arg in ("value", "control", "sink", "mapper"):
            parser.add_argument(arg, type=Path)
        args = parser.parse_args()
        check(args.value, args.control, args.sink, args.mapper)
