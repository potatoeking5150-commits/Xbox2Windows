"""Compare implicit TLS data under the Windows loader and under pe_map.

The tls_data fixture keeps an initialized value, a zero-filled value, a 64-byte aligned array, a
pointer with a base relocation inside the template, a zero-filled range longer than a page and a
C++ thread_local with a dynamic initializer. Each case runs in a fresh child process that loads
the image with LoadLibraryW or maps it with pe_map and records what three threads observe: the
loading thread, a thread that existed before the load and a thread created after it.

pe_map serves the template through a carrier DLL that Windows loads from a private directory
under the temporary directory. Every child gets a temporary directory of its own (TMP and TEMP), so
the test sees each process's carriers. It checks that while an image is mapped its carrier can be
neither renamed nor opened for writing and its directory cannot be renamed, that the carrier is gone
after release, and that the carrier a process leaves when it ends without releasing is not touched
by the next process that maps a TLS image there.

Two failure controls run beside carrier-like files of another process: one where Windows refuses
to load the carrier after it was written (the child's temporary directory carries a low integrity
label that new files inherit, and the child refuses images with that label), and one
where the carrier was loaded and the guest's entry point refuses the process attach. Each failed map
must remove its own carrier and leave the other files as they were.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import struct
import secrets
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading

import pefile

from native_map_check import Mapper, Resolver, STATUS

TIMEOUT = 60
WAIT_MS = 10000
ERROR_BAD_EXE_FORMAT, ERROR_NOACCESS, ERROR_DLL_INIT_FAILED = 193, 998, 1114
ERROR_ACCESS_DENIED = 5  # a low-label image under NoLowMandatoryLabelImages
MAP_MEMORY = 3  # MapStatus in tools/pe_map.cpp; the carrier's failures are reported with it
CARRIER = "pe_map_tls_"
ALIGN_MASK, ALIGN_4096 = 0x00F00000, 0x00D00000
MAX_TEMPLATE = 16 << 20  # MAX_TLS_TEMPLATE in tools/pe_map.cpp

kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.LoadLibraryW.argtypes = [ctypes.c_wchar_p]
kernel.LoadLibraryW.restype = ctypes.c_void_p
kernel.FreeLibrary.argtypes = [ctypes.c_void_p]
kernel.GetProcAddress.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
kernel.GetProcAddress.restype = ctypes.c_void_p
kernel.CreateThread.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_uint32, ctypes.c_void_p]
kernel.CreateThread.restype = ctypes.c_void_p
kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
kernel.WaitForSingleObject.restype = ctypes.c_uint32
kernel.CloseHandle.argtypes = [ctypes.c_void_p]
kernel.GetTempPathW.argtypes = [ctypes.c_uint32, ctypes.c_wchar_p]
kernel.GetEnvironmentVariableW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
kernel.SetProcessMitigationPolicy.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
# ProcessImageLoadPolicy in PROCESS_MITIGATION_POLICY and its NoLowMandatoryLabelImages flag. Only images
# with a low integrity label are refused, so nothing else the child loads is affected.
PROCESS_IMAGE_LOAD_POLICY, NO_LOW_LABEL_IMAGES = 10, 2
kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32,
                               ctypes.c_uint32, ctypes.c_void_p]
kernel.CreateFileW.restype = ctypes.c_void_p
GENERIC_WRITE, SHARE_ALL, OPEN_EXISTING, INVALID_HANDLE = 0x40000000, 7, 3, 2 ** 64 - 1


class Snapshot(ctypes.Structure):
    """Snapshot in tests/cpu/tls_data.cpp."""
    _fields_ = [("init", ctypes.c_int32), ("zero", ctypes.c_int32), ("wide", ctypes.c_uint64),
                ("pointer_ok", ctypes.c_uint32), ("aligned_offset", ctypes.c_uint32),
                ("aligned_first", ctypes.c_uint32), ("big_sum", ctypes.c_uint32),
                ("dynamic_nonzero", ctypes.c_uint32), ("index", ctypes.c_uint32),
                ("block", ctypes.c_uint64)]

    def view(self):
        return {name: getattr(self, name) for name, _ in self._fields_}


FRESH = {"init": 7, "zero": 0, "wide": 0x1122334455667788, "pointer_ok": 1, "aligned_offset": 0,
         "aligned_first": 0x5a, "big_sum": 0, "dynamic_nonzero": 1}


def written(value):
    return {**FRESH, "init": value, "zero": value + 1, "wide": value + 2, "aligned_first": value + 3,
            "big_sum": value + 4}


def temporary_directory():
    buffer = ctypes.create_unicode_buffer(262)
    kernel.GetTempPathW(len(buffer), buffer)
    return Path(buffer.value)


def listing(folder):
    """Every file and directory under folder, relative to it."""
    return sorted(path.relative_to(folder).as_posix() for path in Path(folder).rglob("*"))


def carrier_files(folder=None):
    """Carrier files under a temporary directory, by default this process's."""
    folder = Path(folder) if folder else temporary_directory()
    return sorted(path.relative_to(folder).as_posix() for path in folder.glob(f"{CARRIER}*/{CARRIER}*.dll"))


def open_for_writing(path):
    """CreateFileW with GENERIC_WRITE, sharing everything; raises PermissionError with the Windows error."""
    handle = kernel.CreateFileW(str(path), GENERIC_WRITE, SHARE_ALL, None, OPEN_EXISTING, 0, None)
    if handle in (None, INVALID_HANDLE):
        error = ctypes.get_last_error()
        raise PermissionError(None, "cannot open for writing", str(path), error)
    kernel.CloseHandle(handle)


def held(path):
    """Whether the carrier file and its directory resist a rename and the file resists a writer."""
    path = Path(path)
    results = {}
    for label, attempt in (("rename", lambda: path.rename(path.with_name("renamed.dll"))),
                           ("write", lambda: open_for_writing(path)),
                           ("rename directory", lambda: path.parent.rename(path.parent.with_name("renamed")))):
        try:
            attempt()
            results[label] = "succeeded"
        except PermissionError as error:
            results[label] = error.winerror
    return results


class Loader:
    """One way to load the fixture: the Windows loader or pe_map with system imports bound to the host."""

    def __init__(self, target, mapper_path):
        self.native = target == "native"
        self.mapper = None if self.native else Mapper(mapper_path)
        self.system, self.callbacks = {}, []

    def _resolve(self, dll, name, ordinal):
        module = self.system.setdefault(dll.decode(), ctypes.WinDLL(dll.decode()))
        return kernel.GetProcAddress(module._handle, name) if name else None

    def load(self, image):
        """Returns (module, failure); failure is the Windows error or [status, detail]."""
        if self.native:
            module = kernel.LoadLibraryW(str(image))
            return (module, None) if module else (None, ctypes.get_last_error())
        callback = Resolver(self._resolve)
        self.callbacks.append(callback)
        status, detail = ctypes.c_uint32(0xffffffff), ctypes.create_string_buffer(256)
        module = self.mapper.map(str(image), callback, ctypes.byref(status), detail, len(detail))
        return (module, None) if module else (None, [status.value, detail.value.decode()])

    def export(self, module, name):
        address = kernel.GetProcAddress(module, name) if self.native else self.mapper.export(module, name)
        if not address:
            raise AssertionError(f"missing export {name}")
        return address

    def release(self, module):
        return bool(kernel.FreeLibrary(module)) if self.native else self.mapper.release(module) == 1


class Image:
    def __init__(self, loader, module):
        self.snapshot_address = loader.export(module, b"tls_data_snapshot")
        self._snapshot = ctypes.CFUNCTYPE(None, ctypes.POINTER(Snapshot))(self.snapshot_address)
        self.write = ctypes.CFUNCTYPE(None, ctypes.c_int)(loader.export(module, b"tls_data_write"))
        self.worker = loader.export(module, b"tls_data_worker")

    def snapshot(self):
        out = Snapshot()
        self._snapshot(ctypes.byref(out))
        return out.view()

    def run_worker(self):
        """A thread created after the load: its first view and its view after writing 50."""
        pair = (Snapshot * 2)()
        handle = kernel.CreateThread(None, 0, self.worker, ctypes.byref(pair), 0, None)
        if not handle or kernel.WaitForSingleObject(handle, WAIT_MS) != 0:
            raise AssertionError("worker thread did not finish")
        kernel.CloseHandle(handle)
        return pair[0].view(), pair[1].view()


def emit(report):
    print(json.dumps(report))
    sys.stdout.flush()
    os._exit(0)  # the waiting thread, if any, must not keep a failed child alive


def run(target, mapper_path, image, second, leave="release"):
    """Child: observe three threads on one image, a second copy, and a reload after release."""
    loader = Loader(target, mapper_path)
    go, done, existing, holder = threading.Event(), threading.Event(), {}, []

    def before_load():
        go.wait()
        existing.update(holder[0].snapshot())
        done.set()

    thread = threading.Thread(target=before_load, daemon=True)
    thread.start()  # exists before the image is loaded

    others = carrier_files()  # left by an earlier process; never touched here
    module, failure = loader.load(image)
    if not module:
        emit({"failure": failure, "carriers": carrier_files()})
    first = Image(loader, module)
    holder.append(first)
    own = [name for name in carrier_files() if name not in others]
    report = {"failure": None, "carriers_loaded": len(own)}
    report["held"] = [held(temporary_directory() / name) for name in own]
    report["main"] = first.snapshot()
    first.write(20)
    report["main_written"] = first.snapshot()
    report["worker"], report["worker_written"] = first.run_worker()
    go.set()
    if not done.wait(WAIT_MS / 1000):
        emit({"failure": "the thread that existed before the load did not answer"})
    report["existing"] = dict(existing)
    report["main_end"] = first.snapshot()

    # A second image loaded at the same time has its own index and its own blocks.
    other_module, failure = loader.load(second)
    if not other_module:
        emit({"failure": ["second", failure]})
    other = Image(loader, other_module)
    report["second"] = other.snapshot()
    other.write(30)
    report["second_written"] = other.snapshot()
    report["first_after_second"] = first.snapshot()
    report["carriers_two"] = len(set(carrier_files()) - set(others))
    report["released"] = [loader.release(other_module), loader.release(module)]
    report["carriers_released"] = len(set(carrier_files()) - set(others))

    # Loaded again, the loading thread starts from the template, not from the earlier writes.
    module, failure = loader.load(image)
    if not module:
        emit({"failure": ["reload", failure]})
    report["reload"] = Image(loader, module).snapshot()
    report["carriers"] = [name for name in carrier_files() if name not in others]
    if leave != "leave":  # "leave" ends the process with the image still loaded
        report["released"].append(loader.release(module))
    emit(report)


def fail_run(target, mapper_path, image, kind):
    """Child: a map or load that fails after the carrier was written or after it was loaded.

    low-label: the temporary directory has a low integrity label that new files inherit, and from
    here on the process refuses images with that label, so Windows refuses the carrier pe_map wrote.
    no-policy: the same load without the policy, the control for low-label. refuse: image is the refusing fixture, whose entry point records
    what its thread-locals hold and fails the process attach.
    """
    loader = Loader(target, mapper_path)
    if kind == "low-label":
        flags = ctypes.c_uint32(NO_LOW_LABEL_IMAGES)
        if not kernel.SetProcessMitigationPolicy(PROCESS_IMAGE_LOAD_POLICY, ctypes.byref(flags), 4):
            emit({"failure": ["policy", ctypes.get_last_error()]})
    module, failure = loader.load(image)
    seen = ctypes.create_unicode_buffer(64)
    kernel.GetEnvironmentVariableW("TLS_DATA_ATTACH", seen, len(seen))
    emit({"failure": failure, "loaded": bool(module), "attach": seen.value, "listing": listing(temporary_directory())})


def child(target, mapper, image, second, leave=False, temp=None, fail=None):
    """Runs a child with temp as its temporary directory; fail selects a failure control instead."""
    environment = dict(os.environ)
    if temp:
        Path(temp).mkdir(parents=True, exist_ok=True)
        environment["TMP"] = environment["TEMP"] = str(temp)
    mode = ["--fail", target, str(mapper), str(image), fail] if fail else \
        ["--run", target, str(mapper), str(image), str(second), "leave" if leave else "release"]
    result = subprocess.run([sys.executable, "-B", __file__, *mode], capture_output=True, text=True, timeout=TIMEOUT,
                            env=environment)
    if result.returncode:
        raise AssertionError(f"{target} {Path(image).name}: child exited {result.returncode:#x}: {result.stderr}")
    return json.loads(result.stdout)


def ended_pid():
    """The ID of a process that has ended."""
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait(TIMEOUT)
    return process.pid


def decoys(folder):
    """Carrier-like files of another process: one in the layout of an earlier pe_map, named after a
    process that has ended, and one directory in the current layout. Returns their contents."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    name = secrets.token_hex(16)
    files = {f"{CARRIER}{ended_pid():08x}_00000001_0000000000000000.dll": b"another process's carrier, old layout",
             f"{CARRIER}{name}/{CARRIER}{name}.dll": b"another process's carrier"}
    for relative, data in files.items():
        (folder / relative).parent.mkdir(exist_ok=True)
        (folder / relative).write_bytes(data)
    return files


def low_label(folder):
    """Give folder a low integrity label that files and directories created in it inherit."""
    result = subprocess.run(["icacls", str(folder), "/setintegritylevel", "(OI)(CI)low"], capture_output=True,
                            text=True, timeout=TIMEOUT)
    if result.returncode:
        raise AssertionError(f"icacls could not label {folder}: {result.stdout} {result.stderr}")


def check_decoys(label, folder, files, report):
    """Only the decoys are left, unchanged."""
    expected = sorted({*files, *(Path(relative).parent.as_posix() for relative in files)} - {"."})
    if report["listing"] != expected:
        raise AssertionError(f"{label}: expected only the other process's files {expected}, found {report['listing']}")
    for relative, data in files.items():
        if (Path(folder) / relative).read_bytes() != data:
            raise AssertionError(f"{label}: {relative} was changed")


SEMANTIC = ("main", "main_written", "worker", "worker_written", "existing", "main_end", "second",
            "second_written", "first_after_second", "reload")


def semantic(report, ignore=()):
    """The observations two loaders must agree on: values, not addresses or index numbers."""
    drop = {"index", "block", *ignore}
    return {key: {k: v for k, v in report[key].items() if k not in drop} for key in SEMANTIC}


def check_structure(label, report):
    """Properties of one run that hold under either loader."""
    if report["failure"] is not None:
        raise AssertionError(f"{label}: failed to load: {report}")
    blocks = [report[key]["block"] for key in ("main", "worker", "existing")]
    if len(set(blocks)) != 3:
        raise AssertionError(f"{label}: threads share a TLS block: {blocks}")
    if report["main"]["block"] != report["main_end"]["block"]:
        raise AssertionError(f"{label}: the loading thread's block moved")
    if report["second"]["index"] == report["main"]["index"]:
        raise AssertionError(f"{label}: two loaded images share TLS index {report['main']['index']}")
    if not all(report["released"]):
        raise AssertionError(f"{label}: release failed: {report['released']}")


def patch(source, target, edit):
    """Copy the fixture with its TLS directory or section table changed by edit(data, pe, tls)."""
    data = bytearray(Path(source).read_bytes())
    with pefile.PE(data=bytes(data)) as pe:
        edit(data, pe, pe.DIRECTORY_ENTRY_TLS.struct)
    Path(target).write_bytes(data)


def set_field(name, pack, value):
    def edit(data, pe, tls):
        struct.pack_into(pack, data, tls.get_field_absolute_offset(name), value(pe, tls))
    return edit


def oversize(data, pe, tls):
    """Grow the last section virtually so a template above the limit still lies inside the image."""
    extra = (MAX_TEMPLATE + 0x100000) & ~0xfff
    last = pe.sections[-1]
    struct.pack_into("<I", data, last.get_field_absolute_offset("Misc_VirtualSize"), last.Misc_VirtualSize + extra)
    struct.pack_into("<I", data, pe.OPTIONAL_HEADER.get_field_absolute_offset("SizeOfImage"),
                     pe.OPTIONAL_HEADER.SizeOfImage + extra)
    struct.pack_into("<Q", data, tls.get_field_absolute_offset("EndAddressOfRawData"),
                     tls.StartAddressOfRawData + MAX_TEMPLATE + 1)


ACCEPTED = {
    # Fields Windows 10 build 19045 loads whatever their value; pe_map hands them to Windows unchanged.
    "SizeOfZeroFill 0x10": (set_field("SizeOfZeroFill", "<I", lambda pe, tls: 0x10), ()),
    "4096-byte alignment": (set_field("Characteristics", "<I", lambda pe, tls: ALIGN_4096), ()),
    # Without an alignment the 64-byte array lands wherever the default alignment puts it.
    "no alignment": (set_field("Characteristics", "<I", lambda pe, tls: tls.Characteristics & ~ALIGN_MASK),
                     ("aligned_offset",)),
}

REJECTED = {
    "template past the image": (
        set_field("EndAddressOfRawData", "<Q",
                  lambda pe, tls: pe.OPTIONAL_HEADER.ImageBase + pe.OPTIONAL_HEADER.SizeOfImage + 0x1000),
        ERROR_NOACCESS, [STATUS["format"], "TLS template outside image"]),
    "template ending before its start": (
        set_field("EndAddressOfRawData", "<Q", lambda pe, tls: tls.StartAddressOfRawData - 8),
        ERROR_BAD_EXE_FORMAT, [STATUS["format"], "TLS template ends before it starts"]),
    # Windows is not asked: it would give every thread a block of more than 16 MiB.
    "template above 16 MiB": (oversize, None, [STATUS["unsupported"], "TLS template larger than 16 MiB"]),
}


def check(mapper, fixture, refuse):
    with TemporaryDirectory() as temporary:
        folder = Path(temporary)
        second = folder / "tls_data_second.dll"
        shutil.copy2(fixture, second)

        native = child("native", mapper, fixture, second, temp=folder / "temp_native")
        mapped = child("mapped", mapper, fixture, second, temp=folder / "temp_mapped")
        check_structure("native", native)
        check_structure("mapped", mapped)
        expected = {"main": FRESH, "main_written": written(20), "worker": FRESH, "worker_written": written(50),
                    "existing": FRESH, "main_end": written(20), "second": FRESH, "second_written": written(30),
                    "first_after_second": written(20), "reload": FRESH}
        for label, report in (("native", native), ("mapped", mapped)):
            if semantic(report) != expected:
                raise AssertionError(f"{label}: expected {expected}, got {semantic(report)}")
        # Windows writes no file; pe_map holds one carrier per mapped image and removes it on release.
        counts = [(r["carriers_loaded"], r["carriers_two"], r["carriers_released"]) for r in (native, mapped)]
        if counts != [(0, 0, 0), (1, 2, 0)]:
            raise AssertionError(f"carrier files while loaded, with two images, after release: {counts}")
        if listing(folder / "temp_mapped"):
            raise AssertionError(f"left after release: {listing(folder / 'temp_mapped')}")
        print(f"native and mapped agree on {len(SEMANTIC)} observations: the loading thread, a thread that existed "
              f"before the load and a new thread each start from the template in a block of their own, aligned to "
              f"64 bytes; indexes native {native['main']['index']}/{native['second']['index']}, "
              f"mapped {mapped['main']['index']}/{mapped['second']['index']}")
        # While mapped, the carrier's path keeps naming the file pe_map wrote: the file and its
        # directory cannot be renamed and the file cannot be opened for writing (error 32).
        if mapped["held"] != [{"rename": 32, "write": 32, "rename directory": 32}]:
            raise AssertionError(f"the carrier's path did not hold while mapped: {mapped['held']}")
        # A process that ends with an image still mapped leaves its carrier (Windows cannot delete a
        # mapped image). The next process that maps a TLS image in the same temporary directory
        # removes its own carrier and leaves that one alone.
        shared = folder / "temp_shared"
        left = child("mapped", mapper, fixture, second, leave=True, temp=shared)
        leftover = carrier_files(shared)
        if leftover != left["carriers"] or len(leftover) != 1:
            raise AssertionError(f"expected the ended process's one carrier, found {leftover}, reported {left['carriers']}")
        contents = (shared / leftover[0]).read_bytes()
        later = child("mapped", mapper, fixture, second, temp=shared)
        if later["failure"] is not None or carrier_files(shared) != leftover \
                or (shared / leftover[0]).read_bytes() != contents:
            raise AssertionError(f"another process's carrier was touched: {carrier_files(shared)}, {later['failure']}")
        print("while mapped, renaming the carrier or its directory and opening it for writing fail with error 32; "
              "nothing is left after release; a carrier left by an ended process is not touched by a later map")

        # Failure controls: each failed map removes its own carrier and nothing else.
        controls = {}
        temp = folder / "temp_low_label"
        files = decoys(temp)
        low_label(temp)
        # Control: a copy of the fixture in that directory carries the label, and Windows refuses it.
        labeled = temp / "labeled_copy" / "tls_data.dll"
        labeled.parent.mkdir()
        shutil.copy2(fixture, labeled)
        plain = child("native", mapper, labeled, None, temp=temp, fail="no-policy")
        native = child("native", mapper, labeled, None, temp=temp, fail="low-label")
        shutil.rmtree(labeled.parent)
        if not plain["loaded"]:
            raise AssertionError(f"low-label: the labeled copy does not load without the policy: {plain}")
        if native["failure"] != ERROR_ACCESS_DENIED:
            raise AssertionError(f"low-label: Windows gave {native['failure']} for a labeled copy, "
                                 f"expected error {ERROR_ACCESS_DENIED}")
        report = child("mapped", mapper, fixture, None, temp=temp, fail="low-label")
        expected_failure = [MAP_MEMORY, f"Windows cannot load the TLS carrier (error {ERROR_ACCESS_DENIED})"]
        if report["failure"] != expected_failure:
            raise AssertionError(f"low-label: expected {expected_failure}, got {report}")
        report["listing"] = [name for name in report["listing"] if not name.startswith("labeled_copy")]
        check_decoys("low-label", temp, files, report)
        controls["the carrier was written and Windows refused it"] = report["failure"][1]
        native = child("native", mapper, refuse, None, temp=folder / "temp_refuse_native", fail="refuse")
        if native["failure"] != ERROR_DLL_INIT_FAILED or native["attach"] != "7 1":
            raise AssertionError(f"refuse: Windows gave {native}, expected error {ERROR_DLL_INIT_FAILED} after '7 1'")
        temp = folder / "temp_refuse"
        files = decoys(temp)
        report = child("mapped", mapper, refuse, None, temp=temp, fail="refuse")
        # "7 1": during the attach, t_init and t_wide held their template values in the carrier's block.
        if report["failure"] != [STATUS["init_failed"], "entry point returned FALSE"] or report["attach"] != "7 1":
            raise AssertionError(f"refuse: expected init_failed after the attach saw '7 1', got {report}")
        check_decoys("refuse", temp, files, report)
        controls["the carrier was loaded and the entry point refused"] = report["failure"][1]
        print("failed maps remove only their own carrier, beside an ended process's old-layout file and another "
              f"carrier directory: {'; '.join(f'{k}: {v}' for k, v in controls.items())}; Windows fails the refusing "
              f"fixture with error {ERROR_DLL_INIT_FAILED} after the same attach")

        for label, (edit, ignore) in ACCEPTED.items():
            copy = folder / "accepted.dll"
            patch(fixture, copy, edit)
            native = child("native", mapper, copy, second, temp=folder / "temp_accepted_native")
            mapped = child("mapped", mapper, copy, second, temp=folder / "temp_accepted_mapped")
            check_structure(f"native {label}", native)
            check_structure(f"mapped {label}", mapped)
            if semantic(native, ignore) != semantic(mapped, ignore):
                raise AssertionError(f"{label}: native {semantic(native, ignore)} != mapped {semantic(mapped, ignore)}")
            if semantic(native, ("aligned_offset",)) != {k: {f: v for f, v in e.items() if f != "aligned_offset"}
                                                         for k, e in expected.items()}:
                raise AssertionError(f"{label}: unexpected values {semantic(native)}")
            for report in (native, mapped):
                if any(report[key]["aligned_offset"] % 16 for key in SEMANTIC):
                    raise AssertionError(f"{label}: block below the default 16-byte alignment: {report}")
        print(f"native and mapped agree on copies with {', '.join(ACCEPTED)}")

        for label, (edit, windows_error, rejection) in REJECTED.items():
            copy = folder / "rejected.dll"
            patch(fixture, copy, edit)
            if windows_error is not None:
                native = child("native", mapper, copy, second, temp=folder / "temp_rejected_native")
                if native["failure"] != windows_error:
                    raise AssertionError(f"{label}: Windows gave {native['failure']}, expected {windows_error}")
            mapped = child("mapped", mapper, copy, second, temp=folder / "temp_rejected_mapped")
            if mapped["failure"] != rejection or mapped["carriers"] or listing(folder / "temp_rejected_mapped"):
                raise AssertionError(f"{label}: expected {rejection} and no carrier file, got {mapped}")
        print(f"pe_map rejects a {', a '.join(REJECTED)}; Windows fails the first two with errors "
              f"{ERROR_NOACCESS} and {ERROR_BAD_EXE_FORMAT}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run":
        run(*sys.argv[2:7])
    elif len(sys.argv) > 1 and sys.argv[1] == "--fail":
        fail_run(*sys.argv[2:6])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("mapper", type=Path)
        parser.add_argument("fixture", type=Path)
        parser.add_argument("refuse", type=Path, help="the fixture built with ENTRY_FAILS")
        arguments = parser.parse_args()
        check(arguments.mapper, arguments.fixture, arguments.refuse)
