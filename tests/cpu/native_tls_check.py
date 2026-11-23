"""Compare TLS callback and entry-point notifications of pe_map with the Windows loader."""

import argparse
import ctypes
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory

import pefile

from native_map_check import Mapper, STATUS

ATTACH, DETACH = 1, 0
TLS, ENTRY = 1, 2
ERROR_DLL_INIT_FAILED = 1114


class Call(ctypes.Structure):
    _fields_ = [("caller", ctypes.c_uint32), ("reason", ctypes.c_uint32),
                ("module", ctypes.c_uint64), ("reserved", ctypes.c_uint64),
                ("thread", ctypes.c_uint32), ("padding", ctypes.c_uint32)]


def load_sink(path):
    sink = ctypes.CDLL(str(path))
    sink.sink_read.argtypes = [ctypes.POINTER(Call), ctypes.c_size_t]
    sink.sink_read.restype = ctypes.c_size_t
    sink.sink_clear.argtypes = []
    sink.sink_clear.restype = None
    return sink


def calls(sink, module=None):
    """Recorded process notifications; module=None names the module of the first call (failed loads)."""
    buffer = (Call * 64)()
    count = sink.sink_read(buffer, len(buffer))
    # Thread attach/detach (reasons 2 and 3) depends on unrelated threads; only process events are compared.
    recorded = [c for c in buffer[:count] if c.reason in (ATTACH, DETACH)]
    if module is None and recorded:
        module = recorded[0].module
    return [[c.caller, c.reason, "module" if c.module == module else hex(c.module), c.reserved] for c in recorded]


THREAD_ATTACH, THREAD_DETACH = 2, 3


def new_thread_calls(sink, module, before_resume=None):
    """Start and join one thread; return the thread notifications recorded on that thread.

    The thread is created suspended; before_resume(thread_id), if given, runs before it starts."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    kernel.GetModuleHandleW.restype = ctypes.c_void_p
    kernel.GetProcAddress.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    kernel.GetProcAddress.restype = ctypes.c_void_p
    kernel.CreateThread.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p,
                                    ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
    kernel.CreateThread.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.ResumeThread.argtypes = [ctypes.c_void_p]
    kernel.ResumeThread.restype = ctypes.c_uint32
    # Sleep(0) as the start routine: on x64 the parameter arrives in RCX (null, so Sleep(0)); Sleep returns
    # no value, so the thread's exit code is undefined, and it is never read. The thread exits at once.
    start = kernel.GetProcAddress(kernel.GetModuleHandleW("kernel32"), b"Sleep")
    thread_id = ctypes.c_uint32()
    sink.sink_clear()
    handle = kernel.CreateThread(None, 0, start, None, 0x4, ctypes.byref(thread_id))  # CREATE_SUSPENDED
    if not handle:
        raise AssertionError("test thread could not be created")
    if before_resume:
        before_resume(thread_id.value)
    kernel.ResumeThread(handle)
    if kernel.WaitForSingleObject(handle, 10000) != 0:
        raise AssertionError("test thread did not start or exit")
    kernel.CloseHandle(handle)
    buffer = (Call * 64)()
    count = sink.sink_read(buffer, len(buffer))
    return [[c.caller, c.reason, "module" if c.module == module else hex(c.module), c.reserved]
            for c in buffer[:count] if c.thread == thread_id.value]


def run(target, image, sink_path, mapper_path):
    """Child process per target and image: load, observe attach, one thread, release, observe detach."""
    sink = load_sink(sink_path)
    sink.sink_clear()
    if target == "native":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.LoadLibraryW.argtypes = [ctypes.c_wchar_p]
        kernel.LoadLibraryW.restype = ctypes.c_void_p
        kernel.FreeLibrary.argtypes = [ctypes.c_void_p]
        module = kernel.LoadLibraryW(image)
        failure = None if module else ctypes.get_last_error()
        release = lambda: kernel.FreeLibrary(module)
    else:
        mapper = Mapper(mapper_path)
        module, status, detail, _ = mapper(Path(image), {Path(sink_path).name.casefold(): sink})
        failure = None if module else [status, detail]
        release = lambda: mapper.release(module)
    if not module:
        print(json.dumps({"failure": failure, "attach": calls(sink)}))
        return
    attach = calls(sink, module)
    thread = new_thread_calls(sink, module)
    sink.sink_clear()
    release()
    print(json.dumps({"failure": None, "attach": attach, "thread": thread, "detach": calls(sink, module)}))


def child(target, image, sink, mapper):
    result = subprocess.run([sys.executable, "-B", __file__, "--run", target, str(image), str(sink), str(mapper)],
                            capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise AssertionError(f"{target} {Path(image).name}: child exited {result.returncode:#x}: {result.stderr}")
    return json.loads(result.stdout)


def check(original, refuse, sink, mapper):
    process = lambda reason: [[TLS, reason, "module", 0], [ENTRY, reason, "module", 0]]
    expected = {
        "original": {"failure": None, "attach": process(ATTACH),
                     "thread": process(THREAD_ATTACH) + process(THREAD_DETACH), "detach": process(DETACH)},
        "refuse": {"attach": process(ATTACH) + process(DETACH)},
    }
    for name, image in (("original", original), ("refuse", refuse)):
        native, mapped = child("native", image, sink, mapper), child("mapped", image, sink, mapper)
        for target, observed in (("native", native), ("mapped", mapped)):
            compared = dict(observed)
            if name == "refuse":
                compared.pop("failure")  # Windows and pe_map report failure differently; checked below
            if compared != expected[name]:
                raise AssertionError(f"{name} {target}: {observed} != {expected[name]}")
        if name == "refuse" and (native["failure"] != ERROR_DLL_INIT_FAILED
                                 or mapped["failure"] != [STATUS["init_failed"], "entry point returned FALSE"]):
            raise AssertionError(f"refuse failures: native={native['failure']} mapped={mapped['failure']}")
    print("native and mapped: TLS callback then entry point on process attach, on a new thread's attach and "
          "detach, and on process detach, reserved null; a FALSE entry point is followed by a full detach "
          "and the load fails")
    check_directory_fields(original, sink, mapper)
    check_rejections(original, sink, mapper)


ERROR_NOACCESS = 998


def check_directory_fields(image, sink, mapper):
    """The loader finds the TLS directory by address alone; compare patched copies with Windows."""
    with pefile.PE(str(image)) as pe:
        tls = pe.OPTIONAL_HEADER.DATA_DIRECTORY[9]
        address, size = (tls.get_field_absolute_offset(f) for f in ("VirtualAddress", "Size"))
        built, image_size = tls.Size, pe.OPTIONAL_HEADER.SizeOfImage
    if built != 40:  # sizeof(IMAGE_TLS_DIRECTORY64)
        raise AssertionError(f"TLS directory Size {built}, expected 40")
    same = (("Size 0", size, 0), ("Size 1", size, 1), ("Size 39", size, 39), ("address 0", address, 0))
    failing = (("address past the image", address, image_size + 0x1000),
               ("directory ending past the image", address, image_size - 8))
    for name, offset, value in same + failing:
        copy = bytearray(Path(image).read_bytes())
        struct.pack_into("<I", copy, offset, value)
        with TemporaryDirectory() as temp:
            damaged = Path(temp) / Path(image).name
            damaged.write_bytes(copy)
            native, mapped = child("native", damaged, sink, mapper), child("mapped", damaged, sink, mapper)
        if (name, offset, value) in same:
            if native != mapped or native["failure"] is not None:
                raise AssertionError(f"TLS {name}: native={native} mapped={mapped}")
            callbacks = any(call[0] == TLS for call in native["attach"])
            if callbacks != (offset == size):
                raise AssertionError(f"TLS {name}: callback ran={callbacks} under Windows")
        elif (native != {"failure": ERROR_NOACCESS, "attach": []}
              or mapped != {"failure": [STATUS["format"], "TLS directory outside image"], "attach": []}):
            raise AssertionError(f"TLS {name}: native={native} mapped={mapped}")
    print("TLS directory with Size 0, 1 or 39: callback and entry point run under Windows and pe_map; "
          "address 0: entry point only under both; a directory past the image fails both loads")


def check_rejections(image, sink_path, mapper_path):
    """Damaged copies are rejected without any TLS or entry-point notification."""
    with pefile.PE(str(image)) as pe:
        tls = pe.DIRECTORY_ENTRY_TLS.struct
        zero_fill = tls.get_field_absolute_offset("SizeOfZeroFill")
        characteristics = pe.FILE_HEADER.get_field_absolute_offset("Characteristics")
        flags = pe.FILE_HEADER.Characteristics
        entry = pe.OPTIONAL_HEADER.get_field_absolute_offset("AddressOfEntryPoint")
        data = next(s for s in pe.sections if s.Name.rstrip(b"\0") == b".data")
        rdata = next(s for s in pe.sections if s.Name.rstrip(b"\0") == b".rdata")
        first_callback = pe.get_offset_from_rva(tls.AddressOfCallBacks - pe.OPTIONAL_HEADER.ImageBase)
        data_va = pe.OPTIONAL_HEADER.ImageBase + data.VirtualAddress
        rdata_va = pe.OPTIONAL_HEADER.ImageBase + rdata.VirtualAddress
        index = tls.get_field_absolute_offset("AddressOfIndex")
        image_base, callbacks_va = pe.OPTIONAL_HEADER.ImageBase, tls.AddressOfCallBacks
        image_size = pe.OPTIONAL_HEADER.SizeOfImage
        callback_section = next(s for s in pe.sections if s.Name.rstrip(b"\0") == b".tlscb")
        data_flags = data.get_field_absolute_offset("Characteristics")
        callback_flags = callback_section.get_field_absolute_offset("Characteristics")
        no_access = ~(0x40000000 | 0x80000000) & 0xffffffff  # clear IMAGE_SCN_MEM_READ and _WRITE
        data_unreadable = data.Characteristics & no_access
        callbacks_unreadable = callback_section.Characteristics & no_access
    controls = (
        ("<I", entry, data.VirtualAddress, "format", "entry point outside executable section", False),
        ("<Q", first_callback, data_va, "format", "TLS callback outside executable section", True),
        # Section protections are applied before the TLS directory and list are read.
        ("<I", data_flags, data_unreadable, "format", "TLS directory not readable", True),
        ("<I", callback_flags, callbacks_unreadable, "format", "TLS callback list not readable", True),
        # The loader writes the TLS index, so the slot must be writable. Windows fails
        # these copies with ERROR_NOACCESS; pe_map rejects them before any notification.
        ("<Q", index, image_base + image_size + 0x1000, "format", "TLS index outside image", True),
        ("<Q", index, rdata_va, "format", "TLS index outside writable section", True),
    )
    sink = load_sink(sink_path)
    mapper = Mapper(mapper_path)
    for layout, offset, value, expected, message, after_imports in controls:
        copy = bytearray(Path(image).read_bytes())
        struct.pack_into(layout, copy, offset, value)
        sink.sink_clear()
        with TemporaryDirectory() as temp:
            damaged = Path(temp) / Path(image).name
            damaged.write_bytes(copy)
            base, status, detail, requests = mapper(damaged, {Path(sink_path).name.casefold(): sink})
        if base or status != STATUS[expected] or detail != message or bool(requests) != after_imports:
            raise AssertionError(f"{message} control: {base=} {status=} {detail=} {requests=}")
        if calls(sink):
            raise AssertionError(f"{message} control: notifications were sent: {calls(sink)}")
    print("rejected without notifications: an entry point in .data, a TLS callback "
          "pointing into .data, an unreadable TLS directory, an unreadable callback list, a TLS "
          "index past the image and a TLS index in read-only .rdata")

    # Copies both loaders accept, with the usual notifications.
    # - A valid alternate writable slot (even unaligned): measured on Windows 11, .data+4 runs the
    #   callback and the entry point. This guards the index check against false rejections.
    # - A zero-fill size on the empty template: measured on Windows 10 build 19045, the copy loads.
    #   pe_map hands the size to Windows in the TLS carrier (tools/pe_map.cpp).
    loader = ctypes.WinDLL("kernel32", use_last_error=True)
    loader.LoadLibraryExW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_uint32]
    loader.LoadLibraryExW.restype = ctypes.c_void_p
    loader.FreeLibrary.argtypes = [ctypes.c_void_p]
    expected = [[TLS, ATTACH, "module", 0], [ENTRY, ATTACH, "module", 0]]
    accepted = (
        ("alternate writable TLS index (.data+4, unaligned)", "<Q", index, data_va + 4),
        ("TLS zero-fill size 16 on an empty template", "<I", zero_fill, 16),
    )
    for label, layout, offset, value in accepted:
        copy = bytearray(Path(image).read_bytes())
        struct.pack_into(layout, copy, offset, value)
        sink.sink_clear()
        with TemporaryDirectory() as temp:
            moved = Path(temp) / Path(image).name
            moved.write_bytes(copy)
            shutil.copy(sink_path, Path(temp) / Path(sink_path).name)
            altered = loader.LoadLibraryExW(str(moved), None, 0x8)
            native_attach = calls(sink)
            if altered:
                loader.FreeLibrary(altered)
            # sink_read does not consume: clear the native attach/detach before mapping.
            sink.sink_clear()
            base, status, detail, _ = mapper(moved, {Path(sink_path).name.casefold(): sink})
            mapped_attach = calls(sink, base) if base else None
            if base:
                mapper.release(base)
        if not altered or native_attach != expected:
            raise AssertionError(f"{label}: native={bool(altered)} {native_attach}")
        if not base or status != STATUS["ok"] or mapped_attach != expected:
            raise AssertionError(f"{label}: mapped={base} {status} {detail} {mapped_attach}")
        print(f"{label}: Windows and pe_map run the TLS callback and the entry point")

    # LoadLibrary sends nothing to a non-DLL image (for an EXE the entry point is the program).
    # Clearing IMAGE_FILE_DLL must leave both the TLS callback and the entry point silent.
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LoadLibraryW.argtypes = [ctypes.c_wchar_p]
    kernel.LoadLibraryW.restype = ctypes.c_void_p
    kernel.FreeLibrary.argtypes = [ctypes.c_void_p]
    for variant, entry_value in (("with its entry point", None), ("with its entry point zeroed", 0)):
        copy = bytearray(Path(image).read_bytes())
        struct.pack_into("<H", copy, characteristics, flags & ~0x2000)
        if entry_value is not None:
            struct.pack_into("<I", copy, entry, entry_value)
        sink.sink_clear()
        with TemporaryDirectory() as temp:
            damaged = Path(temp) / Path(image).name
            damaged.write_bytes(copy)
            base, status, detail, _ = mapper(damaged, {Path(sink_path).name.casefold(): sink})
            if status != STATUS["ok"] or calls(sink):
                raise AssertionError(f"non-DLL {variant} attach: {status=} {detail=} {calls(sink)}")
            mapper.release(base)
            if calls(sink):
                raise AssertionError(f"non-DLL {variant} release sent notifications: {calls(sink)}")
            module = kernel.LoadLibraryW(str(damaged))
            attach = calls(sink)
            if module:
                kernel.FreeLibrary(module)
            if not module or attach or calls(sink):
                raise AssertionError(f"Windows non-DLL {variant}: {module=} {attach=} detach={calls(sink)}")
    print("with IMAGE_FILE_DLL cleared, with or without an entry point, Windows and pe_map both load and "
          "release the copy without any TLS or entry-point call")

    # The callback list is re-read on release. If it was overwritten with a non-executable address, or
    # its page was made unreadable after mapping, the entry point must still get its detach.
    kernel.VirtualProtect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32,
                                      ctypes.POINTER(ctypes.c_uint32)]
    old = ctypes.c_uint32()
    for change in ("overwritten", "made unreadable"):
        sink.sink_clear()
        base, status, detail, _ = mapper(Path(image), {Path(sink_path).name.casefold(): sink})
        if status != STATUS["ok"]:
            raise AssertionError(f"map failed: {status} {detail}")
        slot = base + callbacks_va - image_base
        if change == "overwritten":
            if not kernel.VirtualProtect(slot, 8, 0x04, ctypes.byref(old)):  # PAGE_READWRITE
                raise AssertionError("cannot unprotect the callback list")
            ctypes.c_uint64.from_address(slot).value = base  # the headers are not executable
        elif not kernel.VirtualProtect(slot, 8, 0x01, ctypes.byref(old)):  # PAGE_NOACCESS
            raise AssertionError("cannot protect the callback list")
        sink.sink_clear()
        mapper.release(base)
        if calls(sink, base) != [[ENTRY, DETACH, "module", 0]]:
            raise AssertionError(f"release with a callback list {change}: {calls(sink, base)}")
    print("release with a callback list overwritten or made unreadable after mapping still detaches "
          "the entry point only")


if __name__ == "__main__":
    if sys.argv[1] == "--run":
        run(*sys.argv[2:6])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        for arg in ("original", "refuse", "sink", "mapper"):
            parser.add_argument(arg, type=Path)
        args = parser.parse_args()
        check(args.original, args.refuse, args.sink, args.mapper)
