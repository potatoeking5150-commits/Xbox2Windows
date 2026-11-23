"""Bind the archive fixture's typed DLL interface and standard Zig process termination."""
import argparse
from pathlib import Path

from cpu_stdio_exe_aot import translate
from cpu_switch_aot import HostImport

CODEC = "archive_codec.dll"
BINDINGS = {
    (CODEC, b"next_length"): HostImport("host_next", (), 4),
    (CODEC, b"acquire"): HostImport("host_alloc", (4,), 8),
    (CODEC, b"unpack"): HostImport("host_unpack", (8,), 4),
    (CODEC, b"publish"): HostImport("host_publish", (8,), 4),
    (CODEC, b"publish_texture"): HostImport("host_texture", (8,), 4),
    (CODEC, b"release"): HostImport("host_free", (8,), 0),
    (CODEC, b"completed"): HostImport("host_done", (4,), 0),
    ("ntdll.dll", b"RtlExitUserProcess"): HostImport("host_exit", (4,), 0, noreturn=True),
}
HOST = """
struct Record;
extern "C" {
__declspec(dllimport) int next_length();
__declspec(dllimport) Record* acquire(unsigned);
__declspec(dllimport) int unpack(Record*);
__declspec(dllimport) int publish(Record*);
__declspec(dllimport) int publish_texture(Record*);
__declspec(dllimport) void release(Record*);
__declspec(dllimport) void completed(unsigned);
__declspec(dllimport) __declspec(noreturn) void RtlExitUserProcess(unsigned);
}
static word host_next() { return static_cast<unsigned>(next_length()); }
static word host_alloc(word n) { return reinterpret_cast<word>(acquire(static_cast<unsigned>(n))); }
static word host_unpack(word p) { return static_cast<unsigned>(unpack(reinterpret_cast<Record*>(p))); }
static word host_publish(word p) { return static_cast<unsigned>(publish(reinterpret_cast<Record*>(p))); }
static word host_texture(word p) { return static_cast<unsigned>(publish_texture(reinterpret_cast<Record*>(p))); }
static word host_free(word p) { release(reinterpret_cast<Record*>(p)); return 0; }
static word host_done(word n) { completed(static_cast<unsigned>(n)); return 0; }
[[noreturn]] static word host_exit(word n) { RtlExitUserProcess(static_cast<unsigned>(n)); }
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(translate(args.image, BINDINGS, HOST, allow_process_exit=True), encoding="ascii")
