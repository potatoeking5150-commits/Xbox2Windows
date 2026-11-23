// A DLL with implicit thread-local storage read inside its TLS callback.
//
// g_tls lives in the PE TLS data template (non-empty), unlike tls_route.asm
// whose template is empty. The TLS callback and DllMain each forward the
// observed g_tls value in the sink's reserved slot, with the same caller tags
// as tls_route (1 = TLS callback, 2 = entry point), so the harness can compare
// the value seen on the main thread and on one worker thread.
//
// Built with the static MSVC runtime like the life_* fixtures; the callback
// is registered through .CRT$XLB so the linker emits a real TLS directory
// with data. pe_map serves the template through a carrier DLL that Windows
// loads (tools/pe_map.cpp), so both loaders must report the same values.
// See Refs #165.
#include <windows.h>

#include <cstdint>

extern "C" __declspec(dllimport) void sink_record(void* module, unsigned reason, void* reserved,
                                                  unsigned caller);

// Global scope by design: the linker must emit g_tls in the TLS data template
// and the callback address in .CRT$XLB. An anonymous namespace would give both
// internal linkage and obscure the template in a dumpbin review.
__declspec(thread) int g_tls = 7;

constexpr unsigned CALLER_TLS = 1;
constexpr unsigned CALLER_ENTRY = 2;

void NTAPI tls_callback(void* module, unsigned long reason, void* reserved) {
    (void)reserved;  // the observed TLS value travels in the sink's reserved slot by design
    sink_record(module, reason, reinterpret_cast<void*>(static_cast<std::uintptr_t>(g_tls)), CALLER_TLS);
}

#pragma section(".CRT$XLB", read)
extern "C" __declspec(allocate(".CRT$XLB")) PIMAGE_TLS_CALLBACK p_tls_value_callback = tls_callback;

BOOL WINAPI DllMain(HINSTANCE module, DWORD reason, LPVOID reserved) {
    (void)reserved;
    if (reason == DLL_PROCESS_ATTACH || reason == DLL_THREAD_ATTACH || reason == DLL_THREAD_DETACH ||
        reason == DLL_PROCESS_DETACH) {
        sink_record(module, reason, reinterpret_cast<void*>(static_cast<std::uintptr_t>(g_tls)),
                    CALLER_ENTRY);
    }
    return TRUE;
}

extern "C" __declspec(dllexport) int tls_value_read() { return g_tls; }

extern "C" __declspec(dllexport) void tls_value_set(int value) { g_tls = value; }

// Thread start routine: returns the calling thread's TLS value as the exit code.
extern "C" __declspec(dllexport) DWORD WINAPI tls_value_probe(LPVOID) {
    return static_cast<DWORD>(g_tls);
}
