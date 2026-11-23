// Calls the desktop xmem.dll through named imports, as the package DLLs do.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "adapter_identity.h"
#include "xmem_gpu_usage.h"
#include "xmem_test_device.h"

extern "C" __declspec(dllimport) void* XMemAlloc(SIZE_T bytes, ULONGLONG attributes);
extern "C" __declspec(dllimport) void XMemFree(void* allocation, ULONGLONG attributes);
extern "C" __declspec(dllimport) BOOL XMemVirtualQuery(HANDLE process, LPCVOID address, void* buffer);
extern "C" __declspec(dllimport) BOOL XMemGetWorkingSetStatistics(UINT32 selector, void* statistics);

using namespace std::literals;
using Byte = unsigned char;

namespace {
// Attribute words and selector recorded at the Bink, AMD and BVH call sites.
constexpr ULONGLONG bink_word = 0x05e50000;
constexpr ULONGLONG amd_word = 0xa8004000;
constexpr UINT32 amd_selector = 0x20000001;

struct QueryBuffer {
    MEMORY_BASIC_INFORMATION info;
    std::array<Byte, 16> tail;
};
static_assert(offsetof(QueryBuffer, tail) == 0x30 && sizeof(QueryBuffer) == 64);

// The BVH callers' post-query predicates, as modeled in memory_query_check.cpp, applied to the
// 32-bit result they test.
bool caller_accepts(const QueryBuffer& buffer, BOOL returned, bool source)
{
    return returned && (buffer.tail[0] & 3) && (buffer.info.State & 0x1000) &&
           (!source || (buffer.info.Protect & 6));
}

BOOL ask(const void* address, QueryBuffer& buffer, HANDLE process = GetCurrentProcess())
{
    buffer = {};
    std::fill(buffer.tail.begin() + 1, buffer.tail.end(), Byte{0xa5});
    return XMemVirtualQuery(process, address, &buffer);
}

bool export_checks()
{
    HMODULE module = GetModuleHandleW(L"xmem.dll");
    if (!module) return false;
    unsigned found = 0;
    for (const char* name : {"XMemAlloc", "XMemFree", "XMemVirtualQuery", "XMemGetWorkingSetStatistics"}) {
        if (!GetProcAddress(module, name)) return false;
        ++found;
    }
    // The named imports in this executable resolve to the same module.
    HMODULE owner = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(GetProcAddress(module, "XMemAlloc")), &owner) ||
        owner != module) return false;
    // Routing graphics memory through era_title_gpu must not make xmem.dll load D3D12, and
    // XMemGetWorkingSetStatistics loads DXGI only when called. This executable delay-loads both
    // for its GPU scenarios.
    if (GetModuleHandleW(L"d3d12.dll") || GetModuleHandleW(L"dxgi.dll")) return false;
    std::printf("exports=%u\n", found);
    return true;
}

bool allocation_checks()
{
    // Bink: heap memory, 32-byte alignment, zero-filled.
    auto* heap = static_cast<Byte*>(XMemAlloc(97, bink_word));
    if (!heap || reinterpret_cast<std::uintptr_t>(heap) % 32 ||
        !std::all_of(heap, heap + 97, [](Byte b) { return b == 0; })) return false;
    std::fill_n(heap, 97, Byte{0x51});
    XMemFree(heap, bink_word);

    // AMD: graphics memory, committed read/write on the allocation granularity, freed by XMemFree.
    auto* gpu = static_cast<Byte*>(XMemAlloc(512, amd_word));
    MEMORY_BASIC_INFORMATION info{};
    if (!gpu || VirtualQuery(gpu, &info, sizeof(info)) != sizeof(info) || info.State != MEM_COMMIT ||
        info.Protect != PAGE_READWRITE || reinterpret_cast<std::uintptr_t>(gpu) % (std::size_t{64} << 10) ||
        !std::all_of(gpu, gpu + 512, [](Byte b) { return b == 0; })) return false;
    std::fill_n(gpu, 512, Byte{0x51});
    XMemFree(gpu, amd_word);
    if (VirtualQuery(gpu, &info, sizeof(info)) != sizeof(info) || info.State != MEM_FREE) return false;

    // Rejections pass straight through, and freeing null is a no-op for both types.
    XMemFree(nullptr, bink_word);
    XMemFree(nullptr, amd_word);
    return XMemAlloc(0, bink_word) == nullptr && XMemAlloc(64, bink_word | (ULONGLONG{1} << 40)) == nullptr &&
           XMemAlloc(512, (amd_word & ~(ULONGLONG{15} << 29)) | (ULONGLONG{7} << 29)) == nullptr;
}

bool query_checks()
{
    QueryBuffer buffer;
    // Graphics memory from XMemAlloc is registered, so both caller predicates accept it.
    auto* gpu = XMemAlloc(4096, amd_word);
    if (!gpu) return false;
    BOOL ok = ask(gpu, buffer);
    if (!ok || buffer.tail[0] != 1 || !caller_accepts(buffer, ok, false) || !caller_accepts(buffer, ok, true) ||
        !std::all_of(buffer.tail.begin() + 1, buffer.tail.end(), [](Byte b) { return b == 0xa5; })) return false;
    // A real handle to this process behaves like the pseudo-handle.
    HANDLE self = OpenProcess(PROCESS_QUERY_INFORMATION, FALSE, GetCurrentProcessId());
    if (!self) return false;
    ok = ask(static_cast<Byte*>(gpu) + 100, buffer, self);
    CloseHandle(self);
    if (!ok || buffer.tail[0] != 1) return false;
    // After XMemFree the region is gone and no longer registered.
    XMemFree(gpu, amd_word);
    ok = ask(gpu, buffer);
    if (!ok || buffer.info.State != MEM_FREE || buffer.tail[0] != 0 || caller_accepts(buffer, ok, false)) return false;
    // A freshly reused address must not inherit the old registration.
    auto* again = XMemAlloc(4096, amd_word);
    if (!again) return false;
    XMemFree(again, amd_word);

    // Heap memory is not GPU-intended under the desktop policy: the tail check fails.
    auto* heap = XMemAlloc(257, bink_word);
    if (!heap) return false;
    ok = ask(heap, buffer);
    const bool heap_rejected = ok && buffer.info.State == MEM_COMMIT && buffer.tail[0] == 0 &&
                               !caller_accepts(buffer, ok, false);
    XMemFree(heap, bink_word);
    if (!heap_rejected) return false;

    // An address outside user space fails without touching the buffer.
    buffer = {};
    buffer.tail.fill(0xd7);
    const BOOL outside = XMemVirtualQuery(GetCurrentProcess(), reinterpret_cast<LPCVOID>(0xffff800000000000ull), &buffer);
    if (outside || buffer.info.State != 0 ||
        !std::all_of(buffer.tail.begin(), buffer.tail.end(), [](Byte b) { return b == 0xd7; })) return false;
    SetLastError(ERROR_SUCCESS);
    const BOOL null_output = XMemVirtualQuery(GetCurrentProcess(), &buffer, nullptr);
    const DWORD null_error = GetLastError();
    std::printf("query_cases=7 null_output_error=%lu\n", null_error);
    return !null_output && null_error == ERROR_NOACCESS;
}

// The AMD callers' 128-byte stack buffer, filled with a sentinel.
using Statistics = std::array<std::uint64_t, 16>;
static_assert(sizeof(Statistics) == 128);
constexpr std::uint64_t sentinel = 0x5a5a5a5a5a5a5a5aull;

// What an AMD caller reads: the 64-bit value at +8 on success, zero otherwise.
std::uint64_t caller_reads()
{
    Statistics statistics;
    statistics.fill(sentinel);
    return XMemGetWorkingSetStatistics(amd_selector, statistics.data()) ? statistics[1] : 0;
}

bool working_set_checks()
{
    // The AMD selector succeeds and writes only the qword at +8.
    Statistics statistics;
    statistics.fill(sentinel);
    const BOOL ok = XMemGetWorkingSetStatistics(amd_selector, statistics.data());
    statistics[1] = sentinel;
    const auto untouched = [&] {
        return std::all_of(statistics.begin(), statistics.end(), [](std::uint64_t v) { return v == sentinel; });
    };
    const bool rest_untouched = untouched();

    // Another selector stays unsupported and leaves the output untouched.
    SetLastError(ERROR_SUCCESS);
    const BOOL other = XMemGetWorkingSetStatistics(amd_selector + 1, statistics.data());
    const DWORD other_error = GetLastError();
    SetLastError(ERROR_SUCCESS);
    const BOOL null_output = XMemGetWorkingSetStatistics(amd_selector, nullptr);
    const DWORD null_error = GetLastError();
    std::printf("amd_selector=%d rest_untouched=%d other_selector=%d error=%lu null_output=%d error=%lu\n", ok,
                rest_untouched, other, other_error, null_output, null_error);
    return ok && rest_untouched && !other && other_error == ERROR_NOT_SUPPORTED && untouched() && !null_output &&
           null_error == ERROR_INVALID_PARAMETER;
}

// Synthetic PnP key query output for adapter_identity::key_length. `written` counts the terminator;
// anything malformed gives 0, which leaves the LUID's identity unknown.
bool key_checks()
{
    const std::wstring key = LR"(PCI\VEN_10DE&DEV_28E0\4&1&0&0008)";
    const std::wstring terminated = key + L'\0';
    const auto size = static_cast<std::uint32_t>(terminated.size());
    struct Case {
        const char* name;
        std::wstring buffer;
        std::uint32_t capacity;
        std::uint32_t written;
        std::size_t expected;
    };
    const Case cases[] = {
        {"key_well_formed", terminated, size + 8, size, key.size()},
        {"key_fills_buffer", terminated, size, size, key.size()},
        {"key_nothing_written", terminated, size, 0, 0},
        {"key_empty", L"\0"s, 1, 1, 0},
        {"key_past_buffer", terminated + L'\0', size, size + 1, 0},
        {"key_unterminated", key, size - 1, size - 1, 0},
        {"key_embedded_terminator", L"PCI\0VEN_10DE\0"s, 13, 13, 0},
    };
    bool passed = true;
    for (const auto& c : cases) {
        const auto length = adapter_identity::key_length(c.buffer.data(), c.capacity, c.written);
        const bool ok = length == c.expected;
        passed &= ok;
        std::printf("%s length=%zu %s\n", c.name, length, ok ? "ok" : "MISMATCH");
    }
    return passed;
}

// Synthetic QueryVideoMemoryInfo results for xmem_gpu_usage::domain_bytes, one LUID's share of the
// value at +8, passed on to total() as the shim does.
bool node_checks()
{
    using Reading = std::optional<std::uint64_t>; // nothing: the query fails
    using Nodes = std::vector<std::array<Reading, 2>>; // per node: local, non-local
    constexpr std::uint64_t mib = 1ull << 20;
    const std::wstring gpu_a = LR"(PCI\VEN_10DE&DEV_28E0\4&1&0&0008)";
    const std::wstring gpu_b = LR"(PCI\VEN_10DE&DEV_28E0\4&2&0&0010)";
    const std::wstring igpu = LR"(PCI\VEN_8086&DEV_A7A0\3&1&0&0010)";
    struct Case {
        const char* name;
        std::uint32_t nodes; // as adapter_identity reports; 0 when unknown
        Nodes readings;
        std::vector<std::wstring> devices;
        std::optional<std::uint64_t> expected;
    };
    const Case cases[] = {
        {"single_node_segments", 1, {{64 * mib, 2 * mib}}, {gpu_a}, 66 * mib},
        // A linked pair: nothing on node 0, so reading node 0 alone would report 0.
        {"node1_only", 2, {{0, 0}, {64 * mib, 0}}, {gpu_a, gpu_b}, 64 * mib},
        {"node_count_unknown", 0, {{0, 0}}, {}, std::nullopt},
        {"segment_failed_after_read", 1, {{64 * mib, std::nullopt}}, {gpu_a}, std::nullopt},
        {"node1_failed_after_node0", 2, {{64 * mib, 0}, {std::nullopt, 0}}, {gpu_a, gpu_b}, std::nullopt},
        {"segment_overflow", 1, {{UINT64_MAX, 1}}, {gpu_a}, std::nullopt},
        {"node_overflow", 2, {{UINT64_MAX, 0}, {1, 0}}, {gpu_a, gpu_b}, std::nullopt},
    };
    bool passed = true;
    for (const auto& c : cases) {
        const auto read = [&](std::uint32_t node, int group, std::uint64_t& bytes) noexcept {
            if (node >= c.readings.size() || !c.readings[node][group]) return false;
            bytes = *c.readings[node][group];
            return true;
        };
        const auto bytes = xmem_gpu_usage::domain_bytes(c.nodes, read);
        // The domain next to an idle integrated GPU, so total() must decline whenever domain_bytes() does.
        const std::vector<xmem_gpu_usage::Domain> domains{{bytes.has_value(), bytes.value_or(0), c.devices},
                                                          {true, 0, {igpu}}};
        const auto result = xmem_gpu_usage::total(domains, true);
        const bool ok = bytes == c.expected && result == c.expected;
        passed &= ok;
        if (result) {
            std::printf("%s total=%llu %s\n", c.name, static_cast<unsigned long long>(*result), ok ? "ok" : "MISMATCH");
        } else {
            std::printf("%s declined %s\n", c.name, ok ? "ok" : "MISMATCH");
        }
    }
    return passed;
}

// Synthetic per-adapter readings for xmem_gpu_usage::total, the aggregation behind the value at +8.
bool aggregation_checks()
{
    using xmem_gpu_usage::Domain;
    using Domains = std::vector<Domain>;
    constexpr std::uint64_t mib = 1ull << 20;
    const std::wstring gpu_a = LR"(PCI\VEN_10DE&DEV_28E0\4&1&0&0008)";
    const std::wstring gpu_b = LR"(PCI\VEN_10DE&DEV_28E0\4&2&0&0010)";
    const std::wstring igpu = LR"(PCI\VEN_8086&DEV_A7A0\3&1&0&0010)";
    const std::wstring basic = LR"(ROOT\BasicRender\0000)";
    const std::wstring gpu_a_lower = LR"(pci\ven_10de&dev_28e0\4&1&0&0008)";
    const auto domain = [](std::uint64_t bytes, std::vector<std::wstring> devices = {}) {
        return Domain{true, bytes, std::move(devices)};
    };
    const auto failed = [] { return Domain{}; };

    // Regression reference (Windows build 26200, RTX 4060 Laptop, driver 32.0.16.1656): two LUIDs for
    // 10de:28e0, both flags 0, both moving 7,294,976 -> 74,403,840 bytes on one 64 MiB heap; the
    // Intel and Basic Render Driver entries stay at zero. Vendor and device IDs matched and the
    // readings were equal, but neither shows they are one device, so without an identity the
    // call must fail rather than report 2x.
    const auto laptop = [&](std::uint64_t bytes, bool identified) {
        const auto key = [&](const std::wstring& device) {
            return identified ? std::vector<std::wstring>{device} : std::vector<std::wstring>{};
        };
        return Domains{domain(bytes, key(gpu_a)), domain(bytes, key(gpu_a)), domain(0, key(igpu)),
                       domain(0, key(basic))};
    };
    const auto before = xmem_gpu_usage::total(laptop(7294976, false), true);
    const auto after = xmem_gpu_usage::total(laptop(74403840, false), true);
    const auto alias_before = xmem_gpu_usage::total(laptop(7294976, true), true);
    const auto alias_after = xmem_gpu_usage::total(laptop(74403840, true), true);

    struct Case {
        const char* name;
        Domains domains;
        bool complete;
        std::optional<std::uint64_t> expected;
    };
    const Case cases[] = {
        // GPU-less runner: Basic Render Driver with flags 0 and as a software adapter, one device.
        {"basic_render_alias", {domain(64 * mib, {basic}), domain(64 * mib, {basic})}, true, 64 * mib},
        // Two adapters of the same model at different PCI locations are both counted, even with equal readings.
        {"same_model_distinct", {domain(64 * mib, {gpu_a}), domain(64 * mib, {gpu_b}), domain(0, {igpu})}, true,
         128 * mib},
        {"genuine_zero", {domain(0, {gpu_a}), domain(0), domain(0, {basic})}, true, 0},
        {"one_unidentified_active", {domain(0, {gpu_a}), domain(32 * mib)}, true, 32 * mib},
        {"all_queries_failed", {failed(), failed(), failed()}, true, std::nullopt},
        {"one_query_failed", {domain(64 * mib, {gpu_a}), failed()}, true, std::nullopt},
        {"incomplete_enumeration", {domain(64 * mib, {gpu_a})}, false, std::nullopt},
        {"no_adapters", {}, true, std::nullopt},
        {"alias_disagrees", {domain(64 * mib, {basic}), domain(32 * mib, {basic})}, true, std::nullopt},
        {"unidentified_with_active", {domain(64 * mib, {gpu_a}), domain(64 * mib)}, true, std::nullopt},
        {"linked_partial_overlap", {domain(64 * mib, {gpu_a, gpu_b}), domain(64 * mib, {gpu_a})}, true, std::nullopt},
        {"overflow", {domain(UINT64_MAX, {gpu_a}), domain(1, {gpu_b})}, true, std::nullopt},
        // Registry keys match ignoring case, so a case variant of a key is the same device.
        {"mixed_case_alias", {domain(64 * mib, {gpu_a}), domain(64 * mib, {gpu_a_lower})}, true, 64 * mib},
        {"mixed_case_alias_disagrees", {domain(64 * mib, {gpu_a}), domain(32 * mib, {gpu_a_lower})}, true,
         std::nullopt},
        {"mixed_case_partial_overlap", {domain(64 * mib, {gpu_a, gpu_b}), domain(64 * mib, {gpu_a_lower})}, true,
         std::nullopt},
    };
    bool passed = !before && !after && alias_before == 7294976ull && alias_after == 74403840ull &&
                  *alias_after - *alias_before == 64 * mib;
    std::printf("laptop_reference unidentified=%s identified_delta=%lld\n", before || after ? "published" : "declined",
                alias_before && alias_after ? static_cast<long long>(*alias_after - *alias_before) : -1ll);
    for (const auto& c : cases) {
        const auto result = xmem_gpu_usage::total(c.domains, c.complete);
        const bool ok = result == c.expected;
        passed &= ok;
        if (result) {
            std::printf("%s total=%llu %s\n", c.name, static_cast<unsigned long long>(*result), ok ? "ok" : "MISMATCH");
        } else {
            std::printf("%s declined %s\n", c.name, ok ? "ok" : "MISMATCH");
        }
    }
    const bool nodes = node_checks();
    const bool keys = key_checks();
    return nodes && keys && passed;
}

// Lists the DXGI adapters with the identity the shim aggregates by, for GPU reports.
void describe_adapters()
{
    Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return;
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(adapter->GetDesc1(&desc))) continue;
        const auto identity = adapter_identity::query(desc.AdapterLuid);
        std::wstring devices;
        for (const auto& device : identity.devices) {
            devices += (devices.empty() ? L"" : L";") + device;
        }
        std::printf("adapter %u %04x:%04x flags=%u luid=%08lx:%08lx nodes=%u device=%ls\n", i, desc.VendorId,
                    desc.DeviceId, desc.Flags, desc.AdapterLuid.HighPart, desc.AdapterLuid.LowPart, identity.nodes,
                    devices.empty() ? L"(unknown)" : devices.c_str());
    }
}

// Measures as the AMD callers do: reads the value before creating an object, after creating it
// and after releasing it. Creation must add at least the allocation size and release must remove
// as much. The slack allows for allocations the driver makes on its own between two reads.
template <typename Create>
bool bracket(const char* what, std::uint64_t size, Create create)
{
    constexpr std::int64_t slack = 4 << 20;
    const std::uint64_t before = caller_reads();
    auto object = create();
    if (!object) {
        std::printf("%s: creation failed\n", what);
        return false;
    }
    const std::uint64_t created = caller_reads();
    object.Reset();
    const std::uint64_t released = caller_reads();
    const auto n = static_cast<std::int64_t>(size);
    const auto added = static_cast<std::int64_t>(created - before);
    const auto removed = static_cast<std::int64_t>(created - released);
    const bool ok = added >= n && added <= n + slack && removed >= n && removed <= n + slack;
    std::printf("%s size=%lld added=%lld removed=%lld %s\n", what, static_cast<long long>(n),
                static_cast<long long>(added), static_cast<long long>(removed), ok ? "ok" : "MISMATCH");
    return ok;
}

// Returns 0 on pass, 1 on failure (including an unmatched XBOX2WINDOWS_ADAPTER), 77 (skip)
// without the adapter or a D3D12 device.
int working_set_gpu_checks(bool warp)
{
    describe_adapters();
    bool unmatched = false;
    const auto device = create_device(warp, unmatched);
    if (unmatched) return 1;
    if (!device) {
        std::puts(warp ? "No D3D12 device on WARP" : "No D3D12-capable hardware adapter visible");
        return 77;
    }
    constexpr std::uint64_t mib = 1ull << 20;

    bool passed = true;
    const struct {
        D3D12_HEAP_TYPE type;
        const char* name;
    } heaps[] = {{D3D12_HEAP_TYPE_DEFAULT, "heap_default"},
                 {D3D12_HEAP_TYPE_UPLOAD, "heap_upload"},
                 {D3D12_HEAP_TYPE_READBACK, "heap_readback"}};
    for (const auto& heap : heaps) {
        D3D12_HEAP_DESC desc{};
        desc.SizeInBytes = 64 * mib;
        desc.Properties.Type = heap.type;
        desc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        desc.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
        passed &= bracket(heap.name, desc.SizeInBytes, [&] {
            Microsoft::WRL::ComPtr<ID3D12Heap> object;
            device->CreateHeap(&desc, IID_PPV_ARGS(&object));
            return object;
        });
    }

    D3D12_HEAP_PROPERTIES properties{};
    properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = 32 * mib;
    buffer.Height = 1;
    buffer.DepthOrArraySize = 1;
    buffer.MipLevels = 1;
    buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    const std::uint64_t size = device->GetResourceAllocationInfo(0, 1, &buffer).SizeInBytes;
    passed &= bracket("committed_buffer", size, [&] {
        Microsoft::WRL::ComPtr<ID3D12Resource> object;
        device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COMMON,
                                        nullptr, IID_PPV_ARGS(&object));
        return object;
    });
    return passed ? 0 : 1;
}
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        return 2;
    }
    const std::string_view scenario = argv[1];
    bool passed;
    if (scenario == "exports"sv) {
        passed = export_checks();
    } else if (scenario == "allocation"sv) {
        passed = allocation_checks();
    } else if (scenario == "query"sv) {
        passed = query_checks();
    } else if (scenario == "working_set"sv) {
        passed = working_set_checks();
    } else if (scenario == "working_set_aggregation"sv) {
        passed = aggregation_checks();
    } else if (scenario == "working_set_warp"sv || scenario == "working_set_hardware"sv) {
        const int result = working_set_gpu_checks(scenario == "working_set_warp"sv);
        if (result == 77) return result;
        passed = result == 0;
    } else {
        return 2;
    }
    std::printf("Desktop xmem.dll %s: %s\n", argv[1], passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
