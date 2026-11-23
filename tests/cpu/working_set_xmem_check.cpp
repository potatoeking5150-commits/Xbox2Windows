// Feeds the desktop xmem.dll's XMemGetWorkingSetStatistics samples, taken around a real D3D12
// heap creation and release, into the upscaler's own counter updates after the queries at RVAs
// 0x74d2 and 0x75b1 (translated from the local package image by
// tools/ffx_working_set_delta.py --package).
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <string_view>

#include "../xmem_test_device.h"

using word = std::uint64_t;
extern "C" __declspec(dllimport) BOOL XMemGetWorkingSetStatistics(UINT32 selector, void* statistics);
extern "C" __declspec(dllimport) word create_delta(word input, word counters);
extern "C" __declspec(dllimport) word release_delta(word input, word counters);

namespace {
constexpr UINT32 amd_selector = 0x20000001;

// The translated code's input, as in tests/cpu/working_set_delta_check.py: the earlier sample,
// the raw value at +8 and status of the later query, and the aliasable flag.
struct Input {
    word before;
    word after;
    std::uint32_t status;
    std::uint32_t aliasable;
};

struct Sample {
    BOOL status;
    word value;
};

Sample query()
{
    std::array<word, 16> statistics;
    statistics.fill(0x5a5a5a5a5a5a5a5aull);
    const BOOL status = XMemGetWorkingSetStatistics(amd_selector, statistics.data());
    return {status, statistics[1]};
}

// The caller substitutes zero for a failed earlier query (0x7475 and 0x7577) before the
// translated range; the translated range does the same for the later query itself.
Input bracket(const Sample& before, const Sample& after, std::uint32_t aliasable)
{
    return {before.status ? before.value : 0, after.value, static_cast<std::uint32_t>(after.status), aliasable};
}
}

int main(int argc, char** argv)
{
    using namespace std::literals;
    if (argc != 2 || (argv[1] != "warp"sv && argv[1] != "hardware"sv)) return 2;
    const bool warp = argv[1] == "warp"sv;
    bool unmatched = false;
    const auto device = create_device(warp, unmatched);
    if (unmatched) return 1;
    if (!device) {
        std::puts(warp ? "No D3D12 device on WARP" : "No D3D12-capable hardware adapter visible");
        return 77;
    }

    constexpr word size = 64ull << 20;
    constexpr word total = 1000, aliasable_total = 2000;
    bool passed = true;
    for (const std::uint32_t aliasable : {0u, 1u}) {
        // The effect record's counters at +0x40 (total) and +0x48 (aliasable).
        std::array<word, 2> counters{total, aliasable_total};
        D3D12_HEAP_DESC desc{};
        desc.SizeInBytes = size;
        desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        desc.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;

        const Sample before_create = query();
        Microsoft::WRL::ComPtr<ID3D12Heap> heap;
        if (FAILED(device->CreateHeap(&desc, IID_PPV_ARGS(&heap)))) return 1;
        Input create = bracket(before_create, query(), aliasable);
        create_delta(reinterpret_cast<word>(&create), reinterpret_cast<word>(counters.data()));
        const auto created = counters;

        const Sample before_release = query();
        heap.Reset();
        Input release = bracket(before_release, query(), aliasable);
        release_delta(reinterpret_cast<word>(&release), reinterpret_cast<word>(counters.data()));

        const bool ok = created[0] == total + size && created[1] == aliasable_total + (aliasable ? size : 0) &&
                        counters[0] == total && counters[1] == aliasable_total;
        std::printf("aliasable=%u heap=%llu created total=+%llu aliasable=+%llu released total=+%llu aliasable=+%llu %s\n",
                    aliasable, static_cast<unsigned long long>(size),
                    static_cast<unsigned long long>(created[0] - total),
                    static_cast<unsigned long long>(created[1] - aliasable_total),
                    static_cast<unsigned long long>(counters[0] - total),
                    static_cast<unsigned long long>(counters[1] - aliasable_total), ok ? "ok" : "MISMATCH");
        passed &= ok;
    }
    std::printf("Package working-set counters from desktop xmem.dll %s: %s\n", argv[1], passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
