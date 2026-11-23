#include "xmem_gpu_usage.h"

#include "adapter_identity.h"

#include <new>
#include <utility>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

namespace xmem_gpu_usage {
std::optional<std::uint64_t> process_bytes() noexcept
{
    // A new factory per call keeps the adapter list current.
    Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return std::nullopt;
    std::vector<Domain> domains;
    bool complete = true;
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    HRESULT listed;
    for (UINT i = 0; (listed = factory->EnumAdapters1(i, &adapter)) != DXGI_ERROR_NOT_FOUND; ++i) {
        if (FAILED(listed)) {
            complete = false;
            break;
        }
        try {
            domains.emplace_back();
        } catch (const std::bad_alloc&) {
            return std::nullopt;
        }
        auto& domain = domains.back();
        DXGI_ADAPTER_DESC1 desc{};
        Microsoft::WRL::ComPtr<IDXGIAdapter3> memory;
        if (FAILED(adapter->GetDesc1(&desc)) || FAILED(adapter.As(&memory))) continue;
        auto identity = adapter_identity::query(desc.AdapterLuid);
        const auto bytes = domain_bytes(identity.nodes, [&](std::uint32_t node, int group, std::uint64_t& usage) {
            DXGI_QUERY_VIDEO_MEMORY_INFO info{};
            if (FAILED(memory->QueryVideoMemoryInfo(node, static_cast<DXGI_MEMORY_SEGMENT_GROUP>(group), &info))) {
                return false;
            }
            usage = info.CurrentUsage;
            return true;
        });
        domain.queried = bytes.has_value();
        domain.bytes = bytes.value_or(0);
        domain.devices = std::move(identity.devices);
    }
    return total(domains, complete);
}
}
