// Shared WARP/hardware device selection for the shim-draw providers.
// Both providers honor X2W_SHIM_DRAW_ADAPTER (warp default, hardware opts in,
// anything else is a hard error); hardware probing exits 77 when no physical
// D3D12 adapter exists so CTest records a capability skip.
#pragma once
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdio>
#include <cstring>

inline void shim_require(HRESULT hr, const char* operation)
{
    if (FAILED(hr)) {
        std::fprintf(stderr, "%s failed (HRESULT 0x%08lx)\n", operation, static_cast<unsigned long>(hr));
        throw hr;
    }
}

// Prints the failing operation and returns its HRESULT. For code that cannot rely
// on C++ exceptions, such as images mapped by pe_map.
inline HRESULT shim_report(HRESULT hr, const char* operation) noexcept
{
    if (FAILED(hr))
        std::fprintf(stderr, "%s failed (HRESULT 0x%08lx)\n", operation, static_cast<unsigned long>(hr));
    return hr;
}

inline HRESULT shim_try_create_device(Microsoft::WRL::ComPtr<ID3D12Device>& device) noexcept
{
    using Microsoft::WRL::ComPtr;
    ComPtr<IDXGIFactory6> factory;
    HRESULT hr = shim_report(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");
    if (FAILED(hr)) return hr;
    ComPtr<IDXGIAdapter1> adapter;
    char backend[16]{};
    const auto size = GetEnvironmentVariableA("X2W_SHIM_DRAW_ADAPTER", backend, sizeof(backend));
    if (!size || (size < sizeof(backend) && std::strcmp(backend, "warp") == 0)) {
        hr = shim_report(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "EnumWarpAdapter");
        if (FAILED(hr)) return hr;
    } else if (size < sizeof(backend) && std::strcmp(backend, "hardware") == 0) {
        for (UINT i = 0;; ++i) {
            hr = factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter));
            if (hr == DXGI_ERROR_NOT_FOUND) {
                std::fputs("no D3D12 physical adapter\n", stderr);
                ExitProcess(77);
            }
            if (FAILED(shim_report(hr, "EnumAdapterByGpuPreference"))) return hr;
            DXGI_ADAPTER_DESC1 desc{};
            hr = shim_report(adapter->GetDesc1(&desc), "GetDesc1");
            if (FAILED(hr)) return hr;
            if (!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
                const HRESULT support = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                                          __uuidof(ID3D12Device), nullptr);
                if (support != DXGI_ERROR_UNSUPPORTED) {
                    if (FAILED(shim_report(support, "Probe physical device"))) return support;
                    break;
                }
            }
            adapter.Reset();
        }
    } else {
        return shim_report(E_INVALIDARG, "Select draw adapter");
    }
    return shim_report(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)),
                       "D3D12CreateDevice");
}

inline Microsoft::WRL::ComPtr<ID3D12Device> shim_create_device()
{
    Microsoft::WRL::ComPtr<ID3D12Device> device;
    const HRESULT hr = shim_try_create_device(device);
    if (FAILED(hr)) throw hr;
    return device;
}
