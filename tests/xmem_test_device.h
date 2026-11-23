// D3D12 device for the xmem working-set tests (tests/xmem_shim_check.cpp,
// tests/cpu/working_set_xmem_check.cpp).
#pragma once

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

#include <cstdio>
#include <cwchar>

// As in offscreen_check: the first D3D12 hardware adapter in high-performance order, so a hybrid
// laptop measures its discrete GPU. With XBOX2WINDOWS_ADAPTER, the first whose description
// contains the value, ignoring case; a value that matches no adapter sets `unmatched`.
inline Microsoft::WRL::ComPtr<ID3D12Device> create_device(bool warp, bool& unmatched)
{
    unmatched = false;
    Microsoft::WRL::ComPtr<IDXGIFactory6> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return nullptr;
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    Microsoft::WRL::ComPtr<ID3D12Device> device;
    if (warp) {
        if (SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) {
            D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
        }
        return device;
    }
    wchar_t wanted[128]{};
    const DWORD filter = GetEnvironmentVariableW(L"XBOX2WINDOWS_ADAPTER", wanted, 128);
    if (filter >= 128) {
        std::puts("XBOX2WINDOWS_ADAPTER is longer than 127 characters");
        unmatched = true;
        return nullptr;
    }
    for (UINT i = 0; factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                          IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(adapter->GetDesc1(&desc)) || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
        if (filter) {
            bool match = false;
            for (const wchar_t* text = desc.Description; *text && !match; ++text)
                match = _wcsnicmp(text, wanted, filter) == 0;
            if (!match) continue;
        }
        if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) {
            std::printf("device adapter %04x:%04x %ls\n", desc.VendorId, desc.DeviceId, desc.Description);
            return device;
        }
    }
    if (filter) {
        std::printf("No D3D12 hardware adapter matches XBOX2WINDOWS_ADAPTER=%ls\n", wanted);
        unmatched = true;
    }
    return nullptr;
}
