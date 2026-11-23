#include "adapter_identity.h"

#include <climits>
#include <cwchar>
#include <iterator>
#include <new>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winternl.h> // NTSTATUS for d3dkmthk.h
#include <d3dkmthk.h>

namespace adapter_identity {
Identity query(const LUID& luid) noexcept
{
    D3DKMT_OPENADAPTERFROMLUID open{};
    open.AdapterLuid = luid;
    if (D3DKMTOpenAdapterFromLuid(&open) != 0) return {};
    const struct Close {
        D3DKMT_HANDLE adapter;
        ~Close()
        {
            D3DKMT_CLOSEADAPTER close{adapter};
            D3DKMTCloseAdapter(&close);
        }
    } close{open.hAdapter};
    UINT count = 0;
    D3DKMT_QUERYADAPTERINFO query{open.hAdapter, KMTQAITYPE_PHYSICALADAPTERCOUNT, &count, sizeof(count)};
    if (D3DKMTQueryAdapterInfo(&query) != 0) return {};
    Identity identity{count, {}};
    auto& devices = identity.devices;
    bool complete = count > 0;
    for (UINT i = 0; complete && i < count; ++i) {
        WCHAR key[1024];
        UINT length = static_cast<UINT>(std::size(key));
        D3DKMT_QUERY_PHYSICAL_ADAPTER_PNP_KEY pnp{i, D3DKMT_PNP_KEY_HARDWARE, key, &length};
        query = {open.hAdapter, KMTQAITYPE_PHYSICALADAPTERPNPKEY, &pnp, sizeof(pnp)};
        const std::size_t chars =
            D3DKMTQueryAdapterInfo(&query) == 0 ? key_length(key, static_cast<UINT>(std::size(key)), length) : 0;
        complete = chars != 0;
        try {
            if (complete) devices.emplace_back(key, chars);
        } catch (const std::bad_alloc&) {
            devices.clear();
            return identity;
        }
    }
    if (!complete) devices.clear();
    return identity;
}

std::size_t key_length(const wchar_t* buffer, std::uint32_t capacity, std::uint32_t written) noexcept
{
    if (written < 2 || written > capacity || buffer[written - 1] != L'\0') return 0;
    const std::size_t length = written - 1;
    return wcsnlen(buffer, length) == length ? length : 0;
}

bool same_device(const std::wstring& a, const std::wstring& b) noexcept
{
    return a.size() <= INT_MAX && b.size() <= INT_MAX &&
           CompareStringOrdinal(a.c_str(), static_cast<int>(a.size()), b.c_str(), static_cast<int>(b.size()), TRUE) ==
               CSTR_EQUAL;
}
}
