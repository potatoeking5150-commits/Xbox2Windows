#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct _LUID;

// Windows can list one device under several DXGI LUIDs, and vendor/device IDs do not tell them
// apart from separate adapters of the same model; the PnP device keys do.
namespace adapter_identity {
struct Identity {
    std::uint32_t nodes = 0;            // physical adapters behind the LUID (DXGI nodes); 0 if unknown
    std::vector<std::wstring> devices;  // their PnP hardware keys; empty if any cannot be read
};

Identity query(const _LUID& luid) noexcept;

// Characters in the key a PnP key query wrote to `buffer` (`written` counts the terminator, as
// pCchDest reports), or 0 unless they fit `capacity` and form one nonempty terminated string.
std::size_t key_length(const wchar_t* buffer, std::uint32_t capacity, std::uint32_t written) noexcept;

// Whether two PnP keys name the same device. Registry key names are not case sensitive
// (RegOpenKeyExW), so this is CompareStringOrdinal with the OS uppercase table.
bool same_device(const std::wstring& a, const std::wstring& b) noexcept;
}
