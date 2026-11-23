// This process's GPU memory in bytes, the value XMemGetWorkingSetStatistics publishes.
// process_bytes() reads DXGI; total() combines the readings without querying adapters, so
// tests/xmem_shim_check.cpp can feed it synthetic ones.
#pragma once

#include "adapter_identity.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace xmem_gpu_usage {

// One DXGI adapter (one LUID).
struct Domain {
    bool queried = false;              // domain_bytes() gave a value
    std::uint64_t bytes = 0;           // that value
    std::vector<std::wstring> devices; // adapter_identity::Identity::devices; empty if unknown
};

namespace detail {
// Adds `bytes` to `sum`; false, leaving `sum` unchanged, when the result would not fit.
inline bool add(std::uint64_t& sum, std::uint64_t bytes) noexcept
{
    if (bytes > UINT64_MAX - sum) return false;
    sum += bytes;
    return true;
}

// Keys are compared with adapter_identity::same_device, so case variants of one key match.
inline bool has_device(const Domain& domain, const std::wstring& device) noexcept
{
    return std::any_of(domain.devices.begin(), domain.devices.end(),
                       [&](const std::wstring& other) { return adapter_identity::same_device(other, device); });
}

inline bool overlap(const Domain& a, const Domain& b) noexcept
{
    return std::any_of(a.devices.begin(), a.devices.end(), [&](const std::wstring& d) { return has_device(b, d); });
}

inline bool same_devices(const Domain& a, const Domain& b) noexcept
{
    const auto within = [](const Domain& x, const Domain& y) {
        return std::all_of(x.devices.begin(), x.devices.end(), [&](const std::wstring& d) { return has_device(y, d); });
    };
    return within(a, b) && within(b, a);
}

// Whether `domain` is the first view of its devices: false for an alias of an earlier domain,
// nothing when that alias disagrees or the devices partly overlap an earlier domain's.
inline std::optional<bool> first_view(const Domain* begin, const Domain& domain) noexcept
{
    if (domain.devices.empty()) return true;
    for (const Domain* other = begin; other != &domain; ++other) {
        if (same_devices(*other, domain)) {
            if (other->bytes != domain.bytes) return std::nullopt;
            return false;
        }
        if (overlap(*other, domain)) return std::nullopt;
    }
    return true;
}
}

// One LUID's usage: local plus non-local on every node in [0, nodes), or nothing when the node
// count is unknown (0), a read fails or the sum overflows. read(node, group, bytes) returns whether
// QueryVideoMemoryInfo succeeded; group is a DXGI_MEMORY_SEGMENT_GROUP value.
template <class Read>
std::optional<std::uint64_t> domain_bytes(std::uint32_t nodes, const Read& read) noexcept
{
    if (nodes == 0) return std::nullopt;
    std::uint64_t sum = 0;
    for (std::uint32_t node = 0; node < nodes; ++node) {
        for (const int group : {0, 1}) {
            std::uint64_t bytes = 0;
            if (!read(node, group, bytes) || !detail::add(sum, bytes)) return std::nullopt;
        }
    }
    return sum;
}

// Reads every DXGI adapter (xmem_gpu_usage.cpp) and returns total() of the readings.
std::optional<std::uint64_t> process_bytes() noexcept;

// The process total, or nothing unless it is complete and non-duplicated: LUIDs with the same
// devices count once, distinct devices are summed, and an active adapter of unknown identity is
// accepted only when no other adapter is active. Rules: docs/experiments.md, "Desktop xmem.dll".
inline std::optional<std::uint64_t> total(const std::vector<Domain>& domains, bool enumeration_complete) noexcept
{
    const auto failed = [](const Domain& domain) { return !domain.queried; };
    if (!enumeration_complete || domains.empty() || std::any_of(domains.begin(), domains.end(), failed)) {
        return std::nullopt;
    }
    std::uint64_t sum = 0;
    std::size_t active = 0;
    bool unknown_active = false;
    for (const auto& domain : domains) {
        const auto first = detail::first_view(domains.data(), domain);
        if (!first) return std::nullopt;
        if (!*first || domain.bytes == 0) continue;
        if (!detail::add(sum, domain.bytes)) return std::nullopt;
        ++active;
        unknown_active |= domain.devices.empty();
    }
    // An adapter of unknown identity may be a second view of any other active one.
    if (unknown_active && active > 1) return std::nullopt;
    return sum;
}

}
