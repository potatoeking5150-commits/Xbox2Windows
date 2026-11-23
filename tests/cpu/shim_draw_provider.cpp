// Test-only draw observer. The provider renders the
// 64x64 test triangle offscreen, compares the readback against the CPU pixel rule
// and publishes the raw RGBA bytes so the check can compare runners byte for byte.
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "tex_tables.h"

bool shim_draw_gpu(unsigned mode, std::vector<unsigned char>& rgba);
bool shim_chain_gpu(unsigned mode, std::vector<unsigned char>& published, unsigned& mismatches);
bool shim_tex_gpu(unsigned mode, std::vector<unsigned char>& rgba);

bool shim_shadow_wait(void* request, std::vector<unsigned char>& output);

namespace {
constexpr unsigned side = 64;
constexpr unsigned tex_side = 8;
constexpr unsigned tex_scale = side / tex_side;

bool publish(const void* data, DWORD size)
{
    DWORD written = 0;
    return WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), data, size, &written, nullptr) && written == size;
}

unsigned mismatches(const std::vector<unsigned char>& rgba, bool red)
{
    unsigned count = 0;
    for (unsigned y = 0; y < side; ++y) {
        for (unsigned x = 0; x < side; ++x) {
            const auto* pixel = rgba.data() + (y * side + x) * 4;
            const bool inside = x >= 8 && y >= 8 && x + y <= 63;
            const unsigned char expected[4]{static_cast<unsigned char>(inside && red ? 255 : 0),
                                            static_cast<unsigned char>(inside && !red ? 255 : 0), 0, 255};
            if (std::memcmp(pixel, expected, 4) != 0) ++count;
        }
    }
    return count;
}

// The textured quad magnifies each 8x8 table texel to an 8x8 pixel block, so the
// CPU rule is a direct table lookup: the same rule tex_gen.py enforces.
unsigned tex_mismatches(const std::vector<unsigned char>& rgba, const unsigned char* table)
{
    unsigned count = 0;
    for (unsigned y = 0; y < side; ++y) {
        for (unsigned x = 0; x < side; ++x) {
            const auto* pixel = rgba.data() + (y * side + x) * 4;
            const auto* expected = table + ((y / tex_scale) * tex_side + x / tex_scale) * 4;
            if (std::memcmp(pixel, expected, 4) != 0) ++count;
        }
    }
    return count;
}

// The scene draws the RGBA8 table on the left half and the BC7 table on the
// right half, each squeezed to 32x64: texel (x / 4, y / 8) of the half table.
unsigned scene_mismatches(const std::vector<unsigned char>& rgba)
{
    unsigned count = 0;
    for (unsigned y = 0; y < side; ++y) {
        for (unsigned x = 0; x < side; ++x) {
            const auto* pixel = rgba.data() + (y * side + x) * 4;
            const unsigned tx = (x < side / 2 ? x : x - side / 2) / 4;
            const unsigned char* table = x < side / 2 ? tex_rgba8 : tex_bc7;
            const auto* expected = table + ((y / tex_scale) * tex_side + tx) * 4;
            if (std::memcmp(pixel, expected, 4) != 0) ++count;
        }
    }
    return count;
}
} // namespace

extern "C" __declspec(dllexport) HRESULT shadow_wait(void* request)
{
    std::vector<unsigned char> output;
    if (!shim_shadow_wait(request, output)) return E_FAIL;
    if (!publish(output.data(), static_cast<DWORD>(output.size()))) ExitProcess(23);
    return S_OK;
}

extern "C" __declspec(dllexport) DWORD draw_request()
{
    char mode[16]{};
    const DWORD len = GetEnvironmentVariableA("X2W_SHIM_DRAW_CASE", mode, sizeof(mode));
    if (!len) return 0;
    if (len < sizeof(mode)) {
        if (!std::strcmp(mode, "draw")) return 1;
        if (!std::strcmp(mode, "flip")) return 2;
        // The translated compute -> mesh -> pixel chain (tests/cpu/shim_chain_gpu.cpp).
        if (!std::strcmp(mode, "chain0")) return 3;
        if (!std::strcmp(mode, "chain1")) return 4;
        if (!std::strcmp(mode, "chain-stale")) return 5;
        if (!std::strcmp(mode, "two")) return 6;
        if (!std::strcmp(mode, "stale")) return 7;
        if (!std::strcmp(mode, "fail-second")) return 8;
        if (!std::strcmp(mode, "shadow")) return 14;
        if (!std::strcmp(mode, "shadow-stale")) return 15;
        if (!std::strcmp(mode, "shadow-reject")) return 16;
        if (!std::strcmp(mode, "shadow-fail")) return 17;
        if (!std::strcmp(mode, "texture")) return 10;
        if (!std::strcmp(mode, "texture-stale")) return 12;
        // Uploaded textures through a vertex-buffer quad (tests/cpu/shim_tex_gpu.cpp).
        if (!std::strcmp(mode, "tex")) return 20;
        if (!std::strcmp(mode, "texuv")) return 21;
        if (!std::strcmp(mode, "texbad")) return 22;
        if (!std::strcmp(mode, "bc7")) return 23;
        if (!std::strcmp(mode, "scene")) return 24;
    }
    ExitProcess(21);
}

extern "C" __declspec(dllexport) DWORD draw_frame(DWORD mode)
{
    if (mode >= 20 && mode <= 24) {
        std::vector<unsigned char> rgba;
        if (!shim_tex_gpu(mode - 17, rgba) || rgba.size() != side * side * 4) return 0xFFFFFFFF;
        if (!publish(rgba.data(), static_cast<DWORD>(rgba.size()))) ExitProcess(23);
        if (mode == 23) return tex_mismatches(rgba, tex_bc7);
        if (mode == 24) return scene_mismatches(rgba);
        return tex_mismatches(rgba, tex_rgba8);  // texuv and texbad must differ from the exact rule
    }
    if (mode >= 3 && mode <= 5) {
        std::vector<unsigned char> published;
        unsigned count = 0;
        if (!shim_chain_gpu(mode, published, count)) return 0xFFFFFFFF;
        if (!publish(published.data(), static_cast<DWORD>(published.size()))) ExitProcess(23);
        return count;
    }
    if (mode < 1 || mode > 13) ExitProcess(22);
    if (mode >= 6) std::fprintf(stderr, "draw_frame mode=%lu\n", static_cast<unsigned long>(mode));
    std::vector<unsigned char> rgba;
    if (!shim_draw_gpu(mode, rgba) || rgba.size() != side * side * 4) return 0xFFFFFFFF;
    if (!publish(rgba.data(), static_cast<DWORD>(rgba.size()))) ExitProcess(23);
    if (mode >= 10) return 0;  // The cross-runner reference checks all texture pixels.
    return mismatches(rgba, mode == 7 || mode == 8);
}
