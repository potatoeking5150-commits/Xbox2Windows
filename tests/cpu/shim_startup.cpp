// Exercise the desktop providers from a DLL reached through EXE startup.
#include <d3d12.h>
#include <wrl/client.h>

#include <array>
#include <cstddef>
#include <cstdint>

extern "C" __declspec(dllimport) void* XMemAlloc(SIZE_T bytes, ULONGLONG attributes);
extern "C" __declspec(dllimport) void XMemFree(void* allocation, ULONGLONG attributes);
extern "C" __declspec(dllimport) BOOL XMemVirtualQuery(HANDLE process, LPCVOID address, void* buffer);
extern "C" __declspec(dllimport) BOOL XMemGetWorkingSetStatistics(UINT32 selector, void* statistics);

namespace {
using Microsoft::WRL::ComPtr;

unsigned process_attaches = 0;

template <size_t N>
bool trace(const char (&message)[N])
{
    constexpr DWORD bytes = N - 1;
    DWORD written = 0;
    return WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), message, bytes, &written, nullptr)
        && written == bytes;
}

bool check_memory()
{
    // Use a page-backed graphics allocation so release is observable through the query.
    constexpr ULONGLONG attributes = 0xa8004000;
    constexpr SIZE_T bytes = 4096;
    auto* address = static_cast<BYTE*>(XMemAlloc(bytes, attributes));
    if (!address) return false;
    const auto base = reinterpret_cast<std::uintptr_t>(address);
    volatile BYTE* block = address;

    // The desktop query writes a MEMORY_BASIC_INFORMATION prefix into a 64-byte caller buffer.
    struct Query {
        MEMORY_BASIC_INFORMATION info;
        BYTE tail[16];
    } query{};
    static_assert(offsetof(Query, tail) == 0x30 && sizeof(Query) == 64);

    block[0] = 0x5a;
    block[bytes - 1] = 0xa5;
    const HANDLE process = GetCurrentProcess();
    const bool ok = base % 256 == 0 && block[0] == 0x5a && block[bytes - 1] == 0xa5
        && XMemVirtualQuery(process, address, &query)
        && query.info.State == MEM_COMMIT && query.tail[0] == 1;
    XMemFree(address, attributes);
    return ok && XMemVirtualQuery(process, reinterpret_cast<LPCVOID>(base), &query)
        && query.info.State == MEM_FREE && query.tail[0] == 0;
}

bool check_working_set()
{
    constexpr std::uint64_t sentinel = 0x5a5a5a5a5a5a5a5a;
    std::array<std::uint64_t, 8> output;
    output.fill(sentinel);
    const auto expected = output;
    SetLastError(ERROR_SUCCESS);
    // 0x20000001, the selector the AMD callers pass, is supported; other selectors are not.
    const BOOL ok = XMemGetWorkingSetStatistics(0x20000002, output.data());
    const DWORD error = GetLastError();
    return !ok && error == ERROR_NOT_SUPPORTED && output == expected;
}

bool check_root_signature()
{
    const auto tagged = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT | 0x03000000u);
    constexpr D3D12_ROOT_CONSTANTS constants{5, 3, 4}; // Register, space and word count.
    D3D12_ROOT_PARAMETER parameter{};
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameter.Constants = constants;
    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC input{};
    input.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    input.Desc_1_0.NumParameters = 1;
    input.Desc_1_0.pParameters = &parameter;
    input.Desc_1_0.Flags = tagged;
    ComPtr<ID3DBlob> blob;
    if (FAILED(D3D12SerializeVersionedRootSignature(&input, &blob, nullptr)) || !blob) return false;

    ComPtr<ID3D12RootSignatureDeserializer> reader;
    if (FAILED(D3D12CreateRootSignatureDeserializer(blob->GetBufferPointer(), blob->GetBufferSize(),
                                                    IID_PPV_ARGS(&reader))) || !reader) return false;
    const auto* decoded = reader->GetRootSignatureDesc();
    if (input.Desc_1_0.Flags != tagged || !decoded || decoded->NumParameters != 1
        || !decoded->pParameters || decoded->NumStaticSamplers != 0
        || decoded->Flags != D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT) return false;
    const auto& restored = decoded->pParameters[0];
    return restored.ParameterType == D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS
        && restored.ShaderVisibility == D3D12_SHADER_VISIBILITY_PIXEL
        && restored.Constants.ShaderRegister == constants.ShaderRegister
        && restored.Constants.RegisterSpace == constants.RegisterSpace
        && restored.Constants.Num32BitValues == constants.Num32BitValues;
}
}

BOOL WINAPI DllMain(HINSTANCE, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH) ++process_attaches;
    return TRUE;
}

extern "C" __declspec(dllexport) DWORD run_startup_shims()
{
    if (process_attaches != 1 || !trace("companion attached\n")) return 1;
    if (!check_memory() || !trace("xmem allocation, query and free\n")) return 2;
    if (!check_working_set() || !trace("xmem working set unsupported\n")) return 3;
    if (!check_root_signature() || !trace("d3d12_xs root signature round trip\n")) return 4;
    return 42;
}
