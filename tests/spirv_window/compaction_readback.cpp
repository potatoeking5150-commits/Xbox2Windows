// Read back the source-built subgroup compaction fixture and print every word.
//
// The fixture contract (tests/spirv_window/compaction.spvasm and compaction.py) is one 64-thread
// group. Words are 0xc5c5c5c5 unless the shader writes them:
//   [0, 64)    lane index observed by each group thread
//   [64]       subgroup width
//   [65, 129)  four ballot words per wave (16 waves at width 4)
//   [129, 145) first active lane per wave
//   [145, 161) active lane count per wave
//   [161, 177) last active lane per wave
//   [177, 193) compaction rank of the selected (last active) lane per wave
//   [193, 257) per-lane compaction rank
//   [257, 321) compacted payloads in wave-disjoint ranges keyed by rank
//   [321, 328) untouched guards
#include "../desktop_gpu.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {
constexpr UINT words = 328;
constexpr UINT sentinel = 0xc5c5c5c5u;

ComPtr<ID3D12RootSignature> create_compaction_root(ID3D12Device* device)
{
    D3D12_ROOT_PARAMETER parameter{};
    parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    parameter.Descriptor = {0, 0};  // u0, space0
    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC description{1, &parameter, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> bytes, error;
    check(D3D12SerializeRootSignature(&description, D3D_ROOT_SIGNATURE_VERSION_1, &bytes, &error),
          "Serialize compaction root signature");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, bytes->GetBufferPointer(), bytes->GetBufferSize(),
                                      IID_PPV_ARGS(&root)), "Create compaction root signature");
    return root;
}
}

void check_spirv_compaction(ID3D12Device* device, const char* shader_path)
{
    std::ifstream shader(shader_path, std::ios::binary);
    if (!shader) throw std::runtime_error("Cannot open subgroup compaction shader");
    const std::vector<char> bytes(std::istreambuf_iterator<char>(shader), {});
    if (bytes.empty()) throw std::runtime_error("Empty subgroup compaction shader");

    auto root = create_compaction_root(device);
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = root.Get();
    pso.CS = {bytes.data(), bytes.size()};
    ComPtr<ID3D12PipelineState> pipeline;
    check(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&pipeline)),
          "Create subgroup compaction pipeline");

    constexpr UINT output_bytes = words * sizeof(UINT);
    auto upload = make_buffer(device, output_bytes, D3D12_HEAP_TYPE_UPLOAD,
                              D3D12_RESOURCE_STATE_GENERIC_READ);
    auto arena = make_buffer(device, output_bytes, D3D12_HEAP_TYPE_DEFAULT,
                             D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto readback = make_buffer(device, output_bytes, D3D12_HEAP_TYPE_READBACK,
                                D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_RANGE empty{};
    void* mapped = nullptr;
    check(upload->Map(0, &empty, &mapped), "Map compaction upload");
    std::fill_n(static_cast<UINT*>(mapped), words, sentinel);
    upload->Unmap(0, nullptr);

    ComputeCommands commands(device, pipeline.Get());
    auto& list = commands.list;
    transition(list.Get(), arena.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(arena.Get(), 0, upload.Get(), 0, output_bytes);
    transition(list.Get(), arena.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list->SetComputeRootSignature(root.Get());
    list->SetComputeRootUnorderedAccessView(0, arena->GetGPUVirtualAddress());
    list->Dispatch(1, 1, 1);
    transition(list.Get(), arena.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(readback.Get(), 0, arena.Get(), 0, output_bytes);
    commands.submit_and_wait(device);

    const D3D12_RANGE read{0, output_bytes};
    check(readback->Map(0, &read, &mapped), "Map compaction readback");
    const auto* observed = static_cast<const UINT*>(mapped);
    std::printf("wave_compaction words=%u dispatched_groups=1 threads=64\n", words);
    for (UINT i = 0; i < words; ++i) std::printf("observed[%u]=%08x\n", i, observed[i]);
    readback->Unmap(0, &empty);
}
