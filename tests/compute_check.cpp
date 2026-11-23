#include "desktop_gpu.h"
#include "compute_root.h"
#include "compute_tables.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {
constexpr UINT count = 257;
constexpr UINT capacity = 320;
constexpr UINT bytes = capacity * sizeof(UINT);
constexpr UINT constants_offset = 2 * bytes;
constexpr UINT alignment = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
static_assert(constants_offset % alignment == 0);
constexpr UINT sentinel = 0xc5c5c5c5u;
constexpr std::array<UINT, 16> constants{count, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 23, 29};
}

static UINT input_word(UINT i) { return 0x8b02a7f1u + i * 0x413b1u; }

static void fill_upload(ID3D12Resource* upload)
{
    void* mapped = nullptr;
    const D3D12_RANGE no_reads{};
    check(upload->Map(0, &no_reads, &mapped), "Map compute upload");
    auto* words = static_cast<UINT*>(mapped);
    for (UINT i = 0; i < capacity; ++i) {
        words[i] = input_word(i);
        words[capacity + i] = sentinel;
    }
    auto* cb = static_cast<unsigned char*>(mapped) + constants_offset;
    std::memset(cb, 0, 4 * alignment);
    std::memcpy(cb, constants.data(), sizeof(constants));
    for (UINT i = 1; i < 4; ++i) std::memcpy(cb + i * alignment, &constants[12 + i], sizeof(UINT));
    upload->Unmap(0, nullptr);
}

static ComPtr<ID3D12DescriptorHeap> bind_tables(ID3D12Device* device, ID3D12GraphicsCommandList* list,
                                         ID3D12Resource* input, ID3D12Resource* output)
{
    constexpr UINT uavs = 13, srvs = 21;
    D3D12_DESCRIPTOR_HEAP_DESC desc{};
    desc.NumDescriptors = uavs + srvs;
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> heap;
    check(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap)), "Create compute descriptor heap");
    const UINT stride = device->GetDescriptorHandleIncrementSize(desc.Type);
    auto cpu = heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_R32_TYPELESS;
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav.Buffer.NumElements = capacity;
    uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    for (UINT i = 0; i < uavs; ++i, cpu.ptr += stride)
        device->CreateUnorderedAccessView(i == uavs - 1 ? output : nullptr, nullptr, &uav, cpu);
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R32_TYPELESS;
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Buffer.NumElements = capacity;
    srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    for (UINT i = 0; i < srvs; ++i, cpu.ptr += stride)
        device->CreateShaderResourceView(i == srvs - 1 ? input : nullptr, &srv, cpu);
    ID3D12DescriptorHeap* heaps[]{heap.Get()};
    list->SetDescriptorHeaps(1, heaps);
    auto gpu = heap->GetGPUDescriptorHandleForHeapStart();
    list->SetComputeRootDescriptorTable(0, gpu);
    gpu.ptr += UINT64{uavs} * stride;
    list->SetComputeRootDescriptorTable(1, gpu);
    for (UINT i = 0; i < 4; ++i)
        list->SetComputeRootConstantBufferView(2 + i, input->GetGPUVirtualAddress() + constants_offset + i * alignment);
    return heap;
}

struct StoreCase {
    // Group counts: X in bits 0-7, Y in bits 8-15 and Z in bits 16-23; zero Y or Z means one.
    UINT groups;
    std::array<UINT, capacity> expected;
    std::array<UINT, constants.size()> root_constants = constants;
    UINT axis(UINT index) const
    {
        const UINT n = (groups >> (8 * index)) & 0xff;
        return index && !n ? 1 : n;
    }
};

static void run_compute(ID3D12Device* device, bool tables, D3D12_SHADER_BYTECODE shader,
                        D3D12_SHADER_BYTECODE rootBytes, bool conversion_pair = false,
                        const StoreCase* store = nullptr)
{
    ComPtr<ID3D12RootSignature> signature;
    check(device->CreateRootSignature(0, rootBytes.pShaderBytecode, rootBytes.BytecodeLength, IID_PPV_ARGS(&signature)),
          "Create compute root signature");
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = signature.Get();
    pso.CS = shader;
    ComPtr<ID3D12PipelineState> pipeline;
    check(device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&pipeline)), "Create compute pipeline");
    auto input = make_buffer(device, constants_offset + 4 * alignment, D3D12_HEAP_TYPE_UPLOAD,
                             D3D12_RESOURCE_STATE_GENERIC_READ);
    auto output = make_buffer(device, bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto readback = make_buffer(device, bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    fill_upload(input.Get());
    ComputeCommands commands(device, pipeline.Get());
    auto& list = commands.list;
    transition(list.Get(), output.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(output.Get(), 0, input.Get(), bytes, bytes);
    transition(list.Get(), output.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list->SetComputeRootSignature(signature.Get());
    ComPtr<ID3D12DescriptorHeap> heap;
    if (tables) {
        heap = bind_tables(device, list.Get(), input.Get(), output.Get());
    } else {
        list->SetComputeRootShaderResourceView(0, input->GetGPUVirtualAddress());
        list->SetComputeRootUnorderedAccessView(1, output->GetGPUVirtualAddress());
        const auto& words = store ? store->root_constants : constants;
        list->SetComputeRoot32BitConstants(2, static_cast<UINT>(words.size()), words.data(), 0);
    }
    if (store) list->Dispatch(store->axis(0), store->axis(1), store->axis(2));
    else list->Dispatch(capacity / 64, 1, 1);
    transition(list.Get(), output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(readback.Get(), 0, output.Get(), 0, bytes);
    commands.submit_and_wait(device);
    void* mapped = nullptr;
    const D3D12_RANGE read{0, bytes}, no_writes{};
    check(readback->Map(0, &read, &mapped), "Map compute readback");
    const auto* words = static_cast<const UINT*>(mapped);
    UINT mismatches = 0;
    constexpr UINT key = constants[13] + 3 * constants[14] + 5 * constants[15];
    if (conversion_pair)
        std::printf("immediate_uint=0x%08x immediate_int=0x%08x\n", words[0], words[1]);
    const UINT active = conversion_pair ? 2 : count;
    for (UINT i = 0; i < capacity; ++i) {
        const UINT expected = store ? store->expected[i] : i >= active ? sentinel : conversion_pair ? (i == 0 ? 0xffffffffu : 0x7fffffffu)
            : (input_word(i) * 33u) ^ (key + i * constants[1]);
        if (words[i] != expected) {
            if (!mismatches)
                std::fprintf(stderr, "word=%u expected=%08x actual=%08x\n", i, expected, words[i]);
            ++mismatches;
        }
    }
    readback->Unmap(0, &no_writes);
    if (store && store->groups <= 0xff)
        std::printf("groups=%u compared_words=%u mismatches=%u\n", store->groups, capacity, mismatches);
    else if (store)
        std::printf("groups=%ux%ux%u compared_words=%u mismatches=%u\n", store->axis(0), store->axis(1),
                    store->axis(2), capacity, mismatches);
    else std::printf("layout=%s output_words=%u guard_words=%u mismatches=%u\n",
                tables ? "tables" : "root", active, capacity - active, mismatches);
    if (mismatches) throw std::runtime_error("Compute readback differs from CPU reference");
}

void check_compute(ID3D12Device* device, bool tables)
{
    const D3D12_SHADER_BYTECODE shader = tables ? D3D12_SHADER_BYTECODE{compute_tables, sizeof(compute_tables)} :
                                                  D3D12_SHADER_BYTECODE{compute_root, sizeof(compute_root)};
    run_compute(device, tables, shader, shader);
}

void check_compute_file(ID3D12Device* device, const char* path, bool conversion_pair, const char* reference)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error(std::string("Could not open compute shader: ") + path);
    const auto size = file.tellg();
    if (size <= 0) throw std::runtime_error(std::string("Empty compute shader: ") + path);
    std::vector<char> bytes(static_cast<std::size_t>(size));
    file.seekg(0);
    if (!file.read(bytes.data(), static_cast<std::streamsize>(bytes.size())))
        throw std::runtime_error(std::string("Could not read compute shader: ") + path);
    const D3D12_SHADER_BYTECODE shader{bytes.data(), bytes.size()};
    if (reference) {
        StoreCase store{};
        std::ifstream file(reference, std::ios::binary);
        file.read(reinterpret_cast<char*>(&store.groups), sizeof(store.groups));
        file.read(reinterpret_cast<char*>(store.expected.data()), sizeof(store.expected));
        if (!file || store.groups >> 24 || !store.axis(0) || store.axis(0) > 5 || store.axis(1) > 5 ||
            store.axis(2) > 5)
            throw std::runtime_error("Invalid compute store reference");
        if (file.peek() != std::char_traits<char>::eof()) {
            file.read(reinterpret_cast<char*>(store.root_constants.data()), sizeof(store.root_constants));
            if (!file || file.peek() != std::char_traits<char>::eof())
                throw std::runtime_error("Invalid compute store constants");
        }
        run_compute(device, false, shader, {compute_root, sizeof(compute_root)}, false, &store);
    } else {
        run_compute(device, false, shader, shader, conversion_pair);
    }
}

void check_bvh_root_roundtrip(ID3D12Device* device)
{
    const D3D12_SHADER_BYTECODE shader{compute_root, sizeof(compute_root)};
    ComPtr<ID3D12VersionedRootSignatureDeserializer> embedded;
    check(D3D12CreateVersionedRootSignatureDeserializer(shader.pShaderBytecode, shader.BytecodeLength,
          IID_PPV_ARGS(&embedded)), "Read embedded control signature");
    auto desc = *embedded->GetUnconvertedRootSignatureDesc();
    if (desc.Version != D3D_ROOT_SIGNATURE_VERSION_1_1) throw std::runtime_error("Expected a v1.1 control");
    auto& flags = desc.Desc_1_1.Flags;
    constexpr UINT tags = 0x03000000;
    if (flags & tags) throw std::runtime_error("Control already contains the BVH flags");
    ComPtr<ID3DBlob> control, rejected, normalized, roundtrip;
    check(D3D12SerializeVersionedRootSignature(&desc, &control, nullptr), "Serialize control signature");
    flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(flags | tags);
    const HRESULT tagged = D3D12SerializeVersionedRootSignature(&desc, &rejected, nullptr);
    std::printf("tagged_flags=0x%08x serialize_hr=0x%08lx\n", static_cast<UINT>(flags), tagged);
    if (tagged != E_INVALIDARG || rejected.Get()) throw std::runtime_error("Unexpected tagged signature result");

    flags = static_cast<D3D12_ROOT_SIGNATURE_FLAGS>(flags & ~tags);
    check(D3D12SerializeVersionedRootSignature(&desc, &normalized, nullptr), "Serialize normalized signature");
    ComPtr<ID3D12VersionedRootSignatureDeserializer> decoded;
    check(D3D12CreateVersionedRootSignatureDeserializer(normalized->GetBufferPointer(), normalized->GetBufferSize(),
          IID_PPV_ARGS(&decoded)), "Deserialize normalized signature");
    check(D3D12SerializeVersionedRootSignature(decoded->GetUnconvertedRootSignatureDesc(), &roundtrip, nullptr),
          "Reserialize normalized signature");
    if (roundtrip->GetBufferSize() != control->GetBufferSize() ||
        std::memcmp(roundtrip->GetBufferPointer(), control->GetBufferPointer(), control->GetBufferSize()) != 0)
        throw std::runtime_error("Normalized signature differs from independent control serialization");
    run_compute(device, false, shader, {roundtrip->GetBufferPointer(), roundtrip->GetBufferSize()});
    std::puts("normalized_signature=matched_control root_creation=passed pipeline_creation=passed dispatch=passed");
}
