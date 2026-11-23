#include "../desktop_gpu.h"
#include "../mesh_shadow_draw.h"
#include <cstdlib>
#include <filesystem>
// Run the translated compute -> mesh -> pixel chain for one EXE-requested frame.
//
// This repeats the sequence of `offscreen_check --compute-mesh-texture`
// (tests/mesh_entry_draw.cpp) inside the draw provider DLL, because that
// harness's helpers live in the offscreen_check executable. One direct queue
// runs one or two translated gfx1013 compute producers. Each writes two packed
// RGBA8 rows to a raw UAV. The rows are copied into a 2x2 texture, and the
// translated indexed mesh and image_sample_lz pixel stages draw a 32x32 frame.
// The shaders and case files are built by tests/cpu/shim_chain_check.py and
// read from X2W_SHIM_CHAIN_DIR. The frame and the 80-byte mesh export trace
// are returned for the provider to publish.
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdio>
#include <cstddef>
#include <cstring>
#include <cwchar>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace {
using Microsoft::WRL::ComPtr;

constexpr UINT side = 32;
constexpr UINT cb_bytes = 512;
constexpr UINT input_bytes = 3 * cb_bytes;  // two constant buffers, then the mesh vertex SRV
constexpr UINT trace_bytes = 80;
constexpr UINT image_bytes = side * side * 4;

struct Failure {};

struct ShadowDeviceArguments {
    IUnknown* adapter = nullptr;
    const IID* iid = nullptr;
    void* device = nullptr;
};
static_assert(offsetof(ShadowDeviceArguments, adapter) == 0);
static_assert(offsetof(ShadowDeviceArguments, iid) == 8);
static_assert(offsetof(ShadowDeviceArguments, device) == 16);

struct ShadowRequest : ShadowDeviceArguments {
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter1> selected;
    ComPtr<ID3D12Device> created_device;
    std::unique_ptr<MeshShadowDraw> scene;
    DWORD mode = 0;
};

ShadowRequest* shadow_request(void* args)
{
    return static_cast<ShadowRequest*>(static_cast<ShadowDeviceArguments*>(args));
}

void require(HRESULT hr, const char* operation)
{
    if (FAILED(hr)) {
        std::fprintf(stderr, "%s failed (HRESULT 0x%08lx)\n", operation, static_cast<unsigned long>(hr));
        throw Failure{};
    }
}

void fail(const char* message)
{
    std::fprintf(stderr, "%s\n", message);
    throw Failure{};
}

std::vector<char> read_input(const char* name)
{
    char dir[MAX_PATH]{};
    const DWORD length = GetEnvironmentVariableA("X2W_SHIM_CHAIN_DIR", dir, sizeof(dir));
    if (!length || length >= sizeof(dir)) fail("X2W_SHIM_CHAIN_DIR is not set");
    std::ifstream stream(std::string(dir) + "\\" + name, std::ios::binary);
    if (!stream) {
        std::fprintf(stderr, "cannot open chain input %s\n", name);
        throw Failure{};
    }
    return {std::istreambuf_iterator<char>(stream), {}};
}

// WARP by default. "hardware" takes the first physical D3D12 adapter in high-performance
// order or, with XBOX2WINDOWS_ADAPTER, the first whose description contains that value.
// No physical adapter is a capability skip (77); a filter that matches none is an error.
void select_adapter(ComPtr<IDXGIFactory6>& factory, ComPtr<IDXGIAdapter1>& adapter)
{
    require(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");
    char backend[16]{};
    const auto size = GetEnvironmentVariableA("X2W_SHIM_DRAW_ADAPTER", backend, sizeof(backend));
    if (!size || (size < sizeof(backend) && std::strcmp(backend, "warp") == 0)) {
        require(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "EnumWarpAdapter");
    } else if (size < sizeof(backend) && std::strcmp(backend, "hardware") == 0) {
        wchar_t wanted[128]{};
        const DWORD filter = GetEnvironmentVariableW(L"XBOX2WINDOWS_ADAPTER", wanted, 128);
        if (filter >= 128) fail("XBOX2WINDOWS_ADAPTER is longer than 127 characters");
        for (UINT i = 0;; ++i) {
            const HRESULT hr = factory->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                                   IID_PPV_ARGS(adapter.ReleaseAndGetAddressOf()));
            if (hr == DXGI_ERROR_NOT_FOUND) {
                if (filter) {
                    std::fprintf(stderr, "No D3D12 hardware adapter matches XBOX2WINDOWS_ADAPTER=%ls\n", wanted);
                    throw Failure{};
                }
                std::fputs("no D3D12 physical adapter\n", stderr);
                ExitProcess(77);
            }
            require(hr, "EnumAdapterByGpuPreference");
            DXGI_ADAPTER_DESC1 desc{};
            require(adapter->GetDesc1(&desc), "GetDesc1");
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            if (filter) {
                bool match = false;
                for (const wchar_t* text = desc.Description; *text && !match; ++text)
                    match = _wcsnicmp(text, wanted, filter) == 0;
                if (!match) continue;
            }
            if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr)))
                break;
        }
    } else {
        require(E_INVALIDARG, "Select draw adapter");
    }
}

ComPtr<ID3D12Device> create_device()
{
    ComPtr<IDXGIFactory6> factory;
    ComPtr<IDXGIAdapter1> adapter;
    select_adapter(factory, adapter);
    ComPtr<ID3D12Device> device;
    require(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)), "D3D12CreateDevice");
    return device;
}

ComPtr<ID3D12Resource> committed(ID3D12Device* device, const D3D12_RESOURCE_DESC& desc, D3D12_HEAP_TYPE heap,
                                 D3D12_RESOURCE_STATES state, const D3D12_CLEAR_VALUE* clear = nullptr)
{
    D3D12_HEAP_PROPERTIES props{};
    props.Type = heap;
    props.CreationNodeMask = props.VisibleNodeMask = 1;
    ComPtr<ID3D12Resource> result;
    require(device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, clear,
                                            IID_PPV_ARGS(&result)), "Create chain resource");
    return result;
}

ComPtr<ID3D12Resource> buffer(ID3D12Device* device, UINT64 bytes, D3D12_HEAP_TYPE heap,
                              D3D12_RESOURCE_STATES state, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE)
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return committed(device, desc, heap, state);
}

template<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Kind, typename T>
struct alignas(void*) Field {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE kind = Kind;
    T value{};
};

struct MeshPipeline {
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*> root;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, D3D12_SHADER_BYTECODE> ms;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, D3D12_SHADER_BYTECODE> ps;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, D3D12_RT_FORMAT_ARRAY> formats;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, D3D12_RASTERIZER_DESC> raster;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, D3D12_PRIMITIVE_TOPOLOGY_TYPE> topology;
};

ComPtr<ID3D12RootSignature> root_signature(ID3D12Device* device, const D3D12_ROOT_SIGNATURE_DESC& desc)
{
    ComPtr<ID3DBlob> blob;
    require(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr),
            "Serialize chain root signature");
    ComPtr<ID3D12RootSignature> root;
    require(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)),
            "Create chain root signature");
    return root;
}

} // namespace

// mode 3: dispatch palette 0, bind it. mode 4: dispatch palettes 0 and 1, bind 1.
// mode 5: dispatch both, bind the earlier texture (0) against palette 1's reference.
// On success `published` holds the 32x32 RGBA frame and the mesh export trace, and
// `mismatches` counts differing pixels plus 65536 if the trace differs.
bool shim_chain_gpu(unsigned mode, std::vector<unsigned char>& published, unsigned& mismatches)
{
    try {
        const unsigned count = mode == 3 ? 1 : 2;
        const unsigned bound = mode == 4 ? 1 : 0;
        const auto reference_case = read_input(mode == 3 ? "palette-0.case" : "palette-1.case");
        if (reference_case.size() != input_bytes + trace_bytes + image_bytes) fail("invalid chain case size");
        const auto mesh = read_input("mesh.dxil");
        const auto pixel = read_input("pixel.dxil");
        std::vector<char> producer_code[2];
        for (unsigned i = 0; i < count; ++i)
            producer_code[i] = read_input(i ? "palette-1.dxil" : "palette-0.dxil");

        auto device = create_device();
        D3D12_FEATURE_DATA_D3D12_OPTIONS7 options{};
        require(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &options, sizeof(options)),
                "Query mesh shader tier");
        if (options.MeshShaderTier < D3D12_MESH_SHADER_TIER_1) {
            std::fputs("mesh_shader_tier=unsupported\n", stderr);
            ExitProcess(77);
        }

        // The --compute-mesh-texture layout: b0, b1, the u0 export trace, the mesh's t0 vertex
        // buffer, the pixel stage's t0 texture table and a point-clamp s0.
        D3D12_ROOT_PARAMETER params[5]{};
        for (UINT i = 0; i < 2; ++i) {
            params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
            params[i].Descriptor.ShaderRegister = i;
        }
        params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        params[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_MESH;
        const D3D12_DESCRIPTOR_RANGE texture_range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
        params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[4].DescriptorTable = {1, &texture_range};
        params[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        D3D12_STATIC_SAMPLER_DESC sampler{};
        sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.MaxAnisotropy = 1;
        sampler.ComparisonFunc = D3D12_COMPARISON_FUNC_ALWAYS;
        sampler.MaxLOD = D3D12_FLOAT32_MAX;
        sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        D3D12_ROOT_SIGNATURE_DESC layout{5, params, 1, &sampler};
        auto root = root_signature(device.Get(), layout);

        MeshPipeline stream;
        stream.root.value = root.Get();
        stream.ms.value = {mesh.data(), mesh.size()};
        stream.ps.value = {pixel.data(), pixel.size()};
        stream.formats.value.NumRenderTargets = 1;
        stream.formats.value.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        stream.raster.value.FillMode = D3D12_FILL_MODE_SOLID;
        stream.raster.value.CullMode = D3D12_CULL_MODE_NONE;
        stream.raster.value.DepthClipEnable = TRUE;
        stream.topology.value = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        const D3D12_PIPELINE_STATE_STREAM_DESC stream_desc{sizeof(stream), &stream};
        ComPtr<ID3D12Device2> device2;
        require(device.As(&device2), "Query mesh pipeline device");
        ComPtr<ID3D12PipelineState> pipeline;
        require(device2->CreatePipelineState(&stream_desc, IID_PPV_ARGS(&pipeline)), "Create mesh pipeline");

        D3D12_ROOT_PARAMETER output{};
        output.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        auto compute_root = root_signature(device.Get(), {1, &output, 0, nullptr});

        D3D12_RESOURCE_DESC target_desc{};
        target_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        target_desc.Width = target_desc.Height = side;
        target_desc.DepthOrArraySize = target_desc.MipLevels = target_desc.SampleDesc.Count = 1;
        target_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        target_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        D3D12_CLEAR_VALUE clear{};
        clear.Format = target_desc.Format;
        clear.Color[3] = 1;
        auto target = committed(device.Get(), target_desc, D3D12_HEAP_TYPE_DEFAULT,
                                D3D12_RESOURCE_STATE_RENDER_TARGET, &clear);
        D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
        ComPtr<ID3D12DescriptorHeap> rtv_heap;
        require(device->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtv_heap)), "Create chain RTV heap");
        const auto rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
        device->CreateRenderTargetView(target.Get(), nullptr, rtv);
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        UINT64 frame_bytes = 0;
        device->GetCopyableFootprints(&target_desc, 0, 1, 0, &footprint, nullptr, nullptr, &frame_bytes);
        auto readback = buffer(device.Get(), frame_bytes + trace_bytes, D3D12_HEAP_TYPE_READBACK,
                               D3D12_RESOURCE_STATE_COPY_DEST);

        auto upload = buffer(device.Get(), input_bytes + trace_bytes, D3D12_HEAP_TYPE_UPLOAD,
                             D3D12_RESOURCE_STATE_GENERIC_READ);
        void* mapped = nullptr;
        const D3D12_RANGE no_reads{0, 0};
        require(upload->Map(0, &no_reads, &mapped), "Map chain input");
        std::memcpy(mapped, reference_case.data(), input_bytes);
        std::memset(static_cast<char*>(mapped) + input_bytes, 0xcd, trace_bytes);
        upload->Unmap(0, nullptr);

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        ComPtr<ID3D12CommandQueue> queue;
        require(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)), "Create chain queue");
        ComPtr<ID3D12CommandAllocator> allocator;
        require(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)),
                "Create chain allocator");
        ComPtr<ID3D12GraphicsCommandList> list;
        require(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), pipeline.Get(),
                                          IID_PPV_ARGS(&list)), "Create chain command list");
        list->SetGraphicsRootSignature(root.Get());
        const auto base = upload->GetGPUVirtualAddress();
        list->SetGraphicsRootConstantBufferView(0, base);
        list->SetGraphicsRootConstantBufferView(1, base + cb_bytes);
        list->SetGraphicsRootShaderResourceView(3, base + 2 * cb_bytes);

        // Producers: raw UAV rows at byte offsets 0 and 256, copied into a 2x2 texture.
        auto texel_desc = target_desc;
        texel_desc.Width = texel_desc.Height = 2;
        texel_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT texel_layout{};
        UINT64 texel_bytes = 0;
        device->GetCopyableFootprints(&texel_desc, 0, 1, 0, &texel_layout, nullptr, nullptr, &texel_bytes);
        if (texel_layout.Offset != 0 || texel_layout.Footprint.RowPitch != D3D12_TEXTURE_DATA_PITCH_ALIGNMENT)
            fail("unexpected footprint for the compute texture");
        list->SetComputeRootSignature(compute_root.Get());
        ComPtr<ID3D12PipelineState> producers[2];
        ComPtr<ID3D12Resource> rows[2], textures[2];
        for (unsigned i = 0; i < count; ++i) {
            D3D12_COMPUTE_PIPELINE_STATE_DESC compute{};
            compute.pRootSignature = compute_root.Get();
            compute.CS = {producer_code[i].data(), producer_code[i].size()};
            require(device->CreateComputePipelineState(&compute, IID_PPV_ARGS(&producers[i])),
                    "Create texture producer pipeline");
            rows[i] = buffer(device.Get(), texel_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            textures[i] = committed(device.Get(), texel_desc, D3D12_HEAP_TYPE_DEFAULT,
                                    D3D12_RESOURCE_STATE_COPY_DEST);
            transition(list.Get(), rows[i].Get(), D3D12_RESOURCE_STATE_COMMON,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            list->SetPipelineState(producers[i].Get());
            list->SetComputeRootUnorderedAccessView(0, rows[i]->GetGPUVirtualAddress());
            list->Dispatch(1, 1, 1);
            transition(list.Get(), rows[i].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION from{}, to{};
            from.pResource = rows[i].Get();
            from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            from.PlacedFootprint = texel_layout;
            to.pResource = textures[i].Get();
            list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
            transition(list.Get(), textures[i].Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                       D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        }
        list->SetPipelineState(pipeline.Get());
        D3D12_DESCRIPTOR_HEAP_DESC srv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1,
                                            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        ComPtr<ID3D12DescriptorHeap> srv_heap;
        require(device->CreateDescriptorHeap(&srv_desc, IID_PPV_ARGS(&srv_heap)), "Create chain SRV heap");
        device->CreateShaderResourceView(textures[bound].Get(), nullptr, srv_heap->GetCPUDescriptorHandleForHeapStart());
        ID3D12DescriptorHeap* heaps[]{srv_heap.Get()};
        list->SetDescriptorHeaps(1, heaps);
        list->SetGraphicsRootDescriptorTable(4, srv_heap->GetGPUDescriptorHandleForHeapStart());

        auto trace = buffer(device.Get(), trace_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        list->CopyBufferRegion(trace.Get(), 0, upload.Get(), input_bytes, trace_bytes);
        transition(list.Get(), trace.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->SetGraphicsRootUnorderedAccessView(2, trace->GetGPUVirtualAddress());
        const D3D12_VIEWPORT viewport{0, 0, side, side, 0, 1};
        const D3D12_RECT scissor{0, 0, side, side};
        list->RSSetViewports(1, &viewport);
        list->RSSetScissorRects(1, &scissor);
        list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        list->ClearRenderTargetView(rtv, clear.Color, 0, nullptr);
        ComPtr<ID3D12GraphicsCommandList6> mesh_list;
        require(list.As(&mesh_list), "Query mesh dispatch list");
        mesh_list->DispatchMesh(1, 1, 1);

        transition(list.Get(), trace.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(readback.Get(), frame_bytes, trace.Get(), 0, trace_bytes);
        transition(list.Get(), target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
        src.pResource = target.Get();
        dst.pResource = readback.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = footprint;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        require(list->Close(), "Close chain command list");
        ComPtr<ID3D12Fence> fence;
        require(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "Create chain fence");
        ID3D12CommandList* submitted[]{list.Get()};
        queue->ExecuteCommandLists(1, submitted);
        require(queue->Signal(fence.Get(), 1), "Signal chain fence");
        require(fence->SetEventOnCompletion(1, nullptr), "Wait for chain fence");
        require(device->GetDeviceRemovedReason(), "Chain completion");

        const D3D12_RANGE range{0, static_cast<SIZE_T>(frame_bytes + trace_bytes)};
        require(readback->Map(0, &range, &mapped), "Map chain readback");
        const auto* bytes = static_cast<const unsigned char*>(mapped);
        published.resize(image_bytes + trace_bytes);
        for (UINT y = 0; y < side; ++y)
            std::memcpy(published.data() + y * side * 4, bytes + footprint.Offset + y * footprint.Footprint.RowPitch,
                        side * 4);
        std::memcpy(published.data() + image_bytes, bytes + frame_bytes, trace_bytes);
        readback->Unmap(0, &no_reads);

        const auto* expected_trace = reference_case.data() + input_bytes;
        const auto* expected_frame = expected_trace + trace_bytes;
        mismatches = std::memcmp(published.data() + image_bytes, expected_trace, trace_bytes) ? 65536 : 0;
        for (UINT i = 0; i < side * side; ++i)
            mismatches += std::memcmp(published.data() + 4 * i, expected_frame + 4 * i, 4) != 0;
        return true;
    } catch (Failure) {
        return false;
    }
}

extern "C" __declspec(dllexport) void* shadow_device_request(DWORD mode)
{
    try {
        auto request = std::make_unique<ShadowRequest>();
        select_adapter(request->factory, request->selected);
        request->mode = mode;
        request->adapter = request->selected.Get();
        request->iid = mode == 16 ? &__uuidof(IDXGIFactory6) : &__uuidof(ID3D12Device);
        if (mode == 16) request->device = request->adapter;
        return static_cast<ShadowDeviceArguments*>(request.release());
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return nullptr;
    } catch (const Failure&) {
        return nullptr;
    }
}

extern "C" __declspec(dllexport) HRESULT shadow_open(void* args, HRESULT result)
{
    try {
        auto* request = shadow_request(args);
        if (request->mode == 16) {
            if (result != E_NOINTERFACE || request->device || !request->selected || !request->factory)
                fail("Unexpected rejected shim device or missing adapter ownership");
            request->selected.Reset();
            request->factory.Reset();
            std::fputs("Xbox device rejected: no device, released adapter and factory\n", stderr);
            return E_NOINTERFACE;
        }
        require(result, "D3D12XboxCreateDevice");
        if (!request->device) fail("D3D12XboxCreateDevice returned no device");
        request->created_device.Attach(static_cast<ID3D12Device*>(request->device));
        request->selected.Reset();
        request->factory.Reset();
        std::fputs("Xbox device created: EXE import returned a desktop device\n", stderr);
        const auto* path = std::getenv("X2W_SHIM_CHAIN_DIR");
        if (!path || !*path) fail("X2W_SHIM_CHAIN_DIR is not set");
        const auto directory = std::filesystem::path(path);
        request->scene = std::make_unique<MeshShadowDraw>(
            request->created_device.Get(), (directory / "mesh.dxil").string().c_str(),
            (directory / "receiver.dxil").string().c_str(), (directory / "caster.dxil").string().c_str(),
            (directory / "frames.case").string().c_str(), true);
        if (!request->scene->supported()) return DXGI_ERROR_UNSUPPORTED;
        return S_OK;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return E_FAIL;
    } catch (const Failure&) {
        return E_FAIL;
    }
}

extern "C" __declspec(dllexport) HRESULT shadow_record(void* args, UINT frame, UINT map)
{
    try {
        shadow_request(args)->scene->record(frame, map);
        std::fprintf(stderr, "shadow record frame=%u map=%u\n", frame, map);
        return S_OK;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return E_FAIL;
    }
}

extern "C" __declspec(dllexport) HRESULT shadow_submit(void* args, UINT frame)
{
    try {
        const auto fence = shadow_request(args)->scene->submit(frame);
        if (!fence) {
            std::fprintf(stderr, "shadow submit frame=%u rejected\n", frame);
            return E_INVALIDARG;
        }
        std::fprintf(stderr, "shadow submit frame=%u fence=%llu\n", frame, static_cast<unsigned long long>(fence));
        return S_OK;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return E_FAIL;
    }
}

bool shim_shadow_wait(void* args, std::vector<unsigned char>& output)
{
    try {
        const auto fence = shadow_request(args)->scene->wait(output);
        std::fprintf(stderr, "shadow wait fence=%llu\n", static_cast<unsigned long long>(fence));
        return true;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return false;
    }
}

extern "C" __declspec(dllexport) HRESULT shadow_close(void* args)
{
    try {
        std::unique_ptr<ShadowRequest> request(shadow_request(args));
        // An unsupported mesh tier leaves an empty scene with a live device.
        const UINT surfaces = request->scene && request->scene->supported()
            ? request->scene->release_surfaces() : 0;
        request->scene.reset();
        const bool device = static_cast<bool>(request->created_device);
        request->created_device.Reset();
        std::fprintf(stderr, "shadow close surfaces=%u device=%u\n", surfaces, unsigned(device));
        return S_OK;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return E_FAIL;
    }
}
