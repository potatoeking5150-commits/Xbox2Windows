// Two-pass shadow draw with translated mesh and pixel programs.
// Pass 1 writes light-space depth into a depth texture; its color writes are masked off. Pass 2 draws a receiver
// whose pixel program samples that texture with a comparison sampler, then a lit caster, both depth tested.
// The case file holds every draw's inputs, the expected mesh traces, the expected color image and the
// expected light depth, all from an independent CPU reference.
#include "desktop_gpu.h"
#include "mesh_shadow_draw.h"
#include <algorithm>
#include <optional>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>
using Microsoft::WRL::ComPtr;

namespace {
std::vector<char> read_file(const char* path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open shadow draw input");
    return {std::istreambuf_iterator<char>(stream), {}};
}

template<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Kind, typename T>
struct alignas(void*) Field {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE kind = Kind;
    T value{};
};

struct ShadowPipeline {
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*> root;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, D3D12_SHADER_BYTECODE> ms;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, D3D12_SHADER_BYTECODE> ps;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, D3D12_RT_FORMAT_ARRAY> formats;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, D3D12_BLEND_DESC> blend;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, D3D12_RASTERIZER_DESC> raster;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, D3D12_DEPTH_STENCIL_DESC> depth;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, DXGI_FORMAT> depth_format;
    Field<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, D3D12_PRIMITIVE_TOPOLOGY_TYPE> topology;
};

ComPtr<ID3D12PipelineState> shadow_pipeline(ID3D12Device2* device, ID3D12RootSignature* root,
                                            const std::vector<char>& mesh, const std::vector<char>& pixel,
                                            bool write_color)
{
    ShadowPipeline stream;
    stream.root.value = root;
    stream.ms.value = {mesh.data(), mesh.size()};
    stream.ps.value = {pixel.data(), pixel.size()};
    stream.formats.value.NumRenderTargets = 1;
    stream.formats.value.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    // The light pass keeps the caster's pixel program and target but writes depth only. On WARP
    // (10.0.26100.9818) a mesh pipeline without a pixel shader removed the device in this fixture, with or
    // without a bound render target and with no debug-layer message; the RTX 3070 drew it correctly.
    auto& target = stream.blend.value.RenderTarget[0];
    target.SrcBlend = target.SrcBlendAlpha = D3D12_BLEND_ONE;
    target.DestBlend = target.DestBlendAlpha = D3D12_BLEND_ZERO;
    target.BlendOp = target.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    target.LogicOp = D3D12_LOGIC_OP_NOOP;
    target.RenderTargetWriteMask = write_color ? D3D12_COLOR_WRITE_ENABLE_ALL : 0;
    stream.raster.value.FillMode = D3D12_FILL_MODE_SOLID;
    stream.raster.value.CullMode = D3D12_CULL_MODE_NONE;
    stream.raster.value.DepthClipEnable = TRUE;
    stream.depth.value.DepthEnable = TRUE;
    stream.depth.value.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    stream.depth.value.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    stream.depth_format.value = DXGI_FORMAT_D32_FLOAT;
    stream.topology.value = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    D3D12_PIPELINE_STATE_STREAM_DESC desc{sizeof(stream), &stream};
    ComPtr<ID3D12PipelineState> pipeline;
    check(device->CreatePipelineState(&desc, IID_PPV_ARGS(&pipeline)), "Create shadow pipeline");
    return pipeline;
}

void copy_to_readback(ID3D12GraphicsCommandList* list, ID3D12Resource* texture, ID3D12Resource* readback,
                      const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& footprint)
{
    D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
    src.pResource = texture;
    dst.pResource = readback;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = footprint;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
}

unsigned count_mismatches(const unsigned char* actual, const char* expected, size_t bytes)
{
    unsigned mismatches = 0;
    for (size_t offset = 0; offset < bytes; offset += 4)
        mismatches += std::memcmp(actual + offset, expected + offset, 4) != 0;
    return mismatches;
}

void read_rows(ID3D12Resource* readback, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& footprint,
               UINT side, unsigned char* output)
{
    void* mapped = nullptr;
    check(readback->Map(0, nullptr, &mapped), "Map shadow frame");
    const auto* rows = static_cast<const unsigned char*>(mapped) + footprint.Offset;
    for (UINT y = 0; y < side; ++y)
        std::memcpy(output + y * side * 4, rows + y * footprint.Footprint.RowPitch, side * 4);
    const D3D12_RANGE empty{};
    readback->Unmap(0, &empty);
}
}

struct MeshShadowDraw::Scene {
    Scene(ID3D12Device* device, const char* mesh, const char* receiver, const char* caster,
          const char* fixture, bool two_frames);
    void wait_for_gpu();

    static constexpr UINT cb_bytes = 512;
    static constexpr UINT draw_bytes = 3 * cb_bytes;  // b0, b1 and vertex SRV t0
    static constexpr UINT trace_stride = 80;
    ID3D12Device* const device;
    const bool two_frames;
    UINT next_frame;
    UINT recorded_frame = 0;
    bool ready = false;
    UINT64 submitted = 0;
    UINT64 completed = 0;
    std::vector<char> input;
    UINT side = 0, previous = 0, current = 0, draws = 0;
    size_t traces_at = 0, image_at = 0;
    UINT srv_step = 0;
    D3D12_CPU_DESCRIPTOR_HANDLE rtv{};
    std::array<D3D12_CPU_DESCRIPTOR_HANDLE, 3> dsv{};
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT color_footprint{}, depth_footprint{};
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> light_pass, receiver_pass, caster_pass;
    ComPtr<ID3D12Resource> target, upload, trace, color_readback, depth_readback, trace_readback;
    std::array<ComPtr<ID3D12Resource>, 3> depths;
    ComPtr<ID3D12DescriptorHeap> rtv_heap, dsv_heap, srv_heap;
    std::optional<ComputeCommands> commands;
    ComPtr<ID3D12GraphicsCommandList6> mesh_list;
    ComPtr<ID3D12Fence> fence;
};

MeshShadowDraw::Scene::Scene(ID3D12Device* device, const char* mesh_path, const char* receiver_path,
                             const char* caster_path, const char* case_path, bool two_frames)
    : device(device), two_frames(two_frames), next_frame(two_frames ? 0u : 1u)
{
    input = read_file(case_path);
    std::array<uint32_t, 8> header{};
    if (input.size() < sizeof(header)) throw std::runtime_error("Invalid shadow case header");
    std::memcpy(header.data(), input.data(), sizeof(header));
    side = header[0];
    previous = header[1];
    current = header[2];
    const auto filter = D3D12_FILTER(header[3]);
    const auto func = D3D12_COMPARISON_FUNC(header[4]);
    // Light draws for the previous and the current shadow map, then the receiver and the lit caster.
    draws = previous + current + (two_frames ? 4 : 2);
    if (side == 0 || side > 256 || previous == 0 || current == 0 || previous > 16 || current > 16 || draws > 16)
        throw std::runtime_error("Invalid shadow case header");
    const UINT image_bytes = side * side * 4;
    const size_t inputs_at = sizeof(header);
    traces_at = inputs_at + size_t(draws) * draw_bytes;
    image_at = traces_at + size_t(draws) * trace_stride;
    const size_t depth_at = image_at + image_bytes;
    if (input.size() != (two_frames ? image_at : depth_at + image_bytes)) throw std::runtime_error("Shadow case size does not match its header");
    const auto mesh = read_file(mesh_path);
    const auto receiver = read_file(receiver_path);
    const auto caster = read_file(caster_path);

    // b0 and b1 as root CBVs for both stages, the trace UAV and the vertex SRV for the mesh stage, and the
    // shadow map at t0 with a comparison sampler at s0 for the pixel stage.
    D3D12_ROOT_PARAMETER params[5]{};
    for (UINT i = 0; i < 2; ++i) {
        params[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        params[i].Descriptor.ShaderRegister = i;
    }
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_MESH;
    params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_MESH;
    D3D12_DESCRIPTOR_RANGE map_range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
    params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[4].DescriptorTable = {1, &map_range};
    params[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_STATIC_SAMPLER_DESC sampler{};
    sampler.Filter = filter;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.MaxAnisotropy = 1;
    sampler.ComparisonFunc = func;
    sampler.MaxLOD = D3D12_FLOAT32_MAX;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_ROOT_SIGNATURE_DESC layout{5, params, 1, &sampler, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> blob;
    check(D3D12SerializeRootSignature(&layout, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr), "Serialize shadow root");
    check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&root)),
          "Shadow root");
    ComPtr<ID3D12Device2> device2;
    check(device->QueryInterface(IID_PPV_ARGS(&device2)), "Query mesh pipeline device");
    light_pass = shadow_pipeline(device2.Get(), root.Get(), mesh, caster, false);
    receiver_pass = shadow_pipeline(device2.Get(), root.Get(), mesh, receiver, true);
    caster_pass = shadow_pipeline(device2.Get(), root.Get(), mesh, caster, true);

    D3D12_RESOURCE_DESC color_desc{};
    color_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    color_desc.Width = color_desc.Height = side;
    color_desc.DepthOrArraySize = color_desc.MipLevels = color_desc.SampleDesc.Count = 1;
    color_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    color_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_CLEAR_VALUE clear_color{color_desc.Format, {0, 0, 0, 1}};
    target = resource(device, color_desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RENDER_TARGET,
                           &clear_color);
    // Two shadow maps (previous and current light) and the camera depth. The maps are typeless so the
    // camera pass can read them through an R32_FLOAT view.
    auto depth_desc = color_desc;
    depth_desc.Format = DXGI_FORMAT_R32_TYPELESS;
    depth_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_CLEAR_VALUE clear_depth{};
    clear_depth.Format = DXGI_FORMAT_D32_FLOAT;
    clear_depth.DepthStencil.Depth = 1.0f;
    for (auto& depth : depths)
        depth = resource(device, depth_desc, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_DEPTH_WRITE, &clear_depth);
    D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    check(device->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtv_heap)), "Create shadow target heap");
    rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.Get(), nullptr, rtv);
    D3D12_DESCRIPTOR_HEAP_DESC dsv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, UINT(depths.size())};
    check(device->CreateDescriptorHeap(&dsv_desc, IID_PPV_ARGS(&dsv_heap)), "Create shadow depth heap");
    const UINT dsv_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    D3D12_DEPTH_STENCIL_VIEW_DESC dsv_view{};
    dsv_view.Format = DXGI_FORMAT_D32_FLOAT;
    dsv_view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    for (UINT i = 0; i < depths.size(); ++i) {
        dsv[i] = {dsv_heap->GetCPUDescriptorHandleForHeapStart().ptr + i * dsv_step};
        device->CreateDepthStencilView(depths[i].Get(), &dsv_view, dsv[i]);
    }
    D3D12_DESCRIPTOR_HEAP_DESC srv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2,
                                        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    check(device->CreateDescriptorHeap(&srv_desc, IID_PPV_ARGS(&srv_heap)), "Create shadow map heap");
    D3D12_SHADER_RESOURCE_VIEW_DESC map_view{};
    map_view.Format = DXGI_FORMAT_R32_FLOAT;
    map_view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    map_view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    map_view.Texture2D.MipLevels = 1;
    srv_step = device->GetDescriptorHandleIncrementSize(srv_desc.Type);
    for (UINT i = 0; i < 2; ++i) {
        auto handle = srv_heap->GetCPUDescriptorHandleForHeapStart();
        handle.ptr += i * srv_step;
        device->CreateShaderResourceView(depths[i].Get(), &map_view, handle);
    }

    UINT64 color_size = 0, depth_size = 0;
    device->GetCopyableFootprints(&color_desc, 0, 1, 0, &color_footprint, nullptr, nullptr, &color_size);
    device->GetCopyableFootprints(&depth_desc, 0, 1, 0, &depth_footprint, nullptr, nullptr, &depth_size);
    const UINT trace_bytes = draws * trace_stride;
    color_readback = make_buffer(device, color_size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    depth_readback = make_buffer(device, depth_size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    trace_readback = make_buffer(device, trace_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    const UINT upload_bytes = draws * draw_bytes;
    upload = make_buffer(device, upload_bytes + trace_bytes, D3D12_HEAP_TYPE_UPLOAD,
                              D3D12_RESOURCE_STATE_GENERIC_READ);
    void* mapped = nullptr;
    D3D12_RANGE empty{};
    check(upload->Map(0, &empty, &mapped), "Map shadow input");
    std::memcpy(mapped, input.data() + inputs_at, upload_bytes);
    std::memset(static_cast<char*>(mapped) + upload_bytes, 0xcd, trace_bytes);
    upload->Unmap(0, nullptr);
    trace = make_buffer(device, trace_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    commands.emplace(device, light_pass.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
    check(commands->list.As(&mesh_list), "Query mesh dispatch list");
    check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "Create shadow fence");
}

void MeshShadowDraw::Scene::wait_for_gpu()
{
    check(fence->SetEventOnCompletion(submitted, nullptr), "Wait for shadow frame");
    completed = submitted;
    check(device->GetDeviceRemovedReason(), "Shadow completion");
}

MeshShadowDraw::MeshShadowDraw(ID3D12Device* device, const char* mesh, const char* receiver,
                               const char* caster, const char* fixture, bool two_frames)
{
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 options{};
    check(check_feature(device, D3D12_FEATURE_D3D12_OPTIONS7, &options, sizeof(options)), "Query mesh tier");
    if (options.MeshShaderTier < D3D12_MESH_SHADER_TIER_1) {
        std::puts("mesh_shader_tier=unsupported");
        return;
    }
    scene_ = std::make_unique<Scene>(device, mesh, receiver, caster, fixture, two_frames);
}

MeshShadowDraw::~MeshShadowDraw() = default;

bool MeshShadowDraw::supported() const
{
    return static_cast<bool>(scene_);
}

void MeshShadowDraw::record(UINT frame, UINT shadow_map)
{
    auto& scene = *scene_;
    if (frame != scene.next_frame || frame >= 2 || shadow_map > frame || scene.ready ||
        scene.submitted != scene.completed)
        throw std::runtime_error("Unexpected shadow frame or binding");
    auto* list = scene.commands->list.Get();
    if (scene.completed) {
        check(scene.commands->allocator->Reset(), "Reset shadow allocator");
        check(list->Reset(scene.commands->allocator.Get(), scene.light_pass.Get()), "Reset shadow list");
        transition(list, scene.trace.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    } else {
        list->CopyBufferRegion(scene.trace.Get(), 0, scene.upload.Get(),
                               scene.draws * Scene::draw_bytes, scene.draws * Scene::trace_stride);
        transition(list, scene.trace.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }

    list->SetGraphicsRootSignature(scene.root.Get());
    const D3D12_VIEWPORT viewport{0, 0, float(scene.side), float(scene.side), 0, 1};
    const D3D12_RECT scissor{0, 0, LONG(scene.side), LONG(scene.side)};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    const auto base = scene.upload->GetGPUVirtualAddress();
    UINT draw = 0;
    auto dispatch = [&] {
        const auto at = base + UINT64(draw) * Scene::draw_bytes;
        list->SetGraphicsRootConstantBufferView(0, at);
        list->SetGraphicsRootConstantBufferView(1, at + Scene::cb_bytes);
        list->SetGraphicsRootUnorderedAccessView(2, scene.trace->GetGPUVirtualAddress() + UINT64(draw) * Scene::trace_stride);
        list->SetGraphicsRootShaderResourceView(3, at + 2 * Scene::cb_bytes);
        scene.mesh_list->DispatchMesh(1, 1, 1);
        ++draw;
    };
    // The standalone draw builds both light maps before its single camera pass.
    for (UINT map = scene.two_frames ? frame : 0u; map <= frame; ++map) {
        draw = map ? scene.previous : 0;
        list->SetPipelineState(scene.light_pass.Get());
        list->OMSetRenderTargets(1, &scene.rtv, FALSE, &scene.dsv[map]);
        list->ClearDepthStencilView(scene.dsv[map], D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
        for (UINT i = 0; i < (map ? scene.current : scene.previous); ++i) dispatch();
        transition(list, scene.depths[map].Get(), D3D12_RESOURCE_STATE_DEPTH_WRITE,
                   D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    }
    ID3D12DescriptorHeap* heaps[]{scene.srv_heap.Get()};
    list->SetDescriptorHeaps(1, heaps);
    auto bound = scene.srv_heap->GetGPUDescriptorHandleForHeapStart();
    bound.ptr += UINT64(shadow_map) * scene.srv_step;
    list->SetGraphicsRootDescriptorTable(4, bound);
    list->OMSetRenderTargets(1, &scene.rtv, FALSE, &scene.dsv[2]);
    const float clear[4]{0, 0, 0, 1};
    list->ClearRenderTargetView(scene.rtv, clear, 0, nullptr);
    list->ClearDepthStencilView(scene.dsv[2], D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
    draw = scene.previous + scene.current + (scene.two_frames ? 2 * frame : 0);
    list->SetPipelineState(scene.receiver_pass.Get());
    dispatch();
    list->SetPipelineState(scene.caster_pass.Get());
    dispatch();

    transition(list, scene.target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    copy_to_readback(list, scene.target.Get(), scene.color_readback.Get(), scene.color_footprint);
    auto* depth = scene.depths[frame].Get();
    transition(list, depth, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
    copy_to_readback(list, depth, scene.depth_readback.Get(), scene.depth_footprint);
    if (scene.two_frames && frame == 0) {
        transition(list, scene.target.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
        transition(list, depth, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    }
    transition(list, scene.trace.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(scene.trace_readback.Get(), 0, scene.trace.Get(), 0, scene.draws * Scene::trace_stride);
    transition(list, scene.trace.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    scene.recorded_frame = frame;
    scene.ready = true;
}

UINT64 MeshShadowDraw::submit(UINT frame)
{
    auto& scene = *scene_;
    if (!scene.ready || frame != scene.recorded_frame) return 0;
    check(scene.commands->list->Close(), "Close shadow command list");
    ID3D12CommandList* command_lists[]{scene.commands->list.Get()};
    scene.commands->queue->ExecuteCommandLists(1, command_lists);
    const UINT64 value = scene.submitted + 1;
    check(scene.commands->queue->Signal(scene.fence.Get(), value), "Signal shadow fence");
    scene.submitted = value;
    scene.ready = false;
    return value;
}

UINT64 MeshShadowDraw::wait(std::vector<unsigned char>& pixels)
{
    auto& scene = *scene_;
    if (scene.completed == scene.submitted) throw std::runtime_error("No submitted shadow frame");
    scene.wait_for_gpu();
    const auto* expected = scene.input.data() + scene.traces_at;
    const UINT trace_bytes = scene.draws * Scene::trace_stride;
    std::vector<char> partial;
    if (scene.two_frames && scene.recorded_frame == 0) {
        partial.assign(expected, expected + trace_bytes);
        const auto camera = (scene.previous + scene.current) * Scene::trace_stride;
        std::fill(partial.begin() + scene.previous * Scene::trace_stride, partial.begin() + camera, char(0xcd));
        std::fill(partial.begin() + camera + 2 * Scene::trace_stride, partial.end(), char(0xcd));
        expected = partial.data();
    }
    void* mapped = nullptr;
    check(scene.trace_readback->Map(0, nullptr, &mapped), "Map shadow trace");
    const bool trace_ok = std::memcmp(mapped, expected, trace_bytes) == 0;
    const D3D12_RANGE empty{};
    scene.trace_readback->Unmap(0, &empty);
    if (!trace_ok) throw std::runtime_error("Mesh export trace mismatch");
    const auto bytes = scene.side * scene.side * 4;
    pixels.resize(2 * bytes);
    read_rows(scene.color_readback.Get(), scene.color_footprint, scene.side, pixels.data());
    read_rows(scene.depth_readback.Get(), scene.depth_footprint, scene.side, pixels.data() + bytes);
    ++scene.next_frame;
    return scene.completed;
}

UINT MeshShadowDraw::release_surfaces()
{
    auto& scene = *scene_;
    if (scene.completed != scene.submitted) scene.wait_for_gpu();
    // Drop the unsubmitted list in the rejection case before releasing its surfaces.
    scene.mesh_list.Reset();
    scene.commands.reset();
    unsigned releases = 0;
    for (auto& depth : scene.depths) {
        if (depth) {
            depth.Reset();
            ++releases;
        }
    }
    if (scene.target) {
        scene.target.Reset();
        ++releases;
    }
    return releases;
}

void MeshShadowDraw::compare_reference(const std::vector<unsigned char>& pixels, bool stale) const
{
    const auto& scene = *scene_;
    const auto bytes = scene.side * scene.side * 4;
    const auto* color = scene.input.data() + scene.image_at;
    const unsigned mismatches = count_mismatches(pixels.data(), color, bytes);
    const unsigned depth_mismatches = count_mismatches(pixels.data() + bytes, color + bytes, bytes);
    std::printf("light_draws=%u+%u camera_draws=2 bound_map=%s\n", scene.previous, scene.current,
                stale ? "previous" : "current");
    std::printf("pixels=%u mismatches=%u\n", scene.side * scene.side, mismatches);
    std::printf("shadow_map_texels=%u shadow_map_mismatches=%u\n", scene.side * scene.side, depth_mismatches);
    if (depth_mismatches) throw std::runtime_error("Shadow map reference mismatch");
    if (mismatches) throw std::runtime_error("Draw reference mismatch");
}

bool check_mesh_shadow_draw(ID3D12Device* device, const char* mesh, const char* receiver,
                            const char* caster, const char* fixture, bool stale)
{
    MeshShadowDraw draw(device, mesh, receiver, caster, fixture, false);
    if (!draw.supported()) return false;
    draw.record(1, stale ? 0u : 1u);
    if (!draw.submit(1)) throw std::runtime_error("Standalone shadow submission rejected");
    std::vector<unsigned char> pixels;
    draw.wait(pixels);
    draw.compare_reference(pixels, stale);
    return true;
}
