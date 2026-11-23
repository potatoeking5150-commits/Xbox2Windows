#include "quad_frames.hpp"

#include <dxgi1_6.h>

#include <climits>
#include <cstdio>
#include <cstring>

#include "../shim_gpu_device.h"
#include "tex_quad_ps.h"
#include "tex_quad_vs.h"

namespace {
struct Vertex {
    float x, y, z;
    float u, v;
};

// Two triangles covering NDC [-1, 1]^2; V points down the target (as in shim_tex_gpu.cpp).
constexpr Vertex quad[6] = {
    {-1, 1, 0, 0, 0}, {1, 1, 0, 1, 0}, {-1, -1, 0, 0, 1},
    {1, 1, 0, 1, 0},  {1, -1, 0, 1, 1}, {-1, -1, 0, 0, 1},
};

D3D12_HEAP_PROPERTIES heap(D3D12_HEAP_TYPE type)
{
    D3D12_HEAP_PROPERTIES properties{};
    properties.Type = type;
    properties.CreationNodeMask = properties.VisibleNodeMask = 1;
    return properties;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes)
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                                  D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    return barrier;
}

UINT64 fnv1a64(UINT64 hash, const unsigned char* bytes, size_t size)
{
    for (size_t i = 0; i < size; ++i) hash = (hash ^ bytes[i]) * 0x100000001b3ull;
    return hash;
}
} // namespace

HRESULT QuadFrames::create(UINT side) noexcept
{
    side_ = side;
    HRESULT hr = shim_try_create_device(device_);
    if (SUCCEEDED(hr)) hr = create_pipeline();
    if (SUCCEEDED(hr)) hr = create_commands();
    if (SUCCEEDED(hr)) hr = create_target();
    return hr;
}

// Root signature with one SRV table and a static point-clamp sampler, and the tex_quad pipeline.
HRESULT QuadFrames::create_pipeline() noexcept
{
    HRESULT hr = S_OK;
    D3D12_DESCRIPTOR_RANGE range{};
    range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    range.NumDescriptors = 1;
    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    param.DescriptorTable.NumDescriptorRanges = 1;
    param.DescriptorTable.pDescriptorRanges = &range;
    param.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_STATIC_SAMPLER_DESC sampler{};
    sampler.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    sampler.AddressU = sampler.AddressV = sampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    sampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
    D3D12_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.NumParameters = 1;
    root_desc.pParameters = &param;
    root_desc.NumStaticSamplers = 1;
    root_desc.pStaticSamplers = &sampler;
    root_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    Ptr<ID3DBlob> blob;
    hr = shim_report(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr),
                     "Serialize quad root signature");
    if (FAILED(hr)) return hr;
    hr = shim_report(device_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                                  IID_PPV_ARGS(&root_)), "Create quad root signature");
    if (FAILED(hr)) return hr;

    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = root_.Get();
    pso.VS = {tex_quad_vs, sizeof(tex_quad_vs)};
    pso.PS = {tex_quad_ps, sizeof(tex_quad_ps)};
    const D3D12_INPUT_ELEMENT_DESC elements[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };
    pso.InputLayout = {elements, 2};
    pso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    auto& blend = pso.BlendState.RenderTarget[0];
    blend.SrcBlend = blend.SrcBlendAlpha = D3D12_BLEND_ONE;
    blend.DestBlend = blend.DestBlendAlpha = D3D12_BLEND_ZERO;
    blend.BlendOp = blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pso.SampleMask = UINT_MAX;
    pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pso.NumRenderTargets = 1;
    pso.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    pso.SampleDesc.Count = 1;
    hr = shim_report(device_->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&pipeline_)), "Create quad pipeline");
    if (FAILED(hr)) return hr;
    return S_OK;
}

// One direct queue, allocator, command list (closed until the first frame) and fence.
HRESULT QuadFrames::create_commands() noexcept
{
    HRESULT hr = S_OK;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    hr = shim_report(device_->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue_)), "Create quad queue");
    if (FAILED(hr)) return hr;
    hr = shim_report(device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator_)),
                     "Create quad allocator");
    if (FAILED(hr)) return hr;
    hr = shim_report(device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator_.Get(), nullptr,
                                                IID_PPV_ARGS(&list_)), "Create quad command list");
    if (FAILED(hr)) return hr;
    hr = shim_report(list_->Close(), "Close initial quad command list");
    if (FAILED(hr)) return hr;
    hr = shim_report(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)), "Create quad fence");
    if (FAILED(hr)) return hr;
    return S_OK;
}

// Fixed geometry and output: the quad, the render target, its readback buffer and the
// descriptor heaps. The SRV slot is written by each upload.
HRESULT QuadFrames::create_target() noexcept
{
    HRESULT hr = S_OK;
    const auto upload_heap = heap(D3D12_HEAP_TYPE_UPLOAD);
    const auto vb_desc = buffer_desc(sizeof(quad));
    hr = shim_report(device_->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &vb_desc,
                                                      D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                      IID_PPV_ARGS(&vertices_)), "Create quad vertices");
    if (FAILED(hr)) return hr;
    const D3D12_RANGE no_read{0, 0};
    void* mapped = nullptr;
    hr = shim_report(vertices_->Map(0, &no_read, &mapped), "Map quad vertices");
    if (FAILED(hr)) return hr;
    std::memcpy(mapped, quad, sizeof(quad));
    vertices_->Unmap(0, nullptr);
    vertex_view_ = {vertices_->GetGPUVirtualAddress(), sizeof(quad), sizeof(quad[0])};

    D3D12_RESOURCE_DESC target_desc{};
    target_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    target_desc.Width = target_desc.Height = side_;
    target_desc.DepthOrArraySize = target_desc.MipLevels = 1;
    target_desc.SampleDesc.Count = 1;
    target_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    target_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_CLEAR_VALUE clear{};
    clear.Format = target_desc.Format;
    clear.Color[3] = 1;
    const auto default_heap = heap(D3D12_HEAP_TYPE_DEFAULT);
    hr = shim_report(device_->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &target_desc,
                                                      D3D12_RESOURCE_STATE_RENDER_TARGET, &clear,
                                                      IID_PPV_ARGS(&target_)), "Create quad target");
    if (FAILED(hr)) return hr;
    device_->GetCopyableFootprints(&target_desc, 0, 1, 0, &target_footprint_, nullptr, nullptr, nullptr);
    const auto readback_heap = heap(D3D12_HEAP_TYPE_READBACK);
    const auto readback_desc =
        buffer_desc(target_footprint_.Offset + UINT64{target_footprint_.Footprint.RowPitch} * side_);
    hr = shim_report(device_->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                      D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                      IID_PPV_ARGS(&readback_)), "Create quad readback");
    if (FAILED(hr)) return hr;

    D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{};
    rtv_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtv_desc.NumDescriptors = 1;
    hr = shim_report(device_->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtv_heap_)), "Create quad RTV heap");
    if (FAILED(hr)) return hr;
    device_->CreateRenderTargetView(target_.Get(), nullptr, rtv_heap_->GetCPUDescriptorHandleForHeapStart());
    D3D12_DESCRIPTOR_HEAP_DESC srv_desc{};
    srv_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    srv_desc.NumDescriptors = 1;
    srv_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    hr = shim_report(device_->CreateDescriptorHeap(&srv_desc, IID_PPV_ARGS(&srv_heap_)), "Create quad SRV heap");
    if (FAILED(hr)) return hr;
    return S_OK;
}

// Replaces the texture and records its copy. The previous frame has completed,
// so the old texture, staging buffer and SRV slot are no longer in use.
HRESULT QuadFrames::upload(const TextureImage& image, UINT64& hash) noexcept
{
    HRESULT hr = S_OK;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = image.width;
    desc.Height = image.height;
    desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Format = image.format;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{};
    UINT rows = 0;
    UINT64 row_bytes = 0, staging_bytes = 0;
    device_->GetCopyableFootprints(&desc, 0, 1, 0, &layout, &rows, &row_bytes, &staging_bytes);
    // The source must be exactly the rows D3D12 copies (block rows for BC formats).
    if (!rows || row_bytes * rows != image.size) {
        std::fprintf(stderr, "texture geometry mismatch: rows=%u row_bytes=%llu size=%zu\n", rows,
                     static_cast<unsigned long long>(row_bytes), image.size);
        return E_INVALIDARG;
    }
    texture_.Reset();
    staging_.Reset();
    const auto default_heap = heap(D3D12_HEAP_TYPE_DEFAULT);
    hr = shim_report(device_->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                      D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                      IID_PPV_ARGS(&texture_)), "Create quad texture");
    if (FAILED(hr)) return hr;
    const auto upload_heap = heap(D3D12_HEAP_TYPE_UPLOAD);
    const auto staging_desc = buffer_desc(staging_bytes);
    hr = shim_report(device_->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &staging_desc,
                                                      D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                      IID_PPV_ARGS(&staging_)), "Create quad staging");
    if (FAILED(hr)) return hr;
    const D3D12_RANGE no_read{0, 0};
    void* mapped = nullptr;
    hr = shim_report(staging_->Map(0, &no_read, &mapped), "Map quad staging");
    if (FAILED(hr)) return hr;
    auto* stage = static_cast<unsigned char*>(mapped) + layout.Offset;
    hash = 0xcbf29ce484222325ull;
    for (UINT y = 0; y < rows; ++y) {
        auto* row = stage + UINT64{y} * layout.Footprint.RowPitch;
        std::memcpy(row, image.bytes + y * row_bytes, static_cast<size_t>(row_bytes));
        hash = fnv1a64(hash, row, static_cast<size_t>(row_bytes));
    }
    staging_->Unmap(0, nullptr);
    device_->CreateShaderResourceView(texture_.Get(), nullptr, srv_heap_->GetCPUDescriptorHandleForHeapStart());

    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = texture_.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = staging_.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint = layout;
    list_->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    const auto ready = transition(texture_.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                  D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    list_->ResourceBarrier(1, &ready);
    return S_OK;
}

HRESULT QuadFrames::draw(const TextureImage* image, unsigned char* rgba, size_t size,
                         FrameResult& result) noexcept
{
    HRESULT hr = S_OK;
    result = {};
    hr = shim_report(allocator_->Reset(), "Reset quad allocator");
    if (FAILED(hr)) return hr;
    hr = shim_report(list_->Reset(allocator_.Get(), pipeline_.Get()), "Reset quad command list");
    if (FAILED(hr)) return hr;
    if (image) {
        hr = upload(*image, result.upload_hash);  // Reports its own failure.
        if (FAILED(hr)) return hr;
    }
    if (!texture_) return shim_report(E_INVALIDARG, "Draw quad without a texture");

    record_draw();
    hr = shim_report(list_->Close(), "Close quad command list");
    if (FAILED(hr)) return hr;

    ID3D12CommandList* lists[]{list_.Get()};
    queue_->ExecuteCommandLists(1, lists);
    hr = shim_report(queue_->Signal(fence_.Get(), ++fence_value_), "Signal quad fence");
    if (FAILED(hr)) return hr;
    hr = shim_report(fence_->SetEventOnCompletion(fence_value_, nullptr), "Wait for quad frame");
    if (FAILED(hr)) return hr;
    hr = shim_report(device_->GetDeviceRemovedReason(), "Quad frame completion");
    if (FAILED(hr)) return hr;
    result.fence = fence_->GetCompletedValue();
    return read_frame(rgba, size);
}

// Clears the target, draws the quad and copies the target into the readback buffer.
void QuadFrames::record_draw() noexcept
{
    const D3D12_VIEWPORT viewport{0, 0, static_cast<float>(side_), static_cast<float>(side_), 0, 1};
    const D3D12_RECT scissor{0, 0, static_cast<LONG>(side_), static_cast<LONG>(side_)};
    const auto rtv = rtv_heap_->GetCPUDescriptorHandleForHeapStart();
    const float clear[4] = {0, 0, 0, 1};
    ID3D12DescriptorHeap* heaps[]{srv_heap_.Get()};
    list_->SetDescriptorHeaps(1, heaps);
    list_->SetGraphicsRootSignature(root_.Get());
    list_->SetGraphicsRootDescriptorTable(0, srv_heap_->GetGPUDescriptorHandleForHeapStart());
    list_->RSSetViewports(1, &viewport);
    list_->RSSetScissorRects(1, &scissor);
    list_->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    list_->ClearRenderTargetView(rtv, clear, 0, nullptr);
    list_->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    list_->IASetVertexBuffers(0, 1, &vertex_view_);
    list_->DrawInstanced(6, 1, 0, 0);

    const auto to_copy = transition(target_.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET,
                                    D3D12_RESOURCE_STATE_COPY_SOURCE);
    list_->ResourceBarrier(1, &to_copy);
    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = target_.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = readback_.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = target_footprint_;
    list_->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    // Return the target to its initial state so every frame starts the same way.
    const auto to_render = transition(target_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                      D3D12_RESOURCE_STATE_RENDER_TARGET);
    list_->ResourceBarrier(1, &to_render);
}

// Copies the completed frame out of the readback buffer, dropping the row padding.
HRESULT QuadFrames::read_frame(unsigned char* rgba, size_t size) noexcept
{
    HRESULT hr = S_OK;
    if (size != size_t{side_} * side_ * 4) return shim_report(E_INVALIDARG, "Quad frame buffer size");
    const UINT64 bytes = target_footprint_.Offset + UINT64{target_footprint_.Footprint.RowPitch} * side_;
    const D3D12_RANGE read{0, static_cast<SIZE_T>(bytes)};
    void* mapped = nullptr;
    hr = shim_report(readback_->Map(0, &read, &mapped), "Map quad readback");
    if (FAILED(hr)) return hr;
    for (UINT y = 0; y < side_; ++y) {
        const auto* row = static_cast<const unsigned char*>(mapped) + target_footprint_.Offset +
                          UINT64{y} * target_footprint_.Footprint.RowPitch;
        std::memcpy(rgba + size_t{y} * side_ * 4, row, size_t{side_} * 4);
    }
    const D3D12_RANGE no_writes{0, 0};
    readback_->Unmap(0, &no_writes);
    return S_OK;
}

HRESULT QuadFrames::adapter(char* text, size_t size) const noexcept
{
    HRESULT hr = S_OK;
    Ptr<IDXGIFactory4> factory;
    hr = shim_report(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");
    if (FAILED(hr)) return hr;
    Ptr<IDXGIAdapter1> adapter;
    hr = shim_report(factory->EnumAdapterByLuid(device_->GetAdapterLuid(), IID_PPV_ARGS(&adapter)),
                     "EnumAdapterByLuid");
    if (FAILED(hr)) return hr;
    DXGI_ADAPTER_DESC1 desc{};
    hr = shim_report(adapter->GetDesc1(&desc), "GetDesc1");
    if (FAILED(hr)) return hr;
    char name[256]{};
    WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name, sizeof(name) - 1, nullptr, nullptr);
    LARGE_INTEGER version{};
    hr = shim_report(adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &version), "Query driver version");
    if (FAILED(hr)) return hr;
    const auto v = static_cast<unsigned long long>(version.QuadPart);
    std::snprintf(text, size, "%s, driver %llu.%llu.%llu.%llu", name, v >> 48, (v >> 32) & 0xffff,
                  (v >> 16) & 0xffff, v & 0xffff);
    return S_OK;
}

ULONG QuadFrames::release() noexcept
{
    staging_.Reset();
    texture_.Reset();
    srv_heap_.Reset();
    rtv_heap_.Reset();
    readback_.Reset();
    target_.Reset();
    vertices_.Reset();
    fence_.Reset();
    list_.Reset();
    allocator_.Reset();
    queue_.Reset();
    pipeline_.Reset();
    root_.Reset();
    ID3D12Device* device = device_.Detach();
    return device ? device->Release() : 0;
}
