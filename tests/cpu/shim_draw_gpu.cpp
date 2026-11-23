// Render the repository's 64x64 test triangle offscreen and return raw RGBA bytes.
//
// The draw mirrors the non-reference path of tests/offscreen_check.cpp: the same
// triangle.hlsl vertex/pixel shaders, a 64x64 R8G8B8A8 render target, no vertex
// buffer (SV_VertexID positions), and a fence-waited copy to a readback buffer.
// Mode 1 renders the exact triangle; mode 2 renders the FLIP variant, which mirrors
// X in the vertex shader. The provider compares the readback
// against the CPU pixel rule and publishes the bytes for the cross-runner check.
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "shim_draw_color.h"
#include "shim_draw_texture_init.h"
#include "shim_draw_texture_read.h"
#include "shim_draw_texture_swap.h"
#include "triangle_vs.h"
#include "triangle_ps.h"
#include "triangle_vs_flip.h"
#include "triangle_ps_flip.h"

#include "shim_gpu_device.h"

namespace {
using Microsoft::WRL::ComPtr;

constexpr UINT side = 64;
enum class TextureStep { none, initialize, update, stale, reuse };

inline void require(HRESULT hr, const char* operation)
{
    shim_require(hr, operation);
}

inline ComPtr<ID3D12Device> create_device()
{
    return shim_create_device();
}
class DrawSession {
public:
    DrawSession(bool flip, bool persistent, bool textured = false) : device(create_device()), uses_color(persistent)
    {
        D3D12_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        D3D12_ROOT_PARAMETER color{};
        D3D12_DESCRIPTOR_RANGE texture_range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
        if (textured) {
            color.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            color.DescriptorTable = {1, &texture_range};
        } else if (uses_color) {
            color.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            color.Constants = {0, 0, 4};
        }
        if (uses_color || textured) {
            color.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
            root_desc.NumParameters = 1;
            root_desc.pParameters = &color;
        }
        ComPtr<ID3DBlob> blob;
        require(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr),
                "Serialize draw root signature");
        require(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                            IID_PPV_ARGS(&root)), "Create draw root signature");
        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = root.Get();
        if (flip) {
            pso.VS = {triangle_vs_flip, sizeof(triangle_vs_flip)};
            pso.PS = {triangle_ps_flip, sizeof(triangle_ps_flip)};
        } else {
            pso.VS = {triangle_vs, sizeof(triangle_vs)};
            pso.PS = {triangle_ps, sizeof(triangle_ps)};
        }
        if (uses_color) pso.PS = {shim_draw_color, sizeof(shim_draw_color)};
        if (textured) pso.PS = {shim_draw_texture_read, sizeof(shim_draw_texture_read)};
        const D3D12_INPUT_ELEMENT_DESC elements[] = {
            {"SV_Position", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 16, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
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
        require(device->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&pipeline)), "Create draw pipeline");
        if (textured) {
            pso.PS = {shim_draw_texture_swap, sizeof(shim_draw_texture_swap)};
            require(device->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&swap_pipeline)), "Create texture reuse pipeline");
        }

        D3D12_RESOURCE_DESC tex{};
        tex.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        tex.Width = tex.Height = side;
        tex.DepthOrArraySize = tex.MipLevels = 1;
        tex.SampleDesc.Count = 1;
        tex.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        tex.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        D3D12_CLEAR_VALUE clear{};
        clear.Format = tex.Format;
        clear.Color[3] = 1;
        D3D12_HEAP_PROPERTIES target_heap{};
        target_heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        target_heap.CreationNodeMask = target_heap.VisibleNodeMask = 1;
        if (textured) setup_texture(tex, target_heap);
        require(device->CreateCommittedResource(&target_heap, D3D12_HEAP_FLAG_NONE, &tex,
                                                D3D12_RESOURCE_STATE_RENDER_TARGET, &clear,
                                                IID_PPV_ARGS(&target)), "Create draw target");
        device->GetCopyableFootprints(&tex, 0, 1, 0, &footprint, nullptr, nullptr, nullptr);
        byte_count = footprint.Offset + UINT64{footprint.Footprint.RowPitch} * side;
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = byte_count;
        buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        D3D12_HEAP_PROPERTIES readback_heap{};
        readback_heap.Type = D3D12_HEAP_TYPE_READBACK;
        readback_heap.CreationNodeMask = readback_heap.VisibleNodeMask = 1;
        require(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                IID_PPV_ARGS(&readback)), "Create draw readback");
        D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
        heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        heap_desc.NumDescriptors = 1;
        require(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&descriptors)), "Create draw RTV heap");
        rtv = descriptors->GetCPUDescriptorHandleForHeapStart();
        device->CreateRenderTargetView(target.Get(), nullptr, rtv);

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        require(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)), "Create draw queue");
        require(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)),
                "Create draw allocator");
        require(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), pipeline.Get(),
                                          IID_PPV_ARGS(&list)), "Create draw command list");
        require(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "Create draw fence");
    }

    void render(bool red, std::vector<unsigned char>& rgba, TextureStep step = TextureStep::none)
    {
        rgba.resize(side * side * 4);
        if (submitted_frames) {
            require(allocator->Reset(), "Reset draw allocator");
            require(list->Reset(allocator.Get(), pipeline.Get()), "Reset draw command list");
            transition(target.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
        }
        if (step != TextureStep::none) record_texture(step);
        const D3D12_VIEWPORT viewport{0, 0, side, side, 0, 1};
        const D3D12_RECT scissor{0, 0, side, side};
        list->SetGraphicsRootSignature(root.Get());
        if (step != TextureStep::none)
            list->SetGraphicsRootDescriptorTable(0, texture_descriptors->GetGPUDescriptorHandleForHeapStart());
        if (uses_color) {
            const float color[]{red ? 1.0f : 0.0f, red ? 0.0f : 1.0f, 0.0f, 1.0f};
            list->SetGraphicsRoot32BitConstants(0, 4, color, 0);
        }
        list->RSSetViewports(1, &viewport);
        list->RSSetScissorRects(1, &scissor);
        list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        const float clear[]{0, 0, 0, 1};
        list->ClearRenderTargetView(rtv, clear, 0, nullptr);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list->DrawInstanced(3, 1, 0, 0);
        transition(target.Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = target.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = readback.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = footprint;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        require(list->Close(), "Close draw command list");
        ID3D12CommandList* commands[]{list.Get()};
        queue->ExecuteCommandLists(1, commands);
        ++submitted_frames;
        require(queue->Signal(fence.Get(), submitted_frames), "Signal draw fence");
        require(fence->SetEventOnCompletion(submitted_frames, nullptr), "Wait for draw fence");
        require(device->GetDeviceRemovedReason(), "Draw completion");

        void* mapped = nullptr;
        const D3D12_RANGE range{0, static_cast<SIZE_T>(byte_count)};
        require(readback->Map(0, &range, &mapped), "Map draw readback");
        for (UINT y = 0; y < side; ++y) {
            const auto* row = static_cast<const unsigned char*>(mapped) + footprint.Offset +
                              y * footprint.Footprint.RowPitch;
            std::memcpy(rgba.data() + y * side * 4, row, side * 4);
        }
        const D3D12_RANGE no_writes{0, 0};
        readback->Unmap(0, &no_writes);
        if (uses_color || step != TextureStep::none)
            std::fprintf(stderr, "draw fence completed %llu\n", static_cast<unsigned long long>(submitted_frames));
    }

private:
    void setup_texture(D3D12_RESOURCE_DESC tex, const D3D12_HEAP_PROPERTIES& heap)
    {
        tex.Width = tex.Height = 2;
        tex.Format = DXGI_FORMAT_R32_UINT;
        tex.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        require(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &tex,
                                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                IID_PPV_ARGS(&texture)), "Create persistent texture");
        const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2,
                                             D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
        require(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&texture_descriptors)), "Create texture views heap");
        const UINT increment = device->GetDescriptorHandleIncrementSize(desc.Type);
        const auto cpu = texture_descriptors->GetCPUDescriptorHandleForHeapStart();
        device->CreateShaderResourceView(texture.Get(), nullptr, cpu);
        device->CreateUnorderedAccessView(texture.Get(), nullptr, nullptr, {cpu.ptr + increment});
        compute_view = {texture_descriptors->GetGPUDescriptorHandleForHeapStart().ptr + increment};

        D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 0};
        D3D12_ROOT_PARAMETER params[2]{};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[0].DescriptorTable = {1, &range};
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[1].Constants = {0, 0, 1};
        const D3D12_ROOT_SIGNATURE_DESC layout{2, params};
        ComPtr<ID3DBlob> blob;
        require(D3D12SerializeRootSignature(&layout, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr),
                "Serialize texture compute root");
        require(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                            IID_PPV_ARGS(&compute_root)), "Create texture compute root");
        D3D12_COMPUTE_PIPELINE_STATE_DESC compute{};
        compute.pRootSignature = compute_root.Get();
        compute.CS = {shim_draw_texture_init, sizeof(shim_draw_texture_init)};
        require(device->CreateComputePipelineState(&compute, IID_PPV_ARGS(&compute_pipeline)),
                "Create texture compute pipeline");
    }

    void record_texture(TextureStep step)
    {
        ID3D12DescriptorHeap* heaps[]{texture_descriptors.Get()};
        list->SetDescriptorHeaps(1, heaps);
        if (step == TextureStep::initialize || step == TextureStep::update) {
            const UINT partial = step == TextureStep::update;
            if (partial)
                transition(texture.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            list->SetPipelineState(compute_pipeline.Get());
            list->SetComputeRootSignature(compute_root.Get());
            list->SetComputeRootDescriptorTable(0, compute_view);
            list->SetComputeRoot32BitConstant(1, partial, 0);
            list->Dispatch(1, 1, 1);
            transition(texture.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        }
        list->SetPipelineState(step == TextureStep::reuse ? swap_pipeline.Get() : pipeline.Get());
    }

    void transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
    {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {resource, 0, before, after};
        list->ResourceBarrier(1, &barrier);
    }

    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pipeline;
    ComPtr<ID3D12Resource> target, readback;
    ComPtr<ID3D12DescriptorHeap> descriptors;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12Resource> texture;
    ComPtr<ID3D12DescriptorHeap> texture_descriptors;
    ComPtr<ID3D12RootSignature> compute_root;
    ComPtr<ID3D12PipelineState> compute_pipeline, swap_pipeline;
    D3D12_GPU_DESCRIPTOR_HANDLE compute_view{};
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    D3D12_CPU_DESCRIPTOR_HANDLE rtv{};
    UINT64 byte_count = 0;
    UINT64 submitted_frames = 0;
    bool uses_color;
};
} // namespace

bool shim_draw_gpu(unsigned mode, std::vector<unsigned char>& rgba)
{
    static thread_local std::unique_ptr<DrawSession> pending;
    try {
        if (mode <= 2) {
            DrawSession(mode == 2, false).render(false, rgba);
        } else if (mode == 6 || mode == 10) {
            const bool textured = mode == 10;
            pending = std::make_unique<DrawSession>(false, !textured, textured);
            pending->render(false, rgba, textured ? TextureStep::initialize : TextureStep::none);
        } else if (mode == 11 || mode == 13) {
            if (!pending) return false;
            pending->render(false, rgba, mode == 11 ? TextureStep::update : TextureStep::stale);
        } else {
            auto session = std::move(pending);
            if (!session || mode == 9) return false;
            session->render(mode == 7, rgba, mode == 12 ? TextureStep::reuse : TextureStep::none);
        }
        return true;
    } catch (HRESULT) {
        pending.reset();
        return false;
    }
}
