// Render a textured quad offscreen and return raw RGBA bytes.
//
// The quad covers the full 64x64 render target while sampling an 8x8 texture
// with point filtering and clamp addressing, so pixel (x, y) must reproduce
// texel (x / 8, y / 8) exactly. Mode 3 uploads the RGBA8 table, mode 6 the
// BC7 blocks (both from the generated tex_tables.h); mode 4 shifts U by half
// a texel in the uploaded vertex buffer; mode 5 flips texel (7, 7) in the
// texture staging buffer to simulate a corrupted upload. The provider compares
// the readback against the matching CPU rule and publishes the bytes for the
// cross-runner check.
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <cstdio>
#include <cstring>
#include <vector>

#include "shim_gpu_device.h"
#include "tex_quad_vs.h"
#include "tex_quad_ps.h"
#include "tex_tables.h"

namespace {
using Microsoft::WRL::ComPtr;

constexpr UINT side = 64;
constexpr UINT tex_side = 8;

inline void require(HRESULT hr, const char* operation)
{
    shim_require(hr, operation);
}

struct Vertex {
    float x, y, z;
    float u, v;
};

void fill_quad(Vertex* out, bool shift_u)
{
    // Two triangles covering NDC [-1, 1]^2; V points down the target.
    const float du = shift_u ? 1.0f / 16 : 0.0f;
    const Vertex verts[6] = {
        {-1, 1, 0, 0 + du, 0}, {1, 1, 0, 1 + du, 0}, {-1, -1, 0, 0 + du, 1},
        {1, 1, 0, 1 + du, 0}, {1, -1, 0, 1 + du, 1}, {-1, -1, 0, 0 + du, 1},
    };
    std::memcpy(out, verts, sizeof(verts));
}

// Copy the table row by row using the row geometry D3D12 reports for the
// subresource. For block-compressed formats the footprint height is in texels
// but only `rows` block rows exist, so iterating the height would run past the
// source table and the staging buffer.
void fill_staging(ID3D12Resource* staging, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& layout, UINT rows,
                  UINT64 row_bytes, const unsigned char* table, UINT table_row_bytes, size_t table_bytes,
                  bool corrupt_texel = false)
{
    // The copy must consume the source table exactly and stay inside the staging buffer.
    const UINT64 staging_end = layout.Offset + UINT64{rows ? rows - 1 : 0} * layout.Footprint.RowPitch + row_bytes;
    if (rows == 0 || row_bytes != table_row_bytes || UINT64{rows} * table_row_bytes != table_bytes ||
        staging_end > staging->GetDesc().Width) {
        std::fprintf(stderr, "staging geometry mismatch: rows=%u row_bytes=%llu table_row=%u table_bytes=%zu\n",
                     rows, static_cast<unsigned long long>(row_bytes), table_row_bytes, table_bytes);
        throw E_FAIL;
    }
    const D3D12_RANGE no_read{0, 0};
    void* mapped = nullptr;
    require(staging->Map(0, &no_read, &mapped), "Map texture staging");
    auto* stage = static_cast<unsigned char*>(mapped) + layout.Offset;
    for (UINT y = 0; y < rows; ++y)
        std::memcpy(stage + y * layout.Footprint.RowPitch, table + y * table_row_bytes, table_row_bytes);
    if (corrupt_texel) {
        // Flip every byte of texel (7, 7) after the legitimate fill (RGBA8 only).
        auto* victim = stage + 7 * layout.Footprint.RowPitch + 7 * 4;
        for (UINT i = 0; i < 4; ++i) victim[i] ^= 0xFF;
    }
    const D3D12_RANGE no_writes{0, 0};
    staging->Unmap(0, &no_writes);
}
} // namespace

bool shim_tex_gpu(unsigned mode, std::vector<unsigned char>& rgba)
{
    const bool bc7 = mode == 6 || mode == 7;
    const bool shift_u = mode == 4;
    const bool corrupt = mode == 5;
    const bool scene = mode == 7;
    if (mode < 3 || mode > 7) return false;
    rgba.assign(side * side * 4, 0);
    try {
        auto device = shim_create_device();

        D3D12_DESCRIPTOR_RANGE range{};
        range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        range.NumDescriptors = 1;
        range.BaseShaderRegister = 0;
        D3D12_ROOT_PARAMETER param{};
        param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        param.DescriptorTable.NumDescriptorRanges = 1;
        param.DescriptorTable.pDescriptorRanges = &range;
        param.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        D3D12_STATIC_SAMPLER_DESC samp{};
        samp.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        samp.AddressU = samp.AddressV = samp.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        samp.ShaderRegister = 0;
        samp.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
        D3D12_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.NumParameters = 1;
        root_desc.pParameters = &param;
        root_desc.NumStaticSamplers = 1;
        root_desc.pStaticSamplers = &samp;
        root_desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        ComPtr<ID3DBlob> blob;
        require(D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr),
                "Serialize textured root signature");
        ComPtr<ID3D12RootSignature> root;
        require(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                            IID_PPV_ARGS(&root)), "Create textured root signature");

        D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = root.Get();
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
        ComPtr<ID3D12PipelineState> pipeline;
        require(device->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&pipeline)), "Create textured pipeline");

        // Source texture(s) in DEFAULT heap, filled from UPLOAD staging buffers.
        // The scene case adds the second table as another SRV for the right half.
        const unsigned char* tables[2] = {tex_rgba8, nullptr};
        DXGI_FORMAT formats[2] = {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_UNKNOWN};
        UINT table_rows[2] = {tex_side * 4, 0};
        size_t table_bytes[2] = {sizeof(tex_rgba8), 0};
        unsigned table_count = 1;
        if (bc7) {
            tables[0] = tex_bc7_blocks;
            formats[0] = DXGI_FORMAT_BC7_UNORM;
            table_rows[0] = tex_side / 4 * 16;
            table_bytes[0] = sizeof(tex_bc7_blocks);
        }
        if (scene) {
            tables[0] = tex_rgba8;
            formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            table_rows[0] = tex_side * 4;
            table_bytes[0] = sizeof(tex_rgba8);
            tables[1] = tex_bc7_blocks;
            formats[1] = DXGI_FORMAT_BC7_UNORM;
            table_rows[1] = tex_side / 4 * 16;
            table_bytes[1] = sizeof(tex_bc7_blocks);
            table_count = 2;
        }
        D3D12_RESOURCE_DESC tex{};
        tex.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        tex.Width = tex.Height = tex_side;
        tex.DepthOrArraySize = tex.MipLevels = 1;
        tex.SampleDesc.Count = 1;
        tex.Format = formats[0];
        D3D12_HEAP_PROPERTIES tex_heap{};
        tex_heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        tex_heap.CreationNodeMask = tex_heap.VisibleNodeMask = 1;
        ComPtr<ID3D12Resource> texture;
        require(device->CreateCommittedResource(&tex_heap, D3D12_HEAP_FLAG_NONE, &tex,
                                                D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                IID_PPV_ARGS(&texture)), "Create source texture");
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{};
        UINT rows = 0;
        UINT64 row_bytes = 0, staging_bytes = 0;
        device->GetCopyableFootprints(&tex, 0, 1, 0, &layout, &rows, &row_bytes, &staging_bytes);
        D3D12_RESOURCE_DESC staging_desc{};
        staging_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        staging_desc.Width = staging_bytes;
        staging_desc.Height = staging_desc.DepthOrArraySize = staging_desc.MipLevels = 1;
        staging_desc.SampleDesc.Count = 1;
        staging_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        D3D12_HEAP_PROPERTIES upload_heap{};
        upload_heap.Type = D3D12_HEAP_TYPE_UPLOAD;
        upload_heap.CreationNodeMask = upload_heap.VisibleNodeMask = 1;
        ComPtr<ID3D12Resource> staging;
        require(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &staging_desc,
                                                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                IID_PPV_ARGS(&staging)), "Create texture staging");
        const D3D12_RANGE no_read{0, 0};
        fill_staging(staging.Get(), layout, rows, row_bytes, tables[0], table_rows[0], table_bytes[0], corrupt);
        ComPtr<ID3D12Resource> texture2;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout2{};
        ComPtr<ID3D12Resource> staging2;
        if (scene) {
            D3D12_RESOURCE_DESC tex2 = tex;
            tex2.Format = formats[1];
            require(device->CreateCommittedResource(&tex_heap, D3D12_HEAP_FLAG_NONE, &tex2,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                    IID_PPV_ARGS(&texture2)), "Create scene texture");
            UINT rows2 = 0;
            UINT64 row_bytes2 = 0, staging2_bytes = 0;
            device->GetCopyableFootprints(&tex2, 0, 1, 0, &layout2, &rows2, &row_bytes2, &staging2_bytes);
            D3D12_RESOURCE_DESC staging2_desc = staging_desc;
            staging2_desc.Width = staging2_bytes;
            require(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &staging2_desc,
                                                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                    IID_PPV_ARGS(&staging2)), "Create scene staging");
            fill_staging(staging2.Get(), layout2, rows2, row_bytes2, tables[1], table_rows[1], table_bytes[1]);
        }

        // Vertex buffer(s) in UPLOAD heap, like the reference path of offscreen_check.
        // The scene holds two quads: RGBA8 on the left half, BC7 on the right half.
        Vertex quad[12];
        fill_quad(quad, shift_u);
        if (scene) {
            const Vertex halves[12] = {
                {-1, 1, 0, 0, 0}, {0, 1, 0, 1, 0}, {-1, -1, 0, 0, 1},
                {0, 1, 0, 1, 0}, {0, -1, 0, 1, 1}, {-1, -1, 0, 0, 1},
                {0, 1, 0, 0, 0}, {1, 1, 0, 1, 0}, {0, -1, 0, 0, 1},
                {1, 1, 0, 1, 0}, {1, -1, 0, 1, 1}, {0, -1, 0, 0, 1},
            };
            std::memcpy(quad, halves, sizeof(halves));
        }
        D3D12_RESOURCE_DESC vb_desc = staging_desc;
        vb_desc.Width = sizeof(quad);
        ComPtr<ID3D12Resource> vertices;
        require(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &vb_desc,
                                                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                IID_PPV_ARGS(&vertices)), "Create textured vertex buffer");
        void* mapped_vb = nullptr;
        require(vertices->Map(0, &no_read, &mapped_vb), "Map textured vertices");
        std::memcpy(mapped_vb, quad, sizeof(quad));
        vertices->Unmap(0, &no_read);
        const D3D12_VERTEX_BUFFER_VIEW view{vertices->GetGPUVirtualAddress(), sizeof(quad), sizeof(quad[0])};

        D3D12_RESOURCE_DESC target_desc{};
        target_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        target_desc.Width = target_desc.Height = side;
        target_desc.DepthOrArraySize = target_desc.MipLevels = 1;
        target_desc.SampleDesc.Count = 1;
        target_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        target_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        D3D12_CLEAR_VALUE clear{};
        clear.Format = target_desc.Format;
        clear.Color[3] = 1;
        D3D12_HEAP_PROPERTIES target_heap{};
        target_heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        target_heap.CreationNodeMask = target_heap.VisibleNodeMask = 1;
        ComPtr<ID3D12Resource> target;
        require(device->CreateCommittedResource(&target_heap, D3D12_HEAP_FLAG_NONE, &target_desc,
                                                D3D12_RESOURCE_STATE_RENDER_TARGET, &clear,
                                                IID_PPV_ARGS(&target)), "Create textured target");
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        device->GetCopyableFootprints(&target_desc, 0, 1, 0, &footprint, nullptr, nullptr, nullptr);
        const UINT64 byte_count = footprint.Offset + UINT64{footprint.Footprint.RowPitch} * side;
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = byte_count;
        buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        D3D12_HEAP_PROPERTIES readback_heap{};
        readback_heap.Type = D3D12_HEAP_TYPE_READBACK;
        readback_heap.CreationNodeMask = readback_heap.VisibleNodeMask = 1;
        ComPtr<ID3D12Resource> readback;
        require(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                IID_PPV_ARGS(&readback)), "Create textured readback");

        D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{};
        rtv_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        rtv_desc.NumDescriptors = 1;
        ComPtr<ID3D12DescriptorHeap> rtv_heap;
        require(device->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtv_heap)), "Create textured RTV heap");
        const auto rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
        device->CreateRenderTargetView(target.Get(), nullptr, rtv);
        D3D12_DESCRIPTOR_HEAP_DESC srv_desc{};
        srv_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        srv_desc.NumDescriptors = table_count;
        srv_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        ComPtr<ID3D12DescriptorHeap> srv_heap;
        require(device->CreateDescriptorHeap(&srv_desc, IID_PPV_ARGS(&srv_heap)), "Create textured SRV heap");
        const UINT srv_size = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_CPU_DESCRIPTOR_HANDLE srv_cpu = srv_heap->GetCPUDescriptorHandleForHeapStart();
        device->CreateShaderResourceView(texture.Get(), nullptr, srv_cpu);
        D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu = srv_heap->GetGPUDescriptorHandleForHeapStart();
        D3D12_GPU_DESCRIPTOR_HANDLE srv_gpu2 = srv_gpu;
        if (scene) {
            srv_cpu.ptr += srv_size;
            device->CreateShaderResourceView(texture2.Get(), nullptr, srv_cpu);
            srv_gpu2.ptr += srv_size;
        }

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        ComPtr<ID3D12CommandQueue> queue;
        require(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)), "Create textured queue");
        ComPtr<ID3D12CommandAllocator> allocator;
        require(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)),
                "Create textured allocator");
        ComPtr<ID3D12GraphicsCommandList> list;
        require(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), pipeline.Get(),
                                          IID_PPV_ARGS(&list)), "Create textured command list");
        D3D12_TEXTURE_COPY_LOCATION tex_dst{};
        tex_dst.pResource = texture.Get();
        tex_dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION tex_src{};
        tex_src.pResource = staging.Get();
        tex_src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        tex_src.PlacedFootprint = layout;
        list->CopyTextureRegion(&tex_dst, 0, 0, 0, &tex_src, nullptr);
        if (scene) {
            D3D12_TEXTURE_COPY_LOCATION tex2_dst{};
            tex2_dst.pResource = texture2.Get();
            tex2_dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            D3D12_TEXTURE_COPY_LOCATION tex2_src{};
            tex2_src.pResource = staging2.Get();
            tex2_src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            tex2_src.PlacedFootprint = layout2;
            list->CopyTextureRegion(&tex2_dst, 0, 0, 0, &tex2_src, nullptr);
        }
        D3D12_RESOURCE_BARRIER ready[2]{};
        ready[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        ready[0].Transition = {texture.Get(), 0, D3D12_RESOURCE_STATE_COPY_DEST,
                               D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE};
        UINT ready_count = 1;
        if (scene) {
            ready[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            ready[1].Transition = {texture2.Get(), 0, D3D12_RESOURCE_STATE_COPY_DEST,
                                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE};
            ready_count = 2;
        }
        list->ResourceBarrier(ready_count, ready);
        const D3D12_VIEWPORT viewport{0, 0, side, side, 0, 1};
        const D3D12_RECT scissor{0, 0, side, side};
        ID3D12DescriptorHeap* heaps[]{srv_heap.Get()};
        list->SetDescriptorHeaps(1, heaps);
        list->SetGraphicsRootSignature(root.Get());
        list->SetGraphicsRootDescriptorTable(0, srv_heap->GetGPUDescriptorHandleForHeapStart());
        list->RSSetViewports(1, &viewport);
        list->RSSetScissorRects(1, &scissor);
        list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
        list->ClearRenderTargetView(rtv, clear.Color, 0, nullptr);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list->IASetVertexBuffers(0, 1, &view);
        if (!scene) {
            list->DrawInstanced(6, 1, 0, 0);
        } else {
            // Two draw calls with a rebinding between them: RGBA8 left, BC7 right.
            list->DrawInstanced(6, 1, 0, 0);
            list->SetGraphicsRootDescriptorTable(0, srv_gpu2);
            list->DrawInstanced(6, 1, 6, 0);
        }
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {target.Get(), 0, D3D12_RESOURCE_STATE_RENDER_TARGET,
                              D3D12_RESOURCE_STATE_COPY_SOURCE};
        list->ResourceBarrier(1, &barrier);
        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = target.Get();
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource = readback.Get();
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = footprint;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        require(list->Close(), "Close textured command list");
        ComPtr<ID3D12Fence> fence;
        require(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "Create textured fence");
        ID3D12CommandList* submitted[]{list.Get()};
        queue->ExecuteCommandLists(1, submitted);
        require(queue->Signal(fence.Get(), 1), "Signal textured fence");
        require(fence->SetEventOnCompletion(1, nullptr), "Wait for textured draw");
        require(device->GetDeviceRemovedReason(), "Textured completion");

        void* mapped = nullptr;
        const D3D12_RANGE read_range{0, static_cast<SIZE_T>(byte_count)};
        require(readback->Map(0, &read_range, &mapped), "Map textured readback");
        for (UINT y = 0; y < side; ++y) {
            const auto* row = static_cast<const unsigned char*>(mapped) + footprint.Offset +
                              y * footprint.Footprint.RowPitch;
            std::memcpy(rgba.data() + y * side * 4, row, side * 4);
        }
        const D3D12_RANGE no_writes{0, 0};
        readback->Unmap(0, &no_writes);
        return true;
    } catch (HRESULT) {
        return false;
    }
}
