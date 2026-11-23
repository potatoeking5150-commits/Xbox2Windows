// Frames of one point-sampled texture quad on a long-lived D3D12 device.
//
// QuadFrames keeps one device, queue, pipeline, square RGBA8 render target and
// readback buffer for its lifetime. Each draw() optionally replaces the texture
// with a new upload, draws a quad covering the target with the tex_quad shaders
// (point filter, clamp), signals the next fence value, waits for it and returns
// the frame. Callers own the texel bytes only for the duration of the call.
//
// Nothing here throws: every operation returns the first failing HRESULT after
// printing it. C++ exceptions cannot be caught inside an image mapped by pe_map.
#pragma once
#include <d3d12.h>
#include <wrl/client.h>

#include <cstddef>

struct TextureImage {
    const unsigned char* bytes;  // Rows (block rows for BC formats), tightly packed.
    size_t size;
    UINT width;
    UINT height;
    DXGI_FORMAT format;
};

struct FrameResult {
    UINT64 fence;        // Completed fence value of this frame.
    UINT64 upload_hash;  // FNV-1a 64 of the staged texel rows; 0 when the texture was kept.
};

class QuadFrames {
public:
    // Selects the adapter through shim_gpu_device.h and creates the fixed objects.
    HRESULT create(UINT side) noexcept;

    // Uploads `image` unless it is null, then draws, waits and copies the
    // side * side RGBA8 frame into `rgba`, which must hold exactly `size` bytes.
    HRESULT draw(const TextureImage* image, unsigned char* rgba, size_t size, FrameResult& result) noexcept;

    // Adapter description and user-mode driver version: "<description>, driver a.b.c.d".
    HRESULT adapter(char* text, size_t size) const noexcept;

    ID3D12Device* device() const noexcept { return device_.Get(); }

    // Releases every object. Returns the device's remaining reference count,
    // which is 0 when nothing created from it is still alive.
    ULONG release() noexcept;

private:
    HRESULT create_pipeline() noexcept;
    HRESULT create_commands() noexcept;
    HRESULT create_target() noexcept;
    HRESULT upload(const TextureImage& image, UINT64& hash) noexcept;
    void record_draw() noexcept;
    HRESULT read_frame(unsigned char* rgba, size_t size) noexcept;

    template <class T> using Ptr = Microsoft::WRL::ComPtr<T>;
    UINT side_ = 0;
    UINT64 fence_value_ = 0;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT target_footprint_{};
    D3D12_VERTEX_BUFFER_VIEW vertex_view_{};
    Ptr<ID3D12Device> device_;
    Ptr<ID3D12RootSignature> root_;
    Ptr<ID3D12PipelineState> pipeline_;
    Ptr<ID3D12CommandQueue> queue_;
    Ptr<ID3D12CommandAllocator> allocator_;
    Ptr<ID3D12GraphicsCommandList> list_;
    Ptr<ID3D12Fence> fence_;
    Ptr<ID3D12Resource> vertices_;
    Ptr<ID3D12Resource> target_;
    Ptr<ID3D12Resource> readback_;
    Ptr<ID3D12DescriptorHeap> rtv_heap_;
    Ptr<ID3D12DescriptorHeap> srv_heap_;
    Ptr<ID3D12Resource> texture_;
    Ptr<ID3D12Resource> staging_;
};
