#include "desktop_gpu.h"
#include <cstdio>

using Microsoft::WRL::ComPtr;

#if defined(OFFSCREEN_AGILITY_620)
extern "C" const UINT D3D12SDKVersion;
#endif

void check(HRESULT hr, const char* stage)
{
    if (FAILED(hr)) {
        std::fprintf(stderr, "%s failed (HRESULT 0x%08lx)\n", stage, static_cast<unsigned long>(hr));
        throw std::runtime_error(stage);
    }
}

HRESULT check_feature(ID3D12Device* device, D3D12_FEATURE feature, void* data, UINT size)
{
#if defined(OFFSCREEN_AGILITY_620) && defined(__ID3D12Device16_INTERFACE_DEFINED__)
    ComPtr<ID3D12Device16> versioned;
    if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&versioned))))
        return versioned->CheckFeatureSupport1(feature, data, size, D3D12SDKVersion);
#endif
    return device->CheckFeatureSupport(feature, data, size);
}

ComPtr<ID3D12Resource> resource(ID3D12Device* device, const D3D12_RESOURCE_DESC& desc,
                               D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state,
                               const D3D12_CLEAR_VALUE* clear)
{
    D3D12_HEAP_PROPERTIES props{};
    props.Type = heap;
    props.CreationNodeMask = props.VisibleNodeMask = 1;
    ComPtr<ID3D12Resource> result;
    check(device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, state, clear,
                                          IID_PPV_ARGS(&result)), "Create resource");
    return result;
}

ComPtr<ID3D12Resource> make_buffer(ID3D12Device* device, UINT64 length, D3D12_HEAP_TYPE heap,
                                   D3D12_RESOURCE_STATES state, D3D12_RESOURCE_FLAGS flags)
{
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = length;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return resource(device, desc, heap, state);
}

void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* buffer,
                D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {buffer, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    list->ResourceBarrier(1, &barrier);
}

ComputeCommands::ComputeCommands(ID3D12Device* device, ID3D12PipelineState* pipeline, D3D12_COMMAND_LIST_TYPE type)
{
    D3D12_COMMAND_QUEUE_DESC desc{};
    desc.Type = type;
    check(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue)), "Create compute queue");
    check(device->CreateCommandAllocator(desc.Type, IID_PPV_ARGS(&allocator)), "Create compute allocator");
    check(device->CreateCommandList(0, desc.Type, allocator.Get(), pipeline, IID_PPV_ARGS(&list)),
          "Create compute command list");
}

void ComputeCommands::submit_and_wait(ID3D12Device* device)
{
    check(list->Close(), "Close compute command list");
    ComPtr<ID3D12Fence> fence;
    check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "Create compute fence");
    ID3D12CommandList* submitted[]{list.Get()};
    queue->ExecuteCommandLists(1, submitted);
    check(queue->Signal(fence.Get(), 1), "Signal compute fence");
    check(fence->SetEventOnCompletion(1, nullptr), "Wait for compute");
    check(device->GetDeviceRemovedReason(), "Compute completion");
}
