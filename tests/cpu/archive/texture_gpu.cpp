// D3D12 consumer for completed archive records.
//
// Each record's 4x4 RGBA8 texture is uploaded, drawn as a point-sampled quad on
// a 16x16 target and read back to stdout. One device, queue and pipeline serve
// every record; each frame waits on its own fence value. The stderr trace names
// the adapter, each upload hash and fence, and the device's remaining references
// after the release in finish_archive_textures(). Errors are returned, not
// thrown: this DLL also runs mapped by pe_map, where C++ exceptions are not caught.
//
// X2W_SHIM_DRAW_ADAPTER selects WARP (default) or hardware.
// X2W_ARCHIVE_TEXTURE_CASE selects a negative control for the second record:
//   stale  keep the first texture instead of uploading the second;
//   fail   pass a texture one byte short, so the upload is refused;
//   leak   create one more device object and never release it.
#include "texture.hpp"

#include <cstdio>
#include <cstring>
#include <new>

#include "../shim_gpu_device.h"
#include "quad_frames.hpp"

namespace {
constexpr UINT frame_side = 16;

// Heap-allocated so that nothing D3D12-owned is destroyed during DLL detach.
QuadFrames* frames = nullptr;
unsigned records = 0;

// Negative controls exist only to show that the check fails when it should.
enum class Control { none, stale, fail, leak };

Control second_record_control()
{
    char value[8]{};
    const DWORD size = GetEnvironmentVariableA("X2W_ARCHIVE_TEXTURE_CASE", value, sizeof(value));
    if (!size || size >= sizeof(value)) return Control::none;
    if (std::strcmp(value, "stale") == 0) return Control::stale;
    if (std::strcmp(value, "fail") == 0) return Control::fail;
    if (std::strcmp(value, "leak") == 0) return Control::leak;
    return Control::none;
}

// A device object that is never released keeps one device reference alive.
HRESULT leak_device_object(ID3D12Device* device)
{
    ID3D12Fence* fence = nullptr;
    return shim_report(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "Create leaked fence");
}

// Creates the renderer on the first record and names its adapter in the trace.
bool start_frames()
{
    if (frames) return true;
    frames = new (std::nothrow) QuadFrames;
    if (!frames) {
        std::fputs("texture consumer out of memory\n", stderr);
        return false;
    }
    char adapter[320]{};
    if (FAILED(frames->create(frame_side)) || FAILED(frames->adapter(adapter, sizeof(adapter)))) return false;
    std::fprintf(stderr, "gpu device %s\n", adapter);
    return true;
}
} // namespace

int consume_archive_texture(const ArchiveTexture& texture)
{
    if (!start_frames()) return 1;
    TextureImage image{&texture.rgba[0][0][0], sizeof(texture.rgba), 4, 4, DXGI_FORMAT_R8G8B8A8_UNORM};
    const Control control = ++records == 2 ? second_record_control() : Control::none;
    if (control == Control::fail) --image.size;  // Refused by the copy-footprint check.
    if (control == Control::leak && FAILED(leak_device_object(frames->device()))) return 1;
    const TextureImage* upload = control == Control::stale ? nullptr : &image;

    unsigned char rgba[frame_side * frame_side * 4];
    FrameResult frame{};
    if (FAILED(frames->draw(upload, rgba, sizeof(rgba), frame))) return 1;  // Already reported.
    if (upload) {
        std::fprintf(stderr, "upload %016llx\n", static_cast<unsigned long long>(frame.upload_hash));
    } else {
        std::fputs("upload skipped\n", stderr);
    }
    std::fprintf(stderr, "fence %llu\n", static_cast<unsigned long long>(frame.fence));
    return std::fwrite(rgba, 1, sizeof(rgba), stdout) != sizeof(rgba);
}

void finish_archive_textures()
{
    if (!frames) return;
    const ULONG references = frames->release();
    delete frames;
    frames = nullptr;
    std::fprintf(stderr, "gpu release %lu\n", references);
}
