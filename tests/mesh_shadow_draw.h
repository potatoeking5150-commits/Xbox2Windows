#pragma once

#include <d3d12.h>
#include <memory>
#include <vector>

// The common source-built scene used by the standalone and EXE draw checks.
class MeshShadowDraw {
public:
    MeshShadowDraw(ID3D12Device* device, const char* mesh, const char* receiver, const char* caster,
                   const char* fixture, bool two_frames);
    ~MeshShadowDraw();

    bool supported() const;
    void record(UINT frame, UINT shadow_map);
    UINT64 submit(UINT frame);
    UINT64 wait(std::vector<unsigned char>& pixels);
    UINT release_surfaces();
    void compare_reference(const std::vector<unsigned char>& pixels, bool stale) const;

private:
    struct Scene;
    std::unique_ptr<Scene> scene_;
};
