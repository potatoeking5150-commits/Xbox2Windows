#pragma once
#include "bvh_fixture.h"

namespace dxr_fixture {
struct Ray {
    float x, y, z, nearT, dx, dy, dz, farT;
    UINT mask = 255;
    UINT flags = 0; // RAY_FLAG_* bits, optionally with report_facing.
};
static_assert(sizeof(Ray) == 40);
// Not a ray flag: asks the shader to add 2 to the hit word for a front-facing hit.
constexpr UINT report_facing = 0x80000000u;
using Vertex = std::array<float, 3>;
struct SceneUpdate {
    std::vector<Vertex> vertices;
    std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instances;
};
enum class Backend { inline_query, pipeline };
struct Capabilities {
    D3D12_RAYTRACING_TIER tier;
    D3D_SHADER_MODEL model;
    bool supports(Backend backend) const;
};
Capabilities capabilities(ID3D12Device* device);
bool supported(ID3D12Device* device, Backend backend = Backend::inline_query);
D3D12_RAYTRACING_INSTANCE_DESC instance(UINT id, UINT mask, float translateX = 0);
// The pipeline backend uses instance hit-group offset 0. With one opaque
// geometry it supports facing readback and at most one face-cull flag.
std::vector<bvh_fixture::Word> trace(ID3D12Device* device, const std::vector<Vertex>& vertices,
    std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instances, const std::vector<Ray>& rays,
    Backend backend = Backend::inline_query, const SceneUpdate* update = nullptr,
    // When set, each triangle becomes its own geometry with these flags and
    // ray_candidates.hlsl traces the opacity probes on either backend.
    const std::vector<D3D12_RAYTRACING_GEOMETRY_FLAGS>* triangleGeometries = nullptr);
}
