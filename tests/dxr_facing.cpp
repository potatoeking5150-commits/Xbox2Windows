#include "dxr_fixture.h"
#include "dxr_facing_reference.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

static_assert(dxr_facing::cullBack == D3D12_RAY_FLAG_CULL_BACK_FACING_TRIANGLES);
static_assert(dxr_facing::cullFront == D3D12_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES);

namespace {
void check_shared_edge(ID3D12Device* device, dxr_fixture::Backend backend, bool supported) {
    const std::vector<dxr_facing::Vertex> mesh{
        {0,0,1}, {0,1,1}, {1,0,1}, {1,0,1}, {0,1,1}, {1,1,1}};
    const auto identity = dxr_facing::instances().front();
    const std::vector<dxr_facing::Instance> scene{identity};
    // Bit 0 and bit 1 name the incident primitives. On the shared diagonal DXR
    // may commit either one. Just off it there is exactly one possible owner.
    struct Probe { float x, y; UINT allowed; };
    constexpr Probe probes[]{
        {.25f,.75f,3}, {.5f,.5f,3}, {.75f,.25f,3},
        {.5f,.4375f,1}, {.5f,.5625f,2}, {1.25f,.5f,0}};
    std::vector<dxr_fixture::Ray> rays;
    for (float dz : {1.f, -1.f}) {
        const float z = 1 - dz;
        for (const auto& probe : probes) {
            const dxr_facing::Ray ray{{probe.x, probe.y, z}, {0,0,dz}, 0, 2, 0};
            const auto hit = dxr_facing::closest(mesh, scene, ray);
            if (hit.hit != (probe.allowed != 0) ||
                (hit.hit && (!(probe.allowed & (1u << hit.primitive)) ||
                             hit.distance != 1 || hit.front != (dz > 0)))) {
                std::fprintf(stderr, "CPU shared edge x=%g y=%g dz=%g allowed=%u hit=%d primitive=%u\n",
                             probe.x, probe.y, dz, probe.allowed, hit.hit, hit.primitive);
                throw std::runtime_error("CPU shared-edge intersection differs");
            }
            rays.push_back({probe.x, probe.y, z, 0, 0,0,dz, 2, 255, dxr_fixture::report_facing});
        }
    }
    if (!supported) return;
    const auto words = dxr_fixture::trace(device, mesh, {dxr_fixture::instance(identity.id, 255)}, rays, backend);
    UINT edgeOwners[2]{}, interiorHits = 0, misses = 0;
    for (std::size_t i = 0; i < rays.size(); ++i) {
        const UINT allowed = probes[i % std::size(probes)].allowed;
        const UINT* result = words.data() + 4 * i;
        const UINT status = allowed ? (rays[i].dz > 0 ? 3u : 1u) : 0;
        const bool primitiveMatches = allowed ? result[2] < 2 && (allowed & (1u << result[2])) : result[2] == 0;
        float distance;
        std::memcpy(&distance, result + 3, sizeof(distance));
        const bool distanceMatches = allowed ? std::abs(distance - 1.f) <= 1e-5f : result[3] == 0;
        if (result[0] != status || result[1] != (allowed ? identity.id : 0) ||
            !primitiveMatches || !distanceMatches) {
            std::fprintf(stderr, "DXR shared edge ray=%zu allowed=%u status=%u instance=%u primitive=%u t=%g\n",
                         i, allowed, result[0], result[1], result[2], distance);
            throw std::runtime_error("DXR shared-edge intersection differs");
        }
        if (allowed == 3) ++edgeOwners[result[2]];
        else if (allowed) ++interiorHits;
        else ++misses;
    }
    std::printf("dxr_shared_edge rays=%zu primitive0_edge_hits=%u primitive1_edge_hits=%u interior_hits=%u misses=%u\n",
                rays.size(), edgeOwners[0], edgeOwners[1], interiorHits, misses);
}
}

bool check_dxr_facing(ID3D12Device* device, bool usePipeline) {
    // The geometric reference must reproduce the hand-derived table on every
    // machine, including those that skip the GPU comparison.
    const auto scene = dxr_facing::instances();
    const auto probes = dxr_facing::rays();
    const auto mesh = dxr_facing::vertices();
    // Instance 8 lies at world z=1.5. Its hit at t=2 must fall outside the
    // reference's open interval. Keep this exact endpoint check on the CPU.
    const dxr_facing::Ray clipped{{64.25f, .25f, -.5f}, {0,0,1}, 0, 2, 0};
    if (const auto hit = dxr_facing::closest(mesh, scene, clipped); hit.hit) {
        std::fprintf(stderr, "Scaled CPU ray: expected miss at farT=2, instance=%u t=%.17g\n",
                     hit.instance, hit.distance);
        throw std::runtime_error("Scaled ray entered the open interval");
    }
    if (dxr_facing::first_reference_mismatch() != probes.size())
        throw std::runtime_error("Facing reference disagrees with its hand-derived table");
    const auto backend = usePipeline ? dxr_fixture::Backend::pipeline : dxr_fixture::Backend::inline_query;
    const bool supported = dxr_fixture::supported(device, backend);
    check_shared_edge(device, backend, supported);
    if (!supported) return false;
    std::vector<dxr_fixture::Vertex> vertices(mesh.begin(), mesh.end());
    std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instances;
    for (const auto& source : scene) {
        D3D12_RAYTRACING_INSTANCE_DESC desc{};
        for (UINT axis = 0; axis < 3; ++axis) desc.Transform[axis][axis] = source.scale[axis];
        desc.Transform[0][3] = source.translateX;
        desc.InstanceID = source.id;
        desc.InstanceMask = 255;
        desc.Flags = (source.counterclockwise ? D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_FRONT_COUNTERCLOCKWISE : 0u) |
                     (source.cullDisable ? D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE : 0u);
        instances.push_back(desc);
    }
    std::vector<dxr_fixture::Ray> rays;
    for (const auto& probe : probes)
        rays.push_back({probe.origin[0], probe.origin[1], probe.origin[2], probe.nearT,
                        probe.direction[0], probe.direction[1], probe.direction[2], probe.farT,
                        255, probe.flags | dxr_fixture::report_facing});
    const auto words = dxr_fixture::trace(device, vertices, instances, rays, backend);
    // Reuse the unculled ray for each primitive and direction to predict the
    // readback of a backend that silently dropped both face-cull flags.
    UINT missingCullMismatches = 0;
    UINT hits = 0, frontHits = 0, mismatches = 0;
    for (UINT i = 0; i < rays.size(); ++i) {
        const auto expected = dxr_facing::closest(mesh, scene, probes[i]);
        const UINT* value = words.data() + 4 * i;
        float distance;
        std::memcpy(&distance, value + 3, sizeof(distance));
        // Status, instance and primitive are exact. A hit distance may differ
        // from the reference by 1e-5 relative; a miss leaves all four words zero.
        const UINT status = expected.hit ? (expected.front ? 3u : 1u) : 0u;
        const bool distanceMatches = expected.hit
            ? std::abs(distance - expected.distance) <= 1e-5 * expected.distance : value[3] == 0;
        if (value[0] != status || value[1] != expected.instance || value[2] != expected.primitive || !distanceMatches) {
            if (!mismatches)
                std::fprintf(stderr, "ray=%u flags=0x%x expected status=%u instance=%u primitive=%u distance=%g "
                             "actual status=%u instance=%u primitive=%u distance=%g\n", i, probes[i].flags, status,
                             expected.instance, expected.primitive, expected.distance, value[0], value[1], value[2], distance);
            ++mismatches;
        }
        hits += expected.hit;
        frontHits += expected.hit && expected.front;
        const UINT* unculled = words.data() + 4 * (i - i % std::size(dxr_facing::rayFlags));
        missingCullMismatches += unculled[0] != status;
    }
    std::printf("dxr_facing rays=%zu hits=%u front_hits=%u culled=%zu mismatches=%u\n",
                rays.size(), hits, frontHits, rays.size() - hits, mismatches);
    if (mismatches) throw std::runtime_error("DXR facing or culling disagrees with the CPU reference");
    if (hits != 72 || frontHits != 36) throw std::runtime_error("Facing fixture lost its expected coverage");
    std::printf("dxr_facing no_cull_control_status_mismatches=%u\n", missingCullMismatches);
    if (missingCullMismatches != 24) throw std::runtime_error("Missing-cull control lost its expected coverage");
    return true;
}
