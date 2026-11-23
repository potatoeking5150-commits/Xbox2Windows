RaytracingAccelerationStructure scene : register(t0);
RWStructuredBuffer<uint4> results : register(u0);
struct Query { float4 originMin; float4 directionMax; uint mask; uint flags; };
StructuredBuffer<Query> queries : register(t1);

RayDesc make_ray(Query query)
{
    RayDesc ray;
    ray.Origin = query.originMin.xyz;
    ray.Direction = query.directionMax.xyz;
    ray.TMin = query.originMin.w;
    ray.TMax = query.directionMax.w;
    return ray;
}

// Word 0: bit 0 hit, 1 candidate after an automatic commit (inline only),
// 2 anomaly, 3 candidate after a shader commit, 4 shader committed,
// 8-11 presented geometries, 12-15 committed geometry, 16-23 committed primitive.
// Word 2 holds the distance of the latest hit accepted by the shader.

#ifdef DXR_PIPELINE
// GeometryIndex is a Tier 1.1 intrinsic. Tier 1.0 takes the geometry index
// from a local root constant in the hit group record selected by TraceRay.
cbuffer HitRecord : register(b0) { uint geometry; }
struct Payload { uint4 value; };

[shader("raygeneration")]
void RayGen()
{
    uint i = DispatchRaysIndex().x;
    Query query = queries[i];
    Payload payload;
    payload.value = uint4(0x80000000u, 0, 0, 0); // Miss or ClosestHit must clear the sentinel.
    TraceRay(scene, query.flags & 0x3ff, query.mask, 0, 1, 0, make_ray(query), payload);
    results[i] = payload.value;
}

[shader("anyhit")]
void AnyHit(inout Payload payload, BuiltInTriangleIntersectionAttributes attributes)
{
    Query query = queries[DispatchRaysIndex().x];
    if (payload.value.x & 16u) payload.value.x |= 8u;
    float t = RayTCurrent();
    if (geometry >= 4 || PrimitiveIndex() != 0 || !(t > query.originMin.w && t < query.directionMax.w)) {
        payload.value.x |= 4u;
        IgnoreHit();
        return;
    }
    payload.value.x |= 1u << (8 + geometry);
    if (((query.flags >> (16 + geometry)) & 1u) == 0) {
        IgnoreHit();
        return;
    }
    payload.value.x |= 16u;
    payload.value.z = asuint(t);
}

[shader("miss")]
void Miss(inout Payload payload)
{
    payload.value.x &= ~0x80000000u;
}

[shader("closesthit")]
void ClosestHit(inout Payload payload, BuiltInTriangleIntersectionAttributes attributes)
{
    payload.value = uint4((payload.value.x & ~0x80000000u) | 1u | (min(geometry, 15u) << 12) |
                          (min(PrimitiveIndex(), 255u) << 16), InstanceID(), payload.value.z, asuint(RayTCurrent()));
}
#else
// Traces without template flags so geometry, instance and dynamic ray flags
// alone decide opacity, and handles every non-opaque candidate in the shader.
[RootSignature("SRV(t0),UAV(u0),SRV(t1)")]
[numthreads(1, 1, 1)]
void CSMain(uint3 thread : SV_DispatchThreadID)
{
    uint i = thread.x;
    Query query = queries[i];
    RayDesc ray = make_ray(query);
    // Bits 16-19 are a fixture convention: geometries whose non-opaque
    // candidates this shader commits. The low bits are ray flags.
    uint accept = (query.flags >> 16) & 0xf;
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(scene, query.flags & 0x3ff, query.mask, ray);
    uint seen = 0, afterCommit = 0, afterShaderCommit = 0, anomaly = 0, shaderCommitted = 0;
    float shaderCommitT = 0; // CandidateTriangleRayT of the latest candidate this shader committed.
    while (q.Proceed()) {
        // Legal while searching for the closest hit, but a first-hit search
        // must end at its first commit. Bit 3 marks a candidate after this
        // shader committed; bit 1 one after an automatic opaque commit only.
        if (shaderCommitted) afterShaderCommit = 1;
        else if (q.CommittedStatus() != COMMITTED_NOTHING) afterCommit = 1;
        if (q.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) { anomaly = 1; continue; }
        // A candidate must lie inside the ray extents and not behind any
        // committed hit. An implementation may present the committed triangle
        // or one at the same distance again, so equality is allowed.
        float t = q.CandidateTriangleRayT();
        float limit = q.CommittedStatus() == COMMITTED_NOTHING ? ray.TMax : q.CommittedRayT();
        uint geometry = q.CandidateGeometryIndex();
        if (!(t > ray.TMin && t <= limit) || q.CandidatePrimitiveIndex() != 0 || geometry >= 4) { anomaly = 1; continue; }
        seen |= 1u << geometry;
        if ((accept >> geometry) & 1) { q.CommitNonOpaqueTriangleHit(); shaderCommitted = 1; shaderCommitT = t; }
    }
    uint4 value = uint4((afterCommit << 1) | (anomaly << 2) | (afterShaderCommit << 3) | (shaderCommitted << 4) |
                        (seen << 8), 0, asuint(shaderCommitT), 0);
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        value = uint4(value.x | 1 | (min(q.CommittedGeometryIndex(), 15u) << 12) | (min(q.CommittedPrimitiveIndex(), 255u) << 16),
                      q.CommittedInstanceID(), value.z, asuint(q.CommittedRayT()));
    results[i] = value;
}
#endif
