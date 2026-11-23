struct Vertex { float4 uv : TEXCOORD0; float4 position : SV_Position; };
// PIXEL_MIP_LINEAR builds the variant that blends neighboring mips, for MinLod filter checks.
#ifdef PIXEL_MIP_LINEAR
#define PIXEL_FILTER "FILTER_MIN_MAG_POINT_MIP_LINEAR"
#else
#define PIXEL_FILTER "FILTER_MIN_MAG_MIP_POINT"
#endif
[RootSignature("DescriptorTable(SRV(t0,numDescriptors=2),visibility=SHADER_VISIBILITY_PIXEL),CBV(b0,visibility=SHADER_VISIBILITY_PIXEL),"
               "CBV(b1,visibility=SHADER_VISIBILITY_PIXEL),CBV(b2,visibility=SHADER_VISIBILITY_PIXEL),StaticSampler(s0,filter=" PIXEL_FILTER ","
               "addressU=TEXTURE_ADDRESS_CLAMP,addressV=TEXTURE_ADDRESS_CLAMP,addressW=TEXTURE_ADDRESS_CLAMP)")]
Vertex main(uint id : SV_VertexID)
{
    Vertex v;
    const bool partial = id >= 3;
    const uint corner = partial ? id - 3 : id;
    float2 xy = float2((corner << 1) & 2, corner & 2) * (partial ? 7.75f / 16.0f : 1.0f);
    v.uv = float4(xy, 0.25f + 0.5f * xy.x, 1.0f - xy.y);
    v.position = float4(xy * float2(2,-2) + float2(-1,1), 0, 1);
    return v;
}
