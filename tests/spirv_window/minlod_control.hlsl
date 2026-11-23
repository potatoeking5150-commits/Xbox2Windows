// Direct desktop control for minlod_levels.py: the translated sample written in HLSL.
Texture2D<float4> color : register(t0);
SamplerState pointClamp : register(s0);

float4 main() : SV_Target
{
    return color.Sample(pointClamp, float2(0.25, 0.25), int2(0, 0), CLAMP);
}
