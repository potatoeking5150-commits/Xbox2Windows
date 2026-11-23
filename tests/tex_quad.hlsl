struct VSIn
{
    float3 pos : POSITION;
    float2 uv : TEXCOORD;
};

struct VSOut
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD;
};

VSOut VSMain(VSIn v)
{
    VSOut o;
    o.pos = float4(v.pos, 1);
    o.uv = v.uv;
    return o;
}

Texture2D<float4> tex : register(t0);
SamplerState samp : register(s0);

float4 PSMain(VSOut v) : SV_Target
{
    return tex.Sample(samp, v.uv);
}
