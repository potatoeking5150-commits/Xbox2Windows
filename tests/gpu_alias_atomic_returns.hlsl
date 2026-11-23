#define ProbeRoot "DescriptorTable(UAV(u0, space = 1, numDescriptors = 8)), RootConstants(num32BitConstants = 2, b0)"

RWByteAddressBuffer Aliases[8] : register(u0, space1);

cbuffer Counts : register(b0)
{
    uint rangeCount;
    uint threadCount;
};

[RootSignature(ProbeRoot)]
[numthreads(64, 1, 1)]
void ProbeArrayAtomicsWaterfallReturns(uint id : SV_DispatchThreadID)
{
    if (id >= threadCount) return;

    uint selected = id % rangeCount;
    bool remaining = true;

    while (WaveActiveAnyTrue(remaining))
    {
        uint leaderLane = WaveActiveMin(
            remaining ? WaveGetLaneIndex() : 0xffffffff
        );

        uint groupIndex = WaveReadLaneAt(selected, leaderLane);
        bool member = remaining && selected == groupIndex;

        uint groupCount = WaveActiveCountBits(member);
        uint rank = WavePrefixCountBits(member);

        uint previous = 0;

        if (member && WaveGetLaneIndex() == leaderLane)
        {
            Aliases[NonUniformResourceIndex(groupIndex)].InterlockedAdd(
                0, groupCount, previous
            );
        }

        uint base = WaveReadLaneAt(previous, leaderLane);

        if (member)
            Aliases[4].Store(id * 4, base + rank);

        if (member)
            remaining = false;
    }
}
