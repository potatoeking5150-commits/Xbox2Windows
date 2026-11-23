// Source-built desktop HLSL control for the subgroup compaction fixture.
//
// It writes the same output contract as tests/spirv_window/compaction.spvasm so compaction.py can run
// the direct-HLSL path and the SPIR-V-to-HLSL bridge under identical inputs and compare every word:
//   [0, 64)    lane index observed by each group thread
//   [64]       subgroup width
//   [65, 129)  four ballot words per wave
//   [129, 145) first active lane per wave
//   [145, 161) active lane count per wave
//   [161, 177) last active lane per wave
//   [177, 193) compaction rank of the selected (last active) lane per wave
//   [193, 257) per-lane compaction rank
//   [257, 321) compacted payloads in wave-disjoint ranges keyed by rank
//   [321, 328) untouched guards
//
// MASK_LO and MASK_HI are supplied with -D so both paths consume the same deterministic masks.
// The single-high-bit-per-4-lane nibble design keeps ballot word 0 nonzero for every wave at widths
// 4, 8, 16, 32 and 64, which the first-active computation relies on.
RWByteAddressBuffer output : register(u0);

#ifndef MASK_LO
#define MASK_LO 0x00000000u
#endif
#ifndef MASK_HI
#define MASK_HI 0x00000000u
#endif

static const uint lane_word = 0;
static const uint width_word = 64;
static const uint ballot_words = 65;
static const uint first_word = 129;
static const uint count_word = 145;
static const uint last_word = 161;
static const uint selected_word = 177;
static const uint rank_word = 193;
static const uint compact_word = 257;

[numthreads(64, 1, 1)]
void main(uint local : SV_GroupIndex)
{
    const uint lane = WaveGetLaneIndex();
    const uint width = WaveGetLaneCount();
    const uint wave = local / width;

    const uint index = local;
    const uint source = (index >> 5u) != 0u ? MASK_HI : MASK_LO;
    const bool active = (((source >> (index & 31u)) & 0x1u) != 0u);

    const uint4 ballot = WaveActiveBallot(active);
    const uint b0 = ballot.x, b1 = ballot.y, b2 = ballot.z, b3 = ballot.w;
    const uint count = countbits(b0) + countbits(b1) + countbits(b2) + countbits(b3);
    const uint first = countbits((b0 & (0u - b0)) - 1u);
    const uint last = b1 != 0u ? 32u + uint(firstbithigh(b1)) : uint(firstbithigh(b0));

    const uint lane_half = lane >> 5u;
    const uint below = (1u << (lane & 31u)) - 1u;
    const uint rank = countbits(b0 & (lane_half == 0u ? below : 0xffffffffu))
                    + countbits(b1 & (lane_half == 0u ? 0u : below));

    output.Store(lane_word * 4 + local * 4, lane);
    if (local == 0u)
        output.Store(width_word * 4, width);
    if (lane == 0u) {
        output.Store((ballot_words + wave * 4 + 0u) * 4, b0);
        output.Store((ballot_words + wave * 4 + 1u) * 4, b1);
        output.Store((ballot_words + wave * 4 + 2u) * 4, b2);
        output.Store((ballot_words + wave * 4 + 3u) * 4, b3);
        output.Store((first_word + wave) * 4, first);
        output.Store((count_word + wave) * 4, count);
        output.Store((last_word + wave) * 4, last);
        output.Store((selected_word + wave) * 4, count - 1u);
    }
    output.Store((rank_word + local) * 4, rank);
    if (active)
        output.Store((compact_word + wave * width + rank) * 4, 0x80000000u | local);
}
