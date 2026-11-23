// The second provider of two_slots.asm. fold4 records its first, second and fourth integer argument in three
// words at output, as the imports of tail_provider.cpp do, and returns a 16-bit value that depends on all of them;
// the higher bytes of RAX are zero. ALTER_RETURN changes only the returned value.
#include <cstdint>

using Word = std::uint64_t;

extern "C" Word fold4(Word first, Word second, Word* output, Word fourth) {
    output[0] = first;
    output[1] = second;
    output[2] = fourth;
    Word value = ((first ^ second) + 0x2d) * 5 + (fourth >> 7);
#ifdef ALTER_RETURN
    value += 1;
#endif
    return value & 0xffff;
}
