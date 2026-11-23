#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <io.h>
#include "texture.hpp"
extern "C" {
#include "blast/blast.h"
}

struct Record {
    unsigned remaining;
    unsigned size;
    unsigned char input[255];
    // Stage one full blast window and up to another half window before publication.
    unsigned char output[6144];
};

#define API extern "C" __declspec(dllexport)

API int next_length() {
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stderr), _O_BINARY);
    return std::getchar();
}

API Record* acquire(unsigned length) {
    if (length > sizeof(Record::input)) return nullptr;
    auto* record = static_cast<Record*>(std::calloc(1, sizeof(Record)));
    if (record) {
        record->remaining = length;
        std::fprintf(stderr, "alloc %u\n", length);
    }
    return record;
}

static unsigned input(void* context, unsigned char** bytes) {
    auto* record = static_cast<Record*>(context);
    unsigned n = static_cast<unsigned>(std::fread(record->input, 1, record->remaining, stdin));
    record->remaining -= n;
    *bytes = record->input;
    std::fprintf(stderr, "input %u\n", n);
    return n;
}

static int output(void* context, unsigned char* bytes, unsigned n) {
    auto* record = static_cast<Record*>(context);
    std::fprintf(stderr, "output %u\n", n);
    if (n > sizeof(record->output) - record->size) {
        std::fprintf(stderr, "refused after %u\n", record->size);
        return 1;
    }
    std::memcpy(record->output + record->size, bytes, n);
    record->size += n;
    return 0;
}

API int unpack(Record* record) {
    unsigned left = 0;
    unsigned char* unused = nullptr;
    int result = blast(input, record, output, record, &left, &unused);
    return result || left || record->remaining;
}

API int publish(Record* record) {
    return std::fwrite(record->output, 1, record->size, stdout) != record->size;
}

API int publish_texture(Record* record) {
    if (!record->size) return 1;
    ArchiveTexture texture;
    for (unsigned y = 0; y < 4; ++y) {
        for (unsigned x = 0; x < 4; ++x) {
            auto& pixel = texture.rgba[y][x];
            pixel[0] = record->output[(4 * y + x) % record->size];
            pixel[1] = static_cast<unsigned char>(85 * x);
            pixel[2] = static_cast<unsigned char>(85 * y);
            pixel[3] = 255;
        }
    }
    return consume_archive_texture(texture);
}

API void release(Record* record) {
    std::free(record);
    std::fprintf(stderr, "free\n");
}

API void completed(unsigned count) {
    finish_archive_textures();
    std::fprintf(stderr, "completed %u\n", count);
    std::fflush(stdout);
    std::fflush(stderr);
}
