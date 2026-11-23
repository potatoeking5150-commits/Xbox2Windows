#include "texture.hpp"
#include <cstdio>

int consume_archive_texture(const ArchiveTexture& texture) {
    return std::fwrite(texture.rgba, 1, sizeof(texture.rgba), stdout) != sizeof(texture.rgba);
}

void finish_archive_textures() {}
