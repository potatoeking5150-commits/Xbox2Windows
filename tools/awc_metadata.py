"""Read format metadata from an explicit little-endian ADAT or big-endian TADA sound-effect or music bank."""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import struct
from typing import BinaryIO, Optional

from paged_resource import PagedResource, Segment


@dataclass
class FormatGroup:
    sample_rate_hz: int
    codec_id: int
    streams: int = 0
    samples: int = 0


# Sound format tags: minimum length (bytes read) and codec byte offset, as in the public reader's 0x76 and 0xFA cases.
FORMAT_TAGS = {0x76: (32, 0x1c), 0xFA: (0x14, 0x13)}


@dataclass
class MusicHeader:
    channels: int
    block_count: int
    block_chunk: int


@dataclass
class Bank:
    layout: str
    stream_count: int
    formats: list[FormatGroup]
    music: Optional[MusicHeader]


BYTE_ORDERS = {b"ADAT": "<", b"TADA": ">"}  # the public reader switches its 16-, 32- and 64-bit header reads


def _header(bank: PagedResource, size: int) -> tuple[int, int, int, list[int], str]:
    order = BYTE_ORDERS.get(bank.read(0, 4))
    flags, count, header_size = struct.unpack((order or "<") + "III", bank.read(4, 12))
    if order is None or flags not in (0xff000001, 0xff020001):
        raise ValueError("Expected an ADAT or TADA v1 bank without indexing, streaming or encryption flags")
    if not 1 <= count <= 4096:
        raise ValueError("AWC stream count outside the 1 to 4096 inventory limit")
    table = 16 + 4 * count
    if not table <= header_size <= size:
        raise ValueError("AWC stream table outside declared header")
    entries = [word for word, in struct.iter_unpack(order + "I", bank.read(16, 4 * count))]
    return count, table, header_size, entries, order


def read_formats(stream: BinaryIO, offset: int, size: int) -> list[FormatGroup]:
    bank = PagedResource(stream, [Segment(offset, size, size, "raw")])
    count, table, header_size, entries, order = _header(bank, size)
    if entries[0] & 0x1fffffff == 0:
        raise ValueError("Streamed AWC banks are unsupported")
    table_bytes = 8 * sum(word >> 29 for word in entries)
    if table + table_bytes > header_size:
        raise ValueError("AWC tag table outside declared header")
    tags = struct.iter_unpack(order + "Q", bank.read(table, table_bytes))
    formats: list[tuple[int, int]] = []
    spans: list[tuple[int, int]] = []
    for entry in entries:
        data_tags = 0
        format_offsets = []
        for _ in range(entry >> 29):
            word, = next(tags)
            kind = word >> 56
            length = (word >> 28) & 0x0fffffff
            start = word & 0x0fffffff
            if not header_size <= start <= start + length <= size:
                raise ValueError("AWC tag outside supplied bank extent")
            if length:
                spans.append((start, start + length))
            if kind == 0x55 and length:
                data_tags += 1
            elif kind in FORMAT_TAGS:
                if length < FORMAT_TAGS[kind][0]:
                    raise ValueError("Truncated AWC sound format tag")
                format_offsets.append((start, kind))
        if data_tags != 1 or len(format_offsets) != 1:
            raise ValueError("Expected one data tag and one 0x76 or 0xFA format tag per sound")
        formats.append(format_offsets[0])

    # Establish that every format range is separate from data before reading it.
    spans.sort()
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError("Overlapping AWC tag extents")
    groups: dict[tuple[int, int], FormatGroup] = {}
    for start, kind in formats:
        length, codec_offset = FORMAT_TAGS[kind]
        header = bank.read(start, length)
        samples, = struct.unpack_from(order + "I", header)
        rate, = struct.unpack_from(order + "H", header, 8)
        codec = header[codec_offset]
        key = rate, codec
        if key not in groups:
            groups[key] = FormatGroup(rate, codec)
        groups[key].streams += 1
        groups[key].samples += samples
    return [groups[key] for key in sorted(groups)]


def _music(bank: PagedResource, size: int, table: int, header_size: int, entries: list[int], order: str) -> Bank:
    # Only stream 0's tags are used, as in the reference; tags of streams 1.. are never read.
    if len(entries) > 32:
        raise ValueError("Music AWC bank has more than 32 streams")
    if any(entry & 0x1fffffff == 0 for entry in entries[1:]):
        raise ValueError("Duplicate id-0 AWC stream")
    table_bytes = 8 * (entries[0] >> 29)
    if table + table_bytes > header_size:
        raise ValueError("AWC tag table outside declared header")
    headers: list[int] = []
    data_tags = 0
    spans: list[tuple[int, int]] = []
    for word, in struct.iter_unpack(order + "Q", bank.read(table, table_bytes)):
        kind = word >> 56
        length = (word >> 28) & 0x0fffffff
        start = word & 0x0fffffff
        if not header_size <= start <= start + length <= size:
            raise ValueError("AWC tag outside supplied bank extent")
        if length:
            spans.append((start, start + length))
        if kind == 0x48:
            headers.append(start)
            header_length = length
        elif kind == 0x55 and length:
            data_tags += 1
    if len(headers) != 1 or data_tags != 1:
        raise ValueError("Expected one 0x48 music header and one data tag on the id-0 stream")
    spans.sort()
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError("Overlapping AWC tag extents")
    if header_length < 12:
        raise ValueError("AWC music header shorter than 12 bytes")
    block_count, block_chunk, channels = struct.unpack(order + "III", bank.read(headers[0], 12))
    if not 1 <= channels <= 32:
        raise ValueError("Music channel count outside 1 to 32")
    if header_length < 12 + 16 * channels:
        raise ValueError("Truncated AWC music header")
    if block_count < 1 or block_chunk < 1:
        raise ValueError("Music block count or chunk size is zero")
    records = bank.read(headers[0] + 12, 16 * channels)
    if len({records[16 * channel + 12] for channel in range(channels)}) != 1:
        raise ValueError("Mixed codecs across music channels")
    groups: dict[tuple[int, int], FormatGroup] = {}
    for channel in range(channels):
        samples, = struct.unpack_from(order + "I", records, 16 * channel + 4)
        rate, = struct.unpack_from(order + "H", records, 16 * channel + 10)
        key = rate, records[16 * channel + 12]
        if key not in groups:
            groups[key] = FormatGroup(*key)
        groups[key].streams += 1
        groups[key].samples = max(groups[key].samples, samples)
    return Bank("music", len(entries), [groups[key] for key in sorted(groups)],
                MusicHeader(channels, block_count, block_chunk))


def read_bank(stream: BinaryIO, offset: int, size: int) -> Bank:
    bank = PagedResource(stream, [Segment(offset, size, size, "raw")])
    count, table, header_size, entries, order = _header(bank, size)
    if entries[0] & 0x1fffffff:
        return Bank("sfx", len(entries), read_formats(stream, offset, size), None)
    return _music(bank, size, table, header_size, entries, order)


def integer(text: str) -> int:
    return int(text, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--offset", type=integer, required=True, help="Byte offset of the ADAT or TADA header")
    parser.add_argument("--size", type=integer, required=True, help="Bytes in the caller-supplied bank range")
    args = parser.parse_args()
    try:
        with args.archive.open("rb") as stream:
            bank = read_bank(stream, args.offset, args.size)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{args.archive}: {error}\n")
    result: dict = {"stream_count": bank.stream_count}
    if bank.music:
        result |= {"layout": bank.layout, "music": asdict(bank.music)}
    print(json.dumps(result | {"formats": [asdict(g) for g in bank.formats]}))


if __name__ == "__main__":
    main()
