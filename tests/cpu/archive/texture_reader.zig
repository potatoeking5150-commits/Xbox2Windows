const reader = @import("reader.zig");

extern "archive_codec" fn publish_texture(record: *anyopaque) callconv(.C) c_int;

pub fn main() u8 {
    return reader.readRecords(publish_texture);
}
