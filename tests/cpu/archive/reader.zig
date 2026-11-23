extern "archive_codec" fn next_length() callconv(.C) c_int;
extern "archive_codec" fn acquire(length: c_uint) callconv(.C) ?*anyopaque;
extern "archive_codec" fn unpack(record: *anyopaque) callconv(.C) c_int;
extern "archive_codec" fn publish(record: *anyopaque) callconv(.C) c_int;
extern "archive_codec" fn release(record: *anyopaque) callconv(.C) void;
extern "archive_codec" fn completed(count: c_uint) callconv(.C) void;

pub fn main() u8 {
    return readRecords(publish);
}

pub fn readRecords(comptime publishRecord: anytype) u8 {
    var count: u32 = 0;
    var status: u8 = 0;
    while (true) {
        const length = next_length();
        if (length < 0) break;
        const record = acquire(@intCast(length)) orelse {
            status = 3;
            break;
        };
        defer release(record);
        if (unpack(record) != 0) {
            status = 2;
            break;
        }
        if (publishRecord(record) != 0) {
            status = 4;
            break;
        }
        count +%= 1;
    }
    completed(count);
    return status;
}
