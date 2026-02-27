const std = @import("std");

pub const WireType = enum(u3) {
    Varint = 0,
    Fixed64 = 1,
    LengthDelimited = 2,
    StartGroup = 3,
    EndGroup = 4,
    Fixed32 = 5,
    Unknown = 6,
    Reserved = 7,
};

pub const Tag = struct {
    wire_type: WireType,
    field_number: u64,
};

pub const Error = error{
    EndOfBuffer,
    InvalidVarint,
    InvalidWireType,
    InvalidFieldNumber,
    InvalidUtf8,
    InvalidTag,
    UnsupportedWireType,
    LengthTooLong,
    UnexpectedEOF,
};

pub const ProtoReader = struct {
    allocator: std.mem.Allocator,
    buffer: []const u8,
    pos: usize,

    pub fn init(allocator: std.mem.Allocator, buffer: []const u8) ProtoReader {
        return .{
            .allocator = allocator,
            .buffer = buffer,
            .pos = 0,
        };
    }

    pub fn deinit(self: *ProtoReader) void {
        // ProtoReader doesn't own the buffer, so nothing to free
        _ = self;
    }

    pub fn readTag(self: *ProtoReader) !Tag {
        const raw_tag = try self.readVarint();
        const wire_type = @as(WireType, @enumFromInt(@as(u3, @intCast(raw_tag & 0x7))));
        const field_number = @as(u64, @intCast(raw_tag >> 3));
        return Tag{
            .wire_type = wire_type,
            .field_number = field_number,
        };
    }

    pub fn readVarint(self: *ProtoReader) !u64 {
        var result: u64 = 0;
        var shift: u6 = 0;

        while (true) {
            if (self.pos >= self.buffer.len) {
                return Error.EndOfBuffer;
            }

            const byte = self.buffer[self.pos];
            self.pos += 1;

            result |= @as(u64, @intCast(byte & 0x7F)) << shift;
            if (byte & 0x80 == 0) break;
            shift += 7;

            if (shift >= 64) {
                return Error.InvalidVarint;
            }
        }

        return result;
    }

    // Reads 32 bit one after the other, use @bitCast() to change type, pay attention sizeOf(DEST)=sizeOf(SOURCE)
    pub fn readFixed32(self: *ProtoReader) !u32 {
        if (self.pos + 4 > self.buffer.len) {
            return Error.EndOfBuffer;
        }
        const bytes = self.buffer[self.pos..][0..4];
        self.pos += 4;
        return std.mem.bytesToValue(u32, bytes);
    }

    // Reads 64 bit one after the other, use @bitCast() to change type, pay attention sizeOf(DEST)=sizeOf(SOURCE)
    pub fn readFixed64(self: *ProtoReader) !u64 {
        if (self.pos + 8 > self.buffer.len) {
            return Error.EndOfBuffer;
        }
        const bytes = self.buffer[self.pos..][0..8];
        self.pos += 8;
        return std.mem.bytesToValue(u64, bytes[0..8]);
    }

    pub fn available(self: *ProtoReader) usize {
        return self.buffer.len - self.pos;
    }

    pub fn readLengthDelimited(self: *ProtoReader) !ProtoReader {
        const len = try self.readVarint();
        if (len > std.math.maxInt(usize)) {
            return error.LengthTooLong;
        }
        const size: usize = @intCast(len);

        if (size == 0) {
            return ProtoReader{
                .buffer = self.buffer[self.pos..self.pos],
                .pos = 0,
                .allocator = self.allocator,
            };
        }

        if (self.pos + size > self.buffer.len) {
            return error.EndOfBuffer;
        }

        const result = ProtoReader{
            .buffer = self.buffer[self.pos .. self.pos + size],
            .pos = 0,
            .allocator = self.allocator,
        };
        self.pos += size;
        return result;
    }

    pub fn readString(self: *ProtoReader, allocator: std.mem.Allocator) ![]const u8 {
        const len = try self.readVarint();
        if (len > std.math.maxInt(usize)) {
            return error.LengthTooLong;
        }
        const size: usize = @intCast(len);
        if (self.pos + size > self.buffer.len) {
            return error.EndOfBuffer;
        }
        const slice = self.buffer[self.pos .. self.pos + size];
        self.pos += size;
        const result = try allocator.dupe(u8, slice);
        errdefer allocator.free(result);
        return result;
    }

    pub fn readBytes(self: *ProtoReader, allocator: std.mem.Allocator) ![]u8 {
        const len = try self.readVarint();
        if (len > std.math.maxInt(usize)) {
            return error.LengthTooLong;
        }
        const size: usize = @intCast(len);
        if (self.pos + size > self.buffer.len) {
            return error.EndOfBuffer;
        }
        const bytes = try allocator.dupe(u8, self.buffer[self.pos .. self.pos + size]);
        self.pos += size;
        return bytes;
    }

    pub fn readFixedBytes(self: *ProtoReader, allocator: std.mem.Allocator, size: usize) ![]u8 {
        if (size > self.buffer.len - self.pos) {
            return error.EndOfBuffer;
        }
        const bytes = try allocator.dupe(u8, self.buffer[self.pos .. self.pos + size]);
        self.pos += size;
        return bytes;
    }

    pub fn skipField(self: *ProtoReader, wire_type: WireType) !void {
        switch (wire_type) {
            .Varint => _ = try self.readVarint(),
            .Fixed64 => try self.skip(8),
            .LengthDelimited => {
                const len = try self.readVarint();
                try self.skip(len);
            },
            .Fixed32 => try self.skip(4),
            else => {
                std.log.scoped(.protobuf).warn("\n ERROR! wire type {any} not supported", .{wire_type});
                return error.UnsupportedWireType;
            },
        }
    }

    pub fn hasMore(self: *ProtoReader) bool {
        return self.pos < self.buffer.len;
    }

    pub fn readFloat(self: *ProtoReader) !f32 {
        const bytes = try self.readFixedBytes(self.allocator, 4);
        return @as(f32, @bitCast(@as(u32, bytes[0]) |
            (@as(u32, bytes[1]) << 8) |
            (@as(u32, bytes[2]) << 16) |
            (@as(u32, bytes[3]) << 24)));
    }

    pub fn skip(self: *ProtoReader, len: u64) !void {
        if (self.pos + len > self.buffer.len) {
            self.pos = self.buffer.len;
            return;
        }
        self.pos += @intCast(len);
    }
};
