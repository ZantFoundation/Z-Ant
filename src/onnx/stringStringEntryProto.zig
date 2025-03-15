const std = @import("std");
const protobuf = @import("protobuf.zig");
const AttributeType = @import("onnx.zig").AttributeType;
const DataType = @import("onnx.zig").DataType;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var printingAllocator = std.heap.ArenaAllocator.init(gpa.allocator());

// onnx library reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L531
// TAGS:
//  - 1 : key, optional string
//  - 2 : value, optional string
pub const StringStringEntryProto = struct {
    key: ?[]const u8,
    value: ?[]const u8,

    pub fn deinit(self: *StringStringEntryProto, allocator: std.mem.Allocator) void {
        if (self.key) |key| allocator.free(key);
        if (self.value) |value| allocator.free(value);
    }

    pub fn parse(reader: *protobuf.ProtoReader) !StringStringEntryProto {
        var ssep = StringStringEntryProto{
            .key = null,
            .value = null,
        };

        while (reader.hasMore()) {
            const tag = try reader.readTag();
            switch (tag.field_number) {
                1 => {
                    ssep.key = try reader.readString(reader.allocator);
                },
                2 => {
                    ssep.value = try reader.readString(reader.allocator);
                },
                else => {
                    std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE for StringStringEntryProto\n\n", .{tag});
                    return error.TagNotAvailable;
                },
            }
        }

        return ssep;
    }

    pub fn print(self: *StringStringEntryProto, padding: ?[]const u8) void {
        const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
            return;
        };

        std.debug.print("{s}StringStringEntryProto: key:{s}, value:{s} \n", .{
            space,
            if (self.key) |k| k else "(none)",
            if (self.value) |v| v else "(none)",
        });
    }
};
