const std = @import("std");
const protobuf = @import("protobuf.zig");

// onnx library reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L891
// TAGS:
//  - 1 : domain, type: string
//  - 2 : version, type: int64

pub const OperatorSetIdProto = struct {
    version: i64,
    domain: []const u8,

    pub fn deinit(self: *OperatorSetIdProto, allocator: std.mem.Allocator) void {
        allocator.free(self.domain);
    }

    pub fn parse(reader: *protobuf.ProtoReader, allocator: std.mem.Allocator) !OperatorSetIdProto {
        var obj = OperatorSetIdProto{
            .version = 0,
            .domain = "",
        };

        while (reader.hasMore()) {
            const tag = try reader.readTag();
            switch (tag.field_number) {
                1 => { //version (int64)
                    obj.version = try reader.readVarint();
                },
                2 => { // domain (string)
                    obj.domain = try reader.readString(allocator);
                },
                else => {
                    std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE for OperatorSetIdProto\n\n", .{tag});
                    try reader.skipField(tag.wire_type);
                },
            }
        }
        return obj;
    }

    pub fn print(self: *OperatorSetIdProto, padding: ?[]const u8) void {
        const space = std.mem.concat(std.heap.page_allocator, u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch return;

        std.debug.print("{s} OperatorSetIdProto:\n", .{space});
        std.debug.print("{s}Version: {}\n", .{ space, self.version });
        std.debug.print("{s}Domain: {s}\n", .{ space, self.domain });
    }
};
