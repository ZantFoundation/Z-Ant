const std = @import("std");
const protobuf = @import("protobuf.zig");

// onnx library reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L891
// TAGS:
// - 1 : domain, optional string
// - 2 : version, optional int 64

const onnx_log = std.log.scoped(.operatorSetIdProto);

pub const OperatorSetIdProto = struct {
    domain: ?[]const u8,
    version: ?i64,

    pub fn deinit(self: *OperatorSetIdProto, allocator: std.mem.Allocator) void {
        if (self.domain) |d| allocator.free(d);
    }

    pub fn parse(reader: *protobuf.ProtoReader) !OperatorSetIdProto {
        var idProto = OperatorSetIdProto{
            .domain = null,
            .version = null,
        };

        while (reader.hasMore()) {
            const tag = try reader.readTag();
            switch (tag.field_number) {
                1 => { //domain
                    idProto.domain = try reader.readString(reader.allocator);
                },
                2 => { //version
                    const value = try reader.readVarint();
                    idProto.version = @as(i64, @intCast(value));
                },
                else => {
                    onnx_log.info("\n\n ........default readLenghtDelimited, TAG:{any} \n", .{tag});
                    try reader.skipField(tag.wire_type);
                },
            }
        }

        return idProto;
    }

    pub fn print(self: *OperatorSetIdProto, padding: ?[]const u8) void {
        var space_buf: [256]u8 = undefined;
        const space = std.fmt.bufPrint(&space_buf, "{s}   ", .{if (padding) |p| p else ""}) catch return;

        std.debug.print("{s}------------- OPERATOR SET ID PROTO\n", .{space});

        if (self.domain) |d| {
            std.debug.print("{s}OperatorSetID domain: {s}\n", .{ space, d });
        } else {
            std.debug.print("{s}OperatorSetID domain: (none)\n", .{space});
        }

        if (self.version) |v| {
            std.debug.print("{s}OperatorSetID version: {}\n", .{ space, v });
        } else {
            std.debug.print("{s}OperatorSetID version: (none)\n", .{space});
        }
    }
};
