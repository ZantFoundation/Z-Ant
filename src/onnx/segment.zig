const std = @import("std");
const protobuf = @import("protobuf.zig");

const onnx_log = std.log.scoped(.segment);

// onnx library reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L503

//TAGS:
//  - 1 : begin, optional int64
//  - 2 : end, optional int64
pub const Segment = struct {
    begin: ?i64,
    end: ?i64,

    pub fn parse(reader: *protobuf.ProtoReader) !Segment {
        var segment = Segment{
            .begin = null,
            .end = null,
        };

        while (reader.hasMore()) {
            const seg_tag = try reader.readTag();

            switch (seg_tag.field_number) {
                1 => {
                    const value = try reader.readVarint();
                    segment.begin = @intCast(value);
                },
                2 => {
                    const value = try reader.readVarint();
                    segment.end = @intCast(value);
                },
                else => {
                    onnx_log.warn("\n\n ERROR: tag{} NOT AVAILABLE for AttributeProto\n\n ", .{seg_tag});
                    try reader.skipField(seg_tag.wire_type);
                },
            }
        }
        return segment;
    }

    pub fn print(self: *Segment, padding: ?[]const u8) void {
        var space_buf: [256]u8 = undefined;
        const space = std.fmt.bufPrint(&space_buf, "{s}   ", .{if (padding) |p| p else ""}) catch "";

        std.debug.print("{s}------------- SEGMENT\n", .{space});

        if (self.begin) |b| {
            std.debug.print("{s}Segment begin: {}\n", .{ space, b });
        } else {
            std.debug.print("{s}Segment begin: (none)\n", .{space});
        }

        if (self.end) |e| {
            std.debug.print("{s}Segment end: {}\n", .{ space, e });
        } else {
            std.debug.print("{s}Segment end: (none)\n", .{space});
        }
    }
};
