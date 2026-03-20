const std = @import("std");
const protobuf = @import("protobuf.zig");
const AttributeType = @import("onnx.zig").AttributeType;

const onnx_log = std.log.scoped(.tensorProto);

//https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L700
//The struct Dimension is not present, instead the dimensions are saved inside .dims
//TAGS:
//  - 1 : dims, repeatedTensorShapeProto.Dimension
pub const TensorShapeProto = struct {

    //TAGS:
    //  - 1 : dim_value, int64
    //  - 2 : dim_param, string
    //  - 3 : denotation, optional string
    pub const Dimension = struct {
        dim_value: ?i64,
        dim_param: ?[]const u8,
        denotation: ?[]const u8,

        pub fn deinit(self: *Dimension, allocator: std.mem.Allocator) void {
            if (self.denotation) |den| allocator.free(den);
            if (self.dim_param) |p| allocator.free(p);
        }

        pub fn parse(reader: *protobuf.ProtoReader) !Dimension {
            var dim = Dimension{
                .dim_value = null,
                .dim_param = null,
                .denotation = null,
            };

            while (reader.hasMore()) {
                const tag = try reader.readTag();
                switch (tag.field_number) {
                    1 => { //dim_value
                        dim.dim_value = @bitCast(try reader.readVarint());
                    },
                    2 => { //dim_param
                        dim.dim_param = try reader.readString(reader.allocator);
                    },
                    3 => { //denotation
                        dim.dim_param = try reader.readString(reader.allocator);
                    },
                    else => {
                        onnx_log.warn("\n\n ERROR: tag{} NOT AVAILABLE for Dimension\n\n ", .{tag});
                        try reader.skipField(tag.wire_type);
                    },
                }
            }

            return dim;
        }

        pub fn print(self: Dimension, padding: ?[]const u8) void {
            var space_buf: [256]u8 = undefined;
            const space = std.fmt.bufPrint(&space_buf, "{s}   ", .{if (padding) |p| p else ""}) catch return;
            std.debug.print("{s}------------- DIMENSION\n", .{space});

            if (self.dim_value) |value| {
                std.debug.print("{s}Dim Value: {}\n", .{ space, value });
            } else {
                std.debug.print("{s}Dim Value: (none)\n", .{space});
            }

            if (self.dim_param) |param| {
                std.debug.print("{s}Dim Param: {s}\n", .{ space, param });
            } else {
                std.debug.print("{s}Dim Param: (none)\n", .{space});
            }

            if (self.denotation) |d| {
                std.debug.print("{s}Denotation: {s}\n", .{ space, d });
            } else {
                std.debug.print("{s}Denotation: (none)\n", .{space});
            }
        }
    };

    dims: []*Dimension,
    shape: []i64, //not parsed but created

    pub fn deinit(self: *TensorShapeProto, allocator: std.mem.Allocator) void {
        allocator.free(self.shape);

        for (self.dims) |dim| {
            dim.deinit(allocator);
            allocator.destroy(dim);
        }
        allocator.free(self.dims);
    }

    pub fn parse(reader: *protobuf.ProtoReader) !TensorShapeProto {
        var shape = TensorShapeProto{
            .shape = &[_]i64{},
            .dims = undefined,
        };

        var dims_list: std.ArrayList(*Dimension) = .empty;
        defer dims_list.deinit(reader.allocator);

        while (reader.hasMore()) {
            const tag = try reader.readTag();

            switch (tag.field_number) {
                1 => { // dim
                    var dim_reader = try reader.readLengthDelimited(); //var dim_reader
                    const dim_ptr = try reader.allocator.create(Dimension);
                    dim_ptr.* = try Dimension.parse(&dim_reader);
                    try dims_list.append(reader.allocator, dim_ptr);
                },
                else => {
                    onnx_log.warn("\n\n ERROR: tag{} NOT AVAILABLE for TensorShapeProto\n\n ", .{tag});
                    try reader.skipField(tag.wire_type);
                },
            }
        }

        //creating shape []i64
        var shape_list: std.ArrayList(i64) = .empty;
        defer shape_list.deinit(reader.allocator);
        for (dims_list.items) |d| {
            if (d.*.dim_value) |val| try shape_list.append(reader.allocator, val);
        }
        shape.shape = try shape_list.toOwnedSlice(reader.allocator);

        //creating dim []Dimension
        shape.dims = try dims_list.toOwnedSlice(reader.allocator);
        return shape;
    }

    pub fn print(self: *TensorShapeProto, padding: ?[]const u8) void {
        var space_buf: [256]u8 = undefined;
        const space = std.fmt.bufPrint(&space_buf, "{s}   ", .{if (padding) |p| p else ""}) catch return;
        std.debug.print("{s}------------- SHAPE\n", .{space});

        std.debug.print("{s}Shape: [", .{space});
        for (self.shape) |dim| {
            std.debug.print("{}", .{dim});
        }
        std.debug.print("]\n", .{});

        if (self.dims.len != 0) {
            std.debug.print("{s}Dimensions:\n", .{space});
            for (self.dims) |d| d.print(space);
        } else {
            std.debug.print("{s}Dimensions: (none)\n", .{space});
        }
    }
};
