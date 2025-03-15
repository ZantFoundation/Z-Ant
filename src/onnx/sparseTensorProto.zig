const std = @import("std");
const protobuf = @import("protobuf.zig");

const TensorProto = @import("tensorProto.zig").TensorProto;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var printingAllocator = std.heap.ArenaAllocator.init(gpa.allocator());

// onnx library reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L460

//TAGS:
//  - 1 : values, optional TensorProto
//  - 2 : indices, optional TensorProto
//  - 3:  dims, repeated int64
pub const SparseTensorProto = struct {
    values: ?*TensorProto,
    indices: ?*TensorProto,
    dims: []i64,

    pub fn deinit(self: *SparseTensorProto, allocator: std.mem.Allocator) void {
        if (self.values) |v| {
            v.deinit(allocator);
            allocator.destroy(v);
        }
        if (self.indices) |i| {
            i.deinit(allocator);
            allocator.destroy(i);
        }

        allocator.free(self.dims);
    }

    pub fn parse(reader: *protobuf.ProtoReader, allocator: std.mem.Allocator) !SparseTensorProto {
        var sp_tensor = SparseTensorProto{
            .values = null,
            .indices = null,
            .dims = &[_]i64{},
        };

        var dim_list = std.ArrayList(i64).init(allocator);
        defer dim_list.deinit();

        while (reader.hasMore()) {
            const sp_tag = try reader.readTag();

            switch (sp_tag.field_number) {
                1 => {
                    var tensor_reader = try reader.readLengthDelimited();
                    const tensor_ptr = try reader.allocator.create(TensorProto);
                    tensor_ptr.* = try TensorProto.parse(&tensor_reader);
                    sp_tensor.values = tensor_ptr;
                },
                2 => {
                    var tensor_reader = try reader.readLengthDelimited();
                    const tensor_ptr = try reader.allocator.create(TensorProto);
                    tensor_ptr.* = try TensorProto.parse(&tensor_reader);
                    sp_tensor.values = tensor_ptr;
                },
                3 => {
                    const d = try reader.readVarint();
                    try dim_list.append(@intCast(d));
                },
                else => {
                    std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE for sparseTensorProto\n\n ", .{sp_tag});
                    try reader.skipField(sp_tag.wire_type);
                },
            }
        }
        sp_tensor.dims = try dim_list.toOwnedSlice();

        return sp_tensor;
    }

    pub fn print(self: *SparseTensorProto, padding: ?[]const u8) void {
        const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
            return;
        };

        std.debug.print("{s}------------- SparseTensorProto\n", .{space});

        if (self.values) |tensor| {
            std.debug.print("{s}TensorProto:\n", .{space});
            tensor.print(space);
        }

        if (self.indices) |tensor| {
            std.debug.print("{s}TensorProto:\n", .{space});
            tensor.print(space);
        }

        if (self.dims.len > 0) {
            std.debug.print("{s}Dims: [", .{space});
            for (self.dims, 0..) |val, i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("{}", .{val});
            }
            std.debug.print("]\n", .{});
        }
    }
};
