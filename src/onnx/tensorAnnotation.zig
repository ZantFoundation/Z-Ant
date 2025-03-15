const std = @import("std");
const StringStringEntryProto = @import("stringStringEntryProto.zig").StringStringEntryProto;
const protobuf = @import("protobuf.zig");

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var printingAllocator = std.heap.ArenaAllocator.init(gpa.allocator());

// https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L536
//
//TAG:
//  - 1 : tensor_name, optional string
//  - 2 : quant_parameter_tensor_names, repeated StringStringEntryProto
pub const TensorAnnotation = struct {
    tensor_name: ?[]const u8,
    quant_parameter_tensor_names: []*StringStringEntryProto,

    pub fn deinit(self: *TensorAnnotation, allocator: std.mem.Allocator) void {
        if (self.tensor_name) |tn| allocator.free(tn);
        allocator.free(self.quant_parameter_tensor_names);
    }

    pub fn parse(reader: *protobuf.ProtoReader) !TensorAnnotation {
        var tensor = TensorAnnotation{
            .tensor_name = null,
            .quant_parameter_tensor_names = undefined,
        };

        errdefer {
            if (tensor.tensor_name) |n| reader.allocator.free(n);
        }

        var tensorNamesList = std.ArrayList(*StringStringEntryProto).init(reader.allocator);
        defer tensorNamesList.deinit();

        while (reader.hasMore()) {
            const tensor_tag = try reader.readTag();

            switch (tensor_tag.field_number) {
                1 => {
                    const str = try reader.readString(reader.allocator);
                    if (tensor.tensor_name) |old| reader.allocator.free(old);
                    tensor.tensor_name = str;
                },
                2 => {
                    std.debug.print("\n ................ TensorAnnotation READING  quant_parameter_tensor_names", .{});
                    var md_reader = try reader.readLengthDelimited(); //var md_reader
                    const ssep_ptr = try reader.allocator.create(StringStringEntryProto);
                    ssep_ptr.* = try StringStringEntryProto.parse(&md_reader);
                    try tensorNamesList.append(ssep_ptr);
                },
                else => {
                    std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE for AttributeProto\n\n ", .{tensor_tag});
                    try reader.skipField(tensor_tag.wire_type);
                },
            }
        }
        tensor.quant_parameter_tensor_names = try tensorNamesList.toOwnedSlice();

        return tensor;
    }

    pub fn print(self: *TensorAnnotation, padding: ?[]const u8) void {
        const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
            return;
        };

        std.debug.print("{s}------------- TensorAnnotation\n", .{space});

        if (self.tensor_name) |n| {
            std.debug.print("{s}Tensor Name: {s}\n", .{ space, n });
        } else {
            std.debug.print("{s}Tensor Name: (none)\n", .{space});
        }

        std.debug.print("{s}quant_parameter_tensor_names (key, value) [{}]: \n", .{ space, self.quant_parameter_tensor_names.len });
        for (self.quant_parameter_tensor_names) |mp| {
            mp.print(space);
        }
    }
};
