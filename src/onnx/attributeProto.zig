const std = @import("std");
const protobuf = @import("protobuf.zig");
const AttributeType = @import("onnx.zig").AttributeType;
const TensorProto = @import("onnx.zig").TensorProto;
const GraphProto = @import("graphProto.zig").GraphProto;
const SparseTensorProto = @import("sparseTensorProto.zig").SparseTensorProto;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var printingAllocator = std.heap.ArenaAllocator.init(gpa.allocator());

// https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L126
//TAG:
//  - 1 : name, optional string
//  - 2 : f, optional float
//  - 3 : i, optional int64
//  - 4 : s, optional bytes (UTF-8 string)
//  - 5 : t, optional TensorProto (tensor value)
//  - 6 : TODO g, optional GraphProto (graph)
//  - 7 : floats, repeated float
//  - 8 : ints, repeated int64
//  - 9 : TODO strings, repeated bytes
//  - 10: TODO tensors, repeated TensorProto
//  - 11: TODO graphs, repeated GraphProto
//  - 13: TODO doc_string, optional string
//  - 14: TODO tp, optional TypeProto
//  - 15: TODO type_protos, repeated TypeProto
//  - 20: type, optional AttributeType
//  - 21: TODO ref_attr_name, optional string
//  - 23: sparse_tensor, optional SparseTensorProto
//reserved 12, 16 to 19;
//reserved "v";
pub const AttributeProto = struct {
    name: []const u8,
    type: AttributeType,
    f: f32 = 0,
    i: i64 = 0,
    s: []const u8 = "",
    t: ?*TensorProto = null,
    g: ?*GraphProto = null,
    floats: []f32 = &[_]f32{},
    ints: []i64 = &[_]i64{},
    strings: [][]const u8 = &[_][]const u8{},
    sparse_tensor: ?*SparseTensorProto = null,

    pub fn deinit(self: *AttributeProto, allocator: std.mem.Allocator) void {
        allocator.free(self.name);

        switch (self.type) {
            .FLOAT => {},
            .INT => {},
            .STRING => allocator.free(self.s),
            .TENSOR => if (self.t) |t| {
                t.deinit(allocator);
                allocator.destroy(t);
            },
            .GRAPH => if (self.g) |g| {
                g.deinit(allocator);
                allocator.destroy(g);
            },
            .FLOATS => allocator.free(self.floats),
            .INTS => allocator.free(self.ints),
            .STRINGS => {
                for (self.strings) |s| allocator.free(s);
                allocator.free(self.strings);
            },
            .SPARSE_TENSOR => {
                if (self.sparse_tensor) |sp| {
                    sp.deinit(allocator);
                    allocator.destroy(sp);
                }
            },
            else => {},
        }
    }

    pub fn parseSingleAttribute(attr_reader: *protobuf.ProtoReader, allocator: std.mem.Allocator) !AttributeProto {
        var attr = AttributeProto{
            .name = "",
            .type = .UNDEFINED,
        };

        var floats_list = std.ArrayList(f32).init(allocator);
        defer floats_list.deinit();
        var ints_list = std.ArrayList(i64).init(allocator);
        defer ints_list.deinit();
        var strings_list = std.ArrayList([]const u8).init(allocator);
        defer {
            for (strings_list.items) |s| allocator.free(s);
            strings_list.deinit();
        }

        errdefer {
            for (strings_list.items) |s| allocator.free(s);
        }

        while (attr_reader.hasMore()) {
            const attr_tag = try attr_reader.readTag();
            //DEBUG
            //std.debug.print("Parsing attribute field {d} with wire type {}\n", .{ attr_tag.field_number, attr_tag.wire_type });
            switch (attr_tag.field_number) {
                1 => { // name
                    attr.name = try attr_reader.readString(allocator);
                    // Pre-set type for known Conv attributes
                    if (std.mem.eql(u8, attr.name, "dilations") or
                        std.mem.eql(u8, attr.name, "kernel_shape") or
                        std.mem.eql(u8, attr.name, "pads") or
                        std.mem.eql(u8, attr.name, "strides"))
                    {
                        attr.type = .INTS;
                    }
                },
                2 => { // single float (f)
                    const value = try attr_reader.readFixed32();
                    attr.f = @bitCast(value);
                    if (attr.type != .INTS) attr.type = .FLOAT;
                },
                3 => { // single int (i)
                    const value = try attr_reader.readVarint();
                    attr.i = @intCast(value);
                    if (attr.type != .INTS) attr.type = .INT;
                },
                4 => { // single string (s)
                    attr.s = try attr_reader.readString(allocator);
                    if (attr.type != .INTS) attr.type = .STRING;
                },
                5 => { // single tensor (t)
                    var tensor_reader = try attr_reader.readLengthDelimited();
                    const tensor_ptr = try allocator.create(TensorProto);
                    tensor_ptr.* = try TensorProto.parse(&tensor_reader);
                    attr.t = tensor_ptr;
                    if (attr.type != .INTS) attr.type = .TENSOR;
                },
                6 => {},
                7 => { // repeated float (floats)
                    if (attr_tag.wire_type == .LengthDelimited) {
                        var floats_reader = try attr_reader.readLengthDelimited();
                        while (floats_reader.hasMore()) {
                            if (floats_reader.available() < 4) break;
                            const v = try floats_reader.readFixed32();
                            try floats_list.append(@bitCast(v));
                        }
                    } else {
                        const v = try attr_reader.readFixed32();
                        try floats_list.append(@bitCast(v));
                    }
                    if (attr.type != .INTS) attr.type = .FLOATS;
                },
                8 => { // repeated int64 (ints) or potential repeated int
                    const v = try attr_reader.readVarint();
                    try ints_list.append(@intCast(v));
                    //DEBUG
                    //std.debug.print("Added int value {d} to {s}\n", .{ v, attr.name });
                    if (attr.type != .INTS) attr.type = .INTS;
                },
                20 => { // type
                    const value = try attr_reader.readVarint();
                    // Only set type if it's not already set to INTS
                    if (attr.type != .INTS) {
                        attr.type = @enumFromInt(@as(u8, @intCast(value)));
                    }
                },
                23 => {
                    var tensor_reader = try attr_reader.readLengthDelimited();
                    const tensor_ptr = try allocator.create(SparseTensorProto);
                    tensor_ptr.* = try SparseTensorProto.parse(&tensor_reader, allocator);
                    attr.sparse_tensor = tensor_ptr;
                    if (attr.type != .INTS) attr.type = .SPARSE_TENSOR;
                },
                else => {
                    std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE for AttributeProto\n\n ", .{attr_tag});

                    try attr_reader.skipField(attr_tag.wire_type);
                },
            }
        }

        switch (attr.type) {
            .FLOATS => attr.floats = try floats_list.toOwnedSlice(),
            .INTS => attr.ints = try ints_list.toOwnedSlice(),
            .STRINGS => attr.strings = try strings_list.toOwnedSlice(),
            else => {},
        }

        return attr;
    }

    pub fn print(self: *AttributeProto, padding: ?[]const u8) void {
        const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
            return;
        };
        std.debug.print("{s}------------- ATTRIBUTE \n", .{space});

        std.debug.print("{s}Name: {s}\n", .{ space, self.name });
        std.debug.print("{s}Type: {}\n", .{ space, self.type });

        if (self.f != 0) {
            std.debug.print("{s}Float: {}\n", .{ space, self.f });
        }

        if (self.i != 0) {
            std.debug.print("{s}Int: {}\n", .{ space, self.i });
        }

        if (self.s.len > 0) {
            std.debug.print("{s}String: \"{s}\"\n", .{ space, self.s });
        }

        if (self.t) |tensor| {
            std.debug.print("{s}Tensor:\n", .{space});
            tensor.print(space);
        }

        if (self.g) |tensor| {
            std.debug.print("{s}Tensor:\n", .{space});
            tensor.print(space);
        }

        if (self.floats.len > 0) {
            std.debug.print("{s}Floats: [", .{space});
            for (self.floats, 0..) |val, i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("{}", .{val});
            }
            std.debug.print("]\n", .{});
        }

        if (self.ints.len > 0) {
            std.debug.print("{s}Ints: [", .{space});
            for (self.ints, 0..) |val, i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("{}", .{val});
            }
            std.debug.print("]\n", .{});
        }

        if (self.strings.len > 0) {
            std.debug.print("{s}Strings: [", .{space});
            for (self.strings, 0..) |val, i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("\"{s}\"", .{val});
            }
            std.debug.print("]\n", .{});
        }

        if (self.sparse_tensor) |tensor| {
            std.debug.print("{s}Sparse Tensor:\n", .{space});
            tensor.print(space);
        }
    }
};
