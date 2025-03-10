//! Stop whatever you are doing and read this before proceding!
//! https://github.com/onnx/onnx/blob/main/onnx/onnx.proto

const std = @import("std");
const protobuf = @import("protobuf.zig");

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var printingAllocator = std.heap.ArenaAllocator.init(gpa.allocator());

pub const Version = enum(i64) {
    IR_VERSION_2017_10_10 = 0x0000000000000001,
    IR_VERSION_2017_10_30 = 0x0000000000000002,
    IR_VERSION_2017_11_3 = 0x0000000000000003,
    IR_VERSION_2019_1_22 = 0x0000000000000004,
    IR_VERSION_2019_3_18 = 0x0000000000000005,
    IR_VERSION_2019_9_19 = 0x0000000000000006,
    IR_VERSION_2020_5_8 = 0x0000000000000007,
    IR_VERSION_2021_7_30 = 0x0000000000000008,
    IR_VERSION_2023_5_5 = 0x0000000000000009,
    IR_VERSION_2024_3_25 = 0x000000000000000A,
    IR_VERSION = 0x000000000000000B,
};

pub const DataType = enum(i32) {
    UNDEFINED = 0,
    FLOAT = 1,
    UINT8 = 2,
    INT8 = 3,
    UINT16 = 4,
    INT16 = 5,
    INT32 = 6,
    INT64 = 7,
    STRING = 8,
    BOOL = 9,
    FLOAT16 = 10,
    DOUBLE = 11,
    UINT32 = 12,
    UINT64 = 13,
    COMPLEX64 = 14,
    COMPLEX128 = 15,
    BFLOAT16 = 16,
    FLOAT8E4M3FN = 17,
    FLOAT8E4M3FNUZ = 18,
    FLOAT8E5M2 = 19,
    FLOAT8E5M2FNUZ = 20,
    UINT4 = 21,
    INT4 = 22,
    FLOAT4E2M1 = 23,
};

pub const AttributeType = enum {
    UNDEFINED,
    FLOAT,
    INT,
    STRING,
    TENSOR,
    GRAPH,
    SPARSE_TENSOR,
    FLOATS,
    INTS,
    STRINGS,
    TENSORS,
    GRAPHS,
    SPARSE_TENSORS,
};

//https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L193
//TAGS:
//  - 1 : name, string
//  - 2 : type, TypeProto
//  - 3 : doc_string, string
//  - 4 : TODO metadata_props, StringStringEntryProto
pub const ValueInfoProto = struct {
    name: ?[]const u8,
    type: ?*TypeProto,
    doc_string: ?[]const u8,

    pub fn deinit(self: *ValueInfoProto, allocator: std.mem.Allocator) void {
        if (self.name) |n| allocator.free(n);
        if (self.doc_string) |doc_string| allocator.free(doc_string);
        if (self.type) |t| {
            t.deinit(allocator);
            allocator.destroy(t);
        }
    }

    pub fn parse(reader: *protobuf.ProtoReader) !ValueInfoProto {
        var value_info = ValueInfoProto{
            .name = undefined,
            .type = undefined,
            .doc_string = undefined,
        };

        while (reader.hasMore()) {
            const tag = try reader.readTag();
            switch (tag.field_number) {
                1 => { // name
                    std.debug.print("\n ................ ValueInfoProto READING name ", .{});
                    value_info.name = try reader.readString(reader.allocator);
                },
                2 => { // type
                    std.debug.print("\n ................ ValueInfoProto READING type ", .{});

                    var type_reader = try reader.readLengthDelimited(); //var type_reader
                    const type_ptr = try reader.allocator.create(TypeProto);
                    type_ptr.* = try TypeProto.parse(&type_reader);
                    value_info.type = type_ptr;
                },
                3 => { // doc_string
                    std.debug.print("\n ................ ValueInfoProto READING doc_string ", .{});
                    value_info.doc_string = try reader.readString(reader.allocator);
                },
                else => {
                    std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE for ValueInfoProto", .{tag});
                    try reader.skipField(tag.wire_type);
                },
            }
        }

        return value_info;
    }

    pub fn print(self: *ValueInfoProto, padding: ?[]const u8) void {
        const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
            return;
        };
        std.debug.print("{s}------------- VALUEINFO \n", .{space});

        if (self.name) |n| {
            std.debug.print("{s}Name: {s}\n", .{ space, n });
        } else {
            std.debug.print("{s}Name: (none)\n", .{space});
        }

        if (self.type) |t| {
            std.debug.print("{s}Type:\n", .{space});
            t.print(space);
        } else {
            std.debug.print("{s}Type: (none)\n", .{space});
        }

        if (self.doc_string) |doc| {
            std.debug.print("{s}Doc: {s}\n", .{ space, doc });
        } else {
            std.debug.print("{s}Doc: (none)\n", .{space});
        }
    }
};

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
//  - 9 : strings, repeated bytes
//  - 10: TODO tensors, repeated TensorProto
//  - 11: TODO graphs, repeated GraphProto
//  - 13: TODO doc_string, optional string
//  - 14: TODO tp, optional TypeProto
//  - 15: TODO type_protos, repeated TypeProto
//  - 20: TODO type, optional AttributeType
//  - 21: TODO ref_attr_name, optional string
//  - 23: TODO NOT URGENT sparse_tensor, optional SparseTensorProto
//reserved 12, 16 to 19;
//reserved "v";
pub const AttributeProto = struct {
    name: []const u8,
    type: AttributeType,
    f: f32 = 0,
    i: i64 = 0,
    s: []const u8 = "",
    t: ?*TensorProto = null,
    floats: []f32 = &[_]f32{},
    ints: []i64 = &[_]i64{},
    strings: [][]const u8 = &[_][]const u8{},

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
            .FLOATS => allocator.free(self.floats),
            .INTS => allocator.free(self.ints),
            .STRINGS => {
                for (self.strings) |s| allocator.free(s);
                allocator.free(self.strings);
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
                20 => { // type
                    const value = try attr_reader.readVarint();
                    // Only set type if it's not already set to INTS
                    if (attr.type != .INTS) {
                        attr.type = @enumFromInt(@as(u8, @intCast(value)));
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
                6 => { // repeated float (floats)
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
                7, 8 => { // repeated int64 (ints) or potential repeated int
                    const v = try attr_reader.readVarint();
                    try ints_list.append(@intCast(v));
                    //DEBUG
                    //std.debug.print("Added int value {d} to {s}\n", .{ v, attr.name });
                    if (attr.type != .INTS) attr.type = .INTS;
                },
                else => {
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
    }
};

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
                std.debug.print("\n .................................... Dimension TAG: {any} ", .{tag});
                switch (tag.field_number) {
                    1 => { //dim_value
                        std.debug.print("\n .................................... Dimension READING dim_value ", .{});
                        dim.dim_value = @bitCast(try reader.readVarint());
                        std.debug.print("\n .................................... dim.dim_value = {}", .{dim.dim_value.?});
                    },
                    2 => { //dim_param
                        std.debug.print("\n .................................... Dimension READING dim_param ", .{});
                        dim.dim_param = try reader.readString(reader.allocator);
                        std.debug.print("\n .................................... dim.dim_param = {s}", .{dim.dim_param.?});
                    },
                    3 => { //denotation
                        std.debug.print("\n .................................... Dimension READING denotation ", .{});
                        dim.dim_param = try reader.readString(reader.allocator);
                    },
                    else => {
                        std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE for Dimension\n\n ", .{tag});
                        try reader.skipField(tag.wire_type);
                    },
                }
            }

            return dim;
        }

        pub fn print(self: Dimension, padding: ?[]const u8) void {
            const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
                return;
            };
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

        var dims_list = std.ArrayList(*Dimension).init(reader.allocator);
        defer dims_list.deinit();

        while (reader.hasMore()) {
            const tag = try reader.readTag();
            std.debug.print("\n ................................. tensorShape TAG: {any} ", .{tag});

            switch (tag.field_number) {
                1 => { // dim
                    std.debug.print("\n ................................. TensorShapeProto READING dim ", .{});
                    var dim_reader = try reader.readLengthDelimited(); //var dim_reader
                    const dim_ptr = try reader.allocator.create(Dimension);
                    dim_ptr.* = try Dimension.parse(&dim_reader);
                    try dims_list.append(dim_ptr);
                },
                else => {
                    std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE for TensorShapeProto\n\n ", .{tag});
                    try reader.skipField(tag.wire_type);
                },
            }
        }

        //creating shape []i64
        var shape_list = std.ArrayList(i64).init(reader.allocator);
        defer shape_list.deinit();
        for (dims_list.items) |d| {
            if (d.*.dim_value) |val| try shape_list.append(val);
        }
        shape.shape = try shape_list.toOwnedSlice();
        std.debug.print("\n ................................. TensorShapeProto resulting shape = {any}", .{shape.shape});

        //creating dim []Dimension
        shape.dims = try dims_list.toOwnedSlice();
        return shape;
    }

    pub fn print(self: *TensorShapeProto, padding: ?[]const u8) void {
        const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
            return;
        };
        std.debug.print("{s}------------- SHAPE\n", .{space});

        std.debug.print("{s}Shape: [", .{space});
        for (self.shape, 0..) |dim, i| {
            if (i > 0) std.debug.print(", ", .{});
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

//https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L719
//TAG oneof:
//  - 1: tensor_type, type: TypeProto.Tensor
//  - 4: sequence_type, type: TypeProto.Sequence
//  - 5: map_type, type: TypeProto.Map
//  - 6: denotation, type: []const u8
//  - 8: TODO sparse_tensor_type, type: TypeProto.SparseTensor
//  - 9: TODO: optional_type, type: TypeProto.Optional
pub const TypeProto = struct {
    //TENSOR TAG:
    //  - 1: elem_type int32
    //  - 2: shape TensorShapeProto
    pub const Tensor = struct {
        elem_type: u32,
        shape: ?*TensorShapeProto,

        pub fn deinit(self: *Tensor, allocator: std.mem.Allocator) void {
            if (self.shape) |s| {
                s.deinit(allocator);
                allocator.destroy(s);
            }
        }

        pub fn parse(reader: *protobuf.ProtoReader) !Tensor {
            var tensor = Tensor{
                .elem_type = 0,
                .shape = null,
            };

            _ = &tensor;

            while (reader.hasMore()) {
                const tag = try reader.readTag();
                std.debug.print("\n .............................. tensor TAG: {any} ", .{tag});

                switch (tag.field_number) {
                    1 => { //elem_type
                        std.debug.print("\n .............................. Tensor READING elem_type ", .{});
                        tensor.elem_type = @intCast(try reader.readVarint());
                    },
                    2 => { //shape
                        std.debug.print("\n .............................. Tensor READING shape ", .{});

                        var shape_reader = try reader.readLengthDelimited(); //var shape_reader
                        const shape_ptr = try reader.allocator.create(TensorShapeProto);
                        shape_ptr.* = try TensorShapeProto.parse(&shape_reader);
                        tensor.shape = shape_ptr;
                    },
                    else => {
                        std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE for TensorProto\n\n", .{tag});
                        try reader.skipField(tag.wire_type);
                    },
                }
            }

            return tensor;
        }

        pub fn print(self: *Tensor, padding: ?[]const u8) void {
            const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
                return;
            };
            std.debug.print("{s}------------- TENSOR_TYPE\n", .{space});

            std.debug.print("{s}Element Type: {}\n", .{ space, self.elem_type });

            if (self.shape) |s| {
                std.debug.print("{s}Shape:\n", .{space});
                s.print(space);
            } else {
                std.debug.print("{s}Shape: (none)\n", .{space});
            }
        }
    };

    //SEQUENCE TAG:
    //  - 1: elem_type TypeProto
    pub const Sequence = struct {
        elem_type: ?*TypeProto,

        pub fn deinit(self: *Sequence, allocator: std.mem.Allocator) void {
            if (self.elem_type) |e| {
                e.deinit(allocator);
                allocator.destroy(e);
            }
        }

        pub fn parse(reader: *protobuf.ProtoReader) !Sequence {
            var sequence = Sequence{
                .elem_type = null,
            };

            _ = &sequence;

            while (reader.hasMore()) {
                const tag = try reader.readTag();
                std.debug.print("\n .............................. Sequence TAG: {any} ", .{tag});

                switch (tag.field_number) {
                    1 => { //elem_type
                        std.debug.print("\n .............................. Sequence READING elem_type ", .{});
                        _ = try reader.readLengthDelimited();
                    },
                    else => {
                        std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE for ", .{tag});
                        unreachable;
                    },
                }
            }

            return sequence;
        }

        pub fn print(self: *Sequence, padding: ?[]const u8) void {
            const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
                return;
            };
            std.debug.print("{s}------------- SEQUENCE\n", .{space});

            if (self.elem_type) |t| {
                std.debug.print("{s}Element Type:\n", .{space});
                t.print(space);
            } else {
                std.debug.print("{s}Element Type: (none)\n", .{space});
            }
        }
    };

    //MAP TAG:
    //  - 1: key_type u32
    //  - 2: value_type TypeProto
    pub const Map = struct {
        key_type: u32,
        value_type: ?*TypeProto,

        pub fn deinit(self: *Map, allocator: std.mem.Allocator) void {
            if (self.value_type) |v| {
                v.deinit(allocator);
                allocator.destroy(v);
            }
        }

        pub fn parse(reader: *protobuf.ProtoReader) !Tensor {
            var map = Map{
                .key_type = 0,
                .value_type = null,
            };

            _ = &map;

            while (reader.hasMore()) {
                const tag = try reader.readTag();
                std.debug.print("\n .............................. Map TAG: {any} ", .{tag});

                switch (tag.field_number) {
                    1 => { //elem_type
                        std.debug.print("\n .............................. Map READING elem_type ", .{});
                        _ = try reader.readLengthDelimited();
                    },
                    2 => { //value_type
                        std.debug.print("\n .............................. Map READING value_type ", .{});
                        _ = try reader.readLengthDelimited();
                    },
                    else => {
                        std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE ", .{tag});
                        unreachable;
                    },
                }
            }

            return map;
        }

        pub fn print(self: *Map, padding: ?[]const u8) void {
            const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
                return;
            };
            std.debug.print("{s}------------- MAP\n", .{space});

            std.debug.print("{s}Key Type: {}\n", .{ space, self.key_type });

            if (self.value_type) |v| {
                std.debug.print("{s}Value Type:\n", .{space});
                v.print(space);
            } else {
                std.debug.print("{s}Value Type: (none)\n", .{space});
            }
        }
    };

    //SPARSE TENSOR
    //  - 1: elem_type int32
    //  - 2: shape TensorShapeProto
    pub const SparseTensor = struct {
        elem_type: u32,
        shape: ?*TensorShapeProto,

        pub fn deinit(self: *SparseTensor, allocator: std.mem.Allocator) void {
            if (self.shape) |s| {
                s.deinit(allocator);
                allocator.destroy(s);
            }
        }

        pub fn parse(reader: *protobuf.ProtoReader) !SparseTensor {
            var sparse_tensor = SparseTensor{
                .elem_type = 0,
                .shape = null,
            };

            _ = &sparse_tensor;

            while (reader.hasMore()) {
                const tag = try reader.readTag();
                std.debug.print("\n .............................. tensor TAG: {any} ", .{tag});

                switch (tag.field_number) {
                    1 => { //elem_type
                        std.debug.print("\n .............................. Tensor READING elem_type ", .{});
                        _ = try reader.readLengthDelimited();
                    },
                    2 => { //shape
                        std.debug.print("\n .............................. Tensor READING tensor_type ", .{});
                        _ = try reader.readLengthDelimited();
                    },
                    else => {
                        std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE ", .{tag});
                        unreachable;
                    },
                }
            }

            return sparse_tensor;
        }

        pub fn print(self: *SparseTensor, padding: ?[]const u8) void {
            const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
                return;
            };
            std.debug.print("{s}------------- SparseTensor\n", .{space});
            std.debug.print("{s}Element Type: {}\n", .{ space, self.elem_type });

            if (self.shape) |s| {
                std.debug.print("{s}Shape:\n", .{space});
                s.print(space);
            } else {
                std.debug.print("{s}Shape: (none)\n", .{space});
            }
        }
    };

    //TAG OPTIONAL
    //  - 1: elem_type TypeProto
    pub const Optional = struct {
        elem_type: ?*TypeProto,

        pub fn deinit(self: *Optional, allocator: std.mem.Allocator) void {
            if (self.elem_type) |e| {
                e.deinit(allocator);
                allocator.destroy(e);
            }
        }

        pub fn parse(reader: *protobuf.ProtoReader) !Sequence {
            var opt = Optional{
                .elem_type = null,
            };

            _ = &opt;

            while (reader.hasMore()) {
                const tag = try reader.readTag();
                std.debug.print("\n .............................. Optional TAG: {any} ", .{tag});

                switch (tag.field_number) {
                    1 => { //elem_type
                        std.debug.print("\n .............................. Optional READING elem_type ", .{});
                        _ = try reader.readLengthDelimited();
                    },
                    else => {
                        std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE ", .{tag});
                        unreachable;
                    },
                }
            }

            return opt;
        }

        pub fn print(self: *Optional, padding: ?[]const u8) void {
            const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
                return;
            };
            std.debug.print("{s}------------- OPTIONAL\n", .{space});
            if (self.elem_type) |t| {
                std.debug.print("{s}Element Type:\n", .{space});
                t.print(space);
            } else {
                std.debug.print("{s}Element Type: (none)\n", .{space});
            }
        }
    };

    tensor_type: ?*Tensor,
    sequence_type: ?*Sequence,
    map_type: ?*Map,
    sparse_tensor_type: ?*SparseTensor, //TODO
    optional_type: ?*Optional,
    denotation: ?[]const u8,

    pub fn deinit(self: *TypeProto, allocator: std.mem.Allocator) void {
        if (self.tensor_type) |s| {
            s.deinit(allocator);
            allocator.destroy(s);
        }
        if (self.sequence_type) |st| {
            st.deinit(allocator);
            allocator.destroy(st);
        }
        if (self.map_type) |m| {
            m.deinit(allocator);
            allocator.destroy(m);
        }
        if (self.sparse_tensor_type) |stt| {
            stt.deinit(allocator);
            allocator.destroy(stt);
        }
        if (self.optional_type) |ot| {
            ot.deinit(allocator);
            allocator.destroy(ot);
        }
        if (self.denotation) |d| allocator.free(d);
    }

    pub fn parse(reader: *protobuf.ProtoReader) !TypeProto {
        var typeProto = TypeProto{
            .tensor_type = null,
            .sequence_type = null,
            .map_type = null,
            .sparse_tensor_type = null, //TODO
            .optional_type = null,
            .denotation = null,
        };

        _ = &typeProto;

        while (reader.hasMore()) {
            const tag = try reader.readTag();
            std.debug.print("\n ........................ TypeProto TAG: {any} ", .{tag});

            switch (tag.field_number) {
                1 => { //tensor_type
                    std.debug.print("\n ........................ TypeProto READING tensor_type ", .{});

                    var tensor_type_reader = try reader.readLengthDelimited();
                    const ensor_type_ptr = try reader.allocator.create(Tensor);
                    ensor_type_ptr.* = try Tensor.parse(&tensor_type_reader);
                    typeProto.tensor_type = ensor_type_ptr;
                },
                4 => { //TODO sequence_type
                    std.debug.print("\n ........................ TypeProto READING sequence_type ", .{});
                    _ = try reader.readLengthDelimited();
                },
                5 => { //TODO map_type
                    std.debug.print("\n ........................ TypeProto READING map_type ", .{});
                    _ = try reader.readLengthDelimited();
                },
                6 => { // TODO denotation
                    std.debug.print("\n ........................ TypeProto READING denotation ", .{});
                    _ = try reader.readLengthDelimited();
                },
                8 => { // TODO sparse_tensor_type
                    std.debug.print("\n ........................ TypeProto READING sparse_tensor_type ", .{});
                    _ = try reader.readLengthDelimited();
                },
                9 => { // TODO optional_type
                    std.debug.print("\n ........................ TypeProto READING sparse_tensor_type ", .{});
                    _ = try reader.readLengthDelimited();
                },
                else => {
                    std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE for TypeProto", .{tag});
                    try reader.skipField(tag.wire_type);
                },
            }
        }

        return typeProto;
    }

    pub fn print(self: *TypeProto, padding: ?[]const u8) void {
        const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
            return;
        };

        std.debug.print("{s}------------- TYPE\n", .{space});

        if (self.tensor_type) |t| {
            std.debug.print("{s}Tensor Type:\n", .{space});
            t.print(space);
        } else {
            std.debug.print("{s}Tensor Type: (none)\n", .{space});
        }

        if (self.sequence_type) |s| {
            std.debug.print("{s}Sequence Type:\n", .{space});
            s.print(space);
        } else {
            std.debug.print("{s}Sequence Type: (none)\n", .{space});
        }

        if (self.map_type) |m| {
            std.debug.print("{s}Map Type:\n", .{space});
            m.print(space);
        } else {
            std.debug.print("{s}Map Type: (none)\n", .{space});
        }

        if (self.sparse_tensor_type) |st| {
            std.debug.print("{s}Sparse Tensor Type:\n", .{space});
            st.print(space);
        } else {
            std.debug.print("{s}Sparse Tensor Type: (none)\n", .{space});
        }

        if (self.optional_type) |o| {
            std.debug.print("{s}Optional Type:\n", .{space});
            o.print(space);
        } else {
            std.debug.print("{s}Optional Type: (none)\n", .{space});
        }

        if (self.denotation) |d| {
            std.debug.print("{s}Denotation: {s}\n", .{ space, d });
        } else {
            std.debug.print("{s}Denotation: (none)\n", .{space});
        }
    }
};

// onnx library reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L503
//TAGS:
//  - 1 : dims, repeated int64
//  - 2 : data_type, optional int32
//  - 3 : TODO NOT URGENT segment, optional Segment
//  - 4 : float_data, repeated float
//  - 5 : int32_data, repeated int32
//  - 6 : string_data, repeated bytes
//  - 7 : int64_data, repeated int64
//  - 8 : name, optional string
//  - 9 : raw_data, optional bytes
//  - 10: double_data, repeated double
//  - 11: uint64_data, repeated uint64
//  - 12: TODO doc_string, optional string
//  - 13: TODO external_data, repeated StringStringEntryProto
//  - 14: TODO data_location, optional DataLocation
//  - 16: TODO metadata_props, repeated StringStringEntryProto
pub const TensorProto = struct {
    dims: []i64,
    data_type: DataType,
    name: ?[]const u8,
    raw_data: ?[]const u8,
    float_data: ?[]f32,
    int32_data: ?[]i32,
    string_data: ?[][]const u8,
    int64_data: ?[]i64,
    double_data: ?[]f64,
    uint64_data: ?[]u64,

    pub fn deinit(self: *TensorProto, allocator: std.mem.Allocator) void {
        allocator.free(self.dims);
        if (self.raw_data) |data| allocator.free(data);
        if (self.float_data) |data| allocator.free(data);
        if (self.int32_data) |data| allocator.free(data);
        if (self.int64_data) |data| allocator.free(data);
        if (self.double_data) |data| allocator.free(data);
        if (self.uint64_data) |data| allocator.free(data);
        if (self.string_data) |data| {
            for (data) |str| allocator.free(str);
            allocator.free(data);
        }
        if (self.name) |n| allocator.free(n);
    }

    pub fn parse(reader: *protobuf.ProtoReader) !TensorProto {
        var tensor = TensorProto{
            .dims = &[_]i64{},
            .data_type = .UNDEFINED,
            .name = null,
            .raw_data = null,
            .float_data = null,
            .int32_data = null,
            .string_data = null,
            .int64_data = null,
            .double_data = null,
            .uint64_data = null,
        };

        var dims = std.ArrayList(i64).init(reader.allocator);
        defer dims.deinit();

        while (reader.hasMore()) {
            const tag = try reader.readTag();
            switch (tag.field_number) {
                1 => { // dims
                    const value = try reader.readVarint();
                    try dims.append(@as(i64, @intCast(value)));
                },
                2 => { // data_type
                    const value = try reader.readVarint();
                    tensor.data_type = @enumFromInt((value));
                },
                8 => { // name
                    tensor.name = try reader.readString(reader.allocator);
                },
                9 => { // raw_data
                    tensor.raw_data = try reader.readBytes(reader.allocator);
                },
                4 => { // float_data
                    var data = std.ArrayList(f32).init(reader.allocator);
                    defer data.deinit();
                    if (tag.wire_type == .LengthDelimited) {
                        var float_reader = try reader.readLengthDelimited();
                        while (float_reader.hasMore()) {
                            if (float_reader.available() < 4) break;
                            const value = try float_reader.readFixed32();
                            try data.append(@bitCast(value));
                        }
                    } else {
                        const value = try reader.readFixed32();
                        try data.append(@bitCast(value));
                    }
                    tensor.float_data = try data.toOwnedSlice();
                },
                5 => { // int32_data
                    var data = std.ArrayList(i32).init(reader.allocator);
                    defer data.deinit();
                    if (tag.wire_type == .LengthDelimited) {
                        var int_reader = try reader.readLengthDelimited();
                        while (int_reader.hasMore()) {
                            const value = try int_reader.readVarint();
                            try data.append(@intCast(value));
                        }
                    } else {
                        const value = try reader.readVarint();
                        try data.append(@intCast(value));
                    }
                    tensor.int32_data = try data.toOwnedSlice();
                },
                7 => { // int64_data
                    var data = std.ArrayList(i64).init(reader.allocator);
                    defer data.deinit();
                    if (tag.wire_type == .LengthDelimited) {
                        var int_reader = try reader.readLengthDelimited();
                        while (int_reader.hasMore()) {
                            const value = try int_reader.readVarint();
                            try data.append(@intCast(value));
                        }
                    } else {
                        const value = try reader.readVarint();
                        try data.append(@intCast(value));
                    }
                    tensor.int64_data = try data.toOwnedSlice();
                },
                10 => { // double_data
                    var data = std.ArrayList(f64).init(reader.allocator);
                    defer data.deinit();
                    if (tag.wire_type == .LengthDelimited) {
                        var double_reader = try reader.readLengthDelimited();
                        while (double_reader.hasMore()) {
                            if (double_reader.available() < 8) break;
                            const value = try double_reader.readFixed64();
                            try data.append(@bitCast(value));
                        }
                    } else {
                        const value = try reader.readFixed64();
                        try data.append(@bitCast(value));
                    }
                    tensor.double_data = try data.toOwnedSlice();
                },
                11 => { // uint64_data
                    var data = std.ArrayList(u64).init(reader.allocator);
                    defer data.deinit();
                    if (tag.wire_type == .LengthDelimited) {
                        var uint_reader = try reader.readLengthDelimited();
                        while (uint_reader.hasMore()) {
                            const value = try uint_reader.readVarint();
                            try data.append(value);
                        }
                    } else {
                        const value = try reader.readVarint();
                        try data.append(value);
                    }
                    tensor.uint64_data = try data.toOwnedSlice();
                },
                else => {
                    std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE for TensorProto\n\n", .{tag});
                    try reader.skipField(tag.wire_type);
                },
            }
        }

        tensor.dims = try dims.toOwnedSlice();
        return tensor;
    }

    pub fn print(self: *TensorProto, padding: ?[]const u8) void {
        const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
            return;
        };

        std.debug.print("{s}------------- TENSOR\n", .{space});

        if (self.name) |n| {
            std.debug.print("{s}Name: {s}\n", .{ space, n });
        } else {
            std.debug.print("{s}Name: (none)\n", .{space});
        }

        std.debug.print("{s}Data Type: {any}\n", .{ space, self.data_type });

        std.debug.print("{s}Dims: [", .{space});
        for (self.dims, 0..) |dim, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{}", .{dim});
        }
        std.debug.print("]\n", .{});

        if (self.raw_data) |raw| {
            std.debug.print("{s}Raw Data: {} bytes\n", .{ space, raw.len });
        }

        if (self.float_data) |data| {
            std.debug.print("{s}Float Data: [", .{space});
            for (0..if (data.len < 10) data.len else 10) |i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("{}", .{data[i]});
            }
            if (data.len >= 10) std.debug.print(", ... ", .{});
            std.debug.print("]\n", .{});
        }

        if (self.int32_data) |data| {
            std.debug.print("{s}Int32 Data: [", .{space});
            for (0..if (data.len < 10) data.len else 10) |i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("{}", .{data[i]});
            }
            if (data.len >= 10) std.debug.print(", ... ", .{});
            std.debug.print("]\n", .{});
        }

        if (self.int64_data) |data| {
            std.debug.print("{s}Int64 Data: [", .{space});
            for (0..if (data.len < 10) data.len else 10) |i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("{}", .{data[i]});
            }
            if (data.len >= 10) std.debug.print(", ... ", .{});
            std.debug.print("]\n", .{});
        }

        if (self.double_data) |data| {
            std.debug.print("{s}Double Data: [", .{space});
            for (0..if (data.len < 10) data.len else 10) |i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("{}", .{data[i]});
            }
            if (data.len >= 10) std.debug.print(", ... ", .{});
            std.debug.print("]\n", .{});
        }

        if (self.uint64_data) |data| {
            std.debug.print("{s}UInt64 Data: [", .{space});
            for (0..if (data.len < 10) data.len else 10) |i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("{}", .{data[i]});
            }
            if (data.len >= 10) std.debug.print(", ... ", .{});
            std.debug.print("]\n", .{});
        }

        if (self.string_data) |data| {
            std.debug.print("  String Data: [", .{});
            for (data, 0..) |val, i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("\"{s}\"", .{val});
            }
            std.debug.print("]\n", .{});
        }
    }
};

//onnx library reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L212
//TAGS:
//  - 1 : input, repeated string
//  - 2 : output, repeated string
//  - 3 : name, optional string
//  - 4 : op_type, optional string
//  - 5 : attribute, repeated AttributeProto
//  - 6 : doc_string, optional string
//  - 7 : domain, optional string
//  - 8 : overload, optional string
//  - 9 : metadata_props, repeated StringStringEntryProto
pub const NodeProto = struct {
    name: ?[]const u8,
    op_type: []const u8,
    domain: ?[]const u8,
    input: [][]const u8,
    output: [][]const u8,
    attribute: []*AttributeProto,

    pub fn deinit(self: *NodeProto, allocator: std.mem.Allocator) void {
        if (self.name) |name| allocator.free(name);
        allocator.free(self.op_type);
        if (self.domain) |domain| allocator.free(domain);
        for (self.input) |input| {
            allocator.free(input);
        }
        allocator.free(self.input);
        for (self.output) |output| {
            allocator.free(output);
        }
        allocator.free(self.output);
        for (self.attribute) |attr| {
            attr.deinit(allocator);
            allocator.destroy(attr);
        }
        allocator.free(self.attribute);
    }

    pub fn parse(reader: *protobuf.ProtoReader) !NodeProto {
        var node = NodeProto{
            .name = null,
            .op_type = undefined,
            .domain = null,
            .input = &[_][]const u8{},
            .output = &[_][]const u8{},
            .attribute = &[_]*AttributeProto{},
        };

        var inputs = std.ArrayList([]const u8).init(reader.allocator);
        defer inputs.deinit();
        var outputs = std.ArrayList([]const u8).init(reader.allocator);
        defer outputs.deinit();
        var attributes = std.ArrayList(*AttributeProto).init(reader.allocator);
        defer attributes.deinit();

        errdefer {
            if (node.name) |n| reader.allocator.free(n);
            if (node.domain) |d| reader.allocator.free(d);
            reader.allocator.free(node.op_type);

            for (inputs.items) |i| reader.allocator.free(i);
            for (outputs.items) |o| reader.allocator.free(o);
            for (attributes.items) |attr| {
                attr.deinit(reader.allocator);
                reader.allocator.destroy(attr);
            }
        }

        while (reader.hasMore()) {
            const tag = try reader.readTag();
            switch (tag.field_number) {
                1 => { // input
                    const value = try reader.readString(reader.allocator);
                    try inputs.append(value);
                },
                2 => { // output
                    const value = try reader.readString(reader.allocator);
                    try outputs.append(value);
                },
                3 => { // name
                    node.name = try reader.readString(reader.allocator);
                },
                4 => { // op_type
                    node.op_type = try reader.readString(reader.allocator);
                },
                5 => { // attribute (repeated)
                    var attr_reader = try reader.readLengthDelimited();
                    const attr_ptr = try reader.allocator.create(AttributeProto);
                    attr_ptr.* = try AttributeProto.parseSingleAttribute(&attr_reader, reader.allocator);
                    try attributes.append(attr_ptr);
                },
                7 => { // domain
                    node.domain = try reader.readString(reader.allocator);
                },
                else => {
                    std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE for NodeProto\n\n", .{tag});
                    try reader.skipField(tag.wire_type);
                },
            }
        }

        node.input = try inputs.toOwnedSlice();
        node.output = try outputs.toOwnedSlice();
        node.attribute = try attributes.toOwnedSlice();
        return node;
    }

    pub fn print(self: *NodeProto, padding: ?[]const u8) void {
        const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
            return;
        };

        std.debug.print("{s}------------- NODE\n", .{space});

        if (self.name) |n| {
            std.debug.print("{s}Name: {s}\n", .{ space, n });
        } else {
            std.debug.print("{s}Name: (none)\n", .{space});
        }

        std.debug.print("{s}Op Type: {s}\n", .{ space, self.op_type });

        if (self.domain) |d| {
            std.debug.print("{s}Domain: {s}\n", .{ space, d });
        } else {
            std.debug.print("{s}Domain: (none)\n", .{space});
        }

        std.debug.print("{s}Inputs: ", .{space});
        for (self.input, 0..) |inp, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{s}", .{inp});
        }
        std.debug.print("\n", .{});

        std.debug.print("{s}Outputs: ", .{space});
        for (self.output, 0..) |out, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{s}{s} ", .{ space, out });
        }
        std.debug.print("\n", .{});

        std.debug.print("{s}Attributes:\n", .{space});
        for (self.attribute) |attr| {
            attr.print(space);
        }
    }
};

// onnx library reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L460
//TAGS:
//  - 1 : node, type: NodeProto repeated
//  - 2 : name
//  - 5 : initializer, type: TensorProto repeated
//  - 10: doc_string, optional
//  - 11: input, type: ValueInfoProto repeated
//  - 12: output, type: ValueInfoProto repeated
//  - 13: value_info, type: ValueInfoProto repeated
//  - 14: TODO: quantization_annotation, type: TensorAnnotation repeated
//  - 15: TODO: sparse_initializer, type: TensorProto repeated
//  - 16: TODO: metadata_props, type: StringStringEntryProto repeated
//  - 3, 4, 6, 7, 8, 9 are reserved
pub const GraphProto = struct {
    name: ?[]const u8,
    nodes: []*NodeProto,
    initializers: []*TensorProto,
    inputs: []*ValueInfoProto,
    outputs: []*ValueInfoProto,
    value_info: []*ValueInfoProto,

    pub fn deinit(self: *GraphProto, allocator: std.mem.Allocator) void {
        if (self.name) |n| allocator.free(n);
        for (self.nodes) |node| {
            node.deinit(allocator);
            allocator.destroy(node);
        }
        allocator.free(self.nodes);

        for (self.initializers) |init| {
            init.deinit(allocator);
            allocator.destroy(init);
        }
        allocator.free(self.initializers);

        for (self.inputs) |input| {
            input.deinit(allocator);
            allocator.destroy(input);
        }
        allocator.free(self.inputs);

        for (self.outputs) |output| {
            output.deinit(allocator);
            allocator.destroy(output);
        }
        allocator.free(self.outputs);

        for (self.value_info) |vi| {
            vi.deinit(allocator);
            allocator.destroy(vi);
        }
        allocator.free(self.value_info);
    }

    pub fn parse(reader: *protobuf.ProtoReader) !GraphProto {
        var graph = GraphProto{
            .name = null,
            .nodes = &[_]*NodeProto{},
            .initializers = &[_]*TensorProto{},
            .inputs = &[_]*ValueInfoProto{},
            .outputs = &[_]*ValueInfoProto{},
            .value_info = &[_]*ValueInfoProto{},
        };

        var nodes = std.ArrayList(*NodeProto).init(reader.allocator);
        defer nodes.deinit();
        var initializers = std.ArrayList(*TensorProto).init(reader.allocator);
        defer initializers.deinit();
        var inputs = std.ArrayList(*ValueInfoProto).init(reader.allocator);
        defer inputs.deinit();
        var outputs = std.ArrayList(*ValueInfoProto).init(reader.allocator);
        defer outputs.deinit();
        var value_infos = std.ArrayList(*ValueInfoProto).init(reader.allocator);
        defer value_infos.deinit();

        while (reader.hasMore()) {
            const tag = try reader.readTag();
            switch (tag.field_number) {
                1 => { // node
                    var node_reader = try reader.readLengthDelimited();
                    const node_ptr = try reader.allocator.create(NodeProto);
                    node_ptr.* = try NodeProto.parse(&node_reader);
                    try nodes.append(node_ptr);
                },
                2 => { // name
                    graph.name = try reader.readString(reader.allocator);
                },
                5 => { // initializer (repeated)
                    var tensor_reader = try reader.readLengthDelimited();
                    const tensor_ptr = try reader.allocator.create(TensorProto);
                    tensor_ptr.* = try TensorProto.parse(&tensor_reader);
                    try initializers.append(tensor_ptr);
                },
                10 => { // doc_string
                    var str_reader = try reader.readLengthDelimited();
                    _ = try str_reader.readString(reader.allocator);
                },

                11 => { // input
                    std.debug.print("\n\n ........GRAPH PROTO READING input ", .{});
                    var input_reader = try reader.readLengthDelimited();
                    const input_ptr = try reader.allocator.create(ValueInfoProto);
                    input_ptr.* = try ValueInfoProto.parse(&input_reader);
                    try inputs.append(input_ptr);
                },
                12 => { // output
                    // This field contains a list of ValueInfoProto messages, each representing an output of the graph.
                    // It provides information about the outputs' names, types, and shapes.
                    std.debug.print("\n\n ........GRAPH PROTO READING output ", .{});
                    var output_reader = try reader.readLengthDelimited();
                    const output_ptr = try reader.allocator.create(ValueInfoProto);
                    output_ptr.* = try ValueInfoProto.parse(&output_reader);
                    try outputs.append(output_ptr);
                },
                13 => { // value_info
                    //This optional field holds a list of ValueInfoProto messages that describe intermediate values within the graph.
                    //While it's not mandatory for a value to appear in this list, when present, it offers detailed information about the values computed at various stages of the graph.
                    std.debug.print("\n\n ........GRAPH PROTO READING value_info ", .{});
                    var value_info_reader = try reader.readLengthDelimited(); //var value_info_reader
                    const value_info_ptr = try reader.allocator.create(ValueInfoProto);
                    value_info_ptr.* = try ValueInfoProto.parse(&value_info_reader);
                    try value_infos.append(value_info_ptr);
                },
                else => {
                    std.debug.print("\n\n ........default readLenghtDelimited, TAG:{any} \n", .{tag});

                    var unknown_reader = try reader.readLengthDelimited();
                    while (unknown_reader.hasMore()) {
                        _ = try unknown_reader.readVarint();
                    }
                },
            }
        }

        graph.nodes = try nodes.toOwnedSlice();
        graph.initializers = try initializers.toOwnedSlice();
        graph.inputs = try inputs.toOwnedSlice();
        graph.outputs = try outputs.toOwnedSlice();
        graph.value_info = try value_infos.toOwnedSlice();

        return graph;
    }

    pub fn print(self: *GraphProto, padding: ?[]const u8) void {
        const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
            return;
        };

        std.debug.print("{s}------------- GRAPH\n", .{space});

        if (self.name) |n| {
            std.debug.print("{s}Graph Name: {s}\n", .{ space, n });
        } else {
            std.debug.print("{s}Graph Name: (none)\n", .{space});
        }

        std.debug.print("{s}Nodes:\n", .{space});
        for (self.nodes) |node| {
            node.print(space);
        }

        std.debug.print("{s}Initializers:\n", .{space});
        for (self.initializers) |initializer| {
            initializer.print(space);
        }

        std.debug.print("{s}Inputs:\n", .{space});
        for (self.inputs) |input| {
            input.print(space);
        }

        std.debug.print("{s}Outputs:\n", .{space});
        for (self.outputs) |output| {
            output.print(space);
        }
    }
};

// onnx library reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L361
// TAGS:
//  - 1 : ir_version, optional int64
//  - 2 : optional string producer_name, optional string
//  - 3 : producer_version, optional string
//  - 4 : optional string domain
//  - 5 : model_version, optional int64
//  - 6 : doc_string, optional string
//  - 7 : graph, optional GraphProto
//  - 8 : TODO opset_import, repeated OperatorSetIdProto
//  - 14: TODO metadata_props, repeated StringStringEntryProto
//  - 20: TODO NOT IMPORTANT NOT URGENT training_info, repeated TrainingInfoProto
//  - 25: TODO NOT URGENT functions, repeated FunctionProto
pub const ModelProto = struct {
    ir_version: Version,
    producer_name: ?[]const u8,
    producer_version: ?[]const u8,
    domain: ?[]const u8,
    model_version: ?i64,
    doc_string: ?[]const u8,
    graph: ?*GraphProto,

    pub fn deinit(self: *ModelProto, allocator: std.mem.Allocator) void {
        if (self.producer_name) |n| allocator.free(n);
        if (self.producer_version) |v| allocator.free(v);
        if (self.domain) |d| allocator.free(d);
        if (self.doc_string) |d| allocator.free(d);
        if (self.graph) |g| {
            g.deinit(allocator);
            allocator.destroy(g);
        }
    }

    pub fn parse(reader: *protobuf.ProtoReader) !ModelProto {
        var model = ModelProto{
            .ir_version = undefined,
            .producer_name = null,
            .producer_version = null,
            .domain = null,
            .model_version = null,
            .doc_string = null,
            .graph = null,
        };
        errdefer {
            if (model.producer_name) |n| reader.allocator.free(n);
            if (model.producer_version) |v| reader.allocator.free(v);
            if (model.domain) |d| reader.allocator.free(d);
            if (model.doc_string) |d| reader.allocator.free(d);
            if (model.graph) |g| g.deinit(reader.allocator);
        }

        while (reader.hasMore()) {
            const tag = try reader.readTag();
            switch (tag.field_number) {
                1 => { // ir_version
                    const value = try reader.readVarint();
                    model.ir_version = @enumFromInt(value);
                },
                2 => { // producer_name
                    const str = try reader.readString(reader.allocator);
                    if (model.producer_name) |old| reader.allocator.free(old);
                    model.producer_name = str;
                },
                3 => { // producer_version
                    const str = try reader.readString(reader.allocator);
                    if (model.producer_version) |old| reader.allocator.free(old);
                    model.producer_version = str;
                },
                4 => { // domain
                    const str = try reader.readString(reader.allocator);
                    if (model.domain) |old| reader.allocator.free(old);
                    model.domain = str;
                },
                5 => { // model_version
                    const value = try reader.readVarint();
                    model.model_version = @as(i64, @intCast(value));
                },
                6 => { // doc_string
                    const str = try reader.readString(reader.allocator);
                    if (model.doc_string) |old| reader.allocator.free(old);
                    model.doc_string = str;
                },
                7 => { // graph
                    if (model.graph) |g| {
                        g.deinit(reader.allocator);
                        reader.allocator.destroy(g);
                    }
                    var graph_reader = try reader.readLengthDelimited();
                    const graph_ptr = try reader.allocator.create(GraphProto);
                    graph_ptr.* = try GraphProto.parse(&graph_reader);
                    model.graph = graph_ptr;
                },
                else => {
                    std.debug.print("\n\n ........default readLenghtDelimited, TAG:{any} \n", .{tag});
                    try reader.skipField(tag.wire_type);
                },
            }
        }

        return model;
    }

    pub fn print(self: *ModelProto) void {
        std.debug.print("\n\n------------------------- MODEL -------------------------------\n", .{});

        std.debug.print("ModelProto:\n", .{});
        std.debug.print("  IR Version: {}\n", .{self.ir_version});

        if (self.producer_name) |name| {
            std.debug.print("  Producer Name: {s}\n", .{name});
        } else {
            std.debug.print("  Producer Name: (none)\n", .{});
        }

        if (self.producer_version) |version| {
            std.debug.print("  Producer Version: {s}\n", .{version});
        } else {
            std.debug.print("  Producer Version: (none)\n", .{});
        }

        if (self.domain) |d| {
            std.debug.print("  Domain: {s}\n", .{d});
        } else {
            std.debug.print("  Domain: (none)\n", .{});
        }

        if (self.model_version) |v| {
            std.debug.print("  Model Version: {}\n", .{v});
        } else {
            std.debug.print("  Model Version: (none)\n", .{});
        }

        if (self.doc_string) |doc| {
            std.debug.print("  Doc String: {s}\n", .{doc});
        } else {
            std.debug.print("  Doc String: (none)\n", .{});
        }

        if (self.graph) |g| {
            std.debug.print("  Graph:\n", .{});
            g.print(null);
        } else {
            std.debug.print("  Graph: (none)\n", .{});
        }

        printingAllocator.deinit();
    }
};

pub fn parseFromFile(allocator: std.mem.Allocator, file_path: []const u8) !ModelProto {
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();

    const file_size = try file.getEndPos();
    const buffer = try allocator.alloc(u8, @intCast(file_size));
    defer allocator.free(buffer);

    const bytes_read = try file.readAll(buffer);
    if (bytes_read != file_size) {
        return error.UnexpectedEOF;
    }

    var reader = protobuf.ProtoReader.init(allocator, buffer);
    var model = try ModelProto.parse(&reader);
    errdefer model.deinit(allocator);

    return model;
}

fn printTensorData(data: []const u8, data_type: DataType) void {
    switch (data_type) {
        .FLOAT => {
            const float_slice = @as([*]const f32, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 4)];
            const num_to_print = @min(float_slice.len, 10);
            for (float_slice[0..num_to_print]) |val| {
                std.debug.print("{d:.3} ", .{val});
            }
            if (float_slice.len > 10) {
                std.debug.print("...", .{});
            }
        },
        .INT32 => {
            const int_slice = @as([*]const i32, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 4)];
            const num_to_print = @min(int_slice.len, 10);
            for (int_slice[0..num_to_print]) |val| {
                std.debug.print("{d} ", .{val});
            }
            if (int_slice.len > 10) {
                std.debug.print("...", .{});
            }
        },
        .INT64 => {
            const int_slice = @as([*]const i64, @alignCast(@ptrCast(data.ptr)))[0..@divExact(data.len, 8)];
            const num_to_print = @min(int_slice.len, 10);
            for (int_slice[0..num_to_print]) |val| {
                std.debug.print("{d} ", .{val});
            }
            if (int_slice.len > 10) {
                std.debug.print("...", .{});
            }
        },
        else => {
            std.debug.print("(data type {s} not supported for display)", .{@tagName(data_type)});
        },
    }
}

pub fn printStructure(model: *ModelProto) void {
    // Print model info
    std.debug.print("\n=== Model Info ===\n", .{});
    std.debug.print("IR Version: {}\n", .{model.ir_version});
    if (model.producer_name) |name| {
        std.debug.print("Producer: {s}\n", .{name});
    }
    if (model.producer_version) |version| {
        std.debug.print("Version: {s}\n", .{version});
    }

    // Print graph info
    if (model.graph) |graph| {
        std.debug.print("\n=== Graph Info ===\n", .{});
        if (graph.name) |name| {
            std.debug.print("Name: {s}\n", .{name});
        }
        std.debug.print("Nodes: {d}\n", .{graph.nodes.len});
        std.debug.print("Initializers: {d}\n", .{graph.initializers.len});
        std.debug.print("Inputs: {d}\n", .{graph.inputs.len});

        // First, print a high-level view of the graph structure
        std.debug.print("\n=== Graph Structure ===\n", .{});
        for (graph.nodes, 0..) |node_ptr, i| {
            // Print current node
            std.debug.print("\n[{d}] {s}", .{ i, node_ptr.op_type });
            if (node_ptr.name) |name| {
                std.debug.print(" ({s})", .{name});
            }
            std.debug.print("\n", .{});

            // Print inputs with arrows
            std.debug.print("  Inputs:\n", .{});
            for (node_ptr.input) |input| {
                std.debug.print("    ← {s}\n", .{input});
            }

            // Print outputs with arrows
            std.debug.print("  Outputs:\n", .{});
            for (node_ptr.output) |output| {
                std.debug.print("    → {s}\n", .{output});
            }
        }

        // Then print detailed node information
        std.debug.print("\n=== Detailed Node Info ===\n", .{});
        for (graph.nodes, 0..) |node_ptr, i| {
            std.debug.print("\n[Node {d}]\n", .{i});
            if (node_ptr.name) |name| {
                std.debug.print("Name: {s}\n", .{name});
            }
            std.debug.print("Type: {s}\n", .{node_ptr.op_type});
            if (node_ptr.domain) |domain| {
                std.debug.print("Domain: {s}\n", .{domain});
            }

            // Print attributes
            if (node_ptr.attribute.len > 0) {
                std.debug.print("Attributes:\n", .{});
                for (node_ptr.attribute) |attr| {
                    std.debug.print("  {s}: ", .{attr.name});
                    switch (attr.type) {
                        .FLOAT => std.debug.print("float = {d}\n", .{attr.f}),
                        .INT => std.debug.print("int = {d}\n", .{attr.i}),
                        .STRING => std.debug.print("string = {s}\n", .{attr.s}),
                        .TENSOR => {
                            std.debug.print("tensor = ", .{});
                            if (attr.t) |t| {
                                std.debug.print("type: {}, shape: [", .{t.data_type});
                                for (t.dims, 0..) |dim, j| {
                                    if (j > 0) std.debug.print(", ", .{});
                                    std.debug.print("{d}", .{dim});
                                }
                                std.debug.print("]\n", .{});

                                // Print tensor data if available
                                if (t.float_data) |data| {
                                    std.debug.print("    data = [", .{});
                                    for (data[0..@min(data.len, 10)]) |val| {
                                        std.debug.print("{d:.3} ", .{val});
                                    }
                                    if (data.len > 10) {
                                        std.debug.print("...", .{});
                                    }
                                    std.debug.print("]\n", .{});
                                } else if (t.raw_data) |data| {
                                    std.debug.print("    raw_data = [", .{});
                                    printTensorData(data, t.data_type);
                                    std.debug.print("]\n", .{});
                                } else if (t.int32_data) |data| {
                                    std.debug.print("    int32_data = [", .{});
                                    for (data[0..@min(data.len, 10)]) |val| {
                                        std.debug.print("{d} ", .{val});
                                    }
                                    if (data.len > 10) {
                                        std.debug.print("...", .{});
                                    }
                                    std.debug.print("]\n", .{});
                                } else if (t.int64_data) |data| {
                                    std.debug.print("    int64_data = [", .{});
                                    for (data[0..@min(data.len, 10)]) |val| {
                                        std.debug.print("{d} ", .{val});
                                    }
                                    if (data.len > 10) {
                                        std.debug.print("...", .{});
                                    }
                                    std.debug.print("]\n", .{});
                                } else {
                                    std.debug.print("    (no data available)\n", .{});
                                }
                            } else {
                                std.debug.print("null\n", .{});
                            }
                        },
                        .FLOATS => {
                            std.debug.print("floats = [", .{});
                            for (attr.floats) |f| {
                                std.debug.print("{d} ", .{f});
                            }
                            std.debug.print("]\n", .{});
                        },
                        .INTS => {
                            std.debug.print("ints = [", .{});
                            for (attr.ints) |val| {
                                std.debug.print("{d} ", .{val});
                            }
                            std.debug.print("]\n", .{});
                        },
                        .STRINGS => {
                            std.debug.print("strings = [", .{});
                            for (attr.strings) |s| {
                                std.debug.print("{s} ", .{s});
                            }
                            std.debug.print("]\n", .{});
                        },
                        else => std.debug.print("unsupported type\n", .{}),
                    }
                }
            }
        }

        // Print initializer details
        std.debug.print("\n=== Initializers (weights, biases, etc.) ===\n", .{});
        for (graph.initializers, 0..) |init_ptr, i| {
            std.debug.print("\nInitializer {d}:\n", .{i});
            if (init_ptr.name) |name| {
                if (std.mem.indexOf(u8, name, "weight")) |_| {
                    std.debug.print("Name: {s} (weights/filters)\n", .{name});
                } else if (std.mem.indexOf(u8, name, "bias")) |_| {
                    std.debug.print("Name: {s} (bias values)\n", .{name});
                } else if (std.mem.indexOf(u8, name, "running_mean")) |_| {
                    std.debug.print("Name: {s} (batch norm mean)\n", .{name});
                } else if (std.mem.indexOf(u8, name, "running_var")) |_| {
                    std.debug.print("Name: {s} (batch norm variance)\n", .{name});
                } else {
                    std.debug.print("Name: {s}\n", .{name});
                }
            }
            std.debug.print("Type: {}\n", .{init_ptr.data_type});
            std.debug.print("Shape: [", .{});
            for (init_ptr.dims, 0..) |dim, j| {
                if (j > 0) std.debug.print(", ", .{});
                std.debug.print("{d}", .{dim});
            }
            std.debug.print("]\n", .{});

            // Print some data samples
            std.debug.print("Data preview: ", .{});
            if (init_ptr.float_data) |data| {
                std.debug.print(" float_data [", .{});
                for (data[0..@min(data.len, 5)]) |val| {
                    std.debug.print("{d:.3} ", .{val});
                }
                if (data.len > 5) {
                    std.debug.print("...", .{});
                }
                std.debug.print("]\n", .{});
            } else if (init_ptr.raw_data) |data| {
                std.debug.print(" raw_data [", .{});
                printTensorData(data, init_ptr.data_type);
                std.debug.print("]\n", .{});
            } else if (init_ptr.int32_data) |data| {
                std.debug.print(" int32_data [", .{});
                for (data[0..@min(data.len, 5)]) |val| {
                    std.debug.print("{d} ", .{val});
                }
                if (data.len > 5) {
                    std.debug.print("...", .{});
                }
                std.debug.print("]\n", .{});
            } else {
                std.debug.print("(no data available)\n", .{});
            }

            // Add explanation based on shape
            if (init_ptr.dims.len > 0) {
                std.debug.print("Description: ", .{});
                switch (init_ptr.dims.len) {
                    1 => std.debug.print("1D tensor with {d} values (typically bias or batch norm parameter)\n", .{init_ptr.dims[0]}),
                    2 => std.debug.print("2D matrix of size {d}x{d} (typically dense layer weights)\n", .{ init_ptr.dims[0], init_ptr.dims[1] }),
                    4 => std.debug.print("4D tensor of size {d}x{d}x{d}x{d} (convolutional filters: out_channels x in_channels x height x width)\n", .{ init_ptr.dims[0], init_ptr.dims[1], init_ptr.dims[2], init_ptr.dims[3] }),
                    else => std.debug.print("{d}D tensor\n", .{init_ptr.dims.len}),
                }
            }
        }
    }
}
