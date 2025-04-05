const std = @import("std");
const protobuf = @import("protobuf.zig");
const AttributeType = @import("onnx.zig").AttributeType;
const TensorProto = @import("onnx.zig").TensorProto;
const GraphProto = @import("graphProto.zig").GraphProto;
const TypeProto = @import("typeProto.zig").TypeProto;
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
//  - 6 : g, optional GraphProto (graph)
//  - 7 : floats, repeated float
//  - 8 : ints, repeated int64
//  - 9 : strings, repeated bytes
//  - 10: tensors, repeated TensorProto
//  - 11: TODO graphs, repeated GraphProto (not possible to inferr error)
//  - 13: doc_string, optional string
//  - 14: tp, optional TypeProto
//  - 15: type_protos, repeated TypeProto
//  - 20: type, optional AttributeType //TODO !!check how
//  - 21: ref_attr_name, optional string
//  - 23: sparse_tensor, optional SparseTensorProto
//reserved 12, 16 to 19;
//reserved "v";
pub const AttributeProto = struct {
    name: []const u8, //Tag:1
    f: f32 = 0, //Tag:2
    i: i64 = 0, //Tag:3
    s: []const u8, //Tag:4s
    t: ?*TensorProto, //Tag:5
    g: ?*GraphProto, //Tag:6 WIP
    floats: []f32, //Tag:7
    ints: []i64, //Tag:8
    strings: [][]const u8, //Tag:9
    tensors: []*TensorProto, //Tag:10
    graphs: []*GraphProto, //Tag:11
    doc_string: ?[]const u8, //Tag:13
    tp: ?*TypeProto, //Tag:14
    type_protos: []*TypeProto, //Tag:15
    type: AttributeType, //Tag:20
    ref_attr_name: []const u8, //Tag:21
    sparse_tensor: ?*SparseTensorProto, //Tag:23

    pub fn deinit(self: *AttributeProto, allocator: std.mem.Allocator) void {
        //free name
        allocator.free(self.name);
        allocator.free(self.s);

        //free t
        if (self.t) |t| {
            t.deinit(allocator);
            allocator.destroy(t);
        }

        //free g
        if (self.g) |g| {
            g.deinit(allocator);
            allocator.destroy(g);
        }

        //free floats
        allocator.free(self.floats);

        //free ints
        allocator.free(self.ints);

        //free strings
        for (self.strings) |s| {
            allocator.free(s);
        }
        allocator.free(self.strings);

        //free tensors
        for (self.tensors) |ts| {
            ts.deinit(allocator);
            allocator.destroy(ts);
        }
        allocator.free(self.tensors);

        //free graphs
        for (self.graphs) |gs| {
            gs.deinit(allocator);
            allocator.destroy(gs);
        }
        allocator.free(self.graphs);

        //free doc_string
        if (self.doc_string) |doc| {
            allocator.free(doc);
        }

        //free tp
        if (self.tp) |tp| {
            tp.deinit(allocator);
            allocator.destroy(tp);
        }

        //free type_protos
        for (self.type_protos) |tp| {
            tp.deinit(allocator);
            allocator.destroy(tp);
        }

        //free ref_attr_name
        allocator.free(self.ref_attr_name);

        //free sparse_tensor
        if (self.sparse_tensor) |st| {
            st.deinit(allocator);
            allocator.destroy(st);
        }
    }

    pub fn parse(reader: *protobuf.ProtoReader) !AttributeProto {
        var attr = AttributeProto{
            .name = "",
            .f = 0,
            .i = 0,
            .s = "",
            .t = null,
            .g = null,
            .floats = &[_]f32{},
            .ints = &[_]i64{},
            .strings = &[_][]const u8{},
            .tensors = &[_]*TensorProto{},
            .graphs = undefined,
            .doc_string = "",
            .tp = null,
            .type_protos = &[_]*TypeProto{},
            .type = .UNDEFINED,
            .ref_attr_name = "",
            .sparse_tensor = null,
        };

        var floats_list = std.ArrayList(f32).init(reader.allocator);
        defer floats_list.deinit();
        var ints_list = std.ArrayList(i64).init(reader.allocator);
        defer ints_list.deinit();
        var strings_list = std.ArrayList([]const u8).init(reader.allocator);
        defer strings_list.deinit();
        var tensors_list = std.ArrayList(*TensorProto).init(reader.allocator);
        defer tensors_list.deinit();
        var graphs_list = std.ArrayList(*GraphProto).init(reader.allocator);
        defer graphs_list.deinit();
        var type_protos_list = std.ArrayList(*TypeProto).init(reader.allocator);
        defer type_protos_list.deinit();

        while (reader.hasMore()) {
            const attr_tag = try reader.readTag();
            //DEBUG
            //std.debug.print("Parsing attribute field {d} with wire type {}\n", .{ attr_tag.field_number, attr_tag.wire_type });
            switch (attr_tag.field_number) {
                1 => { // name
                    attr.name = try reader.readString(reader.allocator);
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
                    const value = try reader.readFixed32();
                    attr.f = @bitCast(value);
                    attr.type = .FLOAT;
                },
                3 => { // single int (i)
                    const value = try reader.readVarint();
                    attr.i = @intCast(value);
                    attr.type = .INT;
                },
                4 => { // single string (s)
                    attr.s = try reader.readString(reader.allocator);
                    attr.type = .STRING;
                },
                5 => { // single tensor (t)
                    var tensor_reader = try reader.readLengthDelimited();
                    const tensor_ptr = try reader.allocator.create(TensorProto);
                    tensor_ptr.* = try TensorProto.parse(&tensor_reader);
                    attr.t = tensor_ptr;
                    attr.type = .TENSOR;
                },
                6 => { // single tensor (t)
                    var tensor_reader = try reader.readLengthDelimited();
                    const tensor_ptr = try reader.allocator.create(TensorProto);
                    tensor_ptr.* = try TensorProto.parse(&tensor_reader);
                    attr.t = tensor_ptr;
                    if (attr.type != .INTS) attr.type = .TENSOR;
                },
                7 => { // repeated float (floats)
                    if (attr_tag.wire_type == .LengthDelimited) {
                        var floats_reader = try reader.readLengthDelimited();
                        while (floats_reader.hasMore()) {
                            if (floats_reader.available() < 4) break;
                            const v = try floats_reader.readFixed32();
                            try floats_list.append(@bitCast(v));
                        }
                    } else {
                        const v = try reader.readFixed32();
                        try floats_list.append(@bitCast(v));
                    }
                    if (attr.type != .INTS) attr.type = .FLOATS;
                },
                8 => { // repeated int64 (ints) or potential repeated int
                    const v = try reader.readVarint();
                    try ints_list.append(@intCast(v));
                    //DEBUG
                    //std.debug.print("Added int value {d} to {s}\n", .{ v, attr.name });
                    attr.type = .INTS;
                },
                9 => { // strings
                    const value = try reader.readString(reader.allocator);
                    try strings_list.append(value);
                },
                10 => { //tensors
                    _ = try reader.readLengthDelimited(); //var tensor_reader
                    var tensors_reader = try reader.readLengthDelimited();
                    const tensor_ptr = try reader.allocator.create(TensorProto);
                    tensor_ptr.* = try TensorProto.parse(&tensors_reader);
                    try tensors_list.append(tensor_ptr);
                },
                11 => { //graphs TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    _ = try reader.readLengthDelimited(); //var graph_reader
                    // const graph_ptr = try reader.allocator.create(GraphProto);
                    // graph_ptr.* = try GraphProto.parse(&graph_reader);
                    // try graphs_list.append(graph_ptr);
                    // attr.type = .GRAPHS;
                },
                13 => { // doc_string
                    attr.doc_string = try reader.readString(reader.allocator);
                },
                14 => { // tp
                    var tp_reader = try reader.readLengthDelimited();
                    const tp_ptr = try reader.allocator.create(TypeProto);
                    tp_ptr.* = try TypeProto.parse(&tp_reader);
                    attr.tp = tp_ptr;
                },
                15 => { // type_protos
                    var tp_reader = try reader.readLengthDelimited();
                    const tp_ptr = try reader.allocator.create(TypeProto);
                    tp_ptr.* = try TypeProto.parse(&tp_reader);
                    try type_protos_list.append(tp_ptr);
                },
                20 => { // type

                    const value = try reader.readVarint();
                    // Only set type if it's not already set to INTS
                    if (attr.type != .INTS) {
                        attr.type = @enumFromInt(@as(u8, @intCast(value)));
                    }
                },
                21 => { // ref_attr_name
                    attr.ref_attr_name = try reader.readString(reader.allocator);
                },
                23 => {
                    var tensor_reader = try reader.readLengthDelimited();
                    const tensor_ptr = try reader.allocator.create(SparseTensorProto);
                    tensor_ptr.* = try SparseTensorProto.parse(&tensor_reader);
                    attr.sparse_tensor = tensor_ptr;
                    if (attr.type != .INTS) attr.type = .SPARSE_TENSOR;
                },
                else => {
                    std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE for AttributeProto\n\n ", .{attr_tag});

                    try reader.skipField(attr_tag.wire_type);
                },
            }
        }

        attr.floats = try floats_list.toOwnedSlice();
        attr.ints = try ints_list.toOwnedSlice();
        attr.strings = try strings_list.toOwnedSlice();
        attr.graphs = try graphs_list.toOwnedSlice();

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

        if (self.tensors.len > 0) {
            std.debug.print("{s}Tensors: [", .{space});
            for (self.tensors, 0..) |val, i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("TensorProto", .{});
                val.print(space);
            }
            std.debug.print("]\n", .{});
        }

        //if (self.graphs.len > 0) {
        //   std.debug.print("{s}Graphs: [", .{space});
        // for (self.graphs, 0..) |val, i| {
        //   if (i > 0) std.debug.print(", ", .{});
        // std.debug.print("GraphProto", .{});
        //val.print(space);
        //}
        //std.debug.print("]\n", .{});
        //}

        if (self.doc_string) |doc| {
            std.debug.print("{s}Doc String: \"{s}\"\n", .{ space, doc });
        }

        if (self.tp) |tp| {
            std.debug.print("{s}TypeProto:\n", .{space});
            tp.print(space);
        }

        if (self.type_protos.len > 0) {
            std.debug.print("{s}Type Protos: [", .{space});
            for (self.type_protos, 0..) |val, i| {
                if (i > 0) std.debug.print(", ", .{});
                std.debug.print("TypeProto", .{});
                val.print(space);
            }
            std.debug.print("]\n", .{});
        }

        if (self.ref_attr_name.len > 0) {
            std.debug.print("{s}Ref Attr Name: \"{s}\"\n", .{ space, self.ref_attr_name });
        }

        if (self.sparse_tensor) |tensor| {
            std.debug.print("{s}Sparse Tensor:\n", .{space});
            tensor.print(space);
        }
    }
};
