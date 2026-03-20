const std = @import("std");
const protobuf = @import("protobuf.zig");
const AttributeType = @import("onnx.zig").AttributeType;
const TensorProto = @import("onnx.zig").TensorProto;
const GraphProto = @import("graphProto.zig").GraphProto;
const TypeProto = @import("typeProto.zig").TypeProto;
const SparseTensorProto = @import("sparseTensorProto.zig").SparseTensorProto;

const onnx_log = std.log.scoped(.attributeProto);

//--
const parseError = @import("parseErrors.zig");
//--

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
//  - 22: optional SparseTensorProto sparse_tensor = 22;   sparse tensor value
//  - 23: sparse_tensor, optional SparseTensorProto
//reserved 12, 16 to 19;
//reserved "v";
pub const AttributeProto = struct {

    //MUST BE PRESENT FOR THIS VERSION OF THE IR
    name: []const u8, //Tag:1
    type: AttributeType, //Tag:20

    //EXACTLY ONE OF THIS MUST BE PRESENT
    f: f32 = 0, //Tag:2
    i: i64 = 0, //Tag:3
    s: []const u8, //Tag:4s
    t: ?*TensorProto, //Tag:5
    g: ?*GraphProto, //Tag:6 WIP
    //optional SparseTensorProto sparse_tensor = 22;   sparse tensor value

    //OTHERS, OLDER VERSIONS USE THIS BUT NEWER CODE DOESN'T
    floats: []f32, //Tag:7    list of floats
    ints: []i64, //Tag:8         list of ints
    strings: [][]const u8, //Tag:9  list of UTF-8 strings
    tensors: []*TensorProto, //Tag:10  list of tensors
    graphs: []*GraphProto, //Tag:11      list of graph
    doc_string: ?[]const u8, //Tag:13
    tp: ?*TypeProto, //Tag:14
    type_protos: []*TypeProto, //Tag:15.   list of type protos
    ref_attr_name: []const u8, //Tag:21
    sparse_tensor: ?*SparseTensorProto, //Tag:23.   list of sparse tensors

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

    pub fn parse(reader: *protobuf.ProtoReader) parseError.ParseError!AttributeProto {
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

        var floats_list: std.ArrayList(f32) = .empty;
        defer floats_list.deinit(reader.allocator);
        var ints_list: std.ArrayList(i64) = .empty;
        defer ints_list.deinit(reader.allocator);
        var strings_list: std.ArrayList([]const u8) = .empty;
        defer strings_list.deinit(reader.allocator);
        var tensors_list: std.ArrayList(*TensorProto) = .empty;
        defer tensors_list.deinit(reader.allocator);
        var graphs_list: std.ArrayList(*GraphProto) = .empty;
        defer graphs_list.deinit(reader.allocator);
        var type_protos_list: std.ArrayList(*TypeProto) = .empty;
        defer type_protos_list.deinit(reader.allocator);

        while (reader.hasMore()) {
            const attr_tag = try reader.readTag();
            //DEBUG
            //onnx_log.debug("Parsing attribute field {d} with wire type {}\n", .{ attr_tag.field_number, attr_tag.wire_type });
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
                    attr.i = @bitCast(value);
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
                6 => { // single graph (g)
                    var graph_reader = try reader.readLengthDelimited();
                    const graph_ptr = try reader.allocator.create(GraphProto);
                    graph_ptr.* = try GraphProto.parse(&graph_reader);
                    attr.g = graph_ptr;
                    attr.type = .GRAPH;
                },
                7 => { // repeated float (floats)
                    if (attr_tag.wire_type == .LengthDelimited) {
                        var floats_reader = try reader.readLengthDelimited();
                        while (floats_reader.hasMore()) {
                            if (floats_reader.available() < 4) break;
                            const v = try floats_reader.readFixed32();
                            try floats_list.append(reader.allocator, @bitCast(v));
                        }
                    } else {
                        const v = try reader.readFixed32();
                        try floats_list.append(reader.allocator, @bitCast(v));
                    }
                    if (attr.type != .INTS) attr.type = .FLOATS;
                },
                8 => { // repeated int64 (ints) or potential repeated int
                    const v = try reader.readVarint();
                    try ints_list.append(reader.allocator, @intCast(v));
                    //DEBUG
                    //onnx_log.debug("Added int value {d} to {s}\n", .{ v, attr.name });
                    attr.type = .INTS;
                },
                9 => { // strings
                    const value = try reader.readString(reader.allocator);
                    try strings_list.append(reader.allocator, value);
                },
                10 => { //tensors
                    _ = try reader.readLengthDelimited(); //var tensor_reader
                    var tensors_reader = try reader.readLengthDelimited();
                    const tensor_ptr = try reader.allocator.create(TensorProto);
                    tensor_ptr.* = try TensorProto.parse(&tensors_reader);
                    try tensors_list.append(reader.allocator, tensor_ptr);
                },
                11 => {
                    //var graph_reader = try reader.readLengthDelimited(); //var graph_reader
                    //const graph_ptr = try reader.allocator.create(GraphProto);
                    //graph_ptr.* = try GraphProto.parse(&graph_reader);
                    //try graphs_list.append(graph_ptr);
                    //attr.type = .GRAPHS;
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
                    try type_protos_list.append(reader.allocator, tp_ptr);
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
                    onnx_log.warn("\n\n ERROR: tag{} NOT AVAILABLE for AttributeProto\n\n ", .{attr_tag});

                    try reader.skipField(attr_tag.wire_type);
                },
            }
        }

        attr.floats = try floats_list.toOwnedSlice(reader.allocator);
        attr.ints = try ints_list.toOwnedSlice(reader.allocator);
        attr.strings = try strings_list.toOwnedSlice(reader.allocator);
        attr.graphs = try graphs_list.toOwnedSlice(reader.allocator);

        return attr;
    }

    pub fn print(self: *AttributeProto, padding: ?[]const u8) void {
        var space_buf: [256]u8 = undefined;
        const space = std.fmt.bufPrint(&space_buf, "{s}   ", .{if (padding) |p| p else ""}) catch return;
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
            for (self.floats) |val| {
                std.debug.print("{}", .{val});
            }
            std.debug.print("]\n", .{});
        }

        if (self.ints.len > 0) {
            std.debug.print("{s}Ints: [", .{space});
            for (self.ints) |val| {
                std.debug.print("{}", .{val});
            }
            std.debug.print("]\n", .{});
        }

        if (self.strings.len > 0) {
            std.debug.print("{s}Strings: [", .{space});
            for (self.strings) |val| {
                std.debug.print("\"{s}\"", .{val});
            }
            std.debug.print("]\n", .{});
        }

        if (self.tensors.len > 0) {
            std.debug.print("{s}Tensors: [", .{space});
            for (self.tensors) |val| {
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
            for (self.type_protos) |val| {
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
