const std = @import("std");
const protobuf = @import("protobuf.zig");
const NodeProto = @import("onnx.zig").NodeProto;
const TensorProto = @import("onnx.zig").TensorProto;
const ValueInfoProto = @import("onnx.zig").ValueInfoProto;
const DataType = @import("onnx.zig").DataType;
const StringStringEntryProto = @import("stringStringEntryProto.zig").StringStringEntryProto;
const TensorAnnotation = @import("tensorAnnotation.zig").TensorAnnotation;
const SparseTensorProto = @import("sparseTensorProto.zig").SparseTensorProto;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var printingAllocator = std.heap.ArenaAllocator.init(gpa.allocator());

// onnx library reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L460
//TAGS:
//  - 1 : node, type: NodeProto repeated
//  - 2 : name
//  - 5 : initializer, type: TensorProto repeated
//  - 10: doc_string, optional
//  - 11: input, type: ValueInfoProto repeated
//  - 12: output, type: ValueInfoProto repeated
//  - 13: value_info, type: ValueInfoProto repeated
//  - 14: quantization_annotation, type: TensorAnnotation repeated
//  - 15: sparse_initializer, type: SparseTensorProto repeated
//  - 16: metadata_props, type: StringStringEntryProto repeated
//  - 3, 4, 6, 7, 8, 9 are reserved
pub const GraphProto = struct {
    name: ?[]const u8,
    nodes: []*NodeProto,
    initializers: []*TensorProto,
    inputs: []*ValueInfoProto,
    outputs: []*ValueInfoProto,
    value_info: []*ValueInfoProto,
    quantization_annotation: []*TensorAnnotation,
    sparse_initializer: []*SparseTensorProto,
    metadata_props: []*StringStringEntryProto,

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

        for (self.quantization_annotation) |qa| {
            qa.deinit(allocator);
            allocator.destroy(qa);
        }
        allocator.free(self.quantization_annotation);

        for (self.sparse_initializer) |sp| {
            sp.deinit(allocator);
            allocator.destroy(sp);
        }
        allocator.free(self.sparse_initializer);

        for (self.metadata_props) |mp| {
            mp.deinit(allocator);
            allocator.destroy(mp);
        }
        allocator.free(self.metadata_props);
    }

    pub fn parse(reader: *protobuf.ProtoReader) !GraphProto {
        var graph = GraphProto{
            .name = null,
            .nodes = &[_]*NodeProto{},
            .initializers = &[_]*TensorProto{},
            .inputs = &[_]*ValueInfoProto{},
            .outputs = &[_]*ValueInfoProto{},
            .value_info = &[_]*ValueInfoProto{},
            .quantization_annotation = &[_]*TensorAnnotation{},
            .sparse_initializer = &[_]*SparseTensorProto{},
            .metadata_props = &[_]*StringStringEntryProto{},
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
        var quantizationList = std.ArrayList(*TensorAnnotation).init(reader.allocator);
        defer quantizationList.deinit();
        var sparse_inizializers = std.ArrayList(*SparseTensorProto).init(reader.allocator);
        defer sparse_inizializers.deinit();
        var metadataList = std.ArrayList(*StringStringEntryProto).init(reader.allocator);
        defer metadataList.deinit();

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
                    var input_reader = try reader.readLengthDelimited();
                    const input_ptr = try reader.allocator.create(ValueInfoProto);
                    input_ptr.* = try ValueInfoProto.parse(&input_reader);
                    try inputs.append(input_ptr);
                },
                12 => { // output
                    // This field contains a list of ValueInfoProto messages, each representing an output of the graph.
                    // It provides information about the outputs' names, types, and shapes.
                    var output_reader = try reader.readLengthDelimited();
                    const output_ptr = try reader.allocator.create(ValueInfoProto);
                    output_ptr.* = try ValueInfoProto.parse(&output_reader);
                    try outputs.append(output_ptr);
                },
                13 => { // value_info
                    //This optional field holds a list of ValueInfoProto messages that describe intermediate values within the graph.
                    //While it's not mandatory for a value to appear in this list, when present, it offers detailed information about the values computed at various stages of the graph.
                    var value_info_reader = try reader.readLengthDelimited(); //var value_info_reader
                    const value_info_ptr = try reader.allocator.create(ValueInfoProto);
                    value_info_ptr.* = try ValueInfoProto.parse(&value_info_reader);
                    try value_infos.append(value_info_ptr);
                },
                14 => {
                    var quantization_reader = try reader.readLengthDelimited();
                    const quantization_ptr = try reader.allocator.create(TensorAnnotation);
                    quantization_ptr.* = try TensorAnnotation.parse(&quantization_reader);
                    try quantizationList.append(quantization_ptr);
                },
                15 => {
                    var tensor_reader = try reader.readLengthDelimited();
                    const tensor_ptr = try reader.allocator.create(SparseTensorProto);
                    tensor_ptr.* = try SparseTensorProto.parse(&tensor_reader);
                    try sparse_inizializers.append(tensor_ptr);
                },
                16 => {
                    var md_reader = try reader.readLengthDelimited(); //var md_reader
                    const ssep_ptr = try reader.allocator.create(StringStringEntryProto);
                    ssep_ptr.* = try StringStringEntryProto.parse(&md_reader);
                    try metadataList.append(ssep_ptr);
                },
                else => {
                    //std.debug.print("\n\n ........default readLenghtDelimited, TAG:{any} \n", .{tag});

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
        graph.quantization_annotation = try quantizationList.toOwnedSlice();
        graph.sparse_initializer = try sparse_inizializers.toOwnedSlice();
        graph.metadata_props = try metadataList.toOwnedSlice();

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

        std.debug.print("{s}Initializers  [{}]:\n", .{ space, self.initializers.len });
        for (self.initializers) |initializer| {
            initializer.print(space);
        }

        std.debug.print("{s}Inputs [{}]:\n", .{ space, self.inputs.len });
        for (self.inputs) |input| {
            input.print(space);
        }

        std.debug.print("{s}Outputs  [{}]: \n", .{ space, self.outputs.len });
        for (self.outputs) |output| {
            output.print(space);
        }

        std.debug.print("{s}Value_info [{}]:\n", .{ space, self.value_info.len });
        for (self.value_info) |vi| {
            vi.print(space);
        }

        std.debug.print("{s}Quantization Annotations:\n", .{space});
        for (self.quantization_annotation) |qa| {
            qa.print(space);
        }

        std.debug.print("{s}Sparse Initializers:\n", .{space});
        for (self.sparse_initializer) |sp| {
            sp.print(space);
        }

        std.debug.print("{s}metadata_props (key, value) [{}]: \n", .{ space, self.metadata_props.len });
        for (self.metadata_props) |mp| {
            mp.print(space);
        }
    }
};
