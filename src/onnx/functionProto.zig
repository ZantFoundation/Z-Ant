const std = @import("std");
const protobuf = @import("protobuf.zig");
const TensorProto = @import("onnx.zig").TensorProto;
const AttributeProto = @import("onnx.zig").AttributeProto;
const NodeProto = @import("onnx.zig").NodeProto;
const OperatorSetIdProto = @import("onnx.zig").OperatorSetIdProto;
const ValueInfoProto = @import("onnx.zig").ValueInfoProto;
const StringStringEntryProto = @import("onnx.zig").StringStringEntryProto;

// onnx library reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L909
//  TAGS:
// -1: name type: string
// -2: reserved
// - "since_version": reserved
// -3: reserved
// -4: input type: repeated string
// -5: output type: repeated string
// -6: attribute type: repeated string
// -11: attribute_proto type: repeated AttributeProto
// -7: node type: repeated NodeProto
// -8: doc_string type: optional string
// -9: opset_import type: repeated OperatorSetIdProto
// -10: domain type: optional string
// -13: overload type: optional string
// -12: value_info type: repeated ValueInfoProto
// -14: metadata_props type: repeated StringStringEntryProto

pub const FunctionProto = struct {
    name: []const u8,
    inputs: []*TensorProto,
    outputs: []*TensorProto,
    attributes: []*AttributeProto,
    attribute_proto: []*AttributeProto,
    nodes: []*NodeProto,
    doc_string: ?[]const u8,
    opset_import: []*OperatorSetIdProto,
    domain: ?[]const u8,
    overload: ?[]const u8,
    value_info: []*ValueInfoProto,
    metadata_props: []*StringStringEntryProto,

    pub fn deinit(self: *FunctionProto, allocator: std.mem.Allocator) void {
        allocator.free(self.name);

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

        for (self.attributes) |attr| {
            attr.deinit(allocator);
            allocator.destroy(attr);
        }
        allocator.free(self.attributes);

        for (self.attribute_proto) |attr_proto| {
            attr_proto.deinit(allocator);
            allocator.destroy(attr_proto);
        }
        allocator.free(self.attribute_proto);

        for (self.nodes) |node| {
            node.deinit(allocator);
            allocator.destroy(node);
        }
        allocator.free(self.nodes);

        if (self.doc_string) |doc_str| {
            allocator.free(doc_str);
        }

        for (self.opset_import) |opset| {
            opset.deinit(allocator);
            allocator.destroy(opset);
        }
        allocator.free(self.opset_import);

        if (self.domain) |domain_str| {
            allocator.free(domain_str);
        }

        if (self.overload) |overload_str| {
            allocator.free(overload_str);
        }

        for (self.value_info) |value_info| {
            value_info.deinit(allocator);
            allocator.destroy(value_info);
        }
        allocator.free(self.value_info);

        for (self.metadata_props) |metadata_prop| {
            metadata_prop.deinit(allocator);
            allocator.destroy(metadata_prop);
        }
        allocator.free(self.metadata_props);
    }

    pub fn parse(reader: *protobuf.ProtoReader, allocator: std.mem.Allocator) !FunctionProto {
        var function = FunctionProto{
            .name = "",
            .inputs = &[_]TensorProto{},
            .outputs = &[_]TensorProto{},
            .attributes = &[_]AttributeProto{},
            .nodes = &[_]NodeProto{},
            .doc_string = null,
            .opset_import = &[_]OperatorSetIdProto{},
            .domain = null,
            .overload = null,
            .value_info = &[_]ValueInfoProto{},
            .metadata_props = &[_]StringStringEntryProto{},
        };

        var inputs = std.ArrayList(TensorProto).init(allocator);
        defer inputs.deinit();
        var outputs = std.ArrayList(TensorProto).init(allocator);
        defer outputs.deinit();
        var attributes = std.ArrayList(AttributeProto).init(allocator);
        defer attributes.deinit();
        var nodes = std.ArrayList(NodeProto).init(allocator);
        defer nodes.deinit();
        var opset_import = std.ArrayList(OperatorSetIdProto).init(allocator);
        defer opset_import.deinit();
        var value_info = std.ArrayList(ValueInfoProto).init(allocator);
        defer value_info.deinit();
        var metadata_props = std.ArrayList(StringStringEntryProto).init(allocator);
        defer metadata_props.deinit();

        while (reader.hasMore()) {
            const tag = try reader.readTag();
            switch (tag.field_number) {
                1 => { //name(string)
                    function.name = try reader.readString(allocator);
                },
                4 => { //inputs(reapeted TensorProto)
                    var input_reader = try reader.readLengthDelimited();
                    const input_tensor_ptr = try allocator.create(TensorProto);
                    input_tensor_ptr.* = try TensorProto.parse(&input_reader);
                    try function.inputs.append(input_tensor_ptr);
                },
                5 => { // outputs(repeted TensorProto)
                    var output_reader = try reader.readLengthDelimited();
                    const output_tensor_ptr = try allocator.create(TensorProto);
                    output_tensor_ptr.* = try TensorProto.parse(&output_reader);
                    try function.outputs.append(output_tensor_ptr);
                },
                6 => { //attributes( repeted AttributeProto)
                    var attribute_reader = try reader.readLengthDelimited();
                    const attribute_ptr = try allocator.create(AttributeProto);
                    attribute_ptr.* = try AttributeProto.parse(&attribute_reader, allocator);
                    try function.attributes.append(attribute_ptr);
                },
                11 => { // attribute_proto (repeated AttributeProto)
                    var attribute_proto_reader = try reader.readLengthDelimited();
                    const attribute_proto_ptr = try allocator.create(AttributeProto);
                    attribute_proto_ptr.* = try AttributeProto.parse(&attribute_proto_reader, allocator);
                    try function.attribute_proto.append(attribute_proto_ptr);
                },
                7 => { // nodes (repeated NodeProto)
                    var node_reader = try reader.readLengthDelimited();
                    const node_ptr = try allocator.create(NodeProto);
                    node_ptr.* = try NodeProto.parse(&node_reader, allocator);
                    try function.nodes.append(node_ptr);
                },
                8 => { // doc_string (optional string)
                    function.doc_string = try reader.readString(allocator);
                },
                9 => { // opset_import (repeated OperatorSetIdProto)
                    var opset_reader = try reader.readLengthDelimited();
                    const opset_ptr = try allocator.create(OperatorSetIdProto);
                    opset_ptr.* = try OperatorSetIdProto.parse(&opset_reader, allocator);
                    try function.opset_import.append(opset_ptr);
                },
                10 => { // domain (optional string)
                    function.domain = try reader.readString(allocator);
                },
                13 => { // overload (optional string)
                    function.overload = try reader.readString(allocator);
                },
                12 => { // value_info (repeated ValueInfoProto)
                    var value_info_reader = try reader.readLengthDelimited();
                    const value_info_ptr = try allocator.create(ValueInfoProto);
                    value_info_ptr.* = try ValueInfoProto.parse(&value_info_reader, allocator);
                    try function.value_info.append(value_info_ptr);
                },
                14 => { // metadata_props (repeated StringStringEntryProto)
                    var metadata_reader = try reader.readLengthDelimited();
                    const metadata_entry_ptr = try allocator.create(StringStringEntryProto);
                    metadata_entry_ptr.* = try StringStringEntryProto.parse(&metadata_reader, allocator);
                    try function.metadata_props.append(metadata_entry_ptr);
                },
                else => {
                    std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE for FunztionProto \n\n", .{tag});
                    try reader.skipField(tag.wire_type);
                },
            }
        }

        function.inputs = inputs.toSlice();
        function.outputs = outputs.toSlice();
        function.attributes = attributes.toSlice();
        function.nodes = nodes.toSlice();
        function.opset_import = opset_import.toSlice();
        function.value_info = value_info.toSlice();
        function.metadata_props = metadata_props.toSlice();

        return function;
    }

    pub fn print(self: *FunctionProto, padding: ?[]const u8) void {
        const space = std.mem.concat(std.heap.page_allocator, u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch return;

        std.debug.print("{s}FunctionProto:\n", .{space});
        std.debug.print("{s}Name: {s}\n", .{ space, self.name });

        std.debug.print("{s}Inputs:\n", .{space});
        for (self.inputs) |input| {
            input.print(space);
        }

        std.debug.print("{s}Outputs:\n", .{space});
        for (self.outputs) |output| {
            output.print(space);
        }

        std.debug.print("{s}Attributes:\n", .{space});
        for (self.attributes) |attr| {
            attr.print(space);
        }

        std.debug.print("{s}Nodes:\n", .{space});
        for (self.nodes) |node| {
            node.print(space);
        }

        std.debug.print("{s}Doc String: {s}\n", .{ space, self.doc_string orelse "null" });

        std.debug.print("{s}Opset Import:\n", .{space});
        for (self.opset_import) |opset| {
            opset.print(space);
        }

        std.debug.print("{s}Domain: {s}\n", .{ space, self.domain orelse "null" });

        std.debug.print("{s}Overload: {s}\n", .{ space, self.overload orelse "null" });

        std.debug.print("{s}Value Info:\n", .{space});
        for (self.value_info) |value| {
            value.print(space);
        }

        std.debug.print("{s}Metadata Properties:\n", .{space});
        for (self.metadata_props) |metadata| {
            metadata.print(space);
        }
    }
};
