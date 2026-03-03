const std = @import("std");
const protobuf = @import("protobuf.zig");
const NodeProto = @import("onnx.zig").NodeProto;
const TensorProto = @import("onnx.zig").TensorProto;
const ValueInfoProto = @import("onnx.zig").ValueInfoProto;
const DataType = @import("onnx.zig").DataType;
const StringStringEntryProto = @import("stringStringEntryProto.zig").StringStringEntryProto;
const TensorAnnotation = @import("tensorAnnotation.zig").TensorAnnotation;
const AttributeProto = @import("attributeProto.zig").AttributeProto;
const OperatorSetIdProto = @import("onnx.zig").OperatorSetIdProto;

const onnx_log = std.log.scoped(.functionProto);

// onnx library reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L909
//TAGS:
// - 1 : name, optional string
// - 2 :
// - 3 :
// - 4 : input,  repeated string
// - 5 : output, repeated string
// - 6 : attribute,  repeated string
// - 11: attribute_proto, repeated AttrubuteProto
// - 7 : node, repeated NodeProto
// - 8 : doc_string, optional string
// - 9 : opset_import, repeated OperatorSetIdProto
// - 10: domain, optional string
// - 13: overload, optional string
// - 12: value_info, repeated ValueInfoProto
// - 14: metadata_props, repeated StringStringEntryProto

pub const FunctionProto = struct {
    name: ?[]const u8,
    input: [][]const u8,
    output: [][]const u8,
    attribute: [][]const u8,
    attribute_proto: []*AttributeProto,
    node: []*NodeProto,
    doc_string: ?[]const u8,
    opset_import: []*OperatorSetIdProto,
    domain: ?[]const u8,
    overload: ?[]const u8,
    value_info: []*ValueInfoProto,
    metadata_props: []*StringStringEntryProto,

    pub fn deinit(self: *FunctionProto, allocator: std.mem.Allocator) void {
        if (self.name) |name| allocator.free(name);

        for (self.input) |input| {
            allocator.free(input);
        }
        allocator.free(self.input);

        for (self.output) |output| {
            allocator.free(output);
        }
        allocator.free(self.output);

        for (self.attribute) |attribute| {
            allocator.free(attribute);
        }
        allocator.free(self.attribute);

        for (self.attribute_proto) |attributeproto| {
            attributeproto.deinit(allocator);
            allocator.destroy(attributeproto);
        }
        allocator.free(self.attribute_proto);

        for (self.node) |node| {
            node.deinit(allocator);
            allocator.destroy(node);
        }
        allocator.free(self.node);

        if (self.doc_string) |doc_string| allocator.free(doc_string);

        for (self.opset_import) |opset| {
            opset.deinit(allocator);
            allocator.destroy(opset);
        }
        allocator.free(self.opset_import);

        if (self.domain) |domain| allocator.free(domain);

        if (self.overload) |overload| allocator.free(overload);

        for (self.value_info) |valueinfo| {
            valueinfo.deinit(allocator);
            allocator.destroy(valueinfo);
        }
        allocator.free(self.value_info);

        for (self.metadata_props) |metadata| {
            metadata.deinit(allocator);
            allocator.destroy(metadata);
        }
        allocator.free(self.metadata_props);
    }

    pub fn parse(reader: *protobuf.ProtoReader) !FunctionProto {
        var function = FunctionProto{
            .name = null,
            .input = &[_][]const u8{},
            .output = &[_][]const u8{},
            .attribute = &[_][]const u8{},
            .attribute_proto = &[_]*AttributeProto{},
            .node = &[_]*NodeProto{},
            .doc_string = null,
            .opset_import = &[_]*OperatorSetIdProto{},
            .domain = null,
            .overload = null,
            .value_info = &[_]*ValueInfoProto{},
            .metadata_props = undefined,
        };

        var inputs: std.ArrayList([]const u8) = .empty;
        defer inputs.deinit(reader.allocator);

        var outputs: std.ArrayList([]const u8) = .empty;
        defer outputs.deinit(reader.allocator);

        var attributes: std.ArrayList([]const u8) = .empty;
        defer attributes.deinit(reader.allocator);

        var attributes_proto: std.ArrayList(*AttributeProto) = .empty;
        defer attributes_proto.deinit(reader.allocator);

        var nodes: std.ArrayList(*NodeProto) = .empty;
        defer nodes.deinit(reader.allocator);

        var opset_imports: std.ArrayList(*OperatorSetIdProto) = .empty;
        defer opset_imports.deinit(reader.allocator);

        var value_infos: std.ArrayList(*ValueInfoProto) = .empty;
        defer value_infos.deinit(reader.allocator);

        var metadataList: std.ArrayList(*StringStringEntryProto) = .empty;
        defer metadataList.deinit(reader.allocator);

        while (reader.hasMore()) {
            const tag = try reader.readTag();
            switch (tag.field_number) {
                1 => { //name
                    function.name = try reader.readString(reader.allocator);
                },
                4 => { //input
                    const value = try reader.readString(reader.allocator);
                    try inputs.append(reader.allocator, value);
                },
                5 => { //output
                    const value = try reader.readString(reader.allocator);
                    try outputs.append(reader.allocator, value);
                },
                6 => { //attribute
                    const value = try reader.readString(reader.allocator);
                    try attributes.append(reader.allocator, value);
                },
                11 => { //attributes proto
                    var attr_reader = try reader.readLengthDelimited();
                    const attr_ptr = try reader.allocator.create(AttributeProto);
                    attr_ptr.* = try AttributeProto.parse(&attr_reader);
                    try attributes_proto.append(reader.allocator, attr_ptr);
                },
                7 => {
                    var node_reader = try reader.readLengthDelimited();
                    const node_ptr = try reader.allocator.create(NodeProto);
                    node_ptr.* = try NodeProto.parse(&node_reader);
                    try nodes.append(reader.allocator, node_ptr);
                },
                8 => { //doc_string
                    function.doc_string = try reader.readString(reader.allocator);
                },
                9 => { // opset_import
                    var setId_reader = try reader.readLengthDelimited();
                    const setId_ptr = try reader.allocator.create(OperatorSetIdProto);
                    setId_ptr.* = try OperatorSetIdProto.parse(&setId_reader);
                    try opset_imports.append(reader.allocator, setId_ptr);
                },

                10 => { //domain
                    function.domain = try reader.readString(reader.allocator);
                },
                13 => { //overload
                    function.overload = try reader.readString(reader.allocator);
                },
                12 => { //value_info
                    var value_info_reader = try reader.readLengthDelimited(); //var value_info_reader
                    const value_info_ptr = try reader.allocator.create(ValueInfoProto);
                    value_info_ptr.* = try ValueInfoProto.parse(&value_info_reader);
                    try value_infos.append(reader.allocator, value_info_ptr);
                },
                14 => { //metadata_props
                    var md_reader = try reader.readLengthDelimited(); //var md_reader
                    const ssep_ptr = try reader.allocator.create(StringStringEntryProto);
                    ssep_ptr.* = try StringStringEntryProto.parse(&md_reader);
                    try metadataList.append(reader.allocator, ssep_ptr);
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

        function.input = try inputs.toOwnedSlice(reader.allocator);
        function.output = try outputs.toOwnedSlice(reader.allocator);
        function.attribute = try attributes.toOwnedSlice(reader.allocator);
        function.attribute_proto = try attributes_proto.toOwnedSlice(reader.allocator);
        function.node = try nodes.toOwnedSlice(reader.allocator);
        function.opset_import = try opset_imports.toOwnedSlice(reader.allocator);
        function.value_info = try value_infos.toOwnedSlice(reader.allocator);
        function.metadata_props = try metadataList.toOwnedSlice(reader.allocator);

        return function;
    }

    pub fn print(self: *FunctionProto, padding: ?[]const u8) void {
        var space_buf: [256]u8 = undefined;
        const space = std.fmt.bufPrint(&space_buf, "{s}   ", .{if (padding) |p| p else ""}) catch "";

        std.debug.print("{s}------------- FUNCTION\n", .{space});

        if (self.name) |n| {
            std.debug.print("{s}Function Name: {s}\n", .{ space, n });
        } else {
            std.debug.print("{s}Function Name: (none)\n", .{space});
        }

        std.debug.print("{s}Inputs: ", .{space});
        for (self.input) |inp| {
            std.debug.print("{s}", .{inp});
        }
        std.debug.print("\n", .{});

        std.debug.print("{s}Outputs: ", .{space});
        for (self.output) |out| {
            std.debug.print("{s}{s} ", .{ space, out });
        }
        std.debug.print("\n", .{});

        std.debug.print("{s}Attributes: ", .{space});
        for (self.attribute) |attr| {
            std.debug.print("{s}{s} ", .{ space, attr });
        }
        std.debug.print("\n", .{});

        std.debug.print("{s}Attributes Proto:\n", .{space});
        for (self.attribute_proto) |attr| {
            attr.print(space);
        }

        std.debug.print("{s}Nodes:\n", .{space});
        for (self.node) |node| {
            node.print(space);
        }

        if (self.doc_string) |ds| {
            std.debug.print("{s}Function Doc string: {s}\n", .{ space, ds });
        } else {
            std.debug.print("{s}Function Doc string: (none)\n", .{space});
        }

        std.debug.print("{s}Opertor set id:\n", .{space});
        for (self.opset_import) |opset| {
            opset.print(space);
        }

        if (self.domain) |d| {
            std.debug.print("{s}Function Domain: {s}\n", .{ space, d });
        } else {
            std.debug.print("{s}Function Domain: (none)\n", .{space});
        }

        if (self.overload) |o| {
            std.debug.print("{s}Function Overload: {s}\n", .{ space, o });
        } else {
            std.debug.print("{s}Function Overload: (none)\n", .{space});
        }

        std.debug.print("{s}value infos (key, value) [{}]: \n", .{ space, self.metadata_props.len });
        for (self.value_info) |vi| {
            vi.print(space);
        }

        std.debug.print("{s}metadata_props (key, value) [{}]: \n", .{ space, self.metadata_props.len });
        for (self.metadata_props) |mp| {
            mp.print(space);
        }
    }
};
