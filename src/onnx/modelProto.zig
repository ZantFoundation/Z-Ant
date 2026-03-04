const std = @import("std");
const protobuf = @import("protobuf.zig");
const Version = @import("onnx.zig").Version;
const GraphProto = @import("onnx.zig").GraphProto;
const OperatorSetIdProto = @import("onnx.zig").OperatorSetIdProto;
const StringStringEntryProto = @import("onnx.zig").StringStringEntryProto;
const FunctionProto = @import("onnx.zig").FunctionProto;

const onnx_log = std.log.scoped(.modelProto);

// onnx library reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L361
// TAGS:
//  - 1 : ir_version, optional int64
//  - 2 : optional string producer_name, optional string
//  - 3 : producer_version, optional string
//  - 4 : optional string domain
//  - 5 : model_version, optional int64
//  - 6 : doc_string, optional string
//  - 7 : graph, optional GraphProto
//  - 8 : opset_import, repeated OperatorSetIdProto
//  - 14: metadata_props, repeated StringStringEntryProto
//  - 20: TODO NOT IMPORTANT NOT URGENT training_info, repeated TrainingInfoProto
//  - 25: NOT URGENT functions, repeated FunctionProto
pub const ModelProto = struct {
    ir_version: Version,
    producer_name: ?[]const u8,
    producer_version: ?[]const u8,
    domain: ?[]const u8,
    model_version: ?i64,
    doc_string: ?[]const u8,
    graph: ?*GraphProto,
    opset_import: []*OperatorSetIdProto,
    metadata_props: []*StringStringEntryProto,
    functions: []*FunctionProto,

    pub fn deinit(self: *ModelProto, allocator: std.mem.Allocator) void {
        if (self.producer_name) |n| allocator.free(n);
        if (self.producer_version) |v| allocator.free(v);
        if (self.domain) |d| allocator.free(d);
        if (self.doc_string) |d| allocator.free(d);
        if (self.graph) |g| {
            g.deinit(allocator);
            allocator.destroy(g);
        }

        for (self.opset_import) |oi| {
            oi.deinit(allocator);
            allocator.destroy(oi);
        }
        allocator.free(self.opset_import);

        for (self.metadata_props) |mp| {
            mp.deinit(allocator);
            allocator.destroy(mp);
        }
        allocator.free(self.metadata_props);

        for (self.functions) |f| {
            f.deinit(allocator);
            allocator.destroy(f);
        }
        allocator.free(self.functions);
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
            .opset_import = &[_]*OperatorSetIdProto{},
            .metadata_props = &[_]*StringStringEntryProto{},
            .functions = &[_]*FunctionProto{},
        };
        errdefer {
            if (model.producer_name) |n| reader.allocator.free(n);
            if (model.producer_version) |v| reader.allocator.free(v);
            if (model.domain) |d| reader.allocator.free(d);
            if (model.doc_string) |d| reader.allocator.free(d);
            if (model.graph) |g| g.deinit(reader.allocator);
        }

        var opset_imports: std.ArrayList(*OperatorSetIdProto) = .empty;
        defer opset_imports.deinit(reader.allocator);

        var metadataList: std.ArrayList(*StringStringEntryProto) = .empty;
        defer metadataList.deinit(reader.allocator);

        var functions: std.ArrayList(*FunctionProto) = .empty;
        defer functions.deinit(reader.allocator);

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
                8 => { //opset_import
                    var setId_reader = try reader.readLengthDelimited();
                    const setId_ptr = try reader.allocator.create(OperatorSetIdProto);
                    setId_ptr.* = try OperatorSetIdProto.parse(&setId_reader);
                    try opset_imports.append(reader.allocator, setId_ptr);
                },
                14 => { // metadata props
                    var md_reader = try reader.readLengthDelimited();
                    const ssep_ptr = try reader.allocator.create(StringStringEntryProto);
                    ssep_ptr.* = try StringStringEntryProto.parse(&md_reader);
                    try metadataList.append(reader.allocator, ssep_ptr);
                },
                25 => { //functions
                    var fn_reader = try reader.readLengthDelimited();
                    const fn_ptr = try reader.allocator.create(FunctionProto);
                    fn_ptr.* = try FunctionProto.parse(&fn_reader);
                    try functions.append(reader.allocator, fn_ptr);
                },
                else => {
                    std.debug.print("\n\n ........default readLenghtDelimited, TAG:{any} \n", .{tag});
                    try reader.skipField(tag.wire_type);
                },
            }
        }

        model.opset_import = try opset_imports.toOwnedSlice(reader.allocator);
        model.metadata_props = try metadataList.toOwnedSlice(reader.allocator);
        model.functions = try functions.toOwnedSlice(reader.allocator);

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

        std.debug.print("{s}Operator set id:\n", .{" "});
        for (self.opset_import) |opset| {
            opset.print(null);
        }

        std.debug.print("Metadata count: {}\n", .{self.metadata_props.len});
        for (self.metadata_props) |metadata| {
            metadata.print(null);
        }

        std.debug.print("Functions count: {}\n", .{self.functions.len});
        for (self.functions) |fun| {
            fun.print(null);
        }
    }
};
