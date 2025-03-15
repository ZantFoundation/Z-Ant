const std = @import("std");
const protobuf = @import("protobuf.zig");
const Version = @import("onnx.zig").Version;
const GraphProto = @import("onnx.zig").GraphProto;
const OperatorSetIdProto = @import("onnx.zig").OperatorSetIdProto;
const StringStringEntryProto = @import("onnx.zig").StringStringEntryProto;
const FunctionProto = @import("onnx.zig").FunctionProto;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var printingAllocator = std.heap.ArenaAllocator.init(gpa.allocator());

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
//  - 25: functions, repeated FunctionProto
//  - 26: TODO configuration, reapeted DeviceConfigurationProto
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
        for (self.opset_import) |opset| {
            opset.deinit(allocator);
            allocator.destroy(opset);
        }
        allocator.free(self.opset_import);

        for (self.metadata_props) |meta| {
            meta.deinit(allocator);
            allocator.destroy(meta);
        }
        allocator.free(self.metadata_props);

        for (self.functions) |func| {
            func.deinit(allocator);
            allocator.destroy(func);
        }
        allocator.free(self.functions);
    }

    pub fn parse(reader: *protobuf.ProtoReader, allocator: std.mem.Allocator) !ModelProto {
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

            allocator.free(model.opset_import);
            allocator.free(model.metadata_props);
            allocator.free(model.functions);
        }

        var opset_import_list = std.ArrayList(*OperatorSetIdProto).init(allocator);
        defer opset_import_list.deinit();

        var metadata_list = std.ArrayList(*StringStringEntryProto).init(allocator);
        defer metadata_list.deinit();

        var functions_list = std.ArrayList(*FunctionProto).init(allocator);
        defer functions_list.deinit();

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
                    var opset_reader = try reader.readLengthDelimited();
                    const opset_ptr = try allocator.create(OperatorSetIdProto);
                    opset_ptr.* = try OperatorSetIdProto.parse(&opset_reader, allocator);
                    try opset_import_list.append(opset_ptr);
                },
                14 => { // metadata_props
                    var meta_reader = try reader.readLengthDelimited();
                    const meta_ptr = try allocator.create(StringStringEntryProto);
                    meta_ptr.* = try StringStringEntryProto.parse(&meta_reader, allocator);
                    try metadata_list.append(meta_ptr);
                },
                25 => { // functions
                    var function_reader = try reader.readLengthDelimited();
                    const function_ptr = try allocator.create(FunctionProto);
                    function_ptr.* = try FunctionProto.parse(&function_reader, allocator);
                    try functions_list.append(function_ptr);
                },
                else => {
                    std.debug.print("\n\n ........default readLenghtDelimited, TAG:{any} \n", .{tag});
                    try reader.skipField(tag.wire_type);
                },
            }
        }

        model.opset_import = try opset_import_list.toOwnedSlice();
        model.metadata_props = try metadata_list.toOwnedSlice();
        model.functions = try functions_list.toOwnedSlice();

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
