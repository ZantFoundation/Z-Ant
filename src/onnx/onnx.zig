//! Stop whatever you are doing and read this before proceding!
//! https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
const std = @import("std");
const protobuf = @import("protobuf.zig");

pub const ValueInfoProto = @import("valueInfoProto.zig").ValueInfoProto;
pub const AttributeProto = @import("attributeProto.zig").AttributeProto;
pub const TensorShapeProto = @import("tensorShapeProto.zig").TensorShapeProto;
pub const TypeProto = @import("typeProto.zig").TypeProto;
pub const TensorProto = @import("tensorProto.zig").TensorProto;
pub const NodeProto = @import("nodeProto.zig").NodeProto;
pub const GraphProto = @import("graphProto.zig").GraphProto;
pub const ModelProto = @import("modelProto.zig").ModelProto;
pub const StringStringEntryProto = @import("stringStringEntryProto.zig").StringStringEntryProto;
pub const OperatorSetIdProto = @import("operatorSetIdProto.zig").OperatorSetIdProto;
pub const FunctionProto = @import("functionProto.zig").FunctionProto;

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

pub const DataLocation = enum(i32) {
    DEFAULT = 0,
    EXTERNAL = 1,
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

    if (model.graph.?.value_info.len == 0 and model.graph.?.nodes.len > 1) {
        std.debug.print("\n\n+-------------------------------------------+ ", .{});
        std.debug.print("\n   Your model do not contains intermediate tensor shapes,\n   run ' python3 src/onnx/infer_shape.py --path {s} '", .{file_path});
        std.debug.print("\n+-------------------------------------------+ \n\n", .{});

        unreachable;
    }

    return model;
}

// calculates the size in bytes of a tensor based on its data type and dimensions
pub fn tensorSizeInBytes(tensor: *const TensorProto) usize {
    var num_elements: usize = 1;
    for (tensor.dims) |dim| {
        num_elements *= @intCast(dim);
    }

    const type_size: usize = switch (tensor.data_type) {
        .UINT8, .INT8, .BOOL, .FLOAT8E4M3FN, .FLOAT8E4M3FNUZ, .FLOAT8E5M2, .FLOAT8E5M2FNUZ => 1, // 1-byte types
        .UINT16, .INT16, .FLOAT16, .BFLOAT16 => 2, // 2-byte types
        .FLOAT, .INT32, .UINT32 => 4, // 4-byte types
        .DOUBLE, .INT64, .UINT64, .COMPLEX64 => 8, // 8-byte types
        .COMPLEX128 => 16, // 16-byte types
        else => {
            std.debug.print("Warning: Unknown data type {} in tensor, assuming 4 bytes per element\n", .{tensor.data_type});
            return 4 * num_elements; // Default to 4 bytes for unknown types
        },
    };

    return num_elements * type_size;
}

pub fn printModelDetails(model: *const ModelProto) !void {
    const stdout = std.debug;

    // basic model's informations
    stdout.print("\n=========== ONNX Model Details ===========\n", .{});
    stdout.print("Model version: {}\n", .{model.ir_version});
    stdout.print("Producer: {s}\n", .{model.producer_name orelse "Unknown"});

    // graph informations
    if (model.graph) |graph| {
        stdout.print("\nGraph Statistics:\n", .{});
        stdout.print("  Number of nodes: {}\n", .{graph.nodes.len});

        // operator count
        var op_counts = std.StringHashMap(usize).init(std.heap.page_allocator);
        defer op_counts.deinit();
        for (graph.nodes) |node| {
            const op_type = node.op_type;
            const count = op_counts.get(op_type) orelse 0;
            try op_counts.put(op_type, count + 1);
        }
        stdout.print("  Operator distribution:\n", .{});
        var op_iter = op_counts.iterator();
        while (op_iter.next()) |entry| {
            stdout.print("    {s}: {}\n", .{ entry.key_ptr.*, entry.value_ptr.* });
        }

        // Tensors and weights
        var tensor_count: usize = 0;
        for (graph.initializers) |_| tensor_count += 1;
        for (graph.inputs) |_| tensor_count += 1;
        for (graph.outputs) |_| tensor_count += 1;

        var total_weight_size: usize = 0;
        for (graph.initializers) |tensor| {
            total_weight_size += tensorSizeInBytes(tensor);
        }

        stdout.print("\nMemory Requirements:\n", .{});
        stdout.print("  Total tensors: {}\n", .{tensor_count});
        stdout.print("  Total weight size: {} bytes ({d:.2} MB)\n", .{ total_weight_size, @as(f32, @floatFromInt(total_weight_size)) / (1024.0 * 1024.0) });
    } else {
        stdout.print("\nWARNING: No graph found in the model.\n", .{});
    }

    stdout.print("=========================================\n", .{});
}
