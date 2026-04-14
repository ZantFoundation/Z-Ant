//! ONNX protobuf parser and model representation for Zant.
//!
//! Implements a hand-written protobuf decoder that supports IR_VERSION_2024_3_25.
//! Exposes all ONNX proto types (`ModelProto`, `GraphProto`, `NodeProto`,
//! `TensorProto`, etc.) as Zig structs, along with:
//! - `parseFromFile`: load and decode a `.onnx` file from disk.
//! - `DataType`/`Version`: ONNX-spec enumerations.
//! - `OnnxOperator`/`fromString`/`isQlinear`: operator classification helpers.
//!
//! See the ONNX proto spec: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
const std = @import("std");
const protobuf = @import("onnx/protobuf.zig");

pub const ValueInfoProto = @import("onnx/valueInfoProto.zig").ValueInfoProto;
pub const AttributeProto = @import("onnx/attributeProto.zig").AttributeProto;
pub const TensorShapeProto = @import("onnx/tensorShapeProto.zig").TensorShapeProto;
pub const TypeProto = @import("onnx/typeProto.zig").TypeProto;
pub const TensorProto = @import("onnx/tensorProto.zig").TensorProto;
pub const NodeProto = @import("onnx/nodeProto.zig").NodeProto;
pub const GraphProto = @import("onnx/graphProto.zig").GraphProto;
pub const ModelProto = @import("onnx/modelProto.zig").ModelProto;
pub const StringStringEntryProto = @import("onnx/stringStringEntryProto.zig").StringStringEntryProto;
pub const OperatorSetIdProto = @import("onnx/operatorSetIdProto.zig").OperatorSetIdProto;
pub const FunctionProto = @import("onnx/functionProto.zig").FunctionProto;
// --- onnx operator ---
pub const OnnxOperator = @import("onnx/onnxOperator.zig").OnnxOperator;
pub const fromString = @import("onnx/onnxOperator.zig").fromString;
pub const isQlinear = @import("onnx/onnxOperator.zig").isQlinear;

// --- shared types ---
const types = @import("onnx/types.zig");
pub const Version = types.Version;
pub const DataType = types.DataType;
pub const DataLocation = types.DataLocation;
pub const AttributeType = types.AttributeType;

const onnx_log = std.log.scoped(.onnx);

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
        std.debug.print("\n\n\n+-------------------------------------------+ ", .{});
        std.debug.print("\n   Your model do not contains intermediate tensor shapes,\n   run ' python3 src/onnx/shape_thief.py --model modelName '", .{});
        std.debug.print("\n+-------------------------------------------+ \n\n", .{});

        std.debug.print("\n\n+-------------------------------------------+ ", .{});
        std.debug.print("\n   Also ensure that the input shape is well known, otherwise: \n   run ' python3 src/onnx/input_setter.py --model modelName --shape B,C,H,W (eg., \"1,3,10,10\")'", .{});
        std.debug.print("\n+-------------------------------------------+ \n\n\n", .{});

        return error.UnsupportedWireType;
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
            onnx_log.warn("Warning: Unknown data type {} in tensor, assuming 4 bytes per element\n", .{tensor.data_type});
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
