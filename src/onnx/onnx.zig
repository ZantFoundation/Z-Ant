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
