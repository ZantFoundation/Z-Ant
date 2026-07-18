//! Shared utilities for fusion pattern detection tests.
//! Provides helpers to build minimal IR graphs without ONNX data.
const std = @import("std");
const IR_zant = @import("IR_zant");

pub const allocator = IR_zant.pkg_allocator.allocator;
pub const NodeZant = IR_zant.NodeZant_lib.NodeZant;
pub const GraphZant = IR_zant.GraphZant;
pub const Op_union = IR_zant.Op_union;
pub const operators = IR_zant.operators;
pub const tensorZant_lib = IR_zant.tensorZant_lib;
pub const TensorZant = tensorZant_lib.TensorZant;

/// Creates a heap-allocated NodeZant with the given op tag.
/// The op payload is zero-initialized — only the union tag matters for detection.
/// Caller must free via destroyTestNode or graph.deinit.
pub fn createNode(name: []const u8, op: Op_union) !*NodeZant {
    const node = try allocator.create(NodeZant);
    node.* = NodeZant{
        .name = name,
        .op_type = @tagName(op),
        .op = op,
        .next = .empty,
        .nodeProto = null,
        .ready = false,
        .is_fused = false,
    };
    return node;
}

/// Creates a dummy TensorZant and registers it in the global tensorMap.
/// Returns a stable pointer from the map.
pub fn dummyTensor(name: []const u8) !*TensorZant {
    const result = try tensorZant_lib.tensorMap.getOrPut(name);
    if (!result.found_existing) {
        result.value_ptr.* = TensorZant{
            .name = name,
            .ty = .f32,
            .tc = .LINK,
            .ptr = null,
            .shape = &.{},
            .stride = &.{},
        };
    }
    return result.value_ptr;
}

/// Creates a minimal test graph (no GraphProto needed).
pub fn createGraph() GraphZant {
    return GraphZant{
        .name = "test_graph",
        .nodes = .empty,
        .graphProto = undefined,
    };
}

/// Wires node_a -> node_b (adds node_b to node_a's next list).
pub fn wire(from: *NodeZant, to: *NodeZant) !void {
    try from.next.append(allocator, to);
}

/// Adds a node to the graph's node list.
pub fn addToGraph(graph: *GraphZant, node: *NodeZant) !void {
    try graph.nodes.append(allocator, node);
}

/// Destroys a single test node (frees next list + node memory).
pub fn destroyTestNode(node: *NodeZant) void {
    node.next.deinit(allocator);
    allocator.destroy(node);
}

/// Destroys all nodes in a graph and the graph's node list.
pub fn destroyTestGraph(graph: *GraphZant) void {
    for (graph.nodes.items) |node| {
        destroyTestNode(node);
    }
    graph.nodes.deinit(allocator);
}

// ---- Operator constructors (minimal, for detection tests only) ----

pub fn opConv() !Op_union {
    const t = try dummyTensor("_conv_dummy");
    return Op_union{ .conv = .{
        .input_X = t,
        .input_W = t,
        .input_B = null,
        .output_Y = t,
        .auto_pad = "NOTSET",
        .dilations = null,
        .group = 1,
        .kernel_shape = null,
        .pads = null,
        .strides = null,
    } };
}

pub fn opRelu() !Op_union {
    const t = try dummyTensor("_relu_dummy");
    return Op_union{ .relu = .{ .input_X = t, .output_Y = t } };
}

pub fn opClip() !Op_union {
    const t = try dummyTensor("_clip_dummy");
    return Op_union{ .clip = .{ .input = t, .min = null, .max = null, .output = t } };
}

pub fn opSigmoid() !Op_union {
    const t = try dummyTensor("_sigmoid_dummy");
    return Op_union{ .sigmoid = .{ .input_X = t, .output_Y = t } };
}

pub fn opMul() !Op_union {
    const t = try dummyTensor("_mul_dummy");
    return Op_union{ .mul = .{ .input_A = t, .input_B = t, .output_C = t } };
}

pub fn opAdd() !Op_union {
    const t = try dummyTensor("_add_dummy");
    return Op_union{ .add = .{ .input_A = t, .input_B = t, .output_C = t } };
}

pub fn opPad() !Op_union {
    const t = try dummyTensor("_pad_dummy");
    return Op_union{ .pad = .{
        .input_data = t,
        .input_pads = t,
        .input_constant_value = null,
        .input_axes = null,
        .output = t,
        .mode = "constant",
    } };
}

pub fn opDequantizeLinear() !Op_union {
    const t = try dummyTensor("_dequant_dummy");
    return Op_union{ .dequantizeLinear = .{
        .x = t,
        .x_scale = t,
        .x_zero_point = null,
        .y = t,
        .axis = 0,
        .block_size = 0,
        .output_dtype = .undefined,
    } };
}

pub fn opQuantizeLinear() !Op_union {
    const t = try dummyTensor("_quant_dummy");
    return Op_union{ .quantizeLinear = .{
        .x = t,
        .y_scale = t,
        .y_zero_point = null,
        .y = t,
        .axis = 0,
        .block_size = 0,
        .output_dtype = .undefined,
        .precision = 0,
        .saturate = 1,
    } };
}

pub fn opQLinearConv() !Op_union {
    const t = try dummyTensor("_qlinconv_dummy");
    return Op_union{ .qlinearconv = .{
        .input_x = t,
        .input_x_scale = t,
        .input_x_zero_point = t,
        .input_w = t,
        .input_w_scale = t,
        .input_w_zero_point = t,
        .input_y_scale = t,
        .input_y_zero_point = t,
        .input_B = null,
        .output_y = t,
        .auto_pad = "NOTSET",
        .dilations = null,
        .group = 1,
        .kernel_shape = null,
        .pads = null,
        .strides = null,
    } };
}
