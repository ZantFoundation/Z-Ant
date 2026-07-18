//! `NodeZant` — a single computation node in the Zant IR graph.
//!
//! Wraps an ONNX `NodeProto` and resolves it to a concrete `Op_union` variant
//! (one of ~60 supported operators). Stores adjacency information (`next` list)
//! for graph traversal. Fused nodes are created by the pattern-matcher and carry
//! `is_fused = true` to signal that their `Op_union` is a compound kernel.
const std = @import("std");
const IR_zant = @import("IR_zant");

const onnx = @import("onnx.zig");
const allocator = IR_zant.pkg_allocator.allocator;
const Tensor = IR_zant.core.tensor.Tensor;
const Op_union = @import("op_union/op_union.zig").Op_union;

//--- proto structure
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;
const DataType = onnx.DataType;

//--- zant structures
const TensorZant = @import("tensorZant.zig").TensorZant;

pub const NodeZant = struct {
    name: ?[]const u8, //name of the node
    op_type: []const u8, //onnx name of the operation, see here: https://onnx.ai/onnx/operators/
    op: Op_union, // contains all the information of the operation
    next: std.ArrayList(*NodeZant), // points to the following nodes

    is_fused: bool, // This is a list of all the operations fused in this node, by deafult is null. It is only different from Null when the node represent the fusion of two or more nodes. A fused node is created in pattern_matcher.fuse_nodes().
    nodeProto: ?*NodeProto,
    ready: bool,

    /// Initializes a NodeZant instance starting from a NodeProto instance.
    pub fn init(nodeProto: *NodeProto) !NodeZant {
        return NodeZant{
            .name = if (nodeProto.name) |n| n else "unnamed",
            .op_type = @tagName(nodeProto.op_type),
            .op = try Op_union.init(nodeProto),
            .next = .empty,
            .nodeProto = nodeProto,
            .ready = false,
            .is_fused = false,
        };
    }

    /// Deinitializes a NodeZant instance, freeing allocated resources.
    pub fn deinit(self: *NodeZant) void {
        // dam
        // const name = if (self.name) |n| n else "<unnamed>";
        // std.debug.print("\n    {s}.deinit()  ", .{name});
        self.next.deinit(allocator);
    }

    /// Adds a new node to the next list.
    pub fn add_next(self: *NodeZant, next_node: *NodeZant) !void {
        try self.next.append(allocator, next_node);
    }

    pub fn write_op(self: *NodeZant, writer: *std.Io.Writer) !void {
        try self.op.write_op(writer);
    }

    pub fn get_output_tensors(self: *NodeZant) ![]*TensorZant {
        return try self.op.get_output_tensors();
    }

    pub fn get_input_tensors(self: *NodeZant) ![]*TensorZant {
        return try self.op.get_input_tensors();
    }

    pub fn sobstitute_tensors(self: *NodeZant, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        try self.op.sobstitute_tensors(old_tensor, new_tensor);
    }

    pub fn isFused(self: *NodeZant) bool {
        return !self.is_fused == null;
    }

    // Print method for debugging and display purposes
    pub fn print(self: *const NodeZant) void {
        std.debug.print("NodeZant {{\n", .{});

        // Print name
        if (self.name) |name| {
            std.debug.print("  name: \"{s}\"\n", .{name});
        } else {
            std.debug.print("  name: null\n", .{});
        }

        // Print operation type
        std.debug.print("  op_type: \"{s}\"\n", .{self.op_type});

        // Print operation union (simplified - you may want to expand this based on your Op_union variants)
        std.debug.print("  Op_union: {s}\n", .{@tagName(self.op)});

        // Print next nodes count
        std.debug.print("  next: [{}] nodes\n", .{self.next.items.len});
        // for (self.next.items) |next_node| {
        //     next_node.print();
        // }

        // Print fusion list info
        std.debug.print("  is_fused: {}\n", .{self.is_fused});

        // Print nodeProto status
        if (self.nodeProto) |_| {
            std.debug.print("  nodeProto: *NodeProto\n", .{});
        } else {
            std.debug.print("  nodeProto: null\n", .{});
        }

        // Print ready status
        std.debug.print("  ready: {}\n", .{self.ready});

        std.debug.print("}}\n", .{});
    }
};

pub fn getFusedOpsName(fusion_list: std.ArrayList(*NodeZant)) ![]const u8 {
    // Create concatenated name using a string buffer
    var name_buffer: std.ArrayList(u8) = .empty;
    defer name_buffer.deinit(allocator);

    for (fusion_list.items, 0..) |node, i| {
        if (node.name) |node_name| {
            try name_buffer.appendSlice(allocator, node_name);
        } else {
            try name_buffer.appendSlice(allocator, "unnamed");
        }
        // Add separator between names (except for the last one)
        if (i < fusion_list.items.len - 1) {
            try name_buffer.append(allocator, '_');
        }
    }

    return try allocator.dupe(u8, name_buffer.items);
}

pub fn getFusedOpsType(fusion_list: std.ArrayList(*NodeZant)) ![]const u8 {
    // Create concatenated op_type with "fused_" prefix
    var op_type_buffer: std.ArrayList(u8) = .empty;
    defer op_type_buffer.deinit(allocator);

    try op_type_buffer.appendSlice(allocator, "fused");
    for (fusion_list.items) |node| {
        try op_type_buffer.append(allocator, '_');
        try op_type_buffer.appendSlice(allocator, @tagName(node.op));
    }

    return try allocator.dupe(u8, op_type_buffer.items);
}
