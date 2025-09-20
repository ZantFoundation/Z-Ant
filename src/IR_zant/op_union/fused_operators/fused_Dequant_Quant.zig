const std = @import("std");
const allocator = std.heap.page_allocator;
const zant = @import("zant");
const IR_zant = @import("../../IR_zant.zig");

// --- zant IR---
const tensorZant_lib = IR_zant.tensorZant_lib;
const TensorZant = tensorZant_lib.TensorZant;
const TensorCategory = tensorZant_lib.TensorCategory;
const NodeZant_lib = IR_zant.NodeZant_lib;
const NodeZant = NodeZant_lib.NodeZant;
const GraphZant = IR_zant.GraphZant;
const IR_utils = IR_zant.utils;

// --- union ---
const Op_union = @import("../op_union.zig").Op_union;
const operators = IR_zant.operators;

pub const Fused_Dequant_Quant = struct {
    op_name: []const u8,

    // This method is not used since is initialized as a "useless" operator
    pub fn init_fused_op(fusion_list: std.ArrayList(*NodeZant)) !Fused_Dequant_Quant {
        _ = fusion_list;
    }

    /// Pattern detection function for DequantizeLinear -> QuantizeLinear
    pub fn fn_pattern_detection(graph: *GraphZant, root_node: *NodeZant) anyerror!?std.ArrayList(*NodeZant) {
        _ = graph; // Not used in this sequential pattern

        // Only start detection from DequantizeLinear nodes
        if (!std.mem.eql(u8, root_node.op_type, "DequantizeLinear")) {
            std.debug.print(" -> Not a DequantizeLinear node, skipping", .{});
            return null;
        }

        var node_list = std.ArrayList(*NodeZant).init(allocator);
        errdefer node_list.deinit();

        try node_list.append(root_node);
        std.debug.print(" -> DequantizeLinear node found, checking for QuantizeLinear successor", .{});

        // Check DequantizeLinear -> QuantizeLinear
        if (root_node.next.items.len != 1) {
            std.debug.print(" -> DequantizeLinear has {} successors (expected 1)", .{root_node.next.items.len});
            node_list.deinit();
            return null;
        }

        const quant_node = root_node.next.items[0];
        if (!std.mem.eql(u8, quant_node.op_type, "QuantizeLinear")) {
            std.debug.print(" -> DequantizeLinear successor is {s} (expected QuantizeLinear)", .{quant_node.op_type});
            node_list.deinit();
            return null;
        }

        try node_list.append(quant_node);

        return node_list;
    }

    /// Pattern fusion function
    pub fn fn_pattern_fusion(graph: *GraphZant, node_list: std.ArrayList(*NodeZant)) anyerror!NodeZant {
        _ = graph; // Not used in this sequential pattern

        // Validate the pattern
        if (node_list.items.len != 2) return error.InvalidNumberOfOps;
        if (!std.mem.eql(u8, node_list.items[0].op_type, "DequantizeLinear")) return error.UnexpectedOpAtPos0;
        if (!std.mem.eql(u8, node_list.items[1].op_type, "QuantizeLinear")) return error.UnexpectedOpAtPos1;

        const last_node = node_list.items[1]; // QuantizeLinear node

        // Clone the next list instead of direct reference
        var cloned_next = std.ArrayList(*NodeZant).init(allocator);
        for (last_node.next.items) |next_node| {
            try cloned_next.append(next_node);
        }

        //  Clone the fusion_list instead of direct reference
        var cloned_fusion_list = std.ArrayList(*NodeZant).init(allocator);
        for (node_list.items) |node| {
            try cloned_fusion_list.append(node);
        }

        return NodeZant{
            .name = try NodeZant_lib.getFusedOpsName(node_list),
            .op_type = try NodeZant_lib.getFusedOpsType(node_list),
            .op = Op_union{ .useless = operators.Useless{} },
            .next = cloned_next,
            .nodeProto = null,
            .ready = false,
            .fusion_list = null,
        };
    }

    /// Pattern substitution function
    pub fn fn_pattern_sobstitution(graph: *GraphZant, fused_node: *NodeZant, node_list: std.ArrayList(*NodeZant)) anyerror!void {
        _ = fused_node; //the fuses node is totally useless since I removed the pattern completelly

        // Validate inputs
        if (node_list.items.len != 2) return error.InvalidPatternLength;

        const first_node = node_list.items[0]; // DequantizeLinear node
        const last_node = node_list.items[1]; // QuantizeLinear node

        // Step 1: Find all predecessor nodes that point to the first node
        const predecessors = try graph.get_predecessors(first_node);
        const successors = last_node.next;

        // Step 2: Update predecessor nodes to point to the output of the last node
        for (predecessors.items) |predecessor| {
            for (predecessor.next.items, 0..) |next_node, i| {
                if (next_node == first_node) {
                    predecessor.next.items[i] = last_node.next.items[i];
                }
            }
        }

        // Step 4: Remove old nodes from graph
        try graph.removeNodes(node_list);

        //This is a delicate step, read carrefully!!
        //for each successor sobtitute the input equal to old last_node output wiht the new output of the fusion
        //OSS: in this case the output of the fusion is the output of the predecessor since we don't grate any new node
        const prefusion_last_node_outputs: []*TensorZant = try last_node.get_output_tensors();
        // assuming that the putput of the fusion is only one node:
        const post_fusion_output: *TensorZant = (try predecessors.items[0].get_output_tensors())[0];

        for (successors.items) |succ_node| { //for each succerssor nodes
            const inputs = try succ_node.get_input_tensors(); // collect its inputs

            //for each succ input:
            //if the input is equal to an old last_node output sobstitute it with the new output
            for (inputs) |succ_input| {
                for (prefusion_last_node_outputs) |old_out| {
                    if (succ_input == old_out) try succ_node.sobstitute_tensors(old_out, post_fusion_output);
                }
            }
        }
    }

    // Helper functions matching the interface

    pub fn get_output_shape(self: Fused_Dequant_Quant) []usize {
        _ = self;
        return &[_]usize{};
    }

    pub fn get_input_tensors(self: Fused_Dequant_Quant) anyerror![]*TensorZant {
        _ = self;
        return error.ThisIsUseless;
    }

    pub fn get_output_tensors(self: Fused_Dequant_Quant) anyerror![]*TensorZant {
        _ = self;
        return error.ThisIsUseless;
    }

    pub fn write_op(self: Fused_Dequant_Quant, writer: std.fs.File.Writer) !void {
        _ = self;
        _ = writer;
        return error.ThisIsUseless;
    }

    pub fn compute_output_shape(self: Fused_Dequant_Quant) []usize {
        _ = self;
        return &[_]usize{};
    }

    pub fn print(self: Fused_Dequant_Quant) void {
        _ = self;
        return &[_]usize{};
    }
};
