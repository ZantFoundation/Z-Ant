const std = @import("std");
const IR_zant = @import("IR_zant");
const allocator = IR_zant.pkg_allocator.allocator;

// --- zant IR---
const NodeZant = NodeZant_lib.NodeZant;
const NodeZant_lib = IR_zant.NodeZant_lib;
const TensorZant = tensorZant_lib.TensorZant;
const TensorCategory = tensorZant_lib.TensorCategory;
const tensorZant_lib = IR_zant.tensorZant_lib;
const GraphZant = IR_zant.GraphZant;

// --- union ---
const operators = IR_zant.operators;
const Op_union = @import("../../op_union.zig").Op_union;

/// Fused Conv+Sigmoid+Mul operation for better performance
/// This combines convolution followed by a Sigmoid function and a Mul.
/// Known also as: Attention gate or Sqeeze and Extraction
pub const Fused_Conv_Sigmoid_Mul = struct {
    op_name: []const u8,
    op_Conv: operators.Conv,
    op_Sigmoid: operators.Sigmoid,
    op_Mul: operators.Mul,

    //inizialization logic for the new operation given a list of old nodes
    pub fn init_fused_op(fusion_list: std.ArrayList(*NodeZant)) !Fused_Conv_Sigmoid_Mul {
        if (fusion_list.items.len != 3) return error.WrongNumberOfElements;
        if (fusion_list.items[0].op != .conv) return error.WrongOpAtPose0;
        if (fusion_list.items[1].op != .sigmoid) return error.WrongOpAtPose1;
        if (fusion_list.items[2].op != .mul) return error.WrongOpAtPose2;

        // Extract the specific operations from the unions
        const conv_op = switch (fusion_list.items[0].op) {
            .conv => |c| c,
            else => return error.InvalidConvOperation,
        };

        const sigmoid_op = switch (fusion_list.items[1].op) {
            .sigmoid => |s| s,
            else => return error.InvalidSigmoidOperation,
        };

        const mul_op = switch (fusion_list.items[2].op) {
            .mul => |m| m,
            else => return error.InvalidMulOperation,
        };

        // Downgrade LINK tensors between fudes noted to FUSED_LINK tensors
        conv_op.output_Y.set_tensorCategory(TensorCategory.FUSED_LINK);
        sigmoid_op.output_Y.set_tensorCategory(TensorCategory.FUSED_LINK);

        return Fused_Conv_Sigmoid_Mul{
            .op_name = try NodeZant_lib.getFusedOpsName(fusion_list),
            .op_Conv = conv_op,
            .op_Sigmoid = sigmoid_op,
            .op_Mul = mul_op,
        };
    }

    pub fn fn_pattern_detection(_: *GraphZant, root_node: *NodeZant) anyerror!?std.ArrayList(*NodeZant) {
        std.debug.print("\n  Checking Conv_Sigmoid_Mul pattern from node: {s}", .{root_node.op_type});

        if (root_node.op != .conv) {
            return null;
        }

        var node_list: std.ArrayList(*NodeZant) = .empty;
        errdefer node_list.deinit(allocator); // Clean up on error

        try node_list.append(allocator, root_node);
        std.debug.print(" -> Conv node found, checking for Sigmoid/Mul successors", .{});

        // Check if Conv has exactly two successors
        const next_nodes = root_node.next;
        if (next_nodes.items.len != 2) {
            // std.debug.print(" -> Failed: Conv node does not have exactly 2 successors ({d})", .{next_nodes.items.len});
            node_list.deinit(allocator);
            return null;
        }

        // Check if the two successors are Sigmoid and Mul, in any order
        const node_a = next_nodes.items[0];
        const node_b = next_nodes.items[1];

        var sigmoid_node: ?*NodeZant = null;
        var mul_node: ?*NodeZant = null;

        if (node_a.op == .sigmoid and node_b.op == .mul) {
            sigmoid_node = node_a;
            mul_node = node_b;
        } else if (node_a.op == .mul and node_b.op == .sigmoid) {
            sigmoid_node = node_b;
            mul_node = node_a;
        }

        if (sigmoid_node == null or mul_node == null) {
            // std.debug.print(" -> Failed: Successors are not Sigmoid and Mul (found {s} and {s})", .{
            //     node_a.op_type,
            //     node_b.op_type,
            // });
            node_list.deinit(allocator);
            return null;
        }

        std.debug.print(" -> Found Conv_Sigmoid_Mul pattern!", .{});
        try node_list.append(allocator, sigmoid_node.?);
        try node_list.append(allocator, mul_node.?);

        return node_list;
    }

    pub fn fn_pattern_fusion(_: *GraphZant, node_list: std.ArrayList(*NodeZant)) anyerror!NodeZant {
        if (node_list.items.len != 3) return error.InvalidNumberOfOps;

        // PATTERN_MATCHING_STRATEGY
        if (node_list.items[0].op != .conv) return error.UnexpectedOpAtPos0;
        if (node_list.items[1].op != .sigmoid) return error.UnexpectedOpAtPos1;
        if (node_list.items[2].op != .mul) return error.UnexpectedOpAtPos2;

        const sigmoid_node: *NodeZant = node_list.items[1];
        const mul_node: *NodeZant = node_list.items[2];

        //  Clone the next list instead of direct reference
        var cloned_next: std.ArrayList(*NodeZant) = .empty;
        for (sigmoid_node.next.items) |next_node| {
            try cloned_next.append(allocator, next_node);
        }
        for (mul_node.next.items) |next_node| {
            try cloned_next.append(allocator, next_node);
        }

        //  Clone the fusion_list instead of direct reference
        var cloned_fusion_list: std.ArrayList(*NodeZant) = .empty;
        for (node_list.items) |node| {
            try cloned_fusion_list.append(allocator, node);
        }

        return NodeZant{
            .name = try NodeZant_lib.getFusedOpsName(node_list),
            .op_type = try NodeZant_lib.getFusedOpsType(node_list),
            .op = Op_union{ .fused_Conv_Sigmoid_Mul = try init_fused_op(node_list) },
            .next = cloned_next,
            .nodeProto = null,
            .ready = false,
            .fusion_list = cloned_fusion_list,
        };
    }

    pub fn fn_pattern_sobstitution(graph: *GraphZant, fused_node: *NodeZant, node_list: std.ArrayList(*NodeZant)) anyerror!void {
        if (node_list.items.len == 0) return error.EmptyNodeList;
        if (node_list.items.len != 3) return error.InvalidPatternLength; // For Conv+Sigmoid+Mul pattern

        const first_node = node_list.items[0]; // Conv node
        const branch_node_1 = node_list.items[1]; // Sigmoid node
        const branch_node_2 = node_list.items[2]; // Mul node

        // Step 1: Find all predecessor nodes that point to the first node in the pattern
        const predecessors = try graph.get_predecessors(first_node);

        // Step 2: Update predecessor nodes to point to the fused_node instead of first_node
        for (predecessors.items) |predecessor| {
            // Find and replace all references to first_node with fused_node
            for (predecessor.next.items, 0..) |next_node, i| {
                if (next_node == first_node) {
                    predecessor.next.items[i] = fused_node;
                    // Don't break here - there might be multiple edges to the same node
                }
            }
        }

        // Step 3: Ensure fused_node.next points to the correct successors
        // This should already be set in fn_pattern_fusion, but verify
        if (fused_node.next.items.len == 0) {
            // Copy the branch node's successors to the fused node
            for (branch_node_1.next.items) |successor| {
                try fused_node.next.append(allocator, successor);
            }
            for (branch_node_2.next.items) |successor| {
                try fused_node.next.append(allocator, successor);
            }
        }

        // Step 4: Remove old nodes from graph
        try graph.removeNodes(node_list);

        // Step 5: Add the fused_node to the graph's node list
        try graph.nodes.append(allocator, fused_node);

        // //This is a delicate step, read carrefully!!
        // //for each successor sobtitute the input equal to old last_node output wiht the new output of the fusion
        // //OSS: in this case the output of the fusion is the output of the new node!
        // const prefusion_last_node_outputs: []*TensorZant = try last_node.get_output_tensors();
        // // assuming that the putput of the fusion is only one node:
        // const post_fusion_output: *TensorZant = (try fused_node.get_output_tensors())[0];
        // const successors = last_node.next;
        //
        // for (successors.items) |succ_node| { //for each succerssor nodes
        //     const inputs = succ_node.get_input_tensors(); // collect its inputs

        //     //for each succ input:
        //     //if the input is equal to an old last_node output sobstitute it with the new output
        //     for (inputs) |succ_input| {
        //         for (prefusion_last_node_outputs) |old_out| {
        //             if (succ_input == old_out) succ_node.sobstitute_tensors(old_out, post_fusion_output);
        //         }
        //     }
        // }
    }
};
