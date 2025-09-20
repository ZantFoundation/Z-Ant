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
const IR_utils = IR_zant.utils; //this is IR utils

// --- union ---
const Op_union = @import("../op_union.zig").Op_union;
const operators = IR_zant.operators;

pub const Fused_Conv_Relu = struct {
    op_name: []const u8,
    op_Conv: operators.Conv, // Use the actual Conv type
    op_Relu: operators.Relu, // Use the actual Relu type

    //inizialization logic for the new operation given a list of old nodes
    pub fn init_fused_op(fusion_list: std.ArrayList(*NodeZant)) !Fused_Conv_Relu {
        //Ensure that the ArrayList is the correct one
        if (fusion_list.items.len != 2) return error.WrongNumberOfElements;
        if (!std.mem.eql(u8, fusion_list.items[0].op_type, "Conv")) return error.WrongOpAtPose0;
        if (!std.mem.eql(u8, fusion_list.items[1].op_type, "Relu")) return error.WrongOpAtPose2;

        // Extract the specific operations from the unions
        const conv_op = switch (fusion_list.items[0].op) {
            .conv => |c| c,
            else => return error.InvalidConvOperation,
        };

        const relu_op = switch (fusion_list.items[1].op) {
            .relu => |r| r,
            else => return error.InvalidReluOperation,
        };

        return Fused_Conv_Relu{
            .op_name = try NodeZant_lib.getFusedOpsName(fusion_list),
            .op_Conv = conv_op,
            .op_Relu = relu_op,
        };
    }

    pub fn fn_pattern_detection(graph: *GraphZant, root_node: *NodeZant) anyerror!?std.ArrayList(*NodeZant) {
        _ = graph; // Not used in this sequential pattern

        std.debug.print("\n  Checking pattern from node: {s}", .{root_node.op_type});

        // CRITICAL FIX: Only start detection from Conv nodes
        if (!std.mem.eql(u8, root_node.op_type, "Conv")) {
            // std.debug.print(" -> Not a Conv node, skipping", .{});
            return null;
        }

        var node_list = std.ArrayList(*NodeZant).init(allocator);
        errdefer node_list.deinit(); // Clean up on error

        try node_list.append(root_node);
        std.debug.print(" -> Conv node found, checking for Relu successor", .{});

        // Check if Conv has exactly one successor and it's ReLU
        const next_nodes = root_node.next;
        if (next_nodes.items.len != 1) {
            // std.debug.print(" -> Conv has {} successors (expected 1)", .{next_nodes.items.len});
            node_list.deinit();
            return null;
        }

        const successor = next_nodes.items[0];
        if (!std.mem.eql(u8, successor.op_type, "Relu")) {
            // std.debug.print(" -> Conv successor is {s} (expected Relu)", .{successor.op_type});
            node_list.deinit();
            return null;
        }

        std.debug.print(" -> Found Conv->Relu pattern!", .{});
        try node_list.append(successor);
        return node_list;
    }

    pub fn fn_pattern_fusion(graph: *GraphZant, node_list: std.ArrayList(*NodeZant)) anyerror!NodeZant {
        _ = graph; // in this case graph is not used since the pattern is sequential

        //checks
        if (node_list.items.len != 2) return error.InvalidNumberOfOps;

        // PATTERN_MATCHING_STRATEGY
        if (!std.mem.eql(u8, node_list.items[0].op_type, "Conv")) return error.UnexpectedOpAtPos0;
        if (!std.mem.eql(u8, node_list.items[1].op_type, "Relu")) return error.UnexpectedOpAtPos1;

        const relu_node: *NodeZant = node_list.items[1];

        //  Clone the next list instead of direct reference
        var cloned_next = std.ArrayList(*NodeZant).init(allocator);
        for (relu_node.next.items) |next_node| {
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
            .op = Op_union{ .fused_Conv_Relu = try init_fused_op(node_list) },
            .next = cloned_next,
            .nodeProto = null,
            .ready = false,
            .fusion_list = cloned_fusion_list,
        };
    }

    pub fn fn_pattern_sobstitution(graph: *GraphZant, fused_node: *NodeZant, node_list: std.ArrayList(*NodeZant)) anyerror!void {

        // Validate inputs
        if (node_list.items.len == 0) return error.EmptyNodeList;
        if (node_list.items.len != 2) return error.InvalidPatternLength; // For Conv+Relu pattern

        const first_node = node_list.items[0]; // Conv node
        const last_node = node_list.items[1]; // Relu node

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
            // Copy the last node's successors to the fused node
            for (last_node.next.items) |successor| {
                try fused_node.next.append(successor);
            }
        }

        // Step 4: Remove old nodes from graph
        try graph.removeNodes(node_list);

        // Step 5: Add the fused_node to the graph's node list
        try graph.nodes.append(fused_node);

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

    pub fn get_output_shape(self: Fused_Conv_Relu) []usize {
        return self.op_Relu.get_output_shape();
    }

    pub fn get_input_tensors(self: Fused_Conv_Relu) anyerror![]*TensorZant {
        return try self.op_Conv.get_input_tensors();
    }

    pub fn get_output_tensors(self: Fused_Conv_Relu) anyerror![]*TensorZant {
        return try self.op_Relu.get_output_tensors();
    }

    pub fn write_op(self: Fused_Conv_Relu, writer: std.fs.File.Writer) !void {
        _ = try writer.print(
            \\
            \\    //{s}
            \\    tensMath.Fused_Conv_Relu_lean(..., &tensor_{s}) catch return -1;
        , .{
            "fused Conv -> Relu",
            try IR_utils.getSanitizedName(self.op_Relu.output_Y.name), // Output tensor
        });
    }

    pub fn compute_output_shape(self: Fused_Conv_Relu) []usize {
        return self.op_Relu.compute_output_shape();
    }

    pub fn print(self: Fused_Conv_Relu) void {
        std.debug.print("\n Fused_Conv_Relu:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *Fused_Conv_Relu, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        // Try to substitute in the Conv operation
        self.op_Conv.sobstitute_tensors(old_tensor, new_tensor) catch {
            // If not found in Conv, try Relu operation
            return try self.op_Relu.sobstitute_tensors(old_tensor, new_tensor);
        };
    }
};
