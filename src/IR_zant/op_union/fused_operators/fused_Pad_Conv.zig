const std = @import("std");
const zant = @import("zant");
const allocator = zant.utils.allocator.allocator;
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

pub const Fused_Pad_Conv = struct {
    op_name: []const u8,
    op_Pad: operators.Pad,
    op_Conv: operators.Conv,
    // The resulting fused operation
    fused_pad_conv: operators.Conv,

    pub fn init_fused_op(fusion_list: std.ArrayList(*NodeZant)) !Fused_Pad_Conv {
        if (fusion_list.items.len != 2) return error.WrongNumberOfElements;
        if (fusion_list.items[0].op != .pad) return error.WrongOpAtPos0;
        if (fusion_list.items[1].op != .conv) return error.WrongOpAtPos1;

        const pad_op = switch (fusion_list.items[0].op) {
            .pad => |p| p,
            else => return error.InvalidPadOperation,
        };

        const conv_op = switch (fusion_list.items[1].op) {
            .conv => |c| c,
            else => return error.InvalidConvOperation,
        };

        // Initialize the new Conv operation node, with:
        // input_data: pad_op.input
        // input_weight: conv_op.input_weight
        // output: conv_op.output
        var fused_padconv = conv_op;
        fused_padconv.input_X = pad_op.input_data;

        if (pad_op.input_pads.ptr) |pad_data_AnyTensor| {
            // Get existing pads from Conv (should be initialized to zeros in Conv.init)
            var existing_pads: [4]i64 = .{ 0, 0, 0, 0 };
            if (conv_op.pads) |conv_pads| {
                // Conv pads format: [h_begin, w_begin, h_end, w_end] for 2D
                const pads_to_copy = @min(conv_pads.len, 4);
                for (0..pads_to_copy) |i| {
                    existing_pads[i] = conv_pads[i];
                }
            }

            // Extract pad values from Pad operation
            var pad_values: [4]i64 = .{ 0, 0, 0, 0 };
            if (pad_op.input_pads.shape.len > 0) {
                switch (pad_op.input_pads.ty) {
                    .i64 => {
                        const pad_i64 = pad_data_AnyTensor.get_data_as(i64);
                        const pad_len = pad_op.input_pads.shape[0];

                        // ONNX Pad format for NCHW: [N_begin, C_begin, H_begin, W_begin, N_end, C_end, H_end, W_end]
                        if (pad_len >= 8) {
                            // Extract spatial padding only (H and W dimensions)
                            pad_values[0] = pad_i64[2]; // H_begin -> top
                            pad_values[1] = pad_i64[3]; // W_begin -> left
                            pad_values[2] = pad_i64[6]; // H_end -> bottom
                            pad_values[3] = pad_i64[7]; // W_end -> right
                        }
                    },
                    else => return error.UnsupportedPadDataType,
                }
            }

            // Create fused pads
            const final_pads = [4]i64{
                existing_pads[0] + pad_values[0], // top
                existing_pads[1] + pad_values[1], // left
                existing_pads[2] + pad_values[2], // bottom
                existing_pads[3] + pad_values[3], // right
            };

            pad_op.output.set_tensorCategory(TensorCategory.FUSED_LINK);

            // Allocate and set the fused pads
            const fused_pads = try allocator.alloc(i64, 4);
            @memcpy(fused_pads, &final_pads);
            fused_padconv.pads = fused_pads;

            std.debug.print("Fused padding: original={any}, pad_op={any}, final={any}\n", .{ existing_pads, pad_values, final_pads });
        }

        return Fused_Pad_Conv{
            .op_name = try NodeZant_lib.getFusedOpsName(fusion_list),
            .op_Pad = pad_op,
            .op_Conv = conv_op,
            .fused_pad_conv = fused_padconv,
        };
    }

    // Root_node is Pad, searching for Conv successor
    pub fn fn_pattern_detection(graph: *GraphZant, root_node: *NodeZant) anyerror!?std.ArrayList(*NodeZant) {
        _ = graph; // Not used, we only look at successors nodes

        // Start detection from Pad nodes
        if (root_node.op != .pad) return null;

        // Check that the only successor is Conv
        if (root_node.next.items.len != 1) return null;
        const conv_node = root_node.next.items[0];
        if (conv_node.op != .conv) return null;

        var node_list = std.ArrayList(*NodeZant).init(allocator);
        errdefer node_list.deinit();

        // Node_list: Pad -> Conv
        try node_list.append(root_node);
        try node_list.append(conv_node);

        std.debug.print(" -> Found Pad->Conv pattern!", .{});
        return node_list;
    }

    pub fn fn_pattern_fusion(graph: *GraphZant, node_list: std.ArrayList(*NodeZant)) anyerror!NodeZant {
        _ = graph; // Not used, we only look at successors nodes

        // Check the number of nodes
        if (node_list.items.len != 2) return error.InvalidNumberOfOps;

        // Check Pattern matching: Pad->Conv
        if (node_list.items[0].op != .pad) return error.UnexpectedOpAtPos0;
        if (node_list.items[1].op != .conv) return error.UnexpectedOpAtPos1;

        const last_node = node_list.items[1];
        // Clone the next list instead of direct reference
        var cloned_next = std.ArrayList(*NodeZant).init(allocator);
        // Take the successors of the last_node, to make them successors of the fused_node
        for (last_node.next.items) |next_node| {
            try cloned_next.append(next_node);
        }

        return NodeZant{
            .name = try NodeZant_lib.getFusedOpsName(node_list),
            .op_type = try NodeZant_lib.getFusedOpsType(node_list),
            .op = Op_union{ .fused_Pad_Conv = try init_fused_op(node_list) },
            .next = cloned_next,
            .nodeProto = null,
            .ready = false,
            .is_fused = true,
        };
    }

    pub fn fn_pattern_sobstitution(graph: *GraphZant, fused_node: *NodeZant, node_list: std.ArrayList(*NodeZant)) anyerror!void {
        if (node_list.items.len != 2) return error.InvalidPatternLength;

        const first_node = node_list.items[0]; // Pad node
        const last_node = node_list.items[1]; // Conv node

        // Find all predecessor nodes that point to the first node in the pattern
        const predecessors_first_node = try graph.get_predecessors(first_node);

        // Update predecessor nodes to point to the fused_node instead of first_node
        for (predecessors_first_node.items) |predecessor| {
            for (predecessor.next.items, 0..) |next_node, i| {
                if (next_node == first_node)
                    predecessor.next.items[i] = fused_node;
            }
        }

        // Set the successor nodes if it hasn't been done yet
        if (fused_node.next.items.len == 0) {
            for (last_node.next.items) |successor| {
                try fused_node.next.append(successor);
            }
        }

        try graph.removeNodes(node_list);
        try graph.nodes.append(fused_node);
    }

    pub fn get_output_shape(self: Fused_Pad_Conv) []usize {
        return self.fused_pad_conv.get_output_shape();
    }

    pub fn get_input_tensors(self: Fused_Pad_Conv) anyerror![]*TensorZant {
        return try self.fused_pad_conv.get_input_tensors();
    }

    pub fn get_output_tensors(self: Fused_Pad_Conv) anyerror![]*TensorZant {
        return try self.fused_pad_conv.get_output_tensors();
    }

    pub fn write_op(self: Fused_Pad_Conv, writer: std.fs.File.Writer) !void {
        try self.fused_pad_conv.write_op(writer);
    }

    pub fn compute_output_shape(self: Fused_Pad_Conv) []usize {
        return self.fused_pad_conv.compute_output_shape();
    }

    pub fn print(self: Fused_Pad_Conv) void {
        std.debug.print("\n Fused_Pad_Conv:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *Fused_Pad_Conv, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        // Try to substitute in the Pad operation
        self.op_Pad.sobstitute_tensors(old_tensor, new_tensor) catch {
            // If not found in Pad, try Conv operation
            self.op_Conv.sobstitute_tensors(old_tensor, new_tensor) catch {
                // Finally, try the fused result operation
                return try self.fused_pad_conv.sobstitute_tensors(old_tensor, new_tensor);
            };
        };
    }
};
