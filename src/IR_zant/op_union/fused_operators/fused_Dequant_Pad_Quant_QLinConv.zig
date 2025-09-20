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

/// Fused DequantizeLinear -> pad -> QuantizeLinear -> QLinearConv operation for better performance
pub const Fused_Dequant_Pad_Quant_QLinConv = struct {
    op_name: []const u8,
    op_DequantizeLinear: operators.DequantizeLinear,
    op_Pad: operators.Pad,
    op_QuantizeLinear: operators.QuantizeLinear,
    op_QLinearConv: operators.QLinearConv,

    // The resulting fused operation that combines all four
    fused_qlinearconv: operators.QLinearConv,

    /// FIXED fusion initialization with proper tensor handling
    pub fn init_fused_op(fusion_list: std.ArrayList(*NodeZant)) !Fused_Dequant_Pad_Quant_QLinConv {
        // Validation
        if (fusion_list.items.len != 4) return error.WrongNumberOfElements;
        if (!std.mem.eql(u8, fusion_list.items[0].op_type, "DequantizeLinear")) return error.WrongOpAtPos0;
        if (!std.mem.eql(u8, fusion_list.items[1].op_type, "Pad")) return error.WrongOpAtPos1;
        if (!std.mem.eql(u8, fusion_list.items[2].op_type, "QuantizeLinear")) return error.WrongOpAtPos2;
        if (!std.mem.eql(u8, fusion_list.items[3].op_type, "QLinearConv")) return error.WrongOpAtPos3;

        // Extract operations
        const dequant_op = switch (fusion_list.items[0].op) {
            .dequantizeLinear => |d| d,
            else => return error.InvalidDequantizeLinearOperation,
        };

        const pad_op = switch (fusion_list.items[1].op) {
            .pad => |p| p,
            else => return error.InvalidPadOperation,
        };

        const quant_op = switch (fusion_list.items[2].op) {
            .quantizeLinear => |q| q,
            else => return error.InvalidQuantizeLinearOperation,
        };

        const qlinearconv_op = switch (fusion_list.items[3].op) {
            .qlinearconv => |qc| qc,
            else => return error.InvalidQLinearConvOperation,
        };

        // Create the fused QLinearConv operation
        var fused_qconv = qlinearconv_op;

        fused_qconv.input_x = dequant_op.x;
        fused_qconv.input_x_scale = dequant_op.x_scale;
        fused_qconv.input_x_zero_point = dequant_op.x_zero_point.?;

        if (pad_op.input_pads.ptr) |pad_data_AnyTensor| {
            // Get existing pads from QLinearConv (should be initialized to zeros in QLinearConv.init)
            var existing_pads: [4]i64 = .{ 0, 0, 0, 0 };
            if (qlinearconv_op.pads) |conv_pads| {
                // QLinearConv pads format: [top, left, bottom, right] for 2D
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
                    else => {
                        std.debug.print("Warning: Unsupported pad data type in fusion\n", .{});
                    },
                }
            }

            // Create fused pads
            const final_pads = [4]i64{
                existing_pads[0] + pad_values[0], // top
                existing_pads[1] + pad_values[1], // left
                existing_pads[2] + pad_values[2], // bottom
                existing_pads[3] + pad_values[3], // right
            };

            // Allocate and set the fused pads
            const fused_pads = try allocator.alloc(i64, 4);
            @memcpy(fused_pads, &final_pads);
            fused_qconv.pads = fused_pads;

            std.debug.print("Fused padding: original={any}, pad_op={any}, final={any}\n", .{ existing_pads, pad_values, final_pads });
        }

        // The output should be the same as the original QLinearConv output
        fused_qconv.output_y = qlinearconv_op.output_y;

        return Fused_Dequant_Pad_Quant_QLinConv{
            .op_name = try NodeZant_lib.getFusedOpsName(fusion_list),
            .op_DequantizeLinear = dequant_op,
            .op_Pad = pad_op,
            .op_QuantizeLinear = quant_op,
            .op_QLinearConv = qlinearconv_op,
            .fused_qlinearconv = fused_qconv,
        };
    }

    /// Pattern detection function for DequantizeLinear -> Pad -> QuantizeLinear -> QLinearConv
    pub fn fn_pattern_detection(graph: *GraphZant, root_node: *NodeZant) anyerror!?std.ArrayList(*NodeZant) {
        _ = graph; // Not used in this sequential pattern

        std.debug.print("\n  Checking 4-op pattern from node: {s}", .{root_node.op_type});

        // Only start detection from DequantizeLinear nodes
        if (!std.mem.eql(u8, root_node.op_type, "DequantizeLinear")) {
            std.debug.print(" -> Not a DequantizeLinear node, skipping", .{});
            return null;
        }

        var node_list = std.ArrayList(*NodeZant).init(allocator);
        errdefer node_list.deinit();

        try node_list.append(root_node);
        std.debug.print(" -> DequantizeLinear node found, checking for Pad successor", .{});

        // Check DequantizeLinear -> Pad
        if (root_node.next.items.len != 1) {
            std.debug.print(" -> DequantizeLinear has {} successors (expected 1)", .{root_node.next.items.len});
            node_list.deinit();
            return null;
        }

        const pad_node = root_node.next.items[0];
        if (!std.mem.eql(u8, pad_node.op_type, "Pad")) {
            std.debug.print(" -> DequantizeLinear successor is {s} (expected Pad)", .{pad_node.op_type});
            node_list.deinit();
            return null;
        }

        try node_list.append(pad_node);
        std.debug.print(" -> Found DequantizeLinear->Pad, checking for QuantizeLinear", .{});

        // Check Pad -> QuantizeLinear
        if (pad_node.next.items.len != 1) {
            std.debug.print(" -> Pad has {} successors (expected 1)", .{pad_node.next.items.len});
            node_list.deinit();
            return null;
        }

        const quant_node = pad_node.next.items[0];
        if (!std.mem.eql(u8, quant_node.op_type, "QuantizeLinear")) {
            std.debug.print(" -> Pad successor is {s} (expected QuantizeLinear)", .{quant_node.op_type});
            node_list.deinit();
            return null;
        }

        try node_list.append(quant_node);
        std.debug.print(" -> Found Pad->QuantizeLinear, checking for QLinearConv", .{});

        // Check QuantizeLinear -> QLinearConv
        if (quant_node.next.items.len != 1) {
            std.debug.print(" -> QuantizeLinear has {} successors (expected 1)", .{quant_node.next.items.len});
            node_list.deinit();
            return null;
        }

        const qlinearconv_node = quant_node.next.items[0];
        if (!std.mem.eql(u8, qlinearconv_node.op_type, "QLinearConv")) {
            std.debug.print(" -> QuantizeLinear successor is {s} (expected QLinearConv)", .{qlinearconv_node.op_type});
            node_list.deinit();
            return null;
        }

        try node_list.append(qlinearconv_node);
        std.debug.print(" -> Found complete DequantizeLinear->Pad->QuantizeLinear->QLinearConv pattern!", .{});

        return node_list;
    }

    /// Pattern fusion function
    pub fn fn_pattern_fusion(graph: *GraphZant, node_list: std.ArrayList(*NodeZant)) anyerror!NodeZant {
        _ = graph; // Not used in this sequential pattern

        // Validate the pattern
        if (node_list.items.len != 4) return error.InvalidNumberOfOps;
        if (!std.mem.eql(u8, node_list.items[0].op_type, "DequantizeLinear")) return error.UnexpectedOpAtPos0;
        if (!std.mem.eql(u8, node_list.items[1].op_type, "Pad")) return error.UnexpectedOpAtPos1;
        if (!std.mem.eql(u8, node_list.items[2].op_type, "QuantizeLinear")) return error.UnexpectedOpAtPos2;
        if (!std.mem.eql(u8, node_list.items[3].op_type, "QLinearConv")) return error.UnexpectedOpAtPos3;

        const last_node = node_list.items[3]; // QLinearConv node

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
            .op = Op_union{ .fused_Dequant_Pad_Quant_QLinConv = try init_fused_op(node_list) },
            .next = cloned_next,
            .nodeProto = null,
            .ready = false,
            .fusion_list = cloned_fusion_list,
        };
    }

    /// Pattern substitution function
    pub fn fn_pattern_sobstitution(graph: *GraphZant, fused_node: *NodeZant, node_list: std.ArrayList(*NodeZant)) anyerror!void {
        // Validate inputs
        if (node_list.items.len != 4) return error.InvalidPatternLength;

        const first_node = node_list.items[0]; // DequantizeLinear node
        const last_node = node_list.items[3]; // QLinearConv node

        // Step 1: Find all predecessor nodes that point to the first node
        const predecessors = try graph.get_predecessors(first_node);

        // Step 2: Update predecessor nodes to point to fused_node
        for (predecessors.items) |predecessor| {
            for (predecessor.next.items, 0..) |next_node, i| {
                if (next_node == first_node) {
                    predecessor.next.items[i] = fused_node;
                }
            }
        }

        // Step 3: Set up fused node's successors
        if (fused_node.next.items.len == 0) {
            for (last_node.next.items) |successor| {
                try fused_node.next.append(successor);
            }
        }

        // Step 4: Remove old nodes from graph
        try graph.removeNodes(node_list);

        // Step 5: Add fused node to graph
        try graph.nodes.append(fused_node);
    }

    // Helper functions matching the Fused_Conv_Relu interface

    pub fn get_output_shape(self: Fused_Dequant_Pad_Quant_QLinConv) []usize {
        return self.fused_qlinearconv.get_output_shape();
    }

    pub fn get_input_tensors(self: Fused_Dequant_Pad_Quant_QLinConv) anyerror![]*TensorZant {
        return try self.fused_qlinearconv.get_input_tensors();
    }

    pub fn get_output_tensors(self: Fused_Dequant_Pad_Quant_QLinConv) anyerror![]*TensorZant {
        return try self.fused_qlinearconv.get_output_tensors();
    }

    pub fn write_op(self: Fused_Dequant_Pad_Quant_QLinConv, writer: std.fs.File.Writer) !void {
        try self.fused_qlinearconv.write_op(writer);
    }

    pub fn compute_output_shape(self: Fused_Dequant_Pad_Quant_QLinConv) []usize {
        return self.fused_qlinearconv.compute_output_shape();
    }

    pub fn print(self: Fused_Dequant_Pad_Quant_QLinConv) void {
        std.debug.print("\n Fused_Dequant_Pad_Quant_QLinConv:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *Fused_Dequant_Pad_Quant_QLinConv, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        // Try to substitute in DequantizeLinear operation
        self.op_DequantizeLinear.sobstitute_tensors(old_tensor, new_tensor) catch {
            // If not found, try Pad operation
            self.op_Pad.sobstitute_tensors(old_tensor, new_tensor) catch {
                // If not found, try QuantizeLinear operation
                self.op_QuantizeLinear.sobstitute_tensors(old_tensor, new_tensor) catch {
                    // If not found, try QLinearConv operation
                    self.op_QLinearConv.sobstitute_tensors(old_tensor, new_tensor) catch {
                        // Finally, try the fused result operation
                        return try self.fused_qlinearconv.sobstitute_tensors(old_tensor, new_tensor);
                    };
                };
            };
        };
    }
};
