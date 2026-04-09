const std = @import("std");
const allocator = std.heap.page_allocator;
const IR_zant = @import("IR_zant");

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

const utils = IR_zant.utils;

/// Fused
/// DequantizeLinear-->
///                    |--> Add->QuantizeLinear operation for better performance
/// DequantizeLinear-->
///
/// This optimizes the common quantized arithmetic pattern by avoiding intermediate
/// dequantization and requantization, performing the add operation directly
/// on quantized values using QLinearAdd semantics.
pub const Fused_2Dequant_Add_Quant = struct {
    op_name: []const u8,
    op_DequantizeLinear_A: operators.DequantizeLinear,
    op_DequantizeLinear_B: operators.DequantizeLinear,
    op_Add: operators.Add,
    op_QuantizeLinear: operators.QuantizeLinear,

    // The resulting fused operation that combines all four
    fused_qlinear_add: operators.QLinearAdd,

    /// Initialize fused operation from individual operations
    pub fn init_fused_op(fusion_list: std.ArrayList(*NodeZant)) !Fused_2Dequant_Add_Quant {
        // Validation - expects exactly 4 operations: DequantA -> DequantB -> Add -> Quant
        if (fusion_list.items.len != 4) return error.WrongNumberOfElements;

        // Pattern: DequantizeLinear(A) -> DequantizeLinear(B) -> Add -> QuantizeLinear
        if (fusion_list.items[0].op != .dequantizeLinear) return error.WrongOpAtPos0;
        if (fusion_list.items[1].op != .dequantizeLinear) return error.WrongOpAtPos1;
        if (fusion_list.items[2].op != .add) return error.WrongOpAtPos2;
        if (fusion_list.items[3].op != .quantizeLinear) return error.WrongOpAtPos3;

        const dequant_a_op = switch (fusion_list.items[0].op) {
            .dequantizeLinear => |d| d,
            else => return error.InvalidDequantizeLinearOperation,
        };

        const dequant_b_op = switch (fusion_list.items[1].op) {
            .dequantizeLinear => |d| d,
            else => return error.InvalidDequantizeLinearOperation,
        };

        const add_op = switch (fusion_list.items[2].op) {
            .add => |a| a,
            else => return error.InvalidAddOperation,
        };

        const quant_op = switch (fusion_list.items[3].op) {
            .quantizeLinear => |q| q,
            else => return error.InvalidQuantizeLinearOperation,
        };

        // Create the fused QLinearAdd operation
        const fused_qlinear_add = operators.QLinearAdd{
            .input_A = dequant_a_op.x,
            .input_A_scale = dequant_a_op.x_scale,
            .input_A_zero_point = dequant_a_op.x_zero_point.?,
            .input_B = dequant_b_op.x,
            .input_B_scale = dequant_b_op.x_scale,
            .input_B_zero_point = dequant_b_op.x_zero_point.?,
            .output_C = quant_op.y,
            .input_C_scale = quant_op.y_scale,
            .input_C_zero_point = quant_op.y_zero_point.?,
        };

        // Downgrade LINK tensors between fudes noted to FUSED_LINK tensors
        dequant_a_op.y.set_tensorCategory(TensorCategory.FUSED_LINK);
        dequant_b_op.y.set_tensorCategory(TensorCategory.FUSED_LINK);
        add_op.output_C.set_tensorCategory(TensorCategory.FUSED_LINK);

        return Fused_2Dequant_Add_Quant{
            .op_name = try NodeZant_lib.getFusedOpsName(fusion_list),
            .op_DequantizeLinear_A = dequant_a_op,
            .op_DequantizeLinear_B = dequant_b_op,
            .op_Add = add_op,
            .op_QuantizeLinear = quant_op,
            .fused_qlinear_add = fused_qlinear_add,
        };
    }

    /// Pattern detection function for DequantizeLinear -> DequantizeLinear -> Add -> QuantizeLinear
    /// Starting from Add node as root, looking backwards for two DequantizeLinear predecessors
    pub fn fn_pattern_detection(graph: *GraphZant, root_node: *NodeZant) anyerror!?std.ArrayList(*NodeZant) {

        // Only start detection from Add nodes
        if (root_node.op != .add) {
            return null;
        }

        var node_list: std.ArrayList(*NodeZant) = .empty;
        errdefer node_list.deinit(allocator);

        // Get predecessors of the Add node
        var predecessors = try graph.get_predecessors(root_node);
        defer predecessors.deinit(IR_zant.pkg_allocator.allocator);

        // We need exactly 2 predecessors for binary Add operation
        if (predecessors.items.len != 2) {
            return null;
        }

        var dequant_a: ?*NodeZant = null;
        var dequant_b: ?*NodeZant = null;

        // Check that both predecessors are DequantizeLinear nodes
        for (predecessors.items) |pred| {
            if (pred.op != .dequantizeLinear) {
                if (dequant_a == null) {
                    dequant_a = pred;
                } else {
                    dequant_b = pred;
                }
            } else {
                return null;
            }
        }

        if (dequant_a == null or dequant_b == null) {
            return null;
        }

        // Check that Add has exactly one successor which is QuantizeLinear
        if (root_node.next.items.len != 1) {
            return null;
        }

        const quant_node = root_node.next.items[0];
        if (quant_node.op != .quantizeLinear) {
            return null;
        }

        // Build the node list in order: DequantA -> DequantB -> Add -> Quant
        try node_list.append(allocator, dequant_a.?);
        try node_list.append(allocator, dequant_b.?);
        try node_list.append(allocator, root_node); // Add node
        try node_list.append(allocator, quant_node);

        std.debug.print(" -> Found complete DequantizeLinear->DequantizeLinear->Add->QuantizeLinear pattern!", .{});

        return node_list;
    }

    /// Pattern fusion function
    pub fn fn_pattern_fusion(graph: *GraphZant, node_list: std.ArrayList(*NodeZant)) anyerror!NodeZant {
        _ = graph; // Not used in this sequential pattern

        // Validate the pattern - must be exactly 4 nodes
        if (node_list.items.len != 4) return error.InvalidNumberOfOps;

        if (node_list.items[0].op != .dequantizeLinear) return error.UnexpectedOpAtPos0;
        if (node_list.items[1].op != .dequantizeLinear) return error.UnexpectedOpAtPos1;
        if (node_list.items[2].op != .add) return error.UnexpectedOpAtPos2;
        if (node_list.items[3].op != .quantizeLinear) return error.UnexpectedOpAtPos3;

        const last_node = node_list.items[3]; // QuantizeLinear node

        // Clone the next list instead of direct reference
        var cloned_next: std.ArrayList(*NodeZant) = .empty;
        for (last_node.next.items) |next_node| {
            try cloned_next.append(allocator, next_node);
        }

        return NodeZant{
            .name = try NodeZant_lib.getFusedOpsName(node_list),
            .op_type = try NodeZant_lib.getFusedOpsType(node_list),
            .op = Op_union{ .fused_2Dequant_Add_Quant = try init_fused_op(node_list) },
            .next = cloned_next,
            .nodeProto = null,
            .ready = false,
            .is_fused = true,
        };
    }

    /// Pattern substitution function
    pub fn fn_pattern_sobstitution(graph: *GraphZant, fused_node: *NodeZant, node_list: std.ArrayList(*NodeZant)) anyerror!void {
        // Validate inputs - must be exactly 4 nodes
        if (node_list.items.len != 4) return error.InvalidPatternLength;

        const node_A = node_list.items[0]; // First DequantizeLinear node
        const node_B = node_list.items[1]; // Second DequantizeLinear node
        // const last_node = node_list.items[3]; // QuantizeLinear node

        // Step 1: Find all predecessor nodes that point to the first node
        const predecessors_A = try graph.get_predecessors(node_A);
        const predecessors_B = try graph.get_predecessors(node_B);

        // Step 2: Update predecessor nodes to point to fused_node
        for (predecessors_A.items) |predecessor| {
            for (predecessor.next.items, 0..) |next_node, i| {
                if (next_node == node_A) {
                    predecessor.next.items[i] = fused_node;
                }
            }
        }
        for (predecessors_B.items) |predecessor| {
            for (predecessor.next.items, 0..) |next_node, i| {
                if (next_node == node_B) {
                    predecessor.next.items[i] = fused_node;
                }
            }
        }

        // // Step 3: Set up fused node's successors
        // if (fused_node.next.items.len == 0) {
        //     for (last_node.next.items) |successor| {
        //         try fused_node.next.append(successor);
        //     }
        // }

        // Step 4: Remove old nodes from graph
        try graph.removeNodes(node_list);

        // Step 5: Add fused node to graph
        try graph.nodes.append(allocator, fused_node);
    }

    // Helper functions matching the interface

    pub fn get_output_shape(self: Fused_2Dequant_Add_Quant) []usize {
        return self.fused_qlinear_add.get_output_shape();
    }

    pub fn get_input_tensors(self: Fused_2Dequant_Add_Quant) anyerror![]*TensorZant {
        return try self.fused_qlinear_add.get_input_tensors();
    }

    pub fn get_output_tensors(self: Fused_2Dequant_Add_Quant) anyerror![]*TensorZant {
        return try self.fused_qlinear_add.get_output_tensors();
    }

    /// Optimized write operation for quantized add pattern.
    /// This implements QLinearAdd semantics: efficiently adds quantized inputs
    /// without intermediate dequantization.
    pub fn write_op(self: Fused_2Dequant_Add_Quant, writer: *std.Io.Writer) !void {
        try self.fused_qlinear_add.write_op(writer);
    }

    pub fn compute_output_shape(self: Fused_2Dequant_Add_Quant) []usize {
        return self.fused_qlinear_add.compute_output_shape();
    }

    pub fn print(self: Fused_2Dequant_Add_Quant) void {
        std.debug.print("\n Fused_2Dequant_Add_Quant:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *Fused_2Dequant_Add_Quant, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        // Try to substitute in DequantizeLinear_A operation
        self.op_DequantizeLinear_A.sobstitute_tensors(old_tensor, new_tensor) catch {
            // If not found, try DequantizeLinear_B operation
            self.op_DequantizeLinear_B.sobstitute_tensors(old_tensor, new_tensor) catch {
                // If not found, try Add operation
                self.op_Add.sobstitute_tensors(old_tensor, new_tensor) catch {
                    // If not found, try QuantizeLinear operation
                    self.op_QuantizeLinear.sobstitute_tensors(old_tensor, new_tensor) catch {
                        // Finally, try the fused result operation
                        return try self.fused_qlinear_add.sobstitute_tensors(old_tensor, new_tensor);
                    };
                };
            };
        };
    }
};
