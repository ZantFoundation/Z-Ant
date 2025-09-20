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

const utils = IR_zant.utils;

/// Fused DequantizeLinear->Clip->QuantizeLinear operation for better performance
/// This optimizes the common quantized activation pattern by avoiding intermediate
/// dequantization and requantization, performing the clip operation directly
/// on quantized values.
pub const Fused_Dequant_Clip_Quant = struct {
    op_name: []const u8,
    op_DequantizeLinear: operators.DequantizeLinear,
    op_Clip: operators.Clip,
    op_QuantizeLinear: operators.QuantizeLinear,

    /// Initialize fused operation from individual operations
    pub fn init_fused_op(fusion_list: std.ArrayList(*NodeZant)) !Fused_Dequant_Clip_Quant {
        // Validation
        if (fusion_list.items.len != 3) return error.WrongNumberOfElements;
        if (!std.mem.eql(u8, fusion_list.items[0].op_type, "DequantizeLinear")) return error.WrongOpAtPos0;
        if (!std.mem.eql(u8, fusion_list.items[1].op_type, "Clip")) return error.WrongOpAtPos1;
        if (!std.mem.eql(u8, fusion_list.items[2].op_type, "QuantizeLinear")) return error.WrongOpAtPos2;

        // Extract operations
        const dequant_op = switch (fusion_list.items[0].op) {
            .dequantizeLinear => |d| d,
            else => return error.InvalidDequantizeLinearOperation,
        };

        const clip_op = switch (fusion_list.items[1].op) {
            .clip => |c| c,
            else => return error.InvalidClipOperation,
        };

        const quant_op = switch (fusion_list.items[2].op) {
            .quantizeLinear => |q| q,
            else => return error.InvalidQuantizeLinearOperation,
        };

        // Extract clip bounds
        var min_val: f32 = 0.0;
        var max_val: f32 = std.math.floatMax(f32);

        if (clip_op.min) |min_tensor| {
            if (min_tensor.ptr) |tensor_ptr| {
                min_val = tensor_ptr.f32.data[0];
            }
        }

        if (clip_op.max) |max_tensor| {
            if (max_tensor.ptr) |tensor_ptr| {
                max_val = tensor_ptr.f32.data[0];
            }
        }

        return Fused_Dequant_Clip_Quant{
            .op_name = try NodeZant_lib.getFusedOpsName(fusion_list),
            .op_DequantizeLinear = dequant_op,
            .op_Clip = clip_op,
            .op_QuantizeLinear = quant_op,
        };
    }

    /// Pattern detection function for DequantizeLinear -> Clip -> QuantizeLinear
    pub fn fn_pattern_detection(graph: *GraphZant, root_node: *NodeZant) anyerror!?std.ArrayList(*NodeZant) {
        _ = graph; // Not used in this sequential pattern

        std.debug.print("\n  Checking 3-op DequantClipQuant pattern from node: {s}", .{root_node.op_type});

        // Only start detection from DequantizeLinear nodes
        if (!std.mem.eql(u8, root_node.op_type, "DequantizeLinear")) {
            std.debug.print(" -> Not a DequantizeLinear node, skipping", .{});
            return null;
        }

        var node_list = std.ArrayList(*NodeZant).init(allocator);
        errdefer node_list.deinit();

        try node_list.append(root_node);
        std.debug.print(" -> DequantizeLinear node found, checking for Clip successor", .{});

        // Check DequantizeLinear -> Clip
        if (root_node.next.items.len != 1) {
            std.debug.print(" -> DequantizeLinear has {} successors (expected 1)", .{root_node.next.items.len});
            node_list.deinit();
            return null;
        }

        const clip_node = root_node.next.items[0];
        if (!std.mem.eql(u8, clip_node.op_type, "Clip")) {
            std.debug.print(" -> DequantizeLinear successor is {s} (expected Clip)", .{clip_node.op_type});
            node_list.deinit();
            return null;
        }

        try node_list.append(clip_node);
        std.debug.print(" -> Found DequantizeLinear->Clip, checking for QuantizeLinear", .{});

        // Check Clip -> QuantizeLinear
        if (clip_node.next.items.len != 1) {
            std.debug.print(" -> Clip has {} successors (expected 1)", .{clip_node.next.items.len});
            node_list.deinit();
            return null;
        }

        const quant_node = clip_node.next.items[0];
        if (!std.mem.eql(u8, quant_node.op_type, "QuantizeLinear")) {
            std.debug.print(" -> Clip successor is {s} (expected QuantizeLinear)", .{quant_node.op_type});
            node_list.deinit();
            return null;
        }

        try node_list.append(quant_node);
        std.debug.print(" -> Found complete DequantizeLinear->Clip->QuantizeLinear pattern!", .{});

        return node_list;
    }

    /// Pattern fusion function
    pub fn fn_pattern_fusion(graph: *GraphZant, node_list: std.ArrayList(*NodeZant)) anyerror!NodeZant {
        _ = graph; // Not used in this sequential pattern

        // Validate the pattern
        if (node_list.items.len != 3) return error.InvalidNumberOfOps;
        if (!std.mem.eql(u8, node_list.items[0].op_type, "DequantizeLinear")) return error.UnexpectedOpAtPos0;
        if (!std.mem.eql(u8, node_list.items[1].op_type, "Clip")) return error.UnexpectedOpAtPos1;
        if (!std.mem.eql(u8, node_list.items[2].op_type, "QuantizeLinear")) return error.UnexpectedOpAtPos2;

        const last_node = node_list.items[2]; // QuantizeLinear node

        // Clone the next list instead of direct reference
        var cloned_next = std.ArrayList(*NodeZant).init(allocator);
        for (last_node.next.items) |next_node| {
            try cloned_next.append(next_node);
        }

        // Clone the fusion_list instead of direct reference
        var cloned_fusion_list = std.ArrayList(*NodeZant).init(allocator);
        for (node_list.items) |node| {
            try cloned_fusion_list.append(node);
        }

        return NodeZant{
            .name = try NodeZant_lib.getFusedOpsName(node_list),
            .op_type = try NodeZant_lib.getFusedOpsType(node_list),
            .op = Op_union{ .fused_Dequant_Clip_Quant = try init_fused_op(node_list) },
            .next = cloned_next,
            .nodeProto = null,
            .ready = false,
            .fusion_list = cloned_fusion_list,
        };
    }

    /// Pattern substitution function
    pub fn fn_pattern_sobstitution(graph: *GraphZant, fused_node: *NodeZant, node_list: std.ArrayList(*NodeZant)) anyerror!void {
        // Validate inputs
        if (node_list.items.len != 3) return error.InvalidPatternLength;

        const first_node = node_list.items[0]; // DequantizeLinear node
        const last_node = node_list.items[2]; // QuantizeLinear node

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

    // Helper functions matching the interface

    pub fn get_output_shape(self: Fused_Dequant_Clip_Quant) []usize {
        return self.op_QuantizeLinear.y.getShape();
    }

    pub fn get_input_tensors(self: Fused_Dequant_Clip_Quant) anyerror![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();

        try inputs.append(self.op_DequantizeLinear.x);
        try inputs.append(self.op_DequantizeLinear.x_scale);
        if (self.op_DequantizeLinear.x_zero_point) |zp| {
            try inputs.append(zp);
        }

        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: Fused_Dequant_Clip_Quant) anyerror![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();

        try outputs.append(self.op_QuantizeLinear.y);
        return outputs.toOwnedSlice();
    }

    /// Optimized write operation for quantized clip pattern.
    /// This should be called when we detect the pattern:
    /// DequantizeLinear -> Clip -> QuantizeLinear
    pub fn write_op(self: Fused_Dequant_Clip_Quant, writer: std.fs.File.Writer) !void {
        // Extract clip bounds from the clip operation
        var min_val: f32 = 0.0;
        var max_val: f32 = std.math.floatMax(f32);

        if (self.op_Clip.min) |min_tensor| {
            if (min_tensor.ptr) |tensor_ptr| {
                min_val = tensor_ptr.f32.data[0];
            }
        }

        if (self.op_Clip.max) |max_tensor| {
            if (max_tensor.ptr) |tensor_ptr| {
                max_val = tensor_ptr.f32.data[0];
            }
        }

        try self.write_op_quantized_pattern(
            self.op_DequantizeLinear.x,
            self.op_DequantizeLinear.x_scale,
            self.op_DequantizeLinear.x_zero_point.?,
            self.op_QuantizeLinear.y,
            self.op_QuantizeLinear.y_scale,
            self.op_QuantizeLinear.y_zero_point.?,
            min_val,
            max_val,
            writer,
        );
    }

    pub fn write_op_quantized_pattern(
        self: Fused_Dequant_Clip_Quant,
        input_quantized_tensor: *TensorZant,
        input_scale_tensor: *TensorZant,
        input_zero_point_tensor: *TensorZant,
        output_quantized_tensor: *TensorZant,
        output_scale_tensor: *TensorZant,
        output_zero_point_tensor: *TensorZant,
        min_val: f32,
        max_val: f32,
        writer: std.fs.File.Writer,
    ) !void {
        _ = self; // Self is not used in this static-like function
        _ = output_quantized_tensor; // Output is the same as input (in-place)

        // Helper to create tensor strings
        const createTensorStr = struct {
            fn call(tensor: *TensorZant) ![]u8 {
                if (tensor.tc == TensorCategory.INITIALIZER) {
                    return try std.mem.concat(allocator, u8, &[_][]const u8{
                        "@constCast(&param_lib.tensor_",
                        try utils.getSanitizedName(tensor.name),
                        ")",
                    });
                } else {
                    return try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(tensor.name) });
                }
            }
        }.call;

        const str_input_quantized = try createTensorStr(input_quantized_tensor);
        defer allocator.free(str_input_quantized);
        const str_input_scale = try createTensorStr(input_scale_tensor);
        defer allocator.free(str_input_scale);
        const str_input_zero_point = try createTensorStr(input_zero_point_tensor);
        defer allocator.free(str_input_zero_point);
        const str_output_scale = try createTensorStr(output_scale_tensor);
        defer allocator.free(str_output_scale);
        const str_output_zero_point = try createTensorStr(output_zero_point_tensor);
        defer allocator.free(str_output_zero_point);

        try writer.print(
            \\
            \\    // Fused DequantizeLinear->Clip->QuantizeLinear optimization
            \\    tensMath.clip_quantized_lean(
            \\        {s}, // InputType
            \\        {s}, // input tensor
            \\        {s}.data[0], // input_scale
            \\        {s}.data[0], // input_zero_point
            \\        {d:.6}, // min_val
            \\        {d:.6}, // max_val
            \\        @constCast({s}), // output = input (in-place)
            \\        {s}.data[0], // output_scale
            \\        {s}.data[0], // output_zero_point
            \\    ) catch return -1;
            \\
        , .{
            input_quantized_tensor.ty.toString(),
            str_input_quantized,
            str_input_scale,
            str_input_zero_point,
            min_val,
            max_val,
            str_input_quantized,
            str_output_scale,
            str_output_zero_point,
        });
    }

    pub fn compute_output_shape(self: Fused_Dequant_Clip_Quant) []usize {
        // Output shape is the same as input shape for clip operations
        return self.op_DequantizeLinear.x.get_shape();
    }

    pub fn print(self: Fused_Dequant_Clip_Quant) void {
        std.debug.print("\n Fused_Dequant_Clip_Quant:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *Fused_Dequant_Clip_Quant, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        // Try to substitute in each operation
        self.op_DequantizeLinear.sobstitute_tensors(old_tensor, new_tensor) catch {
            self.op_Clip.sobstitute_tensors(old_tensor, new_tensor) catch {
                return try self.op_QuantizeLinear.sobstitute_tensors(old_tensor, new_tensor);
            };
        };
    }
};
