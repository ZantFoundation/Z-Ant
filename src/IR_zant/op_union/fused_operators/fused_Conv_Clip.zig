const std = @import("std");
const zant = @import("zant");
const allocator = zant.utils.allocator.allocator;
const IR_zant = @import("../../IR_zant.zig");

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

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

const tensorMath = zant.core.tensor.math_standard;

const utils = IR_zant.utils;

/// Fused Conv+Clip operation for better performance
/// This combines convolution followed by clipping
/// Common pattern in MobileNet v2 and other efficient architectures
pub const Fused_Conv_Clip = struct {
    op_name: []const u8,
    op_Conv: operators.Conv,
    op_Clip: operators.Clip,

    pub fn init_fused_op(fusion_list: std.ArrayList(*NodeZant)) !Fused_Conv_Clip {
        //ensure correct operations are given
        if (fusion_list.items.len != 2) return error.WrongNumberOfElements;
        if (fusion_list.items[0].op != .conv) return error.WrongOpAtPose0;
        if (fusion_list.items[1].op != .clip) return error.WrongOpAtPose2;

        // Extract the specific operations from the unionsnono
        const conv_op = switch (fusion_list.items[0].op) {
            .conv => |c| c,
            else => return error.InvalidConvOperation,
        };

        const clip_op = switch (fusion_list.items[1].op) {
            .clip => |cl| cl,
            else => return error.InvalidClipOperation,
        };

        // Downgrade LINK tensors between fudes noted to FUSED_LINK tensors
        conv_op.output_Y.set_tensorCategory(TensorCategory.FUSED_LINK);

        return Fused_Conv_Clip{
            .op_name = try NodeZant_lib.getFusedOpsName(fusion_list),
            .op_Conv = conv_op,
            .op_Clip = clip_op,
        };
    }

    pub fn get_output_shape(self: Fused_Conv_Clip) []usize {
        return self.op_Clip.get_output_shape();
    }

    pub fn get_input_tensors(self: Fused_Conv_Clip) ![]*TensorZant {
        //evaluating inputs len
        const count = 2 + // X, W sempre presenti
            (if (self.op_Conv.input_B != null) @as(usize, 1) else 0) +
            (if (self.op_Clip.min != null) @as(usize, 1) else 0) +
            (if (self.op_Clip.max != null) @as(usize, 1) else 0);

        var inputs = try allocator.alloc(*TensorZant, count);
        errdefer allocator.free(inputs);
        var idx: usize = 0;

        inputs[idx] = self.op_Conv.input_X;
        idx += 1;
        inputs[idx] = self.op_Conv.input_W;
        idx += 1;
        if (self.op_Conv.input_B) |b| {
            inputs[idx] = b;
            idx += 1;
        }
        if (self.op_Clip.min) |m| {
            inputs[idx] = m;
            idx += 1;
        }
        if (self.op_Clip.max) |M| {
            inputs[idx] = M;
            idx += 1;
        }

        return inputs;
    }

    pub fn get_output_tensors(self: Fused_Conv_Clip) ![]*TensorZant {
        return self.op_Clip.get_output_tensors();
    }

    pub fn write_op(self: Fused_Conv_Clip, writer: anytype) !void {

        //----create tensor_X_string
        var tensor_X_string: []u8 = undefined;
        defer allocator.free(tensor_X_string);

        if (self.op_Conv.input_X.tc == TensorCategory.INITIALIZER) {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try IR_utils.getSanitizedName(self.op_Conv.input_X.name),
                ")",
            });
        } else {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try IR_utils.getSanitizedName(self.op_Conv.input_X.name), ")" });
        }

        //----create tensor_W_string
        var tensor_W_string: []u8 = undefined;
        defer allocator.free(tensor_W_string);
        if (self.op_Conv.input_W.tc == TensorCategory.INITIALIZER) {
            tensor_W_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try IR_utils.getSanitizedName(self.op_Conv.input_W.name),
                ")",
            });
        } else {
            tensor_W_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try IR_utils.getSanitizedName(self.op_Conv.input_W.name), ")" });
        }

        //----create optional ?bias string
        var bias_string: []u8 = undefined;
        defer allocator.free(bias_string);
        // Bias Tensor B is optional! verify the presence
        if (self.op_Conv.input_B) |input_B| {
            const B_name = try IR_utils.getSanitizedName(input_B.name);
            bias_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", B_name, ")" });
        } else {
            bias_string = try std.mem.concat(allocator, u8, &[_][]const u8{"null"});
        }

        //----create optional min string
        var min_string: []u8 = undefined;
        defer allocator.free(min_string);
        // Tensor Min is optional! verify the presence
        if (self.op_Clip.min) |min| {
            const min_name = try IR_utils.getSanitizedName(min.name);
            min_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", min_name, ")" });
        } else {
            min_string = try std.mem.concat(allocator, u8, &[_][]const u8{"null"});
        }

        //----create optional max string
        var max_string: []u8 = undefined;
        defer allocator.free(max_string);
        // Tensor Max is optional! verify the presence
        if (self.op_Clip.max) |max| {
            const max_name = try IR_utils.getSanitizedName(max.name);
            max_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", max_name, ")" });
        } else {
            max_string = try std.mem.concat(allocator, u8, &[_][]const u8{"null"});
        }

        //?dilation string
        var dilat_string: []const u8 = "null";
        if (self.op_Conv.dilations != null) {
            if (self.op_Conv.dilations.?.len > 0) {
                dilat_string = try IR_utils.i64SliceToUsizeArrayString(self.op_Conv.dilations.?);
            } else {
                dilat_string = "&[_]usize{}";
            }
        } // else dilat_string remains "null"

        //strides string
        if (self.op_Conv.strides == null) return error.StrideNotFound;
        const stride_string = try utils.i64SliceToUsizeArrayString(self.op_Conv.strides.?);

        // &pads string
        var pads_string: []const u8 = "null";
        if (self.op_Conv.pads != null) {
            if (self.op_Conv.pads.?.len > 0) { // Check if the slice is actually non-empty
                pads_string = try IR_utils.i64SliceToUsizeArrayString(self.op_Conv.pads.?);
                // Assuming no allocation needed to be freed, following write_conv
            } else {
                pads_string = "&[_]usize{}"; // Use explicit empty slice literal if input slice is empty
            }
        } // else pads_string remains "null"

        // Check if we need cast operations for mixed precision
        const target_type = self.op_Clip.output.ty.toString();
        const need_kernel_cast = !std.mem.eql(u8, self.op_Conv.input_W.ty.toString(), target_type);
        const need_bias_cast = if (self.op_Conv.input_B) |bias| !std.mem.eql(u8, bias.ty.toString(), target_type) else false;

        //Handle casting
        var final_kernel_string: []const u8 = undefined;
        var final_bias_string: []const u8 = undefined;
        var need_free_kernel = false;
        var need_free_bias = false;
        defer if (need_free_kernel) allocator.free(@constCast(final_kernel_string));
        defer if (need_free_bias) allocator.free(@constCast(final_bias_string));

        //--KERNEL CAST--
        if (need_kernel_cast) {
            // Generate cast for kernel
            const kernel_name = try IR_utils.getSanitizedName(self.op_Conv.input_W.name);
            _ = try writer.print(
                \\
                \\    // Cast kernel from {s} to {s}
                \\    var tensor_{s}_casted = Tensor({s}).fromShape(&allocator, @constCast(param_lib.tensor_{s}.shape)) catch return -2;
                \\    defer tensor_{s}_casted.deinit();
                \\    tensMath.cast_lean({s}, {s}, @constCast(&param_lib.tensor_{s}), &tensor_{s}_casted, zant.onnx.DataType.FLOAT) catch return -{d};
                \\
            , .{
                self.op_Conv.input_W.ty.toString(),
                target_type,
                kernel_name,
                target_type,
                kernel_name,
                kernel_name,
                self.op_Conv.input_W.ty.toString(),
                target_type,
                kernel_name,
                kernel_name,
                utils.getMathErrorReturn(), // Error code for math errors
            });
            final_kernel_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", kernel_name, "_casted)" });
            need_free_kernel = true;
        } else {
            final_kernel_string = tensor_W_string;
        }

        //BIAS CAST
        if (need_bias_cast and self.op_Conv.input_B != null) {
            // Generate cast for bias
            const bias_name = try IR_utils.getSanitizedName(self.op_Conv.input_B.?.name);
            _ = try writer.print(
                \\
                \\    // Cast bias from {s} to {s}
                \\    var tensor_{s}_casted = Tensor({s}).fromShape(&allocator, @constCast(param_lib.tensor_{s}.shape)) catch return -2;
                \\    defer tensor_{s}_casted.deinit();
                \\    tensMath.cast_lean({s}, {s}, @constCast(&param_lib.tensor_{s}), &tensor_{s}_casted, zant.onnx.DataType.FLOAT) catch return -{d};
                \\
            , .{
                self.op_Conv.input_B.?.ty.toString(),
                target_type,
                bias_name,
                target_type,
                bias_name,
                bias_name,
                self.op_Conv.input_B.?.ty.toString(),
                target_type,
                bias_name,
                bias_name,
                utils.getMathErrorReturn(), // Error code for math errors
            });
            final_bias_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", bias_name, "_casted)" });
            need_free_bias = true;
        } else {
            final_bias_string = bias_string;
        }

        // Generate fused call
        _ = try writer.print(
            \\
            \\    @setEvalBranchQuota(10000);
            \\    // Fused Conv+Clip operation
            \\    tensMath.conv_clip_lean(
            \\        {s},
            \\        {s},
            \\        {s},
            \\        &tensor_{s},
            \\        {s},
            \\        {s},
            \\        {s},
            \\        {s},
            \\        {},
            \\        "{s}",
            \\        {s},
            \\        {s},
            \\    ) catch return -{d};
            \\
        , .{
            target_type,
            tensor_X_string,
            final_kernel_string,
            try IR_utils.getSanitizedName(self.op_Clip.output.name),
            final_bias_string,
            stride_string,
            pads_string,
            dilat_string,
            self.op_Conv.group,
            self.op_Conv.auto_pad,
            min_string,
            max_string,
            utils.getMathErrorReturn(), // Error code for math errors
        });
    }

    pub fn compute_output_shape(self: Fused_Conv_Clip) []usize {
        return self.op_Clip.compute_output_shape();
    }

    pub fn print(self: Fused_Conv_Clip) void {
        std.debug.print("\n Fused_Conv_Clip:\n {any}", .{self});
    }

    //if there are problems check the conv.sobtitute_tensors, it changes also the output that here is not considered
    pub fn sobstitute_tensors(self: *Fused_Conv_Clip, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        // Direct access instead of nested try/catch
        if (self.op_Conv.input_X == old_tensor) {
            self.op_Conv.input_X = new_tensor;
            return;
        }
        if (self.op_Conv.input_W == old_tensor) {
            self.op_Conv.input_W = new_tensor;
            return;
        }
        if (self.op_Conv.input_B) |b| {
            if (b == old_tensor) {
                self.op_Conv.input_B = new_tensor;
                return;
            }
        }
        if (self.op_Clip.min) |m| {
            if (m == old_tensor) {
                self.op_Clip.min = new_tensor;
                return;
            }
        }
        if (self.op_Clip.max) |M| {
            if (M == old_tensor) {
                self.op_Clip.max = new_tensor;
                return;
            }
        }
        if (self.op_Clip.output == old_tensor) {
            self.op_Clip.output = new_tensor;
            return;
        }

        return error.OldTensorNotFoundInSubstitution;
    }

    // --- Fusion --
    /// Pattern detection function for Conv -> Clip
    pub fn fn_pattern_detection(graph: *GraphZant, root_node: *NodeZant) anyerror!?std.ArrayList(*NodeZant) {
        _ = graph; // Not used in this sequential pattern

        // Only start detection from DequantizeLinear nodes
        if (root_node.op != .conv) {
            return null;
        }

        if (root_node.next.items.len != 1) {
            return null;
        }

        var node_list: std.ArrayList(*NodeZant) = .empty;
        errdefer node_list.deinit(allocator);

        try node_list.append(allocator, root_node);

        const pad_node = root_node.next.items[0];
        if (pad_node.op != .clip) {
            node_list.deinit(allocator);
            return null;
        }

        try node_list.append(allocator, pad_node);

        std.debug.print(" -> Found complete Conv->Clip pattern!", .{});

        return node_list;
    }

    /// Pattern fusion function
    pub fn fn_pattern_fusion(graph: *GraphZant, node_list: std.ArrayList(*NodeZant)) anyerror!NodeZant {
        _ = graph; // Not used in this sequential pattern

        // Validate the pattern
        if (node_list.items.len != 2) return error.InvalidNumberOfOps;
        if (node_list.items[0].op != .conv) return error.UnexpectedOpAtPos0;
        if (node_list.items[1].op != .clip) return error.UnexpectedOpAtPos1;

        const last_node = node_list.items[1]; // Clip

        // Clone the next list instead of direct reference
        var cloned_next: std.ArrayList(*NodeZant) = .empty;
        for (last_node.next.items) |next_node| {
            try cloned_next.append(allocator, next_node);
        }

        return NodeZant{
            .name = try NodeZant_lib.getFusedOpsName(node_list),
            .op_type = try NodeZant_lib.getFusedOpsType(node_list),
            .op = Op_union{ .fused_Conv_Clip = try init_fused_op(node_list) },
            .next = cloned_next,
            .nodeProto = null,
            .ready = false,
            .is_fused = true,
        };
    }

    /// Pattern substitution function
    pub fn fn_pattern_sobstitution(graph: *GraphZant, fused_node: *NodeZant, node_list: std.ArrayList(*NodeZant)) anyerror!void {
        // Validate inputs
        if (node_list.items.len != 2) return error.InvalidPatternLength;

        const first_node = node_list.items[0]; // DequantizeLinear node
        const last_node = node_list.items[1]; // QLinearConv node

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
                try fused_node.next.append(allocator, successor);
            }
        }

        // Step 4: Remove old nodes from graph
        try graph.removeNodes(node_list);

        // Step 5: Add fused node to graph
        try graph.nodes.append(allocator, fused_node);
    }
};
