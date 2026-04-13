const std = @import("std");
const allocator = IR_zant.pkg_allocator.allocator;
const IR_zant = @import("IR_zant");

// --- zant IR---
const tensorZant_lib = IR_zant.tensorZant_lib;
const TensorZant = tensorZant_lib.TensorZant;
const TensorCategory = tensorZant_lib.TensorCategory;
const NodeZant_lib = IR_zant.NodeZant_lib;
const NodeZant = NodeZant_lib.NodeZant;
const GraphZant = IR_zant.GraphZant;
const utils = IR_zant.utils; //this is IR utils

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
        if (fusion_list.items[0].op != .conv) return error.WrongOpAtPose0;
        if (fusion_list.items[1].op != .relu) return error.WrongOpAtPose2;

        // Extract the specific operations from the unions
        const conv_op = switch (fusion_list.items[0].op) {
            .conv => |c| c,
            else => return error.InvalidConvOperation,
        };

        const relu_op = switch (fusion_list.items[1].op) {
            .relu => |r| r,
            else => return error.InvalidReluOperation,
        };

        // Downgrade LINK tensors between fudes noted to FUSED_LINK tensors
        conv_op.output_Y.set_tensorCategory(TensorCategory.FUSED_LINK);

        return Fused_Conv_Relu{
            .op_name = try NodeZant_lib.getFusedOpsName(fusion_list),
            .op_Conv = conv_op,
            .op_Relu = relu_op,
        };
    }

    pub fn fn_pattern_detection(graph: *GraphZant, root_node: *NodeZant) anyerror!?std.ArrayList(*NodeZant) {
        _ = graph; // Not used in this sequential pattern

        // CRITICAL FIX: Only start detection from Conv nodes
        if (root_node.op != .conv) {
            return null;
        }

        var node_list: std.ArrayList(*NodeZant) = .empty;
        errdefer node_list.deinit(allocator); // Clean up on error

        try node_list.append(allocator, root_node);

        // Check if Conv has exactly one successor and it's ReLU
        const next_nodes = root_node.next;
        if (next_nodes.items.len != 1) {
            node_list.deinit(allocator);
            return null;
        }

        const successor = next_nodes.items[0];
        if (successor.op != .relu) {
            node_list.deinit(allocator);
            return null;
        }

        std.debug.print(" -> Found Conv->Relu pattern!", .{});
        try node_list.append(allocator, successor);
        return node_list;
    }

    pub fn fn_pattern_fusion(graph: *GraphZant, node_list: std.ArrayList(*NodeZant)) anyerror!NodeZant {
        _ = graph; // in this case graph is not used since the pattern is sequential

        //checks
        if (node_list.items.len != 2) return error.InvalidNumberOfOps;

        // PATTERN_MATCHING_STRATEGY
        if (node_list.items[0].op != .conv) return error.UnexpectedOpAtPos0;
        if (node_list.items[1].op != .relu) return error.UnexpectedOpAtPos1;

        const relu_node: *NodeZant = node_list.items[1];

        //  Clone the next list instead of direct reference
        var cloned_next: std.ArrayList(*NodeZant) = .empty;
        for (relu_node.next.items) |next_node| {
            try cloned_next.append(allocator, next_node);
        }

        return NodeZant{
            .name = try NodeZant_lib.getFusedOpsName(node_list),
            .op_type = try NodeZant_lib.getFusedOpsType(node_list),
            .op = Op_union{ .fused_Conv_Relu = try init_fused_op(node_list) },
            .next = cloned_next,
            .nodeProto = null,
            .ready = false,
            .is_fused = true,
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
                try fused_node.next.append(allocator, successor);
            }
        }

        // Step 4: Remove old nodes from graph
        try graph.removeNodes(node_list);

        // Step 5: Add the fused_node to the graph's node list
        try graph.nodes.append(allocator, fused_node);
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

    pub fn write_op(self: Fused_Conv_Relu, writer: *std.Io.Writer) !void {

        //----create tensor_X_string
        var tensor_X_string: []u8 = undefined;
        defer allocator.free(tensor_X_string);

        if (self.op_Conv.input_X.tc == TensorCategory.INITIALIZER) {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.op_Conv.input_X.name),
                ")",
            });
        } else {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.op_Conv.input_X.name), ")" });
        }

        //----create tensor_W_string
        var tensor_W_string: []u8 = undefined;
        defer allocator.free(tensor_W_string);
        if (self.op_Conv.input_W.tc == TensorCategory.INITIALIZER) {
            tensor_W_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.op_Conv.input_W.name),
                ")",
            });
        } else {
            tensor_W_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.op_Conv.input_W.name), ")" });
        }

        //----create ?bias string
        var bias_string: []u8 = undefined;
        // Bias Tensor B is optional! verify the presence
        if (self.op_Conv.input_B) |input_B| {
            const B_name = try utils.getSanitizedName(input_B.name);
            bias_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", B_name, ")" });
        } else {
            bias_string = try std.mem.concat(allocator, u8, &[_][]const u8{"null"});
        }

        //----create stride string (mandatory)
        // TODO: implement default stride, see docs above
        if (self.op_Conv.strides == null) return error.StrideNotFound;
        const stride_string: []const u8 = try utils.i64SliceToUsizeArrayString(self.op_Conv.strides.?);

        //----create ?pads string
        var pads_string: []const u8 = "null";
        if (self.op_Conv.pads != null) {
            if (self.op_Conv.pads.?.len > 0) { // Check if the slice is actually non-empty
                pads_string = try utils.i64SliceToUsizeArrayString(self.op_Conv.pads.?);
                // Assuming no allocation needed to be freed, following write_conv
            } else {
                pads_string = "&[_]usize{}"; // Use explicit empty slice literal if input slice is empty
            }
        } // else pads_string remains "null"

        //----create ?dilatations string
        var dilat_string: []const u8 = "null";
        if (self.op_Conv.dilations != null) {
            if (self.op_Conv.dilations.?.len > 0) {
                dilat_string = try utils.i64SliceToUsizeArrayString(self.op_Conv.dilations.?);
            } else {
                dilat_string = "&[_]usize{}";
            }
        } // else dilat_string remains "null"

        // Check if we need cast operations for mixed precision
        const target_type = self.op_Relu.output_Y.ty.toString();
        const need_kernel_cast = !std.mem.eql(u8, self.op_Conv.input_W.ty.toString(), target_type);
        const need_bias_cast = if (self.op_Conv.input_B) |bias| !std.mem.eql(u8, bias.ty.toString(), target_type) else false;

        var final_kernel_string: []const u8 = undefined;
        var final_bias_string: []const u8 = undefined;
        var need_free_kernel = false;
        var need_free_bias = false;
        defer if (need_free_kernel) allocator.free(@constCast(final_kernel_string));
        defer if (need_free_bias) allocator.free(@constCast(final_bias_string));

        if (need_kernel_cast) {
            // Generate cast for kernel
            const kernel_name = try utils.getSanitizedName(self.op_Conv.input_W.name);
            _ = try writer.print(
                \\
                \\    // Cast kernel from {s} to {s}
                \\    var tensor_{s}_casted = Tensor({s}).fromShape(&allocator, @constCast(param_lib.tensor_{s}.shape)) catch return -2;
                \\    defer tensor_{s}_casted.deinit();
                \\    tensMath.cast_lean({s}, {s}, @constCast(&param_lib.tensor_{s}), &tensor_{s}_casted, IR_zant.onnx.DataType.FLOAT) catch return {d};
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

        if (need_bias_cast and self.op_Conv.input_B != null) {
            // Generate cast for bias
            const bias_name = try utils.getSanitizedName(self.op_Conv.input_B.?.name);
            _ = try writer.print(
                \\
                \\    // Cast bias from {s} to {s}
                \\    var tensor_{s}_casted = Tensor({s}).fromShape(&allocator, @constCast(param_lib.tensor_{s}.shape)) catch return -2;
                \\    defer tensor_{s}_casted.deinit();
                \\    tensMath.cast_lean({s}, {s}, @constCast(&param_lib.tensor_{s}), &tensor_{s}_casted, IR_zant.onnx.DataType.FLOAT) catch return {d};
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

        _ = try writer.print(
            \\    
            \\    @setEvalBranchQuota(10000);
            \\
            \\    // Conv + ReLU operation
            \\    tensMath.conv_relu_lean(
            \\        {s}, //type
            \\        {s}, //input
            \\        {s}, //kernel
            \\        &tensor_{s}, //output
            \\        {s}, //bias
            \\        {s}, //stride
            \\        {s}, //pads
            \\        {s}, //dilatations
            \\        {}, //group
            \\        "{s}", //auto_pad
            \\    ) catch return {d};
        , .{
            target_type,
            tensor_X_string, //Input tensor
            final_kernel_string, //Kernel (possibly casted)
            try utils.getSanitizedName(self.op_Relu.output_Y.name), // Output tensor
            final_bias_string, //Bias (possibly casted)
            stride_string, //Strides
            pads_string, //Pads
            dilat_string, //Dilatations
            self.op_Conv.group, //Group
            self.op_Conv.auto_pad, //auto_pad
            utils.getMathErrorReturn(), // Error code for math errors
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
