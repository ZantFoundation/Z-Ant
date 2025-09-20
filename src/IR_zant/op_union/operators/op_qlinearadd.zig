const std = @import("std");
const allocator = std.heap.page_allocator;
const zant = @import("zant");
const IR_zant = @import("../../IR_zant.zig");

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

// --- zant ---
const tensorZant_lib = IR_zant.tensorZant_lib;
const TensorZant = tensorZant_lib.TensorZant;
const TensorCategory = tensorZant_lib.TensorCategory;

const tensorMath = zant.core.tensor.math_standard;
const utils = IR_zant.utils;

// https://onnx.ai/onnx/operators/onnx__QLinearAdd.html
// INPUTS:
//      - A (heterogeneous) - T: First operand (quantized)
//      - A_scale (heterogeneous) - tensor(float): Scale of quantization of input A
//      - A_zero_point (heterogeneous) - T: Zero point of quantization of input A
//      - B (heterogeneous) - T: Second operand (quantized)
//      - B_scale (heterogeneous) - tensor(float): Scale of quantization of input B
//      - B_zero_point (heterogeneous) - T: Zero point of quantization of input B
//      - C_scale (heterogeneous) - tensor(float): Scale of quantization of output C
//      - C_zero_point (heterogeneous) - T: Zero point of quantization of output C
// OUTPUTS:
//      - C (heterogeneous) - T: Result, has same element type as two inputs (quantized)

pub const QLinearAdd = struct {
    input_A: *TensorZant,
    input_A_scale: *TensorZant,
    input_A_zero_point: *TensorZant,
    input_B: *TensorZant,
    input_B_scale: *TensorZant,
    input_B_zero_point: *TensorZant,
    input_C_scale: *TensorZant,
    input_C_zero_point: *TensorZant,
    output_C: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !QLinearAdd {
        // QLinearAdd has exactly 8 inputs
        if (nodeProto.input.len != 8) {
            return error.QLinearAddInvalidInputCount;
        }

        const input_A = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_A_notFound;
        const input_A_scale = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_A_scale_notFound;
        const input_A_zero_point = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[2])) |ptr| ptr else return error.input_A_zero_point_notFound;
        const input_B = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[3])) |ptr| ptr else return error.input_B_notFound;
        const input_B_scale = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[4])) |ptr| ptr else return error.input_B_scale_notFound;
        const input_B_zero_point = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[5])) |ptr| ptr else return error.input_B_zero_point_notFound;
        const input_C_scale = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[6])) |ptr| ptr else return error.input_C_scale_notFound;
        const input_C_zero_point = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[7])) |ptr| ptr else return error.input_C_zero_point_notFound;

        const output_C = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_C_notFound;

        //set the output type:
        if (output_C.ty == tensorZant_lib.TensorType.undefined) output_C.ty = input_A.ty;

        const qlinear_add = QLinearAdd{
            .input_A = input_A,
            .input_A_scale = input_A_scale,
            .input_A_zero_point = input_A_zero_point,
            .input_B = input_B,
            .input_B_scale = input_B_scale,
            .input_B_zero_point = input_B_zero_point,
            .input_C_scale = input_C_scale,
            .input_C_zero_point = input_C_zero_point,
            .output_C = output_C,
        };

        // Force shape computation during initialization
        _ = qlinear_add.compute_output_shape() catch {};

        return qlinear_add;
    }

    pub fn get_output_shape(self: QLinearAdd) ![]usize {
        return try self.compute_output_shape();
    }

    pub fn get_input_tensors(self: QLinearAdd) ![]*TensorZant {
        var input_tensors = std.ArrayList(*TensorZant).init(allocator);
        defer input_tensors.deinit();

        try input_tensors.append(self.input_A);
        try input_tensors.append(self.input_A_scale);
        try input_tensors.append(self.input_A_zero_point);
        try input_tensors.append(self.input_B);
        try input_tensors.append(self.input_B_scale);
        try input_tensors.append(self.input_B_zero_point);
        try input_tensors.append(self.input_C_scale);
        try input_tensors.append(self.input_C_zero_point);

        return input_tensors.toOwnedSlice();
    }

    pub fn get_output_tensors(self: QLinearAdd) ![]*TensorZant {
        var output_tensors = std.ArrayList(*TensorZant).init(allocator);
        defer output_tensors.deinit();

        try output_tensors.append(self.output_C);

        return output_tensors.toOwnedSlice();
    }

    pub fn write_op(self: QLinearAdd, writer: std.fs.File.Writer) !void {
        // Create tensor string variables for each input
        var tensor_A_string: []u8 = undefined;
        defer allocator.free(tensor_A_string);
        var tensor_A_scale_string: []u8 = undefined;
        defer allocator.free(tensor_A_scale_string);
        var tensor_A_zero_point_string: []u8 = undefined;
        defer allocator.free(tensor_A_zero_point_string);
        var tensor_B_string: []u8 = undefined;
        defer allocator.free(tensor_B_string);
        var tensor_B_scale_string: []u8 = undefined;
        defer allocator.free(tensor_B_scale_string);
        var tensor_B_zero_point_string: []u8 = undefined;
        defer allocator.free(tensor_B_zero_point_string);
        var tensor_C_scale_string: []u8 = undefined;
        defer allocator.free(tensor_C_scale_string);
        var tensor_C_zero_point_string: []u8 = undefined;
        defer allocator.free(tensor_C_zero_point_string);

        // Generate tensor strings for each input
        if (self.input_A.tc == TensorCategory.INITIALIZER) {
            tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_A.name), ")" });
        } else {
            tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_A.name), ")" });
        }

        if (self.input_A_scale.tc == TensorCategory.INITIALIZER) {
            tensor_A_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_A_scale.name), ")" });
        } else {
            tensor_A_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_A_scale.name), ")" });
        }

        if (self.input_A_zero_point.tc == TensorCategory.INITIALIZER) {
            tensor_A_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_A_zero_point.name), ")" });
        } else {
            tensor_A_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_A_zero_point.name), ")" });
        }

        if (self.input_B.tc == TensorCategory.INITIALIZER) {
            tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_B.name), ")" });
        } else {
            tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_B.name), ")" });
        }

        if (self.input_B_scale.tc == TensorCategory.INITIALIZER) {
            tensor_B_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_B_scale.name), ")" });
        } else {
            tensor_B_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_B_scale.name), ")" });
        }

        if (self.input_B_zero_point.tc == TensorCategory.INITIALIZER) {
            tensor_B_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_B_zero_point.name), ")" });
        } else {
            tensor_B_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_B_zero_point.name), ")" });
        }

        if (self.input_C_scale.tc == TensorCategory.INITIALIZER) {
            tensor_C_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_C_scale.name), ")" });
        } else {
            tensor_C_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_C_scale.name), ")" });
        }

        if (self.input_C_zero_point.tc == TensorCategory.INITIALIZER) {
            tensor_C_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_C_zero_point.name), ")" });
        } else {
            tensor_C_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_C_zero_point.name), ")" });
        }

        // Write the operation call
        try writer.print("\n    tensMath.qlinearadd_lean(\n", .{});
        try writer.print("        {s},\n", .{tensor_A_string});
        try writer.print("        {s},\n", .{tensor_A_scale_string});
        try writer.print("        {s},\n", .{tensor_A_zero_point_string});
        try writer.print("        {s},\n", .{tensor_B_string});
        try writer.print("        {s},\n", .{tensor_B_scale_string});
        try writer.print("        {s},\n", .{tensor_B_zero_point_string});
        try writer.print("        &tensor_{s},\n", .{try utils.getSanitizedName(self.output_C.name)});
        try writer.print("        {s},\n", .{tensor_C_scale_string});
        try writer.print("        {s},\n", .{tensor_C_zero_point_string});
        try writer.print("    ) catch return -1;\n", .{});
    }

    pub fn compute_output_shape(self: QLinearAdd) ![]usize {
        // QLinearAdd output shape calculation:
        // For element-wise addition, output shape should be the same as input shapes
        // Both inputs should have the same shape (broadcasting rules apply)

        const input_A_shape = self.input_A.getShape();
        const input_B_shape = self.input_B.getShape();

        // If either input has placeholder shape [1], use the other input's shape
        if (input_A_shape.len == 1 and input_A_shape[0] == 1) {
            if (input_B_shape.len > 1) {
                // Copy B's shape to output
                const output_shape = try allocator.alloc(usize, input_B_shape.len);
                @memcpy(output_shape, input_B_shape);
                self.output_C.shape = output_shape;
                return output_shape;
            }
        } else if (input_B_shape.len == 1 and input_B_shape[0] == 1) {
            if (input_A_shape.len > 1) {
                // Copy A's shape to output
                const output_shape = try allocator.alloc(usize, input_A_shape.len);
                @memcpy(output_shape, input_A_shape);
                self.output_C.shape = output_shape;
                return output_shape;
            }
        } else {
            // Both inputs have valid shapes - check if they match exactly
            if (input_A_shape.len == input_B_shape.len) {
                var shapes_match = true;
                for (input_A_shape, 0..) |dim, i| {
                    if (dim != input_B_shape[i]) {
                        shapes_match = false;
                        break;
                    }
                }
                if (shapes_match) {
                    // Copy A's shape to output (both are the same)
                    const output_shape = try allocator.alloc(usize, input_A_shape.len);
                    @memcpy(output_shape, input_A_shape);
                    self.output_C.shape = output_shape;
                    return output_shape;
                }
            }

            // Shapes don't match exactly - try broadcasting
            const broadcasted_shape = try self.computeBroadcastShape(input_A_shape, input_B_shape);
            if (broadcasted_shape) |shape| {
                self.output_C.shape = shape;
                return shape;
            }
        }

        // Fallback: if both are placeholders or shapes don't match, keep placeholder
        return self.output_C.getShape();
    }

    /// Compute broadcasting shape following NumPy/ONNX broadcasting rules
    fn computeBroadcastShape(self: QLinearAdd, shape_a: []const usize, shape_b: []const usize) !?[]usize {
        _ = self;

        const max_rank = @max(shape_a.len, shape_b.len);
        var result_shape = try allocator.alloc(usize, max_rank);

        // Start from the rightmost dimensions and work backward
        var i: usize = 0;
        while (i < max_rank) : (i += 1) {
            const dim_idx = max_rank - 1 - i;

            // Get dimensions (1 if out of bounds)
            const dim_a = if (i < shape_a.len) shape_a[shape_a.len - 1 - i] else 1;
            const dim_b = if (i < shape_b.len) shape_b[shape_b.len - 1 - i] else 1;

            // Broadcasting rules:
            // - If one dimension is 1, use the other
            // - If both are the same, use that value
            // - Otherwise, incompatible
            if (dim_a == 1) {
                result_shape[dim_idx] = dim_b;
            } else if (dim_b == 1) {
                result_shape[dim_idx] = dim_a;
            } else if (dim_a == dim_b) {
                result_shape[dim_idx] = dim_a;
            } else {
                // Incompatible dimensions
                allocator.free(result_shape);
                return null;
            }
        }

        return result_shape;
    }

    pub fn sobstitute_tensors(self: *QLinearAdd, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input_A == old_tensor) {
            self.input_A = new_tensor;
            return;
        }
        if (self.input_A_scale == old_tensor) {
            self.input_A_scale = new_tensor;
            return;
        }
        if (self.input_A_zero_point == old_tensor) {
            self.input_A_zero_point = new_tensor;
            return;
        }
        if (self.input_B == old_tensor) {
            self.input_B = new_tensor;
            return;
        }
        if (self.input_B_scale == old_tensor) {
            self.input_B_scale = new_tensor;
            return;
        }
        if (self.input_B_zero_point == old_tensor) {
            self.input_B_zero_point = new_tensor;
            return;
        }
        if (self.input_C_scale == old_tensor) {
            self.input_C_scale = new_tensor;
            return;
        }
        if (self.input_C_zero_point == old_tensor) {
            self.input_C_zero_point = new_tensor;
            return;
        }
        if (self.output_C == old_tensor) {
            self.output_C = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
};
