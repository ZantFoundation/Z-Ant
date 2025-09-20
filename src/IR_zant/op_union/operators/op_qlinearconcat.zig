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

// https://onnx.ai/onnx/operators/onnx__QLinearConcat.html
// INPUTS (variable number):
//      - inputs[0..N-3] (heterogeneous) - T: List of tensors for concatenation (quantized)
//      - inputs[N-2] (heterogeneous) - tensor(float): Scale of quantization of output
//      - inputs[N-1] (heterogeneous) - T: Zero point of quantization of output
// OUTPUTS:
//      - concat_result (heterogeneous) - T: Concatenated tensor (quantized)
// ATTRIBUTES:
//      - axis (int, required): Which axis to concat on

pub const QLinearConcat = struct {
    inputs: std.ArrayList(*TensorZant),
    input_scales: std.ArrayList(*TensorZant),
    input_zero_points: std.ArrayList(*TensorZant),
    output_scale: *TensorZant,
    output_zero_point: *TensorZant,
    concat_result: *TensorZant,
    // attributes:
    axis: i64, // default = 0

    pub fn init(nodeProto: *NodeProto) !QLinearConcat {
        // QLinearConcat has variable inputs: N input tensors + N scales + N zero_points + output_scale + output_zero_point
        // Total inputs = 3*N + 2
        if (nodeProto.input.len < 5 or (nodeProto.input.len - 2) % 3 != 0) {
            return error.QLinearConcatInvalidInputCount;
        }

        const num_inputs = (nodeProto.input.len - 2) / 3;

        var inputs = std.ArrayList(*TensorZant).init(allocator);
        var input_scales = std.ArrayList(*TensorZant).init(allocator);
        var input_zero_points = std.ArrayList(*TensorZant).init(allocator);

        // Parse input tensors, scales, and zero points
        for (0..num_inputs) |i| {
            // The model provides inputs interleaved per input: [scale_i, zero_point_i, tensor_i]
            // followed by output scale and output zero point.
            const base = 3 * i;
            const input_scale = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[base + 0])) |ptr| ptr else return error.input_scale_notFound;
            const input_zero_point = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[base + 1])) |ptr| ptr else return error.input_zero_point_notFound;
            const input_tensor = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[base + 2])) |ptr| ptr else return error.input_tensor_notFound;

            try inputs.append(input_tensor);
            try input_scales.append(input_scale);
            try input_zero_points.append(input_zero_point);
        }

        // Parse output scale and zero point
        const output_scale = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[3 * num_inputs + 0])) |ptr| ptr else return error.output_scale_notFound;
        const output_zero_point = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[3 * num_inputs + 1])) |ptr| ptr else return error.output_zero_point_notFound;

        const concat_result = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.concat_result_notFound;

        var axis: i64 = 0;

        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "axis")) {
                if (attr.type != onnx.AttributeType.INT) {
                    return error.InvalidAttributeType;
                }
                axis = attr.i;
            }
        }

        // Set the output type
        if (concat_result.ty == tensorZant_lib.TensorType.undefined) concat_result.ty = inputs.items[0].ty;

        const qlinear_concat = QLinearConcat{
            .inputs = inputs,
            .input_scales = input_scales,
            .input_zero_points = input_zero_points,
            .output_scale = output_scale,
            .output_zero_point = output_zero_point,
            .concat_result = concat_result,
            .axis = axis,
        };

        // Force shape computation during initialization
        _ = qlinear_concat.compute_output_shape() catch {};

        return qlinear_concat;
    }

    pub fn get_output_shape(self: QLinearConcat) []usize {
        return self.concat_result.getShape();
    }

    pub fn get_input_tensors(self: QLinearConcat) ![]*TensorZant {
        var input_tensors = std.ArrayList(*TensorZant).init(allocator);
        defer input_tensors.deinit();

        // Add all input tensors
        for (self.inputs.items) |input| {
            try input_tensors.append(input);
        }
        // Add all scales
        for (self.input_scales.items) |scale| {
            try input_tensors.append(scale);
        }
        // Add all zero points
        for (self.input_zero_points.items) |zp| {
            try input_tensors.append(zp);
        }
        // Add output scale and zero point
        try input_tensors.append(self.output_scale);
        try input_tensors.append(self.output_zero_point);

        return input_tensors.toOwnedSlice();
    }

    pub fn get_output_tensors(self: QLinearConcat) ![]*TensorZant {
        var output_tensors = std.ArrayList(*TensorZant).init(allocator);
        defer output_tensors.deinit();

        try output_tensors.append(self.concat_result);
        return output_tensors.toOwnedSlice();
    }

    pub fn write_op(self: QLinearConcat, writer: std.fs.File.Writer) !void {
        // Decide type strings robustly (prefer OUTPUT types for scale/zp)
        const input_type_s: []const u8 = if (self.inputs.items.len > 0 and self.inputs.items[0].ty != tensorZant_lib.TensorType.undefined) self.inputs.items[0].ty.toString() else "i8";
        const scale_type_s: []const u8 = if (self.output_scale.ty != tensorZant_lib.TensorType.undefined) self.output_scale.ty.toString() else "f32";
        const zp_type_s: []const u8 = if (self.output_zero_point.ty != tensorZant_lib.TensorType.undefined) self.output_zero_point.ty.toString() else "u8";

        // Create arrays for input tensors, scales, and zero points
        _ = try writer.print(
            \\
            \\    // Create arrays for QLinearConcat inputs
            \\    var qlinearconcat_inputs_{s} = [_]*const Tensor({s}){{
        , .{
            try utils.getSanitizedName(self.concat_result.name),
            input_type_s,
        });

        // Write input tensor pointers
        for (self.inputs.items, 0..) |input, idx| {
            if (idx > 0) {
                _ = try writer.print(", ", .{});
            }

            if (input.tc == TensorCategory.INITIALIZER) {
                _ = try writer.print("@as(*const Tensor({s}), @ptrCast(@constCast(&param_lib.tensor_{s})))", .{ input_type_s, try utils.getSanitizedName(input.name) });
            } else {
                _ = try writer.print("@as(*const Tensor({s}), @ptrCast(@constCast(&tensor_{s})))", .{ input_type_s, try utils.getSanitizedName(input.name) });
            }
        }

        _ = try writer.print(
            \\}};
            \\
            \\    var qlinearconcat_scales_{s} = [_]*const Tensor({s}){{
        , .{
            try utils.getSanitizedName(self.concat_result.name),
            scale_type_s,
        });

        // Write scale tensor pointers
        for (self.input_scales.items, 0..) |scale, idx| {
            if (idx > 0) {
                _ = try writer.print(", ", .{});
            }

            if (scale.tc == TensorCategory.INITIALIZER) {
                _ = try writer.print("@as(*const Tensor({s}), @ptrCast(@constCast(&param_lib.tensor_{s})))", .{ scale_type_s, try utils.getSanitizedName(scale.name) });
            } else {
                _ = try writer.print("@as(*const Tensor({s}), @ptrCast(@constCast(&tensor_{s})))", .{ scale_type_s, try utils.getSanitizedName(scale.name) });
            }
        }

        _ = try writer.print(
            \\}};
            \\
            \\    var qlinearconcat_zero_points_{s} = [_]*const Tensor({s}){{
        , .{
            try utils.getSanitizedName(self.concat_result.name),
            zp_type_s,
        });

        // Write zero point tensor pointers
        for (self.input_zero_points.items, 0..) |zp, idx| {
            if (idx > 0) {
                _ = try writer.print(", ", .{});
            }

            if (zp.tc == TensorCategory.INITIALIZER) {
                _ = try writer.print("@as(*const Tensor({s}), @ptrCast(@constCast(&param_lib.tensor_{s})))", .{ zp_type_s, try utils.getSanitizedName(zp.name) });
            } else {
                _ = try writer.print("@as(*const Tensor({s}), @ptrCast(@constCast(&tensor_{s})))", .{ zp_type_s, try utils.getSanitizedName(zp.name) });
            }
        }

        _ = try writer.print("}};", .{});

        // Generate output scale and zero point tensor strings
        var output_scale_string: []u8 = undefined;
        defer allocator.free(output_scale_string);
        var output_zero_point_string: []u8 = undefined;
        defer allocator.free(output_zero_point_string);

        // Rebuild output scale string with explicit cast
        if (self.output_scale.tc == TensorCategory.INITIALIZER) {
            output_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(@as(*const Tensor(", scale_type_s, "), @ptrCast(&param_lib.tensor_", try utils.getSanitizedName(self.output_scale.name), ")))" });
        } else {
            output_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(@as(*const Tensor(", scale_type_s, "), @ptrCast(&tensor_", try utils.getSanitizedName(self.output_scale.name), ")))" });
        }

        // Rebuild output zero_point string with explicit cast
        if (self.output_zero_point.tc == TensorCategory.INITIALIZER) {
            output_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(@as(*const Tensor(", zp_type_s, "), @ptrCast(&param_lib.tensor_", try utils.getSanitizedName(self.output_zero_point.name), ")))" });
        } else {
            output_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(@as(*const Tensor(", zp_type_s, "), @ptrCast(&tensor_", try utils.getSanitizedName(self.output_zero_point.name), ")))" });
        }

        // Write the operation call
        _ = try writer.print(
            \\
            \\
            \\    // Perform QLinearConcat
            \\    tensMath.lean_qlinearconcat(
            \\        {s},
            \\        {s},
            \\        {s},
            \\        &qlinearconcat_inputs_{s},
            \\        &qlinearconcat_scales_{s},
            \\        &qlinearconcat_zero_points_{s},
            \\        {s},
            \\        {s},
            \\        {},
            \\        &tensor_{s},
            \\    ) catch return -1;
        , .{
            input_type_s, // 1. InputType (data)
            scale_type_s, // 2. ScaleType
            zp_type_s, // 3. ZeroPointType
            try utils.getSanitizedName(self.concat_result.name), // 4. array inputs name
            try utils.getSanitizedName(self.concat_result.name), // 5. array scales name
            try utils.getSanitizedName(self.concat_result.name), // 6. array zero_points name
            output_scale_string, // 7. output scale
            output_zero_point_string, // 8. output zero point
            self.axis, // 9. axis (i64)
            try utils.getSanitizedName(self.concat_result.name), // 10. output tensor
        });
    }

    pub fn compute_output_shape(self: QLinearConcat) ![]usize {
        // QLinearConcat output shape calculation is same as regular concat
        var input_shapes = try allocator.alloc([]const usize, self.inputs.items.len);
        defer {
            for (input_shapes) |shape| allocator.free(shape);
            allocator.free(input_shapes);
        }

        // Find the maximum rank among all inputs
        var max_rank: usize = 0;
        for (self.inputs.items) |input| {
            max_rank = @max(max_rank, input.getShape().len);
        }

        for (self.inputs.items, 0..) |input, i| {
            const input_shape = input.getShape();
            var shape = try allocator.alloc(usize, max_rank);

            // Pad with 1s for lower-rank tensors (left-padding)
            const offset = max_rank - input_shape.len;
            for (0..max_rank) |j| {
                if (j < offset) {
                    shape[j] = 1;
                } else {
                    const dim = input_shape[j - offset];
                    shape[j] = if (dim < 0) 1 else @intCast(dim);
                }
            }
            input_shapes[i] = shape;
        }

        const output_shape = try tensorMath.get_concatenate_output_shape(input_shapes, self.axis);
        self.concat_result.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: QLinearConcat) void {
        std.debug.print("\n QLinearConcat:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *QLinearConcat, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        for (self.inputs.items, 0..) |tensor, i| {
            if (tensor == old_tensor) {
                self.inputs.items[i] = new_tensor;
                return;
            }
        }
        for (self.input_scales.items, 0..) |tensor, i| {
            if (tensor == old_tensor) {
                self.input_scales.items[i] = new_tensor;
                return;
            }
        }
        for (self.input_zero_points.items, 0..) |tensor, i| {
            if (tensor == old_tensor) {
                self.input_zero_points.items[i] = new_tensor;
                return;
            }
        }
        if (self.output_scale == old_tensor) {
            self.output_scale = new_tensor;
            return;
        }
        if (self.output_zero_point == old_tensor) {
            self.output_zero_point = new_tensor;
            return;
        }
        if (self.concat_result == old_tensor) {
            self.concat_result = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
};
