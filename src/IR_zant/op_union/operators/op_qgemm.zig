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

// QGemm (Quantized GEMM) operation
// Similar to QLinearMatMul but with bias support like regular Gemm
pub const QGemm = struct {
    // ONNX spec for QGemm typically has these inputs:
    // 0: A (quantized input)
    // 1: A_scale
    // 2: A_zero_point
    // 3: B (quantized weight)
    // 4: B_scale
    // 5: B_zero_point
    // 6: C (quantized bias)
    // 7: Y_scale (output scale)
    // 8: Y_zero_point (output zero point)
    input_A: *TensorZant,
    input_A_scale: *TensorZant,
    input_A_zero_point: *TensorZant,
    input_B: *TensorZant,
    input_B_scale: *TensorZant,
    input_B_zero_point: *TensorZant,
    input_C: *TensorZant, // bias
    output_Y_scale: *TensorZant,
    output_Y_zero_point: *TensorZant,
    output: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !QGemm {
        // QGemm should have 9 inputs
        if (nodeProto.input.len != 9) {
            std.debug.print("\nQGemm expects 9 inputs, got {}\n", .{nodeProto.input.len});
            return error.QGemmInvalidInputCount;
        }

        const input_A = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_A_notFound;
        const input_A_scale = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_A_scale_notFound;
        const input_A_zero_point = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[2])) |ptr| ptr else return error.input_A_zero_point_notFound;
        const input_B = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[3])) |ptr| ptr else return error.input_B_notFound;
        const input_B_scale = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[4])) |ptr| ptr else return error.input_B_scale_notFound;
        const input_B_zero_point = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[5])) |ptr| ptr else return error.input_B_zero_point_notFound;
        const input_C = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[6])) |ptr| ptr else return error.input_C_notFound;
        const output_Y_scale = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[7])) |ptr| ptr else return error.output_Y_scale_notFound;
        const output_Y_zero_point = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[8])) |ptr| ptr else return error.output_Y_zero_point_notFound;
        const output = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        // Set output type if undefined
        if (output.ty == tensorZant_lib.TensorType.undefined) output.ty = input_A.ty;

        // Force compute and set correct output shape
        if (input_A.shape.len >= 2 and input_B.shape.len >= 2) {
            const M = input_A.shape[input_A.shape.len - 2];
            const N = input_B.shape[input_B.shape.len - 2];

            var correct_shape = allocator.alloc(usize, 2) catch blk: {
                var fallback: [2]usize = [_]usize{ 1, 4 };
                fallback[0] = M;
                fallback[1] = N;
                break :blk &fallback;
            };
            correct_shape[0] = M;
            correct_shape[1] = N;

            output.shape = correct_shape;
        }

        return QGemm{
            .input_A = input_A,
            .input_A_scale = input_A_scale,
            .input_A_zero_point = input_A_zero_point,
            .input_B = input_B,
            .input_B_scale = input_B_scale,
            .input_B_zero_point = input_B_zero_point,
            .input_C = input_C,
            .output_Y_scale = output_Y_scale,
            .output_Y_zero_point = output_Y_zero_point,
            .output = output,
        };
    }

    pub fn get_output_shape(self: QGemm) []usize {
        // QGemm output shape: [M, N] where A is [M, K] and B is [N, K] (transposed format)
        // Use the input_A and input_B shapes to determine output
        if (self.input_A.shape.len >= 2 and self.input_B.shape.len >= 2) {
            const M = self.input_A.shape[self.input_A.shape.len - 2];
            const N = self.input_B.shape[self.input_B.shape.len - 2]; // N from first dimension of B
            const result = allocator.alloc(usize, 2) catch unreachable;
            result[0] = M;
            result[1] = N;

            // Force update the output tensor shape
            self.output.shape = result;

            return result;
        }
        // Fallback to default shape if we can't determine
        const result = allocator.alloc(usize, 2) catch unreachable;
        result[0] = 1;
        result[1] = 4;

        // Force update the output tensor shape
        self.output.shape = result;

        return result;
    }

    pub fn compute_output_shape(self: QGemm) ![]usize {
        // QGemm output shape: [M, N] where A is [M, K] and B is [N, K] (transposed format)
        if (self.input_A.shape.len >= 2 and self.input_B.shape.len >= 2) {
            const M = self.input_A.shape[self.input_A.shape.len - 2];
            const N = self.input_B.shape[self.input_B.shape.len - 2]; // N from first dimension of B

            const result = try allocator.alloc(usize, 2);
            result[0] = M;
            result[1] = N;

            // Force update the output tensor's shape permanently
            self.output.shape = result;

            return result;
        }

        return error.InvalidShapeInference;
    }

    pub fn get_input_tensors(self: QGemm) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        try inputs.append(self.input_A);
        try inputs.append(self.input_A_scale);
        try inputs.append(self.input_A_zero_point);
        try inputs.append(self.input_B);
        try inputs.append(self.input_B_scale);
        try inputs.append(self.input_B_zero_point);
        try inputs.append(self.input_C);
        try inputs.append(self.output_Y_scale);
        try inputs.append(self.output_Y_zero_point);
        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: QGemm) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        try outputs.append(self.output);
        return outputs.toOwnedSlice();
    }

    pub fn write_op(self: QGemm, writer: std.fs.File.Writer) !void {
        // Generate proper tensor name strings like other quantized operators do
        var tensor_a_string: []u8 = undefined;
        var tensor_a_scale_string: []u8 = undefined;
        var tensor_a_zero_point_string: []u8 = undefined;
        var tensor_b_string: []u8 = undefined;
        var tensor_b_scale_string: []u8 = undefined;
        var tensor_b_zero_point_string: []u8 = undefined;
        var tensor_c_string: []u8 = undefined;
        var tensor_y_scale_string: []u8 = undefined;
        var tensor_y_zero_point_string: []u8 = undefined;
        var tensor_output_string: []u8 = undefined;

        // Build tensor name strings with proper sanitization
        if (self.input_A.tc == TensorCategory.INITIALIZER) {
            tensor_a_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_A.name), ")" });
        } else {
            tensor_a_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.input_A.name) });
        }

        if (self.input_A_scale.tc == TensorCategory.INITIALIZER) {
            tensor_a_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_A_scale.name), ")" });
        } else {
            tensor_a_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_A_scale.name), ")" });
        }

        if (self.input_A_zero_point.tc == TensorCategory.INITIALIZER) {
            tensor_a_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_A_zero_point.name), ")" });
        } else {
            tensor_a_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_A_zero_point.name), ")" });
        }

        if (self.input_B.tc == TensorCategory.INITIALIZER) {
            tensor_b_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_B.name), ")" });
        } else {
            tensor_b_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.input_B.name) });
        }

        if (self.input_B_scale.tc == TensorCategory.INITIALIZER) {
            tensor_b_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_B_scale.name), ")" });
        } else {
            tensor_b_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_B_scale.name), ")" });
        }

        if (self.input_B_zero_point.tc == TensorCategory.INITIALIZER) {
            tensor_b_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_B_zero_point.name), ")" });
        } else {
            tensor_b_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_B_zero_point.name), ")" });
        }

        if (self.input_C.tc == TensorCategory.INITIALIZER) {
            tensor_c_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_C.name), ")" });
        } else {
            tensor_c_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.input_C.name) });
        }

        if (self.output_Y_scale.tc == TensorCategory.INITIALIZER) {
            tensor_y_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.output_Y_scale.name), ")" });
        } else {
            tensor_y_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.output_Y_scale.name), ")" });
        }

        if (self.output_Y_zero_point.tc == TensorCategory.INITIALIZER) {
            tensor_y_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.output_Y_zero_point.name), ")" });
        } else {
            tensor_y_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.output_Y_zero_point.name), ")" });
        }

        if (self.output.tc == TensorCategory.INITIALIZER) {
            tensor_output_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&param_lib.tensor_", try utils.getSanitizedName(self.output.name) });
        } else {
            tensor_output_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.output.name) });
        }

        // Write the operation call
        try writer.print("\n    // QGemm operation: quantized matrix multiplication (bias ignored for now)\n", .{});
        try writer.print("    tensMath.qgemm_lean(\n", .{});
        try writer.print("        {s},\n", .{tensor_a_string});
        try writer.print("        {s},\n", .{tensor_a_scale_string});
        try writer.print("        {s},\n", .{tensor_a_zero_point_string});
        try writer.print("        {s},\n", .{tensor_b_string});
        try writer.print("        {s},\n", .{tensor_b_scale_string});
        try writer.print("        {s},\n", .{tensor_b_zero_point_string});
        try writer.print("        {s},\n", .{tensor_output_string});
        try writer.print("        {s},\n", .{tensor_y_scale_string});
        try writer.print("        {s}\n", .{tensor_y_zero_point_string});
        try writer.print("    ) catch return -1;\n", .{});

        // Skip deallocation of input tensor to avoid FixedBufferAllocator issues
        // Input tensors will be deallocated by the general tensor management system
        // if (self.input_A.tc != TensorCategory.INITIALIZER) {
        //     try writer.print("    tensor_{s}.deinit();\n", .{try utils.getSanitizedName(self.input_A.name)});
        // }
    }

    pub fn print(self: QGemm) void {
        std.debug.print("\n QGemm:\n", .{});
        std.debug.print("   input_A: {s}\n", .{self.input_A.name});
        std.debug.print("   input_B: {s}\n", .{self.input_B.name});
        std.debug.print("   input_C: {s}\n", .{self.input_C.name});
        std.debug.print("   output: {s}\n", .{self.output.name});
    }

    pub fn sobstitute_tensors(self: *QGemm, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
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
        if (self.input_C == old_tensor) {
            self.input_C = new_tensor;
            return;
        }
        if (self.output_Y_scale == old_tensor) {
            self.output_Y_scale = new_tensor;
            return;
        }
        if (self.output_Y_zero_point == old_tensor) {
            self.output_Y_zero_point = new_tensor;
            return;
        }
        if (self.output == old_tensor) {
            self.output = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
};
