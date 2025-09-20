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

// https://onnx.ai/onnx/operators/onnx__QLinearMul.html
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
//      - C (heterogeneous) - T: Result (quantized)

pub const QLinearMul = struct {
    input_A: *TensorZant,
    input_A_scale: *TensorZant,
    input_A_zero_point: *TensorZant,
    input_B: *TensorZant,
    input_B_scale: *TensorZant,
    input_B_zero_point: *TensorZant,
    input_C_scale: *TensorZant,
    input_C_zero_point: *TensorZant,
    output_C: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !QLinearMul {
        // QLinearMul has exactly 8 inputs
        if (nodeProto.input.len != 8) {
            return error.QLinearMulInvalidInputCount;
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
        if (output_C.ty == tensorZant_lib.TensorType.undefined) {
            // QLinearMul typically outputs same type as inputs (usually u8 or i8)
            if (input_A.ty != tensorZant_lib.TensorType.undefined) {
                output_C.ty = input_A.ty;
            } else {
                // Fallback to u8 for quantized operations when input type is undefined
                output_C.ty = tensorZant_lib.TensorType.u8;
            }
        }

        return QLinearMul{
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
    }

    pub fn get_output_shape(op: *const QLinearMul) ![]usize {
        // QLinearMul uses broadcasting semantics like regular Mul
        const output_shape = try tensorMath.get_qlinearmul_output_shape(
            op.input_A.shape,
            op.input_B.shape,
        );
        // Persist the inferred shape onto the output tensor so downstream code sees it
        op.output_C.shape = output_shape;
        // Also update size to product of dims (runtime tensor will set size on allocation)
        return output_shape;
    }

    pub fn run(op: *const QLinearMul) !void {
        // Determine input types and call appropriate QLinearMul function
        switch (op.input_A.ty) {
            .u8 => try tensorMath.qlinearmul_lean(
                u8,
                f32,
                u8,
                &op.input_A.tensor_u8,
                &op.input_A_scale.tensor_f32,
                &op.input_A_zero_point.tensor_u8,
                &op.input_B.tensor_u8,
                &op.input_B_scale.tensor_f32,
                &op.input_B_zero_point.tensor_u8,
                &op.input_C_scale.tensor_f32,
                &op.input_C_zero_point.tensor_u8,
                &op.output_C.tensor_u8,
            ),
            .i8 => try tensorMath.qlinearmul_lean(
                i8,
                f32,
                i8,
                &op.input_A.tensor_i8,
                &op.input_A_scale.tensor_f32,
                &op.input_A_zero_point.tensor_i8,
                &op.input_B.tensor_i8,
                &op.input_B_scale.tensor_f32,
                &op.input_B_zero_point.tensor_i8,
                &op.input_C_scale.tensor_f32,
                &op.input_C_zero_point.tensor_i8,
                &op.output_C.tensor_i8,
            ),
            else => return error.UnsupportedDataType,
        }
    }

    pub fn get_output_tensors(op: *const QLinearMul) ![]*TensorZant {
        var tensors = try allocator.alloc(*TensorZant, 1);
        tensors[0] = op.output_C;
        return tensors;
    }

    pub fn get_input_tensors(op: *const QLinearMul) ![]*TensorZant {
        var tensors = try allocator.alloc(*TensorZant, 8);
        tensors[0] = op.input_A;
        tensors[1] = op.input_A_scale;
        tensors[2] = op.input_A_zero_point;
        tensors[3] = op.input_B;
        tensors[4] = op.input_B_scale;
        tensors[5] = op.input_B_zero_point;
        tensors[6] = op.input_C_scale;
        tensors[7] = op.input_C_zero_point;
        return tensors;
    }

    pub fn write_op(op: *const QLinearMul, writer: std.fs.File.Writer) !void {
        // Create sanitized tensor name strings
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
        var tensor_output_string: []u8 = undefined;
        defer allocator.free(tensor_output_string);

        // Build tensor name strings with proper sanitization
        if (op.input_A.tc == TensorCategory.INITIALIZER) {
            tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(op.input_A.name), ")" });
        } else {
            tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(op.input_A.name), ")" });
        }

        if (op.input_A_scale.tc == TensorCategory.INITIALIZER) {
            tensor_A_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(op.input_A_scale.name), ")" });
        } else {
            tensor_A_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(op.input_A_scale.name), ")" });
        }

        if (op.input_A_zero_point.tc == TensorCategory.INITIALIZER) {
            tensor_A_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(op.input_A_zero_point.name), ")" });
        } else {
            tensor_A_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(op.input_A_zero_point.name), ")" });
        }

        if (op.input_B.tc == TensorCategory.INITIALIZER) {
            tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(op.input_B.name), ")" });
        } else {
            tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(op.input_B.name), ")" });
        }

        if (op.input_B_scale.tc == TensorCategory.INITIALIZER) {
            tensor_B_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(op.input_B_scale.name), ")" });
        } else {
            tensor_B_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(op.input_B_scale.name), ")" });
        }

        if (op.input_B_zero_point.tc == TensorCategory.INITIALIZER) {
            tensor_B_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(op.input_B_zero_point.name), ")" });
        } else {
            tensor_B_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(op.input_B_zero_point.name), ")" });
        }

        if (op.input_C_scale.tc == TensorCategory.INITIALIZER) {
            tensor_C_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(op.input_C_scale.name), ")" });
        } else {
            tensor_C_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(op.input_C_scale.name), ")" });
        }

        if (op.input_C_zero_point.tc == TensorCategory.INITIALIZER) {
            tensor_C_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(op.input_C_zero_point.name), ")" });
        } else {
            tensor_C_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(op.input_C_zero_point.name), ")" });
        }

        tensor_output_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(op.output_C.name) });

        switch (op.input_A.ty) {
            .u8 => {
                try writer.print("    tensMath.qlinearmul_lean(u8, f32, u8, {s}, {s}, {s}, {s}, {s}, {s}, {s}, {s}, {s}) catch return -1;\n", .{
                    tensor_A_string,
                    tensor_A_scale_string,
                    tensor_A_zero_point_string,
                    tensor_B_string,
                    tensor_B_scale_string,
                    tensor_B_zero_point_string,
                    tensor_C_scale_string,
                    tensor_C_zero_point_string,
                    tensor_output_string,
                });
            },
            .i8 => {
                try writer.print("    tensMath.qlinearmul_lean(i8, f32, i8, {s}, {s}, {s}, {s}, {s}, {s}, {s}, {s}, {s}) catch return -1;\n", .{
                    tensor_A_string,
                    tensor_A_scale_string,
                    tensor_A_zero_point_string,
                    tensor_B_string,
                    tensor_B_scale_string,
                    tensor_B_zero_point_string,
                    tensor_C_scale_string,
                    tensor_C_zero_point_string,
                    tensor_output_string,
                });
            },
            else => return error.UnsupportedDataType,
        }
    }

    pub fn print(op: *const QLinearMul) void {
        std.debug.print("\n QLINEAR_MUL:\n {any}", .{op});
    }

    pub fn sobstitute_tensors(self: *QLinearMul, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
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
