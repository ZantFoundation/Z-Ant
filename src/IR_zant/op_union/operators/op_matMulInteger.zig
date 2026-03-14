const std = @import("std");
const allocator = std.heap.page_allocator;
const zant = @import("zant");
const IR_zant = @import("../../IR_zant.zig");

const onnx = zant.onnx;
const NodeProto = onnx.NodeProto;

const tensorZant_lib = IR_zant.tensorZant_lib;
const TensorZant = tensorZant_lib.TensorZant;
const TensorCategory = tensorZant_lib.TensorCategory;

const tensorMath = zant.core.tensor.math_standard;
const utils = IR_zant.utils;

// https://onnx.ai/onnx/operators/onnx__MatMulInteger.html
pub const MatMulInteger = struct {
    input_A: *TensorZant,
    input_B: *TensorZant,
    input_a_zero_point: ?*TensorZant,
    input_b_zero_point: ?*TensorZant,
    output_Y: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !MatMulInteger {
        if (nodeProto.input.len < 2 or nodeProto.input.len > 4) {
            return error.MatMulIntegerInvalidInputCount;
        }

        const input_A = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_A_notFound;
        const input_B = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_B_notFound;
        const input_a_zero_point = if (nodeProto.input.len > 2 and nodeProto.input[2].len > 0) tensorZant_lib.tensorMap.getPtr(nodeProto.input[2]) else null;
        const input_b_zero_point = if (nodeProto.input.len > 3 and nodeProto.input[3].len > 0) tensorZant_lib.tensorMap.getPtr(nodeProto.input[3]) else null;
        const output_Y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        if (output_Y.ty == tensorZant_lib.TensorType.undefined) {
            output_Y.ty = tensorZant_lib.TensorType.i32;
        }

        return MatMulInteger{
            .input_A = input_A,
            .input_B = input_B,
            .input_a_zero_point = input_a_zero_point,
            .input_b_zero_point = input_b_zero_point,
            .output_Y = output_Y,
        };
    }

    pub fn get_output_shape(self: *MatMulInteger) ![]usize {
        const output_shape = try tensorMath.get_mat_mul_output_shape(self.input_A.shape, self.input_B.shape);
        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn compute_output_shape(self: *MatMulInteger) ![]usize {
        return self.get_output_shape();
    }

    pub fn get_input_tensors(self: *const MatMulInteger) ![]*TensorZant {
        var count: usize = 2;
        if (self.input_a_zero_point != null) count += 1;
        if (self.input_b_zero_point != null) count += 1;

        var tensors = try allocator.alloc(*TensorZant, count);
        tensors[0] = self.input_A;
        tensors[1] = self.input_B;

        var index: usize = 2;
        if (self.input_a_zero_point) |zp| {
            tensors[index] = zp;
            index += 1;
        }
        if (self.input_b_zero_point) |zp| {
            tensors[index] = zp;
        }

        return tensors;
    }

    pub fn get_output_tensors(self: *const MatMulInteger) ![]*TensorZant {
        var tensors = try allocator.alloc(*TensorZant, 1);
        tensors[0] = self.output_Y;
        return tensors;
    }

    fn getTensorString(tensor: *TensorZant) ![]u8 {
        if (tensor.tc == TensorCategory.INITIALIZER) {
            return std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(tensor.name),
                ")",
            });
        }
        return std.mem.concat(allocator, u8, &[_][]const u8{
            "&tensor_",
            try utils.getSanitizedName(tensor.name),
        });
    }

    pub fn write_op(self: *const MatMulInteger, writer: std.fs.File.Writer) !void {
        const tensor_A_string = try getTensorString(self.input_A);
        defer allocator.free(tensor_A_string);
        const tensor_B_string = try getTensorString(self.input_B);
        defer allocator.free(tensor_B_string);

        var a_zero_point_string: []u8 = undefined;
        defer allocator.free(a_zero_point_string);
        if (self.input_a_zero_point) |zp| {
            a_zero_point_string = try getTensorString(zp);
        } else {
            a_zero_point_string = try allocator.dupe(u8, "null");
        }

        var b_zero_point_string: []u8 = undefined;
        defer allocator.free(b_zero_point_string);
        if (self.input_b_zero_point) |zp| {
            b_zero_point_string = try getTensorString(zp);
        } else {
            b_zero_point_string = try allocator.dupe(u8, "null");
        }

        try writer.print(
            \\    tensMath.matMulInteger_lean(
            \\        {s},
            \\        {s},
            \\        {s},
            \\        {s},
            \\        &tensor_{s},
            \\    ) catch return -1;
        , .{
            tensor_A_string,
            tensor_B_string,
            a_zero_point_string,
            b_zero_point_string,
            try utils.getSanitizedName(self.output_Y.name),
        });
    }

    pub fn print(self: *const MatMulInteger) void {
        std.debug.print("\n MAT_MUL_INTEGER:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *MatMulInteger, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input_A == old_tensor) {
            self.input_A = new_tensor;
            return;
        }
        if (self.input_B == old_tensor) {
            self.input_B = new_tensor;
            return;
        }
        if (self.input_a_zero_point == old_tensor) {
            self.input_a_zero_point = new_tensor;
            return;
        }
        if (self.input_b_zero_point == old_tensor) {
            self.input_b_zero_point = new_tensor;
            return;
        }
        if (self.output_Y == old_tensor) {
            self.output_Y = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
};
