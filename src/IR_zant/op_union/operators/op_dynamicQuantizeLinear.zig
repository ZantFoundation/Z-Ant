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

// https://onnx.ai/onnx/operators/onnx__DynamicQuantizeLinear.html
// INPUTS:
//      - x (heterogeneous) - T1: Input tensor
// OUTPUTS:
//      - y (heterogeneous) - T2: Quantized output tensor
//      - y_scale (heterogeneous) - T1: Scale for quantization
//      - y_zero_point (heterogeneous) - T2: Zero point for quantization

pub const DynamicQuantizeLinear = struct {
    input_x: *TensorZant,
    output_y: *TensorZant,
    output_y_scale: *TensorZant,
    output_y_zero_point: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !DynamicQuantizeLinear {
        // DynamicQuantizeLinear has 1 input and 3 outputs
        if (nodeProto.input.len != 1) {
            return error.DynamicQuantizeLinearInvalidInputCount;
        }
        if (nodeProto.output.len != 3) {
            return error.DynamicQuantizeLinearInvalidOutputCount;
        }

        const input_x = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_x_notFound;
        const output_y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_y_notFound;
        const output_y_scale = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[1])) |ptr| ptr else return error.output_y_scale_notFound;
        const output_y_zero_point = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[2])) |ptr| ptr else return error.output_y_zero_point_notFound;

        return DynamicQuantizeLinear{
            .input_x = input_x,
            .output_y = output_y,
            .output_y_scale = output_y_scale,
            .output_y_zero_point = output_y_zero_point,
        };
    }

    pub fn get_output_shape(op: *const DynamicQuantizeLinear) ![]usize {
        // DynamicQuantizeLinear outputs: [y_shape, scalar_shape, scalar_shape]
        // For the main output, it has the same shape as input
        return try allocator.dupe(usize, op.input_x.shape);
    }

    pub fn run(op: *const DynamicQuantizeLinear) !void {
        // Use the DynamicQuantizeLinear_lean operation from tensorMath
        try tensorMath.dynamicQuantizeLinear_lean(
            &op.input_x.tensor_f32,
            &op.output_y.tensor_u8,
            &op.output_y_scale.tensor_f32,
            &op.output_y_zero_point.tensor_u8,
        );
    }

    pub fn get_output_tensors(op: *const DynamicQuantizeLinear) ![]*TensorZant {
        var tensors = try allocator.alloc(*TensorZant, 3);
        tensors[0] = op.output_y;
        tensors[1] = op.output_y_scale;
        tensors[2] = op.output_y_zero_point;
        return tensors;
    }

    pub fn get_input_tensors(op: *const DynamicQuantizeLinear) ![]*TensorZant {
        var tensors = try allocator.alloc(*TensorZant, 1);
        tensors[0] = op.input_x;
        return tensors;
    }

    pub fn write_op(op: *const DynamicQuantizeLinear, writer: std.fs.File.Writer) !void {
        // Create tensor string for input x
        var tensor_x_string: []u8 = undefined;
        defer allocator.free(tensor_x_string);
        if (op.input_x.tc == TensorCategory.INITIALIZER) {
            tensor_x_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(op.input_x.name),
                ")",
            });
        } else {
            tensor_x_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(op.input_x.name) });
        }

        // Generate the function call
        try writer.print(
            \\    tensMath.dynamicQuantizeLinear_lean(
            \\        {s}, // input x (f32)
            \\        &tensor_{s}, // output y (u8)
            \\        &tensor_{s}, // output y_scale (f32)
            \\        &tensor_{s}, // output y_zero_point (u8)
            \\    ) catch return -1;
        , .{
            tensor_x_string, // input x
            try utils.getSanitizedName(op.output_y.name), // output y
            try utils.getSanitizedName(op.output_y_scale.name), // output y_scale
            try utils.getSanitizedName(op.output_y_zero_point.name), // output y_zero_point
        });
    }

    pub fn print(op: *const DynamicQuantizeLinear) void {
        std.debug.print("\n DYNAMIC_QUANTIZE_LINEAR:\n {any}", .{op});
    }

    pub fn sobstitute_tensors(self: *DynamicQuantizeLinear, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input_x == old_tensor) {
            self.input_x = new_tensor;
            return;
        }
        if (self.output_y == old_tensor) {
            self.output_y = new_tensor;
            return;
        }
        if (self.output_y_scale == old_tensor) {
            self.output_y_scale = new_tensor;
            return;
        }
        if (self.output_y_zero_point == old_tensor) {
            self.output_y_zero_point = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
};
