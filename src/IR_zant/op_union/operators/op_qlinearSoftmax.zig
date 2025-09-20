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

// QLinearSoftmax (non-standard, but following same pattern as other QLinear operators)
// INPUTS:
//      - x (heterogeneous) - T: Input tensor (quantized)
//      - x_scale (heterogeneous) - tensor(float): Scale of quantization of input x
//      - x_zero_point (heterogeneous) - T: Zero point of quantization of input x
//      - y_scale (heterogeneous) - tensor(float): Scale of quantization of output y
//      - y_zero_point (heterogeneous) - T: Zero point of quantization of output y
// OUTPUTS:
//      - y (heterogeneous) - T: Output tensor (quantized)
// ATTRIBUTES:
//      - axis - INT (default is -1): The axis along which to compute softmax

pub const QLinearSoftmax = struct {
    input_x: *TensorZant,
    input_x_scale: *TensorZant,
    input_x_zero_point: *TensorZant,
    input_y_scale: *TensorZant,
    input_y_zero_point: *TensorZant,
    output_y: *TensorZant,

    // Attributes
    axis: i64,

    pub fn init(nodeProto: *NodeProto) !QLinearSoftmax {
        // QLinearSoftmax has 5 inputs
        if (nodeProto.input.len != 5) {
            return error.QLinearSoftmaxInvalidInputCount;
        }

        const input_x = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_x_notFound;
        const input_x_scale = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_x_scale_notFound;
        const input_x_zero_point = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[2])) |ptr| ptr else return error.input_x_zero_point_notFound;
        const input_y_scale = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[3])) |ptr| ptr else return error.input_y_scale_notFound;
        const input_y_zero_point = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[4])) |ptr| ptr else return error.input_y_zero_point_notFound;

        const output_y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_y_notFound;

        // Default attributes
        var axis: i64 = -1;

        // Parse attributes
        for (nodeProto.attribute) |attr| {
            if (std.mem.indexOf(u8, attr.name, "axis")) |_| {
                if (attr.type == onnx.AttributeType.INT) axis = attr.i else return error.AxisNotINT;
            }
        }

        //set the output type:
        if (output_y.ty == tensorZant_lib.TensorType.undefined) {
            // QLinearSoftmax always outputs u8 (quantized softmax)
            output_y.ty = tensorZant_lib.TensorType.u8;
        }

        return QLinearSoftmax{
            .input_x = input_x,
            .input_x_scale = input_x_scale,
            .input_x_zero_point = input_x_zero_point,
            .input_y_scale = input_y_scale,
            .input_y_zero_point = input_y_zero_point,
            .output_y = output_y,
            .axis = axis,
        };
    }

    pub fn get_output_shape(op: *const QLinearSoftmax) ![]usize {
        // QLinearSoftmax preserves input shape
        return tensorMath.get_qlinearsoftmax_output_shape(op.input_x.shape);
    }

    pub fn run(op: *const QLinearSoftmax) !void {
        // Determine effective axis for actual input rank (assume ONNX axis refers to 4D [N,C,H,W])
        const nd: i64 = @intCast(op.input_x.shape.len);
        const adjust: i64 = 4 - nd; // if nd==3 => adjust=1 so axis 1 -> 0
        var axis_eff: i64 = op.axis - adjust;
        if (axis_eff < 0) axis_eff = 0;
        if (axis_eff >= nd) axis_eff = nd - 1;

        // Determine input types and call appropriate QLinearSoftmax function
        switch (op.input_x.ty) {
            .u8 => try tensorMath.qlinearsoftmax_lean(
                u8,
                f32,
                u8,
                &op.input_x.tensor_u8,
                &op.input_x_scale.tensor_f32,
                &op.input_x_zero_point.tensor_u8,
                &op.input_y_scale.tensor_f32,
                &op.input_y_zero_point.tensor_u8,
                &op.output_y.tensor_u8,
                @intCast(axis_eff),
            ),
            .i8 => try tensorMath.qlinearsoftmax_lean(
                i8,
                f32,
                i8,
                &op.input_x.tensor_i8,
                &op.input_x_scale.tensor_f32,
                &op.input_x_zero_point.tensor_i8,
                &op.input_y_scale.tensor_f32,
                &op.input_y_zero_point.tensor_i8,
                &op.output_y.tensor_i8,
                @intCast(axis_eff),
            ),
            else => return error.UnsupportedDataType,
        }
    }

    pub fn get_output_tensors(op: *const QLinearSoftmax) ![]*TensorZant {
        var tensors = try allocator.alloc(*TensorZant, 1);
        tensors[0] = op.output_y;
        return tensors;
    }

    pub fn get_input_tensors(op: *const QLinearSoftmax) ![]*TensorZant {
        var tensors = try allocator.alloc(*TensorZant, 5);
        tensors[0] = op.input_x;
        tensors[1] = op.input_x_scale;
        tensors[2] = op.input_x_zero_point;
        tensors[3] = op.input_y_scale;
        tensors[4] = op.input_y_zero_point;
        return tensors;
    }

    pub fn write_op(op: *const QLinearSoftmax, writer: std.fs.File.Writer) !void {
        // Create sanitized tensor name strings
        var tensor_x_string: []u8 = undefined;
        defer allocator.free(tensor_x_string);
        var tensor_x_scale_string: []u8 = undefined;
        defer allocator.free(tensor_x_scale_string);
        var tensor_x_zero_point_string: []u8 = undefined;
        defer allocator.free(tensor_x_zero_point_string);
        var tensor_y_scale_string: []u8 = undefined;
        defer allocator.free(tensor_y_scale_string);
        var tensor_y_zero_point_string: []u8 = undefined;
        defer allocator.free(tensor_y_zero_point_string);
        var tensor_output_string: []u8 = undefined;
        defer allocator.free(tensor_output_string);

        // Build tensor name strings with proper sanitization
        if (op.input_x.tc == TensorCategory.INITIALIZER) {
            tensor_x_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(op.input_x.name), ")" });
        } else {
            tensor_x_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(op.input_x.name), ")" });
        }

        if (op.input_x_scale.tc == TensorCategory.INITIALIZER) {
            tensor_x_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(op.input_x_scale.name), ")" });
        } else {
            tensor_x_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(op.input_x_scale.name), ")" });
        }

        if (op.input_x_zero_point.tc == TensorCategory.INITIALIZER) {
            tensor_x_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(op.input_x_zero_point.name), ")" });
        } else {
            tensor_x_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(op.input_x_zero_point.name), ")" });
        }

        if (op.input_y_scale.tc == TensorCategory.INITIALIZER) {
            tensor_y_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(op.input_y_scale.name), ")" });
        } else {
            tensor_y_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(op.input_y_scale.name), ")" });
        }

        if (op.input_y_zero_point.tc == TensorCategory.INITIALIZER) {
            tensor_y_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(op.input_y_zero_point.name), ")" });
        } else {
            tensor_y_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(op.input_y_zero_point.name), ")" });
        }

        tensor_output_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(op.output_y.name) });

        // Compute effective axis based on actual rank
        const nd: i64 = @intCast(op.input_x.shape.len);
        const adjust: i64 = 4 - nd;
        var axis_eff: i64 = op.axis - adjust;
        if (axis_eff < 0) axis_eff = 0;
        if (axis_eff >= nd) axis_eff = nd - 1;

        switch (op.input_x.ty) {
            .u8 => {
                try writer.print("    tensMath.qlinearsoftmax_lean(u8, f32, u8, {s}, {s}, {s}, {s}, {s}, {s}, {d}) catch return -1;\n", .{
                    tensor_x_string,
                    tensor_x_scale_string,
                    tensor_x_zero_point_string,
                    tensor_y_scale_string,
                    tensor_y_zero_point_string,
                    tensor_output_string,
                    axis_eff,
                });
            },
            .i8 => {
                try writer.print("    tensMath.qlinearsoftmax_lean(i8, f32, i8, {s}, {s}, {s}, {s}, {s}, {s}, {d}) catch return -1;\n", .{
                    tensor_x_string,
                    tensor_x_scale_string,
                    tensor_x_zero_point_string,
                    tensor_y_scale_string,
                    tensor_y_zero_point_string,
                    tensor_output_string,
                    axis_eff,
                });
            },
            else => return error.UnsupportedDataType,
        }
    }

    pub fn compute_output_shape(self: QLinearSoftmax) []usize {
        // Softmax preserves the input shape
        self.output_y.shape = self.input_x.shape;
        return self.output_y.shape;
    }

    pub fn print(op: *const QLinearSoftmax) void {
        std.debug.print("\n QLINEAR_SOFTMAX:\n {any}", .{op});
    }

    pub fn sobstitute_tensors(self: *QLinearSoftmax, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input_x == old_tensor) {
            self.input_x = new_tensor;
            return;
        }
        if (self.input_x_scale == old_tensor) {
            self.input_x_scale = new_tensor;
            return;
        }
        if (self.input_x_zero_point == old_tensor) {
            self.input_x_zero_point = new_tensor;
            return;
        }
        if (self.input_y_scale == old_tensor) {
            self.input_y_scale = new_tensor;
            return;
        }
        if (self.input_y_zero_point == old_tensor) {
            self.input_y_zero_point = new_tensor;
            return;
        }
        if (self.output_y == old_tensor) {
            self.output_y = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
};
