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

// https://onnx.ai/onnx/operators/onnx__QLinearMatMul.html
// INPUTS:
//      - a (heterogeneous) - T1: N-dimensional matrix A (quantized)
//      - a_scale (heterogeneous) - tensor(float): Scale of quantization of input a
//      - a_zero_point (heterogeneous) - T1: Zero point of quantization of input a
//      - b (heterogeneous) - T1: N-dimensional matrix B (quantized)
//      - b_scale (heterogeneous) - tensor(float): Scale of quantization of input b
//      - b_zero_point (heterogeneous) - T1: Zero point of quantization of input b
//      - y_scale (heterogeneous) - tensor(float): Scale of quantization of output y
//      - y_zero_point (heterogeneous) - T1: Zero point of quantization of output y
// OUTPUTS:
//      - y (heterogeneous) - T1: Matrix multiply results from A * B (quantized)

pub const QLinearMatMul = struct {
    input_a: *TensorZant,
    input_a_scale: *TensorZant,
    input_a_zero_point: *TensorZant,
    input_b: *TensorZant,
    input_b_scale: *TensorZant,
    input_b_zero_point: *TensorZant,
    input_y_scale: *TensorZant,
    input_y_zero_point: *TensorZant,
    output_y: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !QLinearMatMul {
        // QLinearMatMul has exactly 8 inputs
        if (nodeProto.input.len != 8) {
            return error.QLinearMatMulInvalidInputCount;
        }

        const input_a = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_a_notFound;
        const input_a_scale = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_a_scale_notFound;
        const input_a_zero_point = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[2])) |ptr| ptr else return error.input_a_zero_point_notFound;
        const input_b = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[3])) |ptr| ptr else return error.input_b_notFound;
        const input_b_scale = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[4])) |ptr| ptr else return error.input_b_scale_notFound;
        const input_b_zero_point = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[5])) |ptr| ptr else return error.input_b_zero_point_notFound;
        const input_y_scale = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[6])) |ptr| ptr else return error.input_y_scale_notFound;
        const input_y_zero_point = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[7])) |ptr| ptr else return error.input_y_zero_point_notFound;

        const output_y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_y_notFound;

        //set the output type:
        if (output_y.ty == tensorZant_lib.TensorType.undefined) output_y.ty = input_a.ty;

        return QLinearMatMul{
            .input_a = input_a,
            .input_a_scale = input_a_scale,
            .input_a_zero_point = input_a_zero_point,
            .input_b = input_b,
            .input_b_scale = input_b_scale,
            .input_b_zero_point = input_b_zero_point,
            .input_y_scale = input_y_scale,
            .input_y_zero_point = input_y_zero_point,
            .output_y = output_y,
        };
    }

    pub fn get_output_shape(self: QLinearMatMul) []usize {
        return self.output_y.getShape();
    }

    pub fn get_input_tensors(self: QLinearMatMul) ![]*TensorZant {
        var input_tensors = std.ArrayList(*TensorZant).init(allocator);
        defer input_tensors.deinit();

        try input_tensors.append(self.input_a);
        try input_tensors.append(self.input_a_scale);
        try input_tensors.append(self.input_a_zero_point);
        try input_tensors.append(self.input_b);
        try input_tensors.append(self.input_b_scale);
        try input_tensors.append(self.input_b_zero_point);
        try input_tensors.append(self.input_y_scale);
        try input_tensors.append(self.input_y_zero_point);

        return input_tensors.toOwnedSlice();
    }

    pub fn get_output_tensors(self: QLinearMatMul) ![]*TensorZant {
        var output_tensors = std.ArrayList(*TensorZant).init(allocator);
        defer output_tensors.deinit();

        try output_tensors.append(self.output_y);

        return output_tensors.toOwnedSlice();
    }

    pub fn write_op(self: QLinearMatMul, writer: std.fs.File.Writer) !void {
        // Create tensor string variables for each input
        var tensor_a_string: []u8 = undefined;
        defer allocator.free(tensor_a_string);
        var tensor_a_scale_string: []u8 = undefined;
        defer allocator.free(tensor_a_scale_string);
        var tensor_a_zero_point_string: []u8 = undefined;
        defer allocator.free(tensor_a_zero_point_string);
        var tensor_b_string: []u8 = undefined;
        defer allocator.free(tensor_b_string);
        var tensor_b_scale_string: []u8 = undefined;
        defer allocator.free(tensor_b_scale_string);
        var tensor_b_zero_point_string: []u8 = undefined;
        defer allocator.free(tensor_b_zero_point_string);
        var tensor_y_scale_string: []u8 = undefined;
        defer allocator.free(tensor_y_scale_string);
        var tensor_y_zero_point_string: []u8 = undefined;
        defer allocator.free(tensor_y_zero_point_string);

        // Generate tensor strings for each input
        if (self.input_a.tc == TensorCategory.INITIALIZER) {
            tensor_a_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_a.name), ")" });
        } else {
            tensor_a_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_a.name), ")" });
        }

        if (self.input_a_scale.tc == TensorCategory.INITIALIZER) {
            tensor_a_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_a_scale.name), ")" });
        } else {
            tensor_a_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_a_scale.name), ")" });
        }

        if (self.input_a_zero_point.tc == TensorCategory.INITIALIZER) {
            tensor_a_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_a_zero_point.name), ")" });
        } else {
            tensor_a_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_a_zero_point.name), ")" });
        }

        if (self.input_b.tc == TensorCategory.INITIALIZER) {
            tensor_b_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_b.name), ")" });
        } else {
            tensor_b_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_b.name), ")" });
        }

        if (self.input_b_scale.tc == TensorCategory.INITIALIZER) {
            tensor_b_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_b_scale.name), ")" });
        } else {
            tensor_b_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_b_scale.name), ")" });
        }

        if (self.input_b_zero_point.tc == TensorCategory.INITIALIZER) {
            tensor_b_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_b_zero_point.name), ")" });
        } else {
            tensor_b_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_b_zero_point.name), ")" });
        }

        if (self.input_y_scale.tc == TensorCategory.INITIALIZER) {
            tensor_y_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_y_scale.name), ")" });
        } else {
            tensor_y_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_y_scale.name), ")" });
        }

        if (self.input_y_zero_point.tc == TensorCategory.INITIALIZER) {
            tensor_y_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_y_zero_point.name), ")" });
        } else {
            tensor_y_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_y_zero_point.name), ")" });
        }

        // Write the operation call
        try writer.print("\n    tensMath.qlinearmatmul_lean(\n", .{});
        try writer.print("        {s},\n", .{tensor_a_string});
        try writer.print("        {s},\n", .{tensor_a_scale_string});
        try writer.print("        {s},\n", .{tensor_a_zero_point_string});
        try writer.print("        {s},\n", .{tensor_b_string});
        try writer.print("        {s},\n", .{tensor_b_scale_string});
        try writer.print("        {s},\n", .{tensor_b_zero_point_string});
        try writer.print("        &tensor_{s},\n", .{try utils.getSanitizedName(self.output_y.name)});
        try writer.print("        {s},\n", .{tensor_y_scale_string});
        try writer.print("        {s},\n", .{tensor_y_zero_point_string});
        try writer.print("    ) catch return -1;\n", .{});
    }

    pub fn sobstitute_tensors(self: *QLinearMatMul, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input_a == old_tensor) {
            self.input_a = new_tensor;
            return;
        }
        if (self.input_a_scale == old_tensor) {
            self.input_a_scale = new_tensor;
            return;
        }
        if (self.input_a_zero_point == old_tensor) {
            self.input_a_zero_point = new_tensor;
            return;
        }
        if (self.input_b == old_tensor) {
            self.input_b = new_tensor;
            return;
        }
        if (self.input_b_scale == old_tensor) {
            self.input_b_scale = new_tensor;
            return;
        }
        if (self.input_b_zero_point == old_tensor) {
            self.input_b_zero_point = new_tensor;
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
