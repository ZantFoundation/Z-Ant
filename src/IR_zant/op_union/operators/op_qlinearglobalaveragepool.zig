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

// https://onnx.ai/onnx/operators/onnx__QLinearGlobalAveragePool.html
// INPUTS:
//      - X (heterogeneous) - T: Input tensor (quantized)
//      - X_scale (heterogeneous) - tensor(float): Scale of quantization of input X
//      - X_zero_point (heterogeneous) - T: Zero point of quantization of input X
//      - Y_scale (heterogeneous) - tensor(float): Scale of quantization of output Y
//      - Y_zero_point (heterogeneous) - T: Zero point of quantization of output Y
// OUTPUTS:
//      - Y (heterogeneous) - T: Output tensor (quantized)

pub const QLinearGlobalAveragePool = struct {
    input_X: *TensorZant,
    input_X_scale: *TensorZant,
    input_X_zero_point: *TensorZant,
    input_Y_scale: *TensorZant,
    input_Y_zero_point: *TensorZant,
    output_Y: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !QLinearGlobalAveragePool {
        // QLinearGlobalAveragePool has exactly 5 inputs
        if (nodeProto.input.len != 5) {
            return error.QLinearGlobalAveragePoolInvalidInputCount;
        }

        const input_X = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const input_X_scale = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_X_scale_notFound;
        const input_X_zero_point = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[2])) |ptr| ptr else return error.input_X_zero_point_notFound;
        const input_Y_scale = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[3])) |ptr| ptr else return error.input_Y_scale_notFound;
        const input_Y_zero_point = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[4])) |ptr| ptr else return error.input_Y_zero_point_notFound;

        const output_Y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        //set the output type:
        if (output_Y.ty == tensorZant_lib.TensorType.undefined) output_Y.ty = input_X.ty;

        const qlinear_gap = QLinearGlobalAveragePool{
            .input_X = input_X,
            .input_X_scale = input_X_scale,
            .input_X_zero_point = input_X_zero_point,
            .input_Y_scale = input_Y_scale,
            .input_Y_zero_point = input_Y_zero_point,
            .output_Y = output_Y,
        };

        // Force shape computation during initialization
        _ = qlinear_gap.compute_output_shape() catch {};

        return qlinear_gap;
    }

    pub fn get_output_shape(self: QLinearGlobalAveragePool) []usize {
        return self.output_Y.getShape();
    }

    pub fn get_input_tensors(self: QLinearGlobalAveragePool) ![]*TensorZant {
        var input_tensors = std.ArrayList(*TensorZant).init(allocator);
        defer input_tensors.deinit();

        try input_tensors.append(self.input_X);
        try input_tensors.append(self.input_X_scale);
        try input_tensors.append(self.input_X_zero_point);
        try input_tensors.append(self.input_Y_scale);
        try input_tensors.append(self.input_Y_zero_point);

        return input_tensors.toOwnedSlice();
    }

    pub fn get_output_tensors(self: QLinearGlobalAveragePool) ![]*TensorZant {
        var output_tensors = std.ArrayList(*TensorZant).init(allocator);
        defer output_tensors.deinit();

        try output_tensors.append(self.output_Y);

        return output_tensors.toOwnedSlice();
    }

    pub fn write_op(self: QLinearGlobalAveragePool, writer: std.fs.File.Writer) !void {
        // Create tensor string variables for each input
        var tensor_X_string: []u8 = undefined;
        defer allocator.free(tensor_X_string);
        var tensor_X_scale_string: []u8 = undefined;
        defer allocator.free(tensor_X_scale_string);
        var tensor_X_zero_point_string: []u8 = undefined;
        defer allocator.free(tensor_X_zero_point_string);
        var tensor_Y_scale_string: []u8 = undefined;
        defer allocator.free(tensor_Y_scale_string);
        var tensor_Y_zero_point_string: []u8 = undefined;
        defer allocator.free(tensor_Y_zero_point_string);

        // Generate tensor strings for each input
        if (self.input_X.tc == TensorCategory.INITIALIZER) {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_X.name), ")" });
        } else {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_X.name), ")" });
        }

        if (self.input_X_scale.tc == TensorCategory.INITIALIZER) {
            tensor_X_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_X_scale.name), ")" });
        } else {
            tensor_X_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_X_scale.name), ")" });
        }

        if (self.input_X_zero_point.tc == TensorCategory.INITIALIZER) {
            tensor_X_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_X_zero_point.name), ")" });
        } else {
            tensor_X_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_X_zero_point.name), ")" });
        }

        if (self.input_Y_scale.tc == TensorCategory.INITIALIZER) {
            tensor_Y_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_Y_scale.name), ")" });
        } else {
            tensor_Y_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_Y_scale.name), ")" });
        }

        if (self.input_Y_zero_point.tc == TensorCategory.INITIALIZER) {
            tensor_Y_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_Y_zero_point.name), ")" });
        } else {
            tensor_Y_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_Y_zero_point.name), ")" });
        }

        // Write the operation call
        try writer.print("\n    tensMath.qlinearglobalaveragepool_lean(\n", .{});
        try writer.print("        {s},\n", .{tensor_X_string});
        try writer.print("        {s},\n", .{tensor_X_scale_string});
        try writer.print("        {s},\n", .{tensor_X_zero_point_string});
        try writer.print("        &tensor_{s},\n", .{try utils.getSanitizedName(self.output_Y.name)});
        try writer.print("        {s},\n", .{tensor_Y_scale_string});
        try writer.print("        {s},\n", .{tensor_Y_zero_point_string});
        try writer.print("    ) catch return -1;\n", .{});
    }

    pub fn compute_output_shape(self: QLinearGlobalAveragePool) ![]usize {
        // Input [N, C, H, W] (or more dims) -> Output [N, C, 1, 1, ...] with same rank
        const input_shape = self.input_X.getShape();
        if (input_shape.len < 2) return error.InvalidDimensions;

        const output_shape = try allocator.alloc(usize, if (input_shape.len < 4) 4 else input_shape.len);
        // Force at least 4D with last 2 dims = 1
        const n = input_shape[0];
        const c = input_shape[1];
        output_shape[0] = n;
        output_shape[1] = c;
        if (output_shape.len >= 3) output_shape[2] = 1;
        if (output_shape.len >= 4) output_shape[3] = 1;
        // If rank > 4, set remaining spatial dims to 1
        if (output_shape.len > 4) {
            for (4..output_shape.len) |i| output_shape[i] = 1;
        }

        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn sobstitute_tensors(self: *QLinearGlobalAveragePool, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input_X == old_tensor) {
            self.input_X = new_tensor;
            return;
        }
        if (self.input_X_scale == old_tensor) {
            self.input_X_scale = new_tensor;
            return;
        }
        if (self.input_X_zero_point == old_tensor) {
            self.input_X_zero_point = new_tensor;
            return;
        }
        if (self.input_Y_scale == old_tensor) {
            self.input_Y_scale = new_tensor;
            return;
        }
        if (self.input_Y_zero_point == old_tensor) {
            self.input_Y_zero_point = new_tensor;
            return;
        }
        if (self.output_Y == old_tensor) {
            self.output_Y = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
};
