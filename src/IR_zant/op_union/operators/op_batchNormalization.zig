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

// https://onnx.ai/onnx/operators/onnx__BatchNormalization.html
// INPUTS:
//      - X (heterogeneous) - T: Input data tensor from the previous operator; dimensions are in the form of (N x C x D1 x D2 … Dn), where N is the batch size, C is the number of channels. Statistics are computed for every channel of C over N and D1 to Dn dimensions. For image data, input dimensions become (N x C x H x W). The op also accepts single dimension input of size N in which case C is assumed to be 1
//      - scale (heterogeneous) - T1: Scale tensor of shape ©.
//      - B (heterogeneous) - T1: Bias tensor of shape ©.
//      - input_mean (heterogeneous) - T2: running (training) or estimated (testing) mean tensor of shape ©.
//      - input_var (heterogeneous) - T2: running (training) or estimated (testing) variance tensor of shape ©.
// OUTPUT:
//      - Y (heterogeneous) - T: The output tensor of the same shape as X
// ATTRIBUTES:
//      - epsilon - FLOAT (default is '1e-05'): The epsilon value to use to avoid division by zero.
//      - momentum - FLOAT (default is '0.9'): Factor used in computing the running mean and variance.e.g., running_mean = running_mean * momentum + mean * (1 - momentum).
//      - training_mode - INT (default is '0'): If set to true, it indicates BatchNormalization is being used for training, and outputs 1 and 2 are to be computed.

pub const BatchNormalization = struct {
    input_X: *TensorZant,
    scale: *TensorZant,
    B: *TensorZant,
    input_mean: *TensorZant,
    input_var: *TensorZant,
    output_Y: *TensorZant,
    //attributes:
    epsilon: f32, // default = 1e-05;
    momentum: f32, //default = 0.9;
    training_mode: bool, //default = flase;

    pub fn init(nodeProto: *NodeProto) !BatchNormalization {
        const input_X = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const scale = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.scale_notFound;
        const B = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[2])) |ptr| ptr else return error.B_notFound;
        const input_mean = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[3])) |ptr| ptr else return error.input_mean_notFound;
        const input_var = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[4])) |ptr| ptr else return error.input_var_notFound;
        const output_Y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var epsilon: f32 = 1e-05;
        var momentum: f32 = 0.9;
        const training_mode: bool = false; // -> NOT USED, ALWAYS FALSE for Zant

        for (nodeProto.attribute) |attr| {
            if (std.mem.indexOf(u8, attr.name, "epsilon")) |_| {
                if (attr.type == onnx.AttributeType.FLOAT) epsilon = attr.f else return error.BatchNorm_epsilon_NotFloat;
            } else if (std.mem.indexOf(u8, attr.name, "momentum")) |_| {
                if (attr.type == onnx.AttributeType.FLOAT) momentum = attr.f else return error.BatchNorm_momentum_NotFloat;
            } else if (std.mem.indexOf(u8, attr.name, "training_mode")) |_| {
                if (attr.type == onnx.AttributeType.INT) if (attr.i != 0) return error.BatchNorm_training_NotAvailable;
            }
        }

        //set the output type:
        if (output_Y.ty == tensorZant_lib.TensorType.undefined) output_Y.ty = input_X.ty;

        return BatchNormalization{
            .input_X = input_X,
            .scale = scale,
            .B = B,
            .input_mean = input_mean,
            .input_var = input_var,
            .output_Y = output_Y,
            .epsilon = epsilon,
            .momentum = momentum,
            .training_mode = training_mode,
        };
    }

    pub fn get_output_shape(self: BatchNormalization) []usize {
        return self.output_Y.getShape();
    }

    pub fn get_input_tensors(self: BatchNormalization) ![]*TensorZant {
        var input_tensors = std.ArrayList(*TensorZant).init(allocator);
        defer input_tensors.deinit();

        try input_tensors.append(self.input_X);
        try input_tensors.append(self.scale);
        try input_tensors.append(self.B);
        try input_tensors.append(self.input_mean);
        try input_tensors.append(self.input_var);

        return input_tensors.toOwnedSlice();
    }

    pub fn get_output_tensors(self: BatchNormalization) ![]*TensorZant {
        var output_tensors = std.ArrayList(*TensorZant).init(allocator);
        defer output_tensors.deinit();

        try output_tensors.append(self.output_Y);

        return output_tensors.toOwnedSlice();
    }

    pub fn write_op(self: BatchNormalization, writer: std.fs.File.Writer) !void {

        //----create tensor_X_string
        var tensor_X_string: []u8 = undefined;
        defer allocator.free(tensor_X_string);

        if (self.input_X.tc == TensorCategory.INITIALIZER) {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_X.name),
                ")",
            });
        } else {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_X.name), ")" });
        }

        //----create tensor_scale_string
        var tensor_scale_string: []u8 = undefined;
        defer allocator.free(tensor_scale_string);

        if (self.scale.tc == TensorCategory.INITIALIZER) {
            tensor_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.scale.name),
                ")",
            });
        } else {
            tensor_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.scale.name), ")" });
        }

        //----create tensor_B_string
        var tensor_B_string: []u8 = undefined;
        defer allocator.free(tensor_B_string);

        if (self.B.tc == TensorCategory.INITIALIZER) {
            tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.B.name),
                ")",
            });
        } else {
            tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.B.name), ")" });
        }

        //----create tensor_input_mean_string
        var tensor_input_mean_string: []u8 = undefined;
        defer allocator.free(tensor_input_mean_string);

        if (self.input_mean.tc == TensorCategory.INITIALIZER) {
            tensor_input_mean_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_mean.name),
                ")",
            });
        } else {
            tensor_input_mean_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_mean.name), ")" });
        }

        //----create tensor_input_var_string
        var tensor_input_var_string: []u8 = undefined;
        defer allocator.free(tensor_input_var_string);

        if (self.input_var.tc == TensorCategory.INITIALIZER) {
            tensor_input_var_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_var.name),
                ")",
            });
        } else {
            tensor_input_var_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_var.name), ")" });
        }

        // pub inline fn batchNormalization_lean( comptime T: anytype, comptime T1: anytype, comptime T2: anytype, input: *Tensor(T), scales: *Tensor(T1), B: *Tensor(T1), input_mean: Tensor(T2), input_var: Tensor(T2), epsilon: f32, momentum: f32, training_mode: bool, output: *Tensor(T))
        _ = try writer.print(
            \\    
            \\
            \\    tensMath.batchNormalization_lean(
            \\        {s}, //type 0
            \\        {s}, //type 1
            \\        {s}, //type 2
            \\        {s}, //input
            \\        {s}, //scales
            \\        {s}, //B
            \\        {s}, //input_mean
            \\        {s}, //input_var
            \\        {}, //epsilon
            \\        {}, //momentum
            \\        false, //training_mode
            \\        &tensor_{s}, //output
            \\    ) catch return -1;
        , .{
            self.input_X.ty.toString(),
            self.scale.ty.toString(),
            self.input_mean.ty.toString(),
            tensor_X_string,
            tensor_scale_string,
            tensor_B_string,
            tensor_input_mean_string,
            tensor_input_var_string,
            self.epsilon,
            self.momentum,
            try utils.getSanitizedName(self.output_Y.name),
        });
    }

    pub fn compute_output_shape(self: BatchNormalization) []usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_batchNormalization_output_shape(self.input_X.get_shape());
        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: BatchNormalization) void { // TODO
        std.debug.print("\n BatchNormalization:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *BatchNormalization, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input_X == old_tensor) {
            self.input_X = new_tensor;
            return;
        }
        if (self.scale == old_tensor) {
            self.scale = new_tensor;
            return;
        }
        if (self.B == old_tensor) {
            self.B = new_tensor;
            return;
        }
        if (self.input_mean == old_tensor) {
            self.input_mean = new_tensor;
            return;
        }
        if (self.input_var == old_tensor) {
            self.input_var = new_tensor;
            return;
        }
        if (self.output_Y == old_tensor) {
            self.output_Y = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
};
