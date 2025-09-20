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

// https://onnx.ai/onnx/operators/onnx__LeakyRelu.html#l-onnx-doc-leakyrelu
// INPUTS:
//      - X (heterogeneous) - T:  input tensor.
// OUTPUTS:
//      - Y (heterogeneous) - T:  output tensor.
// ATTRIBUTES:
//      - alpha (float) - coefficent of leakage. Default is 0.01.

pub const LeakyRelu = struct {
    input_X: *TensorZant,
    output_Y: *TensorZant,
    alpha: f32 = 0.01, // default value

    pub fn init(nodeProto: *NodeProto) !LeakyRelu {
        const input_X = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const output_Y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var alpha: f32 = 0.01; // default value
        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "alpha")) {
                alpha = attr.f;
            }
        }

        //set the output type:
        if (output_Y.ty == tensorZant_lib.TensorType.undefined) output_Y.ty = input_X.ty;

        return LeakyRelu{
            .input_X = input_X,
            .output_Y = output_Y,
            .alpha = alpha,
        };
    }

    pub fn get_output_shape(self: LeakyRelu) []usize { // TODO
        return self.output_Y.getShape();
    }

    pub fn get_input_tensors(self: LeakyRelu) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();

        try inputs.append(self.input_X);

        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: LeakyRelu) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();

        try outputs.append(self.output_Y);

        return outputs.toOwnedSlice();
    }

    pub fn write_op(self: LeakyRelu, writer: std.fs.File.Writer) !void {
        // Create input tensor string
        var input_tensor_string: []u8 = undefined;
        defer allocator.free(input_tensor_string);
        if (self.input_X.tc == TensorCategory.INITIALIZER) {
            input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_X.name),
                ")",
            });
        } else {
            input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.input_X.name) });
        }

        _ = try writer.print(
            \\
            \\    tensMath.leakyReLU_lean({s}, {s}, {d}, &tensor_{s}) catch return -1;
        , .{
            self.input_X.ty.toString(),
            input_tensor_string,
            self.alpha,
            try utils.getSanitizedName(self.output_Y.name),
        });
    }

    pub fn compute_output_shape(self: LeakyRelu) []usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_leaky_relu_output_shape(self.input_X.ptr.?.get_shape());
        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: LeakyRelu) void {
        std.debug.print("\n LeakyRelu:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *LeakyRelu, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input_X == old_tensor) {
            self.input_X = new_tensor;
            return;
        }
        if (self.output_Y == old_tensor) {
            self.output_Y = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
};
