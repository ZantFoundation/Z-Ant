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

//https://onnx.ai/onnx/operators/onnx__Softmax.html
// INPUTS:
//      - X (heterogeneous) - T: Input tensor
// OUTPUTS:
//      - Y (heterogeneous) - T: Output tensor
pub const Softmax = struct {
    input_X: *TensorZant,
    output_Y: *TensorZant,
    axis: i64,

    pub fn init(nodeProto: *NodeProto) !Softmax {
        const input_X = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const output_Y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;
        var axis: i64 = -1;

        for (nodeProto.attribute) |attr| {
            if (std.mem.indexOf(u8, attr.name, "axis")) |_| {
                if (attr.type == onnx.AttributeType.INT) axis = attr.i else return error.Axis_NotINT;
            }
        }
        //set the output type:
        if (output_Y.ty == tensorZant_lib.TensorType.undefined) output_Y.ty = input_X.ty;

        return Softmax{
            .input_X = input_X,
            .output_Y = output_Y,
            .axis = axis,
        };
    }

    pub fn get_output_shape(self: Softmax) []usize {
        return self.output_Y.shape;
    }

    pub fn get_input_tensors(self: Softmax) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();
        try inputs.append(self.input_X);
        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: Softmax) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();
        try outputs.append(self.output_Y);
        return outputs.toOwnedSlice();
    }

    pub fn compute_output_shape(self: Softmax) []usize {
        var output_shape: []usize = undefined;
        output_shape = self.input_X.shape;
        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Softmax) void {
        std.debug.print("\n Softmax: {any}", .{self});
    }

    pub fn write_op(self: Softmax, writer: std.fs.File.Writer) !void {
        var tensor_input_string: []u8 = undefined;
        defer allocator.free(tensor_input_string);

        if (self.input_X.tc == TensorCategory.INITIALIZER) {
            tensor_input_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_X.name),
                ")",
            });
        } else {
            tensor_input_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "&tensor_",
                try utils.getSanitizedName(self.input_X.name),
            });
        }

        _ = try writer.print(
            \\
            \\    tensMath.softmax_lean(
            \\        {s}, //Type
            \\        {s}, // input tensor
            \\        &tensor_{s}, // output tensor
            \\        {},
            \\    ) catch return -1;
            \\
        , .{
            self.output_Y.ty.toString(),
            tensor_input_string,
            try utils.getSanitizedName(self.output_Y.name),
            self.axis,
        });
    }

    pub fn sobstitute_tensors(self: *Softmax, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
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
