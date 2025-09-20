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
// https://onnx.ai/onnx/operators/onnx__Elu.html
// INPUTS:
//      - X (heterogeneous) - T: Input tensor
// OUTPUTS:
//      - Y (heterogeneous) - T: Output tensor
// ATTRIBUTES:
//      - alpha - FLOAT (default is '1.0'): Coefficient of ELU operator
pub const Elu = struct {
    input_X: *TensorZant,
    output_Y: *TensorZant,
    //attributes:
    alpha: f32, // default = 1.0,

    pub fn init(nodeProto: *NodeProto) !Elu {
        const input_X = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const output_Y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var alpha: f32 = 1.0;

        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "alpha")) {
                if (attr.type != onnx.AttributeType.FLOAT) {
                    return error.InvalidAttributeType;
                }
                alpha = attr.f;
            }
        }

        //set the output type:
        if (output_Y.ty == tensorZant_lib.TensorType.undefined) output_Y.ty = input_X.ty;

        return Elu{
            .input_X = input_X,
            .output_Y = output_Y,
            .alpha = alpha,
        };
    }

    pub fn get_output_shape(self: Elu) []usize {
        return self.output_Y.getShape();
    }

    pub fn get_input_tensors(self: Elu) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();

        try inputs.append(self.input_X);

        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: Elu) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();

        try outputs.append(self.output_Y);
        return outputs.toOwnedSlice();
    }

    pub fn write_op(self: Elu, writer: std.fs.File.Writer) !void {
        var input_tensor_string: []u8 = undefined;
        defer allocator.free(input_tensor_string);
        if (self.input_X.tc == TensorCategory.INITIALIZER) {
            input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_X.name),
                ")",
            });
        } else {
            input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&tensor_",
                try utils.getSanitizedName(self.input_X.name),
                ")",
            });
        }

        _ = try writer.print(
            \\
            \\    tensMath.elu_lean(
            \\        {s}, // type
            \\        {s}, // input
            \\        &tensor_{s}, // output
            \\        {d} // alpha
            \\    ) catch return -1;
        , .{
            self.input_X.ty.toString(),
            input_tensor_string,
            try utils.getSanitizedName(self.output_Y.name),
            self.alpha,
        });
    }

    pub fn print(self: Elu) void { // TODO
        std.debug.print("\n ADD:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *Elu, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
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
