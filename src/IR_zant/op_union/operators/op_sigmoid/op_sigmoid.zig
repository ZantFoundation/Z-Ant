const std = @import("std");
const allocator = std.heap.page_allocator;
const IR_zant = @import("IR_zant");

// --- onnx ---
const onnx = IR_zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

// --- zant ---
const tensorZant_lib = IR_zant.tensorZant_lib;
const TensorZant = tensorZant_lib.TensorZant;
const TensorCategory = tensorZant_lib.TensorCategory;

const tensorMath = IR_zant.core.math_standard;

const utils = IR_zant.utils;

//https://onnx.ai/onnx/operators/onnx__Sigmoid.html
// INPUTS:
//      - X (heterogeneous) - T: Input tensor
// OUTPUTS:
//      - Y (heterogeneous) - T: Output tensor
pub const Sigmoid = struct {
    input_X: *TensorZant,
    output_Y: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Sigmoid {
        const input_X = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const output_Y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        //set the output type:
        if (output_Y.ty == tensorZant_lib.TensorType.undefined) output_Y.ty = input_X.ty;

        return Sigmoid{
            .input_X = input_X,
            .output_Y = output_Y,
        };
    }

    pub fn get_output_shape(self: Sigmoid) []usize {
        return self.output_Y.getShape();
    }

    pub fn get_input_tensors(self: Sigmoid) ![]*TensorZant {
        var inputs: std.ArrayList(*TensorZant) = .empty;
        defer inputs.deinit(allocator);
        try inputs.append(allocator, self.input_X);
        return inputs.toOwnedSlice(allocator);
    }

    pub fn get_output_tensors(self: Sigmoid) ![]*TensorZant {
        var outputs: std.ArrayList(*TensorZant) = .empty;
        defer outputs.deinit(allocator);
        try outputs.append(allocator, self.output_Y);
        return outputs.toOwnedSlice(allocator);
    }

    pub fn write_op(self: Sigmoid, writer: *std.Io.Writer) !void {
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
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.input_X.name) });
        }

        _ = try writer.print(
            \\
            \\    tensMath.sigmoid_lean(
            \\      {s},
            \\      {s},
            \\      &tensor_{s},
            \\    ) catch return {d};
        ,
            .{
                self.input_X.ty.toString(),
                tensor_X_string,
                try utils.getSanitizedName(self.output_Y.name),
                utils.getMathErrorReturn(), // Error code for math errors
            },
        );
    }

    pub fn compute_output_shape(self: Sigmoid) []usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_sigmoid_output_shape(self.input_X.shape);
        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Sigmoid) void {
        std.debug.print("\n Sigmoid: {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *Sigmoid, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
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
