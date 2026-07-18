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

// https://onnx.ai/onnx/operators/onnx__Identity.html#l-onnx-doc-identity
// INPUTS:
//      - input (heterogeneous) - V:  input tensor.
// OUTPUTS:
//      - output (heterogeneous) - V:  output tensor.

pub const Identity = struct {
    input: *TensorZant,
    output: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Identity {
        const input = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_notFound;
        const output = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_notFound;

        //set the output type:
        if (output.ty == tensorZant_lib.TensorType.undefined) output.ty = input.ty;

        return Identity{
            .input = input,
            .output = output,
        };
    }

    pub fn get_output_shape(self: Identity) []usize { // TODO
        return self.output.getShape();
    }

    pub fn get_input_tensors(self: Identity) ![]*TensorZant {
        var inputs: std.ArrayList(*TensorZant) = .empty;
        defer inputs.deinit(allocator);

        try inputs.append(allocator, self.input);

        return inputs.toOwnedSlice(allocator);
    }

    pub fn get_output_tensors(self: Identity) ![]*TensorZant {
        var outputs: std.ArrayList(*TensorZant) = .empty;
        defer outputs.deinit(allocator);

        try outputs.append(allocator, self.output);

        return outputs.toOwnedSlice(allocator);
    }

    pub fn write_op(self: Identity, writer: *std.Io.Writer) !void {
        // Create input tensor string
        var input_tensor_string: []u8 = undefined;
        defer allocator.free(input_tensor_string);

        if (self.input.tc == TensorCategory.INITIALIZER) {
            input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input.name),
                ")",
            });
        } else {
            input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.input.name) });
        }

        _ = try writer.print(
            \\
            \\
            \\    tensMath.identity_lean({s}, {s}, &tensor_{s}) catch return {d};
        , .{
            self.input.ty.toString(),
            input_tensor_string,
            try utils.getSanitizedName(self.output.name),
            utils.getMathErrorReturn(), // Error code for math errors
        });
    }

    pub fn compute_output_shape(self: Identity) []usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_identity_output_shape(self.input.ptr.?.get_shape());
        self.output.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Identity) void {
        std.debug.print("\n Identity:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *Identity, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input == old_tensor) {
            self.input = new_tensor;
            return;
        }
        if (self.output == old_tensor) {
            self.output = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }

    // https://onnx.ai/onnx/operators/onnx__Identity.html
};
