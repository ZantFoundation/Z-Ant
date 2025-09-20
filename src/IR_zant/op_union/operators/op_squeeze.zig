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

// https://onnx.ai/onnx/operators/onnx__Squeeze.html
// INPUTS:
//      - data (heterogeneous) - T: Tensors with at least max(dims) dimensions.
//      - axes (optional, heterogeneous) - tensor(int64): List of integers indicating the dimensions to squeeze.
// OUTPUTS:
//      - squeezed (heterogeneous) - T: Reshaped tensor with same data as input.

pub const Squeeze = struct {
    input_data: *TensorZant,
    input_axes: ?*TensorZant, // Optional
    output: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Squeeze {
        const input_data = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_data_notFound;
        const input_axes: ?*TensorZant = if (nodeProto.input.len >= 2)
            if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else null
        else
            null;
        const output = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_notFound;

        //set the output type:
        if (output.ty == tensorZant_lib.TensorType.undefined) output.ty = input_data.ty;

        return Squeeze{
            .input_data = input_data,
            .input_axes = input_axes,
            .output = output,
        };
    }

    pub fn get_output_shape(self: Squeeze) []usize {
        return self.compute_output_shape() catch {
            // Fallback to a default shape in case of error
            std.log.warn("[SQUEEZE DEBUG] Failed to compute output shape, using fallback", .{});
            const fallback_shape = allocator.alloc(usize, 1) catch unreachable;
            fallback_shape[0] = 1;
            return fallback_shape;
        };
    }

    pub fn compute_output_shape(self: Squeeze) ![]usize {
        // Convert axes tensor to slice if present
        const axes_data = if (self.input_axes) |axes|
            if (axes.ptr) |ptr| ptr.i64.data else null
        else
            null;

        const output_shape = try tensorMath.get_squeeze_output_shape(
            self.input_data.shape,
            axes_data,
        );
        self.output.shape = output_shape;
        return output_shape;
    }

    pub fn get_input_tensors(self: Squeeze) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();
        try inputs.append(self.input_data);
        if (self.input_axes) |axes| try inputs.append(axes);
        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: Squeeze) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();
        try outputs.append(self.output);
        return outputs.toOwnedSlice();
    }

    pub fn print(self: Squeeze) void {
        std.debug.print("\n Squeeze: {any}", .{self});
    }

    pub fn write_op(self: Squeeze, writer: std.fs.File.Writer) !void {
        // Input data tensor string
        var tensor_data_string: []u8 = undefined;
        defer allocator.free(tensor_data_string);

        if (self.input_data.tc == TensorCategory.INITIALIZER) {
            tensor_data_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_data.name),
                ")",
            });
        } else {
            tensor_data_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "&tensor_",
                try utils.getSanitizedName(self.input_data.name),
            });
        }

        // Print the squeeze operation (squeeze_lean doesn't take axes parameter)
        _ = try writer.print(
            \\    tensMath.squeeze_lean(
            \\        {s},
            \\        {s}, // Input tensor
            \\        &tensor_{s} // Output tensor
            \\    ) catch return -1;
        , .{
            self.input_data.ty.toString(),
            tensor_data_string,
            try utils.getSanitizedName(self.output.name),
        });
    }

    pub fn sobstitute_tensors(self: *Squeeze, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input_data == old_tensor) {
            self.input_data = new_tensor;
            return;
        }
        if (self.input_axes != null and self.input_axes.? == old_tensor) {
            self.input_axes = new_tensor;
            return;
        }
        if (self.output == old_tensor) {
            self.output = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
};
