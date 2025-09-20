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

//https://onnx.ai/onnx/operators/onnx__Unsqueeze.html
// INPUTS:
//      - X (heterogeneous) - T: Input tensor
//      - axes (heterogeneous) - T: Axes to unsqueeze
// OUTPUTS:
//      - Y (heterogeneous) - T: Output tensor
pub const Unsqueeze = struct {
    input_X: *TensorZant,
    input_axes: *TensorZant,
    output_Y: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Unsqueeze {
        const input_X = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const input_axes = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_axes_notFound;
        const output_Y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        //set the output type:
        if (output_Y.ty == tensorZant_lib.TensorType.undefined) output_Y.ty = input_X.ty;

        return Unsqueeze{
            .input_X = input_X,
            .input_axes = input_axes,
            .output_Y = output_Y,
        };
    }
    pub fn get_output_shape(self: Unsqueeze) []usize {
        return self.compute_output_shape() catch {
            // Fallback to a default shape in case of error
            std.log.warn("[UNSQUEEZE DEBUG] Failed to compute output shape, using fallback", .{});
            const fallback_shape = allocator.alloc(usize, 1) catch unreachable;
            fallback_shape[0] = 1;
            return fallback_shape;
        };
    }

    pub fn get_input_tensors(self: Unsqueeze) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();
        try inputs.append(self.input_X);
        try inputs.append(self.input_axes);
        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: Unsqueeze) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();
        try outputs.append(self.output_Y);
        return outputs.toOwnedSlice();
    }

    pub fn compute_output_shape(self: Unsqueeze) ![]usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_unsqueeze_output_shape(
            self.input_X.shape,
            self.input_axes.ptr.?.i64.data,
        );
        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Unsqueeze) void {
        std.debug.print("\n Unsqueeze: {any}", .{self});
    }

    pub fn write_op(self: Unsqueeze, writer: std.fs.File.Writer) !void {
        // Input tensor string
        var tensor_X_string: []u8 = undefined;
        defer allocator.free(tensor_X_string);

        if (self.input_X.tc == TensorCategory.INITIALIZER) {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_X.name),
                ")",
            });
        } else {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "&tensor_",
                try utils.getSanitizedName(self.input_X.name),
            });
        }

        // Axes tensor string
        var axes_string: []u8 = undefined;
        defer allocator.free(axes_string);

        if (self.input_axes.tc == TensorCategory.INITIALIZER) {
            axes_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_axes.name),
                ")",
            });
        } else {
            axes_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "&tensor_",
                try utils.getSanitizedName(self.input_axes.name),
            });
        }

        // Print the unsqueeze operation
        _ = try writer.print(
            \\    tensMath.unsqueeze_lean(
            \\        {s},
            \\        {s}, // Input tensor
            \\        {s}, // Axes tensor
            \\        &tensor_{s} // Output tensor
            \\    ) catch return -1;
        , .{
            self.input_X.ty.toString(),
            tensor_X_string,
            axes_string,
            try utils.getSanitizedName(self.output_Y.name),
        });
    }

    pub fn sobstitute_tensors(self: *Unsqueeze, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input_X == old_tensor) {
            self.input_X = new_tensor;
            return;
        }
        if (self.input_axes == old_tensor) {
            self.input_axes = new_tensor;
            return;
        }
        if (self.output_Y == old_tensor) {
            self.output_Y = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
};
