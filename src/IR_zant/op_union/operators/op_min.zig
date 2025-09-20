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

// https://onnx.ai/onnx/operators/onnx__Min.html
// INPUTS:
//      - data_0 (variadic, heterogeneous) - T: List of tensors for min.
// OUTPUTS:
//      - min (heterogeneous) - T: Output tensor.

pub const Min = struct {
    inputs: []*TensorZant,
    output: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Min {
        if (nodeProto.input.len == 0) return error.no_inputs;

        var inputs = try allocator.alloc(*TensorZant, nodeProto.input.len);
        errdefer allocator.free(inputs);

        for (nodeProto.input, 0..) |input_name, i| {
            inputs[i] = if (tensorZant_lib.tensorMap.getPtr(input_name)) |ptr| ptr else return error.input_notFound;
        }

        const output = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_notFound;

        // Set the output type based on the first input
        if (output.ty == tensorZant_lib.TensorType.undefined) output.ty = inputs[0].ty;

        return Min{
            .inputs = inputs,
            .output = output,
        };
    }

    pub fn get_output_shape(self: Min) []usize {
        return self.output.getShape();
    }

    pub fn get_input_tensors(self: Min) ![]*TensorZant {
        const result = try allocator.alloc(*TensorZant, self.inputs.len);
        @memcpy(result, self.inputs);
        return result;
    }

    pub fn get_output_tensors(self: Min) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();
        try outputs.append(self.output);
        return outputs.toOwnedSlice();
    }

    pub fn print(self: Min) void {
        std.debug.print("\n Min: inputs={d}, output={s}", .{ self.inputs.len, self.output.name });
    }

    pub fn sobstitute_tensors(self: *Min, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        for (self.inputs, 0..) |tensor, i| {
            if (tensor == old_tensor) {
                self.inputs[i] = new_tensor;
                return;
            }
        }
        if (self.output == old_tensor) {
            self.output = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }

    pub fn write_op(self: Min, writer: std.fs.File.Writer) !void {
        if (self.inputs.len == 0) return;

        // Handle simple case of two inputs (most common)
        if (self.inputs.len == 2) {
            // Generate tensor strings for the two inputs
            var tensor1_string: []u8 = undefined;
            defer allocator.free(tensor1_string);
            var tensor2_string: []u8 = undefined;
            defer allocator.free(tensor2_string);

            if (self.inputs[0].tc == TensorCategory.INITIALIZER) {
                tensor1_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                    "@constCast(&param_lib.tensor_",
                    try utils.getSanitizedName(self.inputs[0].name),
                    ")",
                });
            } else {
                tensor1_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.inputs[0].name) });
            }

            if (self.inputs[1].tc == TensorCategory.INITIALIZER) {
                tensor2_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                    "@constCast(&param_lib.tensor_",
                    try utils.getSanitizedName(self.inputs[1].name),
                    ")",
                });
            } else {
                tensor2_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.inputs[1].name) });
            }

            _ = try writer.print(
                \\    tensMath.min_two_lean(
                \\        {s},
                \\        {s}, // First input
                \\        {s}, // Second input
                \\        &tensor_{s} // Output
                \\    ) catch return -1;
            , .{
                self.inputs[0].ty.toString(),
                tensor1_string,
                tensor2_string,
                try utils.getSanitizedName(self.output.name),
            });
        } else {
            // For multiple inputs, we need to create an array
            _ = try writer.print(
                \\    {{
                \\        var min_inputs = [_]*Tensor({s}){{ 
            , .{self.inputs[0].ty.toString()});

            for (self.inputs, 0..) |input, i| {
                var tensor_string: []u8 = undefined;
                defer allocator.free(tensor_string);

                if (input.tc == TensorCategory.INITIALIZER) {
                    tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                        "@constCast(&param_lib.tensor_",
                        try utils.getSanitizedName(input.name),
                        ")",
                    });
                } else {
                    tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(input.name) });
                }

                if (i == self.inputs.len - 1) {
                    _ = try writer.print("{s} }};\n", .{tensor_string});
                } else {
                    _ = try writer.print("{s}, ", .{tensor_string});
                }
            }

            _ = try writer.print(
                \\        tensMath.min_lean(
                \\            {s},
                \\            &min_inputs,
                \\            &tensor_{s}
                \\        ) catch return -1;
                \\    }}
            , .{
                self.inputs[0].ty.toString(),
                try utils.getSanitizedName(self.output.name),
            });
        }
    }

    pub fn deinit(self: Min) void {
        allocator.free(self.inputs);
    }
};
