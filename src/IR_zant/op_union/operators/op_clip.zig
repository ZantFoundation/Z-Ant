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

// --- uops ---
const cg_v2 = @import("codegen").codegen_v2;
const Uops = cg_v2.uops;
const UOpBuilder = cg_v2.builder;
const DType = Uops.DType;
const Any = Uops.Any;

// https://onnx.ai/onnx/operators/onnx__Clip.html
// INPUTS:
//      - input (T) T: Input tensor whose elements to be clipped
//      - min (T) T: Minimum value, under which element is replaced by min. Optional, default -inf.
//      - max (T) T: Maximum value, above which element is replaced by max. Optional, default +inf.
// OUTPUTS:
//      - output (T) T: Output tensor with clipped input elements
pub const Clip = struct {
    input: *TensorZant,
    min: ?*TensorZant,
    max: ?*TensorZant,
    output: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Clip {
        const input = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_notFound;
        const output = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_notFound;

        // Optional min and max inputs
        const min = if (nodeProto.input.len > 1) tensorZant_lib.tensorMap.getPtr(nodeProto.input[1]) else null;
        const max = if (nodeProto.input.len > 2) tensorZant_lib.tensorMap.getPtr(nodeProto.input[2]) else null;

        //set the output type:
        if (output.ty == tensorZant_lib.TensorType.undefined) output.ty = input.ty;

        return Clip{
            .input = input,
            .min = min,
            .max = max,
            .output = output,
        };
    }

    pub fn get_output_shape(self: Clip) []usize {
        return self.output.getShape();
    }

    pub fn get_input_tensors(self: Clip) ![]*TensorZant {
        var input_tensors = std.ArrayList(*TensorZant).init(allocator);
        defer input_tensors.deinit();

        // Append the mandatory input tensor
        try input_tensors.append(self.input);

        // Append optional min and max tensors if they exist
        if (self.min) |min_tensor| {
            try input_tensors.append(min_tensor);
        }
        if (self.max) |max_tensor| {
            try input_tensors.append(max_tensor);
        }

        return input_tensors.toOwnedSlice();
    }

    pub fn get_output_tensors(self: Clip) ![]*TensorZant {
        var output_tensors = std.ArrayList(*TensorZant).init(allocator);
        defer output_tensors.deinit();

        // Append the single output tensor
        try output_tensors.append(self.output);

        return output_tensors.toOwnedSlice();
    }

    pub fn write_op(self: Clip, writer: std.fs.File.Writer) !void {
        // Create tensor string for input
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

        // Create tensor strings for min and max if they exist
        var min_tensor_str: []const u8 = "null";
        var max_tensor_str: []const u8 = "null";
        var min_tensor_string: []u8 = undefined;
        var max_tensor_string: []u8 = undefined;
        var need_free_min = false;
        var need_free_max = false;
        defer if (need_free_min) allocator.free(min_tensor_string);
        defer if (need_free_max) allocator.free(max_tensor_string);

        if (self.min) |min| {
            if (min.tc == TensorCategory.INITIALIZER) {
                min_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                    "@constCast(&param_lib.tensor_",
                    try utils.getSanitizedName(min.name),
                    ")",
                });
            } else {
                min_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(min.name) });
            }
            min_tensor_str = min_tensor_string;
            need_free_min = true;
        }

        if (self.max) |max| {
            if (max.tc == TensorCategory.INITIALIZER) {
                max_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                    "@constCast(&param_lib.tensor_",
                    try utils.getSanitizedName(max.name),
                    ")",
                });
            } else {
                max_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(max.name) });
            }
            max_tensor_str = max_tensor_string;
            need_free_max = true;
        }

        try writer.print(
            \\    tensMath.clip_lean(
            \\       {s},  //input type
            \\       {s},  //input tensor
            \\       {s},  //min tensor
            \\       {s},  //max tensor
            \\       &tensor_{s},  //output tensor
            \\    ) catch return -1;
            \\
        , .{
            self.input.ty.toString(),
            input_tensor_string,
            min_tensor_str,
            max_tensor_str,
            try utils.getSanitizedName(self.output.name),
        });
    }

    /// Optimized write operation for quantized clip pattern.
    /// This should be called when we detect the pattern:
    /// DequantizeLinear -> Clip -> QuantizeLinear
    pub fn write_op_quantized_pattern(input_quantized_tensor: *TensorZant, input_scale_tensor: *TensorZant, input_zero_point_tensor: *TensorZant, _: *TensorZant, output_scale_tensor: *TensorZant, output_zero_point_tensor: *TensorZant, min_val: f32, max_val: f32, writer: std.fs.File.Writer) !void {
        // Helper to create tensor strings
        const createTensorStr = struct {
            fn call(tensor: *TensorZant) ![]u8 {
                if (tensor.tc == TensorCategory.INITIALIZER) {
                    return try std.mem.concat(allocator, u8, &[_][]const u8{
                        "@constCast(&param_lib.tensor_",
                        try utils.getSanitizedName(tensor.name),
                        ")",
                    });
                } else {
                    return try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(tensor.name) });
                }
            }
        }.call;

        const str_input_quantized = try createTensorStr(input_quantized_tensor);
        defer allocator.free(str_input_quantized);
        const str_input_scale = try createTensorStr(input_scale_tensor);
        defer allocator.free(str_input_scale);
        const str_input_zero_point = try createTensorStr(input_zero_point_tensor);
        defer allocator.free(str_input_zero_point);
        const str_output_scale = try createTensorStr(output_scale_tensor);
        defer allocator.free(str_output_scale);
        const str_output_zero_point = try createTensorStr(output_zero_point_tensor);
        defer allocator.free(str_output_zero_point);

        try writer.print(
            \\    tensMath.clip_quantized_lean(
            \\        {s}, // InputType
            \\        {s}, // input tensor
            \\        {s}.data[0], // input_scale
            \\        {s}.data[0], // input_zero_point
            \\        {d:.6}, // min_val
            \\        {d:.6}, // max_val
            \\        @constCast({s}), // output = input (in-place)
            \\        {s}.data[0], // output_scale
            \\        {s}.data[0], // output_zero_point
            \\    ) catch return -1;
            \\
        , .{
            input_quantized_tensor.ty.toString(),
            str_input_quantized,
            str_input_scale,
            str_input_zero_point,
            min_val,
            max_val,
            str_input_quantized,
            str_output_scale,
            str_output_zero_point,
        });
    }

    pub fn compute_output_shape(self: Clip) []usize {
        // Clip preserves the input shape
        const input_shape = self.input.getShape();
        self.output.shape = input_shape;
        return input_shape;
    }

    pub fn print(self: Clip) void {
        std.debug.print("\n Clip:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *Clip, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input == old_tensor) {
            self.input = new_tensor;
            return;
        }
        if (self.min != null and self.min.? == old_tensor) {
            self.min = new_tensor;
            return;
        }
        if (self.max != null and self.max.? == old_tensor) {
            self.max = new_tensor;
            return;
        }
        if (self.output == old_tensor) {
            self.output = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
};
