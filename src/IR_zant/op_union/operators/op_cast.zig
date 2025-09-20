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
const TensorType = tensorZant_lib.TensorType;

const tensorMath = zant.core.tensor.math_standard;
const utils = IR_zant.utils;

// https://onnx.ai/onnx/operators/onnx__Cast.html
// INPUTS:
//      - input (heterogeneous) - T1: Input tensor to be cast
// OUTPUTS:
//      - output (heterogeneous) - T2: Output tensor with the same shape but different type
// ATTRIBUTES:
//      - to (required) - INT: The data type to which the elements of the input tensor are cast
//      - saturate - INT (default is 1): Controls the saturation behavior when casting

pub const Cast = struct {
    input: *TensorZant,
    output: *TensorZant,

    // Attributes
    to: i64, // Target data type
    saturate: i64,

    pub fn init(nodeProto: *NodeProto) !Cast {
        // Cast has 1 input
        if (nodeProto.input.len != 1) {
            return error.CastInvalidInputCount;
        }

        const input = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_notFound;
        const output = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_notFound;

        // Default attributes
        var to: i64 = 0;
        var saturate: i64 = 1;

        // Parse attributes
        for (nodeProto.attribute) |attr| {
            if (std.mem.indexOf(u8, attr.name, "to")) |_| {
                if (attr.type == onnx.AttributeType.INT) to = attr.i else return error.ToNotINT;
            } else if (std.mem.indexOf(u8, attr.name, "saturate")) |_| {
                if (attr.type == onnx.AttributeType.INT) saturate = attr.i else return error.SaturateNotINT;
            }
        }

        if (to == 0) return error.ToAttributeRequired;

        return Cast{
            .input = input,
            .output = output,
            .to = to,
            .saturate = saturate,
        };
    }

    pub fn get_output_shape(op: *const Cast) ![]usize {
        // Cast maintains the same shape as input
        return try allocator.dupe(usize, op.input.shape);
    }

    pub fn run(op: *const Cast) !void {
        // Use the cast_lean operation from tensorMath based on input/output types
        switch (op.input.ty) {
            .f32 => switch (op.output.ty) {
                .u8 => try tensorMath.cast_lean(f32, u8, &op.input.tensor_f32, &op.output.tensor_u8, @enumFromInt(op.to)),
                .i8 => try tensorMath.cast_lean(f32, i8, &op.input.tensor_f32, &op.output.tensor_i8, @enumFromInt(op.to)),
                .i32 => try tensorMath.cast_lean(f32, i32, &op.input.tensor_f32, &op.output.tensor_i32, @enumFromInt(op.to)),
                .f32 => try tensorMath.cast_lean(f32, f32, &op.input.tensor_f32, &op.output.tensor_f32, @enumFromInt(op.to)),
                else => return error.UnsupportedCastType,
            },
            .u8 => switch (op.output.ty) {
                .f32 => try tensorMath.cast_lean(u8, f32, &op.input.tensor_u8, &op.output.tensor_f32, @enumFromInt(op.to)),
                .i32 => try tensorMath.cast_lean(u8, i32, &op.input.tensor_u8, &op.output.tensor_i32, @enumFromInt(op.to)),
                .u8 => try tensorMath.cast_lean(u8, u8, &op.input.tensor_u8, &op.output.tensor_u8, @enumFromInt(op.to)),
                else => return error.UnsupportedCastType,
            },
            .i8 => switch (op.output.ty) {
                .f32 => try tensorMath.cast_lean(i8, f32, &op.input.tensor_i8, &op.output.tensor_f32, @enumFromInt(op.to)),
                .i32 => try tensorMath.cast_lean(i8, i32, &op.input.tensor_i8, &op.output.tensor_i32, @enumFromInt(op.to)),
                .i8 => try tensorMath.cast_lean(i8, i8, &op.input.tensor_i8, &op.output.tensor_i8, @enumFromInt(op.to)),
                else => return error.UnsupportedCastType,
            },
            .i32 => switch (op.output.ty) {
                .f32 => try tensorMath.cast_lean(i32, f32, &op.input.tensor_i32, &op.output.tensor_f32, @enumFromInt(op.to)),
                .u8 => try tensorMath.cast_lean(i32, u8, &op.input.tensor_i32, &op.output.tensor_u8, @enumFromInt(op.to)),
                .i8 => try tensorMath.cast_lean(i32, i8, &op.input.tensor_i32, &op.output.tensor_i8, @enumFromInt(op.to)),
                .i32 => try tensorMath.cast_lean(i32, i32, &op.input.tensor_i32, &op.output.tensor_i32, @enumFromInt(op.to)),
                else => return error.UnsupportedCastType,
            },
            else => return error.UnsupportedInputType,
        }
    }

    pub fn get_output_tensors(op: *const Cast) ![]*TensorZant {
        var tensors = try allocator.alloc(*TensorZant, 1);
        tensors[0] = op.output;
        return tensors;
    }

    pub fn get_input_tensors(op: *const Cast) ![]*TensorZant {
        var tensors = try allocator.alloc(*TensorZant, 1);
        tensors[0] = op.input;
        return tensors;
    }

    pub fn write_op(op: *const Cast, writer: std.fs.File.Writer) !void {
        // Create tensor string for input
        var tensor_input_string: []u8 = undefined;
        defer allocator.free(tensor_input_string);
        if (op.input.tc == TensorCategory.INITIALIZER) {
            tensor_input_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(op.input.name),
                ")",
            });
        } else {
            tensor_input_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(op.input.name) });
        }

        // Get input and output types
        const input_type = op.input.ty.toString();
        const output_type = op.output.ty.toString();

        // Generate the function call based on input and output types
        try writer.print(
            \\    tensMath.cast_lean(
            \\        {s}, // Input type
            \\        {s}, // Output type
            \\        {s}, // input tensor
            \\        &tensor_{s}, // output tensor
            \\        @enumFromInt({d}), // to DataType
            \\    ) catch return -1;
        , .{
            input_type, // Input type
            output_type, // Output type
            tensor_input_string, // input tensor
            try utils.getSanitizedName(op.output.name), // output tensor
            op.to, // to attribute
        });
    }

    pub fn print(op: *const Cast) void {
        std.debug.print("\n CAST:\n {any}", .{op});
    }

    pub fn sobstitute_tensors(self: *Cast, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
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
};
