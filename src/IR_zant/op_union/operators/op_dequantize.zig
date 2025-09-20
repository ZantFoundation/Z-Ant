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
const AnyTensor = zant.core.tensor.AnyTensor;

const tensorMath = zant.core.tensor.math_standard;

const utils = IR_zant.utils;

const Tensor = zant.core.tensor.Tensor;

const TensorType = tensorZant_lib.TensorType;

/// IMPORTANT:
///
/// This is a Custom Operator!!
/// Dequantize is not part of the ONNX standard
///
///
pub const Dequantize = struct {
    input: *TensorZant,
    output: *TensorZant,
    //scheme: quantScheme,

    pub fn init(comptime input_type: type, inputTensor: *Tensor(input_type), comptime output_type: type) !Dequantize {
        const shape: []usize = inputTensor.shape;

        const input_ptr = try allocator.create(TensorZant);
        input_ptr.* = TensorZant{
            .name = "input",
            .ty = TensorType.fromType(input_type),
            .tc = TensorCategory.INPUT,
            .ptr = AnyTensor{ .input_type = inputTensor },
            .shape = shape,
        };

        const outputTensor = Tensor(input_type).fromShape(&allocator, shape);
        const output_ptr = try allocator.create(TensorZant);
        output_ptr.* = TensorZant{
            .name = "output",
            .ty = TensorType.fromType(output_type),
            .tc = TensorCategory.OUTPUT,
            .ptr = AnyTensor{ .output_type = outputTensor },
            .shape = shape,
        };
        return Dequantize{
            .input = input_ptr,
            .output = output_ptr,
        };
    }

    pub fn get_output_shape(self: Dequantize) []usize {
        return self.output.getShape();
    }

    pub fn get_input_tensors(self: Dequantize) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();

        try inputs.append(self.input);

        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: Dequantize) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();

        try outputs.append(self.output);
        return outputs.toOwnedSlice();
    }

    pub fn compute_output_shape(self: Dequantize) []usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_quantize_output_shape(self.input.getShape());
        return output_shape;
    }

    pub fn write_op(self: Dequantize, writer: std.fs.File.Writer) !void {
        const input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&tensor_",
            try utils.getSanitizedName(self.input.name),
            ")",
        });

        const output_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&tensor_",
            try utils.getSanitizedName(self.output.name),
            ")",
        });

        _ = try writer.print(
            \\
            \\
            \\    tensMath.lean_dequantize(
            \\        {s}, //inputType 
            \\        {s}, //outputType 
            \\        {s}, //input 
            \\        {s}, //output
            \\    ) catch return -1;
        , .{
            self.input.ty.toString(),
            self.output.ty.toString(),
            input_tensor_string,
            output_tensor_string,
        });
    }

    pub fn print(self: Dequantize) void { // TODO
        std.debug.print("\n Dequantize:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *Dequantize, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
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
