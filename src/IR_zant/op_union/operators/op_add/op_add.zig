const std = @import("std");
const allocator = std.heap.page_allocator;
const IR_zant = @import("IR_zant");

// --- onnx ---
const onnx = IR_zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

// --- zant IR---
const tensorZant_lib = IR_zant.tensorZant_lib;
const TensorZant = tensorZant_lib.TensorZant;
const TensorCategory = tensorZant_lib.TensorCategory;
const utils = IR_zant.utils;

// https://onnx.ai/onnx/operators/onnx__Add.html
// INPUTS:
//      - A (heterogeneous) - T: First operand.
//      - B (heterogeneous) - T: Second operand.
// OUTPUTS:
//      - C (heterogeneous) - T: Result, has same element type as two inputs.
pub const Add = struct {
    input_A: *TensorZant,
    input_B: *TensorZant,
    output_C: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Add {
        const input_A = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_A_notFound;
        const input_B = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_B_notFound;
        const output_C = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_C_notFound;

        //set the output type:
        if (output_C.ty == tensorZant_lib.TensorType.undefined) output_C.ty = input_A.ty;

        return Add{
            .input_A = input_A,
            .input_B = input_B,
            .output_C = output_C,
        };
    }

    pub fn get_output_shape(self: Add) []usize {
        return self.output_C.getShape();
    }

    pub fn get_input_tensors(self: Add) ![]*TensorZant {
        var input_tensors: std.ArrayList(*TensorZant) = .empty;
        defer input_tensors.deinit(allocator);

        try input_tensors.append(allocator, self.input_A);
        try input_tensors.append(allocator, self.input_B);

        return input_tensors.toOwnedSlice(allocator);
    }

    pub fn get_output_tensors(self: Add) ![]*TensorZant {
        var output_tensors: std.ArrayList(*TensorZant) = .empty;
        defer output_tensors.deinit(allocator);

        try output_tensors.append(allocator, self.output_C);

        return output_tensors.toOwnedSlice(allocator);
    }

    pub fn write_op(self: Add, writer: *std.Io.Writer) !void {

        //----create tensor_A_string
        var tensor_A_string: []u8 = undefined;
        defer allocator.free(tensor_A_string);

        if (self.input_A.tc == TensorCategory.INITIALIZER) {
            tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_A.name),
                ")",
            });
        } else {
            tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.input_A.name) });
        }

        //----create tensor_B_string
        var tensor_B_string: []u8 = undefined;
        defer allocator.free(tensor_B_string);
        if (self.input_B.tc == TensorCategory.INITIALIZER) {
            tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_B.name),
                ")",
            });
        } else {
            tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.input_B.name) });
        }

        _ = try writer.print(
            \\
            \\
            \\    tensMath.add_lean({s}, {s}, {s}, {s}, &tensor_{s}) catch return {d};
        , .{
            self.input_A.ty.toString(),
            self.output_C.ty.toString(),
            tensor_A_string, // Input tensor A
            tensor_B_string, // Input tensor B
            try utils.getSanitizedName(self.output_C.name), // Output tensor C
            utils.getMathErrorReturn(), // Error code for math errors
        });
    }

    pub fn compute_output_shape(self: Add) []usize {
        var output_shape: []usize = undefined;
        output_shape = try utils.broadcastShapes(allocator, self.input_A.shape, self.input_B.shape);
        self.output_C.shape = output_shape;
        return output_shape;
    }

    pub fn sobstitute_tensors(self: *Add, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        std.debug.print("\n                Add.sobstitute_tensors({s} with {s})", .{ old_tensor.name, new_tensor.name });

        if (self.input_A == old_tensor) {
            self.input_A = new_tensor;
            return;
        }

        if (self.input_B == old_tensor) {
            self.input_B = new_tensor;
            return;
        }

        if (self.output_C == old_tensor) {
            self.output_C = new_tensor;
            return;
        }

        return error.TensorNotFound;
    }

    pub fn print(self: Add) void {
        std.debug.print("\n ADD:\n {any}", .{self});
    }

};
