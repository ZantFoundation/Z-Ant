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
const mathHandler_log = std.log.scoped(.mathHandler);

const utils = IR_zant.utils;

// https://onnx.ai/onnx/operators/onnx__MatMul.html#l-onnx-doc-matmul
// INPUTS:
//      - A (heterogeneous) - T:  input tensor.
// OUTPUTS:
//      - C (heterogeneous) - T:  output tensor.

pub const MatMul = struct {
    input_A: *TensorZant,
    input_B: *TensorZant,
    output_C: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !MatMul {
        const input_A = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_A_notFound;
        const input_B = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_B_notFound;
        const output_C = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_C_notFound;

        //set the output type:
        if (output_C.ty == tensorZant_lib.TensorType.undefined) output_C.ty = input_A.ty;

        return MatMul{
            .input_A = input_A,
            .input_B = input_B,
            .output_C = output_C,
        };
    }

    pub fn get_output_shape(self: MatMul) []usize {
        return self.output_C.getShape();
    }

    pub fn get_input_tensors(self: MatMul) ![]*TensorZant {
        var inputs: std.ArrayList(*TensorZant) = .empty;
        defer inputs.deinit(allocator);

        try inputs.append(allocator, self.input_A);
        try inputs.append(allocator, self.input_B);

        return inputs.toOwnedSlice(allocator);
    }

    pub fn get_output_tensors(self: MatMul) ![]*TensorZant {
        var outputs: std.ArrayList(*TensorZant) = .empty;
        defer outputs.deinit(allocator);

        try outputs.append(allocator, self.output_C);

        return outputs.toOwnedSlice(allocator);
    }

    pub fn write_op(self: MatMul, writer: *std.Io.Writer) !void {
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

        var element_size_bytes: usize = 4; // Default to f32 size as fallback

        // Determine size from DataType enum
        element_size_bytes = switch (self.input_B.ty) {
            .f32 => @sizeOf(f32),
            .f16 => @sizeOf(f16),
            .i64 => @sizeOf(i64),
            .i32 => @sizeOf(i32),
            .i8 => @sizeOf(i8),
            .u8 => @sizeOf(u8),
            // Add other supported types as needed
            else => blk: {
                mathHandler_log.warn("Warning: Unsupported DataType '{any}' for MatMul input B '{s}'. Assuming f32 size.\n", .{ self.input_B.ty, tensor_B_string });
                break :blk 4;
            },
        };

        const b_dims = self.input_B.getShape().len;
        if (b_dims == 0) {
            mathHandler_log.warn("Error: MatMul input B '{s}' has zero dimensions.\n", .{tensor_B_string});
            return error.InvalidShape; // Avoid panic on empty shape
        }

        const b_width_elements: usize = self.input_B.shape[b_dims - 1];
        const b_width_bytes: usize = b_width_elements * element_size_bytes;

        if (b_width_bytes >= std.atomic.cache_line) { //B is large enough for the new mat mul to work;
            _ = try writer.print(
                \\
                \\    tensMath.blocked_mat_mul_lean(
                \\          {s}, 
                \\          {s}, 
                \\          {s},
                \\          &tensor_{s},
                \\    ) catch return {d};
            , .{
                self.input_A.ty.toString(), // Input tensor type
                tensor_A_string, // Input tensor A
                tensor_B_string, // Input tensor B
                try utils.getSanitizedName(self.output_C.name), // Output tensor C
                utils.getMathErrorReturn(), // Error code for math errors
            });
        } else { //B is not large enough, so we keep the old but improved mat_mul
            _ = try writer.print(
                \\
                \\    tensMath.mat_mul_lean({s}, {s}, {s}, &tensor_{s}) catch return {d};
            , .{
                self.output_C.ty.toString(),
                tensor_A_string, // Input tensor A
                tensor_B_string, // Input tensor B
                try utils.getSanitizedName(self.output_C.name), // Output tensor C
                utils.getMathErrorReturn(), // Error code for math errors
            });
        }
    }

    pub fn compute_output_shape(self: MatMul) []usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_mat_mul_output_shape(self.input_A.shape, self.input_B.shape);
        self.output_C.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: MatMul) void {
        std.debug.print("\n MatMul:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *MatMul, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
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
};
