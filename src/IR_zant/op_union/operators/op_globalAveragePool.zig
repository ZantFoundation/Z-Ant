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
const IR_utils = IR_zant.utils; //this is IR utils

pub const GlobalAveragePool = struct {
    input_X: *TensorZant,
    output_Y: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !GlobalAveragePool {
        // Validate that we have exactly one input and one output
        if (nodeProto.input.len != 1) {
            return error.InvalidInputCount;
        }
        if (nodeProto.output.len != 1) {
            return error.InvalidOutputCount;
        }

        const input_X = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const output_Y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        // GlobalAveragePool has no attributes according to ONNX spec
        // But we can validate that no unexpected attributes are present
        for (nodeProto.attribute) |attr| {
            // According to ONNX spec, GlobalAveragePool should have no attributes
            // Log a warning if any attributes are found
            std.debug.print("Warning: GlobalAveragePool received unexpected attribute: {s}\n", .{attr.name});
        }

        // Set the output type to match input type
        if (output_Y.ty == tensorZant_lib.TensorType.undefined) {
            output_Y.ty = input_X.ty;
        }

        // Validate that input type is a floating point type (ONNX constraint)
        switch (input_X.ty) {
            .f16, .f32, .f64 => {}, // Valid floating point types
            else => return error.InvalidInputType, // GlobalAveragePool only supports float types
        }

        return GlobalAveragePool{
            .input_X = input_X,
            .output_Y = output_Y,
        };
    }

    pub fn get_output_shape(self: GlobalAveragePool) []usize {
        return self.output_Y.getShape();
    }

    pub fn get_input_tensors(self: GlobalAveragePool) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();
        try inputs.append(self.input_X);
        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: GlobalAveragePool) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();
        try outputs.append(self.output_Y);
        return outputs.toOwnedSlice();
    }

    pub fn compute_output_shape(self: GlobalAveragePool) []usize {
        // For GlobalAveragePool, output shape is (N, C, 1, 1, ...)
        // where N is batch size, C is channels, and all spatial dimensions become 1
        const input_shape = self.input_X.ptr.?.get_shape();

        // Validate input has at least 2 dimensions (N, C)
        if (input_shape.len < 2) {
            std.debug.panic("GlobalAveragePool input must have at least 2 dimensions (N, C), got {d}\n", .{input_shape.len});
        }

        // Create output shape: same rank as input, but spatial dimensions are 1
        var output_shape = try allocator.alloc(usize, input_shape.len);

        // First two dimensions (batch_size, channels) remain the same
        output_shape[0] = input_shape[0]; // N (batch size)
        output_shape[1] = input_shape[1]; // C (channels)

        // All spatial dimensions become 1
        for (2..input_shape.len) |i| {
            output_shape[i] = 1;
        }

        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn write_op(self: GlobalAveragePool, writer: std.fs.File.Writer) !void {
        var input_tensor_string: []u8 = undefined;
        defer allocator.free(input_tensor_string);

        if (self.input_X.tc == TensorCategory.INITIALIZER) {
            input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_X.name),
                ")",
            });
        } else {
            input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.input_X.name) });
        }

        _ = try writer.print(
            \\
            \\ tensMath.globalAveragePool_lean({s}, {s}, &tensor_{s}) catch return -1;
        , .{
            self.input_X.ty.toString(),
            input_tensor_string,
            try utils.getSanitizedName(self.output_Y.name),
        });
    }

    pub fn print(self: GlobalAveragePool) void {
        std.debug.print("\n GlobalAveragePool:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *GlobalAveragePool, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
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
