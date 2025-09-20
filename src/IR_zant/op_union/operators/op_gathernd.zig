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

// https://onnx.ai/onnx/operators/onnx__GatherND.html
// INPUTS:
//      - data (heterogeneous) - T: Tensor of rank r >= 1.
//      - indices (heterogeneous) - tensor(int64): Tensor of rank q >= 1.
// OUTPUTS:
//      - output (heterogeneous) - T: Tensor of rank q + r - indices_shape[-1] - 1.

pub const GatherND = struct {
    input_data: *TensorZant,
    input_indices: *TensorZant,
    output: *TensorZant,
    //attributes:
    batch_dims: i64 = 0, // default = 0

    pub fn init(nodeProto: *NodeProto) !GatherND {
        const input_data = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_data_notFound;
        const input_indices = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_indices_notFound;
        const output = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_notFound;

        var batch_dims: i64 = 0;
        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "batch_dims")) {
                if (attr.type == onnx.AttributeType.INT) batch_dims = attr.i;
            }
        }

        // Set the output type based on the data input
        if (output.ty == tensorZant_lib.TensorType.undefined) output.ty = input_data.ty;

        return GatherND{
            .input_data = input_data,
            .input_indices = input_indices,
            .output = output,
            .batch_dims = batch_dims,
        };
    }

    pub fn get_output_shape(self: GatherND) []usize {
        return self.compute_output_shape() catch {
            // Fallback to a default shape in case of error
            std.log.warn("[GATHERND DEBUG] Failed to compute output shape, using fallback", .{});
            const fallback_shape = allocator.alloc(usize, 1) catch unreachable;
            fallback_shape[0] = 1;
            return fallback_shape;
        };
    }

    pub fn compute_output_shape(self: GatherND) ![]usize {
        const output_shape = try tensorMath.get_gathernd_output_shape(
            self.input_data.shape,
            self.input_indices.shape,
        );
        self.output.shape = output_shape;
        return output_shape;
    }

    pub fn get_input_tensors(self: GatherND) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();
        try inputs.append(self.input_data);
        try inputs.append(self.input_indices);
        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: GatherND) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();
        try outputs.append(self.output);
        return outputs.toOwnedSlice();
    }

    pub fn print(self: GatherND) void {
        std.debug.print("\n GatherND: data={s}, indices={s}, output={s}, batch_dims={d}", .{ self.input_data.name, self.input_indices.name, self.output.name, self.batch_dims });
    }

    pub fn write_op(self: GatherND, writer: std.fs.File.Writer) !void {
        // Generate tensor strings for inputs
        var tensor_data_string: []u8 = undefined;
        defer allocator.free(tensor_data_string);
        var tensor_indices_string: []u8 = undefined;
        defer allocator.free(tensor_indices_string);

        if (self.input_data.tc == TensorCategory.INITIALIZER) {
            tensor_data_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_data.name),
                ")",
            });
        } else {
            tensor_data_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.input_data.name) });
        }

        if (self.input_indices.tc == TensorCategory.INITIALIZER) {
            tensor_indices_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_indices.name),
                ")",
            });
        } else {
            tensor_indices_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.input_indices.name) });
        }

        _ = try writer.print(
            \\    tensMath.gathernd_lean(
            \\        {s},
            \\        {s}, // Data tensor
            \\        {s}, // Indices tensor
            \\        &tensor_{s} // Output tensor
            \\    ) catch return -1;
        , .{
            self.input_data.ty.toString(),
            tensor_data_string,
            tensor_indices_string,
            try utils.getSanitizedName(self.output.name),
        });
    }

    pub fn sobstitute_tensors(self: *GatherND, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input_data == old_tensor) {
            self.input_data = new_tensor;
            return;
        }
        if (self.input_indices == old_tensor) {
            self.input_indices = new_tensor;
            return;
        }
        if (self.output == old_tensor) {
            self.output = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
};
