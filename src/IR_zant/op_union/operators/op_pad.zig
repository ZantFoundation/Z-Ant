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

// https://onnx.ai/onnx/operators/onnx__Pad.html
// INPUTS:
//      - data (heterogeneous) - T: Input tensor
//      - pads (heterogeneous) - tensor(int64): Tensor of integers indicating the number of padding elements to add or remove
//      - constant_value (optional, heterogeneous) - T: A scalar value to be used if the mode is "constant"
//      - axes (optional, heterogeneous) - Tind: Tensor of integers indicating the axis to pad
// OUTPUTS:
//      - output (heterogeneous) - T: Tensor after padding
// ATTRIBUTES:
//      - mode - STRING (default is "constant"): Padding mode

pub const Pad = struct {
    input_data: *TensorZant,
    input_pads: *TensorZant,
    input_constant_value: ?*TensorZant,
    input_axes: ?*TensorZant,
    output: *TensorZant,

    // Attributes
    mode: []const u8,

    pub fn init(nodeProto: *NodeProto) !Pad {
        // Pad has 2-4 inputs
        if (nodeProto.input.len < 2 or nodeProto.input.len > 4) {
            return error.PadInvalidInputCount;
        }

        const input_data = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_data_notFound;
        const input_pads = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_pads_notFound;
        const input_constant_value = if (nodeProto.input.len > 2) if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[2])) |ptr| ptr else null else null;
        const input_axes = if (nodeProto.input.len > 3) if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[3])) |ptr| ptr else null else null;
        const output = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_notFound;

        // Default attributes
        var mode: []const u8 = "constant";

        // Parse attributes
        for (nodeProto.attribute) |attr| {
            if (std.mem.indexOf(u8, attr.name, "mode")) |_| {
                if (attr.type == onnx.AttributeType.STRING) mode = attr.s else return error.ModeNotSTRING;
            }
        }

        // Set the output type - Pad preserves input data type
        if (output.ty == tensorZant_lib.TensorType.undefined) {
            if (input_data.ty != tensorZant_lib.TensorType.undefined) {
                output.ty = input_data.ty;
            } else {
                // Fallback to u8 for quantized operations when input type is undefined
                output.ty = tensorZant_lib.TensorType.u8;
            }
        }

        return Pad{
            .input_data = input_data,
            .input_pads = input_pads,
            .input_constant_value = input_constant_value,
            .input_axes = input_axes,
            .output = output,
            .mode = mode,
        };
    }

    pub fn get_output_shape(op: *const Pad) []usize {
        return op.compute_output_shape() catch {
            // Fallback to a default shape in case of error
            std.log.warn("[PAD DEBUG] Failed to compute output shape, using fallback", .{});
            const fallback_shape = allocator.alloc(usize, 1) catch unreachable;
            fallback_shape[0] = 1;
            return fallback_shape;
        };
    }

    pub fn compute_output_shape(op: *const Pad) ![]usize {
        // Compute output shape based on pads and optional axes
        const input_shape = op.input_data.shape;

        // Extract pads and axes slices from AnyTensor
        // If data is not available during shape inference, use default zero padding
        const pads_slice: []const i64 = if (op.input_pads.ptr) |at|
            at.get_data_as(i64)
        else blk: {
            // Default to zero padding (no change in shape)
            const default_pads = try allocator.alloc(i64, input_shape.len * 2);
            @memset(default_pads, 0);
            break :blk default_pads;
        };

        const axes_slice_opt: ?[]const i64 = if (op.input_axes) |ax|
            if (ax.ptr) |at| at.get_data_as(i64) else null
        else
            null;

        const out_shape = try tensorMath.get_pad_output_shape(
            input_shape,
            pads_slice,
            axes_slice_opt,
        );

        // Free the allocated default pads if we created them
        if (op.input_pads.ptr == null) {
            allocator.free(pads_slice);
        }

        // Persist onto output tensor for downstream allocation
        op.output.shape = out_shape;
        return out_shape;
    }

    pub fn run(op: *const Pad) !void {
        // Use the pad operation from tensorMath based on input type
        switch (op.input_data.ty) {
            .f32 => try tensorMath.pad(
                f32,
                &op.input_data.tensor_f32,
                &op.input_pads.tensor_i64,
                if (op.input_constant_value) |cv| &cv.tensor_f32 else null,
                if (op.input_axes) |axes| &axes.tensor_i64 else null,
                &op.output.tensor_f32,
                op.mode,
            ),
            .u8 => try tensorMath.pad(
                u8,
                &op.input_data.tensor_u8,
                &op.input_pads.tensor_i64,
                if (op.input_constant_value) |cv| &cv.tensor_u8 else null,
                if (op.input_axes) |axes| &axes.tensor_i64 else null,
                &op.output.tensor_u8,
                op.mode,
            ),
            .i8 => try tensorMath.pad(
                i8,
                &op.input_data.tensor_i8,
                &op.input_pads.tensor_i64,
                if (op.input_constant_value) |cv| &cv.tensor_i8 else null,
                if (op.input_axes) |axes| &axes.tensor_i64 else null,
                &op.output.tensor_i8,
                op.mode,
            ),
            .i32 => try tensorMath.pad(
                i32,
                &op.input_data.tensor_i32,
                &op.input_pads.tensor_i64,
                if (op.input_constant_value) |cv| &cv.tensor_i32 else null,
                if (op.input_axes) |axes| &axes.tensor_i64 else null,
                &op.output.tensor_i32,
                op.mode,
            ),
            else => return error.UnsupportedInputType,
        }
    }

    pub fn get_output_tensors(op: *const Pad) ![]*TensorZant {
        var tensors = try allocator.alloc(*TensorZant, 1);
        tensors[0] = op.output;
        return tensors;
    }

    pub fn get_input_tensors(op: *const Pad) ![]*TensorZant {
        var count: usize = 2;
        if (op.input_constant_value != null) count += 1;
        if (op.input_axes != null) count += 1;

        var tensors = try allocator.alloc(*TensorZant, count);
        tensors[0] = op.input_data;
        tensors[1] = op.input_pads;
        var idx: usize = 2;
        if (op.input_constant_value) |cv| {
            tensors[idx] = cv;
            idx += 1;
        }
        if (op.input_axes) |axes| {
            tensors[idx] = axes;
        }
        return tensors;
    }

    pub fn write_op(op: *const Pad, writer: std.fs.File.Writer) !void {
        // Build input data reference
        var data_ref: []u8 = undefined;
        defer allocator.free(data_ref);
        if (op.input_data.tc == TensorCategory.INITIALIZER or op.input_data.tc == TensorCategory.CONSTANT) {
            data_ref = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&",
                if (op.input_data.tc == TensorCategory.CONSTANT) "" else "param_lib.",
                "tensor_",
                try utils.getSanitizedName(op.input_data.name),
                ")",
            });
        } else {
            data_ref = try std.mem.concat(allocator, u8, &[_][]const u8{
                "&tensor_",
                try utils.getSanitizedName(op.input_data.name),
            });
        }

        // Build pads reference (always i64 tensor)
        var pads_ref: []u8 = undefined;
        defer allocator.free(pads_ref);
        if (op.input_pads.tc == TensorCategory.INITIALIZER or op.input_pads.tc == TensorCategory.CONSTANT) {
            pads_ref = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&",
                if (op.input_pads.tc == TensorCategory.CONSTANT) "" else "param_lib.",
                "tensor_",
                try utils.getSanitizedName(op.input_pads.name),
                ")",
            });
        } else {
            pads_ref = try std.mem.concat(allocator, u8, &[_][]const u8{
                "&tensor_",
                try utils.getSanitizedName(op.input_pads.name),
            });
        }

        // Optional constant value
        var const_ref: []const u8 = "null";
        if (op.input_constant_value) |cv| {
            const cv_ref = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&",
                if (cv.tc == TensorCategory.CONSTANT) "" else "param_lib.",
                "tensor_",
                try utils.getSanitizedName(cv.name),
                ")",
            });
            const_ref = cv_ref;
            // leak on purpose into generated code scope; freed with allocator lifetime at end of codegen
        }

        // Optional axes
        var axes_ref: []const u8 = "null";
        if (op.input_axes) |ax| {
            const ax_ref = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&",
                if (ax.tc == TensorCategory.CONSTANT) "" else "param_lib.",
                "tensor_",
                try utils.getSanitizedName(ax.name),
                ")",
            });
            axes_ref = ax_ref;
        }

        // Output tensor name
        const out_name = try utils.getSanitizedName(op.output.name);

        // No shape fix needed - the mathematical function will handle shape validation

        // Emit pad call (preserve dtype)
        _ = try writer.print(
            \\    tensMath.pad({s},
            \\        {s},
            \\        {s},
            \\        {s},
            \\        {s},
            \\        &tensor_{s},
            \\        "{s}"
            \\    ) catch return -1;
        , .{
            op.output.ty.toString(),
            data_ref,
            pads_ref,
            const_ref,
            axes_ref,
            out_name,
            op.mode,
        });
    }

    pub fn print(op: *const Pad) void {
        std.debug.print("\n PAD:\n {any}", .{op});
    }

    pub fn sobstitute_tensors(self: *Pad, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input_data == old_tensor) {
            self.input_data = new_tensor;
            return;
        }
        if (self.input_pads == old_tensor) {
            self.input_pads = new_tensor;
            return;
        }
        if (self.input_constant_value != null and self.input_constant_value.? == old_tensor) {
            self.input_constant_value = new_tensor;
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
