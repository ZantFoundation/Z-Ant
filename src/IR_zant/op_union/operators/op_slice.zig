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

// https://onnx.ai/onnx/operators/onnx__Slice.html
// INPUTS:
//      - input (heterogeneous) - T: Tensor of data to extract slices from.
//      - starts (heterogeneous) - T1: 1-D tensor of starting indices of corresponding axis in `axes`.
//      - ends (heterogeneous) - T1: 1-D tensor of ending indices (exclusive) of corresponding axis in `axes`.
//      - axes (heterogeneous) - T1: 1-D tensor of axes that `starts` and `ends` apply to.
//      - steps (heterogeneous) - T1: 1-D tensor of slice step of corresponding axis in `axes`.
// OUTPUTS:
//      - output (heterogeneous) - T: Sliced data tensor.

pub const Slice = struct {
    input: *TensorZant,
    starts: *TensorZant,
    ends: *TensorZant,
    axes: ?*TensorZant,
    steps: ?*TensorZant,
    output: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Slice {
        const input = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const starts = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_X_notFound;
        const ends = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[2])) |ptr| ptr else return error.input_X_notFound;
        const output = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;
        // Optional inputs
        const axes: ?*TensorZant = if (nodeProto.input.len >= 4) if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[3])) |ptr| ptr else return error.axes_notFound else null;
        const steps: ?*TensorZant = if (nodeProto.input.len >= 5) if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[4])) |ptr| ptr else return error.steps_notFound else null;

        //set the output type:
        if (output.ty == tensorZant_lib.TensorType.undefined) output.ty = input.ty;

        return Slice{
            .input = input,
            .starts = starts,
            .ends = ends,
            .axes = axes,
            .steps = steps,
            .output = output,
        };
    }

    pub fn get_output_shape(self: Slice) []usize {
        return self.compute_output_shape() catch {
            // Fallback to a default shape in case of error
            std.log.warn("[SLICE DEBUG] Failed to compute output shape, using fallback", .{});
            const fallback_shape = allocator.alloc(usize, 1) catch unreachable;
            fallback_shape[0] = 1;
            return fallback_shape;
        };
    }

    pub fn get_input_tensors(self: Slice) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();
        try inputs.append(self.input);
        try inputs.append(self.starts);
        try inputs.append(self.ends);
        if (self.axes) |a| try inputs.append(a);
        if (self.steps) |s| try inputs.append(s);
        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: Slice) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();
        try outputs.append(self.output);
        return outputs.toOwnedSlice();
    }

    pub fn write_op(self: Slice, writer: std.fs.File.Writer) !void {

        //----create tensor_input_string
        var tensor_input_string: []u8 = undefined;
        defer allocator.free(tensor_input_string);

        if (self.input.tc == TensorCategory.INITIALIZER) {
            tensor_input_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input.name),
                ")",
            });
        } else {
            tensor_input_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try self.input.getNameSanitized(), ")" });
        }

        //----create tensor_starts_string
        var tensor_starts_string: []u8 = undefined;
        defer allocator.free(tensor_starts_string);

        if (self.starts.tc == TensorCategory.INITIALIZER) {
            tensor_starts_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try self.starts.getNameSanitized(),
                ")",
            });
        } else {
            tensor_starts_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try self.starts.getNameSanitized(), ")" });
        }

        //----create tensor_ends_string
        var tensor_ends_string: []u8 = undefined;
        defer allocator.free(tensor_ends_string);

        if (self.ends.tc == TensorCategory.INITIALIZER) {
            tensor_ends_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try self.ends.getNameSanitized(),
                ")",
            });
        } else {
            tensor_ends_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try self.ends.getNameSanitized(), ")" });
        }

        //----create ?axes string
        var tensor_axes_string: []u8 = undefined;
        var axes_allocated = false;
        // Bias Tensor B is optional! verify the presence
        if (self.axes) |axes| {
            tensor_axes_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try axes.getNameSanitized(), ")" });
            axes_allocated = true;
        } else {
            tensor_axes_string = try std.mem.concat(allocator, u8, &[_][]const u8{"null"});
            axes_allocated = true;
        }
        defer if (axes_allocated) allocator.free(tensor_axes_string);

        //----create ?steps string
        var tensor_steps_string: []u8 = undefined;
        var steps_allocated = false;
        // Steps Tensor is optional! verify the presence
        if (self.steps) |steps| {
            tensor_steps_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try steps.getNameSanitized(), ")" });
            steps_allocated = true;
        } else {
            tensor_steps_string = try std.mem.concat(allocator, u8, &[_][]const u8{"null"});
            steps_allocated = true;
        }
        defer if (steps_allocated) allocator.free(tensor_steps_string);

        _ = try writer.print(
            \\    
            \\    @setEvalBranchQuota(10000);
            \\    tensMath.slice_onnx_lean(
            \\        {s}, //type
            \\        {s}, //type 1
            \\        {s}, //input
            \\        {s}, //starts
            \\        {s}, //ends
            \\        {s}, //axes
            \\        {s}, //steps
            \\        &tensor_{s}, //output
            \\    ) catch return -1;
        , .{
            self.input.ty.toString(), //type
            self.starts.ty.toString(), //type 1
            tensor_input_string, //input
            tensor_starts_string, //starts
            tensor_ends_string, //ends
            tensor_axes_string, //axes
            tensor_steps_string, //steps
            try self.output.getNameSanitized(),
        });
    }

    pub fn compute_output_shape(self: Slice) ![]usize {
        // Convert tensor data to slices, handling null cases
        const starts_data = if (self.starts.ptr) |ptr| ptr.i64.data else return error.InvalidSliceData;
        const ends_data = if (self.ends.ptr) |ptr| ptr.i64.data else return error.InvalidSliceData;
        const axes_data = if (self.axes) |axes| if (axes.ptr) |ptr| ptr.i64.data else null else null;
        const steps_data = if (self.steps) |steps| if (steps.ptr) |ptr| ptr.i64.data else null else null;

        const output_shape = try tensorMath.get_slice_output_shape(
            self.input.shape,
            starts_data,
            ends_data,
            axes_data,
            steps_data,
        );
        self.output.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Slice) void {
        std.debug.print("\n Slice: {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *Slice, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input == old_tensor) {
            self.input = new_tensor;
            return;
        }
        if (self.starts == old_tensor) {
            self.starts = new_tensor;
            return;
        }
        if (self.ends == old_tensor) {
            self.ends = new_tensor;
            return;
        }
        if (self.axes != null and self.axes.? == old_tensor) {
            self.axes = new_tensor;
            return;
        }
        if (self.steps != null and self.steps.? == old_tensor) {
            self.steps = new_tensor;
            return;
        }
        if (self.output == old_tensor) {
            self.output = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
};
