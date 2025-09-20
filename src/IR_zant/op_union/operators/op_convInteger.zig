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
const quantTensorMath = zant.core.tensor.math_quant;
const utils = IR_zant.utils;

// https://onnx.ai/onnx/operators/onnx__ConvInteger.html
// INPUTS:
//      - x (heterogeneous) - T1: Input tensor (quantized)
//      - w (heterogeneous) - T2: Weight tensor (quantized)
//      - x_zero_point (optional, heterogeneous) - T1: Zero point for input x
//      - w_zero_point (optional, heterogeneous) - T2: Zero point for weight w
// OUTPUTS:
//      - y (heterogeneous) - T3: Output tensor (i32)
// ATTRIBUTES:
//      - auto_pad - STRING (default is 'NOTSET')
//      - dilations - INTS : dilation value along each spatial axis of the filter
//      - group - INT (default is '1'): number of groups input channels and output channels are divided into
//      - kernel_shape - INTS : The shape of the convolution kernel
//      - pads - INTS : Padding for the beginning and ending along each spatial axis
//      - strides - INTS : Stride along each spatial axis

pub const ConvInteger = struct {
    input_x: *TensorZant,
    input_w: *TensorZant,
    input_x_zero_point: ?*TensorZant,
    input_w_zero_point: ?*TensorZant,
    output_y: *TensorZant,

    // Attributes
    auto_pad: []const u8,
    dilations: ?[]i64,
    group: i64,
    kernel_shape: ?[]i64,
    pads: ?[]i64,
    strides: ?[]i64,

    pub fn init(nodeProto: *NodeProto) !ConvInteger {
        // ConvInteger has 2-4 inputs
        if (nodeProto.input.len < 2 or nodeProto.input.len > 4) {
            return error.ConvIntegerInvalidInputCount;
        }

        const input_x = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_x_notFound;
        const input_w = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_w_notFound;
        const input_x_zero_point = if (nodeProto.input.len > 2) if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[2])) |ptr| ptr else null else null;
        const input_w_zero_point = if (nodeProto.input.len > 3) if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[3])) |ptr| ptr else null else null;
        const output_y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_y_notFound;

        // Default attributes
        var auto_pad: []const u8 = "NOTSET";
        var dilations: ?[]i64 = null;
        var group: i64 = 1;
        var kernel_shape: ?[]i64 = null;
        var pads: ?[]i64 = null;
        var strides: ?[]i64 = null;

        // Parse attributes
        for (nodeProto.attribute) |attr| {
            if (std.mem.indexOf(u8, attr.name, "auto_pad")) |_| {
                if (attr.type == onnx.AttributeType.STRING) auto_pad = attr.s else return error.AutoPadNotSTRING;
            } else if (std.mem.indexOf(u8, attr.name, "dilations")) |_| {
                if (attr.type == onnx.AttributeType.INTS) dilations = attr.ints else return error.DilationsNotINTS;
            } else if (std.mem.indexOf(u8, attr.name, "group")) |_| {
                if (attr.type == onnx.AttributeType.INT) group = attr.i else return error.GroupNotINT;
            } else if (std.mem.indexOf(u8, attr.name, "kernel_shape")) |_| {
                if (attr.type == onnx.AttributeType.INTS) kernel_shape = attr.ints else return error.KernelShapeNotINTS;
            } else if (std.mem.indexOf(u8, attr.name, "pads")) |_| {
                if (attr.type == onnx.AttributeType.INTS) pads = attr.ints else return error.PadsNotINTS;
            } else if (std.mem.indexOf(u8, attr.name, "strides")) |_| {
                if (attr.type == onnx.AttributeType.INTS) strides = attr.ints else return error.StridesNotINTS;
            }
        }

        // Set the output type - ConvInteger always outputs i32
        if (output_y.ty == tensorZant_lib.TensorType.undefined) {
            output_y.ty = tensorZant_lib.TensorType.i32;
        }

        return ConvInteger{
            .input_x = input_x,
            .input_w = input_w,
            .input_x_zero_point = input_x_zero_point,
            .input_w_zero_point = input_w_zero_point,
            .output_y = output_y,
            .auto_pad = auto_pad,
            .dilations = dilations,
            .group = group,
            .kernel_shape = kernel_shape,
            .pads = pads,
            .strides = strides,
        };
    }

    pub fn get_output_shape(op: *const ConvInteger) ![]usize {
        // ConvInteger output shape is the same as regular convolution
        // Use the existing convolution output shape calculation from tensorMath

        // Convert i64 arrays to usize arrays
        var stride_usize: ?[]usize = null;
        var pads_usize: ?[]usize = null;
        var dilations_usize: ?[]usize = null;

        if (op.strides) |s| {
            stride_usize = try allocator.alloc(usize, s.len);
            for (s, 0..) |val, i| {
                stride_usize.?[i] = @intCast(val);
            }
        }

        if (op.pads) |p| {
            pads_usize = try allocator.alloc(usize, p.len);
            for (p, 0..) |val, i| {
                pads_usize.?[i] = @intCast(val);
            }
        }

        if (op.dilations) |d| {
            dilations_usize = try allocator.alloc(usize, d.len);
            for (d, 0..) |val, i| {
                dilations_usize.?[i] = @intCast(val);
            }
        }

        defer {
            if (stride_usize) |s| allocator.free(s);
            if (pads_usize) |p| allocator.free(p);
            if (dilations_usize) |d| allocator.free(d);
        }

        // Calculate output shape using existing convolution calculation
        return tensorMath.get_convolution_output_shape(
            i32, // Output type is always i32 for ConvInteger
            allocator,
            op.input_x.shape,
            op.input_w.shape,
            stride_usize,
            pads_usize,
            dilations_usize,
            op.auto_pad,
        );
    }

    pub fn run(op: *const ConvInteger) !void {
        // Convert i64 arrays to usize arrays for tensor operations
        var strides_usize: []usize = undefined;
        var pads_usize: []usize = undefined;
        var dilations_usize: []usize = undefined;

        if (op.strides) |s| {
            strides_usize = try allocator.alloc(usize, s.len);
            for (s, 0..) |val, i| {
                strides_usize[i] = @intCast(val);
            }
        }

        if (op.pads) |p| {
            pads_usize = try allocator.alloc(usize, p.len);
            for (p, 0..) |val, i| {
                pads_usize[i] = @intCast(val);
            }
        }

        if (op.dilations) |d| {
            dilations_usize = try allocator.alloc(usize, d.len);
            for (d, 0..) |val, i| {
                dilations_usize[i] = @intCast(val);
            }
        }

        // Determine input types and call appropriate ConvInteger function
        switch (op.input_x.ty) {
            .u8 => switch (op.input_w.ty) {
                .u8 => try quantTensorMath.convInteger_lean(
                    u8,
                    u8,
                    &op.input_x.tensor_u8,
                    &op.input_w.tensor_u8,
                    if (op.input_x_zero_point) |zp| &zp.tensor_u8 else null,
                    if (op.input_w_zero_point) |zp| &zp.tensor_u8 else null,
                    &op.output_y.tensor_i32,
                    if (op.strides) |_| strides_usize else null,
                    if (op.pads) |_| pads_usize else null,
                    if (op.dilations) |_| dilations_usize else null,
                    @intCast(op.group),
                    op.auto_pad,
                ),
                .i8 => try quantTensorMath.convInteger_lean(
                    u8,
                    i8,
                    &op.input_x.tensor_u8,
                    &op.input_w.tensor_i8,
                    if (op.input_x_zero_point) |zp| &zp.tensor_u8 else null,
                    if (op.input_w_zero_point) |zp| &zp.tensor_i8 else null,
                    &op.output_y.tensor_i32,
                    if (op.strides) |_| strides_usize else null,
                    if (op.pads) |_| pads_usize else null,
                    if (op.dilations) |_| dilations_usize else null,
                    @intCast(op.group),
                    op.auto_pad,
                ),
                else => return error.UnsupportedWeightType,
            },
            .i8 => switch (op.input_w.ty) {
                .u8 => try quantTensorMath.convInteger_lean(
                    i8,
                    u8,
                    &op.input_x.tensor_i8,
                    &op.input_w.tensor_u8,
                    if (op.input_x_zero_point) |zp| &zp.tensor_i8 else null,
                    if (op.input_w_zero_point) |zp| &zp.tensor_u8 else null,
                    &op.output_y.tensor_i32,
                    if (op.strides) |_| strides_usize else null,
                    if (op.pads) |_| pads_usize else null,
                    if (op.dilations) |_| dilations_usize else null,
                    @intCast(op.group),
                    op.auto_pad,
                ),
                .i8 => try quantTensorMath.convInteger_lean(
                    i8,
                    i8,
                    &op.input_x.tensor_i8,
                    &op.input_w.tensor_i8,
                    if (op.input_x_zero_point) |zp| &zp.tensor_i8 else null,
                    if (op.input_w_zero_point) |zp| &zp.tensor_i8 else null,
                    &op.output_y.tensor_i32,
                    if (op.strides) |_| strides_usize else null,
                    if (op.pads) |_| pads_usize else null,
                    if (op.dilations) |_| dilations_usize else null,
                    @intCast(op.group),
                    op.auto_pad,
                ),
                else => return error.UnsupportedWeightType,
            },
            else => return error.UnsupportedInputType,
        }

        // Cleanup allocated memory
        if (op.strides != null) allocator.free(strides_usize);
        if (op.pads != null) allocator.free(pads_usize);
        if (op.dilations != null) allocator.free(dilations_usize);
    }

    pub fn get_output_tensors(op: *const ConvInteger) ![]*TensorZant {
        var tensors = try allocator.alloc(*TensorZant, 1);
        tensors[0] = op.output_y;
        return tensors;
    }

    pub fn get_input_tensors(op: *const ConvInteger) ![]*TensorZant {
        var count: usize = 2;
        if (op.input_x_zero_point != null) count += 1;
        if (op.input_w_zero_point != null) count += 1;

        var tensors = try allocator.alloc(*TensorZant, count);
        tensors[0] = op.input_x;
        tensors[1] = op.input_w;
        var idx: usize = 2;
        if (op.input_x_zero_point) |zp| {
            tensors[idx] = zp;
            idx += 1;
        }
        if (op.input_w_zero_point) |zp| {
            tensors[idx] = zp;
        }
        return tensors;
    }

    pub fn write_op(op: *const ConvInteger, writer: std.fs.File.Writer) !void {
        // Create tensor string for input x
        var tensor_x_string: []u8 = undefined;
        defer allocator.free(tensor_x_string);
        if (op.input_x.tc == TensorCategory.INITIALIZER) {
            tensor_x_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(op.input_x.name),
                ")",
            });
        } else {
            tensor_x_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(op.input_x.name) });
        }

        // Create tensor string for input w
        var tensor_w_string: []u8 = undefined;
        defer allocator.free(tensor_w_string);
        if (op.input_w.tc == TensorCategory.INITIALIZER) {
            tensor_w_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(op.input_w.name),
                ")",
            });
        } else {
            tensor_w_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(op.input_w.name) });
        }

        // Create tensor string for x_zero_point (optional)
        var x_zero_point_string: []u8 = undefined;
        defer allocator.free(x_zero_point_string);
        if (op.input_x_zero_point) |zp| {
            if (zp.tc == TensorCategory.INITIALIZER) {
                x_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                    "@constCast(&param_lib.tensor_",
                    try utils.getSanitizedName(zp.name),
                    ")",
                });
            } else {
                x_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(zp.name) });
            }
        } else {
            x_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{"null"});
        }

        // Create tensor string for w_zero_point (optional)
        var w_zero_point_string: []u8 = undefined;
        defer allocator.free(w_zero_point_string);
        if (op.input_w_zero_point) |zp| {
            if (zp.tc == TensorCategory.INITIALIZER) {
                w_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                    "@constCast(&param_lib.tensor_",
                    try utils.getSanitizedName(zp.name),
                    ")",
                });
            } else {
                w_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(zp.name) });
            }
        } else {
            w_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{"null"});
        }

        // Create stride string (strides is optional but defaults to [1,1,...])
        var stride_string: []const u8 = "null";
        if (op.strides) |s| {
            if (s.len > 0) {
                stride_string = try utils.i64SliceToUsizeArrayString(s);
            } else {
                stride_string = "&[_]usize{}";
            }
        }

        // Create pads string (optional)
        var pads_string: []const u8 = "null";
        if (op.pads) |p| {
            if (p.len > 0) {
                pads_string = try utils.i64SliceToUsizeArrayString(p);
            } else {
                pads_string = "&[_]usize{}";
            }
        }

        // Create dilations string (optional)
        var dilations_string: []const u8 = "null";
        if (op.dilations) |d| {
            if (d.len > 0) {
                dilations_string = try utils.i64SliceToUsizeArrayString(d);
            } else {
                dilations_string = "&[_]usize{}";
            }
        }

        // Get input and weight types for the function call
        const input_type = op.input_x.ty.toString();
        const weight_type = op.input_w.ty.toString();

        // Generate the function call
        try writer.print(
            \\    tensMath.convInteger_lean(
            \\        {s}, // Input type (T1)
            \\        {s}, // Weight type (T2)
            \\        {s}, // input x
            \\        {s}, // input w
            \\        {s}, // x_zero_point
            \\        {s}, // w_zero_point
            \\        &tensor_{s}, // output (always i32)
            \\        {s}, // stride
            \\        {s}, // pads
            \\        {s}, // dilations
            \\        {d}, // group
            \\        "{s}", // auto_pad
            \\    ) catch return -1;
        , .{
            input_type, // Input type
            weight_type, // Weight type
            tensor_x_string, // input x
            tensor_w_string, // input w
            x_zero_point_string, // x_zero_point
            w_zero_point_string, // w_zero_point
            try utils.getSanitizedName(op.output_y.name), // output
            stride_string, // stride
            pads_string, // pads
            dilations_string, // dilations
            op.group, // group
            op.auto_pad, // auto_pad
        });
    }

    pub fn print(op: *const ConvInteger) void {
        std.debug.print("\n CONV_INTEGER:\n {any}", .{op});
    }

    pub fn sobstitute_tensors(self: *ConvInteger, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input_x == old_tensor) {
            self.input_x = new_tensor;
            return;
        }
        if (self.input_w == old_tensor) {
            self.input_w = new_tensor;
            return;
        }
        if (self.input_x_zero_point != null and self.input_x_zero_point.? == old_tensor) {
            self.input_x_zero_point = new_tensor;
            return;
        }
        if (self.input_w_zero_point != null and self.input_w_zero_point.? == old_tensor) {
            self.input_w_zero_point = new_tensor;
            return;
        }
        if (self.output_y == old_tensor) {
            self.output_y = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
};
