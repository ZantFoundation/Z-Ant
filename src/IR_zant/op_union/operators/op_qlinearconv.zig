const std = @import("std");
const allocator = std.heap.page_allocator;
const zant = @import("zant");
const IR_zant = @import("../../IR_zant.zig");
const accelerators = zant.core.tensor.accelerators;

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
const NodeZant_lib = IR_zant.NodeZant_lib;
const NodeZant = NodeZant_lib.NodeZant;

const tensorMath = zant.core.tensor.math_standard;
const utils = IR_zant.utils;

const cmsis_codegen_enabled = accelerators.canUseCmsisHelium();

// https://onnx.ai/onnx/operators/onnx__QLinearConv.html
// INPUTS:
//      - x (heterogeneous) - T1: Input tensor (quantized)
//      - x_scale (heterogeneous) - T2: Scale of input x quantization
//      - x_zero_point (heterogeneous) - T1: Zero point of input x quantization
//      - w (heterogeneous) - T1: Weight tensor (quantized)
//      - w_scale (heterogeneous) - T2: Scale of weight w quantization
//      - w_zero_point (heterogeneous) - T1: Zero point of weight w quantization
//      - y_scale (heterogeneous) - T2: Scale of output y quantization
//      - y_zero_point (heterogeneous) - T1: Zero point of output y quantization
//      - B (optional, heterogeneous) - T2: Optional 1D bias tensor
// OUTPUTS:
//      - y (heterogeneous) - T1: Output tensor (quantized)
// ATTRIBUTES:
//      - auto_pad - STRING (default is 'NOTSET')
//      - dilations - INTS : dilation value along each spatial axis of the filter
//      - group - INT (default is '1'): number of groups input channels and output channels are divided into
//      - kernel_shape - INTS : The shape of the convolution kernel
//      - pads - INTS : Padding for the beginning and ending along each spatial axis
//      - strides - INTS : Stride along each spatial axis

pub const QLinearConv = struct {
    input_x: *TensorZant,
    input_x_scale: *TensorZant,
    input_x_zero_point: *TensorZant,
    input_w: *TensorZant,
    input_w_scale: *TensorZant,
    input_w_zero_point: *TensorZant,
    input_y_scale: *TensorZant,
    input_y_zero_point: *TensorZant,
    input_B: ?*TensorZant,
    output_y: *TensorZant,

    // Attributes
    auto_pad: []const u8,
    dilations: ?[]i64,
    group: i64,
    kernel_shape: ?[]i64,
    pads: ?[]i64,
    strides: ?[]i64,

    pub fn init(nodeProto: *NodeProto) !QLinearConv {
        // QLinearConv has 8 or 9 inputs (bias is optional)
        if (nodeProto.input.len < 8 or nodeProto.input.len > 9) {
            return error.QLinearConvInvalidInputCount;
        }

        const input_x = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_x_notFound;
        const input_x_scale = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_x_scale_notFound;
        const input_x_zero_point = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[2])) |ptr| ptr else return error.input_x_zero_point_notFound;
        const input_w = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[3])) |ptr| ptr else return error.input_w_notFound;
        const input_w_scale = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[4])) |ptr| ptr else return error.input_w_scale_notFound;
        const input_w_zero_point = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[5])) |ptr| ptr else return error.input_w_zero_point_notFound;
        const input_y_scale = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[6])) |ptr| ptr else return error.input_y_scale_notFound;
        const input_y_zero_point = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[7])) |ptr| ptr else return error.input_y_zero_point_notFound;
        const input_B = if (nodeProto.input.len > 8) if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[8])) |ptr| ptr else return error.input_B_notFound else null;

        const output_y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_y_notFound;

        var auto_pad: []const u8 = "NOTSET";
        var dilations: ?[]i64 = null;
        var group: i64 = 1;
        var kernel_shape: ?[]i64 = null;
        var pads: ?[]i64 = null;
        var strides: ?[]i64 = null;

        for (nodeProto.attribute) |attr| {
            if (std.mem.indexOf(u8, attr.name, "auto_pad")) |_| {
                if (attr.type == onnx.AttributeType.STRING) auto_pad = attr.s else return error.QLinearConvAuto_padNotSTRING;
            } else if (std.mem.indexOf(u8, attr.name, "dilations")) |_| {
                if (attr.type == onnx.AttributeType.INTS) dilations = attr.ints else return error.QLinearConvDilatationNoINTS;
            } else if (std.mem.indexOf(u8, attr.name, "group")) |_| {
                if (attr.type == onnx.AttributeType.INT) group = attr.i else return error.QLinearConvGroupNotINT;
            } else if (std.mem.indexOf(u8, attr.name, "kernel_shape")) |_| {
                if (attr.type == onnx.AttributeType.INTS) kernel_shape = attr.ints else return error.QLinearConvKernelShapeNotINTS;
            } else if (std.mem.indexOf(u8, attr.name, "pads")) |_| {
                if (attr.type == onnx.AttributeType.INTS) pads = attr.ints else return error.QLinearConvPadsNotINTS;
            } else if (std.mem.indexOf(u8, attr.name, "strides")) |_| {
                if (attr.type == onnx.AttributeType.INTS) strides = attr.ints else return error.QLinearConvStridesNotINTS;
            }
        }

        if (pads == null) {
            const input_spatial_dims = input_x.shape.len;
            const pads_len = input_spatial_dims * 2;
            const default_pads = try allocator.alloc(i64, pads_len);

            for (default_pads) |*pad_val| {
                pad_val.* = 0;
            }

            pads = default_pads;
        }

        if (dilations == null) {
            const input_spatial_dims = input_x.shape.len;
            const dilations_len = input_spatial_dims * 2;
            const default_dilations = try allocator.alloc(i64, dilations_len);

            for (default_dilations) |*dil_val| {
                dil_val.* = 1;
            }

            dilations = default_dilations;
        }

        // Set the output type - for quantized convolution, output type should match input quantized type
        if (output_y.ty == tensorZant_lib.TensorType.undefined) output_y.ty = input_x.ty;

        // For QLinearConv, shape inference is complex and depends on convolution parameters
        // The output shape will be computed later in compute_output_shape() method
        // For now, just mark that it needs computation if it's a placeholder
        if (output_y.shape.len == 1 and output_y.shape[0] == 1) {
            // Keep placeholder shape - will be computed later
        }

        const qlinear_conv = QLinearConv{
            .input_x = input_x,
            .input_x_scale = input_x_scale,
            .input_x_zero_point = input_x_zero_point,
            .input_w = input_w,
            .input_w_scale = input_w_scale,
            .input_w_zero_point = input_w_zero_point,
            .input_y_scale = input_y_scale,
            .input_y_zero_point = input_y_zero_point,
            .input_B = input_B,
            .output_y = output_y,
            .auto_pad = auto_pad,
            .dilations = dilations,
            .group = group,
            .kernel_shape = kernel_shape,
            .pads = pads,
            .strides = strides,
        };

        // Force shape computation during initialization
        _ = qlinear_conv.compute_output_shape() catch {};

        return qlinear_conv;
    }

    pub fn get_output_shape(self: QLinearConv) ![]usize {
        return try self.compute_output_shape();
    }

    pub fn get_input_tensors(self: QLinearConv) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();

        try inputs.append(self.input_x);
        try inputs.append(self.input_x_scale);
        try inputs.append(self.input_x_zero_point);
        try inputs.append(self.input_w);
        try inputs.append(self.input_w_scale);
        try inputs.append(self.input_w_zero_point);
        try inputs.append(self.input_y_scale);
        try inputs.append(self.input_y_zero_point);
        if (self.input_B) |bias| {
            try inputs.append(bias);
        }

        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: QLinearConv) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();

        try outputs.append(self.output_y);
        return outputs.toOwnedSlice();
    }

    pub fn write_op(self: QLinearConv, writer: std.fs.File.Writer) !void {
        // Create tensor string for input x
        var tensor_x_string: []u8 = undefined;
        defer allocator.free(tensor_x_string);
        if (self.input_x.tc == TensorCategory.INITIALIZER) {
            tensor_x_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_x.name),
                ")",
            });
        } else {
            tensor_x_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_x.name), ")" });
        }

        // Create tensor strings for all quantization parameters with null safety
        const x_scale_name = if (self.input_x_scale.name.len > 0) try utils.getSanitizedName(self.input_x_scale.name) else "missing_x_scale";
        const x_zero_point_name = if (self.input_x_zero_point.name.len > 0) try utils.getSanitizedName(self.input_x_zero_point.name) else "missing_x_zero_point";
        const w_name = if (self.input_w.name.len > 0) try utils.getSanitizedName(self.input_w.name) else "missing_w";
        const w_scale_name = if (self.input_w_scale.name.len > 0) try utils.getSanitizedName(self.input_w_scale.name) else "missing_w_scale";
        const w_zero_point_name = if (self.input_w_zero_point.name.len > 0) try utils.getSanitizedName(self.input_w_zero_point.name) else "missing_w_zero_point";
        const y_scale_name = if (self.input_y_scale.name.len > 0) try utils.getSanitizedName(self.input_y_scale.name) else "missing_y_scale";
        const y_zero_point_name = if (self.input_y_zero_point.name.len > 0) try utils.getSanitizedName(self.input_y_zero_point.name) else "missing_y_zero_point";

        // Create bias string - handle missing bias
        var bias_string: []u8 = undefined;
        if (self.input_B) |input_B| {
            if (input_B.name.len > 0) {
                const B_name = try utils.getSanitizedName(input_B.name);
                bias_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", B_name, ")" });
            } else {
                bias_string = try std.mem.concat(allocator, u8, &[_][]const u8{"null"});
            }
        } else {
            bias_string = try std.mem.concat(allocator, u8, &[_][]const u8{"null"});
        }

        // Create stride string
        if (self.strides == null) return error.StrideNotFound;
        const stride_string: []const u8 = try utils.i64SliceToUsizeArrayString(self.strides.?);

        // Create pads string
        var pads_string: []const u8 = "null";
        if (self.pads != null) {
            if (self.pads.?.len > 0) {
                pads_string = try utils.i64SliceToUsizeArrayString(self.pads.?);
            } else {
                pads_string = "&[_]usize{}";
            }
        }

        // Create dilations string
        var dilat_string: []const u8 = "null";
        if (self.dilations != null) {
            if (self.dilations.?.len > 0) {
                dilat_string = try utils.i64SliceToUsizeArrayString(self.dilations.?);
            } else {
                dilat_string = "&[_]usize{}";
            }
        }

        const target_type = self.output_y.ty.toString();

        // Determine the bias type
        const bias_type = if (self.input_B) |bias_tensor| bias_tensor.ty.toString() else "f32";

        // Use compile-time dispatch function that chooses implementation based on CMSIS flags
        const qlinearconv_impl = "qlinearconv_dispatch";
        try writer.print(
            \\    tensMath.{s}(
            \\        {s}, // InputType
            \\        {s}, // WeightType
            \\        {s}, // ScaleType
            \\        {s}, // OutputType
            \\        {s}, // BiasType
            \\        {s}, // input x
            \\        @constCast(&param_lib.tensor_{s}), // x_scale
            \\        @constCast(&param_lib.tensor_{s}), // x_zero_point
            \\        @constCast(&param_lib.tensor_{s}), // w
            \\        @constCast(&param_lib.tensor_{s}), // w_scale
            \\        @constCast(&param_lib.tensor_{s}), // w_zero_point
            \\        &tensor_{s}, // output
            \\        @constCast(&param_lib.tensor_{s}), // y_scale
            \\        @constCast(&param_lib.tensor_{s}), // y_zero_point
            \\        {s}, // bias
            \\        {s}, // stride
            \\        {s}, // pads
            \\        {s}, // dilations
            \\        {d}, // group
            \\        "{s}", // auto_pad
            \\    ) catch return -1;
        , .{
            qlinearconv_impl,
            target_type, // InputType
            self.input_w.ty.toString(), // WeightType (use actual weight type)
            "f32", // ScaleType (scales are always f32)
            self.output_y.ty.toString(), // OutputType (use actual output type)
            bias_type, // BiasType (use actual bias type or f32 default)
            tensor_x_string, // input x
            x_scale_name, // x_scale
            x_zero_point_name, // x_zero_point
            w_name, // w
            w_scale_name, // w_scale
            w_zero_point_name, // w_zero_point
            try utils.getSanitizedName(self.output_y.name), // output
            y_scale_name, // y_scale
            y_zero_point_name, // y_zero_point
            bias_string, // bias
            stride_string, // stride
            pads_string, // pads
            dilat_string, // dilations
            self.group, // group
            self.auto_pad, // auto_pad
        });
    }

    fn hasConcreteShape(shape: []const usize) bool {
        return !(shape.len == 0 or (shape.len == 1 and shape[0] == 1));
    }

    pub fn compute_output_shape(self: QLinearConv) ![]usize {
        const input_shape = self.input_x.getShape();
        const kernel_shape = self.input_w.getShape();

        if (!hasConcreteShape(input_shape)) {
            // Upstream nodes have not provided the real shape yet. Keep the placeholder
            // so we can recompute once the producing tensor is inferred.
            return self.output_y.shape;
        }

        if (kernel_shape.len < 4) {
            return error.InvalidKernelShape;
        }

        var normalized_input: [4]usize = .{ 1, 1, 1, 1 };
        switch (input_shape.len) {
            4 => normalized_input = .{ input_shape[0], input_shape[1], input_shape[2], input_shape[3] },
            3 => normalized_input = .{ 1, input_shape[0], input_shape[1], input_shape[2] },
            2 => normalized_input = .{ 1, 1, input_shape[0], input_shape[1] },
            1 => normalized_input = .{ 1, 1, 1, input_shape[0] },
            else => return error.InvalidInputShape,
        }

        const spatial_rank = kernel_shape.len - 2;
        if (spatial_rank == 0 or spatial_rank > 4) {
            return error.InvalidKernelShape;
        }

        var stride_storage: []usize = undefined;
        var stride_allocated = false;
        defer if (stride_allocated) allocator.free(stride_storage);
        if (self.strides) |s| {
            stride_storage = try utils.i64SliceToUsizeSlice(s);
            stride_allocated = true;
        } else {
            stride_storage = try allocator.alloc(usize, spatial_rank);
            stride_allocated = true;
            for (stride_storage) |*val| {
                val.* = 1;
            }
        }

        if (stride_storage.len < spatial_rank) {
            return error.InvalidInputShape;
        }
        const stride_slice = stride_storage[0..spatial_rank];

        var pads_storage: ?[]usize = null;
        defer if (pads_storage) |buf| allocator.free(buf);
        if (self.pads) |pads_vals| {
            pads_storage = try utils.i64SliceToUsizeSlice(pads_vals);
        }

        var dilation_storage: ?[]usize = null;
        defer if (dilation_storage) |buf| allocator.free(buf);
        if (self.dilations) |dilation_vals| {
            dilation_storage = try utils.i64SliceToUsizeSlice(dilation_vals);
        }

        const computed_array = try tensorMath.get_convolution_output_shape(
            f32, // Type parameter
            allocator, // Allocator parameter
            normalized_input[0..],
            kernel_shape,
            stride_slice,
            if (pads_storage) |buf| buf else null,
            if (dilation_storage) |buf| buf else null,
            self.auto_pad,
        );

        const computed_slice = computed_array[0..];
        const new_shape = try allocator.alloc(usize, computed_slice.len);
        errdefer allocator.free(new_shape);
        std.mem.copyForwards(usize, new_shape, computed_slice);

        const new_stride = try TensorZant.computeStride(new_shape);
        errdefer allocator.free(new_stride);

        const old_shape = self.output_y.shape;
        const old_stride = self.output_y.stride;

        self.output_y.shape = new_shape;
        self.output_y.stride = new_stride;

        if (self.output_y.tc != TensorCategory.INITIALIZER and old_shape.len > 0 and old_shape.ptr != new_shape.ptr) {
            allocator.free(old_shape);
        }
        if (old_stride.len > 0 and old_stride.ptr != new_stride.ptr) {
            allocator.free(old_stride);
        }

        return new_shape;
    }

    pub fn print(self: QLinearConv) !void {
        std.debug.print("\n QLINEARCONV:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *QLinearConv, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input_x == old_tensor) {
            self.input_x = new_tensor;
            return;
        }
        if (self.input_x_scale == old_tensor) {
            self.input_x_scale = new_tensor;
            return;
        }
        if (self.input_x_zero_point == old_tensor) {
            self.input_x_zero_point = new_tensor;
            return;
        }
        if (self.input_w == old_tensor) {
            self.input_w = new_tensor;
            return;
        }
        if (self.input_w_scale == old_tensor) {
            self.input_w_scale = new_tensor;
            return;
        }
        if (self.input_w_zero_point == old_tensor) {
            self.input_w_zero_point = new_tensor;
            return;
        }
        if (self.input_y_scale == old_tensor) {
            self.input_y_scale = new_tensor;
            return;
        }
        if (self.input_y_zero_point == old_tensor) {
            self.input_y_zero_point = new_tensor;
            return;
        }
        if (self.input_B != null and self.input_B.? == old_tensor) {
            self.input_B = new_tensor;
            return;
        }
        if (self.output_y == old_tensor) {
            self.output_y = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
};
