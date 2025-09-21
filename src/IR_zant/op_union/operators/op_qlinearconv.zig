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

        // Generate the function call - prefer CMSIS Helium when it is part of the build, otherwise fall back to the embedded-friendly path
        const qlinearconv_impl = if (cmsis_codegen_enabled) "qlinearconv_simd_lean" else "qlinearconv_embedded_lean";
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

    pub fn compute_output_shape(self: QLinearConv) ![]usize {
        var output_shape: []usize = undefined;
        const input_shape = self.input_x.getShape();
        const kernel_shape = self.input_w.getShape();

        // Check if input shape is placeholder (common for intermediate tensors)
        if (input_shape.len == 1 and input_shape[0] == 1) {
            // Try to infer a reasonable shape based on the operation parameters
            return self.inferOutputShapeFromParams();
        }

        // Check if kernel shape is valid
        if (kernel_shape.len != 4) {
            return error.InvalidKernelShape;
        }

        // Normalize input shape to 4D by prepending leading 1s if needed
        var normalized_input: [4]usize = .{ 1, 1, 1, 1 };
        switch (input_shape.len) {
            4 => {
                normalized_input = .{ input_shape[0], input_shape[1], input_shape[2], input_shape[3] };
            },
            3 => {
                // Assume missing batch dimension
                normalized_input = .{ 1, input_shape[0], input_shape[1], input_shape[2] };
            },
            2 => {
                normalized_input = .{ 1, 1, input_shape[0], input_shape[1] };
            },
            1 => {
                normalized_input = .{ 1, 1, 1, input_shape[0] };
            },
            else => {
                // Unsupported rank for convolution input
                return error.InvalidInputShape;
            },
        }

        const stride = self.strides;
        const pads = self.pads;
        const dilations = self.dilations;
        const auto_pad = self.auto_pad;
        output_shape = try tensorMath.get_convolution_output_shape(
            u8, // Type parameter
            allocator, // Allocator parameter
            normalized_input[0..],
            kernel_shape,
            try utils.i64SliceToUsizeSlice(stride.?),
            if (pads != null) try utils.i64SliceToUsizeSlice(pads.?) else null,
            try utils.i64SliceToUsizeSlice(dilations.?),
            auto_pad,
        );
        self.output_y.shape = output_shape;
        return output_shape;
    }

    /// Infer output shape from operation parameters when input shape is placeholder
    fn inferOutputShapeFromParams(self: QLinearConv) ![]usize {
        const kernel_shape = self.input_w.getShape();
        if (kernel_shape.len != 4) {
            return error.InvalidKernelShape;
        }

        // Extract operation parameters
        const stride = self.strides;
        const group = self.group;

        std.debug.print("QLinearConv: Mathematical inference - weight={any}, group={}, stride={any}\n", .{ kernel_shape, group, stride });

        // Calculate input channels mathematically based on weight and group
        var inferred_input_shape: [4]usize = undefined;
        inferred_input_shape[0] = 1; // batch size

        // MATHEMATICAL CALCULATION: in_channels = weight_in_channels * group
        const weight_in_channels = kernel_shape[1];
        const calculated_in_channels = weight_in_channels * @as(usize, @intCast(group));
        inferred_input_shape[1] = calculated_in_channels;

        std.debug.print("QLinearConv: Calculated input channels = {} * {} = {}\n", .{ weight_in_channels, group, calculated_in_channels });

        // For spatial dimensions, we can't know the exact input size without tracing the graph
        // But we can make educated mathematical guesses based on common CNN architectures

        // Start with a base size and adjust based on network depth heuristics
        var estimated_spatial_size: usize = 224; // Common input size

        // Apply heuristic based on number of channels (deeper = smaller spatial size)
        if (calculated_in_channels <= 32) {
            estimated_spatial_size = 224; // Early layers - large spatial size
        } else if (calculated_in_channels <= 64) {
            estimated_spatial_size = 112; // Mid layers
        } else if (calculated_in_channels <= 128) {
            estimated_spatial_size = 56; // Mid-deep layers
        } else {
            estimated_spatial_size = 28; // Deep layers - small spatial size
        }

        // Further adjust based on stride (reverse engineering)
        if (stride) |s| {
            if (s.len >= 2 and s[0] == 2) {
                // Stride 2 suggests this is a downsampling layer
                // Input should be 2x larger than typical output for this depth
                estimated_spatial_size = estimated_spatial_size * 2;
            }
        }

        inferred_input_shape[2] = estimated_spatial_size;
        inferred_input_shape[3] = estimated_spatial_size;

        std.debug.print("QLinearConv: Estimated input spatial size = {}x{} (based on {} channels, stride={any})\n", .{ estimated_spatial_size, estimated_spatial_size, calculated_in_channels, stride });

        // Now compute output shape with mathematically inferred input
        const pads = self.pads;
        const dilations = self.dilations;
        const auto_pad = self.auto_pad;

        const input_shape_slice = inferred_input_shape[0..];
        std.debug.print("QLinearConv: About to call get_convolution_output_shape with input={any} kernel={any}\n", .{ input_shape_slice, kernel_shape });

        const output_shape = tensorMath.get_convolution_output_shape(
            u8, // Type parameter
            allocator, // Allocator parameter
            input_shape_slice,
            kernel_shape,
            try utils.i64SliceToUsizeSlice(stride.?),
            if (pads != null) try utils.i64SliceToUsizeSlice(pads.?) else null,
            try utils.i64SliceToUsizeSlice(dilations.?),
            auto_pad,
        ) catch |err| {
            std.debug.print("QLinearConv: get_convolution_output_shape failed with error: {}\n", .{err});
            return err;
        };

        std.debug.print("QLinearConv: Final calculated output shape = {any}\n", .{output_shape});

        self.output_y.shape = output_shape;
        return output_shape;
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
