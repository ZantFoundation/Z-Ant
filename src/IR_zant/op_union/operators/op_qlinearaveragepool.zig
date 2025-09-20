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

// https://onnx.ai/onnx/operators/onnx__QLinearAveragePool.html
// INPUTS:
//      - X (heterogeneous) - T: Input tensor (quantized)
//      - X_scale (heterogeneous) - tensor(float): Scale of quantization of input X
//      - X_zero_point (heterogeneous) - T: Zero point of quantization of input X
//      - Y_scale (heterogeneous) - tensor(float): Scale of quantization of output Y
//      - Y_zero_point (heterogeneous) - T: Zero point of quantization of output Y
// OUTPUTS:
//      - Y (heterogeneous) - T: Output tensor (quantized)
// ATTRIBUTES:
//      - auto_pad - STRING (default is 'NOTSET'): auto_pad type
//      - ceil_mode - INT (default is '0'): Whether to use ceil or floor
//      - count_include_pad - INT (default is '0'): Whether to include padding in averaging
//      - dilations - INTS: Dilation value along each spatial axis
//      - kernel_shape - INTS (required): Kernel size along each axis
//      - pads - INTS: Padding for each spatial axis
//      - strides - INTS: Stride along each spatial axis

pub const QLinearAveragePool = struct {
    input_X: *TensorZant,
    input_X_scale: *TensorZant,
    input_X_zero_point: *TensorZant,
    input_Y_scale: *TensorZant,
    input_Y_zero_point: *TensorZant,
    output_Y: *TensorZant,
    // attributes:
    auto_pad: []const u8, // default = "NOTSET"
    ceil_mode: i64, // default = 0
    count_include_pad: i64, // default = 0
    dilations: ?[]i64, // default = null
    kernel_shape: ?[]i64, // default = null, but mandatory
    pads: ?[]i64, // default = null
    strides: ?[]i64, // default = null

    pub fn init(nodeProto: *NodeProto) !QLinearAveragePool {
        // QLinearAveragePool has exactly 5 inputs
        if (nodeProto.input.len != 5) {
            return error.QLinearAveragePoolInvalidInputCount;
        }

        const input_X = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const input_X_scale = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_X_scale_notFound;
        const input_X_zero_point = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[2])) |ptr| ptr else return error.input_X_zero_point_notFound;
        const input_Y_scale = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[3])) |ptr| ptr else return error.input_Y_scale_notFound;
        const input_Y_zero_point = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[4])) |ptr| ptr else return error.input_Y_zero_point_notFound;

        const output_Y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var auto_pad: []const u8 = "NOTSET";
        var ceil_mode: i64 = 0;
        var count_include_pad: i64 = 0;
        var dilations: ?[]i64 = null;
        var kernel_shape: ?[]i64 = null; // mandatory
        var pads: ?[]i64 = null;
        var strides: ?[]i64 = null;

        for (nodeProto.attribute) |attr| {
            if (std.mem.indexOf(u8, attr.name, "auto_pad")) |_| {
                if (attr.type == onnx.AttributeType.STRING) auto_pad = attr.s else return error.QLinearAveragePoolAuto_padNotSTRING;
            } else if (std.mem.indexOf(u8, attr.name, "ceil_mode")) |_| {
                if (attr.type == onnx.AttributeType.INT) ceil_mode = attr.i else return error.QLinearAveragePoolCeil_modeNotINT;
            } else if (std.mem.indexOf(u8, attr.name, "count_include_pad")) |_| {
                if (attr.type == onnx.AttributeType.INT) count_include_pad = attr.i else return error.QLinearAveragePoolCountIncludePadNotINT;
            } else if (std.mem.indexOf(u8, attr.name, "dilations")) |_| {
                if (attr.type == onnx.AttributeType.INTS) dilations = attr.ints else return error.QLinearAveragePoolDilatationNotINTS;
            } else if (std.mem.indexOf(u8, attr.name, "kernel_shape")) |_| {
                if (attr.type == onnx.AttributeType.INTS) kernel_shape = attr.ints else return error.QLinearAveragePoolKernelShapeNotINTS;
            } else if (std.mem.indexOf(u8, attr.name, "pads")) |_| {
                if (attr.type == onnx.AttributeType.INTS) pads = attr.ints else return error.QLinearAveragePoolPadsNotINTS;
            } else if (std.mem.indexOf(u8, attr.name, "strides")) |_| {
                if (attr.type == onnx.AttributeType.INTS) strides = attr.ints else return error.QLinearAveragePoolStridesNotINTS;
            }
        }

        if (dilations == null and kernel_shape != null) {
            dilations = try allocator.alloc(i64, kernel_shape.?.len);
            @memset(dilations.?, 1);
        }

        if (strides == null and kernel_shape != null) {
            strides = try allocator.alloc(i64, kernel_shape.?.len);
            @memset(strides.?, 1);
        }

        // Set the output type
        if (output_Y.ty == tensorZant_lib.TensorType.undefined) output_Y.ty = input_X.ty;

        const qlinear_avgpool = QLinearAveragePool{
            .input_X = input_X,
            .input_X_scale = input_X_scale,
            .input_X_zero_point = input_X_zero_point,
            .input_Y_scale = input_Y_scale,
            .input_Y_zero_point = input_Y_zero_point,
            .output_Y = output_Y,
            .auto_pad = auto_pad,
            .ceil_mode = ceil_mode,
            .count_include_pad = count_include_pad,
            .dilations = dilations,
            .kernel_shape = kernel_shape,
            .pads = pads,
            .strides = strides,
        };

        // Force shape computation during initialization
        _ = qlinear_avgpool.compute_output_shape() catch {};

        return qlinear_avgpool;
    }

    pub fn get_output_shape(self: QLinearAveragePool) []usize {
        return self.output_Y.getShape();
    }

    pub fn get_input_tensors(self: QLinearAveragePool) ![]*TensorZant {
        var input_tensors = std.ArrayList(*TensorZant).init(allocator);
        defer input_tensors.deinit();

        try input_tensors.append(self.input_X);
        try input_tensors.append(self.input_X_scale);
        try input_tensors.append(self.input_X_zero_point);
        try input_tensors.append(self.input_Y_scale);
        try input_tensors.append(self.input_Y_zero_point);

        return input_tensors.toOwnedSlice();
    }

    pub fn get_output_tensors(self: QLinearAveragePool) ![]*TensorZant {
        var output_tensors = std.ArrayList(*TensorZant).init(allocator);
        defer output_tensors.deinit();

        try output_tensors.append(self.output_Y);
        return output_tensors.toOwnedSlice();
    }

    pub fn write_op(self: QLinearAveragePool, writer: std.fs.File.Writer) !void {
        // Generate tensor strings for inputs
        var tensor_X_string: []u8 = undefined;
        defer allocator.free(tensor_X_string);
        var tensor_X_scale_string: []u8 = undefined;
        defer allocator.free(tensor_X_scale_string);
        var tensor_X_zero_point_string: []u8 = undefined;
        defer allocator.free(tensor_X_zero_point_string);
        var tensor_Y_scale_string: []u8 = undefined;
        defer allocator.free(tensor_Y_scale_string);
        var tensor_Y_zero_point_string: []u8 = undefined;
        defer allocator.free(tensor_Y_zero_point_string);

        if (self.input_X.tc == TensorCategory.INITIALIZER) {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_X.name), ")" });
        } else {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_X.name), ")" });
        }

        if (self.input_X_scale.tc == TensorCategory.INITIALIZER) {
            tensor_X_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_X_scale.name), ")" });
        } else {
            tensor_X_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_X_scale.name), ")" });
        }

        if (self.input_X_zero_point.tc == TensorCategory.INITIALIZER) {
            tensor_X_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(@as(*const Tensor(", self.input_X_zero_point.ty.toString(), "), @ptrCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_X_zero_point.name), ")))" });
        } else {
            tensor_X_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(@as(*const Tensor(", self.input_X_zero_point.ty.toString(), "), @ptrCast(&tensor_", try utils.getSanitizedName(self.input_X_zero_point.name), ")))" });
        }

        if (self.input_Y_scale.tc == TensorCategory.INITIALIZER) {
            tensor_Y_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_Y_scale.name), ")" });
        } else {
            tensor_Y_scale_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_Y_scale.name), ")" });
        }

        if (self.input_Y_zero_point.tc == TensorCategory.INITIALIZER) {
            tensor_Y_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(@as(*const Tensor(", self.input_Y_zero_point.ty.toString(), "), @ptrCast(&param_lib.tensor_", try utils.getSanitizedName(self.input_Y_zero_point.name), ")))" });
        } else {
            tensor_Y_zero_point_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(@as(*const Tensor(", self.input_Y_zero_point.ty.toString(), "), @ptrCast(&tensor_", try utils.getSanitizedName(self.input_Y_zero_point.name), ")))" });
        }

        // Convert attributes to proper arrays
        _ = try writer.print("\n    // QLinearAveragePool attributes\n", .{});

        // Generate kernel_shape array
        if (self.kernel_shape) |ks| {
            _ = try writer.print("    var kernel_shape_{s} = [_]usize{{", .{try utils.getSanitizedName(self.output_Y.name)});
            for (ks, 0..) |k, idx| {
                if (idx > 0) _ = try writer.print(", ", .{});
                _ = try writer.print("{}", .{@as(usize, @intCast(k))});
            }
            _ = try writer.print("}};\n", .{});
        }

        // Generate strides array
        if (self.strides) |s| {
            _ = try writer.print("    var strides_{s} = [_]usize{{", .{try utils.getSanitizedName(self.output_Y.name)});
            for (s, 0..) |stride, idx| {
                if (idx > 0) _ = try writer.print(", ", .{});
                _ = try writer.print("{}", .{@as(usize, @intCast(stride))});
            }
            _ = try writer.print("}};\n", .{});
        }

        // Generate dilations array
        if (self.dilations) |d| {
            _ = try writer.print("    var dilations_{s} = [_]usize{{", .{try utils.getSanitizedName(self.output_Y.name)});
            for (d, 0..) |dilation, idx| {
                if (idx > 0) _ = try writer.print(", ", .{});
                _ = try writer.print("{}", .{@as(usize, @intCast(dilation))});
            }
            _ = try writer.print("}};\n", .{});
        }

        // Generate pads array
        if (self.pads) |p| {
            _ = try writer.print("    var pads_{s} = [_]usize{{", .{try utils.getSanitizedName(self.output_Y.name)});
            for (p, 0..) |pad, idx| {
                if (idx > 0) _ = try writer.print(", ", .{});
                _ = try writer.print("{}", .{@as(usize, @intCast(pad))});
            }
            _ = try writer.print("}};\n", .{});
        }

        // Convert auto_pad string to enum
        _ = try writer.print("    const auto_pad_{s} = ", .{try utils.getSanitizedName(self.output_Y.name)});
        if (std.mem.eql(u8, self.auto_pad, "NOTSET")) {
            _ = try writer.print("tensMath.AutoPadType.NOTSET;\n", .{});
        } else if (std.mem.eql(u8, self.auto_pad, "SAME_UPPER")) {
            _ = try writer.print("tensMath.AutoPadType.SAME_UPPER;\n", .{});
        } else if (std.mem.eql(u8, self.auto_pad, "SAME_LOWER")) {
            _ = try writer.print("tensMath.AutoPadType.SAME_LOWER;\n", .{});
        } else if (std.mem.eql(u8, self.auto_pad, "VALID")) {
            _ = try writer.print("tensMath.AutoPadType.VALID;\n", .{});
        } else {
            _ = try writer.print("tensMath.AutoPadType.NOTSET;\n", .{});
        }

        // Write the operation call
        _ = try writer.print(
            \\
            \\    // Perform QLinearAveragePool
            \\    tensMath.lean_qlinearaveragepool(
            \\        {s},
            \\        {s},
            \\        {s},
            \\        {s},
            \\        {s},
            \\        {s},
            \\        {s},
            \\        {s},
            \\        &tensor_{s}
        , .{
            self.input_X.ty.toString(),
            self.input_X_scale.ty.toString(),
            self.input_X_zero_point.ty.toString(),
            tensor_X_string,
            tensor_X_scale_string,
            tensor_X_zero_point_string,
            tensor_Y_scale_string,
            tensor_Y_zero_point_string,
            try utils.getSanitizedName(self.output_Y.name),
        });

        // Add parameters
        if (self.kernel_shape != null) {
            _ = try writer.print(",\n        &kernel_shape_{s}", .{try utils.getSanitizedName(self.output_Y.name)});
        } else {
            _ = try writer.print(",\n        &[_]usize{{2, 2}}", .{});
        }

        if (self.strides != null) {
            _ = try writer.print(",\n        &strides_{s}", .{try utils.getSanitizedName(self.output_Y.name)});
        } else {
            _ = try writer.print(",\n        &[_]usize{{1, 1}}", .{});
        }

        if (self.dilations != null) {
            _ = try writer.print(",\n        &dilations_{s}", .{try utils.getSanitizedName(self.output_Y.name)});
        } else {
            _ = try writer.print(",\n        &[_]usize{{1, 1}}", .{});
        }

        if (self.pads != null) {
            _ = try writer.print(",\n        &pads_{s}", .{try utils.getSanitizedName(self.output_Y.name)});
        } else {
            _ = try writer.print(",\n        &[_]usize{{0, 0, 0, 0}}", .{});
        }

        _ = try writer.print(",\n        auto_pad_{s}", .{try utils.getSanitizedName(self.output_Y.name)});
        _ = try writer.print(",\n        {}", .{self.count_include_pad != 0});
        _ = try writer.print(",\n    ) catch return -1;\n", .{});
    }

    pub fn compute_output_shape(self: QLinearAveragePool) ![]usize {
        // QLinearAveragePool output shape calculation is same as regular AveragePool
        const input_shape = self.input_X.getShape();

        // Convert i64 arrays to usize arrays
        var kernel_shape_usize: []usize = undefined;
        var strides_usize: []usize = undefined;
        var dilations_usize: []usize = undefined;
        var pads_usize: []usize = undefined;

        if (self.kernel_shape) |ks| {
            kernel_shape_usize = try allocator.alloc(usize, ks.len);
            for (ks, 0..) |k, i| {
                kernel_shape_usize[i] = @as(usize, @intCast(k));
            }
        } else {
            kernel_shape_usize = try allocator.alloc(usize, 2);
            kernel_shape_usize[0] = 2;
            kernel_shape_usize[1] = 2;
        }
        defer allocator.free(kernel_shape_usize);

        if (self.strides) |s| {
            strides_usize = try allocator.alloc(usize, s.len);
            for (s, 0..) |stride, i| {
                strides_usize[i] = @as(usize, @intCast(stride));
            }
        } else {
            strides_usize = try allocator.alloc(usize, kernel_shape_usize.len);
            @memset(strides_usize, 1);
        }
        defer allocator.free(strides_usize);

        if (self.dilations) |d| {
            dilations_usize = try allocator.alloc(usize, d.len);
            for (d, 0..) |dilation, i| {
                dilations_usize[i] = @as(usize, @intCast(dilation));
            }
        } else {
            dilations_usize = try allocator.alloc(usize, kernel_shape_usize.len);
            @memset(dilations_usize, 1);
        }
        defer allocator.free(dilations_usize);

        if (self.pads) |p| {
            pads_usize = try allocator.alloc(usize, p.len);
            for (p, 0..) |pad, i| {
                pads_usize[i] = @as(usize, @intCast(pad));
            }
        } else {
            pads_usize = try allocator.alloc(usize, kernel_shape_usize.len * 2);
            @memset(pads_usize, 0);
        }
        defer allocator.free(pads_usize);

        // Convert auto_pad string to enum
        const auto_pad_enum = if (std.mem.eql(u8, self.auto_pad, "SAME_UPPER"))
            tensorMath.AutoPadType.SAME_UPPER
        else if (std.mem.eql(u8, self.auto_pad, "SAME_LOWER"))
            tensorMath.AutoPadType.SAME_LOWER
        else if (std.mem.eql(u8, self.auto_pad, "VALID"))
            tensorMath.AutoPadType.VALID
        else
            tensorMath.AutoPadType.NOTSET;

        const output_shape = try tensorMath.get_qlinearaveragepool_output_shape(
            input_shape,
            kernel_shape_usize,
            strides_usize,
            dilations_usize,
            pads_usize,
            auto_pad_enum,
            self.ceil_mode != 0,
        );

        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: QLinearAveragePool) void {
        std.debug.print("\n QLinearAveragePool:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *QLinearAveragePool, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input_X == old_tensor) {
            self.input_X = new_tensor;
            return;
        }
        if (self.input_X_scale == old_tensor) {
            self.input_X_scale = new_tensor;
            return;
        }
        if (self.input_X_zero_point == old_tensor) {
            self.input_X_zero_point = new_tensor;
            return;
        }
        if (self.input_Y_scale == old_tensor) {
            self.input_Y_scale = new_tensor;
            return;
        }
        if (self.input_Y_zero_point == old_tensor) {
            self.input_Y_zero_point = new_tensor;
            return;
        }
        if (self.output_Y == old_tensor) {
            self.output_Y = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
};
