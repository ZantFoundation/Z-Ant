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

// https://onnx.ai/onnx/operators/onnx__Resize.html
// INPUTS:
//      - X (heterogeneous) - T: Input tensor
//      - roi (optional) - T2: ROI (region of interest) tensor
//      - scales (optional, heterogeneous) - tensor(float): The scale array along each dimension
//      - sizes (optional, heterogeneous) - tensor(int64): Target size of the output tensor
// OUTPUTS:
//      - Y (heterogeneous) - T: Resized output tensor
// ATTRIBUTES:
//      - antialias - INT (default is '0')
//      - axes - INTS
//      - coordinate_transformation_mode - STRING (default is 'half_pixel')
//      - cubic_coeff_a - FLOAT (default is '-0.75')
//      - exclude_outside - INT (default is '0')
//      - extrapolation_value - FLOAT (default is '0.0')
//      - keep_aspect_ratio_policy - STRING (default is 'stretch')
//      - mode - STRING (default is 'nearest')
//      - nearest_mode - STRING (default is 'round_prefer_floor')
//
pub const Resize = struct {
    input_X: *TensorZant,
    input_roi: ?*TensorZant,
    input_scales: ?*TensorZant,
    input_sizes: ?*TensorZant,
    output_Y: *TensorZant,
    // attributes:
    antialias: i64 = 0,
    axes: []i64 = &[_]i64{},
    coordinate_transformation_mode: []const u8 = "half_pixel",
    cubic_coeff_a: f64 = -0.75,
    exclude_outside: i64 = 0,
    extrapolation_value: f64 = 0.0,
    keep_aspect_ratio_policy: []const u8 = "stretch",
    mode: []const u8 = "nearest",
    nearest_mode: []const u8 = "round_prefer_floor",

    pub fn init(nodeProto: *NodeProto) !Resize {
        const input_X = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;

        // ---- optional inputs
        const input_roi: ?*TensorZant = if (nodeProto.input.len >= 2) if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_X_notFound else null;
        const input_scales: ?*TensorZant = if (nodeProto.input.len >= 3) if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[2])) |ptr| ptr else return error.input_roi_notFound else null;
        const input_sizes: ?*TensorZant = if (nodeProto.input.len >= 4) if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[3])) |ptr| ptr else return error.input_sizes_notFound else null;

        const output_Y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        // ---- ATTRIBUTES from NodeProto
        var antialias: i64 = 0;
        var axes: []i64 = &[_]i64{};
        var coordinate_transformation_mode: []const u8 = try allocator.dupe(u8, "half_pixel");
        var cubic_coeff_a: f64 = -0.75;
        var exclude_outside: i64 = 0;
        var extrapolation_value: f64 = 0.0;
        var keep_aspect_ratio_policy: []const u8 = try allocator.dupe(u8, "stretch");
        var mode: []const u8 = try allocator.dupe(u8, "nearest");
        var nearest_mode: []const u8 = try allocator.dupe(u8, "round_prefer_floor");

        for (nodeProto.attribute) |attr| {
            if (std.mem.indexOf(u8, attr.name, "antialias")) |_| {
                if (attr.type == onnx.AttributeType.INT) antialias = attr.i else return error.ResizeAnitialiasNotINT;
            } else if (std.mem.indexOf(u8, attr.name, "axes")) |_| {
                if (attr.type == onnx.AttributeType.INTS) axes = attr.ints else return error.ResizeAxesNotINTS;
            } else if (std.mem.indexOf(u8, attr.name, "coordinate_transformation_mode")) |_| {
                if (attr.type == onnx.AttributeType.STRING) coordinate_transformation_mode = attr.s else return error.Resize_coordinate_transformation_mode_NotSTRING;
            } else if (std.mem.indexOf(u8, attr.name, "cubic_coeff_a")) |_| {
                if (attr.type == onnx.AttributeType.FLOAT) cubic_coeff_a = attr.f else return error.Resize_cubic_coeff_a_NotFLOAT;
            } else if (std.mem.indexOf(u8, attr.name, "exclude_outside")) |_| {
                if (attr.type == onnx.AttributeType.INT) exclude_outside = attr.i else return error.Resize_exclude_outside_NotINT;
            } else if (std.mem.indexOf(u8, attr.name, "extrapolation_value")) |_| {
                if (attr.type == onnx.AttributeType.FLOAT) extrapolation_value = attr.f else return error.Resize_extrapolation_value_NotFLOAT;
            } else if (std.mem.indexOf(u8, attr.name, "keep_aspect_ratio_policy")) |_| {
                if (attr.type == onnx.AttributeType.STRING) keep_aspect_ratio_policy = attr.s else return error.Resize_keep_aspect_ratio_policy_NotSTRING;
            } else if (std.mem.indexOf(u8, attr.name, "mode")) |_| {
                if (attr.type == onnx.AttributeType.STRING) mode = attr.s else return error.Resize_mode_NotSTRING;
            } else if (std.mem.indexOf(u8, attr.name, "nearest_mode")) |_| {
                if (attr.type == onnx.AttributeType.STRING) nearest_mode = attr.s else return error.Resize_nearest_mode_NotSTRING;
            }
        }

        //set the output type:
        if (output_Y.ty == tensorZant_lib.TensorType.undefined) output_Y.ty = input_X.ty;

        return Resize{
            .input_X = input_X,
            .input_roi = input_roi,
            .mode = mode,
            .input_scales = input_scales,
            .input_sizes = input_sizes,
            .coordinate_transformation_mode = coordinate_transformation_mode,
            .output_Y = output_Y,
        };
    }

    pub fn get_output_shape(self: Resize) []usize {
        return self.output_Y.getShape();
    }

    pub fn get_input_tensors(self: Resize) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();
        try inputs.append(self.input_X);
        if (self.input_roi) |x| try inputs.append(x);
        if (self.input_scales) |x| try inputs.append(x);
        if (self.input_sizes) |x| try inputs.append(x);
        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: Resize) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();
        try outputs.append(self.output_Y);
        return outputs.toOwnedSlice();
    }

    pub fn write_op(self: Resize, writer: std.fs.File.Writer) !void {
        //----create tensor_X_string
        var tensor_X_string: []u8 = undefined;
        defer allocator.free(tensor_X_string);

        if (self.input_X.tc == TensorCategory.INITIALIZER) {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_X.name),
                ")",
            });
        } else {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.input_X.name) });
        }

        // ---- optional inputs
        var tensor_roi_string: []const u8 = try allocator.dupe(u8, "null");
        defer {
            if (self.input_roi != null) {
                allocator.free(tensor_roi_string);
            }
        }
        var data_scales_string: []const u8 = try allocator.dupe(u8, "null");
        defer {
            if (self.input_scales != null) {
                allocator.free(data_scales_string);
            }
        }
        var data_sizes_string: []const u8 = try allocator.dupe(u8, "null");
        defer {
            if (self.input_sizes != null) {
                allocator.free(data_sizes_string);
            }
        }

        //----create tensor_roi_string
        if (self.input_roi) |roi| {
            if (roi.tc == TensorCategory.INITIALIZER) {
                tensor_roi_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                    "@constCast(&param_lib.tensor_",
                    try utils.getSanitizedName(roi.name),
                    ")",
                });
            } else {
                tensor_roi_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(roi.name) });
            }
        }

        //----create tensor_scales_string
        if (self.input_scales) |scales| {
            if (scales.tc == TensorCategory.INITIALIZER) {
                data_scales_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                    "param_lib.tensor_",
                    try utils.getSanitizedName(scales.name),
                    ".data",
                });
            } else {
                data_scales_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "tensor_", try utils.getSanitizedName(scales.name), ".data" });
            }
        }

        //----create tensor_sizes_string
        if (self.input_sizes) |sizes| {
            if (sizes.tc == TensorCategory.INITIALIZER) {
                data_sizes_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                    "param_lib.tensor_",
                    try utils.getSanitizedName(sizes.name),
                    ".data",
                });
            } else {
                data_sizes_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "tensor_", try utils.getSanitizedName(sizes.name), ".data" });
            }
        }

        // ---- CREATING ATTRIBUTES strings
        const axes_string = try utils.i64SliceToUsizeArrayString(self.axes);
        _ = axes_string;

        //pub fn rezise_lean(comptime T: type, t: *Tensor(T), comptime mode: []const u8, scales: ?[]const f32, sizes: ?[]const usize, coordinate_transformation_mode: []const u8, output_tensor: *Tensor(T)) !void {
        _ = try writer.print(
            \\
            \\    tensMath.resize_lean(
            \\      {s}, 
            \\      {s}, //*Tensor(T)
            \\      "{s}", //mode
            \\      {s}, //scales: ?[]const f32
            \\      {s}, //sizes: ?[]const usize
            \\      "{s}", //coordinate_transformation_mode: []const u8
            \\      &tensor_{s}, //output_tensor: *Tensor(T)
            \\    ) catch return -1;
        ,
            .{
                self.input_X.ty.toString(), // type
                tensor_X_string, // input
                self.mode,
                data_scales_string,
                data_sizes_string,
                self.coordinate_transformation_mode,
                try utils.getSanitizedName(self.output_Y.name), //output
            },
        );
    }

    pub fn compute_output_shape(self: Resize) []usize {
        var output_shape: []usize = undefined;
        const scales = self.input_scales.?.ptr.?.f32.data;
        const sizes = self.input_sizes.?.ptr.?.i64.data;
        output_shape = try tensorMath.get_resize_output_shape(
            self.input_X.shape,
            scales,
            sizes,
        );
        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Resize) void {
        std.debug.print("\n Resize :{any}\n", .{self});
    }

    pub fn sobstitute_tensors(self: *Resize, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input_X == old_tensor) {
            self.input_X = new_tensor;
            return;
        }
        if (self.input_roi != null and self.input_roi.? == old_tensor) {
            self.input_roi = new_tensor;
            return;
        }
        if (self.input_scales != null and self.input_scales.? == old_tensor) {
            self.input_scales = new_tensor;
            return;
        }
        if (self.input_sizes != null and self.input_sizes.? == old_tensor) {
            self.input_sizes = new_tensor;
            return;
        }
        if (self.output_Y == old_tensor) {
            self.output_Y = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
};
