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

const utils = IR_zant.utils;

// https://onnx.ai/onnx/operators/onnx__Conv.html
// INPUTS:
//      - X (heterogeneous) - T: Input data tensor
//      - W (heterogeneous) - T: The weight tensor
//      - B (optional, heterogeneous) - T: Optional 1D bias to be added to the convolution, has size of M.
// OUTPUTS:
//      - Y (heterogeneous) - T: Output data tensor that contains the result of the convolution
// ATTRIBUTES:
//      - auto_pad - STRING (default is 'NOTSET'): auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET
//      - dilations - INTS : dilation value along each spatial axis of the filter. If not present, the dilation defaults is 1 along each spatial axis.
//      - group - INT (default is '1'): number of groups input channels and output channels are divided into
//      - kernel_shape - INTS : The shape of the convolution kernel. If not present, should be inferred from input W
//      - pads - INTS : Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0.
//      - strides - INTS : Stride along each spatial axis. If not present, the stride defaults is 1 along each spatial axis.

pub const Conv = struct {
    input_X: *TensorZant,
    input_W: *TensorZant,
    input_B: ?*TensorZant,
    output_Y: *TensorZant,
    //attributes:
    auto_pad: []const u8,
    dilations: ?[]i64,
    group: i64,
    kernel_shape: ?[]i64,
    pads: ?[]i64,
    strides: ?[]i64,

    pub fn init(nodeProto: *NodeProto) !Conv {
        const input_X = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const input_W = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_W_notFound;
        const input_B = if (nodeProto.input.len > 2) if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[2])) |ptr| ptr else return error.input_B_notFound else null;
        const output_Y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var auto_pad: []const u8 = "NOTSET";
        var dilations: ?[]i64 = null;
        var group: i64 = 1;
        var kernel_shape: ?[]i64 = null;
        var pads: ?[]i64 = null;
        var strides: ?[]i64 = null; //mandatory

        for (nodeProto.attribute) |attr| {
            if (std.mem.indexOf(u8, attr.name, "auto_pad")) |_| {
                if (attr.type == onnx.AttributeType.STRING) auto_pad = attr.s else return error.ConvAuto_padNotSTRING;
            } else if (std.mem.indexOf(u8, attr.name, "dilations")) |_| {
                if (attr.type == onnx.AttributeType.INTS) dilations = attr.ints else return error.ConvDilatationNoINTS;
            } else if (std.mem.indexOf(u8, attr.name, "group")) |_| {
                if (attr.type == onnx.AttributeType.INT) group = attr.i else return error.ConvGroupNotINT;
            } else if (std.mem.indexOf(u8, attr.name, "kernel_shape")) |_| {
                if (attr.type == onnx.AttributeType.INTS) kernel_shape = attr.ints else return error.ConvKernelShapeNotINTS;
            } else if (std.mem.indexOf(u8, attr.name, "pads")) |_| {
                if (attr.type == onnx.AttributeType.INTS) pads = attr.ints else return error.ConvPadsNotINTS;
            } else if (std.mem.indexOf(u8, attr.name, "strides")) |_| {
                if (attr.type == onnx.AttributeType.INTS) strides = attr.ints else return error.ConvStridesNotINTS;
            }
        }

        //set the output type:
        if (output_Y.ty == tensorZant_lib.TensorType.undefined) output_Y.ty = input_W.ty;

        return Conv{
            .input_X = input_X,
            .input_W = input_W,
            .input_B = input_B,
            .output_Y = output_Y,
            .auto_pad = auto_pad,
            .dilations = dilations,
            .group = group,
            .kernel_shape = kernel_shape,
            .pads = pads,
            .strides = strides,
        };
    }

    pub fn get_output_shape(self: Conv) []usize {
        return self.output_Y.getShape();
    }

    pub fn get_input_tensors(self: Conv) ![]*TensorZant {
        var inputs: std.ArrayList(*TensorZant) = .empty;
        defer inputs.deinit(allocator);

        try inputs.append(allocator, self.input_X);
        try inputs.append(allocator, self.input_W);
        if (self.input_B) |bias| {
            try inputs.append(allocator, bias);
        }

        return inputs.toOwnedSlice(allocator);
    }

    pub fn get_output_tensors(self: Conv) ![]*TensorZant {
        var outputs: std.ArrayList(*TensorZant) = .empty;
        defer outputs.deinit(allocator);

        try outputs.append(allocator, self.output_Y);
        return outputs.toOwnedSlice(allocator);
    }

    pub fn write_op(self: Conv, writer: *std.Io.Writer) !void {

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
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_X.name), ")" });
        }

        //----create tensor_W_string
        var tensor_W_string: []u8 = undefined;
        defer allocator.free(tensor_W_string);
        if (self.input_W.tc == TensorCategory.INITIALIZER) {
            tensor_W_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_W.name),
                ")",
            });
        } else {
            tensor_W_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_W.name), ")" });
        }

        //----create ?bias string
        var bias_string: []u8 = undefined;
        // Bias Tensor B is optional! verify the presence
        if (self.input_B) |input_B| {
            const B_name = try utils.getSanitizedName(input_B.name);
            bias_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", B_name, ")" });
        } else {
            bias_string = try std.mem.concat(allocator, u8, &[_][]const u8{"null"});
        }

        //----create stride string (mandatory)
        // TODO: implement default stride, see docs above
        if (self.strides == null) return error.StrideNotFound;
        const stride_string: []const u8 = try utils.i64SliceToUsizeArrayString(self.strides.?);

        //----create ?pads string
        var pads_string: []const u8 = "null";
        if (self.pads != null) {
            if (self.pads.?.len > 0) { // Check if the slice is actually non-empty
                pads_string = try utils.i64SliceToUsizeArrayString(self.pads.?);
                // Assuming no allocation needed to be freed, following write_conv
            } else {
                pads_string = "&[_]usize{}"; // Use explicit empty slice literal if input slice is empty
            }
        } // else pads_string remains "null"

        //----create ?dilatations string
        var dilat_string: []const u8 = "null";
        if (self.dilations != null) {
            if (self.dilations.?.len > 0) {
                dilat_string = try utils.i64SliceToUsizeArrayString(self.dilations.?);
            } else {
                dilat_string = "&[_]usize{}";
            }
        } // else dilat_string remains "null"

        // Check if we need cast operations for mixed precision
        const target_type = self.output_Y.ty.toString();
        const need_kernel_cast = !std.mem.eql(u8, self.input_W.ty.toString(), target_type);
        const need_bias_cast = if (self.input_B) |bias| !std.mem.eql(u8, bias.ty.toString(), target_type) else false;

        var final_kernel_string: []const u8 = undefined;
        var final_bias_string: []const u8 = undefined;
        var need_free_kernel = false;
        var need_free_bias = false;
        defer if (need_free_kernel) allocator.free(@constCast(final_kernel_string));
        defer if (need_free_bias) allocator.free(@constCast(final_bias_string));

        if (need_kernel_cast) {
            // Generate cast for kernel
            const kernel_name = try utils.getSanitizedName(self.input_W.name);
            _ = try writer.print(
                \\
                \\    // Cast kernel from {s} to {s}
                \\    var tensor_{s}_casted = Tensor({s}).fromShape(&allocator, @constCast(param_lib.tensor_{s}.shape)) catch return -2;
                \\    defer tensor_{s}_casted.deinit();
                \\    tensMath.cast_lean({s}, {s}, @constCast(&param_lib.tensor_{s}), &tensor_{s}_casted, IR_zant.onnx.DataType.FLOAT) catch return {d};
                \\
            , .{
                self.input_W.ty.toString(),
                target_type,
                kernel_name,
                target_type,
                kernel_name,
                kernel_name,
                self.input_W.ty.toString(),
                target_type,
                kernel_name,
                kernel_name,
                utils.getMathErrorReturn(), // Error code for math errors
            });
            final_kernel_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", kernel_name, "_casted)" });
            need_free_kernel = true;
        } else {
            final_kernel_string = tensor_W_string;
        }

        if (need_bias_cast and self.input_B != null) {
            // Generate cast for bias
            const bias_name = try utils.getSanitizedName(self.input_B.?.name);
            _ = try writer.print(
                \\
                \\    // Cast bias from {s} to {s}
                \\    var tensor_{s}_casted = Tensor({s}).fromShape(&allocator, @constCast(param_lib.tensor_{s}.shape)) catch return -2;
                \\    defer tensor_{s}_casted.deinit();
                \\    tensMath.cast_lean({s}, {s}, @constCast(&param_lib.tensor_{s}), &tensor_{s}_casted, IR_zant.onnx.DataType.FLOAT) catch return {d};
                \\
            , .{
                self.input_B.?.ty.toString(),
                target_type,
                bias_name,
                target_type,
                bias_name,
                bias_name,
                self.input_B.?.ty.toString(),
                target_type,
                bias_name,
                bias_name,
                utils.getMathErrorReturn(), // Error code for math errors
            });
            final_bias_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", bias_name, "_casted)" });
            need_free_bias = true;
        } else {
            final_bias_string = bias_string;
        }

        // pub fn OnnxConvLean(comptime T: type, input: *Tensor(T), kernel: *Tensor(T), output: *Tensor(T), bias: ?*const Tensor(T), stride: []const usize, pads: ?[]const usize, dilations: ?[]const usize, group: ?usize, auto_pad: ?[]const u8) !void
        _ = try writer.print(
            \\    
            \\    @setEvalBranchQuota(10000);
            \\    tensMath.conv_lean(
            \\        {s}, //type
            \\        {s}, //input
            \\        {s}, //kernel
            \\        &tensor_{s}, //output
            \\        {s}, //bias
            \\        {s}, //stride
            \\        {s}, //pads
            \\        {s}, //dilatations
            \\        {}, //group
            \\        "{s}", //auto_pad
            \\    ) catch return {d};
        , .{
            target_type,
            tensor_X_string, //Input
            final_kernel_string, //Kernel (possibly casted)
            try utils.getSanitizedName(self.output_Y.name), //Output
            final_bias_string, //Bias (possibly casted)
            stride_string, //Strides
            pads_string, //Pads
            dilat_string, //Dilatations
            self.group, //Group
            self.auto_pad, //auto_pad
            utils.getMathErrorReturn(), // Error code for math errors
        });
    }

    pub fn compute_output_shape(self: Conv) ![]usize {
        var output_shape: []usize = undefined;
        const input_shape = self.input_X.getShape();
        const kernel_shape = self.input_W.getShape();
        const stride = self.strides;
        const pads = self.pads;
        const dilations = self.dilations;
        const auto_pad = self.auto_pad;
        output_shape = try tensorMath.get_convolution_output_shape(
            f32, // Type parameter
            allocator, // Allocator parameter
            input_shape,
            kernel_shape,
            try utils.i64SliceToUsizeSlice(stride.?),
            if (pads != null) try utils.i64SliceToUsizeSlice(pads.?) else null,
            try utils.i64SliceToUsizeSlice(dilations.?),
            auto_pad,
        );
        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Conv) void { //TODO
        std.debug.print("\n CONV:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *Conv, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input_X == old_tensor) {
            self.input_X = new_tensor;
            return;
        }
        if (self.input_W == old_tensor) {
            self.input_W = new_tensor;
            return;
        }
        if (self.input_B != null and self.input_B.? == old_tensor) {
            self.input_B = new_tensor;
            return;
        }
        if (self.output_Y == old_tensor) {
            self.output_Y = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
};
