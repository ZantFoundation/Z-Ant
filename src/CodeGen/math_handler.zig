const std = @import("std");
const os = std.os;

const zant = @import("zant");

const Tensor = zant.core.tensor.Tensor;
const tensorMath = zant.core.tensor.math_standard;
const onnx = zant.onnx;
const ModelOnnx = onnx.ModelProto;
const DataType = onnx.DataType;
const allocator = zant.utils.allocator.allocator;

// --- proto libs
const TensorProto = onnx.TensorProto;
const NodeProto = onnx.NodeProto;
const GraphProto = onnx.GraphProto;
const AttributeType = onnx.AttributeType;

// --- codeGen libs
const ReadyNode = @import("globals.zig").ReadyNode;
const ReadyTensor = @import("globals.zig").ReadyTensor;
const codegen = @import("codegen.zig");
const utils = codegen.utils;
const parameters = codegen.parameters;
const codegen_options = @import("codegen_options");
const globals = @import("globals.zig");

// ----------------------------------- MATH -----------------------------------

/// This method map and write the ONNX operations with the Zant LeanTensorMath mathods
/// Follow the link for details: https://onnx.ai/onnx/operators/?utm_source=chatgpt.com
pub fn write_math_op(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    if (codegen_options.comm) {
        try write_op_info(writer, node);
    }
    if (codegen_options.log) {
        try writer.print(
            \\ 
            \\
            \\    if (log_function) |log| {{
            \\        log(@constCast(@ptrCast("Running {s} operation...\n")));
            \\    }}
        , .{node.*.nodeProto.*.op_type});
    }

    if (std.mem.eql(u8, node.nodeProto.op_type, "Add")) {
        try write_add(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "AveragePool")) {
        try writer.writeAll("// Handle AveragePool\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "BatchNormalization")) {
        try writer.writeAll("// Handle BatchNormalization\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Ceil")) {
        try write_ceil(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Concat")) {
        try write_concat(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Constant")) {
        try write_constant(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Conv")) {
        try write_conv(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Div")) {
        try write_div(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Flatten")) {
        try writer.writeAll("// Handle Flatten\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Gather")) {
        try write_gather(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Gemm")) {
        try write_gemm(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Identity")) {
        try write_identity(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "LeakyRelu")) {
        try write_leaky_relu(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "LogSoftmax")) {
        try writer.writeAll("// Handle LogSoftmax\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "MatMul")) {
        try write_matmul(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "MaxPool")) {
        try write_maxPool(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Mul")) {
        try write_mul(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Neg")) {
        try write_neg(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "OneHot")) {
        try writer.writeAll("// Handle OneHot\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "ReduceMean")) {
        try write_reduceMean(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Relu")) {
        try write_ReLU(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Reshape")) {
        try write_reshape(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Resize")) {
        try write_resize(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Sigmoid")) {
        try write_sigmoid(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Softmax")) {
        try write_softmax(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Slice")) {
        try write_slice(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Split")) {
        try write_split(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Sub")) {
        try write_sub(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Sum")) {
        try write_sum(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Transpose")) {
        try write_transpose(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Shape")) {
        try write_shape(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Unsqueeze")) {
        try write_unsqueeze(writer, node);
    } else {
        return error.OperationNotSupported;
    }

    try writer.writeAll(" catch return;");
}

fn write_op_info(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    try writer.print(
        \\
        \\
        \\   //forwarding operation : {s}
        \\   //parameters:
        \\   //   inputs: 
    , .{node.*.nodeProto.*.op_type});

    //write the inputs
    for (node.inputs.items) |input| {
        try writer.print(
            \\
            \\   //      -> {s} 
        , .{input.name});
    }
    try writer.print(
        \\
        \\   //    outputs: 
    , .{});

    //write the outputs
    for (node.outputs.items) |output| {
        try writer.print(
            \\
            \\   //      <- {s} 
        , .{output.name});
    }
}

inline fn write_add(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Add.html
    // INPUTS:
    //      - A (heterogeneous) - T: First operand.
    //      - B (heterogeneous) - T: Second operand.
    // OUTPUTS:
    //      - C (heterogeneous) - T: Result, has same element type as two inputs.

    //----create tensor_A_string
    var tensor_A_string: []u8 = undefined;
    defer allocator.free(tensor_A_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    //----create tensor_B_string
    var tensor_B_string: []u8 = undefined;
    defer allocator.free(tensor_B_string);
    if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[1].?.name),
            ")",
        });
    } else {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[1].?.name) });
    }

    _ = try writer.print(
        \\
        \\    tensMath.sum_tensors_lean(T, T, {s}, {s}, &tensor_{s})
    , .{
        tensor_A_string, // Input tensor A
        tensor_B_string, // Input tensor B
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor C
    });
}

inline fn write_sub(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Sub.html
    // INPUTS:
    //      - A (heterogeneous) - T: First operand.
    //      - B (heterogeneous) - T: Second operand.
    // OUTPUTS:
    //      - C (heterogeneous) - T: Result, has same element type as two inputs.

    //----create tensor_A_string
    var tensor_A_string: []u8 = undefined;
    defer allocator.free(tensor_A_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name), ")" });
    }

    //----create tensor_B_string
    var tensor_B_string: []u8 = undefined;
    defer allocator.free(tensor_B_string);
    if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[1].?.name),
            ")",
        });
    } else {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(node.inputs.items[1].?.name), ")" });
    }

    _ = try writer.print(
        \\
        \\    tensMath.sub_tensors_lean(T, T, {s}, ({s}), &tensor_{s})
    , .{
        tensor_A_string, // Input tensor A
        tensor_B_string, // Input tensor B
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor C
    });
}

inline fn write_conv(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Conv.html
    // INPUTS:
    //      - X (heterogeneous) - T: Input data tensor
    //      - W (heterogeneous) - T: The weight tensor
    //      - B (optional, heterogeneous) - T: Optional 1D bias to be added to the convolution, has size of M.
    // OUTPUT:
    //      - Y (heterogeneous) - T: Output data tensor that contains the result of the convolution
    // ATTRIBUTES:
    //      - auto_pad - STRING (default is 'NOTSET'): auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET
    //      - dilations - INTS : dilation value along each spatial axis of the filter. If not present, the dilation defaults is 1 along each spatial axis.
    //      - group - INT (default is '1'): number of groups input channels and output channels are divided into
    //      - kernel_shape - INTS : The shape of the convolution kernel. If not present, should be inferred from input W
    //      - pads - INTS : Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0.
    //      - strides - INTS : Stride along each spatial axis. If not present, the stride defaults is 1 along each spatial axis.

    var auto_pad: []const u8 = "NOTSET";
    var dilations: ?[]i64 = null;
    var group: i64 = 1;
    var kernel_shape: ?[]i64 = null;
    var pads: ?[]i64 = null;
    var strides: ?[]i64 = null; //mandatory

    for (node.nodeProto.attribute) |attr| {
        if (std.mem.indexOf(u8, attr.name, "auto_pad")) |_| {
            if (attr.type == AttributeType.STRING) auto_pad = attr.s else return error.ConvAuto_padNotSTRING;
        } else if (std.mem.indexOf(u8, attr.name, "dilations")) |_| {
            if (attr.type == AttributeType.INTS) dilations = attr.ints else return error.ConvDilatationNoINTS;
        } else if (std.mem.indexOf(u8, attr.name, "group")) |_| {
            if (attr.type == AttributeType.INT) group = attr.i else return error.ConvGroupNotINT;
        } else if (std.mem.indexOf(u8, attr.name, "kernel_shape")) |_| {
            if (attr.type == AttributeType.INTS) kernel_shape = attr.ints else return error.ConvKernelShapeNotINTS;
        } else if (std.mem.indexOf(u8, attr.name, "pads")) |_| {
            if (attr.type == AttributeType.INTS) pads = attr.ints else return error.ConvPadsNotINTS;
        } else if (std.mem.indexOf(u8, attr.name, "strides")) |_| {
            if (attr.type == AttributeType.INTS) strides = attr.ints else return error.ConvStridesNotINTS;
        }
    }

    //----create tensor_X_string
    var tensor_X_string: []u8 = undefined;
    defer allocator.free(tensor_X_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    //----create tensor_W_string
    var tensor_W_string: []u8 = undefined;
    defer allocator.free(tensor_W_string);
    if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_W_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[1].?.name),
            ")",
        });
    } else {
        tensor_W_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[1].?.name) });
    }

    //----create ?bias string
    var bias_string: []u8 = undefined;
    // Bias Tensor B is optional! verify the presence
    if (node.inputs.items.len == 3) {
        const B_name = try utils.getSanitizedName(node.inputs.items[2].?.name);
        bias_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", B_name, ")" });
    } else {
        bias_string = try std.mem.concat(allocator, u8, &[_][]const u8{"null"});
    }

    //----create stride string (mandatory)
    // TODO: implement default stride, see docs above
    if (strides == null) return error.StrideNotFound;
    const stride_string: []const u8 = try utils.i64SliceToUsizeArrayString(strides.?);

    //----create ?pads string
    var pads_string: []const u8 = undefined;
    if (pads != null) {
        pads_string = try utils.i64SliceToUsizeArrayString(pads.?);
    } else {
        pads_string = try std.mem.concat(allocator, u8, &[_][]const u8{" null"});
    }

    //----create ?dilatations string
    var dilat_string: []const u8 = undefined;
    if (dilations != null) {
        dilat_string = try utils.i64SliceToUsizeArrayString(dilations.?);
    } else {
        dilat_string = try std.mem.concat(allocator, u8, &[_][]const u8{" null"});
    }

    // pub fn OnnxConvLean(comptime T: type, input: *Tensor(T), kernel: *Tensor(T), output: *Tensor(T), bias: ?*const Tensor(T), stride: []const usize, pads: ?[]const usize, dilations: ?[]const usize, group: ?usize, auto_pad: ?[]const u8) !void
    _ = try writer.print(
        \\    
        \\
        \\    tensMath.conv_lean(
        \\        T, //type
        \\        {s}, //input
        \\        {s}, //kernel
        \\        &tensor_{s}, //output
        \\        {s}, //bias
        \\        {s}, //stride
        \\        {s}, //pads
        \\        {s}, //dilatations
        \\        {}, //group
        \\        "{s}", //auto_pad
        \\    )
    , .{
        tensor_X_string, //Input
        tensor_W_string, //Kernel
        try utils.getSanitizedName(node.outputs.items[0].name), //Output
        bias_string, //Bias
        stride_string, //Strides
        pads_string, //Pads
        dilat_string, //Dilatations
        group, //Group
        auto_pad, //auto_pad
    });
}

inline fn write_concat(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Concat.html
    // INPUTS:
    //      - inputs (variadic, heterogeneous) - T: List of tensors for concatenation
    // OUTPUTS:
    //      - concat_result (heterogeneous) - T: Concatenated tensor
    // ATTRIBUTES:
    //      - axis (int, required): Which axis to concat on

    // Get the axis attribute
    var axis: i64 = 0;
    var axis_found = false;

    for (node.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "axis")) {
            if (attr.type == AttributeType.INT) {
                axis = attr.i;
                axis_found = true;
            } else {
                return error.ConcatAxisNotINT;
            }
        }
    }

    if (!axis_found) {
        return error.ConcatAxisNotFound;
    }

    // Special case for axis 0 with different ranks
    if (axis == 0) {
        // Find if there are tensors with different ranks
        var has_different_ranks = false;
        const first_rank = node.inputs.items[0].?.shape.len;

        for (node.inputs.items[1..]) |input| {
            if (input.?.shape.len != first_rank) {
                has_different_ranks = true;
                break;
            }
        }

        if (has_different_ranks) {
            _ = try writer.print(
                \\
                \\    // Special case for concatenation along axis 0 with different ranks
                \\    // This requires custom handling as the standard concatenate function expects same rank
                \\    std.debug.print("\\nWarning: Concatenating tensors with different ranks along axis 0\\n", .{{}});
                \\
                \\    // Create a list of tensors to concatenate
                \\    var concat_tensor_list_{s} = [_]Tensor(T){{
            , .{try utils.getSanitizedName(node.outputs.items[0].name)});

            for (node.inputs.items, 0..) |input, idx| {
                if (idx > 0) {
                    _ = try writer.print(", ", .{});
                }

                var tensor_string: []u8 = undefined;
                defer allocator.free(tensor_string);
                if (input.?.tag == globals.TensorTag.INITIALIZER) {
                    tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                        "@constCast(&param_lib.tensor_",
                        try utils.getSanitizedName(input.?.name),
                        ")",
                    });
                } else {
                    tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(input.?.name) });
                }
                _ = try writer.print("{s}", .{tensor_string});
            }

            _ = try writer.print(
                \\}};
                \\
                \\    // Perform concatenation with special handling for different ranks
                \\     try tensMath.concatenate_lean(T, &allocator, &concat_tensor_list_{s}, {},tensor_{s})
            , .{
                try utils.getSanitizedName(node.outputs.items[0].name),
                axis,
                try utils.getSanitizedName(node.outputs.items[0].name),
            });

            return;
        }
    }

    // Standard case: all tensors have the same rank
    // Create a tensor list with all input tensors
    _ = try writer.print(
        \\
        \\    // Create a list of tensors to concatenate
        \\    var concat_tensor_list_{s} = [_]Tensor(T){{
    , .{try utils.getSanitizedName(node.outputs.items[0].name)});

    for (node.inputs.items, 0..) |input, idx| {
        if (idx > 0) {
            _ = try writer.print(", ", .{});
        }

        if (input.?.tag == globals.TensorTag.INITIALIZER) {
            _ = try writer.print("param_lib.tensor_{s}", .{try utils.getSanitizedName(input.?.name)});
        } else {
            _ = try writer.print("tensor_{s}", .{try utils.getSanitizedName(input.?.name)});
        }
    }

    _ = try writer.print(
        \\}};
        \\
        \\    // Perform concatenation
        \\    tensMath.concatenate_lean(T, &allocator, &concat_tensor_list_{s}, {}, &tensor_{s} )
    , .{
        try utils.getSanitizedName(node.outputs.items[0].name),
        axis,
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_constant(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Constant.html
    // Outputs:
    // - output (heterogeneous) - T: Output tensor containing the same value of the provided tensor.
    // Attributes - only one of these should be specified:
    // - value (TENSOR): The value for the elements of the output tensor.
    // - sparse_value (SPARSE_TENSOR): The value for the elements of the output tensor in sparse format.
    // - value_float (FLOAT): The value for the sole element for the scalar, float32, output tensor.
    // - value_floats (FLOATS): The values for the elements for the 1D, float32, output tensor.
    // - value_int (INT): The value for the sole element for the scalar, int64, output tensor.
    // - value_ints (INTS): The values for the elements for the 1D, int64, output tensor.
    // - value_string (STRING): The value for the sole element for the scalar, UTF-8 string, output tensor.
    // - value_strings (STRINGS): The values for the elements for the 1D, UTF-8 string, output tensor.

    const output_name = try utils.getSanitizedName(node.outputs.items[0].name);

    for (node.nodeProto.attribute) |attr| {
        if (attr.type == onnx.AttributeType.TENSOR) {
            try writer.print(
                \\
                \\    // Constant tensor_{s} already declared and inizialized in predict.zig write_constantTensor()
            , .{output_name});

            return;
        } else if (std.mem.eql(u8, attr.name, "value_float")) {
            if (attr.type != AttributeType.FLOAT) return error.ConstantAttributeTypeMismatch;

            // Create a scalar tensor with a float value
            try writer.print(
                \\
                \\    // Initialize scalar float constant
                \\    tensor_{s} = Tensor(T).initScalar(&allocator, {d}) catch return;
            , .{ output_name, attr.f });
            return;
        } else if (std.mem.eql(u8, attr.name, "value_floats")) {
            if (attr.type != AttributeType.FLOATS) return error.ConstantAttributeTypeMismatch;

            // Create 1D tensor with float values
            try writer.print(
                \\
                \\    // Initialize 1D float array constant
                \\    const data_{s} = [_]T{{
            , .{output_name});

            // Write array elements
            for (attr.floats, 0..) |val, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("{d}", .{val});
            }

            try writer.print(
                \\
                \\    }};
                \\    tensor_{s} = Tensor(T).fromSlice(&allocator, &data_{s}, &[_]usize{{{d}}}) catch return;
            , .{ output_name, output_name, attr.floats.len });
            return;
        } else if (std.mem.eql(u8, attr.name, "value_int")) {
            if (attr.type != AttributeType.INT) return error.ConstantAttributeTypeMismatch;

            // Create a scalar tensor with an int value
            try writer.print(
                \\
                \\    // Initialize scalar int constant
                \\    tensor_{s} = Tensor(T).initScalar(&allocator, @as(T, @floatFromInt({d}))) catch return;
            , .{ output_name, attr.i });
            return;
        } else if (std.mem.eql(u8, attr.name, "value_ints")) {
            if (attr.type != AttributeType.INTS) return error.ConstantAttributeTypeMismatch;

            // Create 1D tensor with int values
            try writer.print(
                \\
                \\    // Initialize 1D int array constant
                \\    const data_{s} = [_]T{{
            , .{output_name});

            // Write array elements
            for (attr.ints, 0..) |val, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("@as(T, @floatFromInt({d}))", .{val});
            }

            try writer.print(
                \\
                \\    }};
                \\    tensor_{s} = Tensor(T).fromSlice(&allocator, &data_{s}, &[_]usize{{{d}}}) catch return;
            , .{ output_name, output_name, attr.ints.len });
            return;
        } else if (std.mem.eql(u8, attr.name, "value_string")) {
            if (attr.type != AttributeType.STRING) return error.ConstantAttributeTypeMismatch;

            // String constants are not directly supported in this numeric tensor library
            try writer.print(
                \\
                \\    // String constants are not directly supported in this numeric tensor library
                \\    // For now, we'll create a placeholder tensor with a single value
                \\    tensor_{s} = Tensor(T).initScalar(&allocator, 0) catch return;
                \\    // The actual string value was: "{s}"
            , .{ output_name, attr.s });
            return;
        } else if (std.mem.eql(u8, attr.name, "value_strings")) {
            if (attr.type != AttributeType.STRINGS) return error.ConstantAttributeTypeMismatch;

            // String array constants are not directly supported in this numeric tensor library
            try writer.print(
                \\
                \\    // String array constants are not directly supported in this numeric tensor library
                \\    // For now, we'll create a placeholder tensor with zeros
                \\    const data_{s} = [_]T{{
            , .{output_name});

            // Create a placeholder array of zeros with the same length
            for (attr.strings, 0..) |_, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("0", .{});
            }

            try writer.print(
                \\
                \\    }};
                \\    tensor_{s} = Tensor(T).fromSlice(&allocator, &data_{s}, &[_]usize{{{d}}}) catch return;
                \\    // Note: This is a placeholder for string values that cannot be directly represented
            , .{ output_name, output_name, attr.strings.len });
            return;
        } else if (std.mem.eql(u8, attr.name, "sparse_value")) {
            // Sparse tensor constants require special handling
            try writer.print(
                \\
                \\    // Sparse tensor constants are not yet fully supported
                \\    // Creating a placeholder tensor for sparse_value
                \\    tensor_{s} = Tensor(T).initScalar(&allocator, 0) catch return;
                \\    std.debug.print("Warning: sparse_value attribute used but not fully supported\\n", .{{}});
            , .{output_name});
            return;
        }
    }

    // If we get here, no valid constant value was found
    try writer.writeAll(
        \\
        \\    return error.ConstantValueNotFound;
    );
}

inline fn write_div(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Div.html
    // INPUTS:
    //      - A (heterogeneous) - T: First operand.
    //      - B (heterogeneous) - T: Second operand.
    // OUTPUTS:
    //      - C (heterogeneous) - T: Result, has same element type as two inputs.

    //----create tensor_A_string
    var tensor_A_string: []u8 = undefined;
    defer allocator.free(tensor_A_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",

            try utils.getSanitizedName(node.inputs.items[0].?.name),

            ")",
        });
    } else {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    //----create tensor_B_string
    var tensor_B_string: []u8 = undefined;
    defer allocator.free(tensor_B_string);
    if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",

            try utils.getSanitizedName(node.inputs.items[1].?.name),

            ")",
        });
    } else {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(node.inputs.items[1].?.name), ")" });
    }

    _ = try writer.print(
        \\
        \\    tensMath.div_lean(T, {s}, ({s}), &tensor_{s})
    , .{
        tensor_A_string, // Input tensor A
        tensor_B_string, // Input tensor B
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor C
    });
}

inline fn write_gather(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Gather.html
    // INPUTS:
    //      - data (heterogeneous) - T: Tensor of rank r >= 1.
    //      - indices (heterogeneous) - tensor(int64): Tensor of int64 indices, of any rank q.
    // OUTPUTS:
    //      - output (heterogeneous) - T: Tensor of rank q + r - 1.
    // ATTRIBUTES:
    //      - axis (int, default is 0): Which axis to gather on. Negative value means counting dimensions from the back.

    var axis: i64 = 0;
    for (node.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "axis")) {
            if (attr.type == AttributeType.INT) axis = attr.i;
        }
    }

    // Create data tensor string
    var data_tensor_string: []u8 = undefined;
    defer allocator.free(data_tensor_string);
    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        data_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        data_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    // Create indices tensor string
    const indices_name = try utils.getSanitizedName(node.inputs.items[1].?.name);
    var indices_tensor_string: []u8 = undefined;
    defer allocator.free(indices_tensor_string);
    if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
        indices_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "param_lib.tensor_",
            indices_name,
        });
    } else {
        indices_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "tensor_",
            indices_name,
        });
    }

    _ = try writer.print(
        \\    
        \\
        \\    //creating the indices Tensor(usize)
        \\    
        \\    const usize_slice_{s} = utils.sliceToUsizeSlice({s}.data);
        \\    var usize_tensor_{s} = Tensor(usize).fromConstBuffer(&allocator, usize_slice_{s}, {s}.shape);
        \\    defer usize_tensor_{s}.deinit();
        \\    defer allocator.free(usize_slice_{s});
        \\    
    , .{
        indices_name, //usize_slice_
        indices_tensor_string, //tensor_
        indices_name, //usize_tensor_
        indices_name, //usize_slice_
        indices_tensor_string, //tensor_.shape
        indices_name, //usize_tensor_.deinit
        indices_name, //usize_slice_ for free
    });

    _ = try writer.print(
        \\
        \\
        \\    tensMath.gather_lean(
        \\        T, //type
        \\        {s}, //data tensor
        \\        &usize_tensor_{s}, //indices tensor
        \\        {}, //axis
        \\        &tensor_{s}, //output tensor
        \\    )
    , .{
        data_tensor_string, // Input data tensor
        indices_name, // Input indices tensor
        axis, // Selected axis
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor
    });
}

inline fn write_gemm(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Gemm.html
    // INPUTS:
    //      - Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M) if transA is non-zero.
    //      - Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K) if transB is non-zero.
    //      - Optional input tensor C. If not specified, the computation is done as if C is a scalar 0. The shape of C should be unidirectional broadcastable to (M, N).
    //OUTPUTS:
    //      - Output tensor of shape (M, N).
    // ATTRIBUTES:
    //      - alpha. FLOAT (default is '1.0'): Scalar multiplier for the product of input tensors A * B.
    //      - beta - FLOAT (default is '1.0'): Scalar multiplier for input tensor C.
    //      - transA - INT (default is '0'): Whether A should be transposed
    //      - transB - INT (default is '0'): Whether B should be transposed

    var alpha: f32 = 1.0;
    var beta: f32 = 1.0;
    var transA: bool = false;
    var transB: bool = false;

    for (node.nodeProto.attribute) |attr| {
        if (std.mem.indexOf(u8, attr.name, "alpha")) |_| {
            if (attr.type == AttributeType.FLOAT) alpha = attr.f else return error.GemmAphaNotFLOAT;
        } else if (std.mem.indexOf(u8, attr.name, "beta")) |_| {
            if (attr.type == AttributeType.FLOAT) beta = attr.f else return error.GemmBetaNotFLOAT;
        } else if (std.mem.indexOf(u8, attr.name, "transA")) |_| {
            if (attr.type == AttributeType.INT) transA = if (attr.i != 0) true else false else return error.GemmTransANotINT;
        } else if (std.mem.indexOf(u8, attr.name, "transB")) |_| {
            if (attr.type == AttributeType.INT) transB = if (attr.i != 0) true else false else return error.GemmTransBNotINT;
        }
    }

    //----create tensor_A_string
    var tensor_A_string: []u8 = undefined;
    defer allocator.free(tensor_A_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    //----create tensor_B_string
    var tensor_B_string: []u8 = undefined;
    defer allocator.free(tensor_B_string);
    if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[1].?.name),
            ")",
        });
    } else {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[1].?.name) });
    }

    // Input Tensor C is optional! verify the presence
    var tensor_C_string: []u8 = undefined;
    if (node.inputs.items.len == 3) {
        const sanitized_tensor_C = try utils.getSanitizedName(node.inputs.items[2].?.name);
        tensor_C_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&",
            if (globals.tensorHashMap.getPtr(node.inputs.items[2].?.name).?.tag == globals.TensorTag.INITIALIZER) "param_lib." else "",
            "tensor_",
            sanitized_tensor_C,
            ")",
        });
    } else {
        tensor_C_string = try std.mem.concat(allocator, u8, &[_][]const u8{" null"});
    }

    _ = try writer.print(
        \\
        \\
        \\    tensMath.gemm_lean(T, {s}, {s}, {s}, {}, {}, {s}, {s}, &tensor_{s} )
    , .{
        tensor_A_string, // Input tensor A
        tensor_B_string, // Input tensor B
        tensor_C_string,
        alpha,
        beta,
        if (transA) "true" else "false",
        if (transB) "true" else "false",
        try utils.getSanitizedName(node.outputs.items[0].name), // Output
    });
}

inline fn write_matmul(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__MatMul.html
    // INPUTS:
    //      - A (heterogeneous) - T: First operand.
    //      - B (heterogeneous) - T: Second operand.
    // OUTPUTS:
    //      - C (heterogeneous) - T: Result, has same element type as two inputs.

    //----create tensor_A_string
    var tensor_A_string: []u8 = undefined;
    defer allocator.free(tensor_A_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    //----create tensor_B_string
    var tensor_B_string: []u8 = undefined;
    defer allocator.free(tensor_B_string);
    if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[1].?.name),
            ")",
        });
    } else {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[1].?.name) });
    }

    _ = try writer.print(
        \\
        \\    tensMath.mat_mul_lean(T, {s}, {s}, &tensor_{s})
    , .{
        tensor_A_string, // Input tensor A
        tensor_B_string, // Input tensor B
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor C
    });
}

inline fn write_maxPool(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    //https://onnx.ai/onnx/operators/onnx__MaxPool.html
    // INPUTS:
    //      - X (heterogeneous) - T: Input data tensor
    // OUTPUTS:
    //      - Y (heterogeneous) - T: Output data tensor from average or max pooling across the input tensor.
    //      - (NOT IMPLEMENTED) Indices (optional, heterogeneous) - I: Indices tensor from max pooling across the input tensor.
    // ATTRIBUTES:
    //      - auto_pad - STRING (default is 'NOTSET'): auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID
    //      - ceil_mode - INT (default is '0'): Whether to use ceil or floor (default) to compute the output shape
    //      - dilations - INTS : Dilation value along each spatial axis of filter. If not present, the dilation defaults to 1 along each spatial axis
    //      - kernel_shape - INTS (required) : The size of the kernel along each axis.
    //      - pads - INTS : Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0.
    //      - storage_order - INT (default is '0'): The storage order of the tensor. 0 is row major, and 1 is column major. This attribute is used only to convert an n-tuple index value into a single integer value for producing the second output.
    //      - strides - INTS : Stride along each spatial axis. If not present, the stride defaults to 1 along each spatial axis.

    var auto_pad: []const u8 = "NOTSET";

    var ceil_mode: i64 = 0;

    var dilations: ?[]i64 = null;

    var kernel_shape: ?[]i64 = null; //mandatory

    var pads: ?[]i64 = null;

    var storage_order: i64 = 0;

    var strides: ?[]i64 = null;

    for (node.nodeProto.attribute) |attr| {
        if (std.mem.indexOf(u8, attr.name, "auto_pad")) |_| {
            if (attr.type == AttributeType.STRING) auto_pad = attr.s else return error.MaxPoolAuto_padNotSTRING;
        } else if (std.mem.indexOf(u8, attr.name, "ceil_mode")) |_| {
            if (attr.type == AttributeType.INT) ceil_mode = attr.i else return error.MaxPoolCeil_modeNotINT;
        } else if (std.mem.indexOf(u8, attr.name, "dilations")) |_| {
            if (attr.type == AttributeType.INTS) dilations = attr.ints else return error.MaxPoolDilatationNoINTS;
        } else if (std.mem.indexOf(u8, attr.name, "kernel_shape")) |_| {
            if (attr.type == AttributeType.INTS) kernel_shape = attr.ints else return error.MaxPoolKernelShapeNotINTS;
        } else if (std.mem.indexOf(u8, attr.name, "pads")) |_| {
            if (attr.type == AttributeType.INTS) pads = attr.ints else return error.MaxPoolPadsNotINTS;
        } else if (std.mem.indexOf(u8, attr.name, "storage_order")) |_| {
            if (attr.type == AttributeType.INT) storage_order = attr.i else return error.MaxPoolStorage_orderNotINT;
        } else if (std.mem.indexOf(u8, attr.name, "strides")) |_| {
            if (attr.type == AttributeType.INTS) strides = attr.ints else return error.MaxPoolStridesNotINTS;
        }
    }

    //----create tensor_X_string
    var tensor_X_string: []u8 = undefined;
    defer allocator.free(tensor_X_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    //----create kernel_shape string
    var kernel_shape_string: []const u8 = undefined;
    if (kernel_shape != null) {
        kernel_shape_string = try utils.i64SliceToUsizeArrayString(kernel_shape.?);
    } else {
        return error.Kernel_shapeNotFound;
    }

    //----create strides string
    var strides_string: []const u8 = undefined;
    if (strides != null) {
        strides_string = try utils.i64SliceToUsizeArrayString(strides.?);
    } else {
        return error.StridesNotFound;
    }

    //----create dilations string
    var dilations_string: []const u8 = undefined;
    if (dilations != null) {
        dilations_string = try utils.i64SliceToUsizeArrayString(dilations.?);
    } else {
        dilations_string = try utils.i64SliceToUsizeArrayString(&[_]i64{ 1, 1, 1, 1 }); // TODO: It is hardcoded in 4D, not the most elegant solution
    }

    //----create pads string
    var pads_string: []const u8 = undefined;
    if (pads != null) {
        pads_string = try utils.i64SliceToUsizeArrayString(pads.?);
    } else {
        return error.PadsNotFound;
    }

    _ = try writer.print(
        \\
        \\
        \\    tensMath.onnx_maxpool_lean(
        \\        T,
        \\        {s}, //Input
        \\        &tensor_{s}, //Output
        \\        {s}, //kernel_shape
        \\        {s}, //strides
        \\        {s}, //dilations
        \\        {s}, //pads
        \\        tensMath.AutoPadType.{s}, //auto_pad
        \\    )
    , .{
        tensor_X_string, //Input
        try utils.getSanitizedName(node.outputs.items[0].name), //Output
        kernel_shape_string, //kernel_shape
        strides_string, //strides
        dilations_string, //dilatations
        pads_string, //pads
        auto_pad, //auto_pad
    });
}

inline fn write_mul(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Mul.html
    // INPUTS:
    //      - A (heterogeneous) - T: First operand.
    //      - B (heterogeneous) - T: Second operand.
    // OUTPUTS:
    //      - C (heterogeneous) - T: Result, has same element type as two inputs.

    //----create tensor_A_string
    var tensor_A_string: []u8 = undefined;
    defer allocator.free(tensor_A_string);
    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    //----create tensor_B_string
    var tensor_B_string: []u8 = undefined;
    defer allocator.free(tensor_B_string);
    if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[1].?.name),
            ")",
        });
    } else {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(node.inputs.items[1].?.name), ")" });
    }

    _ = try writer.print(
        \\
        \\
        \\    tensMath.mul_lean(T, {s}, ({s}), &tensor_{s})
    , .{
        tensor_A_string, // Input tensor A
        tensor_B_string, // Input tensor B
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor C
    });
}

inline fn write_reduceMean(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__ReduceMean.html
    // INPUTS:
    //      - data (heterogeneous) - T: An input tensor.
    //      - axes (optional, heterogeneous) - tensor(int64): A list of integers, along which to reduce. The default is to reduce over all the dimensions of the input tensor if 'keepdims' is true.
    // OUTPUTS:
    //      - reduced (heterogeneous) - T: Reduced output tensor.
    // ATTRIBUTES:
    //      - keepdims (int, default is 1): Keep the reduced dimension or not, default 1 means keep the reduced dimension.
    //      - noop_with_empty_axes (int, default is 0): Defines behavior if 'axes' is empty. Default behavior is to reduce all axes.

    // Get attributes
    var keepdims: bool = true;
    var noop_with_empty_axes: bool = false;
    var axes_attr: ?[]i64 = null;

    for (node.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "keepdims")) {
            if (attr.type == AttributeType.INT) keepdims = attr.i != 0;
        } else if (std.mem.eql(u8, attr.name, "noop_with_empty_axes")) {
            if (attr.type == AttributeType.INT) noop_with_empty_axes = attr.i != 0;
        } else if (std.mem.eql(u8, attr.name, "axes")) {
            if (attr.type == AttributeType.INTS) axes_attr = attr.ints;
        }
    }

    // Create input tensor string
    var input_tensor_string: []u8 = undefined;
    defer allocator.free(input_tensor_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    // Handle axes - either from attribute, input tensor, or as null
    var axes_str: []const u8 = "null";
    var needs_free = false;

    // First check if axes is defined as an attribute
    if (axes_attr != null) {
        // Create a static array from the axes attribute
        const axes_array_name = try std.fmt.allocPrint(allocator, "axes_{s}", .{try utils.getSanitizedName(node.outputs.items[0].name)});
        defer allocator.free(axes_array_name);

        try writer.print(
            \\
            \\    // Define axes array from attribute
            \\    const {s} = [_]i64{{
        , .{axes_array_name});

        for (axes_attr.?, 0..) |axis, i| {
            if (i > 0) try writer.writeAll(", ");
            try writer.print("{d}", .{axis});
        }

        try writer.print(
            \\}};
            \\
        , .{});

        axes_str = try std.fmt.allocPrint(allocator, "&{s}", .{axes_array_name});
        needs_free = true;
    }
    // If not found in attributes, check if provided as an input tensor
    else if (node.inputs.items.len > 1) {
        // Get axes from second input
        const axes_name = try utils.getSanitizedName(node.inputs.items[1].?.name);

        if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
            // For initializer tensors, we need to extract the data directly
            axes_str = try std.fmt.allocPrint(allocator, "(@ptrCast([*]const i64, param_lib.tensor_{s}.data.ptr))[0..param_lib.tensor_{s}.size]", .{ axes_name, axes_name });
        } else {
            // For regular tensors
            axes_str = try std.fmt.allocPrint(allocator, "(@ptrCast([*]const i64, tensor_{s}.data.ptr))[0..tensor_{s}.size]", .{ axes_name, axes_name });
        }
        needs_free = true;
    }
    defer if (needs_free) allocator.free(axes_str);

    _ = try writer.print(
        \\
        \\    tensMath.reduce_mean_lean(
        \\        T, // type
        \\        {s}, // input tensor
        \\        &tensor_{s}, // output tensor
        \\        {s}, // axes
        \\        {s}, // keepdims
        \\        {s} // noop_with_empty_axes
        \\    )
    , .{
        input_tensor_string,
        try utils.getSanitizedName(node.outputs.items[0].name),
        axes_str,
        if (keepdims) "true" else "false",
        if (noop_with_empty_axes) "true" else "false",
    });
}

inline fn write_ReLU(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    //node.inputs.items[0].? -> input
    //node.outputs.items[0] -> output

    //----create tensor_A_string
    var tensor_A_string: []u8 = undefined;
    defer allocator.free(tensor_A_string);
    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    _ = try writer.print(
        \\
        \\
        \\    tensMath.ReLU_lean(T, {s}, &tensor_{s})
    , .{
        tensor_A_string,
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_reshape(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Reshape.html
    // INPUTS:
    //      - data (heterogeneous) - T: An input tensor.
    //      - shape (heterogeneous) - tensor(int64): Specified shape for output.
    // OUTPUTS:
    //      - reshaped (heterogeneous) - T: Reshaped data.
    // ATTRIBUTES:
    //      - allowzero - INT (default is '0'): Whether to allow zeros in shape tensor

    var allowzer0: bool = false;
    for (node.nodeProto.attribute) |attr| {
        if (std.mem.indexOf(u8, attr.name, "allowzero")) |_| {
            if (attr.type == AttributeType.INT) allowzer0 = attr.i != 0;
        }
    }

    //input string creation
    var input_string: []u8 = undefined;
    const sanitized_input_name = try utils.getSanitizedName(node.inputs.items[0].?.name);
    input_string = try std.mem.concat(allocator, u8, &[_][]const u8{
        if (globals.tensorHashMap.getPtr(node.inputs.items[0].?.name).?.tag == globals.TensorTag.INITIALIZER) "param_lib." else "",
        "tensor_",
        sanitized_input_name,
    });

    //shape string creation
    const sanitized_shape_name = try utils.getSanitizedName(node.inputs.items[1].?.name);
    const shape_tensor_name = try std.mem.concat(allocator, u8, &[_][]const u8{
        if (globals.tensorHashMap.getPtr(node.inputs.items[1].?.name).?.tag == globals.TensorTag.INITIALIZER) "param_lib." else "",
        "tensor_",
        sanitized_shape_name,
    });
    defer allocator.free(shape_tensor_name);

    _ = try writer.print(
        \\
        \\    // Convert shape tensor data to isize slice
        \\    const shape_slice_{s} = utils.sliceToIsizeSlice({s}.data);
        \\    defer allocator.free(shape_slice_{s});
        \\
        \\    tensMath.reshape_lean(
        \\        T, //type
        \\        @constCast(&{s}), //Input tensor
        \\        shape_slice_{s}, //New shape (converted to isize)
        \\        {s}, //allowzero
        \\        &tensor_{s}, //Output tensor
        \\    )
    , .{
        sanitized_shape_name, // For the temporary slice name
        shape_tensor_name, // Shape tensor
        sanitized_shape_name, // For freeing the slice
        input_string, // Input tensor
        sanitized_shape_name, // The converted shape slice
        if (allowzer0) "true" else "false", //allowzer0
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor
    });
}

inline fn write_sigmoid(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    //node.inputs.items[0].? -> input
    //node.outputs.items[0] -> output

    //----create tensor_A_string
    var tensor_A_string: []u8 = undefined;
    defer allocator.free(tensor_A_string);
    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    _ = try writer.print(
        \\
        \\
        \\    tensMath.sigmoid_lean(T, {s}, &tensor_{s})
    , .{
        tensor_A_string,
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

const Converter = zant.utils.type_converter;

/// Helper function to cast tensor data to i64 array
fn castTensorDataToI64Array(tensor_string: []const u8) ![]const u8 {
    // Create a temporary array and initialize with the tensor data
    return try std.fmt.allocPrint(allocator,
        \\blk: {{
        \\    const data_slice = {s};
        \\    // Define the result array in one go, directly applying the conversion
        \\    var temp_i64_arr =  allocator.alloc(i64, data_slice.len) catch return;
        \\    for (data_slice, 0..) |val, i| {{
        \\        temp_i64_arr[i] = if (@typeInfo(@TypeOf(val)) == .int) val else @intFromFloat(val);
        \\    }}
        \\    // The result array will be managed by the caller
        \\    break :blk temp_i64_arr;
        \\}}
    , .{tensor_string});
}

inline fn write_slice(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Slice.html
    // INPUTS:
    //      - input (heterogeneous) - T: Tensor of data to extract slices from.
    //      - starts (heterogeneous) - T1: 1-D tensor of starting indices of corresponding axis in `axes`.
    //      - ends (heterogeneous) - T1: 1-D tensor of ending indices (exclusive) of corresponding axis in `axes`.
    //      - axes (heterogeneous) - T1: 1-D tensor of axes that `starts` and `ends` apply to.
    //      - steps (heterogeneous) - T1: 1-D tensor of slice step of corresponding axis in `axes`.
    // OUTPUTS:
    //      - output (heterogeneous) - T: Sliced data tensor.

    // First, get the sanitized names for all tensors
    const input_name = try utils.getSanitizedName(node.inputs.items[0].?.name);
    const starts_name = try utils.getSanitizedName(node.inputs.items[1].?.name);
    const ends_name = try utils.getSanitizedName(node.inputs.items[2].?.name);
    const output_name = try utils.getSanitizedName(node.outputs.items[0].name);

    // Create input tensor string
    var input_tensor_string: []u8 = undefined;
    defer allocator.free(input_tensor_string);
    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", input_name, ")" });
    } else {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", input_name, ")" });
    }

    // Create starts tensor string
    var starts_tensor_string: []u8 = undefined;
    defer allocator.free(starts_tensor_string);
    if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
        starts_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "param_lib.tensor_", starts_name, ".data" });
    } else {
        starts_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "tensor_", starts_name, ".data" });
    }

    // Create cast code for starts
    const starts_i64_code = try castTensorDataToI64Array(starts_tensor_string);
    defer allocator.free(starts_i64_code);

    // Create ends tensor string
    var ends_tensor_string: []u8 = undefined;
    defer allocator.free(ends_tensor_string);
    if (node.inputs.items[2].?.tag == globals.TensorTag.INITIALIZER) {
        ends_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "param_lib.tensor_", ends_name, ".data" });
    } else {
        ends_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "tensor_", ends_name, ".data" });
    }

    // Create cast code for ends
    const ends_i64_code = try castTensorDataToI64Array(ends_tensor_string);
    defer allocator.free(ends_i64_code);

    // Handle optional axes and steps inputs
    var axes_str: []const u8 = "null";
    var axes_i64_code: []const u8 = "null";
    var steps_str: []const u8 = "null";
    var steps_i64_code: []const u8 = "null";

    if (node.inputs.items.len > 3) {
        const axes_name = try utils.getSanitizedName(node.inputs.items[3].?.name);
        if (node.inputs.items[3].?.tag == globals.TensorTag.INITIALIZER) {
            axes_str = try std.fmt.allocPrint(allocator, "param_lib.tensor_{s}.data", .{axes_name});
        } else {
            axes_str = try std.fmt.allocPrint(allocator, "tensor_{s}.data", .{axes_name});
        }
        axes_i64_code = try castTensorDataToI64Array(axes_str);
        defer if (axes_str.len > 4) allocator.free(axes_str);
    }

    if (node.inputs.items.len > 4) {
        const steps_name = try utils.getSanitizedName(node.inputs.items[4].?.name);
        if (node.inputs.items[4].?.tag == globals.TensorTag.INITIALIZER) {
            steps_str = try std.fmt.allocPrint(allocator, "param_lib.tensor_{s}.data", .{steps_name});
        } else {
            steps_str = try std.fmt.allocPrint(allocator, "tensor_{s}.data", .{steps_name});
        }
        steps_i64_code = try castTensorDataToI64Array(steps_str);
        defer if (steps_str.len > 4) allocator.free(steps_str);
    }

    // Generate defer code for axes and steps
    var axes_defer_code: []const u8 = "";
    var steps_defer_code: []const u8 = "";
    var axes_var_code: []const u8 = "null";
    var steps_var_code: []const u8 = "null";
    var axes_decl_code: []const u8 = "";
    var steps_decl_code: []const u8 = "";

    if (axes_str.len > 4) {
        axes_defer_code = try std.fmt.allocPrint(allocator, "defer allocator.free(axes_arr_{s});", .{output_name});
        axes_var_code = try std.fmt.allocPrint(allocator, "axes_arr_{s}", .{output_name});
        axes_decl_code = try std.fmt.allocPrint(allocator, "const axes_arr_{s} = {s};", .{ output_name, axes_i64_code });
    }

    if (steps_str.len > 4) {
        steps_defer_code = try std.fmt.allocPrint(allocator, "defer allocator.free(steps_arr_{s});", .{output_name});
        steps_var_code = try std.fmt.allocPrint(allocator, "steps_arr_{s}", .{output_name});
        steps_decl_code = try std.fmt.allocPrint(allocator, "const steps_arr_{s} = {s};", .{ output_name, steps_i64_code });
    }

    defer {
        if (axes_defer_code.len > 0) allocator.free(axes_defer_code);
        if (steps_defer_code.len > 0) allocator.free(steps_defer_code);
        if (axes_var_code.len > 4) allocator.free(axes_var_code);
        if (steps_var_code.len > 4) allocator.free(steps_var_code);
        if (axes_decl_code.len > 0) allocator.free(axes_decl_code);
        if (steps_decl_code.len > 0) allocator.free(steps_decl_code);
    }

    _ = try writer.print(
        \\
        \\
        \\    // Allocate arrays for slice operation
        \\    const starts_arr_{s} = {s};
        \\    const ends_arr_{s} = {s};
        \\    {s}
        \\    {s}
        \\    defer allocator.free(starts_arr_{s});
        \\    defer allocator.free(ends_arr_{s});
        \\    {s}
        \\    {s}
        \\
        \\    tensMath.slice_onnx_lean(
        \\        T, //type
        \\        {s}, //input tensor
        \\        starts_arr_{s}, //starts (casted to i64)
        \\        ends_arr_{s}, //ends (casted to i64)
        \\        {s}, //axes (casted to i64 if not null)
        \\        {s}, //steps (casted to i64 if not null)
        \\        &tensor_{s}, //output tensor
        \\    )
    , .{
        // Variable names with unique suffixes
        output_name,
        starts_i64_code,
        output_name,
        ends_i64_code,
        // Only declare axes and steps if they're used
        if (axes_decl_code.len > 0) axes_decl_code else "// no axes needed",
        if (steps_decl_code.len > 0) steps_decl_code else "// no steps needed",
        output_name,
        output_name,
        if (axes_defer_code.len > 0) axes_defer_code else "// no axes to free",
        if (steps_defer_code.len > 0) steps_defer_code else "// no steps to free",
        input_tensor_string,
        output_name,
        output_name,
        axes_var_code,
        steps_var_code,
        output_name,
    });

    // Free any allocated memory for the i64 code strings
    if (axes_i64_code.len > 4 and axes_str.len > 4) allocator.free(axes_i64_code);
    if (steps_i64_code.len > 4 and steps_str.len > 4) allocator.free(steps_i64_code);
}

inline fn write_softmax(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    //node.inputs.items[0].? -> input
    //node.outputs.items[0] -> output

    //----create tensor_A_string
    var tensor_A_string: []u8 = undefined;
    defer allocator.free(tensor_A_string);
    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    _ = try writer.print(
        \\
        \\
        \\    tensMath.softmax_lean(T, {s}, &tensor_{s})
    , .{
        tensor_A_string,
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_sum(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Sum.html
    // INPUTS:
    //      - list of tensors
    // OUTPUTS:
    //      - sum (heterogeneous) - T: Output tensor.

    //Writing the tensor list with all the inputs
    _ = try writer.print(
        \\
        \\
        \\    const my_tensor_list = [_]*const Tensor(T){{
    , .{});

    for (node.inputs.items, 0..) |tens, idx| {
        if (idx > 0) {
            _ = try writer.print(", ", .{});
        }

        var new_tensor_string: []u8 = undefined;
        const sanitized_tensor_name = try utils.getSanitizedName(tens.?.name);

        new_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            if (globals.tensorHashMap.getPtr(tens.?.name).?.tag == globals.TensorTag.INITIALIZER) "param_lib." else "",
            "tensor_",
            sanitized_tensor_name,
        });

        _ = try writer.print(
            \\{s}
        , .{try utils.getSanitizedName(new_tensor_string)});
    }

    _ = try writer.print("}}", .{});

    _ = try writer.print(
        \\
        \\    tensMath.sum_tensor_list_lean(T, T, &my_tensor_list, &tensor_{s})
    , .{try utils.getSanitizedName(node.outputs.items[0].name)});
}

inline fn write_shape(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Shape.html
    // INPUTS:
    //      - data (heterogeneous) - T: An input tensor.
    // OUTPUTS:
    //      - shape (heterogeneous) - T1: Shape of the input tensor
    // ATTRIBUTES:
    //      - start - INT: First dimension to take
    //      - end - INT: Last dimension to take

    var start: ?i64 = null;
    var end: ?i64 = null;

    for (node.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "start")) {
            if (attr.type == AttributeType.INT) start = attr.i;
        } else if (std.mem.eql(u8, attr.name, "end")) {
            if (attr.type == AttributeType.INT) end = attr.i;
        }
    }

    //----create tensor_A_string
    var tensor_A_string: []u8 = undefined;
    defer allocator.free(tensor_A_string);
    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&param_lib.tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    } else {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    _ = try writer.print(
        \\
        \\    tensMath.shape_onnx_lean(
        \\        T,
        \\        T, //type
        \\        @constCast({s}), //input tensor
        \\        {s}, //start
        \\        {s}, //end
        \\        &tensor_{s}, //output tensor,
        \\    )
    , .{
        tensor_A_string,
        if (start) |s| try std.fmt.allocPrint(allocator, "{}", .{s}) else "null",
        if (end) |e| try std.fmt.allocPrint(allocator, "{}", .{e}) else "null",
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_unsqueeze(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Unsqueeze.html
    // INPUTS:
    //      - data (heterogeneous) - T: Original tensor
    //      - axes (optional) - tensor(int64): List of integers indicating the dimensions to be inserted.
    //        Negative value means counting dimensions from the back.
    // OUTPUTS:
    //      - expanded (heterogeneous) - T: Reshaped tensor with same data as input.
    // ATTRIBUTES (deprecated in opset 13):
    //      - axes - INTS: List of integers indicating the dimensions to be inserted.

    if (node.inputs.items[0]) |input_tensor| {
        const input_name = try utils.getSanitizedName(input_tensor.name);
        const output_name = try utils.getSanitizedName(node.outputs.items[0].name);

        // Create input tensor string
        var input_tensor_string: []u8 = undefined;
        defer allocator.free(input_tensor_string);
        if (input_tensor.tag == globals.TensorTag.INITIALIZER) {
            input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", input_name, ")" });
        } else {
            input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", input_name, ")" });
        }

        // Determine if axes is provided as an input tensor or as an attribute
        var axes_str: []const u8 = "null";
        var needs_free = false;

        if (node.inputs.items.len > 1) {
            // Axes is provided as an input tensor (opset 13+)
            const axes_tensor_name = try utils.getSanitizedName(node.inputs.items[1].?.name);
            if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
                axes_str = try std.fmt.allocPrint(allocator, "@constCast(&param_lib.tensor_{s})", .{axes_tensor_name});
            } else {
                axes_str = try std.fmt.allocPrint(allocator, "&tensor_{s}", .{axes_tensor_name});
            }
            needs_free = true;
        } else {
            // Axes is provided as an attribute (opset < 13)
            for (node.nodeProto.attribute) |attr| {
                if (std.mem.eql(u8, attr.name, "axes")) {
                    if (attr.type == AttributeType.INTS) {
                        axes_str = try utils.i64ToI64ArrayString(attr.ints);
                        needs_free = true;
                        break;
                    }
                }
            }
        }

        defer if (needs_free) allocator.free(axes_str);

        // Generate code for the unsqueeze operation
        try writer.print(
            \\     
            \\    tensMath.unsqueeze_lean(
            \\        T, //type
            \\        {s}, //input tensor
            \\        {s}, //axes tensor
            \\        &tensor_{s}, //output tensor
            \\    )
        , .{
            input_tensor_string, //input tensor
            axes_str, //axes tensor
            output_name, //output tensor
        });
    } else {
        return error.InvalidInput;
    }
}

inline fn write_transpose(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Transpose.html
    // INPUTS:
    //      - data (heterogeneous) - T: An input tensor.
    // OUTPUTS:
    //      - transposed (heterogeneous) - T: Transposed output.
    // ATTRIBUTES:
    //      - perm - INTS: A list of integers. By default, reverse the dimensions,
    //        otherwise permute the axes according to the values given.

    // Get the perm attribute if it exists
    var perm_str: []const u8 = "null";
    for (node.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "perm")) {
            if (attr.type == AttributeType.INTS) {
                perm_str = try utils.i64SliceToUsizeArrayString(attr.ints);
            }
        }
    }

    //----create tensor_A_string
    var tensor_A_string: []u8 = undefined;
    defer allocator.free(tensor_A_string);
    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&param_lib.tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    } else {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    _ = try writer.print(
        \\
        \\
        \\    tensMath.transpose_onnx_lean(
        \\        T, //type
        \\        @constCast({s}), //input tensor
        \\        {s}, //perm
        \\        &tensor_{s}, //output tensor
        \\    )
    , .{
        tensor_A_string, // Input tensor
        perm_str, // Permutation array
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor
    });
}

inline fn write_ceil(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Ceil.html
    // INPUTS:
    //      - X (heterogeneous) - T: Input tensor
    // OUTPUTS:
    //      - Y (heterogeneous) - T: Output tensor with ceiling of input elements

    // Create input tensor string
    var input_tensor_string: []u8 = undefined;
    defer allocator.free(input_tensor_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    _ = try writer.print(
        \\
        \\
        \\    tensMath.ceil_lean(T, {s}, &tensor_{s})
    , .{
        input_tensor_string,
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_identity(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Identity.html
    // INPUTS:
    //      - input (heterogeneous) - T: Input tensor
    // OUTPUTS:
    //      - output (heterogeneous) - T: Tensor with same shape and contents as input

    // Create input tensor string
    var input_tensor_string: []u8 = undefined;
    defer allocator.free(input_tensor_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    _ = try writer.print(
        \\
        \\
        \\    tensMath.identity_lean(T, {s}, &tensor_{s})
    , .{
        input_tensor_string,
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_leaky_relu(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__LeakyRelu.html
    // INPUTS:
    //      - X (heterogeneous) - T: Input tensor
    // OUTPUTS:
    //      - Y (heterogeneous) - T: Output tensor
    // ATTRIBUTES:
    //      - alpha (float, default is 0.01): Coefficient of leakage

    // Get alpha attribute, default to 0.01 if not specified
    var alpha: f32 = 0.01;
    for (node.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "alpha")) {
            if (attr.type == AttributeType.FLOAT) alpha = attr.f;
        }
    }

    // Create input tensor string
    var input_tensor_string: []u8 = undefined;
    defer allocator.free(input_tensor_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    _ = try writer.print(
        \\
        \\    tensMath.leakyReLU_lean(T, {s}, {d}, &tensor_{s})
    , .{
        input_tensor_string,
        alpha,
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}

inline fn write_split(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Split.html
    // INPUTS:
    //      - input (heterogeneous) - T: The tensor to split
    //      - split (optional, heterogeneous) - tensor(int64): Optional tensor specifying the size of each split
    // OUTPUTS:
    //      - outputs (variadic, heterogeneous) - T: One or more outputs forming splits of the input
    // ATTRIBUTES:
    //      - axis (int, default is 0): Which axis to split on
    //      - split (list of ints, deprecated): Length of each output. This attribute is deprecated in favor of the 'split' input

    // Get axis attribute (default is 0)
    var axis: i64 = 0;
    var split_sizes_attr: ?[]i64 = null;

    for (node.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "axis")) {
            if (attr.type == AttributeType.INT) axis = attr.i;
        } else if (std.mem.eql(u8, attr.name, "split")) {
            if (attr.type == AttributeType.INTS) split_sizes_attr = attr.ints;
        }
    }

    // Create input tensor string
    var input_tensor_string: []u8 = undefined;
    defer allocator.free(input_tensor_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    // Handle split sizes - either from input tensor or attribute
    var split_sizes_str: []const u8 = "null";
    var needs_free = false;

    if (node.inputs.items.len > 1 and node.inputs.items[1].?.tensorProto != null) {
        // Split sizes from input tensor (opset 13+)
        const output_name = try utils.getSanitizedName(node.outputs.items[0].name);

        // Extract split sizes from the input tensor
        try writer.print(
            \\
            \\    // Extract split sizes from the input tensor
        , .{});

        // For initializers, access directly from the parameter library
        if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
            try writer.print(
                \\
                \\    const split_sizes_tensor_{0s} = param_lib.tensor_{1s};
                \\    var split_sizes_{0s} = allocator.alloc(usize, split_sizes_tensor_{0s}.size) catch @panic("Out of memory");
                \\    defer allocator.free(split_sizes_{0s});
                \\    
                \\    // Convert int64 data to usize
                \\    for (split_sizes_tensor_{0s}.data, 0..) |val, i| {{
                \\        split_sizes_{0s}[i] = @as(usize, @intFromFloat(val));
                \\    }}
            , .{ output_name, try utils.getSanitizedName(node.inputs.items[1].?.name) });
        } else {
            try writer.print(
                \\
                \\    const split_sizes_tensor_{0s} = tensor_{1s};
                \\    var split_sizes_{0s} = allocator.alloc(usize, split_sizes_tensor_{0s}.size) catch @panic("Out of memory");
                \\    defer allocator.free(split_sizes_{0s});
                \\    
                \\    // Convert int64 data to usize
                \\    for (split_sizes_tensor_{0s}.data, 0..) |val, i| {{
                \\        split_sizes_{0s}[i] = @as(usize, @intFromFloat(val));
                \\    }}
            , .{ output_name, try utils.getSanitizedName(node.inputs.items[1].?.name) });
        }

        split_sizes_str = try std.fmt.allocPrint(allocator, "split_sizes_{s}", .{output_name});
        needs_free = true;
    } else if (split_sizes_attr != null) {
        // Split sizes from attribute (deprecated but still supported)
        const split_array_name = try std.fmt.allocPrint(allocator, "split_sizes_{s}", .{try utils.getSanitizedName(node.outputs.items[0].name)});
        defer allocator.free(split_array_name);

        try writer.print(
            \\
            \\    // Define split sizes array
            \\    const {s} = [_]i64{{
        , .{split_array_name});

        for (split_sizes_attr.?, 0..) |size, i| {
            if (i > 0) try writer.writeAll(", ");
            try writer.print("{d}", .{size});
        }

        try writer.print(
            \\}};
            \\
        , .{});

        split_sizes_str = try std.fmt.allocPrint(allocator, "&{s}", .{split_array_name});
        needs_free = true;
    }
    defer if (needs_free) allocator.free(split_sizes_str);

    // Create a different approach that works with the expected types
    try writer.print(
        \\
        \\    // Create array for output tensor pointers to store final results
        \\    var output_ptrs_{s} = [_]*Tensor(T){{
    , .{try utils.getSanitizedName(node.outputs.items[0].name)});

    for (node.outputs.items, 0..) |output, i| {
        if (i > 0) try writer.writeAll(", ");
        try writer.print("&tensor_{s}", .{try utils.getSanitizedName(output.name)});
    }

    try writer.print(
        \\}};
        \\
        \\    // Create temporary tensors that split_lean can operate on
        \\    var temp_tensors_{0s} = allocator.alloc(Tensor(T), {1d}) catch @panic("Out of memory");
        \\    defer {{
        \\        for (temp_tensors_{0s}) |*t| t.deinit();
        \\        allocator.free(temp_tensors_{0s});
        \\    }}
        \\
        \\    // Initialize the temporary tensors
        \\    for (temp_tensors_{0s}) |*t| {{
        \\        t.* = Tensor(T).init(&allocator) catch @panic("Failed to initialize tensor");
        \\    }}
    , .{ try utils.getSanitizedName(node.outputs.items[0].name), node.outputs.items.len });

    // Convert split sizes to usize if provided
    if (!std.mem.eql(u8, split_sizes_str, "null")) {
        try writer.print(
            \\
            \\    // Call split_lean with the extracted split sizes
            \\    tensMath.split_lean(T, {2s}, {3d}, {1s}, &temp_tensors_{0s}) catch unreachable;
        , .{ try utils.getSanitizedName(node.outputs.items[0].name), split_sizes_str, input_tensor_string, axis });
    } else {
        // Get the proper axis value string
        const axis_str = if (axis < 0)
            try std.fmt.allocPrint(allocator, "@intCast((@as(i64, @intCast({s}.shape.len)) + {d}) %% @as(i64, @intCast({s}.shape.len)))", .{ input_tensor_string, axis, input_tensor_string })
        else
            try std.fmt.allocPrint(allocator, "{d}", .{axis});
        defer allocator.free(axis_str);

        try writer.print(
            \\
            \\    // Create default split size array for evenly dividing the tensor
            \\    const dim_size = {0s}.shape[{1s}];
            \\    const num_splits = {2d};
            \\    if (dim_size % num_splits != 0) @panic("Cannot evenly split dimension");
            \\    const split_size = dim_size / num_splits;
            \\    
            \\    const default_split_sizes_{4s} = allocator.alloc(usize, num_splits) catch @panic("Out of memory");
            \\    defer allocator.free(default_split_sizes_{4s});
            \\    for (default_split_sizes_{4s}) |*split_size_item| {{
            \\        split_size_item.* = split_size;
            \\    }}
            \\
            \\    // Call split_lean with default split sizes
            \\    tensMath.split_lean(T, {0s}, {3d}, default_split_sizes_{4s}, &temp_tensors_{4s}) catch unreachable;
        , .{ input_tensor_string, axis_str, node.outputs.items.len, axis, try utils.getSanitizedName(node.outputs.items[0].name) });
    }

    // Now copy the data from temp_tensors to the output tensors
    try writer.print(
        \\
        \\    // Copy data to existing output tensor arrays
        \\    for (temp_tensors_{0s}, 0..) |*src, i| {{
        \\        // Copy data directly to the existing array
        \\        const size_to_copy = @min(src.size, output_ptrs_{0s}[i].size);
        \\        if (size_to_copy > 0) {{
        \\            @memcpy(output_ptrs_{0s}[i].data[0..size_to_copy], src.data[0..size_to_copy]);
        \\        }}
        \\
        \\        // Update the shape if needed
        \\        if (!output_ptrs_{0s}[i].owns_memory) {{
        \\            // Shape is pre-allocated statically, just update if needed
        \\            const shape_size_to_copy = @min(src.shape.len, output_ptrs_{0s}[i].shape.len);
        \\            if (shape_size_to_copy > 0) {{
        \\                @memcpy(output_ptrs_{0s}[i].shape[0..shape_size_to_copy], src.shape[0..shape_size_to_copy]);
        \\            }}
        \\        }} else {{
        \\            // This is a dynamically allocated shape, replace it
        \\            if (output_ptrs_{0s}[i].shape.len > 0) {{
        \\                allocator.free(output_ptrs_{0s}[i].shape);
        \\            }}
        \\            output_ptrs_{0s}[i].shape = allocator.dupe(usize, src.shape) catch @panic("Out of memory");
        \\        }}
        \\
        \\        // Update the size
        \\        output_ptrs_{0s}[i].size = src.size;
        \\    }}
    , .{try utils.getSanitizedName(node.outputs.items[0].name)});

    // End with a function that returns an error union
    try writer.writeAll(
        \\
        \\    // Final dummy operation that returns an error union
        \\    _ = @import("std").fmt.bufPrint(&[_]u8{}, "", .{})
    );
}

inline fn write_resize(writer: std.fs.File.Writer, node: *ReadyNode) !void {
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

    //----create tensor_X_string
    var tensor_X_string: []u8 = undefined;
    defer allocator.free(tensor_X_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    // ---- optional inputs
    var tensor_roi_string: []const u8 = try allocator.dupe(u8, "null");
    defer {
        if (node.inputs.items.len >= 2 and node.inputs.items[1] != null) {
            allocator.free(tensor_roi_string);
        }
    }
    var data_scales_string: []const u8 = try allocator.dupe(u8, "null");
    defer {
        if (node.inputs.items.len >= 3 and node.inputs.items[2] != null) {
            allocator.free(data_scales_string);
        }
    }
    var data_sizes_string: []const u8 = try allocator.dupe(u8, "null");
    defer {
        if (node.inputs.items.len >= 4 and node.inputs.items[3] != null) {
            allocator.free(data_sizes_string);
        }
    }

    if (node.inputs.items.len >= 2 and node.inputs.items[1] != null) { //----create tensor_roi_string
        if (node.inputs.items[1].?.tag == globals.TensorTag.INITIALIZER) {
            tensor_roi_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(node.inputs.items[1].?.name),
                ")",
            });
        } else {
            tensor_roi_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[1].?.name) });
        }
    }

    if (node.inputs.items.len >= 3 and node.inputs.items[2] != null) { //----create tensor_scales_string
        if (node.inputs.items[2].?.tag == globals.TensorTag.INITIALIZER) {
            data_scales_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "param_lib.tensor_",
                try utils.getSanitizedName(node.inputs.items[2].?.name),
                ".data",
            });
        } else {
            data_scales_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "tensor_", try utils.getSanitizedName(node.inputs.items[2].?.name), ".data" });
        }
    }

    if (node.inputs.items.len >= 4 and node.inputs.items[3] != null) { //----create tensor_sizes_string
        if (node.inputs.items[3].?.tag == globals.TensorTag.INITIALIZER) {
            data_sizes_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "param_lib.tensor_",
                try utils.getSanitizedName(node.inputs.items[3].?.name),
                ".data",
            });
        } else {
            data_sizes_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "tensor_", try utils.getSanitizedName(node.inputs.items[3].?.name), ".data" });
        }
    }

    // ---- gasthering ATTRIBUTES from protoNode
    var antialias: i64 = 0;
    var axes: []i64 = &[_]i64{};
    defer allocator.free(axes);
    var coordinate_transformation_mode: []const u8 = try allocator.dupe(u8, "half_pixel");

    var cubic_coeff_a: f64 = -0.75;
    var exclude_outside: i64 = 0;
    var extrapolation_value: f64 = 0.0;
    var keep_aspect_ratio_policy: []const u8 = try allocator.dupe(u8, "stretch");
    defer allocator.free(keep_aspect_ratio_policy);
    var mode: []const u8 = try allocator.dupe(u8, "nearest");

    var nearest_mode: []const u8 = try allocator.dupe(u8, "round_prefer_floor");
    defer allocator.free(nearest_mode);

    for (node.nodeProto.attribute) |attr| {
        if (std.mem.indexOf(u8, attr.name, "antialias")) |_| {
            if (attr.type == AttributeType.INT) antialias = attr.i else return error.ResizeAnitialiasNotINT;
        } else if (std.mem.indexOf(u8, attr.name, "axes")) |_| {
            if (attr.type == AttributeType.INTS) axes = attr.ints else return error.ResizeAxesNotINTS;
        } else if (std.mem.indexOf(u8, attr.name, "coordinate_transformation_mode")) |_| {
            if (attr.type == AttributeType.STRING) coordinate_transformation_mode = attr.s else return error.Resize_coordinate_transformation_mode_NotSTRING;
        } else if (std.mem.indexOf(u8, attr.name, "cubic_coeff_a")) |_| {
            if (attr.type == AttributeType.FLOAT) cubic_coeff_a = attr.f else return error.Resize_cubic_coeff_a_NotFLOAT;
        } else if (std.mem.indexOf(u8, attr.name, "exclude_outside")) |_| {
            if (attr.type == AttributeType.INT) exclude_outside = attr.i else return error.Resize_exclude_outside_NotINT;
        } else if (std.mem.indexOf(u8, attr.name, "extrapolation_value")) |_| {
            if (attr.type == AttributeType.FLOAT) extrapolation_value = attr.f else return error.Resize_extrapolation_value_NotFLOAT;
        } else if (std.mem.indexOf(u8, attr.name, "keep_aspect_ratio_policy")) |_| {
            if (attr.type == AttributeType.STRING) keep_aspect_ratio_policy = attr.s else return error.Resize_keep_aspect_ratio_policy_NotSTRING;
        } else if (std.mem.indexOf(u8, attr.name, "mode")) |_| {
            if (attr.type == AttributeType.STRING) mode = attr.s else return error.Resize_mode_NotSTRING;
        } else if (std.mem.indexOf(u8, attr.name, "nearest_mode")) |_| {
            if (attr.type == AttributeType.STRING) nearest_mode = attr.s else return error.Resize_nearest_mode_NotSTRING;
        }
    }

    // ---- CREATING ATTRIBUTES strings
    const axes_string = try utils.i64SliceToUsizeArrayString(axes);
    _ = axes_string;

    //pub fn rezise_lean(comptime T: type, t: *Tensor(T), comptime mode: []const u8, scales: ?[]const f32, sizes: ?[]const usize, coordinate_transformation_mode: []const u8, output_tensor: *Tensor(T)) !void {
    _ = try writer.print(
        \\
        \\    tensMath.resize_lean(
        \\      T, 
        \\      {s}, //*Tensor(T)
        \\      "{s}", //mode
        \\      {s}, //scales: ?[]const f32
        \\      {s}, //sizes: ?[]const usize
        \\      "{s}", //coordinate_transformation_mode: []const u8
        \\      &tensor_{s}, //output_tensor: *Tensor(T)
        \\    )
    ,
        .{
            tensor_X_string, // input
            mode,
            data_scales_string,
            data_sizes_string,
            coordinate_transformation_mode,
            try utils.getSanitizedName(node.outputs.items[0].name), //output
        },
    );
}

inline fn write_neg(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    // https://onnx.ai/onnx/operators/onnx__Neg.html
    // INPUTS:
    //      - X (heterogeneous) - T: Input tensor
    // OUTPUTS:
    //      - Y (heterogeneous) - T: Output tensor with flipped elements

    // Create input tensor string
    var input_tensor_string: []u8 = undefined;
    defer allocator.free(input_tensor_string);

    if (node.inputs.items[0].?.tag == globals.TensorTag.INITIALIZER) {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].?.name),
            ")",
        });
    } else {
        input_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].?.name) });
    }

    _ = try writer.print(
        \\
        \\
        \\    tensMath.neg_lean(T, {s}, &tensor_{s})
    , .{
        input_tensor_string,
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
}
