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
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "LeakyRelu")) {
        try writer.writeAll("// Handle LeakyRelu\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "LogSoftmax")) {
        try writer.writeAll("// Handle LogSoftmax\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "MatMul")) {
        try write_matmul(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "MaxPool")) {
        try write_maxPool(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Mul")) {
        try write_mul(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "OneHot")) {
        try writer.writeAll("// Handle OneHot\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "ReduceMean")) {
        try write_reduceMean(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Relu")) {
        try write_ReLU(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Reshape")) {
        try write_reshape(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Resize")) {
        try writer.writeAll("// Handle Resize\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Sigmoid")) {
        try write_sigmoid(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Softmax")) {
        try write_softmax(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Slice")) {
        try write_slice(writer, node);
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Split")) {
        try writer.writeAll("// Handle Split\n");
    } else if (std.mem.eql(u8, node.nodeProto.op_type, "Sub")) {
        try writer.writeAll("// Handle Sub\n");
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
    if (node.inputs.items[0].tag == globals.TensorTag.INITIALIZER) {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[0].name),
            ")",
        });
    } else {
        tensor_A_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[0].name) });
    }

    //----create tensor_B_string
    var tensor_B_string: []u8 = undefined;
    if (node.inputs.items[1].tag == globals.TensorTag.INITIALIZER) {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&param_lib.tensor_",
            try utils.getSanitizedName(node.inputs.items[1].name),
            ")",
        });
    } else {
        tensor_B_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(node.inputs.items[1].name) });
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

    //----create ?bias string
    var bias_string: []u8 = undefined;
    // Bias Tensor B is optional! verify the presence
    if (node.inputs.items.len == 3) {
        const B_name = try utils.getSanitizedName(node.inputs.items[2].name);
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
        \\        &tensor_{s}, //input
        \\        @constCast(&param_lib.tensor_{s}), //kernel
        \\        &tensor_{s}, //output
        \\        {s}, //bias
        \\        {s}, //stride
        \\        {s}, //pads
        \\        {s}, //dilatations
        \\        {}, //group
        \\        "{s}", //auto_pad
        \\    )
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name), //Input
        try utils.getSanitizedName(node.inputs.items[1].name), //Kernel
        try utils.getSanitizedName(node.outputs.items[0].name), //Output
        bias_string, //Bias
        stride_string, //Strides
        pads_string, //Pads
        dilat_string, //Dilatations
        group, //Group
        auto_pad,
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
        const first_rank = node.inputs.items[0].shape.len;

        for (node.inputs.items[1..]) |input| {
            if (input.shape.len != first_rank) {
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
                \\    var concat_tensor_list = [_]Tensor(T){{
            , .{});

            for (node.inputs.items, 0..) |input, idx| {
                if (idx > 0) {
                    _ = try writer.print(", ", .{});
                }
                _ = try writer.print("tensor_{s}", .{try utils.getSanitizedName(input.name)});
            }

            _ = try writer.print(
                \\}};
                \\
                \\    // Perform concatenation with special handling for different ranks
                \\     try tensMath.concatenate_lean(T, &allocator, &concat_tensor_list, {},tensor_{s})
            , .{
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
        \\    var concat_tensor_list = [_]Tensor(T){{
    , .{});

    for (node.inputs.items, 0..) |input, idx| {
        if (idx > 0) {
            _ = try writer.print(", ", .{});
        }
        _ = try writer.print("tensor_{s}", .{try utils.getSanitizedName(input.name)});
    }

    _ = try writer.print(
        \\}};
        \\
        \\    // Perform concatenation
        \\    tensMath.concatenate_lean(T, &allocator, &concat_tensor_list, {}, &tensor_{s} )
    , .{
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
        if (std.mem.eql(u8, attr.name, "value")) {
            // For TENSOR value, the tensor is already initialized during model loading
            // No additional code needed for initialization
            try writer.print(
                \\
                \\    // Constant tensor was already initialized during model loading
                \\    // No additional code needed for tensor_{s}
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

    _ = try writer.print(
        \\
        \\    tensMath.div_lean(T, &tensor_{s}, &param_lib.tensor_{s}, &tensor_{s})
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name), // Input tensor A
        try utils.getSanitizedName(node.inputs.items[1].name), // Input tensor B
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor C
    });
}

//TODO : add param_lib. where necessary
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

    const indices_name = try utils.getSanitizedName(node.inputs.items[1].name);

    _ = try writer.print(
        \\    
        \\
        \\    //creating the indices Tensor(usize)
        \\    
        \\    const usize_slice_{s} =  utils.sliceToUsizeSlice(tensor_{s}.data);
        \\    var usize_tensor_{s} = Tensor(usize).fromConstBuffer(&allocator, usize_slice_{s}, tensor_{s}.shape);
        \\    defer usize_tensor_{s}.deinit();
        \\    defer allocator.free(usize_slice_{s});
        \\    
    , .{
        indices_name, //usize_slice_
        indices_name, //tensor_
        indices_name, //usize_tensor_
        indices_name, //usize_slice_
        indices_name, //tensor_.shape
        indices_name, //usize_tensor_.deinit
        indices_name, //usize_slice_ for free
    });

    _ = try writer.print(
        \\
        \\
        \\    tensMath.gather_lean(
        \\        T, //type
        \\        @constCast(&tensor_{s}), //data tensor
        \\        &usize_tensor_{s}, //indices tensor
        \\        {}, //axis
        \\        &tensor_{s}, //output tensor
        \\    )
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name), // Input data tensor
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

    // --- generating the tensors name depending if they are initializers or not:
    var b_tensor_string: []u8 = undefined;
    const sanitized_tensor_B = try utils.getSanitizedName(node.inputs.items[1].name);
    b_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
        if (globals.tensorHashMap.getPtr(node.inputs.items[1].name).?.tag == globals.TensorTag.INITIALIZER) "param_lib." else "",
        "tensor_",
        sanitized_tensor_B,
    });

    // Input Tensor C is optional! verify the presence
    var c_tensor_string: []u8 = undefined;
    if (node.inputs.items.len == 3) {
        const sanitized_tensor_C = try utils.getSanitizedName(node.inputs.items[2].name);
        c_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            "@constCast(&",
            if (globals.tensorHashMap.getPtr(node.inputs.items[2].name).?.tag == globals.TensorTag.INITIALIZER) "param_lib." else "",
            "tensor_",
            sanitized_tensor_C,
            ")",
        });
    } else {
        c_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{" null"});
    }

    _ = try writer.print(
        \\
        \\
        \\    tensMath.gemm_lean(T, &tensor_{s}, @constCast(&{s}), {s}, {}, {}, {s}, {s}, &tensor_{s} )
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name), // Input tensor A
        b_tensor_string, // Input tensor B
        c_tensor_string,
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

    //if B is a static parameter must be imported from parameter_lib.zig
    var b_tensor_string: []u8 = undefined;
    const sanitized_tensor_B = try utils.getSanitizedName(node.inputs.items[1].name);
    b_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
        if (globals.tensorHashMap.getPtr(node.inputs.items[1].name).?.tag == globals.TensorTag.INITIALIZER) "param_lib." else "",
        "tensor_",
        sanitized_tensor_B,
    });

    _ = try writer.print(
        \\
        \\    tensMath.mat_mul_lean(T, &tensor_{s}, @constCast(&{s}), &tensor_{s})
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name), // Input tensor A
        b_tensor_string, // Input tensor B
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

    // pub fn lean_onnx_maxpool(
    //     comptime T: type,
    //     input: *Tensor(T),
    //     output: *Tensor(T),
    //     kernel_shape: []const usize,
    //     strides: []const usize,
    //     dilations: []const usize,
    //     pads: []const usize,
    //     auto_pad: AutoPadType,
    // ) !void

    _ = try writer.print(
        \\
        \\
        \\    tensMath.onnx_maxpool_lean(
        \\        T,
        \\        &tensor_{s}, //Input
        \\        &tensor_{s}, //Output
        \\        {s}, //kernel_shape
        \\        {s}, //strides
        \\        {s}, //dilations
        \\        {s}, //pads
        \\        tensMath.AutoPadType.{s}, //auto_pad
        \\    )
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name), //Input
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

    //if B is a static parameter must be imported from parameter_lib.zig
    var b_tensor_string: []u8 = undefined;
    const sanitized_tensor_B = try utils.getSanitizedName(node.inputs.items[1].name);
    b_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
        if (globals.tensorHashMap.getPtr(node.inputs.items[1].name).?.tag == globals.TensorTag.INITIALIZER) "param_lib." else "",
        "tensor_",
        sanitized_tensor_B,
    });

    _ = try writer.print(
        \\
        \\
        \\    tensMath.mul_lean(T, &tensor_{s}, @constCast(&tensor_{s}), &tensor_{s})
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name), // Input tensor A
        b_tensor_string, // Input tensor B
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor C
    });
}

inline fn write_reduceMean(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    var keepdims: bool = true;
    var noop_with_empty_axes: bool = false;

    for (node.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "keepdims")) {
            if (attr.type == AttributeType.INT) keepdims = attr.i != 0;
        } else if (std.mem.eql(u8, attr.name, "noop_with_empty_axes")) {
            if (attr.type == AttributeType.INT) noop_with_empty_axes = attr.i != 0;
        }
    }

    var axes_str: []const u8 = "null";
    if (node.inputs.items.len > 1) {
        axes_str = try std.fmt.allocPrint(allocator, "&tensor_{s}.data", .{try utils.getSanitizedName(node.inputs.items[1].name)});
    }

    _ = try writer.print(
        \\
        \\
        \\    tensMath.reduce_mean_lean(T, &tensor_{s}, &tensor_{s}, {s}, {s}, {s})
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name),
        try utils.getSanitizedName(node.outputs.items[0].name),
        axes_str,
        if (keepdims) "true" else "false",
        if (noop_with_empty_axes) "true" else "false",
    });
}

inline fn write_ReLU(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    //node.inputs.items[0] -> input
    //node.outputs.items[0] -> output

    _ = try writer.print(
        \\
        \\
        \\    tensMath.ReLU_lean(T, &tensor_{s}, &tensor_{s})
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name),
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
    const sanitized_input_name = try utils.getSanitizedName(node.inputs.items[0].name);
    input_string = try std.mem.concat(allocator, u8, &[_][]const u8{
        if (globals.tensorHashMap.getPtr(node.inputs.items[0].name).?.tag == globals.TensorTag.INITIALIZER) "param_lib." else "",
        "tensor_",
        sanitized_input_name,
    });

    //shape string creation
    var input_shape: []u8 = undefined;
    const sanitized_shape_name = try utils.getSanitizedName(node.inputs.items[1].name);
    input_shape = try std.mem.concat(allocator, u8, &[_][]const u8{
        if (globals.tensorHashMap.getPtr(node.inputs.items[1].name).?.tag == globals.TensorTag.INITIALIZER) "param_lib." else "",
        "tensor_",
        sanitized_shape_name,
        ".data",
    });

    _ = try writer.print(
        \\
        \\
        \\    const newShape_tensor_{s}: []usize = utils.sliceToUsizeSlice({s});
        \\    defer allocator.free(newShape_tensor_{s});
    , .{
        try utils.getSanitizedName(node.inputs.items[1].name),
        input_shape,
        try utils.getSanitizedName(node.inputs.items[1].name),
    });

    _ = try writer.print(
        \\
        \\    tensMath.reshape_lean(
        \\        T, //type
        \\        @constCast(&{s}), //Input tensor
        \\        newShape_tensor_{s}, //New shape
        \\        {s}, //allowzero
        \\        &tensor_{s}, //Output tensor
        \\    )
    , .{
        input_string, // Input tensor
        try utils.getSanitizedName(node.inputs.items[1].name), // Input shape tensor
        if (allowzer0) "true" else "false", //allowzer0
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor
    });
}

inline fn write_sigmoid(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    //node.inputs.items[0] -> input
    //node.outputs.items[0] -> output

    _ = try writer.print(
        \\
        \\
        \\    tensMath.sigmoid_lean(T, &tensor_{s}, &tensor_{s})
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name),
        try utils.getSanitizedName(node.outputs.items[0].name),
    });
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
    const input_name = try utils.getSanitizedName(node.inputs.items[0].name);
    const starts_name = try utils.getSanitizedName(node.inputs.items[1].name);
    const ends_name = try utils.getSanitizedName(node.inputs.items[2].name);
    const output_name = try utils.getSanitizedName(node.outputs.items[0].name);

    // Handle optional axes and steps inputs
    var axes_str: []const u8 = "null";
    var steps_str: []const u8 = "null";

    if (node.inputs.items.len > 3) {
        const axes_name = try utils.getSanitizedName(node.inputs.items[3].name);
        axes_str = try std.fmt.allocPrint(allocator, "&tensor_{s}.data", .{axes_name});
    }

    if (node.inputs.items.len > 4) {
        const steps_name = try utils.getSanitizedName(node.inputs.items[4].name);
        steps_str = try std.fmt.allocPrint(allocator, "&tensor_{s}.data", .{steps_name});
    }

    _ = try writer.print(
        \\
        \\
        \\    tensMath.lean_slice_onnx(
        \\        T, //type
        \\        @constCast(&tensor_{s}), //input tensor
        \\        &tensor_{s}.data, //starts
        \\        &tensor_{s}.data, //ends
        \\        {s}, //axes
        \\        {s}, //steps
        \\        &tensor_{s}, //output tensor
        \\    )
    , .{
        input_name,
        starts_name,
        ends_name,
        axes_str,
        steps_str,
        output_name,
    });

    if (axes_str.len > 4) allocator.free(axes_str);
    if (steps_str.len > 4) allocator.free(steps_str);
}

inline fn write_softmax(writer: std.fs.File.Writer, node: *ReadyNode) !void {
    //node.inputs.items[0] -> input
    //node.outputs.items[0] -> output

    _ = try writer.print(
        \\
        \\
        \\    tensMath.softmax_lean(T, &tensor_{s}, &tensor_{s})
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name),
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
        \\    const my_tensor_list = [_]*Tensor(T){{
    , .{});

    for (node.inputs.items, 0..) |tens, idx| {
        if (idx > 0) {
            _ = try writer.print(", ", .{});
        }

        var new_tensor_string: []u8 = undefined;
        const sanitized_tensor_name = try utils.getSanitizedName(tens.name);

        new_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            if (globals.tensorHashMap.getPtr(tens.name).?.tag == globals.TensorTag.INITIALIZER) "param_lib." else "",
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

    _ = try writer.print(
        \\
        \\    tensMath.shape_onnx_lean(
        \\        T,
        \\        T, //type
        \\        @constCast(&tensor_{s}), //input tensor
        \\        {s}, //start
        \\        {s}, //end
        \\        &tensor_{s}, //output tensor,
        \\    )
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name),
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

    const input_name = try utils.getSanitizedName(node.inputs.items[0].name);
    const output_name = try utils.getSanitizedName(node.outputs.items[0].name);

    // Determine if axes is provided as an input tensor or as an attribute
    var axes_str: []const u8 = "null";
    var needs_free = false;

    if (node.inputs.items.len > 1) {
        // Axes is provided as an input tensor (opset 13+)
        const axes_tensor_name = try utils.getSanitizedName(node.inputs.items[1].name);
        axes_str = try std.fmt.allocPrint(allocator, "&tensor_{s}.data", .{axes_tensor_name});
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

    // Generate code to convert the input shape to the output shape
    try writer.print(
        \\     
        \\    var axes_shape_{s} = [_]usize{{1}};
        \\    var axes_tensor_{s} = Tensor(i64).fromArray(&allocator, {s}, &axes_shape_{s}) catch return;
        \\    defer allocator.free(axes_tensor_{s}.data);
        \\    defer allocator.free(axes_tensor_{s}.shape);
        \\    tensMath.unsqueeze_lean(
        \\        T, //type
        \\        @constCast(&tensor_{s}), //input tensor
        \\        &axes_tensor_{s}, //axes
        \\        &tensor_{s}, //output tensor
        \\    )
    , .{
        input_name,
        input_name,
        axes_str,
        input_name,
        input_name,
        input_name,
        input_name,
        input_name,
        output_name,
    });
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

    _ = try writer.print(
        \\
        \\
        \\    tensMath.transpose_onnx_lean(
        \\        T, //type
        \\        @constCast(&tensor_{s}), //input tensor
        \\        {s}, //perm
        \\        &tensor_{s}, //output tensor
        \\    )
    , .{
        try utils.getSanitizedName(node.inputs.items[0].name), // Input tensor
        perm_str, // Permutation array
        try utils.getSanitizedName(node.outputs.items[0].name), // Output tensor
    });
}
