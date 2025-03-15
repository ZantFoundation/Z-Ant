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

// ----------------------------------- SHAPE inference -----------------------------------

pub fn compute_output_shape(readyNode: *ReadyNode) !void {
    if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Add")) {
        //https://onnx.ai/onnx/operators/onnx__Add.html
        readyNode.outputs.items[0].shape = readyNode.inputs.items[0].shape;
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Concat")) {
        //https://onnx.ai/onnx/operators/onnx__Concat.html
        try compute_concat_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Constant")) {
        //https://onnx.ai/onnx/operators/onnx__Constant.html
        try compute_constant_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Conv")) {
        //https://onnx.ai/onnx/operators/onnx__Conv.html
        try compute_conv_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Div")) {
        //https://onnx.ai/onnx/operators/onnx__Div.html
        readyNode.outputs.items[0].shape = readyNode.inputs.items[1].shape;
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Flatten")) {
        // TODO
        return error.OperationWIP;
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Gather")) {
        try compute_gather_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Gemm")) {
        //https://onnx.ai/onnx/operators/onnx__Gemm.html
        try compute_gemm_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "LeakyRelu")) {
        // TODO
        return error.OperationWIP;
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "LogSoftmax")) {
        // TODO
        return error.OperationWIP;
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "MatMul")) {
        try compute_mul_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "MaxPool")) {
        try compute_maxPool_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Mul")) {
        //https://onnx.ai/onnx/operators/onnx__Mul.html
        try compute_mul_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "OneHot")) {
        // TODO
        return error.OperationWIP;
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "ReduceMean")) {
        try compute_reduceMean_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Relu")) {
        //https://onnx.ai/onnx/operators/onnx__Relu.html
        try compute_ReLU_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Reshape")) {
        // https://onnx.ai/onnx/operators/onnx__Reshape.html
        try compute_reshape_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Resize")) {
        // TODO
        return error.OperationWIP;
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Shape")) {
        try compute_shape_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Sigmoid")) {
        try compute_sigmoid_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Softmax")) {
        //https://onnx.ai/onnx/operators/onnx__Softmax.html
        try compute_softmax_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Slice")) {
        try compute_slice_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Split")) {
        // TODO
        return error.OperationWIP;
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Sub")) {
        // TODO
        return error.OperationWIP;
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Transpose")) {
        try compute_transpose_output_shape(readyNode);
    } else if (std.mem.eql(u8, readyNode.nodeProto.op_type, "Unsqueeze")) {
        try compute_unsqueeze_output_shape(readyNode);
    } else {
        std.debug.print("\n\n ERROR! output shape computation for {s} is not available in codeGen_math_handler.compute_output_shape() \n\n", .{readyNode.nodeProto.op_type});
        return error.OperationNotSupported;
    }
}

// ---------------- SHAPE COMPUTATION METHODS ----------------

inline fn compute_constant_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_constant_output_shape node: {s}======", .{readyNode.nodeProto.name.?});

    // Check each possible attribute type for the Constant node
    for (readyNode.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "value")) {
            // Handle tensor value - use existing utility
            const shape = try utils.getConstantTensorDims(readyNode.nodeProto);

            // If the shape is empty (scalar in ONNX), use [1] instead
            if (shape.len == 0) {
                readyNode.outputs.items[0].shape = try allocator.dupe(i64, &[_]i64{1});
            } else {
                readyNode.outputs.items[0].shape = shape;
            }

            std.debug.print("\n output_shape from tensor: []i64 = {any}", .{readyNode.outputs.items[0].shape});
            return;
        } else if (std.mem.eql(u8, attr.name, "value_float") or std.mem.eql(u8, attr.name, "value_int") or
            std.mem.eql(u8, attr.name, "value_string"))
        {
            // These are scalar values - output shape is [1]
            readyNode.outputs.items[0].shape = try allocator.dupe(i64, &[_]i64{1});
            std.debug.print("\n output_shape scalar: []i64 = {any}", .{readyNode.outputs.items[0].shape});
            return;
        } else if (std.mem.eql(u8, attr.name, "value_floats") or std.mem.eql(u8, attr.name, "value_ints")) {
            // These are 1D arrays - shape is [length]
            var length: i64 = 0;
            if (attr.type == AttributeType.FLOATS) {
                length = @intCast(attr.floats.len);
            } else if (attr.type == AttributeType.INTS) {
                length = @intCast(attr.ints.len);
            }
            readyNode.outputs.items[0].shape = try allocator.dupe(i64, &[_]i64{length});
            std.debug.print("\n output_shape 1D array: []i64 = {any}", .{readyNode.outputs.items[0].shape});
            return;
        } else if (std.mem.eql(u8, attr.name, "value_strings")) {
            // 1D array of strings - shape is [length]
            const length: i64 = @intCast(attr.strings.len);
            readyNode.outputs.items[0].shape = try allocator.dupe(i64, &[_]i64{length});
            std.debug.print("\n output_shape string array: []i64 = {any}", .{readyNode.outputs.items[0].shape});
            return;
        } else if (std.mem.eql(u8, attr.name, "sparse_value")) {
            // For sparse tensor, we need to handle it differently
            std.debug.print("\n Warning: Sparse tensor support is limited", .{});

            // Use a placeholder shape for sparse tensors - assuming scalar for now
            readyNode.outputs.items[0].shape = try allocator.dupe(i64, &[_]i64{1});
            std.debug.print("\n output_shape from sparse tensor (placeholder): []i64 = {any}", .{readyNode.outputs.items[0].shape});
            return;
        }
    }

    return error.ConstantValueNotFound;
}

inline fn compute_ReLU_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_ReLU_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    std.debug.print("\n input_shape: []i64 = {any}", .{readyNode.inputs.items[0].shape});
    readyNode.outputs.items[0].shape = readyNode.inputs.items[0].shape;
}

inline fn compute_reshape_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_reshape_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    std.debug.print("\n input_shape: []i64 = {any}", .{readyNode.inputs.items[0].shape});

    // Check if the second input has a tensorProto with int64_data
    if (readyNode.inputs.items[1].tensorProto != null and
        readyNode.inputs.items[1].tensorProto.?.int64_data != null)
    {
        std.debug.print("\n new shape from tensorProto: []i64 = {any}", .{readyNode.inputs.items[1].tensorProto.?.int64_data.?});
        readyNode.outputs.items[0].shape = readyNode.inputs.items[1].tensorProto.?.int64_data.?;
    } else {
        // If not, use the shape of the second input directly
        std.debug.print("\n new shape from input shape: []i64 = {any}", .{readyNode.inputs.items[1].shape});
        readyNode.outputs.items[0].shape = try allocator.dupe(i64, readyNode.inputs.items[1].shape);
    }

    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_softmax_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_softmax_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    std.debug.print("\n input_shape: []i64 = {any}", .{readyNode.inputs.items[0].shape});
    readyNode.outputs.items[0].shape = readyNode.inputs.items[0].shape;
}

inline fn compute_gemm_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_gemm_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    std.debug.print("\n input_shape: []i64 = {any}", .{readyNode.inputs.items[0].shape});
    std.debug.print("\n weight_shape: []i64 = {any}", .{readyNode.inputs.items[1].shape});
    std.debug.print("\n bias_shape: []i64 = {any}", .{readyNode.inputs.items[2].shape});
    readyNode.outputs.items[0].shape = readyNode.inputs.items[2].shape;
}

fn compute_mul_output_shape(readyNode: *ReadyNode) !void {
    const input_a = readyNode.inputs.items[0];
    const input_b = readyNode.inputs.items[1];

    std.debug.print("\n====== compute_mul_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    std.debug.print("\n input_a_shape: []i64 = {any}", .{input_a.shape});
    std.debug.print("\n input_b_shape: []i64 = {any}", .{input_b.shape});

    if (input_a.shape.len < 2 or input_b.shape.len < 2) {
        return error.InvalidShape;
    }

    const a_rows = input_a.shape[input_a.shape.len - 2];
    const a_cols = input_a.shape[input_a.shape.len - 1];
    const b_rows = input_b.shape[input_b.shape.len - 2];
    const b_cols = input_b.shape[input_b.shape.len - 1];

    if (a_cols != b_rows) {
        return error.ShapeMismatch;
    }

    var output_shape: std.ArrayList(i64) = std.ArrayList(i64).init(allocator);
    defer output_shape.deinit();

    // Broadcast the batch dimensions
    const a_batch = input_a.shape.len - 2;
    const b_batch = input_b.shape.len - 2;
    const batch_dims = if (a_batch < b_batch) a_batch else b_batch;

    for (0..batch_dims) |i| {
        try output_shape.append(if (input_a.shape[i] > input_b.shape[i]) input_a.shape[i] else input_b.shape[i]);
    }

    // Append output matrix dimensions
    try output_shape.append(a_rows);
    try output_shape.append(b_cols);

    // Update the shape of the output tensor
    readyNode.outputs.items[0].shape = try output_shape.toOwnedSlice();
}

inline fn compute_conv_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_conv_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape: []const i64 = readyNode.inputs.items[0].shape;
    const kernel_shape: []const i64 = readyNode.inputs.items[1].shape;

    var stride: ?[]i64 = null;
    var dilation: ?[]i64 = null;
    var auto_pad: []const u8 = "NOTSET";
    var pads: ?[]i64 = null;
    for (readyNode.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "strides")) {
            if (attr.type == AttributeType.INTS) stride = attr.ints;
        } else if (std.mem.eql(u8, attr.name, "dilations")) {
            if (attr.type == AttributeType.INTS) dilation = attr.ints;
        } else if (std.mem.eql(u8, attr.name, "auto_pad")) {
            if (attr.type == AttributeType.STRING) auto_pad = attr.s;
        }
        if (std.mem.eql(u8, attr.name, "pads")) {
            if (attr.type == AttributeType.INTS) pads = attr.ints;
        }
    }

    if (stride == null) return error.StridesNotFound;
    if (dilation == null) return error.DilationsNotFound;

    std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});
    std.debug.print("\n kernel_shape: []i64 = {any}", .{kernel_shape});
    std.debug.print("\n stride: []i64 = {any}", .{stride.?});
    //std.debug.print("\n pads: []i64 = {any}", .{pads.?});
    readyNode.outputs.items[0].shape = try utils.usizeSliceToI64Slice(
        @constCast(
            &try tensorMath.get_convolution_output_shape(
                try utils.i64SliceToUsizeSlice(input_shape),
                try utils.i64SliceToUsizeSlice(kernel_shape),
                try utils.i64SliceToUsizeSlice(stride.?),
                if (pads != null) try utils.i64SliceToUsizeSlice(pads.?) else null,
                try utils.i64SliceToUsizeSlice(dilation.?),
                auto_pad,
            ),
        ),
    );
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_maxPool_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_maxPool_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape: []const i64 = readyNode.inputs.items[0].shape;

    var kernel_shape: ?[]i64 = null;
    var stride: ?[]i64 = null;

    for (readyNode.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "kernel_shape")) {
            if (attr.type == AttributeType.INTS) kernel_shape = attr.ints;
        } else if (std.mem.eql(u8, attr.name, "strides")) {
            if (attr.type == AttributeType.INTS) stride = attr.ints;
        }
    }

    if (kernel_shape == null) return error.KernelShapeNotFound;
    if (stride == null) return error.StridesNotFound;

    std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});
    std.debug.print("\n kernel_shape: []i64 = {any}", .{kernel_shape.?});
    std.debug.print("\n stride: []i64 = {any}", .{stride.?});

    const kernel_2d = [2]usize{ @intCast(kernel_shape.?[0]), @intCast(kernel_shape.?[1]) };
    const stride_2d = [2]usize{ @intCast(stride.?[0]), @intCast(stride.?[1]) };

    readyNode.outputs.items[0].shape = try utils.usizeSliceToI64Slice(
        @constCast(
            &try tensorMath.get_pooling_output_shape(
                try utils.i64SliceToUsizeSlice(input_shape),
                kernel_2d,
                stride_2d,
            ),
        ),
    );
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_reduceMean_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_reduceMean_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape: []const i64 = readyNode.inputs.items[0].shape;

    var keepdims: bool = true;
    for (readyNode.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "keepdims")) {
            if (attr.type == AttributeType.INT) keepdims = attr.i != 0;
        }
    }

    // If we have axes input tensor
    if (readyNode.inputs.items.len > 1) {
        const axes = readyNode.inputs.items[1].shape;
        std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});
        std.debug.print("\n axes: []i64 = {any}", .{axes});

        // If keepdims is true, the reduced dimensions are retained with length 1
        if (keepdims) {
            var newShape = try allocator.alloc(i64, input_shape.len);
            @memcpy(newShape, input_shape);
            for (axes) |axis| {
                newShape[@as(usize, @intCast(axis))] = 1;
            }
            readyNode.outputs.items[0].shape = newShape;
        } else {
            // If keepdims is false, the reduced dimensions are removed
            var newShape = try allocator.alloc(i64, input_shape.len - axes.len);
            var j: usize = 0;
            outer: for (input_shape, 0..) |dim, i| {
                for (axes) |axis| {
                    if (i == @as(usize, @intCast(axis))) continue :outer;
                }
                newShape[j] = dim;
                j += 1;
            }
            readyNode.outputs.items[0].shape = newShape;
        }
    } else {
        // If no axes specified, reduce all dimensions to 1 if keepdims is true, or to a scalar if false
        if (keepdims) {
            var newShape = try allocator.alloc(i64, input_shape.len);
            for (newShape, 0..) |_, i| {
                newShape[i] = 1;
            }
            readyNode.outputs.items[0].shape = newShape;
        } else {
            readyNode.outputs.items[0].shape = &[_]i64{1};
        }
    }
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_slice_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_slice_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape = readyNode.inputs.items[0].shape;
    const starts = readyNode.inputs.items[1].tensorProto.?.int64_data.?;
    const ends = readyNode.inputs.items[2].tensorProto.?.int64_data.?;

    var axes: ?[]i64 = null;
    var steps: ?[]i64 = null;

    // Get axes if provided (input 3)
    if (readyNode.inputs.items.len > 3) {
        axes = readyNode.inputs.items[3].tensorProto.?.int64_data.?;
    }

    // Get steps if provided (input 4)
    if (readyNode.inputs.items.len > 4) {
        steps = readyNode.inputs.items[4].tensorProto.?.int64_data.?;
    }

    std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});
    std.debug.print("\n starts: []i64 = {any}", .{starts});
    std.debug.print("\n ends: []i64 = {any}", .{ends});
    std.debug.print("\n axes: []i64 = {any}", .{axes});
    std.debug.print("\n steps: []i64 = {any}", .{steps});

    const output_shape = try tensorMath.get_slice_output_shape(
        try utils.i64SliceToUsizeSlice(input_shape),
        starts,
        ends,
        axes,
        steps,
    );

    readyNode.outputs.items[0].shape = try utils.usizeSliceToI64Slice(output_shape);
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_shape_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_shape_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape = readyNode.inputs.items[0].shape;

    // Get start and end attributes if they exist
    var start: ?i64 = null;
    var end: ?i64 = null;

    for (readyNode.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "start")) {
            if (attr.type == AttributeType.INT) start = attr.i;
        } else if (std.mem.eql(u8, attr.name, "end")) {
            if (attr.type == AttributeType.INT) end = attr.i;
        }
    }

    // Calculate output size
    const rank = input_shape.len;
    var start_axis: i64 = start orelse 0;
    if (start_axis < 0) start_axis += @as(i64, @intCast(rank));
    start_axis = @max(0, @min(start_axis, @as(i64, @intCast(rank - 1))));

    var end_axis: i64 = end orelse @as(i64, @intCast(rank));
    if (end_axis < 0) end_axis += @as(i64, @intCast(rank));
    end_axis = @max(start_axis, @min(end_axis, @as(i64, @intCast(rank))));

    const output_size = @max(0, end_axis - start_axis);

    // Shape operator always outputs a 1D tensor
    readyNode.outputs.items[0].shape = try allocator.dupe(i64, &[_]i64{@intCast(output_size)});
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_gather_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_gather_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const data_shape = readyNode.inputs.items[0].shape;
    const indices_shape = readyNode.inputs.items[1].shape;

    // Get axis attribute, default is 0
    var axis: i64 = 0;
    for (readyNode.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "axis")) {
            if (attr.type == AttributeType.INT) axis = attr.i;
        }
    }

    // Handle negative axis
    if (axis < 0) {
        axis += @as(i64, @intCast(data_shape.len));
    }

    std.debug.print("\n data_shape: []i64 = {any}", .{data_shape});
    std.debug.print("\n indices_shape: []i64 = {any}", .{indices_shape});
    std.debug.print("\n axis: {}", .{axis});

    // Calculate output shape:
    // - Take all dimensions from data before axis
    // - Add all dimensions from indices
    // - Take all dimensions from data after axis
    const output_rank = data_shape.len + indices_shape.len - 1;
    var output_shape = try allocator.alloc(i64, output_rank);

    // Copy dimensions from data before axis
    var out_idx: usize = 0;
    for (0..@as(usize, @intCast(axis))) |i| {
        output_shape[out_idx] = data_shape[i];
        out_idx += 1;
    }

    // Copy dimensions from indices
    for (indices_shape) |dim| {
        output_shape[out_idx] = dim;
        out_idx += 1;
    }

    // Copy remaining dimensions from data after axis
    for (@as(usize, @intCast(axis + 1))..data_shape.len) |i| {
        output_shape[out_idx] = data_shape[i];
        out_idx += 1;
    }

    readyNode.outputs.items[0].shape = output_shape;
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_sigmoid_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_sigmoid_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape = readyNode.inputs.items[0].shape;
    std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});

    // Sigmoid is element-wise, output shape is identical to input shape
    readyNode.outputs.items[0].shape = try allocator.dupe(i64, input_shape);
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_transpose_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_transpose_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape = readyNode.inputs.items[0].shape;

    // Get the perm attribute if it exists
    var perm: ?[]i64 = null;
    for (readyNode.nodeProto.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "perm")) {
            if (attr.type == AttributeType.INTS) {
                perm = attr.ints;
            }
        }
    }

    std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});
    std.debug.print("\n perm: []i64 = {any}", .{perm});

    // If no perm is provided, reverse the dimensions
    var output_shape = try allocator.alloc(i64, input_shape.len);
    if (perm) |p| {
        // Validate perm length
        if (p.len != input_shape.len) {
            return error.InvalidPermutation;
        }
        // Apply permutation
        for (p, 0..) |axis, i| {
            if (axis < 0 or axis >= @as(i64, @intCast(input_shape.len))) {
                return error.InvalidPermutation;
            }
            output_shape[i] = input_shape[@intCast(axis)];
        }
    } else {
        // Reverse dimensions
        for (input_shape, 0..) |_, i| {
            output_shape[i] = input_shape[input_shape.len - 1 - i];
        }
    }

    readyNode.outputs.items[0].shape = output_shape;
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

inline fn compute_unsqueeze_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n====== compute_unsqueeze_output_shape node: {s}======", .{readyNode.nodeProto.name.?});
    const input_shape = readyNode.inputs.items[0].shape;
    std.debug.print("\n input_shape: []i64 = {any}", .{input_shape});

    // Get axes from attributes or from the second input tensor
    var axes: ?[]const i64 = null;

    // First check if axes is provided as an input tensor (ONNX opset 13+)
    if (readyNode.inputs.items.len > 1 and readyNode.inputs.items[1].tensorProto != null) {
        axes = readyNode.inputs.items[1].tensorProto.?.int64_data.?;
        std.debug.print("\n axes from input tensor: []i64 = {any}", .{axes.?});
    } else {
        // Otherwise, check for axes attribute (ONNX opset < 13)
        for (readyNode.nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "axes")) {
                if (attr.type == AttributeType.INTS) {
                    axes = attr.ints;
                    std.debug.print("\n axes from attribute: []i64 = {any}", .{axes.?});
                    break;
                }
            }
        }
    }

    if (axes == null) return error.UnsqueezeAxesNotFound;

    // Calculate output shape by inserting dimensions at the specified axes
    const output_rank = input_shape.len + axes.?.len;
    var output_shape = try allocator.alloc(i64, output_rank);

    // Initialize with 1s
    @memset(output_shape, 1);

    // Create a mask to track which positions are for the new dimensions
    var is_unsqueezed_axis = try allocator.alloc(bool, output_rank);
    defer allocator.free(is_unsqueezed_axis);
    @memset(is_unsqueezed_axis, false);

    // Mark the positions where dimensions will be inserted
    for (axes.?) |axis| {
        var normalized_axis = axis;
        if (normalized_axis < 0) normalized_axis += @as(i64, @intCast(output_rank));

        if (normalized_axis < 0 or normalized_axis >= @as(i64, @intCast(output_rank))) {
            return error.UnsqueezeInvalidAxis;
        }

        is_unsqueezed_axis[@intCast(normalized_axis)] = true;
    }

    // Fill in the output shape with the input dimensions
    var input_idx: usize = 0;
    for (0..output_rank) |output_idx| {
        if (!is_unsqueezed_axis[output_idx]) {
            output_shape[output_idx] = input_shape[input_idx];
            input_idx += 1;
        }
    }

    readyNode.outputs.items[0].shape = output_shape;
    std.debug.print("\n output_shape: []i64 = {any}", .{readyNode.outputs.items[0].shape});
}

pub fn compute_concat_output_shape(readyNode: *ReadyNode) !void {
    std.debug.print("\n compute_concat_output_shape for node: {s}", .{readyNode.nodeProto.name.?});

    // Get the axis attribute (required)
    var axis: i64 = 0;
    var axis_found = false;

    for (readyNode.nodeProto.attribute) |attr| {
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

    std.debug.print("\n   axis: {}", .{axis});

    // Ensure there's at least one input tensor
    if (readyNode.inputs.items.len == 0) {
        return error.ConcatNoInputs;
    }

    // Special case for concatenation along axis 0 with different ranks
    if (axis == 0) {
        // Find the maximum rank among all input tensors
        var max_rank: usize = 0;
        for (readyNode.inputs.items) |input| {
            max_rank = @max(max_rank, input.shape.len);
        }

        // Calculate the output shape for axis 0 concatenation
        var total_dim0: i64 = 0;
        for (readyNode.inputs.items) |input| {
            if (input.shape.len == 0) {
                // Scalar tensor contributes 1 to the first dimension
                total_dim0 += 1;
            } else {
                // For tensors with at least one dimension, add their first dimension
                total_dim0 += input.shape[0];
            }
        }

        // Create the output shape
        var new_shape = try allocator.alloc(i64, max_rank);
        errdefer allocator.free(new_shape);

        // Set the first dimension to the sum
        new_shape[0] = total_dim0;

        // For the remaining dimensions, use the dimensions from the first tensor with the highest rank
        if (max_rank > 1) {
            // Find the first tensor with the maximum rank
            for (readyNode.inputs.items) |input| {
                if (input.shape.len == max_rank) {
                    // Copy the remaining dimensions from this tensor
                    for (1..max_rank) |d| {
                        new_shape[d] = input.shape[d];
                    }
                    break;
                }
            }
        }

        std.debug.print("\n   output shape (special case axis 0): ", .{});
        for (new_shape) |dim| {
            std.debug.print("{} ", .{dim});
        }

        // Set the output shape
        readyNode.outputs.items[0].shape = new_shape;
        return;
    }

    // Standard case: all tensors must have the same rank
    // Get the rank from the first input tensor
    const rank = readyNode.inputs.items[0].shape.len;

    // Normalize negative axis
    var normalized_axis = axis;
    if (normalized_axis < 0) {
        normalized_axis += @as(i64, @intCast(rank));
    }

    // Check if axis is valid
    if (normalized_axis < 0 or normalized_axis >= @as(i64, @intCast(rank))) {
        return error.ConcatAxisOutOfBounds;
    }

    const axis_usize = @as(usize, @intCast(normalized_axis));

    // Validate that all tensors have the same rank and matching shapes except along the concatenation axis
    for (readyNode.inputs.items) |input| {
        if (input.shape.len != rank) {
            return error.ConcatMismatchedRank;
        }

        for (0..rank) |d| {
            if (d != axis_usize and input.shape[d] != readyNode.inputs.items[0].shape[d]) {
                return error.ConcatMismatchedShape;
            }
        }
    }

    // Calculate the new shape after concatenation
    var new_shape = try allocator.alloc(i64, rank);
    errdefer allocator.free(new_shape);

    for (0..rank) |d| {
        if (d == axis_usize) {
            var sum: i64 = 0;
            for (readyNode.inputs.items) |input| {
                sum += input.shape[d];
            }
            new_shape[d] = sum;
        } else {
            new_shape[d] = readyNode.inputs.items[0].shape[d];
        }
    }

    std.debug.print("\n   output shape: ", .{});
    for (new_shape) |dim| {
        std.debug.print("{} ", .{dim});
    }

    // Set the output shape
    readyNode.outputs.items[0].shape = new_shape;
}
