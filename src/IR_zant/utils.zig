const std = @import("std");
const zant = @import("zant");
const Tensor = zant.core.tensor.Tensor;
pub const AnyTensor = zant.core.tensor.AnyTensor;

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;
const ValueInfoProto = onnx.ValueInfoProto;
const DataType = onnx.DataType;

const allocator = std.heap.page_allocator;

// --- zant ---
pub const tensorZant_lib = @import("tensorZant.zig");
const TensorType = tensorZant_lib.TensorType;
pub const TensorZant = tensorZant_lib.TensorZant;
pub const TensorCategory = tensorZant_lib.TensorCategory;

// --- uops ---
const DType = zant.uops.DType;

pub fn getValueInfoTensorFromGraphInfo(name: []const u8, protoGraph: *GraphProto) ?*ValueInfoProto {
    for (protoGraph.value_info) |vi| {
        if (std.mem.eql(u8, vi.name.?, name)) {
            return vi;
        }
    }
    return null;
}

pub fn calculateQLinearAveragePoolShape(node: *NodeProto, tensorMap: anytype) ?[]usize {
    // Get input tensor name (first input is the data tensor)
    if (node.input.len == 0) return null;
    const input_name = node.input[0];

    // Get input tensor from map
    const input_tensor = tensorMap.getPtr(input_name) orelse return null;
    const input_shape = input_tensor.getShape();

    if (input_shape.len < 3) return null; // Need at least batch, channel, spatial dims

    // Get attributes
    var kernel_shape: ?[]i64 = null;
    var strides: ?[]i64 = null;
    var pads: ?[]i64 = null;
    var ceil_mode: bool = false;

    for (node.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "kernel_shape")) {
            kernel_shape = attr.ints;
        } else if (std.mem.eql(u8, attr.name, "strides")) {
            strides = attr.ints;
        } else if (std.mem.eql(u8, attr.name, "pads")) {
            pads = attr.ints;
        } else if (std.mem.eql(u8, attr.name, "ceil_mode")) {
            ceil_mode = attr.i != 0;
        }
    }

    if (kernel_shape == null) return null;

    // Default values
    const actual_strides = strides orelse kernel_shape.?;
    const spatial_dims = kernel_shape.?.len;

    // Allocate output shape
    var output_shape = allocator.alloc(usize, input_shape.len) catch return null;

    // Copy batch and channel dimensions
    output_shape[0] = input_shape[0]; // batch
    output_shape[1] = input_shape[1]; // channels

    // Calculate spatial dimensions
    for (0..spatial_dims) |i| {
        const input_size = @as(i64, @intCast(input_shape[2 + i]));
        const kernel_size = kernel_shape.?[i];
        const stride = actual_strides[i];

        // Handle padding
        var pad_begin: i64 = 0;
        var pad_end: i64 = 0;
        if (pads) |p| {
            if (i < p.len) pad_begin = p[i];
            if (i + spatial_dims < p.len) pad_end = p[i + spatial_dims];
        }

        // Calculate output size using ONNX formula
        const output_size = if (ceil_mode) blk: {
            const numerator = input_size + pad_begin + pad_end - kernel_size;
            break :blk @divTrunc(numerator + stride, stride);
        } else blk: {
            const numerator = input_size + pad_begin + pad_end - kernel_size;
            break :blk @divTrunc(numerator, stride) + 1;
        };

        output_shape[2 + i] = @as(usize, @max(1, @as(usize, @intCast(output_size))));
    }

    return output_shape;
}

pub fn calculateQLinearConcatShape(node: *NodeProto, tensorMap: anytype) ?[]usize {
    // Get axis attribute
    var axis: i64 = 1; // default axis
    for (node.attribute) |attr| {
        if (std.mem.eql(u8, attr.name, "axis")) {
            axis = attr.i;
            break;
        }
    }

    // QLinearConcat inputs: output_scale, output_zp, tensor1, scale1, zp1, tensor2, scale2, zp2, ...
    // Pattern: output_scale, output_zp, [tensor, scale, zp] repeating

    if (node.input.len < 5) return null; // Need at least output_scale, output_zp, tensor, scale, zp

    // Find first tensor input (should be index 2)
    const first_tensor_name = node.input[2];
    const first_tensor = tensorMap.getPtr(first_tensor_name) orelse return null;
    const input_shape = first_tensor.getShape();

    // Calculate number of input tensors
    // Format: output_scale, output_zp, [tensor, scale, zp] repeating
    const num_inputs = (node.input.len - 2) / 3;
    if (num_inputs == 0) return null;

    // Calculate output shape - same as input except for the concatenation axis
    var output_shape = allocator.alloc(usize, input_shape.len) catch return null;

    // Copy all dimensions
    for (input_shape, 0..) |dim, i| {
        output_shape[i] = dim;
    }

    // Calculate concatenated dimension size
    var concat_size: usize = 0;
    for (0..num_inputs) |i| {
        const tensor_idx = 2 + i * 3; // skip to tensor: 2, 5, 8, 11, ...
        if (tensor_idx >= node.input.len) break;

        const tensor_name = node.input[tensor_idx];
        if (tensorMap.getPtr(tensor_name)) |tensor| {
            const tensor_shape = tensor.getShape();
            if (tensor_shape.len != input_shape.len) continue; // shape mismatch

            const normalized_axis = if (axis < 0) @as(usize, @intCast(@as(i64, @intCast(tensor_shape.len)) + axis)) else @as(usize, @intCast(axis));
            if (normalized_axis >= tensor_shape.len) continue;

            concat_size += tensor_shape[normalized_axis];
        }
    }

    // Set the concatenated dimension
    const normalized_axis = if (axis < 0) @as(usize, @intCast(@as(i64, @intCast(input_shape.len)) + axis)) else @as(usize, @intCast(axis));
    if (normalized_axis < output_shape.len) {
        output_shape[normalized_axis] = concat_size;
    }

    return output_shape;
}

pub fn getShapeFromModelInfo(model: *ModelProto) ?[]i64 {
    for (model.value_info) |vi| {
        return getTensorShapeFromValueInfo(vi);
    }
}

pub fn getTensorShapeFromValueInfo(vi: *ValueInfoProto) ?[]i64 {
    if (vi.type) |type_info| {
        if (type_info.tensor_type) |tensor_type| {
            if (tensor_type.shape) |shape| {
                return shape.shape;
            }
        }
    }
    return null;
}

pub fn getTypeFromValueInfo(vi: *ValueInfoProto) !TensorType {

    // Derive and store the input element type string (e.g., "f32", "u8")
    if (vi.type) |type_info| {
        if (type_info.tensor_type) |tensor_type| {
            const raw_et: u32 = tensor_type.elem_type;
            const int_val = @as(i32, @intCast(raw_et));
            const input_dt = @as(DataType, @enumFromInt(int_val));
            // Store the calculated DataType globally
            return switch (input_dt) {
                .INT64 => TensorType.i64,
                .DOUBLE => TensorType.f64,
                .UINT64 => TensorType.u64,
                .FLOAT => TensorType.f32,
                .INT32 => TensorType.i32,
                .UINT32 => TensorType.u32,
                .FLOAT16 => TensorType.f16,
                .INT16 => TensorType.i16,
                .UINT16 => TensorType.u16,
                .INT8 => TensorType.i8,
                .UINT8 => TensorType.u8,
                else => error.DataTypeNotAvailable,
            };
        }
    }
    return error.DataTypeNotAvailable;
}

pub fn getAnyTensorType(anyTensor: AnyTensor) TensorType {
    return switch (anyTensor) {
        .i64 => TensorType.i64,
        .f64 => TensorType.f64,
        .u64 => TensorType.u64,
        .f32 => TensorType.f32,
        .i32 => TensorType.i32,
        .u32 => TensorType.u32,
        .f16 => TensorType.f16,
        .i16 => TensorType.i16,
        .u16 => TensorType.u16,
        .i8 => TensorType.i8,
        .u8 => TensorType.u8,
    };
}

//Returns the sanitized tensor's name, removes all non alphanumeric chars
pub inline fn getSanitizedName(name: []const u8) ![]const u8 {
    var sanitized = try allocator.alloc(u8, name.len);

    for (name, 0..) |char, i| {
        sanitized[i] = if (std.ascii.isAlphanumeric(char) or char == '_')
            std.ascii.toLower(char)
        else
            '_';
    }

    //std.log.debug("\nfrom {s} to {s} ", .{ name, sanitized });

    return sanitized;
}

// ----------------- DATA TYPE management -------------

pub inline fn i64SliceToUsizeSlice(input: []const i64) ![]usize {
    var output = try allocator.alloc(usize, input.len);

    const maxUsize = std.math.maxInt(usize);

    for (input, 0..) |value, index| {
        if (value < 0) {
            return error.NegativeValue;
        }
        if (value > maxUsize) {
            return error.ValueTooLarge;
        }
        output[index] = @intCast(value);
    }

    return output;
}

pub fn i64SliceToUsizeArrayString(values: []const i64) ![]const u8 {
    var list = std.ArrayList(u8).init(allocator);
    defer list.deinit(); // Frees all memory

    try list.appendSlice("&[_]usize{");
    for (values, 0..) |val, i| {
        if (i > 0) try list.append(',');
        try list.writer().print("{}", .{val});
    }
    try list.append('}');

    return try list.toOwnedSlice(); // Caller must free this!
}

pub fn i64SliceToi64ArrayString(values: []const i64) ![]const u8 {
    var list = std.ArrayList(u8).init(allocator);
    defer list.deinit(); // Frees all memory

    try list.appendSlice("&[_]i64{");
    for (values, 0..) |val, i| {
        if (i > 0) try list.append(',');
        try list.writer().print("{}", .{val});
    }
    try list.append('}');

    return try list.toOwnedSlice(); // Caller must free this!
}

pub fn usizeSliceToI64Slice(input: []usize) ![]const i64 {
    var output = try allocator.alloc(i64, input.len);

    for (input, 0..) |value, index| {
        if (value > std.math.maxInt(i64)) {
            return error.ValueTooLarge;
        }
        output[index] = @intCast(value);
    }

    return output;
}

// TODO where to get out_Dtype?
pub fn tensorTypeToDtype(tensor_type: TensorType) DType {
    return switch (tensor_type) {
        .f16 => DType.f16,
        .f32 => DType.f32,
        .f64 => DType.f64,
        .i8 => DType.i8,
        .i16 => DType.i16,
        .i32 => DType.i32,
        .i64 => DType.i64,
        .u8 => DType.u8,
        .u16 => DType.u16,
        .u32 => DType.u32,
        .u64 => DType.u64,
        .bool => DType.bool,
        .undefined => DType.undefined,
    };
}
// ----------------- STRUCT TYPE management -------------

pub fn protoTensor2AnyTensor(proto: *TensorProto) !AnyTensor {
    // Allocate shape array
    var shape = try allocator.alloc(usize, proto.dims.len);
    for (proto.dims, 0..) |dim, i| {
        if (dim < 0) {
            return error.NegativeDimension;
        }
        shape[i] = @intCast(dim);
    }
    defer allocator.free(shape);

    if (proto.float16_data) |float16_data| {
        const tensor = try allocator.create(Tensor(f16));
        tensor.* = try Tensor(f16).fromArray(&allocator, float16_data, shape);
        return AnyTensor{ .f16 = tensor };
    } else if (proto.float_data) |float_data| {
        const tensor = try allocator.create(Tensor(f32));
        tensor.* = try Tensor(f32).fromArray(&allocator, float_data, shape);
        return AnyTensor{ .f32 = tensor };
    } else if (proto.int32_data) |int32_data| {
        if (proto.data_type == .UINT8) return truncate_to_UINT8(i32, int32_data, shape);
        const tensor = try allocator.create(Tensor(i32));
        tensor.* = try Tensor(i32).fromArray(&allocator, int32_data, shape);
        return AnyTensor{ .i32 = tensor };
    } else if (proto.int64_data) |int64_data| {
        const tensor = try allocator.create(Tensor(i64));
        tensor.* = try Tensor(i64).fromArray(&allocator, int64_data, shape);
        return AnyTensor{ .i64 = tensor };
    } else if (proto.int8_data) |int8_data| {
        const tensor = try allocator.create(Tensor(i8));
        tensor.* = try Tensor(i8).fromArray(&allocator, int8_data, shape);
        return AnyTensor{ .i8 = tensor };
    } else if (proto.uint8_data) |uint8_data| {
        const tensor = try allocator.create(Tensor(u8));
        tensor.* = try Tensor(u8).fromArray(&allocator, uint8_data, shape);
        return AnyTensor{ .u8 = tensor };
    } else if (proto.double_data) |double_data| {
        const tensor = try allocator.create(Tensor(f64));
        tensor.* = try Tensor(f64).fromArray(&allocator, double_data, shape);
        return AnyTensor{ .f64 = tensor };
    } else if (proto.uint64_data) |uint64_data| {
        const tensor = try allocator.create(Tensor(u64));
        tensor.* = try Tensor(u64).fromArray(&allocator, uint64_data, shape);
        return AnyTensor{ .u64 = tensor };
    } else if (proto.uint16_data) |uint16_data| {
        const tensor = try allocator.create(Tensor(u16));
        tensor.* = try Tensor(u16).fromArray(&allocator, uint16_data, shape);
        return AnyTensor{ .u16 = tensor };
    } else if (proto.raw_data) |raw| {
        // Handle raw data based on data_type
        switch (proto.data_type) {
            .FLOAT => return try fromRawData(f32, raw, shape),
            .FLOAT16 => return try fromRawData(f16, raw, shape),
            .INT32 => return try fromRawData(i32, raw, shape),
            .INT8 => return try fromRawData(i8, raw, shape),
            .INT64 => return try fromRawData(i64, raw, shape),
            .DOUBLE => return try fromRawData(f64, raw, shape),
            .UINT64 => return try fromRawData(u64, raw, shape),
            .UINT16 => return try fromRawData(u16, raw, shape),
            .UINT8 => return try fromRawData(u8, raw, shape),
            else => {
                std.log.info("\n[writeArray] Error: Unsupported raw data type {any} for tensor {s}", .{ proto.data_type, proto.name.? });
                std.log.err("Unsupported raw data type: {any}", .{proto.data_type});
                return error.DataTypeNotAvailable;
            },
        }
    } else {
        std.debug.print("\n\nERROR: Unsupported data type for tensor {s}", .{proto.name.?});
        std.debug.print("\nTensorProto.print():", .{});
        proto.print(null);
        return error.UnsupportedDataType;
    }
}

fn fromRawData(T: type, raw_data: []const u8, shape: []usize) !AnyTensor {
    const elem_size = @sizeOf(T);
    const num_elements = raw_data.len / elem_size;

    // Ensure raw_data length is a multiple of element size
    if (raw_data.len % elem_size != 0) {
        std.log.err("Raw data length {d} is not a multiple of element size {d} for type {any}", .{ raw_data.len, elem_size, T });
        return error.InvalidRawDataLength;
    }

    const data = try allocator.alloc(T, num_elements);
    for (data, 0..) |*dest, i| {
        const offset = i * elem_size;
        dest.* = std.mem.bytesToValue(T, raw_data[offset .. offset + elem_size]);
    }

    const tensor = try allocator.create(Tensor(T));
    tensor.* = try Tensor(T).fromArray(&allocator, data, shape);
    return switch (T) {
        f32 => AnyTensor{ .f32 = tensor },
        f16 => AnyTensor{ .f16 = tensor },
        i32 => AnyTensor{ .i32 = tensor },
        i8 => AnyTensor{ .i8 = tensor },
        i64 => AnyTensor{ .i64 = tensor },
        f64 => AnyTensor{ .f64 = tensor },
        u64 => AnyTensor{ .u64 = tensor },
        u16 => AnyTensor{ .u16 = tensor },
        u8 => AnyTensor{ .u8 = tensor },
        else => {
            error.TypeNotAvailable;
        },
    };
}

//sometimes the onnx saves the UINT8 arrays as int32, not even raw data, just int32. You have to notice it by checking if the destination type is U8 when parsing an int32, like above.
fn truncate_to_UINT8(inputType: type, data: []inputType, shape: []usize) !AnyTensor {
    const len = data.len;

    // Allocate output buffer
    const output: []u8 = try allocator.alloc(u8, len);

    // Convert each element to u8 with truncation
    for (data, output) |val, *out| {
        out.* = @intCast(val); // or use @truncate(u8, val) if you're sure
    }

    const tensor = try allocator.create(Tensor(u8));
    tensor.* = try Tensor(u8).fromArray(&allocator, output, shape);
    return AnyTensor{ .u8 = tensor };
}

pub fn broadcastShapes(general_allocator: std.mem.Allocator, shape1: []usize, shape2: []usize) ![]usize {
    const max_len = std.math.max(shape1.len, shape2.len);

    var output = try general_allocator.alloc(usize, max_len);
    errdefer general_allocator.free(output);

    // Compute padding
    const pad1 = max_len - shape1.len;
    const pad2 = max_len - shape2.len;

    for (0..max_len) |i| {
        // Get the corresponding dimensions, using 1 for padding if necessary
        const dim1 = if (i < pad1) @as(usize, 1) else shape1[i - pad1];
        const dim2 = if (i < pad2) @as(usize, 1) else shape2[i - pad2];

        // Check compatibility and compute the output dimension
        if (dim1 == dim2) {
            output[i] = dim1;
        } else if (dim1 == 1) {
            output[i] = dim2;
        } else if (dim2 == 1) {
            output[i] = dim1;
        } else {
            allocator.free(output);
            return error.IncompatibleShapes;
        }
    }
    return output;
}

pub fn getInitializers(hashMap: *std.StringHashMap(TensorZant)) ![]TensorZant {
    var initializers = std.ArrayList(TensorZant).init(allocator);
    var it = hashMap.iterator();
    while (it.next()) |entry| {
        if (entry.value_ptr.tc == TensorCategory.INITIALIZER) {
            try initializers.append(entry.value_ptr.*);
        }
    }
    return initializers.toOwnedSlice();
}

pub fn getConstants(hashMap: *std.StringHashMap(TensorZant)) ![]TensorZant {
    var constants = std.ArrayList(TensorZant).init(allocator);
    var it = hashMap.iterator();
    while (it.next()) |entry| {
        if (entry.value_ptr.tc == TensorCategory.CONSTANT) {
            try constants.append(entry.value_ptr.*);
        }
    }
    return constants.toOwnedSlice();
}

pub fn getLinkers(hashMap: *std.StringHashMap(TensorZant)) ![]TensorZant {
    var linkers = std.ArrayList(TensorZant).init(allocator);
    var it = hashMap.iterator();
    while (it.next()) |entry| {
        if (entry.value_ptr.tc == TensorCategory.LINK) {
            try linkers.append(entry.value_ptr.*);
        }
    }
    return linkers.toOwnedSlice();
}

pub fn getOutputs(hashMap: *std.StringHashMap(TensorZant)) ![]TensorZant {
    var outputs = std.ArrayList(TensorZant).init(allocator);
    var it = hashMap.iterator();
    while (it.next()) |entry| {
        if (entry.value_ptr.tc == TensorCategory.OUTPUT) {
            try outputs.append(entry.value_ptr.*);
        }
    }
    return outputs.toOwnedSlice();
}

pub fn getInputs(hashMap: *std.StringHashMap(TensorZant)) ![]TensorZant {
    var inputs = std.ArrayList(TensorZant).init(allocator);
    var it = hashMap.iterator();
    while (it.next()) |entry| {
        if (entry.value_ptr.tc == TensorCategory.INPUT) {
            try inputs.append(entry.value_ptr.*);
        }
    }
    return inputs.toOwnedSlice();
}

pub fn getAllTensors(hashMap: *std.StringHashMap(TensorZant)) ![]TensorZant {
    var inputs = std.ArrayList(TensorZant).init(allocator);
    var it = hashMap.iterator();
    while (it.next()) |entry| {
        if (entry.value_ptr.tc == TensorCategory.OUTPUT) {
            try inputs.append(entry.value_ptr.*);
        }
    }
    return inputs.toOwnedSlice();
}
