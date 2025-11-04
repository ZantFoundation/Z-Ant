const std = @import("std");
const zant = @import("../../../zant.zig");
const quant = zant.core.quantization;
const Tensor = zant.core.tensor.Tensor;
const TensorType = zant.core.tensor.TensorType;

const pkgAllocator = zant.utils.allocator.allocator;

pub const quantScheme = enum {
    SYMM,
    ASYM,
};

/// Quantizes the input Tensor to the outputType.
/// Parameters:
/// - inputType: data type of the tensor to be quantized.
/// - outputType: data type to be quantized to.
/// - input: pointer to the input tensor to be quantized
/// - scheme: quantization scheme (symmetric or asymmetric)
/// note: as of now the quantization scheme is ignored and hardcoded to asymmetric
/// Returns the quantized Tensor.
pub fn quantize(comptime inputType: type, comptime outputType: type, input: *Tensor(inputType), scheme: quantScheme) !Tensor(outputType) {
    const hardcodedScheme = quantScheme.ASYM; // asymm hardcoded
    _ = scheme;

    const output_shape = try get_quantize_output_shape(input.shape);
    defer pkgAllocator.free(output_shape);

    var output = try Tensor(outputType).fromShape(&pkgAllocator, output_shape);
    errdefer output.deinit();

    // minmax quantization "hardcoded"
    try lean_quantize_minmax(inputType, outputType, input, &output, hardcodedScheme);

    return output;
}

pub fn lean_quantize_minmax(comptime inputType: type, comptime outputType: type, input: *Tensor(inputType), output: *Tensor(outputType), scheme: quantScheme) !void {
    const result = try minmax_array_quant(inputType, outputType, scheme, input.data);
    defer pkgAllocator.free(result.quantizedArray);

    @memcpy(output.data, result.quantizedArray);
}

pub fn get_quantize_output_shape(input_shape: []const usize) ![]usize {
    // Allocate and copy the input shape
    const output_shape = try pkgAllocator.alloc(usize, input_shape.len);
    errdefer pkgAllocator.free(output_shape);

    @memcpy(output_shape, input_shape);

    return output_shape;
}

// ------------------------- Quantization methods and helper functions -------------------------

// ========== helper functions
pub fn clamp(comptime T: type, comptime U: type, value: T, scale: T, zero: i32, minInt: U, maxInt: U) U {
    var roundedVal: T = undefined;

    if (scale == 0) {
        roundedVal = value;
    } else {
        roundedVal = @round(value / scale + @as(T, @floatFromInt(zero)));
    }

    if (roundedVal <= @as(T, @floatFromInt(minInt)))
        return minInt;
    if (roundedVal >= @as(T, @floatFromInt(maxInt)))
        return maxInt;

    const roundedValInt: U = @as(U, @intFromFloat(roundedVal));

    return roundedValInt;
}

pub inline fn get_scale_factor(comptime T: type, comptime U: type, minFloat: T, maxFloat: T) T {
    const num: T = maxFloat - minFloat;

    const num_elements = (1 << @bitSizeOf(U)) - 1; // 2^b - 1 values
    const denom: T = @as(T, @floatFromInt(num_elements));

    return num / denom;
}

/// Computes and returns the zero point.
/// Params:
/// - T: input type
/// - U: outputtype
/// - scale: scale factor
/// - minFloat: minimum value in the floating point range
/// - minInt: minimum value of the quantized data type range
/// Returns: the zero point as a i32
pub inline fn get_zero_point(comptime T: type, comptime U: type, scale: T, minFloat: T, minInt: U) i32 {
    if (scale == 0)
        return @as(i32, @intCast(minInt)) - @as(i32, @intFromFloat(minFloat));

    const zeroPointFloat: T = minFloat / scale;

    return @as(i32, @intCast(minInt)) - @as(i32, @intFromFloat(zeroPointFloat));
}

// ========== quantization

/// This function quantizes the input monodimensional array, using the given parameters:
/// scale factor, zero point, minInt/maxInt (aka the integer grid limits).
/// Parameters:
/// - T: input data type
/// - U: output data type, to be quantized to
/// - inputArray: array to be quantized
/// - scale: the scale factor
/// - zero: the zero point
/// - minInt: chosen min value of the quantized range
/// - maxInt: chosen max value of the quantized range
/// Returns the quantized array.
/// The caller is responsible for freeing the returned array.
pub fn quantize_array(comptime T: type, comptime U: type, inputArray: anytype, scale: T, zero: i32, minInt: U, maxInt: U) ![]U {
    var output = try pkgAllocator.alloc(U, inputArray.len);

    for (inputArray, 0..) |val, i| {
        // quantize each val
        output[i] = clamp(T, U, val, scale, zero, minInt, maxInt);
    }

    return output;
}

/// This function quantizes the input monodimensional array using min/max method.
/// Parameters:
/// - T: input data type
/// - U: output data type, to be quantized to
/// - input: array to be quantized
/// - scheme: quantization scheme (symmetric or asymmetric)
/// Returns a tuple with the result quantized array, scale factor, zero point.
/// note: as of now the quantization scheme is ignored and hardcoded to asymmetric
/// The caller is responsible for freeing the returned quantized array.
pub fn minmax_array_quant(comptime T: type, comptime U: type, scheme: quantScheme, input: anytype) !struct { quantizedArray: []U, scale: T, zero: i32 } {
    var minFloat: T = input[0];
    var maxFloat: T = input[0];

    // compute the min and max value if the input tensor
    for (input[1..]) |val| {
        if (minFloat > val)
            minFloat = val;
        if (maxFloat < val)
            maxFloat = val;
    }

    // compute minInt and maxInt
    const minInt = if (@typeInfo(U) == .int) std.math.minInt(U) else std.math.floatMin(U);
    const maxInt = if (@typeInfo(U) == .int) std.math.maxInt(U) else std.math.floatMax(U);

    const scale: T = get_scale_factor(T, U, minFloat, maxFloat);

    var zero: i32 = undefined;
    switch (scheme) {
        quantScheme.SYMM => zero = 0,
        quantScheme.ASYM => zero = get_zero_point(T, U, scale, minFloat, minInt),
    }

    const quantizedArray: []U = try quantize_array(T, U, input, scale, zero, minInt, maxInt);
    const immutableZero: i32 = zero;
    return .{
        .quantizedArray = quantizedArray,
        .scale = scale,
        .zero = immutableZero,
    };
}

// Quantize (all arrays without distinctions for channels) but returns also the scale factor and zero point used
pub fn quantize_struct(comptime inputType: type, comptime outputType: type, input: *Tensor(inputType), scheme: quantScheme) !struct {
    tensor: Tensor(outputType),
    scale: inputType,
    zero: i32,
} {
    const hardcodedScheme = quantScheme.ASYM; // asymm hardcoded
    _ = scheme;

    const output_shape = try get_quantize_output_shape(input.shape);
    defer pkgAllocator.free(output_shape);

    var output = try Tensor(outputType).fromShape(&pkgAllocator, output_shape);
    errdefer output.deinit();

    const result = try minmax_array_quant(inputType, outputType, hardcodedScheme, input.data);
    defer pkgAllocator.free(result.quantizedArray);
    @memcpy(output.data, result.quantizedArray);

    return .{
        .tensor = output,
        .scales = result.scale,
        .zeros = result.zero,
    };
}

/// Quantizes the input tensor per-channel on [N, C, H, W] layout.
/// Each channel C gets its own scale factor and zero point, shared across all batches N.
/// Returns: struct containing the quantized tensor and arrays of scale/zero per channel
/// The caller is responsible for:
/// - calling .tensor.deinit() on the returned tensor
/// - freeing .scales with pkgAllocator.free()
/// - freeing .zeros with pkgAllocator.free()
pub fn quantize_per_channel(
    comptime inputType: type,
    comptime outputType: type,
    input: *Tensor(inputType),
    scheme: quantScheme,
) !struct {
    tensor: Tensor(outputType),
    scales: []inputType,
    zeros: []i32,
} {
    // Use local variable instead of reassigning parameter
    const hardcodedScheme = quantScheme.ASYM;
    _ = scheme; // Ignore parameter for now

    // Validate tensor layout (must be 4D: [N, C, H, W])
    if (input.shape.len != 4) {
        return error.InvalidDimensions;
    }

    const num_channels = input.shape[1]; // C

    // Validate number of channels
    if (num_channels == 0) {
        return error.InvalidDimensions;
    }

    // Allocate output tensor
    const output_shape = try get_quantize_output_shape(input.shape);
    defer pkgAllocator.free(output_shape);

    var output = try Tensor(outputType).fromShape(&pkgAllocator, output_shape);
    errdefer output.deinit();

    // Allocate arrays for scale factors and zero points (one per channel)
    const scales = try pkgAllocator.alloc(inputType, num_channels);
    errdefer pkgAllocator.free(scales);

    const zeros = try pkgAllocator.alloc(i32, num_channels);
    errdefer pkgAllocator.free(zeros);

    // Call the lean version to do the actual work
    try lean_quantize_per_channel(
        inputType,
        outputType,
        input,
        &output,
        scales,
        zeros,
        hardcodedScheme,
    );

    return .{
        .tensor = output,
        .scales = scales,
        .zeros = zeros,
    };
}

/// Lean version of per-channel quantization that writes directly to pre-allocated output.
/// Requires output, scales, and zeros to be pre-allocated.
///
/// This function processes each channel independently:
/// 1. Collects all spatial data for a channel across all batches
/// 2. Quantizes the channel using min-max quantization
/// 3. Distributes quantized data back to the corresponding batch locations
///
/// Layout assumption: [N, C, H, W]
pub fn lean_quantize_per_channel(
    comptime inputType: type,
    comptime outputType: type,
    input: *Tensor(inputType),
    output: *Tensor(outputType),
    scales: []inputType,
    zeros: []i32,
    scheme: quantScheme,
) !void {
    // Validate dimensions (must be 4D tensors)
    if (input.shape.len != 4 or output.shape.len != 4) {
        return error.InvalidDimensions;
    }

    // Verify that input and output shapes match
    if (input.shape[0] != output.shape[0] or
        input.shape[1] != output.shape[1] or
        input.shape[2] != output.shape[2] or
        input.shape[3] != output.shape[3])
    {
        return error.ShapeMismatch;
    }

    const batch_size = input.shape[0];
    const num_channels = input.shape[1];
    const height = input.shape[2];
    const width = input.shape[3];

    // Validate parameter array dimensions
    if (scales.len != num_channels or zeros.len != num_channels) {
        return error.ShapeMismatch;
    }

    // Validate that tensor has data
    if (num_channels == 0 or height == 0 or width == 0) {
        return error.InvalidDimensions;
    }

    // Calculate indexing offsets
    const channel_spatial_size = height * width;
    const batch_channel_size = num_channels * channel_spatial_size;
    const channel_total_size = batch_size * channel_spatial_size;

    // Temporary buffer to collect channel data across all batches
    var channel_buffer = try pkgAllocator.alloc(inputType, channel_total_size);
    defer pkgAllocator.free(channel_buffer);

    // Process each channel independently
    for (0..num_channels) |c| {
        // Gather all data for this channel from all batches
        for (0..batch_size) |n| {
            const batch_offset = n * batch_channel_size;
            const channel_offset = c * channel_spatial_size;
            const src_start = batch_offset + channel_offset;
            const src_end = src_start + channel_spatial_size;

            const dst_start = n * channel_spatial_size;
            const dst_end = dst_start + channel_spatial_size;

            // Bounds check
            if (src_end > input.data.len or dst_end > channel_buffer.len) {
                return error.OutOfBounds;
            }

            @memcpy(channel_buffer[dst_start..dst_end], input.data[src_start..src_end]);
        }

        // Quantize this channel using min-max quantization
        const result = try minmax_array_quant(
            inputType,
            outputType,
            scheme,
            channel_buffer,
        );
        defer pkgAllocator.free(result.quantizedArray);

        // Save quantization parameters for this channel
        scales[c] = result.scale;
        zeros[c] = result.zero;

        // Distribute quantized data back to respective batches
        for (0..batch_size) |n| {
            const batch_offset = n * batch_channel_size;
            const channel_offset = c * channel_spatial_size;
            const dst_start = batch_offset + channel_offset;
            const dst_end = dst_start + channel_spatial_size;

            const src_start = n * channel_spatial_size;
            const src_end = src_start + channel_spatial_size;

            // Bounds check
            if (dst_end > output.data.len or src_end > result.quantizedArray.len) {
                return error.OutOfBounds;
            }

            @memcpy(output.data[dst_start..dst_end], result.quantizedArray[src_start..src_end]);
        }
    }

    // To access all per-channel parameters, use the returned scales/zeros arrays
    output.details = .{
        .quant = .{
            .tensorType = .QuantTensor,
            .scale_factor = @floatCast(scales[0]),
            .zero_point = zeros[0],
        },
    };
}
