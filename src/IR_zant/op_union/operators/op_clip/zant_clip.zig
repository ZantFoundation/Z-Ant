const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const error_handler = zant.utils.error_handler;
const TensorMathError = error_handler.TensorMathError;
const TensorError = error_handler.TensorError;

/// Clips tensor elements element-wise into the range [min_val, max_val].
/// Writes results into outputTensor. Does not perform allocations or extensive checks.
pub inline fn clip_lean(
    comptime T: type,
    inputTensor: *const Tensor(T),
    minTensor: ?*const Tensor(T),
    maxTensor: ?*const Tensor(T),
    outputTensor: *Tensor(T),
) !void {
    // Get default min/max values based on type
    const min_val = if (minTensor) |t| t.data[0] else switch (@typeInfo(T)) {
        .int => std.math.minInt(T),
        .float => -std.math.floatMax(T),
        else => @as(T, 0),
    };

    const max_val = if (maxTensor) |t| t.data[0] else switch (@typeInfo(T)) {
        .int => std.math.maxInt(T),
        .float => std.math.floatMax(T),
        else => @as(T, 0),
    };

    // Handle the case where min > max: set all values to max
    if (min_val > max_val) {
        // According to ONNX spec, if min > max, fill with min value.
        @memset(outputTensor.data, min_val);
        return;
    }

    // Process elements in small chunks for better cache locality
    var i: usize = 0;
    const chunk_size = 32;

    while (i + chunk_size <= inputTensor.size) : (i += chunk_size) {
        comptime var j = 0;
        inline while (j < chunk_size) : (j += 1) {
            outputTensor.data[i + j] = @min(@max(inputTensor.data[i + j], min_val), max_val);
        }
    }

    // Handle remaining elements
    while (i < inputTensor.size) : (i += 1) {
        outputTensor.data[i] = @min(@max(inputTensor.data[i], min_val), max_val);
    }
}

/// Clips tensor elements element-wise into the range [min_val, max_val].
/// Allocates and returns a new tensor. Performs input validation.
pub fn clip(
    comptime T: type,
    allocator: std.mem.Allocator,
    inputTensor: *const Tensor(T),
    minTensor: ?*const Tensor(T),
    maxTensor: ?*const Tensor(T),
) !Tensor(T) {
    // --- Checks ---
    if (inputTensor.size == 0) {
        return TensorError.EmptyTensor;
    }
    if (minTensor) |t| {
        // Consider a tensor a scalar if its size is 1
        if (t.size != 1) return TensorMathError.InputTensorNotScalar;
        if (t.size == 0) return TensorError.EmptyTensor;
    }
    if (maxTensor) |t| {
        // Consider a tensor a scalar if its size is 1
        if (t.size != 1) return TensorMathError.InputTensorNotScalar;
        if (t.size == 0) return TensorError.EmptyTensor;
    }

    // Create a copy of the input tensor with the same shape
    var result = try Tensor(T).init(inputTensor.allocator);
    errdefer result.deinit();

    // Allocate memory for shape and data
    result.shape = try allocator.alloc(usize, inputTensor.shape.len);
    @memcpy(result.shape, inputTensor.shape);

    // Calculate total size
    var total_size: usize = 1;
    for (result.shape) |dim| {
        total_size *= dim;
    }

    // Allocate data
    result.data = try allocator.alloc(T, total_size);
    result.size = total_size;

    // Perform the clip operation
    try clip_lean(T, inputTensor, minTensor, maxTensor, &result);

    return result;
}
