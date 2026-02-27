const std = @import("std");
const zant = @import("zant");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;
const error_handler = zant.utils.error_handler;
const TensorMathError = error_handler.TensorMathError;
const TensorError = error_handler.TensorError;
const quantScheme = zant.core.quantization.quantScheme;
const QuantTensMath = @import("quant_tensor_math_standard.zig");

pub fn quant_sum_tensors(comptime inputType: anytype, comptime outputType: anytype, t1: *const Tensor(inputType), t2: *const Tensor(inputType)) !Tensor(outputType) {
    // CHECKS:
    if (t1.size != t2.size) return TensorMathError.InputTensorDifferentSize;

    // if (@bitSizeOf(outputType) <= 16) { // quantized
    //     if (@bitSizeOf(outputType) <= (@bitSizeOf(inputType) * 2)) return TensorMathError.TooSmallOutputType;
    // } else { // non-quant
    //     if (@bitSizeOf(outputType) < @bitSizeOf(inputType)) return TensorMathError.TooSmallOutputType;
    // }

    // Create output tensor initialized to zero
    var out_tensor = try Tensor(outputType).fromShape(t1.allocator, t2.shape);
    // Propagate details
    out_tensor.details = t1.details;

    try quant_lean_sum_tensors(inputType, outputType, t1, t2, &out_tensor);

    return out_tensor;
}

pub inline fn quant_lean_sum_tensors(comptime inputType: anytype, comptime outputType: anytype, t1: *const Tensor(inputType), t2: *const Tensor(inputType), outputTensor: *Tensor(outputType)) !void {
    switch (outputTensor.details) {
        .quant => |*qd| {
            if (@typeInfo(inputType) == .int) {

                // INTEGER OPERATIONS
                // If the input tensors have the same scale factor operate directly in quantized integer
                if (t1.details.quant.scale_factor == t2.details.quant.scale_factor) {

                    // Set output tensor details
                    qd.zero_point = (t1.details.quant.zero_point + t2.details.quant.zero_point) / 2;
                    qd.scale_factor = t1.details.quant.scale_factor;

                    // Max and min values for clamping
                    const max_value: usize = std.math.maxInt(outputType);
                    const min_value: usize = std.math.minInt(outputType);

                    // Simple case: same size tensors
                    if (t1.size == t2.size) {

                        // Use unrolled loop for small sizes to avoid SIMD overhead
                        if (t1.size <= 8) {
                            comptime var unroll = 0;
                            inline while (unroll < 8) : (unroll += 1) {
                                if (unroll < t1.size and unroll < t2.size) {
                                    // Calculate the quantized result in vectors and clamp it
                                    const result: i32 = qd.zero_point + t1.data[unroll] - t1.details.quant.zero_point + t2.data[unroll] - t2.details.quant.zero_point;
                                    outputTensor.data[unroll] = if (result > max_value) {
                                        max_value;
                                    } else if (result < min_value) {
                                        min_value;
                                    } else {
                                        @as(outputType, @intCast(result));
                                    };
                                }
                            }
                        }

                        // Use SIMD for larger sizes
                        else {
                            const vector_len = std.simd.suggestVectorLength(inputType) orelse 4;
                            const Vec = @Vector(vector_len, inputType);
                            const i32Vec = @Vector(vector_len, i32);

                            // Process 4 vectors at once to exploit instruction-level parallelism
                            const chunk_size = vector_len * 4;
                            const chunks = t1.size / chunk_size;
                            var i: usize = 0;

                            while (i < chunks * chunk_size) : (i += chunk_size) {
                                inline for (0..4) |offset| {
                                    // Calculate the quantized result in vectors and clamp it
                                    const v1: Vec = t1.data[i + offset * vector_len ..][0..vector_len].* - t1.details.quant.zero_point;
                                    const v2: Vec = t2.data[i + offset * vector_len ..][0..vector_len].* - t2.details.quant.zero_point;
                                    const result: i32Vec = @as(i32Vec, v1 + v2) - @as(i32Vec, qd.zero_point);
                                    comptime var j = 0;
                                    inline while (j < vector_len) : (j += 1) {
                                        outputTensor.data[i + offset * vector_len + j] = if (result[j] > max_value) {
                                            max_value;
                                        } else if (result[j] < min_value) {
                                            min_value;
                                        } else {
                                            @as(outputType, @intCast(result[j]));
                                        };
                                    }
                                }
                            }

                            // Handle remaining elements with simple loop
                            while (i < t1.size) : (i += 1) {
                                // Calculate the quantized result and clamp it
                                const result: i32 = qd.zero_point + t1.data[i] - t1.details.quant.zero_point + t2.data[i] - t2.details.quant.zero_point;
                                outputTensor.data[i] = if (result > max_value) {
                                    max_value;
                                } else if (result < min_value) {
                                    min_value;
                                } else {
                                    @as(outputType, @intCast(result));
                                };
                            }
                        }
                    }

                    // Broadcasting is needed
                    else {

                        // Broadcasting case - use stack arrays for small ranks to avoid allocations
                        const rank1 = t1.shape.len;
                        const rank2 = t2.shape.len;
                        const max_rank = @max(rank1, rank2);

                        // Use stack arrays for common tensor ranks (up to 4D)
                        var stack_shape1: [4]usize = [_]usize{1} ** 4;
                        var stack_shape2: [4]usize = [_]usize{1} ** 4;
                        var stack_strides1: [4]usize = undefined;
                        var stack_strides2: [4]usize = undefined;
                        var stack_out_strides: [4]usize = undefined;
                        var stack_indices: [4]usize = [_]usize{0} ** 4;

                        const shape1 = if (max_rank <= 4) stack_shape1[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
                        const shape2 = if (max_rank <= 4) stack_shape2[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
                        const strides1 = if (max_rank <= 4) stack_strides1[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
                        const strides2 = if (max_rank <= 4) stack_strides2[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
                        const out_strides = if (max_rank <= 4) stack_out_strides[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
                        const indices = if (max_rank <= 4) stack_indices[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);

                        // Free heap-allocated arrays at block scope exit (only if actually heap-allocated)
                        defer if (max_rank > 4) pkg_allocator.free(shape1);
                        defer if (max_rank > 4) pkg_allocator.free(shape2);
                        defer if (max_rank > 4) pkg_allocator.free(strides1);
                        defer if (max_rank > 4) pkg_allocator.free(strides2);
                        defer if (max_rank > 4) pkg_allocator.free(out_strides);
                        defer if (max_rank > 4) pkg_allocator.free(indices);

                        // Copy original shapes from right to left
                        var i: usize = 0;
                        while (i < rank1) : (i += 1) {
                            shape1[max_rank - rank1 + i] = t1.shape[i];
                        }
                        i = 0;
                        while (i < rank2) : (i += 1) {
                            shape2[max_rank - rank2 + i] = t2.shape[i];
                        }

                        // Verify shapes and calculate output shape
                        for (0..max_rank) |dim| {
                            if (shape1[dim] != shape2[dim] and shape1[dim] != 1 and shape2[dim] != 1) {
                                return TensorMathError.IncompatibleBroadcastShapes;
                            }
                            outputTensor.shape[dim] = @max(shape1[dim], shape2[dim]);
                        }

                        // Calculate strides from right to left
                        var stride: usize = 1;
                        i = max_rank;
                        while (i > 0) {
                            i -= 1;
                            out_strides[i] = stride;
                            strides1[i] = if (shape1[i] > 1) stride else 0;
                            strides2[i] = if (shape2[i] > 1) stride else 0;
                            stride *= outputTensor.shape[i];
                        }

                        // Perform addition with broadcasting
                        @memset(indices, 0);

                        i = 0;
                        while (i < outputTensor.size) : (i += 1) {
                            // Calculate indices for current position
                            var temp = i;
                            for (0..max_rank) |dim| {
                                const idx = max_rank - 1 - dim;
                                indices[idx] = temp / out_strides[idx];
                                temp = temp % out_strides[idx];
                            }

                            // Calculate input indices considering broadcasting
                            var idx1: usize = 0;
                            var idx2: usize = 0;

                            // For same shape tensors, use the same index calculation
                            if (std.mem.eql(usize, shape1, shape2)) {
                                idx1 = i;
                                idx2 = i;
                            } else {
                                // For broadcasting case
                                for (0..max_rank) |dim| {
                                    const pos = indices[dim];
                                    // For t1: if dimension is 1, don't increment index (broadcasting)
                                    if (shape1[dim] > 1) {
                                        idx1 += pos * strides1[dim];
                                    }
                                    // For t2: if dimension is 1, don't increment index (broadcasting)
                                    if (shape2[dim] > 1) {
                                        const t2_pos = pos % shape2[dim];
                                        idx2 += t2_pos * strides2[dim];
                                    }
                                }
                            }
                            // Calculate the quantized result and clamp it
                            const result: i32 = qd.zero_point + t1.data[idx1] - t1.details.quant.zero_point + t2.data[idx2] - t2.details.quant.zero_point;
                            outputTensor.data[i] = if (result > max_value) {
                                max_value;
                            } else if (result < min_value) {
                                min_value;
                            } else {
                                @as(outputType, @intCast(result));
                            };
                        }
                    }
                }

                // FLOAT OPERATIONS
                // Else we need to dequantize and then requantize
                else {

                    // Dequantize C and result
                    const dequant_t1 = try Tensor(f32).fromShape(&pkg_allocator, t1.shape);
                    defer dequant_t1.deinit();
                    QuantTensMath.lean_dequantize(inputType, f32, t1, dequant_t1);

                    const dequant_t2 = try Tensor(f32).fromShape(&pkg_allocator, t2.shape);
                    defer dequant_t2.deinit();
                    QuantTensMath.lean_dequantize(inputType, f32, t2, dequant_t2);

                    // Create temporal tensor for the result
                    const dequant_result = try Tensor(f32).fromShape(&pkg_allocator, outputTensor.shape);
                    defer dequant_result.deinit();

                    // Simple case: same size tensors
                    if (dequant_t1.size == dequant_t2.size) {

                        // Use unrolled loop for small sizes to avoid SIMD overhead
                        if (dequant_t1.size <= 8) {
                            comptime var unroll = 0;
                            inline while (unroll < 8) : (unroll += 1) {
                                if (unroll < dequant_t1.size and unroll < dequant_t2.size) {
                                    dequant_result.data[unroll] = @as(outputType, dequant_t1.data[unroll] + dequant_t2.data[unroll]);
                                }
                            }
                        }

                        // Use SIMD for larger sizes
                        else {
                            const vector_len = std.simd.suggestVectorLength(inputType) orelse 4;
                            const Vec = @Vector(vector_len, f32);

                            // Process 4 vectors at once to exploit instruction-level parallelism
                            const chunk_size = vector_len * 4;
                            const chunks = t1.size / chunk_size;
                            var i: usize = 0;

                            while (i < chunks * chunk_size) : (i += chunk_size) {
                                inline for (0..4) |offset| {
                                    const v1: Vec = dequant_t1.data[i + offset * vector_len ..][0..vector_len].*;
                                    const v2: Vec = dequant_t2.data[i + offset * vector_len ..][0..vector_len].*;
                                    const result = v1 + v2;
                                    comptime var j = 0;
                                    inline while (j < vector_len) : (j += 1) {
                                        dequant_result.data[i + offset * vector_len + j] = result[j];
                                    }
                                }
                            }

                            // Handle remaining elements with simple loop
                            while (i < t1.size) : (i += 1) {
                                dequant_result.data[i] = dequant_t1.data[i] + dequant_t2.data[i];
                            }
                        }
                    }
                    // Broadcasting is needed
                    else {

                        // Broadcasting case - use stack arrays for small ranks to avoid allocations
                        const rank1 = dequant_t1.shape.len;
                        const rank2 = dequant_t2.shape.len;
                        const max_rank = @max(rank1, rank2);

                        // Use stack arrays for common tensor ranks (up to 4D)
                        var stack_shape1: [4]usize = [_]usize{1} ** 4;
                        var stack_shape2: [4]usize = [_]usize{1} ** 4;
                        var stack_strides1: [4]usize = undefined;
                        var stack_strides2: [4]usize = undefined;
                        var stack_out_strides: [4]usize = undefined;
                        var stack_indices: [4]usize = [_]usize{0} ** 4;

                        const shape1 = if (max_rank <= 4) stack_shape1[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
                        const shape2 = if (max_rank <= 4) stack_shape2[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
                        const strides1 = if (max_rank <= 4) stack_strides1[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
                        const strides2 = if (max_rank <= 4) stack_strides2[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
                        const out_strides = if (max_rank <= 4) stack_out_strides[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);
                        const indices = if (max_rank <= 4) stack_indices[0..max_rank] else try pkg_allocator.alloc(usize, max_rank);

                        // Free heap-allocated arrays at block scope exit (only if actually heap-allocated)
                        defer if (max_rank > 4) pkg_allocator.free(shape1);
                        defer if (max_rank > 4) pkg_allocator.free(shape2);
                        defer if (max_rank > 4) pkg_allocator.free(strides1);
                        defer if (max_rank > 4) pkg_allocator.free(strides2);
                        defer if (max_rank > 4) pkg_allocator.free(out_strides);
                        defer if (max_rank > 4) pkg_allocator.free(indices);

                        // Copy original shapes from right to left
                        var i: usize = 0;
                        while (i < rank1) : (i += 1) {
                            shape1[max_rank - rank1 + i] = dequant_t1.shape[i];
                        }
                        i = 0;
                        while (i < rank2) : (i += 1) {
                            shape2[max_rank - rank2 + i] = dequant_t2.shape[i];
                        }

                        // Verify shapes and calculate output shape
                        for (0..max_rank) |dim| {
                            if (shape1[dim] != shape2[dim] and shape1[dim] != 1 and shape2[dim] != 1) {
                                return TensorMathError.IncompatibleBroadcastShapes;
                            }
                            dequant_result.shape[dim] = @max(shape1[dim], shape2[dim]);
                        }

                        // Calculate strides from right to left
                        var stride: usize = 1;
                        i = max_rank;
                        while (i > 0) {
                            i -= 1;
                            out_strides[i] = stride;
                            strides1[i] = if (shape1[i] > 1) stride else 0;
                            strides2[i] = if (shape2[i] > 1) stride else 0;
                            stride *= dequant_result.shape[i];
                        }

                        // Perform addition with broadcasting
                        @memset(indices, 0);

                        i = 0;
                        while (i < dequant_result.size) : (i += 1) {
                            // Calculate indices for current position
                            var temp = i;
                            for (0..max_rank) |dim| {
                                const idx = max_rank - 1 - dim;
                                indices[idx] = temp / out_strides[idx];
                                temp = temp % out_strides[idx];
                            }

                            // Calculate input indices considering broadcasting
                            var idx1: usize = 0;
                            var idx2: usize = 0;

                            // For same shape tensors, use the same index calculation
                            if (std.mem.eql(usize, shape1, shape2)) {
                                idx1 = i;
                                idx2 = i;
                            } else {
                                // For broadcasting case
                                for (0..max_rank) |dim| {
                                    const pos = indices[dim];
                                    // For t1: if dimension is 1, don't increment index (broadcasting)
                                    if (shape1[dim] > 1) {
                                        idx1 += pos * strides1[dim];
                                    }
                                    // For t2: if dimension is 1, don't increment index (broadcasting)
                                    if (shape2[dim] > 1) {
                                        const t2_pos = pos % shape2[dim];
                                        idx2 += t2_pos * strides2[dim];
                                    }
                                }
                            }

                            dequant_result.data[i] = dequant_t1.data[idx1] + dequant_t2.data[idx2];
                        }
                    }

                    // Requantize the result
                    outputTensor.deinit();
                    try QuantTensMath.lean_quantize_minmax(outputType, dequant_result, outputTensor, quantScheme.ASYM);
                }
            } else return TensorError.NotQuantizedTensor;
        },

        else => {
            return TensorError.NotQuantizedTensor;
        },
    }
}
