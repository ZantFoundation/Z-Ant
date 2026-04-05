const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;

const pkg_allocator = zant.utils.allocator.allocator;

/// Implements the ONNX transpose operator (version 21)
/// Transposes the input tensor similar to numpy.transpose.
/// If perm is not provided, reverses the dimensions.
/// If perm is provided, permutes the axes according to the values given.
pub fn transpose(comptime T: type, input: *Tensor(T), perm: ?[]const usize) !Tensor(T) {
    var output = try Tensor(T).fromShape(&pkg_allocator, input.shape);
    errdefer output.deinit();

    try transpose_lean(T, input, perm, &output);

    return output;
}

/// Lean version of transpose that operates on an existing output tensor
/// Handles implicit broadcasting if the permutation length is greater than the input rank.
//TODO SHAPETRACKER we'll gonna love you
//Pass allocator until we have a shape tracker
pub fn transpose_lean(
    comptime T: type,
    input: *Tensor(T),
    perm: ?[]const usize,
    output: *Tensor(T),
    output_allocator: std.mem.Allocator,
) !void {
    const input_rank = input.shape.len;
    const perm_rank = if (perm) |p| p.len else input_rank; // Effective rank
    // Use pkg_allocator for temporary internal calculations
    const allocator = pkg_allocator;

    if (perm_rank == 0 and input_rank == 0) {
        // Handle scalar case (rank 0)
        if (output.size < 1) {
            // Use output_allocator to manage output tensor's memory
            output.data = try output_allocator.alloc(T, 1);
        }
        output.size = 1;
        if (input.size > 0) output.data[0] = input.data[0]; // Copy scalar value
        // Output shape handling needs clarification for scalar case
        return;
    }

    // -----------------------------
    // 1) Build the actual perm array
    // -----------------------------
    var real_perm = try allocator.alloc(usize, perm_rank);
    defer allocator.free(real_perm);

    if (perm) |p| {
        // Validate length
        if (p.len != perm_rank) {
            return error.InvalidPermutation;
        }
        if (input_rank > perm_rank) {
            return error.UnsupportedBroadcast;
        }

        // Validate that p is a valid permutation of [0..perm_rank)
        var used = try allocator.alloc(bool, perm_rank);
        defer allocator.free(used);
        @memset(used, false);

        for (p) |idx| {
            if (idx >= perm_rank) return error.InvalidPermutation;
            if (used[idx]) return error.InvalidPermutation;
            used[idx] = true;
        }
        @memcpy(real_perm, p);
    } else {
        // If no perm given, ONNX says reverse the dimension order for perm_rank
        for (0..perm_rank) |i| {
            real_perm[i] = perm_rank - 1 - i;
        }
    }

    // -----------------------------
    // 2) Create effective input shape and compute input strides
    // -----------------------------
    var effective_input_shape = try allocator.alloc(usize, perm_rank);
    defer allocator.free(effective_input_shape);

    const leading_ones = if (perm_rank > input_rank) perm_rank - input_rank else 0;
    for (0..leading_ones) |i| {
        effective_input_shape[i] = 1;
    }
    @memcpy(effective_input_shape[leading_ones..], input.shape);

    var input_strides = try allocator.alloc(usize, input_rank);
    defer allocator.free(input_strides);

    if (input_rank > 0) {
        var stride: usize = 1;
        var i: usize = input_rank;
        while (i > 0) {
            i -= 1;
            input_strides[i] = stride;
            // Use actual input shape for stride calculation
            stride *= input.shape[i];
        }
    }

    // -----------------------------
    // 3) Compute the output shape and output strides
    // -----------------------------
    var output_shape = try allocator.alloc(usize, perm_rank);
    defer allocator.free(output_shape);

    var total_size: usize = 1;
    for (0..perm_rank) |j| {
        const effective_input_dim_index = real_perm[j];
        output_shape[j] = effective_input_shape[effective_input_dim_index];
        total_size *= output_shape[j];
    }

    // Check if output tensor needs resizing or shape update
    if (output.shape.len != perm_rank) {
        // Use output_allocator to manage output tensor's shape memory
        output.shape = try output.allocator.alloc(usize, perm_rank);
    }
    // Copy calculated shape into output tensor's shape slice
    @memcpy(output.shape, output_shape);

    var output_strides = try allocator.alloc(usize, perm_rank);
    defer allocator.free(output_strides);

    if (perm_rank > 0) {
        var stride: usize = 1;
        var i: usize = perm_rank;
        while (i > 0) {
            i -= 1;
            output_strides[i] = stride;
            stride *= output.shape[i]; // Use output tensor's actual shape
        }
    }

    // -----------------------------
    // 4) Allocate output data if needed, using output_allocator
    // -----------------------------
    const size_matches = (output.data.len == total_size);

    if (size_matches) {
        // Size is correct, proceed using existing buffer.
    } else {
        // Size mismatch, reallocation needed.
        // Do not free previous buffer because it may be a static const buffer.
        if (total_size > 0) {
            output.data = try output.allocator.alloc(T, total_size);
        } else {
            output.data = &[_]T{};
        }
    }
    output.size = total_size;

    // -----------------------------
    // 5) Fill output by iterating over all output coords
    // -----------------------------
    const out_data = output.data;
    const in_data = input.data;

    if (total_size == 0) return; // Handle empty tensor case

    // Optimization: If it's just a copy (no permutation or broadcasting)
    var is_simple_copy = (perm_rank == input_rank);
    if (is_simple_copy) {
        for (0..perm_rank) |i| {
            if (real_perm[i] != i) {
                is_simple_copy = false;
                break;
            }
        }
    }
    if (is_simple_copy) {
        @memcpy(out_data, in_data[0..total_size]);
        return;
    }

    // Calculate inverse permutation: maps effective input dim -> output dim
    var inv_perm = try allocator.alloc(usize, perm_rank);
    defer allocator.free(inv_perm);
    for (0..perm_rank) |i| inv_perm[real_perm[i]] = i;

    var out_coord = try allocator.alloc(usize, perm_rank);
    defer allocator.free(out_coord);

    for (0..total_size) |flat_out_idx| {
        // Unflatten flat_out_idx -> out_coord [o0, o1, ...]
        var tmp = flat_out_idx;
        for (0..perm_rank) |d| {
            if (output_strides[d] == 0) { // Avoid division by zero for dims of size 1
                out_coord[d] = 0;
            } else {
                out_coord[d] = tmp / output_strides[d];
                tmp %= output_strides[d];
            }
        }

        // Map output coords to actual input index
        var in_idx: usize = 0;
        var current_stride_idx = input_rank - 1;
        for (0..perm_rank) |eff_in_dim_rev| { // Iterate effective dims reverse (perm_rank-1 down to 0)
            const eff_in_dim = perm_rank - 1 - eff_in_dim_rev;

            // Find the corresponding output dimension using inverse perm
            const out_dim = inv_perm[eff_in_dim];
            const coord = out_coord[out_dim];

            // Only add to in_idx if it corresponds to an actual input dimension
            if (eff_in_dim >= leading_ones) {
                if (current_stride_idx < input_rank) { // Boundary check
                    // Use the stride for the *actual* input dimension
                    in_idx += coord * input_strides[current_stride_idx];
                    if (current_stride_idx > 0) {
                        current_stride_idx -= 1;
                    } else if (current_stride_idx == 0) {
                        // Avoid wrapping below zero for usize
                        current_stride_idx = input_rank; // Signal we are done with strides
                    }
                } else {
                    // This case might indicate an issue if coord != 0 for a broadcasted dim
                }
            }
        }
        if (in_idx >= input.size) {
            return error.IndexOutOfBounds; // Or another appropriate error
        }

        // Do the copy
        out_data[flat_out_idx] = in_data[in_idx];
    }
}
