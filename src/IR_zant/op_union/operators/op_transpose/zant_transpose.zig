const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const pkg_allocator = zant.utils.allocator.allocator;

/// Returns a Tensor self transposed. Does not modify self.
/// It sobstitute init(), but defer yourTensor.deinit() is still necessary.
pub fn transpose2D(comptime T: type, t: *Tensor(T)) !Tensor(T) {
    if (t.shape.len != 2) {
        return error.InvalidDimension; // For simplicity, let's focus on 2D for now
    }

    const allocator = t.allocator;

    // Shape of the transposed tensor
    const transposed_shape: [2]usize = [_]usize{ t.shape[1], t.shape[0] };
    const tensorShape = try allocator.alloc(usize, t.shape.len);
    @memcpy(tensorShape, &transposed_shape);

    // Allocate space for transposed data
    const transposed_data = try allocator.alloc(T, t.size);

    // Perform the transposition
    for (0..t.shape[0]) |i| {
        for (0..t.shape[1]) |j| {
            // For 2D tensor, flatten the index and swap row/column positions
            const old_idx = i * t.shape[1] + j;
            const new_idx = j * t.shape[0] + i;
            transposed_data[new_idx] = t.data[old_idx];
        }
    }

    return Tensor(T){
        .data = transposed_data,
        .size = t.size,
        .shape = tensorShape,
        .allocator = allocator,
    };
}

/// Returns a Tensor self transposed.
/// OSS! Does not modify self.data!! it only changes the shape! so it is necessary to acces it trough get_flatten_index()
/// By default, it transposes the tensor to the reverse shape.
pub fn transposeDefault(comptime T: type, t: *Tensor(T)) !Tensor(T) {
    // Reverse the shape of the tensor
    const tensorShape = try t.allocator.alloc(usize, t.shape.len);
    for (0..t.shape.len) |i| {
        tensorShape[i] = t.shape.len - 1 - i;
    }

    return transpose(T, t, tensorShape);
}

/// Returns a Tensor self transposed. Does not modify self.
fn transpose(comptime T: type, t: *Tensor(T), perms: []usize) !Tensor(T) {
    defer t.allocator.free(perms);
    const num_dims = t.shape.len;
    if (perms.len != num_dims) {
        return error.InvalidDimension;
    }

    // Check that the permutation is valid
    var bitmap = try t.allocator.alloc(bool, perms.len);
    defer t.allocator.free(bitmap);

    for (perms) |perm| {
        if (perm >= perms.len) {
            return error.InvalidPermutation;
        }
        if (bitmap[perm] == true) {
            return error.InvalidPermutation;
        }
        bitmap[perm] = true;
    }

    // Allocate space for the new shape
    const new_shape = try t.allocator.alloc(usize, num_dims);
    for (0..num_dims) |i| {
        new_shape[i] = t.shape[perms[i]];
    }
    defer t.allocator.free(new_shape);

    // Create the new tensor
    var new_tensor = try Tensor(T).fromShape(t.allocator, new_shape);

    // Copy data to the new tensor
    for (0..t.size) |i| {
        new_tensor.data[i] = t.data[i];
    }

    return new_tensor;
}

/// Given a 4D tensor it returns the tensor with the last 2 dimensions transposed. Operates on both data and shape, does not modify self, used by gemm.
pub fn transposeLastTwo(comptime T: anytype, tensor: *const Tensor(T)) !Tensor(T) {
    //std.log.debug("\n[DEBUG] transposeLastTwo:", .{});
    //std.log.debug("\n  Input tensor shape: ", .{});
    //for (tensor.shape) |s| std.log.debug("{d} ", .{s});

    // Verifying correct shape
    if (tensor.shape.len != 2 and tensor.shape.len != 4) {
        //std.log.debug("\n  Error: Expected 2D or 4D tensor, got {d}D", .{tensor.shape.len});
        return TensorMathError.InputTensorsWrongShape;
    }

    var rows: usize = undefined;
    var cols: usize = undefined;
    var total: usize = undefined;
    var newShape: []usize = undefined;

    if (tensor.shape.len == 2) {
        rows = tensor.shape[0];
        cols = tensor.shape[1];
        total = rows * cols;
        newShape = try pkg_allocator.alloc(usize, 2);
        errdefer pkg_allocator.free(newShape);
        newShape[0] = cols;
        newShape[1] = rows;
    } else { // 4D case
        const batch = tensor.shape[0];
        const channel = tensor.shape[1];
        rows = tensor.shape[2];
        cols = tensor.shape[3];
        total = batch * channel * rows * cols;
        newShape = try pkg_allocator.alloc(usize, 4);
        errdefer pkg_allocator.free(newShape);
        newShape[0] = batch;
        newShape[1] = channel;
        newShape[2] = cols;
        newShape[3] = rows;
    }

    //std.log.debug("\n  Rows: {d}, Cols: {d}, Total: {d}", .{ rows, cols, total });
    //std.log.debug("\n  New shape: ", .{});
    //for (newShape) |s| std.log.debug("{d} ", .{s});

    // Create a non-const copy of the input data using pkg_allocator
    const outData = try pkg_allocator.alloc(T, total);
    errdefer pkg_allocator.free(outData);

    //std.log.debug("\n  Transposing data...", .{});

    if (tensor.shape.len == 2) {
        // Simple 2D transpose - Fixed indexing
        for (0..rows) |i| {
            for (0..cols) |j| {
                outData[j * rows + i] = tensor.data[i * cols + j];
            }
        }
    } else {
        // 4D transpose of last two dimensions
        const batch = tensor.shape[0];
        const channel = tensor.shape[1];
        for (0..batch) |b| {
            for (0..channel) |c| {
                for (0..rows) |i| {
                    for (0..cols) |j| {
                        const index_in = (((b * channel) + c) * rows + i) * cols + j;
                        const index_out = (((b * channel) + c) * cols + j) * rows + i;
                        outData[index_out] = tensor.data[index_in];
                    }
                }
            }
        }
    }

    //std.log.debug("\n  Transpose complete", .{});

    return Tensor(T){
        .data = outData,
        .size = total,
        .shape = newShape,
        .allocator = &pkg_allocator,
    };
}

/// Implements the ONNX transpose operator (version 21)
/// Transposes the input tensor similar to numpy.transpose.
/// If perm is not provided, reverses the dimensions.
/// If perm is provided, permutes the axes according to the values given.
pub fn transpose_onnx(comptime T: type, input: *Tensor(T), perm: ?[]const usize) !Tensor(T) {
    var output = try Tensor(T).fromShape(&pkg_allocator, input.shape);
    errdefer output.deinit();

    try transpose_onnx_lean(T, input, perm, &output);

    return output;
}

/// Lean version of transpose_onnx that operates on an existing output tensor
/// Handles implicit broadcasting if the permutation length is greater than the input rank.
//TODO SHAPETRACKER we'll gonna love you
//Pass allocator until we have a shape tracker
pub fn transpose_onnx_lean(
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
            //std.log.debug("ERROR: transpose_onnx_lean perm.len = {}, perm_rank = {}. Should be equal.\\n", .{ p.len, perm_rank });
            return error.InvalidPermutation;
        }
        if (input_rank > perm_rank) {
            //std.log.debug("ERROR: transpose_onnx_lean input_rank ({}) > perm_rank ({}) not supported\\n", .{ input_rank, perm_rank });
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
    //    (Uses 'allocator' for temporary effective_input_shape)
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
    //    (Uses 'allocator' for temporary output_shape and output_strides)
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
    //    Map output coords -> effective input coords -> actual input index
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
                    // For now, assume coord is 0 if eff_in_dim < leading_ones
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

/// Calculate the output shape for an ONNX transpose operation without performing the transpose
/// Handles implicit broadcasting if the permutation length is greater than the input rank.
pub fn get_transpose_output_shape(input_shape: []const usize, perm: ?[]const usize) ![]usize {
    const input_rank = input_shape.len;
    const perm_rank = if (perm) |p| p.len else input_rank; // Effective rank

    // Allocate space for the permutation array
    var real_perm = try pkg_allocator.alloc(usize, perm_rank);
    defer pkg_allocator.free(real_perm);

    // Validate perm if provided and build real_perm
    if (perm) |p| {
        if (p.len != perm_rank) {
            // This case should technically not happen if perm_rank is derived from p.len, but good practice
            std.log.warn("ERROR: perm length mismatch internal logic! p.len={} perm_rank={}\\n", .{ p.len, perm_rank });
            return error.InvalidPermutation;
        }

        // Validate that p is a valid permutation of [0..perm_rank)
        var used = try pkg_allocator.alloc(bool, perm_rank);
        defer pkg_allocator.free(used);
        @memset(used, false);

        for (p) |idx| {
            if (idx >= perm_rank) return error.InvalidPermutation;
            if (used[idx]) return error.InvalidPermutation;
            used[idx] = true;
        }
        @memcpy(real_perm, p);
    } else {
        // If no perm given, reverse the dimension order for perm_rank
        for (0..perm_rank) |i| {
            real_perm[i] = perm_rank - 1 - i;
        }
    }

    // Create effective input shape (with leading 1s for broadcasting)
    var effective_input_shape = try pkg_allocator.alloc(usize, perm_rank);
    defer pkg_allocator.free(effective_input_shape);

    const leading_ones = if (perm_rank > input_rank) perm_rank - input_rank else 0;
    for (0..leading_ones) |i| {
        effective_input_shape[i] = 1;
    }
    @memcpy(effective_input_shape[leading_ones..], input_shape);

    // Allocate output shape array
    var output_shape = try pkg_allocator.alloc(usize, perm_rank);
    errdefer pkg_allocator.free(output_shape);

    // Fill output shape based on permuting the effective_input_shape
    for (0..perm_rank) |i| {
        const input_dim_index = real_perm[i];
        if (input_dim_index >= perm_rank) { // Sanity check
            std.log.warn("ERROR: real_perm[{}]={} is out of bounds for perm_rank={}\\n", .{ i, input_dim_index, perm_rank });
            return error.InvalidPermutation;
        }
        output_shape[i] = effective_input_shape[input_dim_index];
    }

    return output_shape;
}
