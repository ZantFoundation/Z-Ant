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
        .owns_memory = true,
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
    const new_tensor = try Tensor(T).fromShape(t.allocator, new_shape);

    // Copy data to the new tensor
    for (0..t.size) |i| {
        new_tensor.data[i] = t.data[i];
    }

    return new_tensor;
}

/// Given a 4D tensor it returns the tensor with the last 2 dimensions transposed. Operates on both data and shape, does not modify self, used by gemm.
pub fn transposeLastTwo(comptime T: anytype, tensor: *const Tensor(T)) !Tensor(T) {
    //std.debug.print("\n[DEBUG] transposeLastTwo:", .{});
    //std.debug.print("\n  Input tensor shape: ", .{});
    //for (tensor.shape) |s| std.debug.print("{d} ", .{s});

    // Verifying correct shape
    if (tensor.shape.len != 2 and tensor.shape.len != 4) {
        //std.debug.print("\n  Error: Expected 2D or 4D tensor, got {d}D", .{tensor.shape.len});
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

    //std.debug.print("\n  Rows: {d}, Cols: {d}, Total: {d}", .{ rows, cols, total });
    //std.debug.print("\n  New shape: ", .{});
    //for (newShape) |s| std.debug.print("{d} ", .{s});

    // Create a non-const copy of the input data using pkg_allocator
    const outData = try pkg_allocator.alloc(T, total);
    errdefer pkg_allocator.free(outData);

    //std.debug.print("\n  Transposing data...", .{});

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

    //std.debug.print("\n  Transpose complete", .{});

    return Tensor(T){
        .data = outData,
        .size = total,
        .shape = newShape,
        .allocator = &pkg_allocator,
        .owns_memory = true,
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
//TODO SHAPETRACKER we'll gonna love you
pub fn transpose_onnx_lean(
    comptime T: type,
    input: *Tensor(T),
    perm: ?[]const usize,
    output: *Tensor(T),
) !void {
    // Validate rank
    const rank = input.shape.len;
    if (output.shape.len != rank) {
        return error.InvalidRank; // or however you handle shape mismatch
    }

    // -----------------------------
    // 1) Build the actual perm array
    // -----------------------------
    var real_perm = try pkg_allocator.alloc(usize, rank);
    defer pkg_allocator.free(real_perm);

    if (perm) |p| {
        // Validate length
        if (p.len != rank) return error.InvalidPermutation;

        // Validate that p is a valid permutation of [0..rank)
        var used = try pkg_allocator.alloc(bool, rank);
        defer pkg_allocator.free(used);
        @memset(used, false);

        for (p) |idx| {
            if (idx >= rank) return error.InvalidPermutation;
            if (used[idx]) return error.InvalidPermutation;
            used[idx] = true;
        }
        // Copy into real_perm
        for (0..rank) |i| {
            real_perm[i] = p[i];
        }
    } else {
        // If no perm given, ONNX says reverse the dimension order
        for (0..rank) |i| {
            real_perm[i] = rank - 1 - i;
        }
    }

    // -----------------------------
    // 2) Compute input strides
    // -----------------------------
    var input_strides = try pkg_allocator.alloc(usize, rank);
    defer pkg_allocator.free(input_strides);

    var stride: usize = 1;
    var i: usize = rank;
    while (i > 0) {
        i -= 1;
        input_strides[i] = stride;
        stride *= input.shape[i];
    }

    // -----------------------------
    // 3) Compute the output shape
    //    and output strides (row-major)
    // -----------------------------
    var output_shape = try pkg_allocator.alloc(usize, rank);
    defer pkg_allocator.free(output_shape);

    // shape comes from perm
    for (0..rank) |j| {
        output_shape[j] = input.shape[real_perm[j]];
    }

    var output_strides = try pkg_allocator.alloc(usize, rank);
    defer pkg_allocator.free(output_strides);

    stride = 1;
    i = rank;
    while (i > 0) {
        i -= 1;
        output_strides[i] = stride;
        stride *= output_shape[i];
    }

    // -----------------------------
    // 4) Allocate output data if needed
    // -----------------------------
    const total_size = stride; // product of all dims
    if (output.data.len != total_size) {
        // free old data if needed
        if (output.data.len > 0) pkg_allocator.free(output.data);
        output.data = try pkg_allocator.alloc(T, total_size);
    }
    output.size = total_size;

    // Copy shape into output
    @memcpy(output.shape, output_shape);

    // -----------------------------
    // 5) Fill output by iterating over all output coords
    // -----------------------------
    // We'll do a simple nested‐index iteration by flattening the output coordinate.
    // Then we un‐flatten to get [o0, o1, ..., o_{rank-1}].
    const out_data = output.data;
    const in_data = input.data;

    for (0..total_size) |flat_out_idx| {
        // Convert flat_out_idx -> array of indices in [o0, o1, ...]
        var tmp = flat_out_idx;
        var out_coord = try pkg_allocator.alloc(usize, rank);
        defer pkg_allocator.free(out_coord);

        // Unflatten in row-major order
        for (0..rank) |d| {
            out_coord[d] = tmp / output_strides[d];
            tmp %= output_strides[d];
        }

        // Now map output coords back to input coords via perm
        var in_idx: usize = 0;
        for (0..rank) |d| {
            const input_dim_index = real_perm[d]; // which input dimension
            in_idx += out_coord[d] * input_strides[input_dim_index];
        }

        // Do the copy
        out_data[flat_out_idx] = in_data[in_idx];
    }
}

/// Calculate the output shape for an ONNX transpose operation without performing the transpose
pub fn get_transpose_output_shape(input_shape: []const usize, perm: ?[]const usize) ![]usize {
    const rank = input_shape.len;

    // Validate perm if provided
    if (perm) |p| {
        if (p.len != rank) return error.InvalidPermutation;

        // Validate that p is a valid permutation of [0..rank)
        var used = try pkg_allocator.alloc(bool, rank);
        defer pkg_allocator.free(used);
        @memset(used, false);

        for (p) |idx| {
            if (idx >= rank) return error.InvalidPermutation;
            if (used[idx]) return error.InvalidPermutation;
            used[idx] = true;
        }
    }

    // Allocate output shape array
    var output_shape = try pkg_allocator.alloc(usize, rank);
    errdefer pkg_allocator.free(output_shape);

    // Fill output shape based on permutation
    if (perm) |p| {
        for (0..rank) |i| {
            output_shape[i] = input_shape[p[i]];
        }
    } else {
        // If no perm given, reverse the dimension order
        for (0..rank) |i| {
            output_shape[i] = input_shape[rank - 1 - i];
        }
    }

    return output_shape;
}
