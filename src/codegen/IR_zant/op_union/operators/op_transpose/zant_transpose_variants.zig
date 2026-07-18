const IR_zant = @import("IR_zant");

const Tensor = IR_zant.core.tensor.Tensor;

const pkg_allocator = IR_zant.pkg_allocator.allocator;

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

    return transpose_internal(T, t, tensorShape);
}

/// Returns a Tensor self transposed. Does not modify self.
fn transpose_internal(comptime T: type, t: *Tensor(T), perms: []usize) !Tensor(T) {
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
    // Verifying correct shape
    if (tensor.shape.len != 2 and tensor.shape.len != 4) {
        return error.InputTensorsWrongShape;
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

    // Create a non-const copy of the input data using pkg_allocator
    const outData = try pkg_allocator.alloc(T, total);
    errdefer pkg_allocator.free(outData);

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

    return Tensor(T){
        .data = outData,
        .size = total,
        .shape = newShape,
        .allocator = &pkg_allocator,
    };
}
