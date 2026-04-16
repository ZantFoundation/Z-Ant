const std = @import("std");
const IR_zant = @import("IR_zant");

const Tensor = IR_zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = IR_zant.pkg_allocator.allocator;

const utils_maxPool = @import("utils_maxPool.zig");
const get_onnx_maxpool_output_shape = utils_maxPool.get_max_pool_output_shape;

const pooling = @import("../op_utils/zant_pooling.zig");
pub const PoolingType = pooling.PoolingType;
pub const AutoPadType = pooling.AutoPadType;

pub fn max_pool(
    comptime T: type,
    input: *Tensor(T),
    kernel_shape: []const usize,
    strides: []const usize,
    dilations: []const usize,
    pads: []const usize,
    auto_pad: AutoPadType,
    ceil_mode: bool,
) !struct { output: Tensor(T), used_input: Tensor(u8) } {
    // Calculate output shape
    const output_shape = try get_onnx_maxpool_output_shape(
        input.shape,
        kernel_shape,
        strides,
        dilations,
        pads,
        auto_pad,
        ceil_mode,
    );
    defer pkg_allocator.free(output_shape);

    var output = try Tensor(T).fromShape(&pkg_allocator, output_shape);
    errdefer output.deinit();

    var used_input = try Tensor(u8).fromShape(&pkg_allocator, input.shape);
    errdefer used_input.deinit();

    try max_pool_lean(
        T,
        input,
        &output,
        kernel_shape,
        strides,
        dilations,
        pads,
        auto_pad,
    );

    return .{ .output = output, .used_input = used_input };
}

/// ONNX-compliant MaxPool implementation
pub fn max_pool_lean(
    comptime T: type,
    input: *Tensor(T),
    output: *Tensor(T),
    kernel_shape: []const usize,
    strides: []const usize,
    dilations: []const usize,
    pads: []const usize,
    auto_pad: AutoPadType,
) !void {
    const input_rank = input.shape.len;
    const output_rank = output.shape.len;
    if (input_rank != output_rank) return error.InputOutputDifferentRank;

    // Validate minimum rank (must be at least 3D: [N, C, spatial_dims...])
    if (input_rank < 3) return error.InvalidInputRank;

    const spatial_dims = input_rank - 2; // Number of spatial dimensions
    if (kernel_shape.len != spatial_dims) return error.KernelShapeMismatch;
    if (strides.len != spatial_dims) return error.StridesMismatch;
    if (dilations.len != spatial_dims) return error.DilationsMismatch;

    const batch_size = input.shape[0];
    const channels = input.shape[1];

    // Extract spatial dimensions
    var input_spatial = try pkg_allocator.alloc(usize, spatial_dims);
    defer pkg_allocator.free(input_spatial);
    var output_spatial = try pkg_allocator.alloc(usize, spatial_dims);
    defer pkg_allocator.free(output_spatial);

    for (0..spatial_dims) |i| {
        input_spatial[i] = input.shape[2 + i];
        output_spatial[i] = output.shape[2 + i];
    }

    // Calculate effective padding
    var effective_pads = try pkg_allocator.alloc(usize, spatial_dims * 2);
    defer pkg_allocator.free(effective_pads);

    switch (auto_pad) {
        .NOTSET => {
            // Use provided pads, ensuring we have enough values
            for (0..spatial_dims * 2) |i| {
                effective_pads[i] = if (i < pads.len) pads[i] else 0;
            }
        },
        .VALID => {
            // No padding
            @memset(effective_pads, 0);
        },
        .SAME_UPPER, .SAME_LOWER => {
            // Calculate padding to maintain spatial dimensions
            for (0..spatial_dims) |i| {
                const required_input = (output_spatial[i] - 1) * strides[i] +
                    ((kernel_shape[i] - 1) * dilations[i] + 1);

                const total_pad = if (required_input > input_spatial[i])
                    required_input - input_spatial[i]
                else
                    0;

                if (auto_pad == .SAME_UPPER) {
                    effective_pads[i] = total_pad / 2; // pad_begin
                    effective_pads[i + spatial_dims] = total_pad - effective_pads[i]; // pad_end
                } else {
                    effective_pads[i + spatial_dims] = total_pad / 2; // pad_end
                    effective_pads[i] = total_pad - effective_pads[i + spatial_dims]; // pad_begin
                }
            }
        },
    }

    // Calculate input and output strides for efficient indexing
    var input_strides = try pkg_allocator.alloc(usize, input_rank);
    defer pkg_allocator.free(input_strides);
    var output_strides = try pkg_allocator.alloc(usize, output_rank);
    defer pkg_allocator.free(output_strides);

    input_strides[input_rank - 1] = 1;
    output_strides[output_rank - 1] = 1;

    var i: usize = input_rank - 1;
    while (i > 0) {
        i -= 1;
        input_strides[i] = input_strides[i + 1] * input.shape[i + 1];
        output_strides[i] = output_strides[i + 1] * output.shape[i + 1];
    }

    // Main pooling loop
    for (0..batch_size) |n| {
        for (0..channels) |c| {
            const base_input_idx = n * input_strides[0] + c * input_strides[1];
            const base_output_idx = n * output_strides[0] + c * output_strides[1];

            // Iterate through all output spatial positions
            const output_coords = try pkg_allocator.alloc(usize, spatial_dims);
            defer pkg_allocator.free(output_coords);

            try poolSpatialRecursive(
                T,
                input,
                output,
                kernel_shape,
                strides,
                dilations,
                effective_pads,
                input_spatial,
                output_spatial,
                input_strides[2..],
                output_strides[2..],
                base_input_idx,
                base_output_idx,
                output_coords,
                0,
            );
        }
    }
}

/// Recursive function to handle N-dimensional spatial pooling
fn poolSpatialRecursive(
    comptime T: type,
    input: *Tensor(T),
    output: *Tensor(T),
    kernel_shape: []const usize,
    strides: []const usize,
    dilations: []const usize,
    pads: []const usize,
    input_spatial: []const usize,
    output_spatial: []const usize,
    input_strides: []const usize,
    output_strides: []const usize,
    base_input_idx: usize,
    base_output_idx: usize,
    output_coords: []usize,
    dim: usize,
) !void {
    if (dim == output_spatial.len) {
        // Base case: perform pooling for this spatial position
        var max_val = -std.math.inf(T);

        // Generate all kernel positions
        const kernel_coords = try pkg_allocator.alloc(usize, kernel_shape.len);
        defer pkg_allocator.free(kernel_coords);

        try poolKernelRecursive(
            T,
            input,
            kernel_shape,
            strides,
            dilations,
            pads,
            input_spatial,
            input_strides,
            base_input_idx,
            output_coords,
            kernel_coords,
            0,
            &max_val,
        );

        // Calculate output index
        var output_idx = base_output_idx;
        for (0..output_spatial.len) |i| {
            output_idx += output_coords[i] * output_strides[i];
        }

        output.data[output_idx] = max_val;
        return;
    }

    // Recursive case: iterate through this dimension
    for (0..output_spatial[dim]) |pos| {
        output_coords[dim] = pos;
        try poolSpatialRecursive(
            T,
            input,
            output,
            kernel_shape,
            strides,
            dilations,
            pads,
            input_spatial,
            output_spatial,
            input_strides,
            output_strides,
            base_input_idx,
            base_output_idx,
            output_coords,
            dim + 1,
        );
    }
}

/// Recursive function to iterate through kernel positions
fn poolKernelRecursive(
    comptime T: type,
    input: *Tensor(T),
    kernel_shape: []const usize,
    strides: []const usize,
    dilations: []const usize,
    pads: []const usize,
    input_spatial: []const usize,
    input_strides: []const usize,
    base_input_idx: usize,
    output_coords: []const usize,
    kernel_coords: []usize,
    dim: usize,
    max_val: *T,
) !void {
    if (dim == kernel_shape.len) {
        // Base case: check this kernel position
        var input_idx = base_input_idx;
        var valid = true;

        for (0..kernel_shape.len) |i| {
            const output_pos = output_coords[i];
            const kernel_pos = kernel_coords[i];

            const input_pos_signed = @as(isize, @intCast(output_pos * strides[i] + kernel_pos * dilations[i])) -
                @as(isize, @intCast(pads[i]));

            if (input_pos_signed < 0 or input_pos_signed >= @as(isize, @intCast(input_spatial[i]))) {
                valid = false;
                break;
            }

            const input_pos = @as(usize, @intCast(input_pos_signed));
            input_idx += input_pos * input_strides[i];
        }

        if (valid) {
            const val = input.data[input_idx];
            max_val.* = @max(max_val.*, val);
        }
        return;
    }

    // Recursive case: iterate through this kernel dimension
    for (0..kernel_shape[dim]) |pos| {
        kernel_coords[dim] = pos;
        try poolKernelRecursive(
            T,
            input,
            kernel_shape,
            strides,
            dilations,
            pads,
            input_spatial,
            input_strides,
            base_input_idx,
            output_coords,
            kernel_coords,
            dim + 1,
            max_val,
        );
    }
}
