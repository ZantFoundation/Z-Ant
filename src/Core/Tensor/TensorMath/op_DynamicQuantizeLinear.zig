const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const pkg_allocator = zant.utils.allocator.allocator;

const Q_MIN_U8: f32 = 0.0;
const Q_MAX_U8: f32 = 255.0;

/// Get output shapes for DynamicQuantizeLinear operation
fn get_dynamicQuantizeLinear_output_shape(input_shape: []const usize) ![][]usize {
    const allocator = pkg_allocator;

    // DynamicQuantizeLinear produces 3 outputs:
    // 1. quantized tensor (same shape as input)
    // 2. scale (scalar)
    // 3. zero_point (scalar)

    const output_shapes = try allocator.alloc([]usize, 3);

    // Output 0: quantized tensor - same shape as input
    output_shapes[0] = try allocator.dupe(usize, input_shape);

    // Output 1: scale - scalar (empty shape)
    output_shapes[1] = try allocator.alloc(usize, 0);

    // Output 2: zero_point - scalar (empty shape)
    output_shapes[2] = try allocator.alloc(usize, 0);

    return output_shapes;
}

/// DynamicQuantizeLinear: Computes scale, zero point, and quantizes FP32 input to UINT8.
/// Returns an array containing [quantized_tensor, scale_tensor, zero_point_tensor].
/// The caller must free the returned array and its tensors.
pub fn dynamicQuantizeLinear(x: *Tensor(f32)) ![]*Tensor(anyopaque) {
    // 1. Get output shapes
    const output_shapes = try get_dynamicQuantizeLinear_output_shape(x.shape);
    defer {
        for (output_shapes) |shape| {
            if (shape.len > 0) { // Don't free potentially static empty shapes
                x.allocator.free(shape);
            }
        }
        x.allocator.free(output_shapes);
    }

    // 2. Allocate output tensors
    // We need 3 outputs: y (u8), y_scale (f32), y_zero_point (u8)
    const y_data = try x.allocator.alloc(u8, Tensor(u8).calculateSize(output_shapes[0]));
    errdefer x.allocator.free(y_data);
    var y = Tensor(u8){
        .allocator = x.allocator,
        .shape = &[_]usize{}, // Will be assigned later
        .data = y_data,
        .size = y_data.len,
    };

    const y_scale_data = try x.allocator.alloc(f32, Tensor(f32).calculateSize(output_shapes[1]));
    errdefer x.allocator.free(y_scale_data);
    var y_scale = Tensor(f32){
        .allocator = x.allocator,
        .shape = &[_]usize{}, // Will be assigned later
        .data = y_scale_data,
        .size = y_scale_data.len,
    };

    const y_zero_point_data = try x.allocator.alloc(u8, Tensor(u8).calculateSize(output_shapes[2]));
    errdefer x.allocator.free(y_zero_point_data);
    var y_zero_point = Tensor(u8){
        .allocator = x.allocator,
        .shape = &[_]usize{}, // Will be assigned later
        .data = y_zero_point_data,
        .size = y_zero_point_data.len,
    };

    // 3. Call the lean implementation
    try dynamicQuantizeLinear_lean(x, &y, &y_scale, &y_zero_point);

    // 4. Package results
    const results = try x.allocator.alloc(*Tensor(anyopaque), 3);
    errdefer x.allocator.free(results); // Free array if packing fails

    results[0] = @ptrCast(&y);
    results[1] = @ptrCast(&y_scale);
    results[2] = @ptrCast(&y_zero_point);

    // Transfer ownership of shapes (allocated in get_dynamicQuantizeLinear_output_shape)
    // to the output tensors. The deferred free in this function won't run.
    y.shape = output_shapes[0];
    y_scale.shape = output_shapes[1];
    y_zero_point.shape = output_shapes[2];

    // Prevent the deferred free of shapes now that ownership is transferred
    output_shapes[0] = &.{};
    output_shapes[1] = &.{};
    output_shapes[2] = &.{};

    return results;
}

pub fn dynamicQuantizeLinear_lean(
    x: *const Tensor(f32),
    y: *Tensor(u8),
    y_scale: *Tensor(f32),
    y_zero_point: *Tensor(u8),
) !void {
    // Anti-aliasing difensivo: y non deve condividere memoria con y_scale/zp
    if (@intFromPtr(y.data.ptr) == @intFromPtr(y_zero_point.data.ptr)) {
        return error.InvalidOutputAlias; // meglio fallire che corrompere i dati
    }
    if (@intFromPtr(y.data.ptr) == @intFromPtr(@as([*]u8, @ptrCast(y_scale.data.ptr)))) {
        return error.InvalidOutputAlias;
    }

    if (x.size == 0) {
        y_scale.data[0] = 1.0;
        y_zero_point.data[0] = 0;
        return;
    }

    // 1) min/max dell’input
    var xmin: f32 = x.data[0];
    var xmax: f32 = x.data[0];
    for (x.data) |v| {
        xmin = @min(xmin, v);
        xmax = @max(xmax, v);
    }

    // 2) Aggiusta il range per includere 0 (ONNX)
    const xmin_adj: f32 = @min(xmin, 0.0);
    const xmax_adj: f32 = @max(xmax, 0.0);

    // 3) Scala
    const range: f32 = xmax_adj - xmin_adj;
    const scale: f32 = if (range == 0.0) 1.0 else range / Q_MAX_U8;

    // 4) Zero-point (qmin = 0)
    const initial_zp_fp: f32 = if (scale == 0.0) 0.0 else (-xmin_adj) / scale;
    const clipped_zp_fp: f32 = std.math.clamp(initial_zp_fp, Q_MIN_U8, Q_MAX_U8);
    const rounded_zp_fp: f32 = std.math.round(clipped_zp_fp); // ties-to-even
    const zp: u8 = @as(u8, @intFromFloat(rounded_zp_fp));

    // 5) Quantizza x -> y (QuantizeLinear)
    const zp_f: f32 = @as(f32, @floatFromInt(zp));
    var i: usize = 0;
    while (i < x.size) : (i += 1) {
        const xs: f32 = if (scale == 0.0) 0.0 else x.data[i] / scale;
        const qf: f32 = std.math.round(xs + zp_f); // ties-to-even
        const qc: f32 = std.math.clamp(qf, Q_MIN_U8, Q_MAX_U8); // saturazione [0,255]
        y.data[i] = @as(u8, @intFromFloat(qc));
    }

    // 6) Scrivi scale e zero-point
    y_scale.data[0] = scale;
    y_zero_point.data[0] = zp;
}
