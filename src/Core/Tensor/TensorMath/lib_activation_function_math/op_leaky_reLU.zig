const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;
const error_handler = zant.utils.error_handler;
const TensorMathError = error_handler.TensorMathError;
const TensorError = error_handler.TensorError;
const ArchitectureError = error_handler.ArchitectureError;
const Converter = zant.utils.type_converter;

const Uops = zant.uops;
const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const Any = Uops.Any;

/// Leaky ReLU is a variant of ReLU that allows a small, positive gradient when the input is negative.
/// This can help prevent the dying ReLU problem, where neurons stop learning if they get stuck in the negative side of the ReLU function.
/// The leaky parameter is a small value that determines how much the function leaks into the negative side.
/// A common value for the leaky parameter is 0.01.
/// The Leaky ReLU function is defined as:
/// f(x) = x if x > 0
/// f(x) = alpha * x if x <= 0
/// where alpha is a small constant.
/// The derivative of the Leaky ReLU function is:
/// f'(x) = 1 if x > 0
/// f'(x) = alpha if x <= 0
pub inline fn leakyReLU(comptime T: anytype, tensor: *Tensor(T), slope: T) !Tensor(T) {
    //checks
    if (tensor.size <= 0) return TensorError.ZeroSizeTensor;

    var output_tensor = try Tensor(T).fromShape(&pkg_allocator, tensor.shape);
    errdefer output_tensor.deinit();

    try lean_leakyReLU(T, tensor, slope, &output_tensor);

    return output_tensor;
}

pub inline fn lean_leakyReLU(comptime T: anytype, input_tensor: *Tensor(T), slope: T, output_tensor: *Tensor(T)) !void {
    //apply Leaky ReLU suing relu self.relu() - (-neg_slope*self).relu()
    for (0..input_tensor.size) |i| {
        if (input_tensor.data[i] <= 0) {
            output_tensor.data[i] = slope * input_tensor.data[i];
        } else {
            output_tensor.data[i] = input_tensor.data[i];
        }
    }
}

pub fn leakyReLU_backward(comptime T: anytype, gradient: *Tensor(T), act_relu_input: *Tensor(T), slope: T) !void {

    //checks
    if (gradient.size <= 0 or act_relu_input.size <= 0) return TensorError.ZeroSizeTensor;
    if (gradient.size != act_relu_input.size) return TensorMathError.InputTensorDifferentSize;

    //apply Leaky ReLU derivative: f'(x) = 1 if x > 0, slope if x <= 0
    for (0..gradient.size) |i| {
        gradient.data[i] *= if (act_relu_input.data[i] > 0) 1 else slope;
    }
}

pub fn get_leaky_relu_output_shape(input_shape: []const usize) ![]usize {
    const output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    errdefer pkg_allocator.free(output_shape);

    @memcpy(output_shape, input_shape);

    return output_shape;
}


/// https://onnx.ai/onnx/operators/onnx__LeakyRelu.html
pub fn lowerLeakyReLU(
    b: *UOpBuilder,
    X_id: usize, // SSA id of input tensor X
    x_shape: []const usize, // Shape of input tensor X
    alpha: f32, // Slope for negative values (typically small, e.g., 0.01)
    out_dtype: DType,
) usize {

    // ── Tiny helpers to reduce boilerplate ────────────────────────────
    const r = struct {
        fn rng(bi: *UOpBuilder, end: usize) usize { // RANGE 0..end-1
            return bi.push(.RANGE, .i32, &.{}, Any{ .loop_bounds = .{ .start = 0, .end = end } });
        }
        fn kconst(bi: *UOpBuilder, v: f32) usize { // CONST <v> (float)
            return bi.push(.CONST, out_dtype, &.{}, Any{ .float = v });
        }
    };

    // ── 1. Create a logical view for the input tensor ─────────────────
    const id_viewX = b.push(.VIEW, out_dtype, &.{X_id}, Any{ .view_meta = .{ .shape = x_shape } });

    // ── 2. Create output tensor with the same shape as input ──────────
    const id_Y = b.push(.DEFINE_GLOBAL, out_dtype, &.{}, Any{ .shape = x_shape });

    // ── 3. Create constants for the operation ─────────────────────────
    const id_alpha = r.kconst(b, alpha); // Alpha value for negative numbers
    const id_zero = r.kconst(b, 0.0); // Zero for comparison

    // ── 4. Create nested loops for each dimension of the tensor ───────
    var nelem: usize = 1;
    for (x_shape) |d| nelem *= d;

    const id_range = b.push(.RANGE, .i32, &.{}, Any{ .loop_bounds = .{ .start = 0, .end = nelem } });

    // ── 5. Create GEP operation for current element ───────────────────
    const id_gepX = b.push(.GEP, out_dtype, &.{id_viewX, id_range}, Any{ .mem_info = .{ .base = id_viewX, .offset = 0, .stride = 1 } });
    
    // ── 6. Load the input value ─────────────────────────────────────────
    const id_x = b.push(.LOAD, out_dtype, &.{id_gepX}, null);

    // ── 7. Implement LeakyReLU: f(x) = alpha * x if x < 0 else x ────────
    // Compare x with zero
    const id_lt = b.push(.CMPLT, .bool, &.{id_x, id_zero}, null);
    
    // Calculate alpha * x for negative inputs
    const id_mul = b.push(.MUL, out_dtype, &.{id_alpha, id_x}, null);
    
    // Select between x and alpha*x based on comparison result
    const id_result = b.push(.SELECT, out_dtype, &.{id_lt, id_mul, id_x}, null);

    // ── 8. Store the result to the output tensor ───────────────────────
    const id_gepY = b.push(.GEP, out_dtype, &.{id_Y, id_range}, Any{ .mem_info = .{ .base = id_Y, .offset = 0, .stride = 1 } });
    _ = b.push(.STORE, out_dtype, &.{id_gepY, id_result}, null);

    // ── 9. Close all the nested loops ───────────────────────────────────
    _ = b.push(.ENDRANGE, .bool, &.{id_range}, null);

    return id_Y; // SSA id of the produced output tensor Y
}