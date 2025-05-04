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

/// The Sigmoid activation function is a smooth, S-shaped function that maps any input
/// to a value between 0 and 1.
/// it can suffer from vanishing gradients, especially for large positive or negative
/// inputs, slowing down training in deep networks.
pub inline fn sigmoid(comptime T: anytype, tensor: *Tensor(T)) !Tensor(T) {
    //checks
    if (tensor.size <= 0) return TensorError.ZeroSizeTensor;

    var output_tensor = try Tensor(T).fromShape(&pkg_allocator, tensor.shape);
    errdefer output_tensor.deinit();

    try sigmoid_lean(T, tensor, &output_tensor);

    return output_tensor;
}

pub inline fn sigmoid_lean(comptime T: anytype, input_tensor: *Tensor(T), output_tensor: *Tensor(T)) !void {
    @setEvalBranchQuota(100000);
    //std.debug.print("\n[DEBUG] sigmoid_lean:", .{});
    //std.debug.print("\n  Input shape: ", .{});
    //for (input_tensor.shape) |s| std.debug.print("{d} ", .{s});

    //std.debug.print("\n  Output shape: ", .{});
    //for (output_tensor.shape) |s| std.debug.print("{d} ", .{s});

    //apply Sigmoid
    for (0..input_tensor.size) |i| {
        const input_val = input_tensor.data[i];
        output_tensor.data[i] = 1.0 / (1.0 + @exp(-input_val));
        //std.debug.print("\n  sigmoid({d:.6}) = {d:.6}", .{ input_val, output_tensor.data[i] });
    }
    //std.debug.print("\n[DEBUG] sigmoid_lean completed\n", .{});
}

pub fn sigmoid_backward(comptime T: anytype, gradient: *Tensor(T), act_forward_out: *Tensor(T)) !void {
    //checks
    if (gradient.size <= 0 or act_forward_out.size <= 0) return TensorError.ZeroSizeTensor;
    if (gradient.size != act_forward_out.size) return TensorMathError.InputTensorDifferentSize;

    //apply Sigmoid derivative: f'(x) = f(x) * (1 - f(x))
    for (0..gradient.size) |i| {
        const sigmoid_output = act_forward_out.data[i];
        gradient.data[i] *= sigmoid_output * (1 - sigmoid_output);
    }
}

pub fn get_sigmoid_output_shape(input_shape: []const usize) ![]usize {
    // Allocate and copy the input shape
    const output_shape = try pkg_allocator.alloc(usize, input_shape.len);
    errdefer pkg_allocator.free(output_shape);

    std.mem.copyForwards(usize, output_shape, input_shape);

    return output_shape;
}




/// https://onnx.ai/onnx/operators/onnx__Sigmoid.html
pub fn lowerSigmoid(
    b: *UOpBuilder,
    X_id: usize, // SSA id of input tensor X
    x_shape: []const usize, // Shape of input tensor X
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
    const id_one = r.kconst(b, 1.0); // Zero for comparison

    // ── 4. Create nested loops for each dimension of the tensor ───────
    var nelem: usize = 1;
    for (x_shape) |d| nelem *= d;

    const id_range = b.push(.RANGE, .i32, &.{}, Any{ .loop_bounds = .{ .start = 0, .end = nelem } });

    // ── 5. Create GEP operation for current element ───────────────────
    const id_gepX = b.push(.GEP, out_dtype, &.{id_viewX, id_range}, Any{ .mem_info = .{ .base = id_viewX, .offset = 0, .stride = 1 } });
    
    // ── 6. Load the input value ─────────────────────────────────────────
    const id_x = b.push(.LOAD, out_dtype, &.{id_gepX}, null);

    // ── 7. Implement Sigmoid: f(x) = 1 / (1 + exp(-x)) ────────

    const id_neg_x = b.push(.NEG, out_dtype, &.{id_x}, null);
    const id_exp_neg_x = b.push(.EXP, out_dtype, &.{id_neg_x}, null);
    const id_one_plus_exp_neg_x = b.push(.ADD, out_dtype, &.{id_one, id_exp_neg_x}, null);
    const id_result = b.push(.DIV, out_dtype, &.{id_one, id_one_plus_exp_neg_x}, null);

    
    // ── 8. Store the result to the output tensor ───────────────────────
    const id_gepY = b.push(.GEP, out_dtype, &.{id_Y, id_range}, Any{ .mem_info = .{ .base = id_Y, .offset = 0, .stride = 1 } });
    _ = b.push(.STORE, out_dtype, &.{id_gepY, id_result}, null);

    // ── 9. Close all the nested loops ───────────────────────────────────
    _ = b.push(.ENDRANGE, .bool, &.{id_range}, null);

    return id_Y; // SSA id of the produced output tensor Y
}