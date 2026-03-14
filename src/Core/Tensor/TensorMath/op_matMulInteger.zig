const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const matmul = @import("op_mat_mul.zig");

fn isSupportedIntegerType(comptime T: type) bool {
    return T == i8 or T == u8;
}

fn broadcastZeroPointValue(zero_point: anytype, target_shape: []const usize, flat_index: usize) !i32 {
    const zp = switch (@typeInfo(@TypeOf(zero_point))) {
        .null => return 0,
        .optional => zero_point orelse return 0,
        .pointer => zero_point,
        else => @compileError("Unsupported zero point argument type"),
    };

    if (zp.size == 1) {
        return @as(i32, zp.data[0]);
    }

    if (zp.shape.len > target_shape.len) {
        return TensorMathError.InvalidDimensions;
    }

    var remaining_index = flat_index;
    var zp_index: usize = 0;
    var zp_stride: usize = 1;
    var target_dim_rev = target_shape.len;
    var zp_dim_rev = zp.shape.len;

    while (target_dim_rev > 0) : (target_dim_rev -= 1) {
        const target_dim = target_dim_rev - 1;
        const coord = remaining_index % target_shape[target_dim];
        remaining_index /= target_shape[target_dim];

        if (zp_dim_rev == 0) continue;

        const zp_dim = zp_dim_rev - 1;
        const zp_extent = zp.shape[zp_dim];
        const target_extent = target_shape[target_dim];

        if (zp_extent != 1 and zp_extent != target_extent) {
            return TensorMathError.InvalidDimensions;
        }

        const zp_coord = if (zp_extent == 1) 0 else coord;
        zp_index += zp_coord * zp_stride;
        zp_stride *= zp_extent;
        zp_dim_rev -= 1;
    }

    while (zp_dim_rev > 0) : (zp_dim_rev -= 1) {
        if (zp.shape[zp_dim_rev - 1] != 1) {
            return TensorMathError.InvalidDimensions;
        }
    }

    return @as(i32, zp.data[zp_index]);
}

/// ONNX MatMulInteger operator.
/// https://onnx.ai/onnx/operators/onnx__MatMulInteger.html
pub fn matMulInteger(
    comptime AType: type,
    comptime BType: type,
    A: *const Tensor(AType),
    B: *const Tensor(BType),
    a_zero_point: ?*const Tensor(AType),
    b_zero_point: ?*const Tensor(BType),
) !Tensor(i32) {
    if (!isSupportedIntegerType(AType) or !isSupportedIntegerType(BType)) {
        return TensorMathError.InvalidDataType;
    }

    if (A.shape.len != B.shape.len) {
        return TensorMathError.InputTensorDifferentShape;
    }

    const output_shape = try matmul.get_mat_mul_output_shape(A.shape, B.shape);
    defer pkg_allocator.free(output_shape);

    var output = try Tensor(i32).fromShape(&pkg_allocator, output_shape);
    errdefer output.deinit();

    try matMulInteger_lean(A, B, a_zero_point, b_zero_point, &output);
    return output;
}

pub fn matMulInteger_lean(
    A: anytype,
    B: anytype,
    a_zero_point: anytype,
    b_zero_point: anytype,
    output: anytype,
) !void {
    const AType = @TypeOf(A.data[0]);
    const BType = @TypeOf(B.data[0]);
    const OutputType = @TypeOf(output.data[0]);

    if (!isSupportedIntegerType(AType) or !isSupportedIntegerType(BType) or OutputType != i32) {
        return TensorMathError.InvalidDataType;
    }

    if (A.shape.len != B.shape.len) {
        return TensorMathError.InputTensorDifferentShape;
    }

    const dim_num = A.shape.len;
    if (dim_num == 0) {
        return TensorMathError.InvalidDimensions;
    }

    if (dim_num == 1) {
        if (A.shape[0] != B.shape[0]) {
            return TensorMathError.InputTensorsWrongShape;
        }
        if (output.shape.len != 1 or output.shape[0] != 1) {
            return TensorMathError.OutputTensorWrongShape;
        }

        var acc: i32 = 0;
        for (0..A.shape[0]) |k| {
            const a_zp = try broadcastZeroPointValue(a_zero_point, A.shape, k);
            const b_zp = try broadcastZeroPointValue(b_zero_point, B.shape, k);
            const a_val = @as(i32, A.data[k]) - a_zp;
            const b_val = @as(i32, B.data[k]) - b_zp;
            acc += a_val * b_val;
        }
        output.data[0] = acc;
        return;
    }

    if (A.shape[dim_num - 1] != B.shape[dim_num - 2]) {
        return TensorMathError.InputTensorsWrongShape;
    }

    const expected_shape = try matmul.get_mat_mul_output_shape(A.shape, B.shape);
    defer pkg_allocator.free(expected_shape);
    if (!std.mem.eql(usize, expected_shape, output.shape)) {
        return TensorMathError.OutputTensorWrongShape;
    }

    const M = A.shape[dim_num - 2];
    const K = A.shape[dim_num - 1];
    const N = B.shape[dim_num - 1];

    var batch_size: usize = 1;
    for (0..dim_num - 2) |i| {
        if (A.shape[i] != B.shape[i]) {
            return TensorMathError.InputTensorDifferentShape;
        }
        batch_size *= A.shape[i];
    }

    const a_batch_stride = M * K;
    const b_batch_stride = K * N;
    const out_batch_stride = M * N;

    for (0..batch_size) |batch| {
        const a_batch_offset = batch * a_batch_stride;
        const b_batch_offset = batch * b_batch_stride;
        const out_batch_offset = batch * out_batch_stride;

        for (0..M) |i| {
            for (0..N) |j| {
                var acc: i32 = 0;
                for (0..K) |k| {
                    const a_idx = a_batch_offset + i * K + k;
                    const b_idx = b_batch_offset + k * N + j;

                    const a_zp = try broadcastZeroPointValue(a_zero_point, A.shape, a_idx);
                    const b_zp = try broadcastZeroPointValue(b_zero_point, B.shape, b_idx);
                    const a_val = @as(i32, A.data[a_idx]) - a_zp;
                    const b_val = @as(i32, B.data[b_idx]) - b_zp;
                    acc += a_val * b_val;
                }
                output.data[out_batch_offset + i * N + j] = acc;
            }
        }
    }
}
