const std = @import("std");
const zant = @import("zant");

const Tensor = zant.core.tensor.Tensor;
const tensor_math = zant.core.tensor.math_standard;

test "qlinearconv embedded rounding matches float path for negative accumulations" {
    const allocator = std.testing.allocator;

    var input_shape = [_]usize{ 1, 1, 1, 1 };
    var input_data: [1][1][1][1]i8 = .{ .{ .{ .{ 1 } } } };
    var input = try Tensor(i8).fromArray(&allocator, &input_data, &input_shape);
    defer input.deinit();

    var weight_shape = [_]usize{ 1, 1, 1, 1 };
    var weight_data: [1][1][1][1]i8 = .{ .{ .{ .{ -1 } } } };
    var weight = try Tensor(i8).fromArray(&allocator, &weight_data, &weight_shape);
    defer weight.deinit();

    var output_shape = [_]usize{ 1, 1, 1, 1 };
    var embedded_output = try Tensor(i8).fromShape(&allocator, &output_shape);
    defer embedded_output.deinit();
    var reference_output = try Tensor(i8).fromShape(&allocator, &output_shape);
    defer reference_output.deinit();

    var scalar_shape = [_]usize{ 1 };

    var x_scale_data: [1]f32 = .{ 1.0 };
    var x_scale = try Tensor(f32).fromArray(&allocator, &x_scale_data, &scalar_shape);
    defer x_scale.deinit();

    var w_scale_data: [1]f32 = .{ 1.0 };
    var w_scale = try Tensor(f32).fromArray(&allocator, &w_scale_data, &scalar_shape);
    defer w_scale.deinit();

    var y_scale_data: [1]f32 = .{ 1.0 };
    var y_scale = try Tensor(f32).fromArray(&allocator, &y_scale_data, &scalar_shape);
    defer y_scale.deinit();

    var x_zp_data: [1]i8 = .{ 0 };
    var x_zp = try Tensor(i8).fromArray(&allocator, &x_zp_data, &scalar_shape);
    defer x_zp.deinit();

    var w_zp_data: [1]i8 = .{ 0 };
    var w_zp = try Tensor(i8).fromArray(&allocator, &w_zp_data, &scalar_shape);
    defer w_zp.deinit();

    var y_zp_data: [1]i8 = .{ 0 };
    var y_zp = try Tensor(i8).fromArray(&allocator, &y_zp_data, &scalar_shape);
    defer y_zp.deinit();

    var bias_shape = [_]usize{ 1 };
    var bias_data: [1]f32 = .{ 0.0 };
    var bias = try Tensor(f32).fromArray(&allocator, &bias_data, &bias_shape);
    defer bias.deinit();

    const auto_pad = std.mem.span("NOTSET");

    const x_real = (@as(f32, @floatFromInt(input_data[0][0][0][0])) - @as(f32, @floatFromInt(x_zp_data[0]))) * x_scale_data[0];
    const w_real = (@as(f32, @floatFromInt(weight_data[0][0][0][0])) - @as(f32, @floatFromInt(w_zp_data[0]))) * w_scale_data[0];

    const cases = [_]struct {
        bias: f32,
        expected_real: f32,
    }{
        .{ .bias = -0.5, .expected_real = -1.5 },
        .{ .bias = 0.5, .expected_real = -0.5 },
    };

    for (cases) |case| {
        bias.data[0] = case.bias;

        try tensor_math.qlinearconv_embedded_lean(
            i8,
            i8,
            f32,
            void,
            f32,
            &input,
            &x_scale,
            &x_zp,
            &weight,
            &w_scale,
            &w_zp,
            &embedded_output,
            &y_scale,
            &y_zp,
            &bias,
            null,
            null,
            null,
            null,
            auto_pad,
        );

        try tensor_math.qlinearconv_lean(
            i8,
            i8,
            f32,
            void,
            f32,
            &input,
            &x_scale,
            &x_zp,
            &weight,
            &w_scale,
            &w_zp,
            &reference_output,
            &y_scale,
            &y_zp,
            &bias,
            null,
            null,
            null,
            null,
            auto_pad,
        );

        const expected_real = x_real * w_real + case.bias;
        try std.testing.expectApproxEqAbs(case.expected_real, expected_real, 1e-6);

        const expected_q = @as(i8, @intFromFloat(@round(expected_real)));
        try std.testing.expectEqual(expected_q, embedded_output.data[0]);
        try std.testing.expectEqual(reference_output.data[0], embedded_output.data[0]);
    }
}
