const std = @import("std");
const zant = @import("zant");

const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const tests_log = std.log.scoped(.lib_elementWise);

test "test sqrt with valid f32 tensor" {
    const allocator = pkgAllocator.allocator;

    var inputArray: [2][2]f32 = [_][2]f32{
        [_]f32{ -5.0, 7.0 },
        [_]f32{ 0.0, 11.0 },
    };

    var shape = [_]usize{ 2, 2 };

    var input = try Tensor(f32).fromArray(&allocator, &inputArray, &shape);
    defer input.deinit();

    var result = try TensMath.sqrt(f32, &input);
    defer result.deinit();

    try std.testing.expect(std.math.isNan(result.data[0]));
    try std.testing.expectEqual(std.math.pow(f32, 7.0, 0.5), result.data[1]);
    try std.testing.expectEqual(std.math.pow(f32, 0.0, 0.5), result.data[2]);
    try std.testing.expectEqual(std.math.pow(f32, 11.0, 0.5), result.data[3]);
}
