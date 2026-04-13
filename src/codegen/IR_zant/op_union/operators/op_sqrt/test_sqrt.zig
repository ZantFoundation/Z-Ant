const std = @import("std");
const IR_zant = @import("IR_zant");

const pkgAllocator = IR_zant.pkg_allocator;
const TensMath = IR_zant.core.math_standard;
const Tensor = IR_zant.core.tensor.Tensor;

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
