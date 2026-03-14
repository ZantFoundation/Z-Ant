const std = @import("std");
const zant = @import("zant");

const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;

test "MatMulInteger 2x2 with scalar zero points" {
    const allocator = pkgAllocator.allocator;

    var a_shape = [_]usize{ 2, 2 };
    var b_shape = [_]usize{ 2, 2 };
    var out_shape = [_]usize{ 2, 2 };

    var a_data = [_][2]u8{
        [_]u8{ 1, 2 },
        [_]u8{ 3, 4 },
    };
    var b_data = [_][2]i8{
        [_]i8{ 1, 0 },
        [_]i8{ -1, 2 },
    };
    var a_zp_data = [_]u8{1};
    var b_zp_data = [_]i8{0};
    var zp_shape = [_]usize{1};

    var A = try Tensor(u8).fromArray(&allocator, &a_data, &a_shape);
    defer A.deinit();
    var B = try Tensor(i8).fromArray(&allocator, &b_data, &b_shape);
    defer B.deinit();
    var a_zp = try Tensor(u8).fromArray(&allocator, &a_zp_data, &zp_shape);
    defer a_zp.deinit();
    var b_zp = try Tensor(i8).fromArray(&allocator, &b_zp_data, &zp_shape);
    defer b_zp.deinit();

    var output = try TensMath.matMulInteger(u8, i8, &A, &B, &a_zp, &b_zp);
    defer output.deinit();

    try std.testing.expectEqualSlices(usize, &out_shape, output.shape);
    try std.testing.expectEqual(@as(i32, -1), output.data[0]);
    try std.testing.expectEqual(@as(i32, 2), output.data[1]);
    try std.testing.expectEqual(@as(i32, -1), output.data[2]);
    try std.testing.expectEqual(@as(i32, 6), output.data[3]);
}

test "MatMulInteger lean without zero points" {
    const allocator = pkgAllocator.allocator;

    var a_shape = [_]usize{ 2, 3 };
    var b_shape = [_]usize{ 3, 2 };
    var out_shape = [_]usize{ 2, 2 };

    var a_data = [_][3]i8{
        [_]i8{ 1, 2, 3 },
        [_]i8{ 4, 5, 6 },
    };
    var b_data = [_][2]u8{
        [_]u8{ 1, 2 },
        [_]u8{ 3, 4 },
        [_]u8{ 5, 6 },
    };

    var A = try Tensor(i8).fromArray(&allocator, &a_data, &a_shape);
    defer A.deinit();
    var B = try Tensor(u8).fromArray(&allocator, &b_data, &b_shape);
    defer B.deinit();
    var output = try Tensor(i32).fromShape(&allocator, &out_shape);
    defer output.deinit();

    try TensMath.matMulInteger_lean(&A, &B, null, null, &output);

    try std.testing.expectEqual(@as(i32, 22), output.data[0]);
    try std.testing.expectEqual(@as(i32, 28), output.data[1]);
    try std.testing.expectEqual(@as(i32, 49), output.data[2]);
    try std.testing.expectEqual(@as(i32, 64), output.data[3]);
}
