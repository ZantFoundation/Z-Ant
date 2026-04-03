const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorMathError = zant.utils.error_handler.TensorMathError;

test "onehot_standard f64 - basic case" {
    std.debug.print("\n     test: onehot_standard f64 - basic case", .{});
    const allocator = pkgAllocator.allocator;

    var indices_array = [_]i64{ 0, 1 };
    var indices_shape = [_]usize{2};

    var depth_array = [_]i64{3};
    var depth_shape = [_]usize{1};

    var values_array = [_]f64{ 0.0, 10.0 };
    var values_shape = [_]usize{2};

    var indices = try Tensor(i64).fromArray(&allocator, &indices_array, &indices_shape);
    var depth = try Tensor(i64).fromArray(&allocator, &depth_array, &depth_shape);
    var values = try Tensor(f64).fromArray(&allocator, &values_array, &values_shape);

    defer indices.deinit();
    defer depth.deinit();
    defer values.deinit();

    var result = try TensMath.oneHot(f64, &indices, &depth, &values, -1);
    defer result.deinit();

    try std.testing.expectEqualSlices(f64, result.data, &[6]f64{ 10.0, 0.0, 0.0, 0.0, 10.0, 0.0 });
}

test "onehot_standard f64 - indices out of range" {
    std.debug.print("\n     test: onehot_standard f64 - indices out of range", .{});
    const allocator = pkgAllocator.allocator;

    var indices_array = [_]i64{ -4, 0, 3, 1 }; // -4 e 3 sono fuori range [-3, 2]
    var indices_shape = [_]usize{4};

    var depth_array = [_]i64{3};
    var depth_shape = [_]usize{1};

    var values_array = [_]f64{ 0.0, 10.0 };
    var values_shape = [_]usize{2};

    var indices = try Tensor(i64).fromArray(&allocator, &indices_array, &indices_shape);
    var depth = try Tensor(i64).fromArray(&allocator, &depth_array, &depth_shape);
    var values = try Tensor(f64).fromArray(&allocator, &values_array, &values_shape);

    defer indices.deinit();
    defer depth.deinit();
    defer values.deinit();

    var result = try TensMath.oneHot(f64, &indices, &depth, &values, -1);
    defer result.deinit();

    try std.testing.expectEqualSlices(f64, result.data, &[12]f64{ 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0 });
    try std.testing.expectEqualSlices(usize, &[_]usize{ 4, 3 }, result.shape);
}

test "onehot_standard bool - axis 0" {
    std.debug.print("\n     test: onehot_standard bool - axis 0", .{});
    const allocator = pkgAllocator.allocator;

    var indices_array = [_]i64{ 0, 1 };
    var indices_shape = [_]usize{2};

    var depth_array = [_]i64{3};
    var depth_shape = [_]usize{1};

    var values_array = [_]bool{ false, true };
    var values_shape = [_]usize{2};

    var indices = try Tensor(i64).fromArray(&allocator, &indices_array, &indices_shape);
    var depth = try Tensor(i64).fromArray(&allocator, &depth_array, &depth_shape);
    var values = try Tensor(bool).fromArray(&allocator, &values_array, &values_shape);

    defer indices.deinit();
    defer depth.deinit();
    defer values.deinit();

    var result = try TensMath.oneHot(bool, &indices, &depth, &values, 0);
    defer result.deinit();

    try std.testing.expectEqualSlices(bool, result.data, &[6]bool{ true, false, false, true, false, false });
    try std.testing.expectEqualSlices(usize, &[_]usize{ 3, 2 }, result.shape);
}

test "onehot_standard f64 - 2D indices" {
    std.debug.print("\n     test: onehot_standard f64 - 2D indices", .{});
    const allocator = pkgAllocator.allocator;

    var indices_array = [_]i64{ 0, 1, 2, 0 }; // Matrice 2x2
    var indices_shape = [_]usize{ 2, 2 };

    var depth_array = [_]i64{3};
    var depth_shape = [_]usize{1};

    var values_array = [_]f64{ 0.0, 10.0 };
    var values_shape = [_]usize{2};

    var indices = try Tensor(i64).fromArray(&allocator, &indices_array, &indices_shape);
    var depth = try Tensor(i64).fromArray(&allocator, &depth_array, &depth_shape);
    var values = try Tensor(f64).fromArray(&allocator, &values_array, &values_shape);

    defer indices.deinit();
    defer depth.deinit();
    defer values.deinit();

    var result = try TensMath.oneHot(f64, &indices, &depth, &values, -1);
    defer result.deinit();

    try std.testing.expectEqualSlices(f64, result.data, &[12]f64{
        10.0, 0.0, 0.0,  0.0,  10.0, 0.0,
        0.0,  0.0, 10.0, 10.0, 0.0,  0.0,
    });
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 2, 3 }, result.shape);
}

test "onehot_standard f64 - invalid depth" {
    std.debug.print("\n     test: onehot_standard f64 - invalid depth", .{});
    const allocator = pkgAllocator.allocator;

    var indices_array = [_]i64{ 0, 1 };
    var indices_shape = [_]usize{2};

    var depth_array = [_]i64{0}; // invalid depth
    var depth_shape = [_]usize{1};

    var values_array = [_]f64{ 0.0, 10.0 };
    var values_shape = [_]usize{2};

    var indices = try Tensor(i64).fromArray(&allocator, &indices_array, &indices_shape);
    var depth = try Tensor(i64).fromArray(&allocator, &depth_array, &depth_shape);
    var values = try Tensor(f64).fromArray(&allocator, &values_array, &values_shape);

    defer indices.deinit();
    defer depth.deinit();
    defer values.deinit();

    try std.testing.expectError(TensorMathError.InvalidDepthValue, TensMath.oneHot(f64, &indices, &depth, &values, -1));
}

test "onehot_standard f64 - invalid axis" {
    std.debug.print("\n     test: onehot_standard f64 - invalid axis", .{});
    const allocator = pkgAllocator.allocator;

    var indices_array = [_]i64{ 0, 1 };
    var indices_shape = [_]usize{2};

    var depth_array = [_]i64{3};
    var depth_shape = [_]usize{1};

    var values_array = [_]f64{ 0.0, 10.0 };
    var values_shape = [_]usize{2};

    var indices = try Tensor(i64).fromArray(&allocator, &indices_array, &indices_shape);
    var depth = try Tensor(i64).fromArray(&allocator, &depth_array, &depth_shape);
    var values = try Tensor(f64).fromArray(&allocator, &values_array, &values_shape);

    defer indices.deinit();
    defer depth.deinit();
    defer values.deinit();

    // axis = 2 è fuori range per rank=1 (range valido: [-2, 1])
    try std.testing.expectError(TensorMathError.InvalidAxes, TensMath.oneHot(f64, &indices, &depth, &values, 2));
}
