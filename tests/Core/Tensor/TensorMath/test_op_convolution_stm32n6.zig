const std = @import("std");
const build_options = @import("build_options");
const zant = @import("zant");

const accelerators = zant.core.tensor.accelerators;
const Tensor = zant.core.tensor.Tensor;

inline fn isForceNativeEnabled() bool {
    return @hasDecl(build_options, "stm32n6_force_native") and build_options.stm32n6_force_native;
}

test "STM32N6 accelerator f32 convolution matches reference" {
    if (!accelerators.isStm32n6Enabled()) {
        return error.SkipZigTest;
    }
    if (!isForceNativeEnabled()) {
        return error.SkipZigTest;
    }

    const allocator = std.testing.allocator;

    var input_shape = [_]usize{ 1, 1, 3, 3 };
    var input_data: [1][1][3][3]f32 = .{
        .{
            .{
                .{ 1.0, 2.0, 3.0 },
                .{ 4.0, 5.0, 6.0 },
                .{ 7.0, 8.0, 9.0 },
            },
        },
    };

    var weight_shape = [_]usize{ 1, 1, 2, 2 };
    var weight_data: [1][1][2][2]f32 = .{
        .{
            .{
                .{ 1.0, 0.0 },
                .{ 0.0, 1.0 },
            },
        },
    };

    var output_shape = [_]usize{ 1, 1, 2, 2 };

    var input_tensor = try Tensor(f32).fromArray(&allocator, &input_data, &input_shape);
    defer input_tensor.deinit();

    var weight_tensor = try Tensor(f32).fromArray(&allocator, &weight_data, &weight_shape);
    defer weight_tensor.deinit();

    var output_tensor = try Tensor(f32).fromShape(&allocator, &output_shape);
    defer output_tensor.deinit();

    const params = accelerators.ConvPreparedParams{
        .stride = .{ 1, 1 },
        .dilations = .{ 1, 1 },
        .pads = .{ 0, 0, 0, 0 },
        .group = 1,
        .filters_per_group = 1,
        .channels_per_group = 1,
        .auto_pad = .notset,
    };

    accelerators.resetTestHooks();

    const accelerated = try accelerators.tryConvLean(f32, &input_tensor, &weight_tensor, &output_tensor, null, params);
    try std.testing.expect(accelerated);

    const expect_ethos = @hasDecl(build_options, "stm32n6_use_ethos") and build_options.stm32n6_use_ethos;
    const expect_cmsis = @hasDecl(build_options, "stm32n6_use_cmsis") and build_options.stm32n6_use_cmsis;

    if (expect_ethos) {
        try std.testing.expect(accelerators.ethosUsed());
    } else if (expect_cmsis) {
        try std.testing.expect(accelerators.cmsisUsed());
    } else {
        try std.testing.expect(!accelerators.ethosUsed());
        try std.testing.expect(!accelerators.cmsisUsed());
    }

    const expected: [1][1][2][2]f32 = .{
        .{
            .{
                .{ 6.0, 8.0 },
                .{ 12.0, 14.0 },
            },
        },
    };

    var coord = [_]usize{ 0, 0, 0, 0 };
    for (0..output_shape[0]) |n| {
        coord[0] = n;
        for (0..output_shape[1]) |c| {
            coord[1] = c;
            for (0..output_shape[2]) |h| {
                coord[2] = h;
                for (0..output_shape[3]) |w| {
                    coord[3] = w;
                    const got = try output_tensor.get_at(coord[0..]);
                    try std.testing.expectEqual(expected[n][c][h][w], got);
                }
            }
        }
    }
}
