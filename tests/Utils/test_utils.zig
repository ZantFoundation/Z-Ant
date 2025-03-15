const std = @import("std");
const zant = @import("zant");
const conv = zant.utils.type_converter;
const init = zant.utils.tensorInitializer;

test "Utils description test" {
    std.debug.print("\n--- Running utils test\n", .{});
}

test "convert integer to float" {
    std.debug.print("\n     convert integer to float", .{});
    const result = conv.convert(i32, f64, 42);
    const a: f64 = 42.0;
    try std.testing.expectEqual(a, result);
    try std.testing.expectEqual(f64, @TypeOf(result));
}

test "convert float to integer" {
    std.debug.print("\n     convert float to integer", .{});
    const result = conv.convert(f64, i32, 42.9);
    const a: i32 = 42;
    try std.testing.expectEqual(a, result);
    try std.testing.expectEqual(i32, @TypeOf(result));
}

test "convert integer to bool" {
    std.debug.print("\n     convert integer to bool", .{});
    const result = conv.convert(i32, bool, 1);
    try std.testing.expectEqual(bool, @TypeOf(result));
}

test "convert float to bool" {
    std.debug.print("\n     convert float to bool", .{});
    const result = conv.convert(f64, bool, 0.0);
    try std.testing.expectEqual(false, result);
}

test "convert true bool to integer" {
    std.debug.print("\n     convert bool to integer", .{});
    const result = conv.convert(bool, i32, true);
    try std.testing.expectEqual(i32, @TypeOf(result));
    try std.testing.expectEqual(1, result);
}

test "convert false bool to integer" {
    std.debug.print("\n     convert bool to integer", .{});
    const result = conv.convert(bool, i32, false);
    try std.testing.expectEqual(i32, @TypeOf(result));
    try std.testing.expectEqual(0, result);
}

test "convert bool to float" {
    std.debug.print("\n     convert bool to float", .{});
    const result = conv.convert(bool, f64, false);
    try std.testing.expectEqual(0.0, result);
    try std.testing.expectEqual(f64, @TypeOf(result));
}

test "convert bool to bool" {
    std.debug.print("\n     convert bool to bool", .{});
    const result = conv.convert(bool, bool, true);
    try std.testing.expectEqual(true, result);
    try std.testing.expectEqual(bool, @TypeOf(result));
}

test "convert comptime int to float" {
    std.debug.print("\n     convert comptime int to float", .{});
    comptime {
        const a = 123;
        const result = conv.convert(@TypeOf(a), f64, a);
        try std.testing.expectEqual(123.0, result);
        try std.testing.expectEqual(f64, @TypeOf(result));
    }
}

test "generateRandomSlice allocates correctly" {
    std.debug.print("\n     Checking allocation in tensorInitializer\n", .{});
    var allocator = std.testing.allocator;
    const slice = try init.generateRandomSlice(f32, allocator, 10, init.InitMethod.Dumb);
    defer allocator.free(slice);

    try std.testing.expectEqual(slice.len, 10);
}

test "generateRandomSlice produces only 0 and 1 for Binary" {
    std.debug.print("\n     Checking binary values in tensorInitializer\n", .{});
    var allocator = std.testing.allocator;
    const slice = try init.generateRandomSlice(u8, allocator, 100, init.InitMethod.Binary);
    defer allocator.free(slice);

    for (slice) |val| {
        try std.testing.expect(val == 0 or val == 1);
    }
}

test "generateRandomSlice respects LimitedRange" {
    std.debug.print("\n     Checking limited range in tensorInitializer\n", .{});
    var allocator = std.testing.allocator;
    const slice = try init.generateRandomSlice(i32, allocator, 100, init.InitMethod.LimitedRange);
    defer allocator.free(slice);

    for (slice) |val| {
        try std.testing.expect(val >= 10 and val <= 100);
    }
}

test "generateRandomSlice respects Gaussian distribution" {
    std.debug.print("\n     Checking Gaussian distribution in tensorInitializer\n", .{});
    var allocator = std.testing.allocator;
    const slice = try init.generateRandomSlice(f64, allocator, 10000, init.InitMethod.Gaussian);
    defer allocator.free(slice);

    var sum: f64 = 0;
    for (slice) |val| {
        sum += val;
    }
    const mean = sum / @as(f64, slice.len);
    try std.testing.expect(mean > -0.2 and mean < 0.2);
}
