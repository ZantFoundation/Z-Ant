const std = @import("std");
const testing = std.testing;

const zant = @import("zant");
const timeseries_to_image_api = zant.TensorToImage.timeseries_to_image_api;
const compound = zant.TensorToImage.compound;

test "zant_timeseries_to_image: tutti i valori in [0, 1]" {
    var input: [64]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.01 - 0.32;

    var output: [3 * 64 * 64]f32 = undefined;
    timeseries_to_image_api.zant_timeseries_to_image(&input, &output);

    for (output) |v| {
        try testing.expect(v >= 0.0 and v <= 1.0);
    }
}

test "zant_timeseries_to_image: corrisponde alla reference lean_compound" {
    var input: [64]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.03 - 0.96;

    var output: [3 * 64 * 64]f32 = undefined;
    timeseries_to_image_api.zant_timeseries_to_image(&input, &output);

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const ref = try compound.toRGBImageF32(arena.allocator(), &input, .MinusOneToOne, 4);

    for (ref, output) |r, got| {
        try testing.expectApproxEqAbs(r, got, 1e-6);
    }
}

test "zant_timeseries_to_image: due chiamate successive producono risultati indipendenti" {
    var input_a: [64]f32 = undefined;
    var input_b: [64]f32 = undefined;
    for (&input_a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.01;
    for (&input_b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(64 - i)) * 0.01;

    var out_a: [3 * 64 * 64]f32 = undefined;
    var out_b: [3 * 64 * 64]f32 = undefined;
    timeseries_to_image_api.zant_timeseries_to_image(&input_a, &out_a);
    timeseries_to_image_api.zant_timeseries_to_image(&input_b, &out_b);

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const ref_a = try compound.toRGBImageF32(arena.allocator(), &input_a, .MinusOneToOne, 4);
    const ref_b = try compound.toRGBImageF32(arena.allocator(), &input_b, .MinusOneToOne, 4);

    for (ref_a, out_a) |r, got| try testing.expectApproxEqAbs(r, got, 1e-6);
    for (ref_b, out_b) |r, got| try testing.expectApproxEqAbs(r, got, 1e-6);
}
