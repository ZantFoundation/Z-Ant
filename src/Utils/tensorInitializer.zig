const std = @import("std");
const Allocator = std.mem.Allocator;
const rand = std.Random;
const zant = @import("zant");

const TensorProto = zant.onnx.TensorProto;
const AnyTensor = zant.core.tensor.AnyTensor;
const Tensor = zant.core.tensor.Tensor;

pub const InitMethod = enum {
    Dumb, // Simple random generation
    Uniform, // Uniform distribution
    Gaussian, // Normal distribution (mean 0, standard deviation 1 for normalization)
    Binary, // Only 0 and 1
    LimitedRange, // Values limited to a certain range
    Sparse, // Mostly zeros with a few random values
};

// Function for generating a random tensor
pub fn generateRandomSlice(comptime T: type, allocator: Allocator, shape: usize, method: InitMethod) ![]T {
    const timestamp = @as(u64, @intCast(@as(u64, @intCast(std.time.nanoTimestamp() & 0xFFFFFFFFFFFFFFFF))));
    var random = rand.DefaultPrng.init(timestamp);
    const rng = random.random();
    const slice = try allocator.alloc(T, shape);

    for (slice) |*val| {
        val.* = switch (method) {
            .Dumb, .Uniform => if (@typeInfo(T) == .float) rng.float(T) else rng.int(T),
            .Gaussian => blk: {
                const x = rng.float(f64);
                const safe_x = if (x == 0.0) 0.0001 else x;
                const gaussian = std.math.sqrt(-2.0 * std.math.log(f64, std.math.e, safe_x)) * std.math.cos(2.0 * std.math.pi * rng.float(f64));

                if (@typeInfo(T) == .float) {
                    break :blk @as(T, @floatCast(gaussian));
                } else if (@typeInfo(T) == .int) {
                    const scaled = gaussian * 10.0;
                    const clamped = @max(@min(scaled, @as(f64, @floatFromInt(std.math.maxInt(T)))), @as(f64, @floatFromInt(std.math.minInt(T))));
                    break :blk @as(T, @intFromFloat(clamped));
                } else {
                    break :blk @as(T, gaussian * 10.0);
                }
            },

            .Binary => blk: {
                if (@typeInfo(T) == .float) {
                    break :blk @as(T, @floatFromInt(@intFromBool(rng.boolean())));
                } else {
                    break :blk @as(T, @intFromBool(rng.boolean()));
                }
            },

            .LimitedRange => if (@typeInfo(T) == .float) rng.float(T) * 90.0 + 10.0 else rng.intRangeAtMost(T, 10, 100),
            .Sparse => if (rng.float(f64) < 0.8) @as(T, 0) else if (@typeInfo(T) == .float) rng.float(T) else rng.int(T),
        };
    }

    return slice;
}
