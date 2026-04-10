const std = @import("std");

/// Possible errors during normalization
pub const NormalizeError = error{ EmptyInput, LengthMismatch };

/// Defines the rescaling interval for the time series
/// - ZeroToOne: Bijective mapping, preserves accurate inverse map
/// - MinusOneToOne: Standard mapping, cosine values fall into [0, pi]
pub const NormRange = enum {
    ZeroToOne,
    MinusOneToOne,
};

/// Normalize a time series to [0, 1] or [-1, 1] interval.
/// Includes floating-point clamping to strictly enforce bounds and prevent NaN errors.
pub fn normalize(input: []const f32, output: []f32, range_type: NormRange) !void {
    if (input.len == 0) return NormalizeError.EmptyInput;
    if (input.len != output.len) return NormalizeError.LengthMismatch;

    var min_val: f32 = input[0];
    var max_val: f32 = input[0];
    for (input) |val| {
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
    }

    const range = max_val - min_val;

    // Handle flat time series to avoid division by zero
    if (range == 0.0) {
        @memset(output, 0.0);
        return;
    }

    switch (range_type) {
        //Normalizes in the [0, 1] interval
        //Equation: (x_i - min(X)) / (max(X) - min(X))
        .ZeroToOne => {
            for (input, output) |x, *o| {
                var normalized = (x - min_val) / range;
                // Clamp to [0, 1] to guard against floating point errors
                normalized = @max(0.0, @min(1.0, normalized));
                o.* = normalized;
            }
        },
        //Normalizes in the [-1, 1] interval
        //Equation: ((x_i - max(X)) + (x_i - min(X))) / (max(X) - min(X))
        .MinusOneToOne => {
            for (input, output) |x, *o| {
                var normalized = ((x - max_val) + (x - min_val)) / range;
                // Clamp to [-1, 1] to guard against floating point errors
                normalized = @max(-1.0, @min(1.0, normalized));
                o.* = normalized;
            }
        },
    }
}
