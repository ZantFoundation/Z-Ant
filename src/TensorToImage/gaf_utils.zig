const std = @import("std");

/// Possible errors during normalization
pub const NormalizeError = error{ EmptyInput, LenghtMismatch };

/// Defines the rescaling interval for the time series
/// - ZeroToOne: Bijective mapping, preserves accurate inverse map
/// - MinusOneToOne: Standard mapping, cosine values fall into [0, pi]
pub const NormRange = enum {
    ZeroToOne,
    MinusOneToOne,
};

/// Normalizes a time series to [-1, 1] using min-max scaling.
///
/// Formula: x̃ᵢ = ((xᵢ - max) + (xᵢ - min)) / (max - min)
///          which simplifies to: x̃ᵢ = (2·xᵢ - max - min) / (max - min)
///
/// Flatline case (max == min): output is all zeros.
/// Final clamp to [-1.0, 1.0] to guard against floating point errors.
///
/// Precondition: output.len == input.len
pub fn normalize_minmax(input: []const f32, output: []f32) void {
    std.debug.assert(input.len == output.len);

    if (input.len == 0) return;

    // Find min and max
    var min_val: f32 = input[0];
    var max_val: f32 = input[0];
    for (input[1..]) |v| {
        if (v < min_val) min_val = v;
        if (v > max_val) max_val = v;
    }

    const range = max_val - min_val;

    // Flatline case: all values identical → output all zeros
    if (range == 0.0) {
        for (output) |*o| o.* = 0.0;
        return;
    }

    for (input, output) |x, *o| {
        var normalized = ((x - max_val) + (x - min_val)) / range;
        // Clamp to [-1, 1] to guard against floating point errors
        normalized = @max(-1.0, @min(1.0, normalized));
        o.* = normalized;
    }
}

/// Applies polar encoding: φᵢ = arccos(x̃ᵢ), φᵢ ∈ [0, π].
///
/// Input is clamped to [-1.0, 1.0] before acos to avoid NaN
/// on hardware without FPU or with different rounding behaviour.
///
/// Precondition: angles.len == normalized.len
pub fn arccos_polar(normalized: []const f32, angles: []f32) void {
    std.debug.assert(normalized.len == angles.len);

    for (normalized, angles) |x, *a| {
        // Mandatory clamp to avoid NaN
        const clamped = @max(-1.0, @min(1.0, x));
        a.* = std.math.acos(clamped);
    }
}

/// Normalize a time series to [0, 1] or [-1, 1] interval.
/// Includes floating-point clamping to strictly enforce bounds and prevent NaN errors.
pub fn normalize(input: []const f32, output: []f32, range_type: NormRange) !void {
    if (input.len == 0) return NormalizeError.EmptyInput;
    if (input.len != output.len) return NormalizeError.LenghtMismatch;

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
