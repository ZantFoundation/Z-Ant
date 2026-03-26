const std = @import("std");

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
