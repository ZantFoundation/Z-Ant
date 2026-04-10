const std = @import("std");

pub const Colormap = enum { Grayscale, Viridis, Jet };

/// Maps a value in [-1, 1] to an RGB triplet [R, G, B].
/// Values outside [-1, 1] are clamped.
pub fn mapToRgb(val: f32, cmap: Colormap) [3]u8 {
    const t = @max(0.0, @min(1.0, (val + 1.0) * 0.5));
    return switch (cmap) {
        .Grayscale => grayscale(t),
        .Viridis   => interpolate(t, &viridis_stops),
        .Jet       => interpolate(t, &jet_stops),
    };
}

fn grayscale(t: f32) [3]u8 {
    const v: u8 = @intFromFloat(@min(255.0, t * 255.0));
    return .{ v, v, v };
}

const Stop = struct { t: f32, r: u8, g: u8, b: u8 };

fn interpolate(t: f32, stops: []const Stop) [3]u8 {
    // Find the bracketing pair: last stop where a.t <= t
    var lo = stops[0];
    var hi = stops[1];
    for (stops[0 .. stops.len - 1], stops[1..]) |a, b| {
        if (a.t <= t) {
            lo = a;
            hi = b;
        }
    }
    if (lo.t == hi.t) return .{ lo.r, lo.g, lo.b };
    const s = @min(1.0, (t - lo.t) / (hi.t - lo.t));
    return .{ lerpU8(lo.r, hi.r, s), lerpU8(lo.g, hi.g, s), lerpU8(lo.b, hi.b, s) };
}

fn lerpU8(a: u8, b: u8, s: f32) u8 {
    const af: f32 = @floatFromInt(a);
    const bf: f32 = @floatFromInt(b);
    return @intFromFloat(@min(255.0, @max(0.0, af + (bf - af) * s)));
}

const viridis_stops = [_]Stop{
    .{ .t = 0.00, .r = 68,  .g = 1,   .b = 84  },
    .{ .t = 0.25, .r = 59,  .g = 82,  .b = 139 },
    .{ .t = 0.50, .r = 33,  .g = 145, .b = 140 },
    .{ .t = 0.75, .r = 94,  .g = 201, .b = 98  },
    .{ .t = 1.00, .r = 253, .g = 231, .b = 37  },
};

const jet_stops = [_]Stop{
    .{ .t = 0.000, .r = 0,   .g = 0,   .b = 128 },
    .{ .t = 0.125, .r = 0,   .g = 0,   .b = 255 },
    .{ .t = 0.375, .r = 0,   .g = 255, .b = 255 },
    .{ .t = 0.625, .r = 255, .g = 255, .b = 0   },
    .{ .t = 0.875, .r = 255, .g = 0,   .b = 0   },
    .{ .t = 1.000, .r = 128, .g = 0,   .b = 0   },
};
