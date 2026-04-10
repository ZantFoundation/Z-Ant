const std = @import("std");
const testing = std.testing;
const zant = @import("zant");
const colormap = zant.TensorToImage.colormap;

test "Grayscale: val=-1.0 maps to black" {
    const rgb = colormap.mapToRgb(-1.0, .Grayscale);
    try testing.expectEqual(@as(u8, 0), rgb[0]);
    try testing.expectEqual(@as(u8, 0), rgb[1]);
    try testing.expectEqual(@as(u8, 0), rgb[2]);
}

test "Grayscale: val=1.0 maps to white" {
    const rgb = colormap.mapToRgb(1.0, .Grayscale);
    try testing.expectEqual(@as(u8, 255), rgb[0]);
    try testing.expectEqual(@as(u8, 255), rgb[1]);
    try testing.expectEqual(@as(u8, 255), rgb[2]);
}

test "Grayscale: val=0.0 maps to mid-grey" {
    const rgb = colormap.mapToRgb(0.0, .Grayscale);
    try testing.expect(rgb[0] >= 126 and rgb[0] <= 128);
    try testing.expectEqual(rgb[0], rgb[1]);
    try testing.expectEqual(rgb[0], rgb[2]);
}

test "Viridis: val=-1.0 maps to dark purple (low end)" {
    const rgb = colormap.mapToRgb(-1.0, .Viridis);
    // Exact first stop: (68, 1, 84)
    try testing.expectEqual(@as(u8, 68),  rgb[0]);
    try testing.expectEqual(@as(u8, 1),   rgb[1]);
    try testing.expectEqual(@as(u8, 84),  rgb[2]);
}

test "Viridis: val=1.0 maps to yellow (high end)" {
    const rgb = colormap.mapToRgb(1.0, .Viridis);
    // Exact last stop: (253, 231, 37)
    try testing.expectEqual(@as(u8, 253), rgb[0]);
    try testing.expectEqual(@as(u8, 231), rgb[1]);
    try testing.expectEqual(@as(u8, 37),  rgb[2]);
}

test "Jet: val=-1.0 maps to dark blue (low end)" {
    const rgb = colormap.mapToRgb(-1.0, .Jet);
    // Exact first stop: (0, 0, 128)
    try testing.expectEqual(@as(u8, 0),   rgb[0]);
    try testing.expectEqual(@as(u8, 0),   rgb[1]);
    try testing.expectEqual(@as(u8, 128), rgb[2]);
}

test "Jet: val=1.0 maps to dark red (high end)" {
    const rgb = colormap.mapToRgb(1.0, .Jet);
    // Exact last stop: (128, 0, 0)
    try testing.expectEqual(@as(u8, 128), rgb[0]);
    try testing.expectEqual(@as(u8, 0),   rgb[1]);
    try testing.expectEqual(@as(u8, 0),   rgb[2]);
}

test "clamping: values outside [-1, 1] are clamped" {
    const low  = colormap.mapToRgb(-2.0, .Grayscale);
    const high = colormap.mapToRgb( 2.0, .Grayscale);
    const low_ref  = colormap.mapToRgb(-1.0, .Grayscale);
    const high_ref = colormap.mapToRgb( 1.0, .Grayscale);
    try testing.expectEqualSlices(u8, &low_ref,  &low);
    try testing.expectEqualSlices(u8, &high_ref, &high);
}
