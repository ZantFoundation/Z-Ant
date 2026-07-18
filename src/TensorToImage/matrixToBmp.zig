const std = @import("std");
const colormap_mod = @import("colormap.zig");

const FILE_HEADER_SIZE: u32 = 14;
const DIB_HEADER_SIZE:  u32 = 40;
const PIXEL_OFFSET:     u32 = FILE_HEADER_SIZE + DIB_HEADER_SIZE;
const BYTES_PER_PIXEL:  u32 = 3;
const BORDER_PX: usize = 4;

/// Writes a single n×n f32 matrix as a BMP. Values in [-1, 1].
pub fn matrixToBmp(
    matrix: []const f32,
    n: usize,
    path: []const u8,
    cmap: colormap_mod.Colormap,
) !void {
    std.debug.assert(matrix.len == n * n);
    const mats = [_][]const f32{matrix};
    try writeBmp(&mats, n, 0, path, cmap);
}

/// Tiles multiple n×n f32 matrices horizontally in one BMP (4px white border between).
pub fn matricesToBmp(
    matrices: []const []const f32,
    n: usize,
    path: []const u8,
    cmap: colormap_mod.Colormap,
) !void {
    try writeBmp(matrices, n, BORDER_PX, path, cmap);
}

fn writeBmp(
    matrices: []const []const f32,
    n: usize,
    border: usize,
    path: []const u8,
    cmap: colormap_mod.Colormap,
) !void {
    std.debug.assert(matrices.len > 0);
    for (matrices) |m| std.debug.assert(m.len == n * n);
    const num = matrices.len;
    const width: u32  = @intCast(n * num + border * (num - 1));
    const height: u32 = @intCast(n);
    const row_raw:    u32 = width * BYTES_PER_PIXEL;
    const row_stride: u32 = (row_raw + 3) & ~@as(u32, 3);
    const pixel_size: u32 = row_stride * height;
    const file_size:  u32 = PIXEL_OFFSET + pixel_size;

    const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();

    var file_buf: [4096]u8 = undefined;
    var file_writer = file.writer(&file_buf);
    const w = &file_writer.interface;

    // BITMAPFILEHEADER (14 bytes)
    try w.writeByte('B');
    try w.writeByte('M');
    try w.writeInt(u32, file_size,    .little);
    try w.writeInt(u32, 0,            .little);
    try w.writeInt(u32, PIXEL_OFFSET, .little);

    // BITMAPINFOHEADER (40 bytes)
    try w.writeInt(u32, DIB_HEADER_SIZE,              .little);
    try w.writeInt(i32, @intCast(width),               .little);
    try w.writeInt(i32, -@as(i32, @intCast(height)),   .little);
    try w.writeInt(u16, 1,                             .little);
    try w.writeInt(u16, 24,                            .little);
    try w.writeInt(u32, 0,                             .little);
    try w.writeInt(u32, pixel_size,                    .little);
    try w.writeInt(i32, 2835,                          .little);
    try w.writeInt(i32, 2835,                          .little);
    try w.writeInt(u32, 0,                             .little);
    try w.writeInt(u32, 0,                             .little);

    // Pixel data (top-down, BGR)
    const padding = row_stride - row_raw;
    const pad_buf = [_]u8{0} ** 3;
    const white   = [_]u8{ 255, 255, 255 };

    for (0..n) |row| {
        for (matrices, 0..) |matrix, mi| {
            for (0..n) |col| {
                const rgb = colormap_mod.mapToRgb(matrix[row * n + col], cmap);
                try w.writeByte(rgb[2]); // B
                try w.writeByte(rgb[1]); // G
                try w.writeByte(rgb[0]); // R
            }
            if (mi + 1 < matrices.len) {
                for (0..border) |_| try w.writeAll(&white);
            }
        }
        if (padding > 0) try w.writeAll(pad_buf[0..padding]);
    }

    try w.flush();
}
