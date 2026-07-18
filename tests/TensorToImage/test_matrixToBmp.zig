const std = @import("std");
const testing = std.testing;
const TensorToImage = @import("TensorToImage");
const matBmp = TensorToImage.matrixToBmp;
const Colormap = TensorToImage.colormap.Colormap;

test "matrixToBmp: file size correct for 4x4 grayscale" {
    // width=4, row_stride=(12+3)&~3=12, pixel_data=12*4=48, total=14+40+48=102
    const n: usize = 4;
    const matrix = [_]f32{0.0} ** (n * n);
    const path = "test_single_4x4.bmp";
    defer std.fs.cwd().deleteFile(path) catch {};

    try matBmp.matrixToBmp(&matrix, n, path, .Grayscale);

    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    try testing.expectEqual(@as(u64, 102), try file.getEndPos());
}

test "matricesToBmp: file size correct for 3 matrices of 4x4 with 4px border" {
    // width=4*3+4*2=20, row_stride=(60+3)&~3=60, pixel_data=60*4=240, total=14+40+240=294
    const n: usize = 4;
    const m1 = [_]f32{-1.0} ** (n * n);
    const m2 = [_]f32{0.0} ** (n * n);
    const m3 = [_]f32{1.0} ** (n * n);
    const matrices = [_][]const f32{ &m1, &m2, &m3 };
    const path = "test_tiled_3x4x4.bmp";
    defer std.fs.cwd().deleteFile(path) catch {};

    try matBmp.matricesToBmp(&matrices, n, path, .Viridis);

    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    try testing.expectEqual(@as(u64, 294), try file.getEndPos());
}

test "matrixToBmp: BMP signature is BM" {
    const n: usize = 2;
    const matrix = [_]f32{0.5} ** (n * n);
    const path = "test_sig.bmp";
    defer std.fs.cwd().deleteFile(path) catch {};

    try matBmp.matrixToBmp(&matrix, n, path, .Jet);

    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    var header: [2]u8 = undefined;
    _ = try file.read(&header);
    try testing.expectEqual(@as(u8, 'B'), header[0]);
    try testing.expectEqual(@as(u8, 'M'), header[1]);
}

test "matrixToBmp: file size correct for n=1 (exercises row padding)" {
    // width=1, row_raw=3, row_stride=(3+3)&~3=4, pixel_data=4*1=4, total=14+40+4=58
    const n: usize = 1;
    const matrix = [_]f32{0.0} ** (n * n);
    const path = "test_single_1x1.bmp";
    defer std.fs.cwd().deleteFile(path) catch {};

    try matBmp.matrixToBmp(&matrix, n, path, .Grayscale);

    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    try testing.expectEqual(@as(u64, 58), try file.getEndPos());
}

test "matricesToBmp: single matrix produces same file size as matrixToBmp" {
    // One matrix in tiled = same as single, no border written
    const n: usize = 4;
    const matrix = [_]f32{0.5} ** (n * n);
    const matrices = [_][]const f32{&matrix};
    const path_tiled = "test_tiled_1x4x4.bmp";
    const path_single = "test_single_4x4_ref.bmp";
    defer std.fs.cwd().deleteFile(path_tiled) catch {};
    defer std.fs.cwd().deleteFile(path_single) catch {};

    try matBmp.matricesToBmp(&matrices, n, path_tiled, .Grayscale);
    try matBmp.matrixToBmp(&matrix, n, path_single, .Grayscale);

    const f1 = try std.fs.cwd().openFile(path_tiled, .{});
    defer f1.close();
    const f2 = try std.fs.cwd().openFile(path_single, .{});
    defer f2.close();
    try testing.expectEqual(try f1.getEndPos(), try f2.getEndPos());
}
