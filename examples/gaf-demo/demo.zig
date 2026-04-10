const std = @import("std");
const zant = @import("zant");

const gasf_fn = zant.TensorToImage.gasf.gasf;
const gadf_fn = zant.TensorToImage.gadf.gadf;
const mtf_fn  = zant.TensorToImage.mtf.mtf.mtf;
const matBmp  = zant.TensorToImage.matrixToBmp;
const Colormap = zant.TensorToImage.colormap.Colormap;

const MTF_BINS: usize = 8;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print(
            "Usage: gaf-demo <input.csv> [--split] [--colormap viridis|jet|grayscale]\n", .{});
        return error.MissingArgument;
    }

    const csv_path = args[1];
    var split = false;
    var cmap: Colormap = .Viridis;

    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--split")) {
            split = true;
        } else if (std.mem.eql(u8, args[i], "--colormap") and i + 1 < args.len) {
            i += 1;
            if (std.mem.eql(u8, args[i], "grayscale")) {
                cmap = .Grayscale;
            } else if (std.mem.eql(u8, args[i], "jet")) {
                cmap = .Jet;
            } else if (std.mem.eql(u8, args[i], "viridis")) {
                cmap = .Viridis;
            } else {
                std.debug.print("Unknown colormap '{s}'. Options: viridis, jet, grayscale\n", .{args[i]});
                return error.UnknownColormap;
            }
        }
    }

    const series = try readCsv(allocator, csv_path);
    defer allocator.free(series);
    std.debug.print("Loaded {d} samples from {s}\n", .{ series.len, csv_path });

    const gasf_mat = try gasf_fn(allocator, series, .MinusOneToOne);
    defer allocator.free(gasf_mat);

    const gadf_mat = try gadf_fn(allocator, series, .MinusOneToOne);
    defer allocator.free(gadf_mat);

    const mtf_mat = try mtf_fn(allocator, series, MTF_BINS);
    defer allocator.free(mtf_mat);

    const n = series.len;

    if (split) {
        try matBmp.matrixToBmp(gasf_mat, n, "output_gasf.bmp", cmap);
        try matBmp.matrixToBmp(gadf_mat, n, "output_gadf.bmp", cmap);
        try matBmp.matrixToBmp(mtf_mat,  n, "output_mtf.bmp",  cmap);
        std.debug.print("Written: output_gasf.bmp, output_gadf.bmp, output_mtf.bmp\n", .{});
    } else {
        const matrices = [_][]const f32{ gasf_mat, gadf_mat, mtf_mat };
        try matBmp.matricesToBmp(&matrices, n, "output.bmp", cmap);
        std.debug.print("Written: output.bmp\n", .{});
    }
}

fn readCsv(allocator: std.mem.Allocator, path: []const u8) ![]f32 {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    const content = try file.readToEndAlloc(allocator, 1024 * 1024);
    defer allocator.free(content);

    var list: std.ArrayList(f32) = .empty;
    errdefer list.deinit(allocator);

    var it = std.mem.tokenizeAny(u8, content, ",\n\r \t");
    while (it.next()) |token| {
        const trimmed = std.mem.trim(u8, token, " \t");
        if (trimmed.len == 0) continue;
        const val = try std.fmt.parseFloat(f32, trimmed);
        try list.append(allocator, val);
    }

    return list.toOwnedSlice(allocator);
}
