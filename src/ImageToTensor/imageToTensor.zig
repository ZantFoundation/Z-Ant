const std = @import("std");
const zant = @import("zant");
pub const jpeg = @import("jpeg/jpegDecoder.zig");
pub const utils = @import("utils.zig");
const formatVerifier = @import("formatVerifier.zig");

const writeBmp = @import("writerBMP.zig").writeBmp;
const findFormat = formatVerifier.findFormat;

const ImageFormat = formatVerifier.ImageFormat;
const Tensor = zant.core.tensor.Tensor;
const ColorChannels = utils.ColorChannels;
pub const SegmentReader = jpeg.SegmentReader;
const ImToTensorError = utils.ImToTensorError;
const ColorSpace = jpeg.ColorSpace;

/// Internal unified function for converting an image file to a Tensor.
/// All public `imageTo*` functions delegate to this.
fn imageToTensorInternal(
    allocator: std.mem.Allocator,
    image_path: []const u8,
    norm_type: usize,
    comptime T: anytype,
    color_space: ColorSpace,
) !Tensor(T) {
    // open the file
    const file = try std.fs.cwd().openFile(image_path, .{});
    defer file.close();

    const file_size = try file.getEndPos();
    const buffer = try allocator.alloc(u8, @intCast(file_size));
    defer allocator.free(buffer);

    const bytes_read = try file.readAll(buffer);
    if (bytes_read != file_size) {
        return ImToTensorError.UnexpectedEOF;
    }

    // find the format of the image
    const format = try findFormat(buffer);

    // create the reader
    var block_reader = try SegmentReader.init(buffer, format);

    // decode the image using the appropriate decoder
    var channels: ColorChannels = switch (format) {
        ImageFormat.JPEG => try jpeg.jpegDecode(&block_reader, allocator, color_space),
        else => return ImToTensorError.InvalidImageFormat,
    };
    defer channels.deinit(allocator);

    // determine number of output channels
    const num_channels: usize = if (color_space == .grayscale) 1 else channels.component_num;

    // allocate the image tensor [channels][height][width]
    var image = try allocator.alloc([][]T, num_channels);
    for (0..num_channels) |i| {
        image[i] = try allocator.alloc([]T, channels.height);
        for (0..channels.height) |j| {
            image[i][j] = try allocator.alloc(T, channels.width);
        }
    }
    defer {
        for (0..num_channels) |i| {
            for (0..channels.height) |j| {
                allocator.free(image[i][j]);
            }
            allocator.free(image[i]);
        }
        allocator.free(image);
    }

    // normalize image:
    // norm_type = 0 -> normalization between 0 and 1
    // norm_type = 1 -> normalization between -1 and 1
    // if norm_type > 1 -> automatic normalization between 0 and 1
    if (norm_type == 1) {
        try utils.normalizeSigned(T, &channels, image);
    } else {
        try utils.normalize(T, &channels, image);
    }

    // create the tensor
    var shape = [_]usize{ image.len, image[0].len, image[0][0].len };
    return try Tensor(T).fromArray(allocator, image, shape[0..]);
}

// Public API — backward-compatible wrappers

pub fn imageToRGB(
    allocator: std.mem.Allocator,
    image_path: []const u8,
    norm_type: usize,
    comptime T: anytype,
) !Tensor(T) {
    return imageToTensorInternal(allocator, image_path, norm_type, T, .rgb);
}

pub fn imageToYCbCr(
    allocator: std.mem.Allocator,
    image_path: []const u8,
    norm_type: usize,
    comptime T: anytype,
) !Tensor(T) {
    return imageToTensorInternal(allocator, image_path, norm_type, T, .ycbcr);
}

pub fn imageToGray(
    allocator: std.mem.Allocator,
    image_path: []const u8,
    norm_type: usize,
    comptime T: anytype,
) !Tensor(T) {
    return imageToTensorInternal(allocator, image_path, norm_type, T, .grayscale);
}
