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

pub fn imageToRGB(
    allocator: std.mem.Allocator,
    image_path: []const u8,
    norm_type: usize,
    comptime T: anytype,
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
    var channels: ColorChannels = undefined;
    defer channels.deinit(allocator);
    // decode the image using the appropriate decoder
    switch (format) {
        ImageFormat.JPEG => {
            channels = try jpeg.jpegToRGB(&block_reader, allocator);
        },
        else => {
            // unsupported format
            return ImToTensorError.InvalidImageFormat;
        },
    }

    // normalize image:
    // norm_type = 0 -> normalization between 0 and 1
    // norm_type = 1 -> normalization beetwen -1 and 1
    // if norm_type > 1 -> automatic normalization between 0 and 1
    // retrurn a tensor with the same shape of the imag
    var image = try allocator.alloc([][]T, channels.component_num);
    for (0..channels.component_num) |i| {
        image[i] = try allocator.alloc([]T, channels.height);
        for (0..channels.height) |j| {
            image[i][j] = try allocator.alloc(T, channels.width);
        }
    }
    defer {
        for (0..channels.component_num) |i| {
            for (0..channels.height) |j| {
                allocator.free(image[i][j]);
            }
            allocator.free(image[i]);
        }
        allocator.free(image);
    }

    if (norm_type == 1) {
        try utils.normalizeSigned(T, &channels, image);
    } else {
        try utils.normalize(T, &channels, image);
    }
    // create the tensor
    var shape = [_]usize{ image.len, image[0].len, image[0][0].len };
    return try Tensor(T).fromArray(allocator, image, shape[0..]);
}

pub fn imageToYCbCr(
    allocator: std.mem.Allocator,
    image_path: []const u8,
    norm_type: usize,
    comptime T: anytype,
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
    var channels: ColorChannels = undefined;
    defer channels.deinit(allocator);
    // decode the image using the appropriate decoder
    switch (format) {
        ImageFormat.JPEG => {
            channels = try jpeg.jpegToYCbCr(&block_reader, allocator);
        },
        else => {
            // unsupported format
            return ImToTensorError.InvalidImageFormat;
        },
    }

    // normalize image:
    // norm_type = 0 -> normalization between 0 and 1
    // norm_type = 1 -> normalization beetwen -1 and 1
    // if norm_type > 1 -> automatic normalization between 0 and 1
    // retrurn a tensor with the same shape of the imag
    var image = try allocator.alloc([][]T, channels.component_num);
    for (0..channels.component_num) |i| {
        image[i] = try allocator.alloc([]T, channels.height);
        for (0..channels.height) |j| {
            image[i][j] = try allocator.alloc(T, channels.width);
        }
    }
    defer {
        for (0..channels.component_num) |i| {
            for (0..channels.height) |j| {
                allocator.free(image[i][j]);
            }
            allocator.free(image[i]);
        }
        allocator.free(image);
    }

    if (norm_type == 1) {
        try utils.normalizeSigned(T, &channels, image);
    } else {
        try utils.normalize(T, &channels, image);
    }
    // create the tensor
    var shape = [_]usize{ image.len, image[0].len, image[0][0].len };
    return try Tensor(T).fromArray(allocator, image, shape[0..]);
}

pub fn imageToGray(
    allocator: std.mem.Allocator,
    image_path: []const u8,
    norm_type: usize,
    comptime T: anytype,
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
    var channels: ColorChannels = undefined;

    // decode the image using the appropriate decoder
    switch (format) {
        ImageFormat.JPEG => {
            channels = try jpeg.jpegToYCbCr(&block_reader, allocator);
        },
        else => {
            // unsupported format
            return ImToTensorError.InvalidImageFormat;
        },
    }

    // normalize image:
    // norm_type = 0 -> normalization between 0 and 1
    // norm_type = 1 -> normalization beetwen -1 and 1
    // if norm_type > 1 -> automatic normalization between 0 and 1
    // retrurn a tensor with the same shape of the imag
    var image = try allocator.alloc([][]T, 1);
    for (0..1) |i| {
        image[i] = try allocator.alloc([]T, channels.height);
        for (0..channels.height) |j| {
            image[i][j] = try allocator.alloc(T, channels.widht);
        }
    }
    defer {
        for (0..1) |i| {
            for (0..channels.height) |j| {
                allocator.free(image[i][j]);
            }
            allocator.free(image[i]);
        }
        allocator.free(image);
    }

    if (norm_type == 1) {
        try utils.normalizeSigned(T, &channels, image);
    } else {
        try utils.normalize(T, &channels, image);
    }

    var shape = [_]usize{ image.len, image[0].len, image[0][0].len };
    channels.deinit(allocator);
    // create the tensor
    return try Tensor(T).fromArray(allocator, image, shape[0..]);
}
