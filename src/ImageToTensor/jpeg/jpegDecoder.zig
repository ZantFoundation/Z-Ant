const std = @import("std");
const alg = @import("jpegAlgorithms.zig");
const parser = @import("jpegParser.zig");
const bmp = @import("../writerBMP.zig");
const utils = @import("../utils.zig");

const JPEG = utils.ImageFormat.JPEG;
pub const SegmentReader = utils.SegmentReader;
const ColorChannels = utils.ColorChannels;
const JpegData = parser.JpegData;
const MCU = alg.MCU;
const ImToTensorError = utils.ImToTensorError;

const writeBmp = bmp.writeBmp;

/// Color space for JPEG decoding output.
pub const ColorSpace = enum {
    rgb,
    ycbcr,
    grayscale,
};

/// Unified JPEG decode pipeline. Parses, decodes Huffman data, dequantizes,
/// applies IDCT, and converts to the requested color space.
pub fn jpegDecode(segment_reader: *SegmentReader, allocator: std.mem.Allocator, color_space: ColorSpace) !ColorChannels {
    // parse the Jpeg file
    var header = try parser.jpegParser(allocator, segment_reader);
    defer header.deinit(allocator);

    // Allocate MCUs
    const mcus = try allocator.alloc(MCU, header.mcu_true_height * header.mcu_true_width);
    defer {
        for (mcus) |mcu| {
            allocator.free(mcu.y);
            allocator.free(mcu.cb);
            allocator.free(mcu.cr);
        }
        allocator.free(mcus);
    }

    // Decode Huffman entropy data
    try alg.decodeHuffmanData(header, allocator, mcus);

    // Dequantize MCUs
    try alg.dequantize(header, mcus);

    // Inverse Discrete Cosine Transform for each MCU
    try alg.inverseDCT(header, mcus);

    // Color space conversion
    switch (color_space) {
        .rgb => try alg.yCbCrToRgb(header, mcus),
        .ycbcr, .grayscale => try alg.yCbCrUpsampling(header, mcus),
    }

    // Convert to color channels
    return try alg.writeChannels(header, mcus, allocator);
}

// Backward-compatible public wrappers

pub fn jpegToRGB(segment_reader: *SegmentReader, allocator: std.mem.Allocator) !ColorChannels {
    return jpegDecode(segment_reader, allocator, .rgb);
}

pub fn jpegToYCbCr(segment_reader: *SegmentReader, allocator: std.mem.Allocator) !ColorChannels {
    return jpegDecode(segment_reader, allocator, .ycbcr);
}

pub fn jpegToGray(segment_reader: *SegmentReader, allocator: std.mem.Allocator) !ColorChannels {
    return jpegDecode(segment_reader, allocator, .grayscale);
}

//------------------------------------------------------------------------------------------------------//
//--------------------------BMP IMAGE GENERATING FUNCTIONS for debugging--------------------------------//
//------------------------------------------------------------------------------------------------------//

fn debugJpegDecode(
    allocator: std.mem.Allocator,
    image_path: []const u8,
    color_space: ColorSpace,
) !void {
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

    // create the reader
    var block_reader = try SegmentReader.init(buffer, JPEG);

    // decode the image using the appropriate decoder
    var channels = try jpegDecode(&block_reader, allocator, color_space);
    defer channels.deinit(allocator);

    // write bmp file
    const bmp_mode: u8 = switch (color_space) {
        .rgb => 0,
        .ycbcr => 1,
        .grayscale => 2,
    };
    try writeBmp(channels, image_path, bmp_mode);
}

// Backward-compatible public wrappers for debug functions

pub fn debug_jpegToRGB(allocator: std.mem.Allocator, image_path: []const u8) !void {
    return debugJpegDecode(allocator, image_path, .rgb);
}

pub fn debug_jpegToYCbCr(allocator: std.mem.Allocator, image_path: []const u8) !void {
    return debugJpegDecode(allocator, image_path, .ycbcr);
}

pub fn debug_jpegToGrayscale(allocator: std.mem.Allocator, image_path: []const u8) !void {
    return debugJpegDecode(allocator, image_path, .grayscale);
}
