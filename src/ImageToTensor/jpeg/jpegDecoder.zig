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

pub fn jpegToYCbCr(segment_reader: *SegmentReader, allocator: std.mem.Allocator) !ColorChannels {

    // parse the Jpeg file
    var header = try parser.jpegParser(allocator, segment_reader);
    defer header.deinit(allocator);

    // Decode Huffman entropy data
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

    // Dequantize MCUS
    try alg.dequantize(header, mcus);

    // Inverse Discrete Cosine Transform for each MCU
    try alg.inverseDCT(header, mcus);

    // Upsampling
    try alg.yCbCrUpsampling(header, mcus);

    // convert to 3 color channels:
    const channels = try alg.writeChannels(header, mcus, allocator);

    return channels;
}

pub fn jpegToRGB(segment_reader: *SegmentReader, allocator: std.mem.Allocator) !ColorChannels {
    // parse the Jpeg file
    var header = try parser.jpegParser(allocator, segment_reader);
    defer header.deinit(allocator);
    // Decode Huffman entropy data
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

    // Dequantize MCUS
    try alg.dequantize(header, mcus);

    // Inverse Discrete Cosine Transform for each MCU
    try alg.inverseDCT(header, mcus);

    // ycbcr to rgb
    try alg.yCbCrToRgb(header, mcus);

    // convert to 3 color channels:
    return try alg.writeChannels(header, mcus, allocator);
}

pub fn jpegToGray(segment_reader: *SegmentReader, allocator: std.mem.Allocator) !ColorChannels {
    const yCbCrChannels: ColorChannels = try jpegToYCbCr(segment_reader, allocator);
    // convert to 3 cde the image using the appropriate decoderolor channels:
    return yCbCrChannels;
}

//------------------------------------------------------------------------------------------------------//
//--------------------------BMP IMAGE GENERATING FUNCTIONS for debugging--------------------------------//
//------------------------------------------------------------------------------------------------------//

pub fn debug_jpegToRGB(
    allocator: std.mem.Allocator,
    image_path: []const u8,
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
    var channels: ColorChannels = undefined;

    // decode the image using the appropriate decoder
    channels = try jpegToRGB(&block_reader, allocator);
    defer channels.deinit(allocator);
    // write bmp file
    try writeBmp(channels, image_path, 0);
}

pub fn debug_jpegToYCbCr(
    allocator: std.mem.Allocator,
    image_path: []const u8,
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

    // create the reader
    var block_reader = try SegmentReader.init(buffer, JPEG);
    var channels: ColorChannels = undefined;

    // decode the image using the appropriate decoder
    channels = try jpegToYCbCr(&block_reader, allocator);
    defer channels.deinit(allocator);
    // write bmp file
    try writeBmp(channels, image_path, 1);
}

pub fn debug_jpegToGrayscale(
    allocator: std.mem.Allocator,
    image_path: []const u8,
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

    // create the reader
    var block_reader = try SegmentReader.init(buffer, JPEG);
    var channels: ColorChannels = undefined;

    // decode the image using the appropriate decoder
    channels = try jpegToGray(&block_reader, allocator);
    defer channels.deinit(allocator);
    try writeBmp(channels, image_path, 2);
}
