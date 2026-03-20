const std = @import("std");
const fv = @import("formatVerifier.zig");
const jpeg = @import("jpeg/jpegParser.zig");

pub const ImageFormat = fv.ImageFormat;
const jpegMarker = jpeg.jpegMarker;

// --------------------------------- Blocks Structures of all format ----------------------------------------
pub const PngChunk = struct {
    type: []u8, //chuck type
    length: usize,
    data: []u8,
    crc: u32,
};

pub const JpegSegment = struct {
    type: jpegMarker, // JPEG marker
    length: usize,
    data: []u8,
    idx: usize = 0,

    pub fn nextByte(self: *JpegSegment) !u8 {
        if (self.idx + 1 > self.data.len) {
            return ImToTensorError.UnexpectedEOF;
        }
        const byte = self.data[self.idx];
        defer self.idx += 1;
        return byte;
    }
};

pub const ImageUnit = union(enum) {
    PngChunk: PngChunk,
    JpegSegment: JpegSegment,
};

// ---------------------------------------  Block Reader ----------------------------------------------------
// this struct is used to read the blocks of the image,
// each block is preceded by a marker that variates depending on the format
// the reader is initialized with the image data and the format
// the reader will read the blocks one by one, returning the block data
// the reader will also check if the format is correct and if the end of the segment is reached
pub const SegmentReader = struct {
    format: ImageFormat, // format type
    data: []u8, //
    idx: usize, // index pointing to the next block marker
    end_of_segment: bool,

    pub fn init(data: []u8, format: ImageFormat) !SegmentReader {
        if (!fv.verifyFormat(data, format)) {
            return ImToTensorError.WrongFileFormat;
        }
        return SegmentReader{
            .format = format,
            .data = data,
            .idx = 0,
            .end_of_segment = false,
        };
    }

    pub fn deinit(self: *SegmentReader, allocator: std.mem.Allocator) void {
        if (self.data.len > 0)
            allocator.free(self.data);
    }

    // read the next block of the image
    // the block is read using specific methods for each format
    // the block is returned as a union of the specific type
    // the reader will also check if the format is correct and if the end of the segment is reached
    pub fn nextUnit(self: *SegmentReader) !ImageUnit {
        return switch (self.format) {
            .JPEG => ImageUnit{ .JpegSegment = try self.nextJpegSegment() },
            //.PNG => png.nextPngChunk(self),
            else => ImToTensorError.InvalidImageFormat,
        };
    }

    // read the next len bytes of the image
    // the bytes are returned as a slice
    // the reader will also check if the end of the segment is reached
    pub fn tryAdvance(self: *SegmentReader, len: usize) ![]u8 {
        if (self.idx + len > self.data.len) {
            return ImToTensorError.UnexpectedEOF;
        }
        const slice = self.data[self.idx .. self.idx + len];
        self.idx += len;
        return slice;
    }

    // read the next byte of the image and return it as a u8
    pub fn nextByte(self: *SegmentReader) !u8 {
        if (self.idx + 1 > self.data.len) {
            return ImToTensorError.UnexpectedEOF;
        }
        const byte_slice = try self.tryAdvance(1);
        const byte = byte_slice[0];
        return byte;
    }

    // JPEG SPECIFIC METHODS
    // specific methods to read the jpeg segments
    // is invoked by the nextUnit method
    fn nextJpegSegment(self: *SegmentReader) !JpegSegment {
        // gets the marker
        if (try self.nextByte() != 0xFF) {
            return ImToTensorError.NotStartOfSegment;
        }

        var marker = try self.nextByte();
        while (marker == 0xFF) {
            marker = try self.nextByte();
        }
        if (marker == 0xD8)
            return JpegSegment{
                .type = .SOI,
                .length = 0,
                .data = undefined,
                .idx = 0,
            };
        if (marker == 0xDD) {
            const dri_bytes = try self.tryAdvance(2);

            return JpegSegment{
                .type = .DRI,
                .length = 0,
                .data = dri_bytes,
                .idx = 0,
            };
        }
        // gest payload length include itself
        const length_bytes = try self.tryAdvance(2);
        var temp: [2]u8 = undefined;
        @memcpy(&temp, length_bytes); // copy the 2 bytes in temp
        const length = std.mem.readInt(u16, &temp, .big);
        // gets payload
        const data = try self.tryAdvance(length - 2);

        const segment = JpegSegment{
            .type = @enumFromInt(marker),
            .data = data,
            .length = length - 2,
            .idx = 0,
        };
        return segment;
    }
};

// ---------------------------------------- Bit Reader ------------------------------------------------------
// this struct is used to read the bitstreame of the image
pub const BitReader = struct {
    data: []const u8, // readable data
    nextByte: usize = 0, // current byte index
    nextBit: u4 = 0, // bit to return (0-7) within data[nextByte]

    pub fn init(data: []const u8) BitReader {
        return .{ .data = data };
    }

    // read a single bit from the bitstream
    // the bit is returned as a u8
    pub fn readBit(self: *BitReader) !u8 {
        if (self.nextByte >= self.data.len) return ImToTensorError.UnexpectedEOF;

        const shift: u3 = @intCast(7 - self.nextBit);
        const bit: u8 = (self.data[self.nextByte] >> shift) & 1;

        self.nextBit += 1;
        if (self.nextBit == 8) {
            self.nextBit = 0;
            self.nextByte += 1;
        }
        return bit;
    }

    // read N bits
    pub fn readBits(self: *BitReader, len: u32) !i32 {
        var out: i32 = 0;
        var i: u32 = 0;
        while (i < len) : (i += 1) {
            const bit = self.readBit() catch |err| {
                return err; // propaga Eof
            };
            out = (out << 1) | @as(i32, bit);
        }
        return out;
    }

    // allign the bit reader to the next byte
    // this is used to discard the remaining bits of the current byte and move to the next byte
    pub fn bitAlign(self: *BitReader) !void {
        if (self.nextBit != 0) {
            self.nextBit = 0;
            self.nextByte += 1;
        }
    }
};

// ---------------------------------------- Normalization ---------------------------------------------------
// struct to hold the channels of the image
// the channels are stored as slices of u8
// the channels are used to store the pixel values of the image
// the pixel values can be stored as Rgb, YCbCr, or Grayscale
pub const ColorChannels = struct {
    component_num: usize = 3,

    height: u16 = 0,
    width: u16 = 0,

    // used for RGB or YCbCr
    ch1: []u8 = undefined,
    ch2: []u8 = undefined,
    ch3: []u8 = undefined,

    // used for alpha channel in RGBA format
    alpha: []u8 = undefined,

    pub fn init(allocator: std.mem.Allocator, len: u32, component_num: usize) !ColorChannels {
        if (component_num <= 0 or component_num == 2 or component_num > 4) {
            return ImToTensorError.InvalidComponentNum;
        }

        if (component_num == 1) {
            return ColorChannels{
                .ch1 = try allocator.alloc(u8, len),
                .ch2 = &[_]u8{},
                .ch3 = &[_]u8{},
            };
        }

        if (component_num == 4) {
            return ColorChannels{
                .ch1 = try allocator.alloc(u8, len),
                .ch2 = try allocator.alloc(u8, len),
                .ch3 = try allocator.alloc(u8, len),
                .alpha = try allocator.alloc(u8, len),
            };
        }

        return ColorChannels{
            .ch1 = try allocator.alloc(u8, len),
            .ch2 = try allocator.alloc(u8, len),
            .ch3 = try allocator.alloc(u8, len),
            .alpha = &[_]u8{},
        };
    }
    pub fn deinit(self: *ColorChannels, allocator: std.mem.Allocator) void {
        allocator.free(self.ch1);
        allocator.free(self.ch2);
        allocator.free(self.ch3);
    }

    pub fn get(self: *ColorChannels, component: usize) ![]u8 {
        return switch (component) {
            0 => self.ch1,
            1 => self.ch2,
            2 => self.ch3,
            3 => self.alpha,
            else => ImToTensorError.InvalidComponent,
        };
    }
};

// normalize the 3 channels from 0 - 255 to 0 - 1
pub fn normalize(comptime T: type, channel: *ColorChannels, output: [][][]T) !void {
    var idx: usize = 0;
    for (0..channel.height) |row| {
        for (0..channel.width) |col| {
            for (0..output.len) |comp| {
                const ch = try channel.get(comp);
                output[comp][row][col] = @as(T, @floatFromInt(ch[idx])) / 255.0;
            }
            idx += 1;
        }
    }
}

// normalize the 3 channels from 0 - 255 to -1 - 1
pub fn normalizeSigned(comptime T: type, channel: *ColorChannels, output: [][][]T) !void {
    var idx: usize = 0;
    for (0..channel.height) |row| {
        for (0..channel.width) |col| {
            for (0..output.len) |comp| {
                const ch = try channel.get(comp);
                output[comp][row][col] = @as(T, @floatFromInt(ch[idx])) / 127.5 - 1.0;
            }
            idx += 1;
        }
    }
}

pub const ImToTensorError = error{
    InvalidImageFormat,
    UnexpectedEOF,
    WrongFileFormat,
    NotStartOfSegment,
    InvalidComponentNum,
    InvalidComponent,
    InvalidColorspace,
    NameTooLong,
    UnexpectedEof,
    InvalidDcValue,
    DcCoefficientLenghtGreaterThan11,
    InvalidAcValue,
    InvalidAcSymbol,
    ZeroRunExceedeMCU,
    DecodeAcCoefficientLenGreaterThan10,
    InvalidHuffmanCode,
    InvalidQuantizationTableId,
    UnsupportedPrecision,
    SegmentTooShort,
    InvalidHuffmanTableId,
    TooManyHTSymbols,
    UnexpectedEndOfSegment,
    SamplingFactorsNotSupported,
    SamplingOnCbCrNotSupported,
    SosDetectedBeforeSof,
    InvalidSpectralSeletion,
    InvalidSuccessiveApproximation,
    UnexpectedMarker,
    ArithmeticEncodingNotSupported,
    SofMarkerNotSupported,
    RstNDetectedBeforeSos,
    SOFMarkerNotSupported,
    RSTNDetectedBeforeSOS,
};
