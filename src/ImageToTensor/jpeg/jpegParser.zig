// contains all the specific methods to unpack and decode a JPEG file
const std = @import("std");
const fv = @import("../formatVerifier.zig");
const utils = @import("../utils.zig");

const ImageFormat = fv.ImageFormat;
const SegmentReader = utils.SegmentReader;
const ImageUnit = utils.ImageUnit;
const JpegSegment = utils.JpegSegment;
const ImToTensorError = utils.ImToTensorError;

const common = @import("jpegCommon.zig");
const zigZagMap = common.zigZagMap;

// marker enum: contains all the markers used in the jpeg file
// the markers are used to identify the different segments of the jpeg file
pub const jpegMarker = enum(u8) {
    // Start of Frame markers, non-differential, Huffman coding
    SOF0 = 0xC0, // Baseline DCT
    SOF1 = 0xC1, // Extended sequential DCT
    SOF2 = 0xC2, // Progressive DCT
    SOF3 = 0xC3, // Lossless (sequential)

    // Start of Frame markers, differential, Huffman coding
    SOF5 = 0xC5, // Differential sequential DCT
    SOF6 = 0xC6, // Differential progressive DCT
    SOF7 = 0xC7, // Differential lossless (sequential)

    // Start of Frame markers, non-differential, arithmetic coding
    SOF9 = 0xC9, // Extended sequential DCT
    SOF10 = 0xCA, // Progressive DCT
    SOF11 = 0xCB, // Lossless (sequential)

    // Start of Frame markers, differential, arithmetic coding
    SOF13 = 0xCD, // Differential sequential DCT
    SOF14 = 0xCE, // Differential progressive DCT
    SOF15 = 0xCF, // Differential lossless (sequential)

    // Define Huffman Table(s)
    DHT = 0xC4,

    // JPEG extensions
    JPG = 0xC8,

    // Define Arithmetic Coding Conditioning(s)
    DAC = 0xCC,

    // Restart interval Markers
    RST0 = 0xD0,
    RST1 = 0xD1,
    RST2 = 0xD2,
    RST3 = 0xD3,
    RST4 = 0xD4,
    RST5 = 0xD5,
    RST6 = 0xD6,
    RST7 = 0xD7,

    // Other Markers
    SOI = 0xD8, // Start of Image
    EOI = 0xD9, // End of Image
    SOS = 0xDA, // Start of Scan
    DQT = 0xDB, // Define Quantization Table(s)
    DNL = 0xDC, // Define Number of Lines
    DRI = 0xDD, // Define Restart Interval
    DHP = 0xDE, // Define Hierarchical Progression
    EXP = 0xDF, // Expand Reference Component(s)

    // APPN Markers
    APP0 = 0xE0,
    APP1 = 0xE1,
    APP2 = 0xE2,
    APP3 = 0xE3,
    APP4 = 0xE4,
    APP5 = 0xE5,
    APP6 = 0xE6,
    APP7 = 0xE7,
    APP8 = 0xE8,
    APP9 = 0xE9,
    APP10 = 0xEA,
    APP11 = 0xEB,
    APP12 = 0xEC,
    APP13 = 0xED,
    APP14 = 0xEE,
    APP15 = 0xEF,

    // Misc Markers
    JPG0 = 0xF0,
    JPG1 = 0xF1,
    JPG2 = 0xF2,
    JPG3 = 0xF3,
    JPG4 = 0xF4,
    JPG5 = 0xF5,
    JPG6 = 0xF6,
    JPG7 = 0xF7,
    JPG8 = 0xF8,
    JPG9 = 0xF9,
    JPG10 = 0xFA,
    JPG11 = 0xFB,
    JPG12 = 0xFC,
    JPG13 = 0xFD,
    COM = 0xFE,
    TEM = 0x01,
};

// DATA STRUCTURES
// header struct: contains all the info parsed from the header of the jpeg file
pub const JpegData = struct {
    // header info
    sos_info: SosInfo = undefined,
    frame_info: SofInfo = undefined,
    quant_tables: []?QuantTable = undefined,
    huffman_tables_dc: []?HuffmanTable = undefined,
    huffman_tables_ac: []?HuffmanTable = undefined,
    restart_interval: u16 = 0,

    // bitstream containing the huffman entropy encoded data
    huffman_data: []u8 = undefined,

    // MUCs info
    mcu_height: u32 = 0,
    mcu_width: u32 = 0,

    // MCUs info for chroma subsampling, includes padding
    mcu_true_height: u32 = 0,
    mcu_true_width: u32 = 0,
    horizontal_sampling_factor: u8 = 1,
    vertical_sampling_factor: u8 = 1,

    pub fn init(allocator: std.mem.Allocator) !JpegData {
        var quant_tables = try allocator.alloc(?QuantTable, 4);
        errdefer allocator.free(quant_tables);
        var huffman_tables_dc = try allocator.alloc(?HuffmanTable, 4);
        errdefer allocator.free(huffman_tables_dc);
        var huffman_tables_ac = try allocator.alloc(?HuffmanTable, 4);
        errdefer allocator.free(huffman_tables_ac);
        for (0..4) |i| {
            quant_tables[i] = null;
            huffman_tables_dc[i] = null;
            huffman_tables_ac[i] = null;
        }
        return JpegData{
            .quant_tables = quant_tables,
            .huffman_tables_dc = huffman_tables_dc,
            .huffman_tables_ac = huffman_tables_ac,
        };
    }

    pub fn deinit(self: *JpegData, allocator: std.mem.Allocator) void {
        for (0..4) |i| {
            if (self.huffman_tables_ac[i] != null) {
                self.huffman_tables_ac[i].?.deinit(allocator);
            }
            if (self.huffman_tables_dc[i] != null) {
                self.huffman_tables_dc[i].?.deinit(allocator);
            }
        }

        allocator.free(self.quant_tables);
        allocator.free(self.huffman_tables_dc);
        allocator.free(self.huffman_tables_ac);
        allocator.free(self.huffman_data);
    }
};

// Huffman tables
pub const HuffmanTable = struct {
    class: u8 = undefined, // 0 = DC, 1 = AC
    id: u8 = undefined, // Table ID (0–3)
    code_lengths: []u8 = undefined, // Number of codes per bit length (1–16)
    symbols: []u8 = undefined, // Symbols (values associated to codes)
    codes: []u32 = undefined,

    pub fn deinit(self: *HuffmanTable, allocator: std.mem.Allocator) void {
        allocator.free(self.code_lengths);
        allocator.free(self.symbols);
        allocator.free(self.codes);
    }
};

// Quantizations Tables
pub const QuantTable = struct {
    id: u8,
    table: [64]u8,

    fn init(table: [64]u8, id: u8) !QuantTable {
        if (id > 15)
            return ImToTensorError.InvalidQuantizationTableId;
        return QuantTable{
            .id = id,
            .table = table,
        };
    }
};

// Color Component
pub const ColorComponent = struct { id: u8, h_sampling_factor: u8 = 1, v_sampling_factor: u8 = 1, quant_table_id: u8 = 0, huffmanACTableID: u8 = 0, huffmanDCTableID: u8 = 0, used: bool = false };

// Start of Frame
pub const SofInfo = struct {
    precision: u8,
    height: u16,
    width: u16,
    components_num: u8,
    components: [3]ColorComponent,
    zero_based: bool = false,
};

// Start of Scan
pub const SosInfo = struct {
    start_of_selection: u8 = 0,
    end_of_selection: u8 = 63,
    successive_approx_high: u8 = 0,
    successive_approx_low: u8 = 0,
};

// PARSING FUNCTIONS
// Quantization Tables parsing
pub fn parseDQT(segment: *JpegSegment, result: *JpegData) !void {
    while (segment.idx < segment.length - 1) {
        const info_byte = try segment.nextByte();
        const precision = info_byte >> 4;
        const id = info_byte & 0x0F;

        if (precision != 0 and precision != 1)
            return ImToTensorError.UnsupportedPrecision;

        var table_slice: [64]u8 = undefined;
        if (precision == 0) {
            for (0..64) |i| {
                table_slice[zigZagMap[i]] = try segment.nextByte();
            }
        } else {
            for (0..64) |i| {
                const byte1 = try segment.nextByte();
                const byte2 = try segment.nextByte();
                table_slice[zigZagMap[i]] = @as(u8, @intCast(std.mem.readInt(u16, &[2]u8{ byte1, byte2 }, .big)));
            }
        }

        result.quant_tables[id] = try QuantTable.init(table_slice, id);
    }
}

// Huffman Tables parsing
pub fn parseDHT(segment: *JpegSegment, result: *JpegData, allocator: std.mem.Allocator) !void {
    while (segment.idx < segment.length - 1) {
        if (segment.length - segment.idx < 17) {
            return ImToTensorError.SegmentTooShort;
        }

        const ht_info = try segment.nextByte();
        const class: u8 = ht_info >> 4; // upper 4 bits
        const id: u8 = @intCast(ht_info & 0x0F); // lower 4 bits

        if (class > 1 or id > 3) {
            return ImToTensorError.InvalidHuffmanTableId;
        }

        var total_symbols: u8 = 0;
        var code_lengths = try allocator.alloc(u8, 17);
        code_lengths[0] = 0;
        for (1..17) |i| {
            total_symbols += try segment.nextByte();
            code_lengths[i] = total_symbols;
        }

        if (total_symbols > 162)
            return ImToTensorError.TooManyHTSymbols;

        const expected_len = 1 + 16 + total_symbols;

        if (expected_len > segment.length) {
            return ImToTensorError.UnexpectedEndOfSegment;
        }

        var symbols = try allocator.alloc(u8, total_symbols);
        const codes = try allocator.alloc(u32, total_symbols);
        for (0..total_symbols) |i| {
            symbols[i] = try segment.nextByte();
        }
        //dc table
        if (class == 0) {
            result.huffman_tables_dc[id] = HuffmanTable{
                .class = class,
                .id = id,
                .code_lengths = code_lengths,
                .symbols = symbols,
                .codes = codes,
            };
        }
        //ac table
        else {
            result.huffman_tables_ac[id] = HuffmanTable{
                .class = class,
                .id = id,
                .code_lengths = code_lengths,
                .symbols = symbols,
                .codes = codes,
            };
        }
    }
}

// SOF parsing
// at the moment only the SOF0 format is supported (baseline). SOF2 (progressive) is not supported but can be easily implemented
pub fn parseSOF0(data: []u8, result: *JpegData) !void {
    if (data.len < 6)
        return ImToTensorError.SegmentTooShort;

    const precision = data[0];
    const height = std.mem.readInt(u16, data[1..3], .big);
    const width = std.mem.readInt(u16, data[3..5], .big);
    const components_num = data[5];

    if (components_num != 1 and components_num != 3) {
        return ImToTensorError.InvalidComponentNum;
    }

    const expected_length = 6 + components_num * 3;
    if (data.len < expected_length)
        return ImToTensorError.SegmentTooShort;

    result.frame_info.precision = precision;
    result.frame_info.height = height;
    result.frame_info.width = width;
    result.frame_info.components_num = components_num;

    result.mcu_height = (height + 7) / 8;
    result.mcu_width = (width + 7) / 8;
    result.mcu_true_height = result.mcu_height;
    result.mcu_true_width = result.mcu_width;

    for (0..components_num) |idx| {
        const offset = 6 + idx * 3;
        var id = data[offset];
        if (id == 0 or result.frame_info.zero_based) {
            id += 1;
            result.frame_info.zero_based = true;
        }

        const sampling_factors = data[offset + 1];
        const h_sampling = @as(u8, @intCast((sampling_factors >> 4) & 0x0F));
        const v_sampling = @as(u8, @intCast(sampling_factors & 0x0F));
        const quant_table_id = data[offset + 2];

        if (id == 1 or id == 0) {
            if ((h_sampling != 1 and h_sampling != 2) or (v_sampling != 1 and v_sampling != 2)) {
                return ImToTensorError.SamplingFactorsNotSupported;
            }
            if (h_sampling == 2 and result.mcu_width % 2 == 1) {
                result.mcu_true_width += 1;
            }
            if (v_sampling == 2 and result.mcu_height % 2 == 1) {
                result.mcu_true_height += 1;
            }
            result.horizontal_sampling_factor = h_sampling;
            result.vertical_sampling_factor = v_sampling;
        } else {
            if (h_sampling != 1 or v_sampling != 1) {
                return ImToTensorError.SamplingOnCbCrNotSupported;
            }
        }

        result.frame_info.components[idx] = ColorComponent{
            .id = id,
            .h_sampling_factor = h_sampling,
            .v_sampling_factor = v_sampling,
            .quant_table_id = quant_table_id,
        };
    }
}

// Start of Scan parsing
pub fn parseSOS(segment: *JpegSegment, decoder: *SegmentReader, result: *JpegData, allocator: std.mem.Allocator) !void {
    if (result.frame_info.components_num == 0) {
        return ImToTensorError.SosDetectedBeforeSof;
    }

    const components_num = try segment.nextByte();
    for (0..components_num) |_| {
        var component_id = try segment.nextByte();
        if (result.frame_info.zero_based) {
            component_id += 1;
        }
        if (component_id > result.frame_info.components_num) {
            return ImToTensorError.InvalidComponentNum;
        }
        if (result.frame_info.components[component_id - 1].used) {
            return ImToTensorError.InvalidComponent;
        }
        result.frame_info.components[component_id - 1].used = true;

        const ht_ids = try segment.nextByte();
        result.frame_info.components[component_id - 1].huffmanDCTableID = ht_ids >> 4;
        result.frame_info.components[component_id - 1].huffmanACTableID = ht_ids & 0x0F;

        if ((ht_ids >> 4) > 3 or (ht_ids & 0x0F) > 3) {
            return ImToTensorError.InvalidHuffmanTableId;
        }
    }

    result.sos_info.start_of_selection = try segment.nextByte();
    result.sos_info.end_of_selection = try segment.nextByte();
    const successive_approx = try segment.nextByte();
    result.sos_info.successive_approx_high = successive_approx >> 4;
    result.sos_info.successive_approx_low = successive_approx & 0x0F;

    if (result.sos_info.start_of_selection != 0 or result.sos_info.end_of_selection != 63) {
        return ImToTensorError.InvalidSpectralSeletion;
    }
    if (result.sos_info.successive_approx_high != 0 or result.sos_info.successive_approx_low != 0) {
        return ImToTensorError.InvalidSuccessiveApproximation;
    }

    result.huffman_data = try readSosData(decoder, allocator);
}

// Read Sos Data
pub fn readSosData(segment: *SegmentReader, allocator: std.mem.Allocator) ![]u8 {
    const data = segment.data[segment.idx..];
    var size: usize = 0;
    var i: usize = 0;

    while (i < data.len) : (i += 1) {
        const b = data[i];

        if (b != 0xFF) {
            size += 1;
            continue;
        }

        if (i + 1 >= data.len)
            return ImToTensorError.UnexpectedEof;

        const next = data[i + 1];

        if (next == @intFromEnum(jpegMarker.EOI)) { // FF D9  -> end of image
            break;
        } else if (next == 0x00) { // FF 00 -> byte stuffed, discard the 0x00
            size += 1;
            i += 1;
        } else if (next >= @intFromEnum(jpegMarker.RST0) and
            next <= @intFromEnum(jpegMarker.RST7))
        { // FF D0..D7 -> restart marker, discard the byte
            i += 1;
        } else {
            return ImToTensorError.UnexpectedMarker;
        }
    }

    var out = try allocator.alloc(u8, size);
    errdefer allocator.free(out);

    var idx: usize = 0;
    var last = try segment.nextByte();

    while (true) {
        const cur = try segment.nextByte();

        if (last == 0xFF) {
            if (cur == @intFromEnum(jpegMarker.EOI)) // FF D9  -> end of image
                break;

            if (cur == 0x00) { // FF 00 -> stuff byte, copy the 0xFF, skip the 0x00
                out[idx] = 0xFF;
                idx += 1;
                last = try segment.nextByte();
                continue;
            }

            if (cur >= @intFromEnum(jpegMarker.RST0) and
                cur <= @intFromEnum(jpegMarker.RST7))
            { // FF D0..D7 -> restart marker, discard the byte
                last = try segment.nextByte();
                continue;
            }

            return ImToTensorError.UnexpectedMarker;
        }

        out[idx] = last;
        idx += 1;
        last = cur;
    }

    return out[0..idx];
}

// main function to parse the header of a jpeg file takes in input the jpeg file and the result structure wich is updated with all the info parsed from the file
pub fn jpegParser(
    allocator: std.mem.Allocator,
    jpegDecoder: *SegmentReader,
) !JpegData {

    // Create the return structure with all the collected data
    var result = try JpegData.init(allocator);
    //defer result.deinit(allocator);
    // first unit contains the signature (SOI) so we can ignore it
    var unit = try jpegDecoder.nextUnit();
    var segment = unit.JpegSegment;

    while (segment.type != jpegMarker.EOI) {
        unit = try jpegDecoder.nextUnit();
        segment = unit.JpegSegment;
        switch (segment.type) {
            // parse quantization tables
            jpegMarker.DQT => {
                try parseDQT(&segment, &result);
            },

            // parse huffman tables
            jpegMarker.DHT => {
                try parseDHT(&segment, &result, allocator);
            },

            // parse start of frame
            jpegMarker.SOF0 => {
                try parseSOF0(segment.data, &result);
            },

            // parse SOS
            jpegMarker.SOS => {
                try parseSOS(&segment, jpegDecoder, &result, allocator);
                break;
            },

            // parse restart interval
            jpegMarker.DRI => {
                const temp = try jpegDecoder.tryAdvance(2);
                var twoBytes: [2]u8 = undefined;
                @memcpy(&twoBytes, temp);
                result.restart_interval = std.mem.readInt(u16, &twoBytes, .big);
            },

            // not supported markers
            else => {
                if (segment.type == jpegMarker.DAC) {
                    return ImToTensorError.ArithmeticEncodingNotSupported;
                } else if (@intFromEnum(segment.type) > @intFromEnum(jpegMarker.SOF0) and @intFromEnum(segment.type) <= @intFromEnum(jpegMarker.SOF15)) {
                    return ImToTensorError.SOFMarkerNotSupported;
                } else if (@intFromEnum(segment.type) >= @intFromEnum(jpegMarker.RST0) and @intFromEnum(segment.type) <= @intFromEnum(jpegMarker.RST7)) {
                    return ImToTensorError.RSTNDetectedBeforeSOS;
                }
            },
        }
    }
    return result;
}
