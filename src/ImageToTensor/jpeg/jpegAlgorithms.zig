const std = @import("std");
const utils = @import("../utils.zig");
const jpeg_parser = @import("jpegParser.zig");

const JpegData = jpeg_parser.JpegData;
const HuffmanTable = jpeg_parser.HuffmanTable;
const QuantTable = jpeg_parser.QuantTable;
const BitReader = utils.BitReader;
const ColorChannels = utils.ColorChannels;
const ImToTensorError = utils.ImToTensorError;

// -------------------------IDCT CONSTANTS-------------------------
const zigZagMap: [64]u8 = .{
    0,  1,  8,  16, 9,  2,  3,  10,
    17, 24, 32, 25, 18, 11, 4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6,  7,  14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
};

const pi = std.math.pi; // π as f64

// ----------------- const "m" (multiplicators) -----------------
pub const m0: f32 = @floatCast(2.0 * std.math.cos((1.0 / 16.0) * 2.0 * pi));

pub const m1: f32 = @floatCast(2.0 * std.math.cos((2.0 / 16.0) * 2.0 * pi));

pub const m3: f32 = m1;

pub const m5: f32 = @floatCast(2.0 * std.math.cos((3.0 / 16.0) * 2.0 * pi));

pub const m2: f32 = m0 - m5;
pub const m4: f32 = m0 + m5;

// ----------------- const "s" (scale / √8) ---------------------
pub const s0: f32 = @floatCast(std.math.cos((0.0 / 16.0) * pi) / std.math.sqrt(8.0));

pub const s1: f32 = @floatCast(std.math.cos((1.0 / 16.0) * pi) / 2.0);

pub const s2: f32 = @floatCast(std.math.cos((2.0 / 16.0) * pi) / 2.0);

pub const s3: f32 = @floatCast(std.math.cos((3.0 / 16.0) * pi) / 2.0);

pub const s4: f32 = @floatCast(std.math.cos((4.0 / 16.0) * pi) / 2.0);

pub const s5: f32 = @floatCast(std.math.cos((5.0 / 16.0) * pi) / 2.0);

pub const s6: f32 = @floatCast(std.math.cos((6.0 / 16.0) * pi) / 2.0);

pub const s7: f32 = @floatCast(std.math.cos((7.0 / 16.0) * pi) / 2.0);

// MCU is the minimum unit of data in JPEG
// it is a 8x8 block of pixels
// it contains 3 color channels (Y, Cb, Cr) for each pixel
pub const MCU = struct {
    y: []i32 = undefined,
    cb: []i32 = undefined,
    cr: []i32 = undefined,

    pub fn init(allocator: *const std.mem.Allocator, num_components: usize) !MCU {
        if (num_components == 0 or num_components > 3) {
            return ImToTensorError.InvalidComponentNum;
        }
        if (num_components == 1) {
            var mcu = MCU{
                .y = try allocator.alloc(i32, 64),
                .cb = try allocator.alloc(i32, 64),
                .cr = try allocator.alloc(i32, 64),
            };
            for (0..64) |i| {
                mcu.cb[i] = 0;
                mcu.cr[i] = 0;
            }
            return mcu;
        }

        return MCU{
            .y = try allocator.alloc(i32, 64),
            .cb = try allocator.alloc(i32, 64),
            .cr = try allocator.alloc(i32, 64),
        };
    }

    pub fn deinit(self: *MCU, allocator: *const std.mem.Allocator) void {
        allocator.free(self.y);
        allocator.free(self.cb);
        allocator.free(self.cr);
    }

    pub fn get(self: *MCU, index: usize) *[]i32 {
        return switch (index) {
            0 => &self.y,
            1 => &self.cb,
            2 => &self.cr,
            else => unreachable,
        };
    }
};

//------------------------------MCU to 3 CHANNELS-------------------------------------
// Converts pixel data from MCU to 3 color channels
pub fn writeChannels(header: JpegData, mcus: []MCU, allocator: *const std.mem.Allocator) !ColorChannels {
    //const mcuHeight: u16= (header.frame_info.height + 7) / 8;

    //const paddingSize = header.frame_info.width % 4;
    const channels_len: u32 = @as(u32, header.frame_info.height) * @as(u32, header.frame_info.width);
    var channels = try ColorChannels.init(allocator, channels_len, header.frame_info.components_num);
    channels.height = header.frame_info.height;
    channels.width = header.frame_info.width;
    channels.component_num = header.frame_info.components_num;

    var idx: u32 = 0;
    for (0..header.frame_info.height) |y| {
        const mcuRow = y / 8;
        const pixelRow = y % 8;
        for (0..header.frame_info.width) |x| {
            const mcuCol = x / 8;
            const pixelCol = x % 8;
            const mcuIndex = mcuRow * header.mcu_true_width + mcuCol;
            const pixelIndex = pixelRow * 8 + pixelCol;

            channels.ch1[idx] = @as(u8, @intCast(std.math.clamp(mcus[mcuIndex].y[pixelIndex], 0, 255)));

            if (channels.component_num == 3) {
                channels.ch2[idx] = @as(u8, @intCast(mcus[mcuIndex].cb[pixelIndex]));
                channels.ch3[idx] = @as(u8, @intCast(mcus[mcuIndex].cr[pixelIndex]));
            }
            idx += 1;
        }
    }
    return channels;
}

// -----------------------------HUFFMAN DECODING--------------------------------------
pub fn decodeHuffmanData(header: JpegData, allocator: *const std.mem.Allocator, mcus: []MCU) !void {
    for (0..header.mcu_true_height * header.mcu_true_width) |i| {
        mcus[i] = try MCU.init(allocator, header.frame_info.components_num);
    }

    // generate codes of huffman tables
    for (0..4) |i| {
        if (header.huffman_tables_dc[i] != null) {
            try generateCodes(&header.huffman_tables_dc[i].?);
        }
        if (header.huffman_tables_ac[i] != null) {
            try generateCodes(&header.huffman_tables_ac[i].?);
        }
    }

    // bitstream reader
    var b: BitReader = BitReader.init(header.huffman_data);

    var previousDCs: [3]i32 = .{ 0, 0, 0 };
    const restartInerval: u32 = @as(u32, header.restart_interval) * @as(u32, header.horizontal_sampling_factor) * @as(u32, header.vertical_sampling_factor);
    // loops through the MCUs decoding each one of them
    var y: usize = 0;
    while (y < header.mcu_height) {
        var x: usize = 0;
        while (x < header.mcu_width) {
            if (restartInerval != 0 and ((y * header.mcu_true_width + x) % restartInerval == 0)) {
                // std.debug.print("reset\n", .{}); DEBUG
                previousDCs[0] = 0;
                previousDCs[1] = 0;
                previousDCs[2] = 0;
                try b.bitAlign();
            }

            // these for loops dont increase the complexity (wich is still header.mcu_height * header.mcu_width)
            for (0..header.frame_info.components_num) |j| {
                for (0..header.frame_info.components[j].v_sampling_factor) |v| {
                    for (0..header.frame_info.components[j].h_sampling_factor) |h| {
                        try decodeMCUComponent(&b, mcus[(y + v) * header.mcu_true_width + (x + h)].get(j).*, &previousDCs[j], header.huffman_tables_dc[header.frame_info.components[j].huffmanDCTableID].?, header.huffman_tables_ac[header.frame_info.components[j].huffmanACTableID].?);
                    }
                }
            }
            x += header.horizontal_sampling_factor;
        }
        y += header.vertical_sampling_factor;
    }
}

// generates huffman codes based on symbols in huffman table
pub fn generateCodes(hTable: *HuffmanTable) !void {
    var code: u32 = 0;

    for (0..16) |i| {
        for (hTable.code_lengths[i]..hTable.code_lengths[i + 1]) |j| {
            hTable.codes[j] = code;
            code += 1;
        }
        code = code << 1;
    }
}

pub fn decodeMCUComponent(b: *BitReader, component: []i32, previousDC: *i32, dcTable: HuffmanTable, acTable: HuffmanTable) !void {
    const one: u32 = 1;

    // ---------- DC ----------
    const len: u8 = try getNextSymbol(b, dcTable);
    if (len == 0xFF) return ImToTensorError.InvalidDcValue;
    if (len > 11) return ImToTensorError.DcCoefficientLenghtGreaterThan11;

    var coeff = try b.readBits(len);
    if (coeff == -1) return ImToTensorError.InvalidDcValue;

    // calculate the shifts
    var len_m1_shift: u5 = 31;
    var len_shift: u5 = 0;

    if (len != 0) {
        len_m1_shift = @intCast(len - 1);
        len_shift = @intCast(len);
        if (coeff < (one << len_m1_shift))
            coeff -= @intCast((one << len_shift) - 1);
    }

    component[0] = @intCast(coeff + previousDC.*);
    previousDC.* = component[0];

    // ---------- AC ----------
    var i: usize = 1;
    while (i < 64) {
        const symbol: u8 = try getNextSymbol(b, acTable);
        if (symbol == 0xFF) return ImToTensorError.InvalidAcValue;

        if (symbol == 0x00) { // EOB
            for (i..64) |j| component[zigZagMap[j]] = 0;
            return;
        }

        var run: u8 = symbol >> 4;
        var size: u8 = symbol & 0x0F;

        if (symbol == 0xF0) { // ZRL
            run = 16;
            size = 0;
        } else if (size == 0) { // size == 0 allowed only in 0xF0
            return ImToTensorError.InvalidAcSymbol;
        }

        if (i + run > 64)
            return ImToTensorError.ZeroRunExceedeMCU;

        // fill the zeros
        for (0..run) |_| {
            component[zigZagMap[i]] = 0;
            i += 1;
        }

        if (size != 0) {
            if (size > 10) return ImToTensorError.DecodeAcCoefficientLenGreaterThan10;

            coeff = try b.readBits(size);
            if (coeff == -1) return ImToTensorError.InvalidAcValue;

            var size_m1_shift: u5 = 31;
            var size_shift: u5 = 0;

            if (size != 0) {
                size_m1_shift = @intCast(size - 1);
                size_shift = @intCast(size);
                if (coeff < (one << size_m1_shift))
                    coeff -= @intCast((one << size_shift) - 1);
            }
            component[zigZagMap[i]] = coeff;
            i += 1;
        }
    }
}

pub fn getNextSymbol(b: *BitReader, hTable: HuffmanTable) !u8 {
    var code_curr: usize = 0;
    for (0..16) |i| {
        const bit = try b.readBit();
        if (bit == -1) {
            return ImToTensorError.InvalidHuffmanCode;
        }
        code_curr = (code_curr << 1) | bit;
        for (hTable.code_lengths[i]..hTable.code_lengths[i + 1]) |j| {
            if (code_curr == hTable.codes[j]) {
                return hTable.symbols[j];
            }
        }
    }
    return ImToTensorError.InvalidHuffmanCode;
}

//------------------------------------DEQUANTIZATION-----------------------------------
pub fn dequantize(header: JpegData, mcus: []MCU) !void {
    var y: usize = 0;
    while (y < header.mcu_height) {
        var x: usize = 0;
        while (x < header.mcu_width) {
            for (0..header.frame_info.components_num) |j| {
                for (0..header.frame_info.components[j].v_sampling_factor) |v| {
                    for (0..header.frame_info.components[j].h_sampling_factor) |h| {
                        try dequantizeMCUComponent(header.quant_tables[header.frame_info.components[j].quant_table_id].?, mcus[(y + v) * header.mcu_true_width + (x + h)].get(j).*);
                    }
                }
            }
            x += header.horizontal_sampling_factor;
        }
        y += header.vertical_sampling_factor;
    }
}

pub fn dequantizeMCUComponent(qTable: QuantTable, component: []i32) !void {
    for (0..64) |i| {
        component[i] *= qTable.table[i];
    }
}

//---------------------------------------IDCT-----------------------------------------
pub fn inverseDCT(header: JpegData, mcus: []MCU) !void {
    var y: usize = 0;
    while (y < header.mcu_height) {
        var x: usize = 0;
        while (x < header.mcu_width) {
            for (0..header.frame_info.components_num) |j| {
                for (0..header.frame_info.components[j].v_sampling_factor) |v| {
                    for (0..header.frame_info.components[j].h_sampling_factor) |h| {
                        try inverseDCTComponent(mcus[(y + v) * header.mcu_true_width + (x + h)].get(j).*);
                    }
                }
            }
            x += header.horizontal_sampling_factor;
        }
        y += header.vertical_sampling_factor;
    }
}

pub fn inverseDCTComponent(component: []i32) !void {
    // temporary buffer for the transform on the columns
    var intermediate: [64]f32 = undefined;

    // ---------- first pass: columns ----------
    var i: usize = 0;
    while (i < 8) : (i += 1) {
        const g0: f32 = @as(f32, @floatFromInt(component[0 * 8 + i])) * s0;
        const g1: f32 = @as(f32, @floatFromInt(component[4 * 8 + i])) * s4;
        const g2: f32 = @as(f32, @floatFromInt(component[2 * 8 + i])) * s2;
        const g3: f32 = @as(f32, @floatFromInt(component[6 * 8 + i])) * s6;
        const g4: f32 = @as(f32, @floatFromInt(component[5 * 8 + i])) * s5;
        const g5: f32 = @as(f32, @floatFromInt(component[1 * 8 + i])) * s1;
        const g6: f32 = @as(f32, @floatFromInt(component[7 * 8 + i])) * s7;
        const g7: f32 = @as(f32, @floatFromInt(component[3 * 8 + i])) * s3;

        const f0: f32 = g0;
        const f1: f32 = g1;
        const f2: f32 = g2;
        const f3: f32 = g3;
        const f4: f32 = g4 - g7;
        const f5: f32 = g5 + g6;
        const f6: f32 = g5 - g6;
        const f7: f32 = g4 + g7;

        const e0: f32 = f0;
        const e1: f32 = f1;
        const e2: f32 = f2 - f3;
        const e3: f32 = f2 + f3;
        const e4: f32 = f4;
        const e5: f32 = f5 - f7;
        const e6: f32 = f6;
        const e7: f32 = f5 + f7;
        const e8: f32 = f4 + f6;

        const d0: f32 = e0;
        const d1: f32 = e1;
        const d2: f32 = e2 * m1;
        const d3: f32 = e3;
        const d4: f32 = e4 * m2;
        const d5: f32 = e5 * m3;
        const d6: f32 = e6 * m4;
        const d7: f32 = e7;
        const d8: f32 = e8 * m5;

        const c0: f32 = d0 + d1;
        const c1: f32 = d0 - d1;
        const c2: f32 = d2 - d3;
        const c3: f32 = d3;
        const c4: f32 = d4 + d8;
        const c5: f32 = d5 + d7;
        const c6: f32 = d6 - d8;
        const c7: f32 = d7;
        const c8: f32 = c5 - c6;

        const b0: f32 = c0 + c3;
        const b1: f32 = c1 + c2;
        const b2: f32 = c1 - c2;
        const b3: f32 = c0 - c3;
        const b4: f32 = c4 - c8;
        const b5: f32 = c8;
        const b6: f32 = c6 - c7;
        const b7: f32 = c7;

        intermediate[0 * 8 + i] = b0 + b7;
        intermediate[1 * 8 + i] = b1 + b6;
        intermediate[2 * 8 + i] = b2 + b5;
        intermediate[3 * 8 + i] = b3 + b4;
        intermediate[4 * 8 + i] = b3 - b4;
        intermediate[5 * 8 + i] = b2 - b5;
        intermediate[6 * 8 + i] = b1 - b6;
        intermediate[7 * 8 + i] = b0 - b7;
    }

    // ---------- second pass: rows ----------
    i = 0;
    while (i < 8) : (i += 1) {
        const g0: f32 = intermediate[i * 8 + 0] * s0;
        const g1: f32 = intermediate[i * 8 + 4] * s4;
        const g2: f32 = intermediate[i * 8 + 2] * s2;
        const g3: f32 = intermediate[i * 8 + 6] * s6;
        const g4: f32 = intermediate[i * 8 + 5] * s5;
        const g5: f32 = intermediate[i * 8 + 1] * s1;
        const g6: f32 = intermediate[i * 8 + 7] * s7;
        const g7: f32 = intermediate[i * 8 + 3] * s3;

        const f0: f32 = g0;
        const f1: f32 = g1;
        const f2: f32 = g2;
        const f3: f32 = g3;
        const f4: f32 = g4 - g7;
        const f5: f32 = g5 + g6;
        const f6: f32 = g5 - g6;
        const f7: f32 = g4 + g7;

        const e0: f32 = f0;
        const e1: f32 = f1;
        const e2: f32 = f2 - f3;
        const e3: f32 = f2 + f3;
        const e4: f32 = f4;
        const e5: f32 = f5 - f7;
        const e6: f32 = f6;
        const e7: f32 = f5 + f7;
        const e8: f32 = f4 + f6;

        const d0: f32 = e0;
        const d1: f32 = e1;
        const d2: f32 = e2 * m1;
        const d3: f32 = e3;
        const d4: f32 = e4 * m2;
        const d5: f32 = e5 * m3;
        const d6: f32 = e6 * m4;
        const d7: f32 = e7;
        const d8: f32 = e8 * m5;

        const c0: f32 = d0 + d1;
        const c1: f32 = d0 - d1;
        const c2: f32 = d2 - d3;
        const c3: f32 = d3;
        const c4: f32 = d4 + d8;
        const c5: f32 = d5 + d7;
        const c6: f32 = d6 - d8;
        const c7: f32 = d7;
        const c8: f32 = c5 - c6;

        const b0: f32 = c0 + c3;
        const b1: f32 = c1 + c2;
        const b2: f32 = c1 - c2;
        const b3: f32 = c0 - c3;
        const b4: f32 = c4 - c8;
        const b5: f32 = c8;
        const b6: f32 = c6 - c7;
        const b7: f32 = c7;

        // round + store (Arai/IJPG: +0.5 and cast a int)
        component[i * 8 + 0] = @as(i32, @intFromFloat(b0 + b7 + 0.5));
        component[i * 8 + 1] = @as(i32, @intFromFloat(b1 + b6 + 0.5));
        component[i * 8 + 2] = @as(i32, @intFromFloat(b2 + b5 + 0.5));
        component[i * 8 + 3] = @as(i32, @intFromFloat(b3 + b4 + 0.5));
        component[i * 8 + 4] = @as(i32, @intFromFloat(b3 - b4 + 0.5));
        component[i * 8 + 5] = @as(i32, @intFromFloat(b2 - b5 + 0.5));
        component[i * 8 + 6] = @as(i32, @intFromFloat(b1 - b6 + 0.5));
        component[i * 8 + 7] = @as(i32, @intFromFloat(b0 - b7 + 0.5));
    }
}

//----------------------------------COLORSPACE CONVERSION--------------------------------
// Convert YCbCr to RGB
pub fn yCbCrToRgb(header: JpegData, mcus: []MCU) !void {
    var y: usize = 0;
    while (y < header.mcu_height) {
        var x: usize = 0;
        while (x < header.mcu_width) {
            const cbcr = mcus[y * header.mcu_true_width + x];
            var v = header.vertical_sampling_factor - 1;
            while (v >= 0) : (v -= 1) {
                var h = header.horizontal_sampling_factor - 1;
                while (h >= 0) : (h -= 1) {
                    const mcu = mcus[(y + v) * header.mcu_true_width + (x + h)];
                    yCbCrToRgbMCU(header, mcu, cbcr, v, h);
                    if (h == 0) {
                        break;
                    }
                }
                if (v == 0) {
                    break;
                }
            }
            x += header.horizontal_sampling_factor;
        }
        y += header.vertical_sampling_factor;
    }
}

// Convert YCbCr to RGB for a single MCU and upsample
pub fn yCbCrToRgbMCU(header: JpegData, mcu: MCU, cbcr: MCU, v: usize, h: usize) void {
    var y: usize = 7;
    while (y >= 0) : (y -= 1) {
        var x: usize = 7;
        while (x >= 0) : (x -= 1) {
            const pixel = y * 8 + x;

            const cbcr_pixel_row = y / header.vertical_sampling_factor + 4 * v;
            const cbcr_pixel_col = x / header.horizontal_sampling_factor + 4 * h;
            const cbcr_pixel = cbcr_pixel_row * 8 + cbcr_pixel_col;

            var r_temp = mcu.y[pixel] + @as(i32, @intFromFloat(1.402 * @as(f32, @floatFromInt(cbcr.cr[cbcr_pixel]))));
            r_temp += 128;
            var g_temp = mcu.y[pixel] - @as(i32, @intFromFloat(0.34414 * @as(f32, @floatFromInt(cbcr.cb[cbcr_pixel])))) - @as(i32, @intFromFloat(0.71414 * @as(f32, @floatFromInt(cbcr.cr[cbcr_pixel]))));
            g_temp += 128;

            var b_temp = mcu.y[pixel] + @as(i32, @intFromFloat(1.772 * @as(f32, @floatFromInt(cbcr.cb[cbcr_pixel]))));
            b_temp += 128;

            r_temp = std.math.clamp(r_temp, 0, 255);
            g_temp = std.math.clamp(g_temp, 0, 255);
            b_temp = std.math.clamp(b_temp, 0, 255);

            mcu.y[pixel] = r_temp;
            mcu.cb[pixel] = g_temp;
            mcu.cr[pixel] = b_temp;
            if (x == 0) {
                break;
            }
        }
        if (y == 0) {
            break;
        }
    }
}

//----------------------------------YCbCr Upsampling--------------------------------
pub fn yCbCrUpsampling(header: JpegData, mcus: []MCU) !void {
    var y: usize = 0;
    while (y < header.mcu_height) {
        var x: usize = 0;
        while (x < header.mcu_width) {
            const cbcr = mcus[y * header.mcu_true_width + x];
            var v = header.vertical_sampling_factor - 1;
            while (v >= 0) : (v -= 1) {
                var h = header.horizontal_sampling_factor - 1;
                while (h >= 0) : (h -= 1) {
                    const mcu = mcus[(y + v) * header.mcu_true_width + (x + h)];
                    yCbCrUpsamplingMCU(header, mcu, cbcr, v, h);
                    if (h == 0) {
                        break;
                    }
                }
                if (v == 0) {
                    break;
                }
            }
            x += header.horizontal_sampling_factor;
        }
        y += header.vertical_sampling_factor;
    }
}

// Convert YCbCr to RGB for a single MCU
pub fn yCbCrUpsamplingMCU(header: JpegData, mcu: MCU, cbcr: MCU, v: usize, h: usize) void {
    var y: usize = 7;
    while (y >= 0) : (y -= 1) {
        var x: usize = 7;
        while (x >= 0) : (x -= 1) {
            const pixel = y * 8 + x;

            const cbcr_pixel_row = y / header.vertical_sampling_factor + 4 * v;
            const cbcr_pixel_col = x / header.horizontal_sampling_factor + 4 * h;
            const cbcr_pixel = cbcr_pixel_row * 8 + cbcr_pixel_col;

            mcu.cb[pixel] = cbcr.cb[cbcr_pixel] + 128;
            mcu.cr[pixel] = cbcr.cr[cbcr_pixel] + 128;
            mcu.y[pixel] = mcu.y[pixel] + 128;
            mcu.cb[pixel] = std.math.clamp(mcu.cb[pixel], 0, 255);
            mcu.cr[pixel] = std.math.clamp(mcu.cr[pixel], 0, 255);
            mcu.y[pixel] = std.math.clamp(mcu.y[pixel], 0, 255);
            if (x == 0) {
                break;
            }
        }
        if (y == 0) {
            break;
        }
    }
}
