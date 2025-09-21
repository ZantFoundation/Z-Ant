const std = @import("std");
const zant = @import("../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const pkg_allocator = zant.utils.allocator.allocator;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const AcceleratorView = zant.core.tensor.AcceleratorView;
const AcceleratorLayout = zant.core.tensor.AcceleratorLayout;
const AcceleratorBufferKind = zant.core.tensor.AcceleratorBufferKind;

// Import existing conv operation to reuse shape calculation and structure
const conv = @import("op_convolution.zig");

fn toF32(comptime T: type, v: T) f32 {
    return switch (@typeInfo(T)) {
        .float => @as(f32, @floatCast(v)),
        .int, .comptime_int => @as(f32, @floatFromInt(v)),
        else => @compileError("Unsupported type for float cast"),
    };
}

fn tensorDataPtrOrZero(comptime T: type, tensor: *const Tensor(T)) usize {
    if (tensor.data.len == 0) return 0;
    return @intFromPtr(tensor.data.ptr);
}

fn zeroPointTag(zp_any: anytype) usize {
    const ZPType = @TypeOf(zp_any);
    const info = @typeInfo(ZPType);
    return switch (info) {
        .pointer => switch (info.pointer.size) {
            .one => zeroPointTag(zp_any.*),
            .slice => if (zp_any.len == 0) 0 else @intFromPtr(zp_any.ptr),
            .many, .c => @intFromPtr(zp_any),
        },
        .optional => if (zp_any) |payload| zeroPointTag(payload) else 0,
        .array => if (info.array.len == 0) 0 else @intFromPtr(&zp_any),
        .vector => if (info.vector.len == 0) 0 else @intFromPtr(&zp_any),
        .@"struct" => if (@hasField(ZPType, "data")) {
            const data = zp_any.data;
            return if (data.len == 0) 0 else @intFromPtr(data.ptr);
        } else 0,
        .int, .comptime_int => @as(usize, @intCast(zp_any)),
        else => 0,
    };
}

const CacheKey = struct {
    weight_ptr: usize,
    x_scale_ptr: usize,
    y_scale_ptr: usize,
    w_scale_ptr: usize,
    bias_ptr: usize,
    w_zero_point_tag: usize,
    group: usize,
    out_channels: usize,
    weight_in_channels: usize,
    kernel_height: usize,
    kernel_width: usize,
};

const CacheKeyContext = struct {
    pub fn hash(_: CacheKeyContext, key: CacheKey) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(std.mem.asBytes(&key.weight_ptr));
        hasher.update(std.mem.asBytes(&key.x_scale_ptr));
        hasher.update(std.mem.asBytes(&key.y_scale_ptr));
        hasher.update(std.mem.asBytes(&key.w_scale_ptr));
        hasher.update(std.mem.asBytes(&key.bias_ptr));
        hasher.update(std.mem.asBytes(&key.w_zero_point_tag));
        hasher.update(std.mem.asBytes(&key.group));
        hasher.update(std.mem.asBytes(&key.out_channels));
        hasher.update(std.mem.asBytes(&key.weight_in_channels));
        hasher.update(std.mem.asBytes(&key.kernel_height));
        hasher.update(std.mem.asBytes(&key.kernel_width));
        return hasher.final();
    }

    pub fn eql(_: CacheKeyContext, a: CacheKey, b: CacheKey) bool {
        return a.weight_ptr == b.weight_ptr and
            a.x_scale_ptr == b.x_scale_ptr and
            a.y_scale_ptr == b.y_scale_ptr and
            a.w_scale_ptr == b.w_scale_ptr and
            a.bias_ptr == b.bias_ptr and
            a.w_zero_point_tag == b.w_zero_point_tag and
            a.group == b.group and
            a.out_channels == b.out_channels and
            a.weight_in_channels == b.weight_in_channels and
            a.kernel_height == b.kernel_height and
            a.kernel_width == b.kernel_width;
    }
};

const CmsisConvCacheEntry = struct {
    id: usize = 0,
    out_channels: usize = 0,
    weight_in_channels: usize = 0,
    kernel_height: usize = 0,
    kernel_width: usize = 0,
    group: usize = 0,
    multipliers: []i32 = &[_]i32{},
    shifts: []i32 = &[_]i32{},
    packed_weights: []i8 = &[_]i8{},
    bias: ?[]i32 = null,
    scratch: []u8 = &[_]u8{},
    input_buffer: []i8 = &[_]i8{},
    output_buffer: []i8 = &[_]i8{},
    grouped_input: []i8 = &[_]i8{},
    grouped_output: []i8 = &[_]i8{},
    x_scale_ptr: usize = 0,
    y_scale_ptr: usize = 0,
    w_scale_ptr: usize = 0,
    bias_ptr: usize = 0,
    w_zero_point_tag: usize = 0,
    x_scale_val: f32 = 0,
    y_scale_val: f32 = 0,
    is_u8_input: bool = false,
    output_generation: usize = 0,

    fn freeQuant(self: *CmsisConvCacheEntry) void {
        if (self.multipliers.len > 0) pkg_allocator.free(self.multipliers);
        if (self.shifts.len > 0) pkg_allocator.free(self.shifts);
        if (self.packed_weights.len > 0) pkg_allocator.free(self.packed_weights);
        if (self.bias) |b| {
            if (b.len > 0) pkg_allocator.free(b);
        }
        self.multipliers = &[_]i32{};
        self.shifts = &[_]i32{};
        self.packed_weights = &[_]i8{};
        self.bias = null;
    }

    fn ensurePrepacked(
        self: *CmsisConvCacheEntry,
        comptime InputType: type,
        comptime WeightType: type,
        comptime ScaleType: type,
        comptime BiasType: type,
        key: CacheKey,
        w: *const Tensor(WeightType),
        w_scale: *const Tensor(ScaleType),
        w_zero_point_any: anytype,
        bias_tensor: ?*const Tensor(BiasType),
        x_scale_val: f32,
        y_scale_val: f32,
        is_u8_input: bool,
    ) !void {
        const dims_changed = self.out_channels != key.out_channels or
            self.weight_in_channels != key.weight_in_channels or
            self.kernel_height != key.kernel_height or
            self.kernel_width != key.kernel_width or
            self.group != key.group;

        if (dims_changed) {
            self.freeQuant();
            const total_weights = key.out_channels * key.weight_in_channels * key.kernel_height * key.kernel_width;
            self.multipliers = try pkg_allocator.alloc(i32, key.out_channels);
            self.shifts = try pkg_allocator.alloc(i32, key.out_channels);
            self.packed_weights = try pkg_allocator.alloc(i8, total_weights);
            self.out_channels = key.out_channels;
            self.weight_in_channels = key.weight_in_channels;
            self.kernel_height = key.kernel_height;
            self.kernel_width = key.kernel_width;
            self.group = key.group;
        }

        const quant_params_changed = dims_changed or
            self.x_scale_ptr != key.x_scale_ptr or
            self.y_scale_ptr != key.y_scale_ptr or
            self.w_scale_ptr != key.w_scale_ptr or
            self.x_scale_val != x_scale_val or
            self.y_scale_val != y_scale_val;

        if (quant_params_changed) {
            for (0..key.out_channels) |ch| {
                const scale_index = if (w_scale.data.len == key.out_channels) ch else 0;
                const w_scale_val = toF32(ScaleType, w_scale.data[scale_index]);
                const scale_ratio = (x_scale_val * w_scale_val) / y_scale_val;
                quantizeMultiplier(scale_ratio, &self.multipliers[ch], &self.shifts[ch]);
            }
            self.x_scale_ptr = key.x_scale_ptr;
            self.y_scale_ptr = key.y_scale_ptr;
            self.w_scale_ptr = key.w_scale_ptr;
            self.x_scale_val = x_scale_val;
            self.y_scale_val = y_scale_val;
        }

        const weights_changed = dims_changed or self.w_zero_point_tag != key.w_zero_point_tag;
        if (weights_changed) {
            const total_weights = key.out_channels * key.weight_in_channels * key.kernel_height * key.kernel_width;
            if (self.packed_weights.len != total_weights) {
                if (self.packed_weights.len > 0) pkg_allocator.free(self.packed_weights);
                self.packed_weights = try pkg_allocator.alloc(i8, total_weights);
            }
            if (key.group == 1) {
                var wp: usize = 0;
                for (0..key.out_channels) |m| {
                    const w_zp_m: i32 = readPerChannelZP(w_zero_point_any, m, key.out_channels);
                    for (0..key.kernel_height) |kh| {
                        for (0..key.kernel_width) |kw| {
                            for (0..key.weight_in_channels) |c| {
                                const weight_idx = ((m * key.weight_in_channels + c) * key.kernel_height + kh) * key.kernel_width + kw;
                                const w_q_i32 = @as(i32, @intCast(w.data[weight_idx]));
                                var val = w_q_i32 - w_zp_m;
                                if (val < -128) val = -128;
                                if (val > 127) val = 127;
                                self.packed_weights[wp] = @as(i8, @intCast(val));
                                wp += 1;
                            }
                        }
                    }
                }
            } else if (key.weight_in_channels == 1 and key.group > 1) {
                var wp: usize = 0;
                for (0..key.kernel_height) |kh| {
                    for (0..key.kernel_width) |kw| {
                        for (0..key.out_channels) |m| {
                            const w_zp_m: i32 = readPerChannelZP(w_zero_point_any, m, key.out_channels);
                            const weight_idx = ((m * key.weight_in_channels + 0) * key.kernel_height + kh) * key.kernel_width + kw;
                            const w_q_i32 = @as(i32, @intCast(w.data[weight_idx]));
                            var val = w_q_i32 - w_zp_m;
                            if (val < -128) val = -128;
                            if (val > 127) val = 127;
                            self.packed_weights[wp] = @as(i8, @intCast(val));
                            wp += 1;
                        }
                    }
                }
            } else {
                var wp: usize = 0;
                for (0..key.out_channels) |m| {
                    const w_zp_m: i32 = readPerChannelZP(w_zero_point_any, m, key.out_channels);
                    for (0..key.kernel_height) |kh| {
                        for (0..key.kernel_width) |kw| {
                            for (0..key.weight_in_channels) |c| {
                                const weight_idx = ((m * key.weight_in_channels + c) * key.kernel_height + kh) * key.kernel_width + kw;
                                const w_q_i32 = @as(i32, @intCast(w.data[weight_idx]));
                                var val = w_q_i32 - w_zp_m;
                                if (val < -128) val = -128;
                                if (val > 127) val = 127;
                                self.packed_weights[wp] = @as(i8, @intCast(val));
                                wp += 1;
                            }
                        }
                    }
                }
            }
            self.w_zero_point_tag = key.w_zero_point_tag;
        }

        const bias_changed = quant_params_changed or dims_changed or self.bias_ptr != key.bias_ptr;
        if (bias_tensor) |b_tensor| {
            if (self.bias == null or self.bias.?.len != key.out_channels or bias_changed) {
                if (self.bias) |old| {
                    if (old.len > 0) pkg_allocator.free(old);
                }
                self.bias = try pkg_allocator.alloc(i32, key.out_channels);
            }
            var bias_slice = self.bias.?;
            for (0..key.out_channels) |ch| {
                const bias_val = if (b_tensor.data.len == 1) b_tensor.data[0] else b_tensor.data[ch];
                const scale_index = if (w_scale.data.len == key.out_channels) ch else 0;
                const w_scale_val = toF32(ScaleType, w_scale.data[scale_index]);
                const bias_scale = x_scale_val * w_scale_val;
                const bias_float = toF32(BiasType, bias_val);
                const bias_quantized = @as(i32, @intFromFloat(@round(bias_float / bias_scale)));
                bias_slice[ch] = bias_quantized;
            }
            self.bias_ptr = key.bias_ptr;
        } else if (self.bias) |old| {
            if (old.len > 0) pkg_allocator.free(old);
            self.bias = null;
            self.bias_ptr = 0;
        }

        self.is_u8_input = is_u8_input;
    }

    fn ensureScratch(self: *CmsisConvCacheEntry, size: usize) !void {
        if (size == 0) {
            if (self.scratch.len > 0) {
                pkg_allocator.free(self.scratch);
                self.scratch = &[_]u8{};
            }
            return;
        }
        if (self.scratch.len != size) {
            if (self.scratch.len > 0) pkg_allocator.free(self.scratch);
            self.scratch = try pkg_allocator.alloc(u8, size);
        }
    }

    fn ensureInputBuffer(self: *CmsisConvCacheEntry, len: usize) ![]i8 {
        if (len == 0) {
            if (self.input_buffer.len > 0) {
                pkg_allocator.free(self.input_buffer);
            }
            self.input_buffer = &[_]i8{};
            return self.input_buffer;
        }
        if (self.input_buffer.len != len) {
            if (self.input_buffer.len > 0) pkg_allocator.free(self.input_buffer);
            self.input_buffer = try pkg_allocator.alloc(i8, len);
        }
        return self.input_buffer;
    }

    fn ensureOutputBuffer(self: *CmsisConvCacheEntry, len: usize) ![]i8 {
        if (len == 0) {
            if (self.output_buffer.len > 0) {
                pkg_allocator.free(self.output_buffer);
                self.output_generation += 1;
            }
            self.output_buffer = &[_]i8{};
            return self.output_buffer;
        }
        if (self.output_buffer.len != len) {
            if (self.output_buffer.len > 0) pkg_allocator.free(self.output_buffer);
            self.output_buffer = try pkg_allocator.alloc(i8, len);
            self.output_generation += 1;
        }
        return self.output_buffer;
    }

    fn ensureGroupedBuffers(self: *CmsisConvCacheEntry, in_len: usize, out_len: usize) !void {
        if (in_len == 0) {
            if (self.grouped_input.len > 0) {
                pkg_allocator.free(self.grouped_input);
                self.grouped_input = &[_]i8{};
            }
        } else if (self.grouped_input.len != in_len) {
            if (self.grouped_input.len > 0) pkg_allocator.free(self.grouped_input);
            self.grouped_input = try pkg_allocator.alloc(i8, in_len);
        }

        if (out_len == 0) {
            if (self.grouped_output.len > 0) {
                pkg_allocator.free(self.grouped_output);
                self.grouped_output = &[_]i8{};
            }
        } else if (self.grouped_output.len != out_len) {
            if (self.grouped_output.len > 0) pkg_allocator.free(self.grouped_output);
            self.grouped_output = try pkg_allocator.alloc(i8, out_len);
        }
    }
};

const CacheMap = std.HashMap(CacheKey, CmsisConvCacheEntry, CacheKeyContext, std.hash_map.default_max_load_percentage);

var cmsis_conv_cache = CacheMap.init(pkg_allocator);
var cache_entry_counter: usize = 1;

fn findCacheEntryById(id: usize) ?*CmsisConvCacheEntry {
    var it = cmsis_conv_cache.iterator();
    while (it.next()) |entry| {
        if (entry.value_ptr.id == id) {
            return entry.value_ptr;
        }
    }
    return null;
}

fn getOrCreateCacheEntry(
    comptime InputType: type,
    comptime WeightType: type,
    comptime ScaleType: type,
    comptime BiasType: type,
    key: CacheKey,
    w: *const Tensor(WeightType),
    w_scale: *const Tensor(ScaleType),
    w_zero_point_any: anytype,
    bias: ?*const Tensor(BiasType),
    x_scale_val: f32,
    y_scale_val: f32,
    is_u8_input: bool,
) !*CmsisConvCacheEntry {
    const entry_result = try cmsis_conv_cache.getOrPut(key);
    if (!entry_result.found_existing) {
        entry_result.value_ptr.* = CmsisConvCacheEntry{};
        entry_result.value_ptr.id = cache_entry_counter;
        cache_entry_counter += 1;
    }
    var entry = entry_result.value_ptr;
    try entry.ensurePrepacked(InputType, WeightType, ScaleType, BiasType, key, w, w_scale, w_zero_point_any, bias, x_scale_val, y_scale_val, is_u8_input);
    return entry;
}

// HELPER FUNCTIONS FOR CORRECT QUANTIZATION
inline fn readScalarZP(comptime T: type, zp_any: anytype) i32 {
    _ = T;
    return readScalarZPInternal(zp_any);
}

fn readScalarZPInternal(zp_any: anytype) i32 {
    const ZPType = @TypeOf(zp_any);
    const info = @typeInfo(ZPType);

    return switch (info) {
        .pointer => switch (info.pointer.size) {
            .one => readScalarZPInternal(zp_any.*),
            .slice => blk: {
                if (zp_any.len == 0) break :blk 0;
                break :blk @as(i32, @intCast(zp_any[0]));
            },
            .many, .c => blk: {
                // Treat bare pointers as single-value buffers and read the first element.
                break :blk @as(i32, @intCast(zp_any[0]));
            },
        },
        .optional => if (zp_any) |payload| readScalarZPInternal(payload) else 0,
        .array => blk: {
            if (info.array.len == 0) break :blk 0;
            break :blk @as(i32, @intCast(zp_any[0]));
        },
        .vector => blk: {
            if (info.vector.len == 0) break :blk 0;
            break :blk @as(i32, @intCast(zp_any[0]));
        },
        .@"struct" => if (@hasField(ZPType, "data")) blk: {
            const data = zp_any.data;
            if (data.len == 0) break :blk 0;
            break :blk @as(i32, @intCast(data[0]));
        } else @compileError("unsupported zero-point struct representation"),
        .int, .comptime_int => @as(i32, @intCast(zp_any)),
        else => @compileError("unsupported zero-point representation"),
    };
}

inline fn readPerChannelScale(comptime T: type, s: *const Tensor(T), m: usize, M: usize) f32 {
    if (s.shape.len == 1 and s.shape[0] == M) return @as(f32, @floatCast(s.data[m]));
    return @as(f32, @floatCast(s.data[0])); // broadcast
}

inline fn selectChannelIndex(len: usize, channel: usize) usize {
    if (len <= 1) return 0;
    return if (channel < len) channel else len - 1;
}

inline fn readPerChannelZP(zp_any: anytype, m: usize, M: usize) i32 {
    _ = M;
    return readPerChannelZPInternal(zp_any, m);
}

fn readPerChannelZPInternal(zp_any: anytype, m: usize) i32 {
    const ZPType = @TypeOf(zp_any);
    const info = @typeInfo(ZPType);

    return switch (info) {
        .pointer => switch (info.pointer.size) {
            .one => readPerChannelZPInternal(zp_any.*, m),
            .slice => blk: {
                if (zp_any.len == 0) break :blk 0;
                const idx = selectChannelIndex(zp_any.len, m);
                break :blk @as(i32, @intCast(zp_any[idx]));
            },
            .many, .c => @compileError("unsupported zero-point pointer representation"),
        },
        .optional => if (zp_any) |payload| readPerChannelZPInternal(payload, m) else 0,
        .array => blk: {
            if (info.array.len == 0) break :blk 0;
            const idx = selectChannelIndex(info.array.len, m);
            break :blk @as(i32, @intCast(zp_any[idx]));
        },
        .vector => blk: {
            if (info.vector.len == 0) break :blk 0;
            const idx = selectChannelIndex(info.vector.len, m);
            break :blk @as(i32, @intCast(zp_any[idx]));
        },
        .@"struct" => if (@hasField(ZPType, "data")) blk: {
            const data = zp_any.data;
            if (data.len == 0) break :blk 0;
            const idx = selectChannelIndex(data.len, m);
            break :blk @as(i32, @intCast(data[idx]));
        } else @compileError("unsupported zero-point struct representation"),
        .int, .comptime_int => @as(i32, @intCast(zp_any)),
        else => @compileError("unsupported zero-point representation"),
    };
}

const SHIFT: u5 = 16;
inline fn q16(x: f32) i32 {
    return @as(i32, @intFromFloat(x * @as(f32, @floatFromInt(@as(u32, 1) << SHIFT))));
}

inline fn rshift_round_s64(x: i64, comptime shift_bits: u5) i64 {
    const bias: i64 = if (x >= 0) (1 << (shift_bits - 1)) else -(1 << (shift_bits - 1));
    return (x + bias) >> shift_bits;
}

/// QLinearConv operation following ONNX specification
/// Performs quantized convolution using linear quantization scheme
///
/// INPUTS:
/// - x: quantized input tensor (typically int8/uint8)
/// - x_scale: scale factor for input quantization
/// - x_zero_point: zero point for input quantization
/// - w: quantized weight tensor
/// - w_scale: scale factor for weight quantization
/// - w_zero_point: zero point for weight quantization
/// - y_scale: scale factor for output quantization
/// - y_zero_point: zero point for output quantization
/// - bias: optional bias tensor (can be null)
///
/// OUTPUT:
/// - y: quantized output tensor
///
/// Formula: quantized_output = quantize(conv(dequantize(x), dequantize(w)) + bias, y_scale, y_zero_point)
pub fn qlinearconv(
    comptime InputType: anytype,
    comptime WeightType: anytype,
    comptime ScaleType: anytype,
    comptime ZeroPointType: anytype,
    comptime BiasType: anytype,
    x: *const Tensor(InputType),
    x_scale: *const Tensor(ScaleType),
    x_zero_point: *const Tensor(ZeroPointType),
    w: *const Tensor(WeightType),
    w_scale: *const Tensor(ScaleType),
    w_zero_point: *const Tensor(ZeroPointType),
    y_scale: *const Tensor(ScaleType),
    y_zero_point: *const Tensor(ZeroPointType),
    bias: ?*const Tensor(BiasType),
    // Convolution parameters
    stride: ?[]const usize,
    pads: ?[]const usize,
    dilations: ?[]const usize,
    group: ?usize,
    auto_pad: ?[]const u8,
) !Tensor(InputType) {
    // Input validation
    if (x.shape.len != 3 and x.shape.len != 4) {
        return TensorMathError.InvalidDimensions;
    }
    if (w.shape.len != 4) {
        return TensorMathError.InvalidDimensions;
    }

    // Handle 3D input by assuming batch size = 1
    var input_shape: [4]usize = undefined;
    var temp_input: ?Tensor(InputType) = null;
    var input_ptr = x;

    if (x.shape.len == 3) {
        input_shape[0] = 1; // batch
        input_shape[1] = x.shape[0]; // channels
        input_shape[2] = x.shape[1]; // height
        input_shape[3] = x.shape[2]; // width

        const temp = try Tensor(InputType).fromArray(&pkg_allocator, x.data, &input_shape);
        temp_input = temp;
        input_ptr = &temp_input.?;
    } else {
        @memcpy(&input_shape, x.shape[0..4]);
    }
    defer if (temp_input) |*t| t.deinit();

    // Calculate output shape using existing conv calculation
    const output_shape = try conv.calculateOutputShape(InputType, &input_shape, w.shape, stride, pads, dilations, auto_pad);

    // Create output tensor
    var output = try Tensor(InputType).fromShape(&pkg_allocator, &output_shape);
    errdefer output.deinit();

    // Perform quantized convolution
    try qlinearconv_lean(InputType, WeightType, ScaleType, ZeroPointType, BiasType, input_ptr, x_scale, x_zero_point, w, w_scale, w_zero_point, &output, y_scale, y_zero_point, bias, stride, pads, dilations, group, auto_pad.?);

    return output;
}

/// OTTIMIZZATO: Lean version of QLinearConv with pre-computation and cache optimizations
pub fn qlinearconv_lean(
    comptime InputType: anytype,
    comptime WeightType: anytype,
    comptime ScaleType: anytype,
    comptime _: anytype, // ZeroPointType unused due to anytype zero_point parameters
    comptime BiasType: anytype,
    x: *const Tensor(InputType),
    x_scale: *const Tensor(ScaleType),
    x_zero_point: anytype, // Accept any tensor type for zero_point
    w: *const Tensor(WeightType),
    w_scale: *const Tensor(ScaleType),
    w_zero_point: anytype, // Accept any tensor type for zero_point
    output: *Tensor(InputType),
    y_scale: *const Tensor(ScaleType),
    y_zero_point: anytype, // Accept any tensor type for zero_point
    bias: ?*const Tensor(BiasType),
    stride: ?[]const usize,
    pads: ?[]const usize,
    dilations: ?[]const usize,
    group: ?usize,
    auto_pad: []const u8,
) !void {
    _ = auto_pad; // non gestito: usare pads espliciti

    // Check tensor shapes
    if (x.shape.len != 4 or w.shape.len != 4 or output.shape.len != 4) {
        // std.log.err("QLinearConv: InvalidDimensions x.shape={any} w.shape={any} y.shape={any}", .{ x.shape, w.shape, output.shape });
        return TensorMathError.InvalidDimensions;
    }

    const isInt = struct {
        fn call(comptime T: type) bool {
            return switch (@typeInfo(T)) {
                .int, .comptime_int => true,
                else => false,
            };
        }
    }.call;

    const asF32 = struct {
        fn call(comptime T: type, v: T) f32 {
            return switch (@typeInfo(T)) {
                .float => @as(f32, @floatCast(v)),
                .int, .comptime_int => @as(f32, @floatFromInt(v)),
                else => @compileError("Unsupported type for float cast"),
            };
        }
    }.call;

    // Handle scalar tensors (shape [1])
    if (x.shape.len == 1 and x.shape[0] == 1 and output.shape.len == 1 and output.shape[0] == 1) {
        output.data[0] = if (isInt(InputType)) @as(InputType, 0) else @as(InputType, 0.0);
        return;
    }

    // Estrai dimensioni
    const batch_size = x.shape[0]; // N
    const in_channels = x.shape[1]; // C
    const in_height = x.shape[2]; // H
    const in_width = x.shape[3]; // W

    const out_channels = w.shape[0]; // M
    const weight_in_channels = w.shape[1]; // C/group
    const kernel_height = w.shape[2]; // kH
    const kernel_width = w.shape[3]; // kW

    const out_height = output.shape[2]; // oH
    const out_width = output.shape[3]; // oW

    // Parametri
    const actual_group = group orelse 1;
    const stride_h = if (stride) |s| (if (s.len > 0) s[0] else 1) else 1;
    const stride_w = if (stride) |s| (if (s.len > 1) s[1] else stride_h) else stride_h;
    const dilation_h = if (dilations) |d| (if (d.len > 0) d[0] else 1) else 1;
    const dilation_w = if (dilations) |d| (if (d.len > 1) d[1] else dilation_h) else dilation_h;

    // Gruppi validation
    if (in_channels % actual_group != 0) {
        return TensorMathError.InvalidGroupParameter;
    }
    if (out_channels % actual_group != 0) {
        return TensorMathError.InvalidGroupParameter;
    }
    if (weight_in_channels != in_channels / actual_group) {
        return TensorMathError.InvalidDimensions;
    }

    // Padding
    var pad_h_begin: usize = 0;
    var pad_w_begin: usize = 0;
    if (pads) |p| {
        if (p.len >= 2) {
            pad_h_begin = p[0];
            pad_w_begin = p[1];
        }
    }

    // ===== OTTIMIZZAZIONE 1: Pre-calcolo scale e bias =====
    const x_scale_val: f32 = asF32(ScaleType, x_scale.data[0]);
    const y_scale_val: f32 = asF32(ScaleType, y_scale.data[0]);
    const x_zp_f: f32 = if (x_zero_point.data.len > 0) asF32(@TypeOf(x_zero_point.data[0]), x_zero_point.data[0]) else 0.0;
    const y_zp_f: f32 = if (y_zero_point.data.len > 0) asF32(@TypeOf(y_zero_point.data[0]), y_zero_point.data[0]) else 0.0;

    // Pre-calcola scale e bias per tutti i canali output (evita calcoli ridondanti)
    var channel_scales = std.ArrayList(f32).init(pkg_allocator);
    defer channel_scales.deinit();
    var channel_zps = std.ArrayList(f32).init(pkg_allocator);
    defer channel_zps.deinit();
    var channel_bias = std.ArrayList(f32).init(pkg_allocator);
    defer channel_bias.deinit();

    try channel_scales.ensureTotalCapacity(out_channels);
    try channel_zps.ensureTotalCapacity(out_channels);
    try channel_bias.ensureTotalCapacity(out_channels);

    for (0..out_channels) |m| {
        const w_scale_val: f32 = if (w_scale.data.len == out_channels)
            asF32(ScaleType, w_scale.data[m])
        else
            asF32(ScaleType, w_scale.data[0]);

        const w_zp_f: f32 = if (w_zero_point.data.len == out_channels)
            asF32(@TypeOf(w_zero_point.data[0]), w_zero_point.data[m])
        else if (w_zero_point.data.len > 0)
            asF32(@TypeOf(w_zero_point.data[0]), w_zero_point.data[0])
        else
            0.0;

        const bias_f: f32 = if (bias) |b| blk: {
            const b_raw = if (b.data.len == 1) b.data[0] else b.data[m];
            const b_val: f32 = if (isInt(BiasType))
                asF32(BiasType, b_raw) * x_scale_val * w_scale_val
            else
                asF32(BiasType, b_raw);
            break :blk b_val;
        } else 0.0;

        channel_scales.appendAssumeCapacity(w_scale_val);
        channel_zps.appendAssumeCapacity(w_zp_f);
        channel_bias.appendAssumeCapacity(bias_f);
    }

    // Pre-calcola i limiti di quantizzazione
    const q_min: f32 = asF32(InputType, std.math.minInt(InputType));
    const q_max: f32 = asF32(InputType, std.math.maxInt(InputType));

    // ===== OTTIMIZZAZIONE 2: Specialized paths per kernel comuni =====
    if (kernel_height == 3 and kernel_width == 3 and dilation_h == 1 and dilation_w == 1) {
        // Ottimizzato per 3x3 (MobileNet style)
        try conv3x3Optimized(x, w, output, batch_size, actual_group, in_channels, out_channels, weight_in_channels, in_height, in_width, out_height, out_width, stride_h, stride_w, pad_h_begin, pad_w_begin, x_scale_val, x_zp_f, channel_scales.items, channel_zps.items, channel_bias.items, y_scale_val, y_zp_f, q_min, q_max, InputType, WeightType);
    } else if (kernel_height == 1 and kernel_width == 1) {
        // Ottimizzato per 1x1 (pointwise)
        try conv1x1Optimized(x, w, output, batch_size, actual_group, in_channels, out_channels, weight_in_channels, in_height, in_width, out_height, out_width, x_scale_val, x_zp_f, channel_scales.items, channel_zps.items, channel_bias.items, y_scale_val, y_zp_f, q_min, q_max, InputType, WeightType);
    } else {
        // ===== OTTIMIZZAZIONE 3: Loop originale con pre-calcoli =====
        for (0..batch_size) |n| {
            for (0..actual_group) |g| {
                const in_c_start = g * (in_channels / actual_group);
                const in_c_end = (g + 1) * (in_channels / actual_group);
                const out_c_start = g * (out_channels / actual_group);
                const out_c_end = (g + 1) * (out_channels / actual_group);

                // Process output channels in blocks for better cache locality
                const block_size = 4;
                var m_block = out_c_start;
                while (m_block < out_c_end) {
                    const m_end = @min(m_block + block_size, out_c_end);

                    for (m_block..m_end) |m| {
                        const w_scale_val = channel_scales.items[m];
                        const w_zp_f = channel_zps.items[m];
                        const bias_f = channel_bias.items[m];

                        for (0..out_height) |oh| {
                            const in_h_start = @as(isize, @intCast(oh * stride_h)) - @as(isize, @intCast(pad_h_begin));

                            for (0..out_width) |ow| {
                                const in_w_start = @as(isize, @intCast(ow * stride_w)) - @as(isize, @intCast(pad_w_begin));
                                var acc: f32 = bias_f;

                                for (0..kernel_height) |kh| {
                                    const in_h = in_h_start + @as(isize, @intCast(kh * dilation_h));
                                    if (in_h < 0 or in_h >= @as(isize, @intCast(in_height))) continue;

                                    for (0..kernel_width) |kw| {
                                        const in_w = in_w_start + @as(isize, @intCast(kw * dilation_w));
                                        if (in_w < 0 or in_w >= @as(isize, @intCast(in_width))) continue;

                                        const ih = @as(usize, @intCast(in_h));
                                        const iw = @as(usize, @intCast(in_w));

                                        for (in_c_start..in_c_end) |c| {
                                            const k_c = c - in_c_start;
                                            const input_idx = ((n * in_channels + c) * in_height + ih) * in_width + iw;
                                            const weight_idx = ((m * weight_in_channels + k_c) * kernel_height + kh) * kernel_width + kw;

                                            const x_real: f32 = if (isInt(InputType)) blk: {
                                                const qx = asF32(InputType, x.data[input_idx]);
                                                break :blk x_scale_val * (qx - x_zp_f);
                                            } else asF32(InputType, x.data[input_idx]);

                                            const w_real: f32 = if (isInt(WeightType)) blk: {
                                                const qw = asF32(WeightType, w.data[weight_idx]);
                                                break :blk w_scale_val * (qw - w_zp_f);
                                            } else asF32(WeightType, w.data[weight_idx]);

                                            acc += x_real * w_real;
                                        }
                                    }
                                }

                                const output_idx = ((n * out_channels + m) * out_height + oh) * out_width + ow;
                                const q_unrounded: f32 = acc / y_scale_val + y_zp_f;
                                const q_clamped = std.math.clamp(@round(q_unrounded), q_min, q_max);
                                output.data[output_idx] = @as(InputType, @intFromFloat(q_clamped));
                            }
                        }
                    }
                    m_block += block_size;
                }
            }
        }
    }
}

// ===== SPECIALIZZAZIONI OTTIMIZZATE =====

/// Optimized 3x3 convolution with loop unrolling and cache blocking
fn conv3x3Optimized(x: anytype, w: anytype, output: anytype, batch_size: usize, actual_group: usize, in_channels: usize, out_channels: usize, weight_in_channels: usize, in_height: usize, in_width: usize, out_height: usize, out_width: usize, stride_h: usize, stride_w: usize, pad_h_begin: usize, pad_w_begin: usize, x_scale_val: f32, x_zp_f: f32, channel_scales: []const f32, channel_zps: []const f32, channel_bias: []const f32, y_scale_val: f32, y_zp_f: f32, q_min: f32, q_max: f32, comptime InputType: type, comptime WeightType: type) !void {
    const isInt = struct {
        fn call(comptime T: type) bool {
            return switch (@typeInfo(T)) {
                .int, .comptime_int => true,
                else => false,
            };
        }
    }.call;

    const asF32 = struct {
        fn call(comptime T: type, v: T) f32 {
            return switch (@typeInfo(T)) {
                .float => @as(f32, @floatCast(v)),
                .int, .comptime_int => @as(f32, @floatFromInt(v)),
                else => @compileError("Unsupported type for float cast"),
            };
        }
    }.call;

    for (0..batch_size) |n| {
        for (0..actual_group) |g| {
            const in_c_start = g * (in_channels / actual_group);
            const in_c_end = (g + 1) * (in_channels / actual_group);
            const out_c_start = g * (out_channels / actual_group);
            const out_c_end = (g + 1) * (out_channels / actual_group);

            for (out_c_start..out_c_end) |m| {
                const w_scale_val = channel_scales[m];
                const w_zp_f = channel_zps[m];
                const bias_f = channel_bias[m];

                for (0..out_height) |oh| {
                    const in_h_start = @as(isize, @intCast(oh * stride_h)) - @as(isize, @intCast(pad_h_begin));

                    for (0..out_width) |ow| {
                        const in_w_start = @as(isize, @intCast(ow * stride_w)) - @as(isize, @intCast(pad_w_begin));
                        var acc: f32 = bias_f;

                        // Unroll 3x3 kernel manually per migliori performance
                        for (in_c_start..in_c_end) |c| {
                            const k_c = c - in_c_start;

                            // kh=0, kw=0
                            const in_h_0 = in_h_start;
                            const in_w_0 = in_w_start;
                            if (in_h_0 >= 0 and in_h_0 < in_height and in_w_0 >= 0 and in_w_0 < in_width) {
                                const ih_0 = @as(usize, @intCast(in_h_0));
                                const iw_0 = @as(usize, @intCast(in_w_0));
                                const input_idx = ((n * in_channels + c) * in_height + ih_0) * in_width + iw_0;
                                const weight_idx = ((m * weight_in_channels + k_c) * 3 + 0) * 3 + 0;

                                const x_real: f32 = if (isInt(InputType)) blk: {
                                    const qx = asF32(InputType, x.data[input_idx]);
                                    break :blk x_scale_val * (qx - x_zp_f);
                                } else asF32(InputType, x.data[input_idx]);

                                const w_real: f32 = if (isInt(WeightType)) blk: {
                                    const qw = asF32(WeightType, w.data[weight_idx]);
                                    break :blk w_scale_val * (qw - w_zp_f);
                                } else asF32(WeightType, w.data[weight_idx]);

                                acc += x_real * w_real;
                            }

                            // Continue unrolling for all 9 positions (kh=0,1,2 x kw=0,1,2)
                            // Unroll rimanenti per brevità...
                            inline for (0..3) |kh| {
                                inline for (0..3) |kw| {
                                    if (kh == 0 and kw == 0) continue; // già fatto sopra

                                    const in_h = in_h_start + @as(isize, @intCast(kh));
                                    const in_w = in_w_start + @as(isize, @intCast(kw));

                                    if (in_h >= 0 and in_h < in_height and in_w >= 0 and in_w < in_width) {
                                        const ih = @as(usize, @intCast(in_h));
                                        const iw = @as(usize, @intCast(in_w));
                                        const input_idx = ((n * in_channels + c) * in_height + ih) * in_width + iw;
                                        const weight_idx = ((m * weight_in_channels + k_c) * 3 + kh) * 3 + kw;

                                        const x_real: f32 = if (isInt(InputType)) blk: {
                                            const qx = asF32(InputType, x.data[input_idx]);
                                            break :blk x_scale_val * (qx - x_zp_f);
                                        } else asF32(InputType, x.data[input_idx]);

                                        const w_real: f32 = if (isInt(WeightType)) blk: {
                                            const qw = asF32(WeightType, w.data[weight_idx]);
                                            break :blk w_scale_val * (qw - w_zp_f);
                                        } else asF32(WeightType, w.data[weight_idx]);

                                        acc += x_real * w_real;
                                    }
                                }
                            }
                        }

                        const output_idx = ((n * out_channels + m) * out_height + oh) * out_width + ow;
                        const q_unrounded: f32 = acc / y_scale_val + y_zp_f;
                        const q_clamped = std.math.clamp(@round(q_unrounded), q_min, q_max);
                        output.data[output_idx] = @as(InputType, @intFromFloat(q_clamped));
                    }
                }
            }
        }
    }
}

/// Optimized 1x1 convolution (pointwise - essentially matrix multiplication)
fn conv1x1Optimized(x: anytype, w: anytype, output: anytype, batch_size: usize, actual_group: usize, in_channels: usize, out_channels: usize, weight_in_channels: usize, in_height: usize, in_width: usize, out_height: usize, out_width: usize, x_scale_val: f32, x_zp_f: f32, channel_scales: []const f32, channel_zps: []const f32, channel_bias: []const f32, y_scale_val: f32, y_zp_f: f32, q_min: f32, q_max: f32, comptime InputType: type, comptime WeightType: type) !void {
    const isInt = struct {
        fn call(comptime T: type) bool {
            return switch (@typeInfo(T)) {
                .int, .comptime_int => true,
                else => false,
            };
        }
    }.call;

    const asF32 = struct {
        fn call(comptime T: type, v: T) f32 {
            return switch (@typeInfo(T)) {
                .float => @as(f32, @floatCast(v)),
                .int, .comptime_int => @as(f32, @floatFromInt(v)),
                else => @compileError("Unsupported type for float cast"),
            };
        }
    }.call;

    for (0..batch_size) |n| {
        for (0..actual_group) |g| {
            const in_c_start = g * (in_channels / actual_group);
            const in_c_end = (g + 1) * (in_channels / actual_group);
            const out_c_start = g * (out_channels / actual_group);
            const out_c_end = (g + 1) * (out_channels / actual_group);

            for (out_c_start..out_c_end) |m| {
                const w_scale_val = channel_scales[m];
                const w_zp_f = channel_zps[m];
                const bias_f = channel_bias[m];

                // 1x1 conv è matrix multiplication - ottimizzato di conseguenza
                for (0..out_height) |oh| {
                    for (0..out_width) |ow| {
                        var acc: f32 = bias_f;

                        // Nessun kernel spaziale, solo channel mixing
                        for (in_c_start..in_c_end) |c| {
                            const k_c = c - in_c_start;
                            const input_idx = ((n * in_channels + c) * in_height + oh) * in_width + ow;
                            const weight_idx = m * weight_in_channels + k_c;

                            const x_real: f32 = if (isInt(InputType)) blk: {
                                const qx = asF32(InputType, x.data[input_idx]);
                                break :blk x_scale_val * (qx - x_zp_f);
                            } else asF32(InputType, x.data[input_idx]);

                            const w_real: f32 = if (isInt(WeightType)) blk: {
                                const qw = asF32(WeightType, w.data[weight_idx]);
                                break :blk w_scale_val * (qw - w_zp_f);
                            } else asF32(WeightType, w.data[weight_idx]);

                            acc += x_real * w_real;
                        }

                        const output_idx = ((n * out_channels + m) * out_height + oh) * out_width + ow;
                        const q_unrounded: f32 = acc / y_scale_val + y_zp_f;
                        const q_clamped = std.math.clamp(@round(q_unrounded), q_min, q_max);
                        output.data[output_idx] = @as(InputType, @intFromFloat(q_clamped));
                    }
                }
            }
        }
    }
}

const ChannelParams = struct {
    weight_scale_q16: i32,
    weight_zero_point: i32,
    bias_q16: i64,
};

const QuantParams = struct {
    scale_shift: u5,
    input_zero_point: i32,
    input_scale_i64: i64,
    output_inv_scale_i64: i64,
    output_zero_point_q16: i64,
    rounding: i64,
    q_min: i32,
    q_max: i32,
};

const ConvDims = struct {
    batch: usize,
    in_channels: usize,
    in_height: usize,
    in_width: usize,
    out_channels: usize,
    out_height: usize,
    out_width: usize,
    kernel_height: usize,
    kernel_width: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    groups: usize,
    group_in_channels: usize,
    group_out_channels: usize,
};

const ConvLayout = struct {
    input_batch_stride: usize,
    input_channel_stride: usize,
    input_row_stride: usize,
    weight_out_stride: usize,
    weight_channel_stride: usize,
    weight_row_stride: usize,
    output_batch_stride: usize,
    output_channel_stride: usize,
    output_row_stride: usize,
};

inline fn quantizeAccumulator(acc: i64, quant: QuantParams) i32 {
    const acc_q16 = (acc * quant.output_inv_scale_i64) >> quant.scale_shift;
    const acc_with_zp = acc_q16 + quant.output_zero_point_q16;
    var q = @as(i32, @intCast((acc_with_zp + quant.rounding) >> quant.scale_shift));
    if (q < quant.q_min) q = quant.q_min;
    if (q > quant.q_max) q = quant.q_max;
    return q;
}

/// Embedded-optimized version using fixed-point arithmetic (Q15.16)
/// Reduces floating-point operations for better performance on embedded targets
pub fn qlinearconv_embedded_lean(
    comptime InputType: anytype,
    comptime WeightType: anytype,
    comptime ScaleType: anytype,
    comptime _: anytype, // ZeroPointType unused due to anytype zero_point parameters
    comptime BiasType: anytype,
    x: *const Tensor(InputType),
    x_scale: *const Tensor(ScaleType),
    x_zero_point: anytype, // Accept any tensor type for zero_point
    w: *const Tensor(WeightType),
    w_scale: *const Tensor(ScaleType),
    w_zero_point: anytype, // Accept any tensor type for zero_point
    output: *Tensor(InputType),
    y_scale: *const Tensor(ScaleType),
    y_zero_point: anytype, // Accept any tensor type for zero_point
    bias: ?*const Tensor(BiasType),
    stride: ?[]const usize,
    pads: ?[]const usize,
    dilations: ?[]const usize,
    group: ?usize,
    auto_pad: []const u8,
) !void {
    if (auto_pad.len != 0 and !std.mem.eql(u8, auto_pad, "NOTSET")) {
        return TensorMathError.InvalidPadding;
    }

    const isInt = struct {
        fn call(comptime T: type) bool {
            return switch (@typeInfo(T)) {
                .int, .comptime_int => true,
                else => false,
            };
        }
    }.call;

    const asF32 = struct {
        fn call(comptime T: type, v: T) f32 {
            return switch (@typeInfo(T)) {
                .float => @as(f32, @floatCast(v)),
                .int, .comptime_int => @as(f32, @floatFromInt(v)),
                else => @compileError("Unsupported type for float cast"),
            };
        }
    }.call;

    if (!isInt(InputType) or !isInt(WeightType)) {
        return qlinearconv_lean(InputType, WeightType, ScaleType, void, BiasType, x, x_scale, x_zero_point, w, w_scale, w_zero_point, output, y_scale, y_zero_point, bias, stride, pads, dilations, group, auto_pad);
    }

    // Pure reference implementation - no CMSIS dispatch overhead

    if (x.shape.len != 4 or w.shape.len != 4 or output.shape.len != 4) {
        return TensorMathError.InvalidDimensions;
    }

    const batch_size = x.shape[0];
    const in_channels = x.shape[1];
    const in_height = x.shape[2];
    const in_width = x.shape[3];

    const out_channels = w.shape[0];
    const weight_in_channels = w.shape[1];
    const kernel_height = w.shape[2];
    const kernel_width = w.shape[3];

    const out_height = output.shape[2];
    const out_width = output.shape[3];

    const stride_h = if (stride) |s| s[0] else 1;
    const stride_w = if (stride) |s| (if (s.len > 1) s[1] else s[0]) else 1;
    const pads_arr = pads orelse &[_]usize{ 0, 0, 0, 0 };
    const pad_h_begin = pads_arr[0];
    const pad_w_begin = pads_arr[1];
    const dilation_h = if (dilations) |d| d[0] else 1;
    const dilation_w = if (dilations) |d| (if (d.len > 1) d[1] else d[0]) else 1;
    const actual_group = group orelse 1;

    if (in_channels % actual_group != 0 or out_channels % actual_group != 0) {
        return TensorMathError.InvalidDimensions;
    }
    if (weight_in_channels * actual_group != in_channels) {
        return TensorMathError.InvalidDimensions;
    }

    const SCALE_SHIFT: u5 = 16;
    const scale_factor = @as(f32, @floatFromInt(@as(u32, 1) << SCALE_SHIFT));
    const x_scale_val = asF32(ScaleType, x_scale.data[0]);
    const y_scale_val = asF32(ScaleType, y_scale.data[0]);
    const input_zero_point = if (@typeInfo(@TypeOf(x_zero_point)) == .pointer and x_zero_point.data.len == 0)
        0
    else
        readScalarZP(InputType, x_zero_point);
    const output_zero_point = if (@typeInfo(@TypeOf(y_zero_point)) == .pointer and y_zero_point.data.len == 0)
        0
    else
        readScalarZP(InputType, y_zero_point);

    const quant = QuantParams{
        .scale_shift = SCALE_SHIFT,
        .input_zero_point = input_zero_point,
        .input_scale_i64 = @as(i64, q16(x_scale_val)),
        .output_inv_scale_i64 = @as(i64, q16(1.0 / y_scale_val)),
        .output_zero_point_q16 = @as(i64, output_zero_point) << SCALE_SHIFT,
        .rounding = @as(i64, 1) << (SCALE_SHIFT - 1),
        .q_min = @as(i32, @intCast(std.math.minInt(InputType))),
        .q_max = @as(i32, @intCast(std.math.maxInt(InputType))),
    };

    const dims = ConvDims{
        .batch = batch_size,
        .in_channels = in_channels,
        .in_height = in_height,
        .in_width = in_width,
        .out_channels = out_channels,
        .out_height = out_height,
        .out_width = out_width,
        .kernel_height = kernel_height,
        .kernel_width = kernel_width,
        .stride_h = stride_h,
        .stride_w = stride_w,
        .pad_h = pad_h_begin,
        .pad_w = pad_w_begin,
        .dilation_h = dilation_h,
        .dilation_w = dilation_w,
        .groups = actual_group,
        .group_in_channels = in_channels / actual_group,
        .group_out_channels = out_channels / actual_group,
    };

    const layout = ConvLayout{
        .input_batch_stride = in_channels * in_height * in_width,
        .input_channel_stride = in_height * in_width,
        .input_row_stride = in_width,
        .weight_out_stride = weight_in_channels * kernel_height * kernel_width,
        .weight_channel_stride = kernel_height * kernel_width,
        .weight_row_stride = kernel_width,
        .output_batch_stride = out_channels * out_height * out_width,
        .output_channel_stride = out_height * out_width,
        .output_row_stride = out_width,
    };

    var channel_params = try pkg_allocator.alloc(ChannelParams, out_channels);
    defer pkg_allocator.free(channel_params);

    const bias_tensor = if (bias) |b| b else null;
    const bias_is_int = isInt(BiasType);

    for (0..out_channels) |m| {
        const w_scale_val: f32 = if (w_scale.data.len == out_channels)
            asF32(ScaleType, w_scale.data[m])
        else
            asF32(ScaleType, w_scale.data[0]);

        const weight_scale_q16 = q16(w_scale_val);
        const weight_zero_point = if (@typeInfo(@TypeOf(w_zero_point)) == .pointer and w_zero_point.data.len == 0)
            0
        else
            readPerChannelZP(w_zero_point, m, out_channels);

        const bias_q16 = if (bias_tensor) |b_tensor| blk: {
            if (b_tensor.data.len == 0) break :blk 0;
            const raw = if (b_tensor.data.len == 1) b_tensor.data[0] else b_tensor.data[m];
            var bias_real = asF32(BiasType, raw);
            if (bias_is_int) {
                bias_real *= x_scale_val * w_scale_val;
            }
            break :blk @as(i64, @intFromFloat(@round(bias_real * scale_factor)));
        } else 0;

        channel_params[m] = .{
            .weight_scale_q16 = weight_scale_q16,
            .weight_zero_point = weight_zero_point,
            .bias_q16 = bias_q16,
        };
    }

    if (kernel_height == 3 and kernel_width == 3 and dilation_h == 1 and dilation_w == 1) {
        conv3x3EmbeddedOptimized(InputType, WeightType, x.data, w.data, output.data, dims, layout, quant, channel_params);
    } else if (kernel_height == 1 and kernel_width == 1) {
        conv1x1EmbeddedOptimized(InputType, WeightType, x.data, w.data, output.data, dims, layout, quant, channel_params);
    } else {
        convGenericEmbeddedOptimized(InputType, WeightType, x.data, w.data, output.data, dims, layout, quant, channel_params);
    }
}

inline fn conv3x3EmbeddedOptimized(
    comptime InputType: type,
    comptime WeightType: type,
    x_data: []const InputType,
    w_data: []const WeightType,
    out_data: []InputType,
    dims: ConvDims,
    layout: ConvLayout,
    quant: QuantParams,
    channel_params: []const ChannelParams,
) void {
    const shift = quant.scale_shift;
    const input_zp = quant.input_zero_point;
    const x_scale_i64 = quant.input_scale_i64;
    const in_height_isize = @as(isize, @intCast(dims.in_height));
    const in_width_isize = @as(isize, @intCast(dims.in_width));

    for (0..dims.batch) |n| {
        const input_batch_base = n * layout.input_batch_stride;
        const output_batch_base = n * layout.output_batch_stride;

        for (0..dims.groups) |g| {
            const in_group_base = g * dims.group_in_channels;
            const out_group_base = g * dims.group_out_channels;

            for (0..dims.group_out_channels) |oc| {
                const m = out_group_base + oc;
                const channel = channel_params[m];
                const weight_scale_i64 = @as(i64, channel.weight_scale_q16);
                const weight_base = m * layout.weight_out_stride;
                const output_channel_base = output_batch_base + m * layout.output_channel_stride;

                for (0..dims.out_height) |oh| {
                    const ih_origin = @as(isize, @intCast(oh * dims.stride_h)) - @as(isize, @intCast(dims.pad_h));
                    const output_row_base = output_channel_base + oh * layout.output_row_stride;

                    for (0..dims.out_width) |ow| {
                        const iw_origin = @as(isize, @intCast(ow * dims.stride_w)) - @as(isize, @intCast(dims.pad_w));
                        var acc: i64 = channel.bias_q16;

                        for (0..dims.group_in_channels) |ic| {
                            const c = in_group_base + ic;
                            const input_channel_base = input_batch_base + c * layout.input_channel_stride;
                            const weight_channel_base = weight_base + ic * layout.weight_channel_stride;

                            var kh: usize = 0;
                            while (kh < 3) : (kh += 1) {
                                const ih = ih_origin + @as(isize, @intCast(kh));
                                if (ih < 0 or ih >= in_height_isize) continue;

                                const input_row_base = input_channel_base + @as(usize, @intCast(ih)) * layout.input_row_stride;
                                const weight_row_base = weight_channel_base + kh * layout.weight_row_stride;

                                var kw: usize = 0;
                                var weight_index = weight_row_base;
                                while (kw < 3) : (kw += 1) {
                                    const iw = iw_origin + @as(isize, @intCast(kw));
                                    if (iw >= 0 and iw < in_width_isize) {
                                        const input_index = input_row_base + @as(usize, @intCast(iw));
                                        const x_q = @as(i32, @intCast(x_data[input_index]));
                                        const w_q = @as(i32, @intCast(w_data[weight_index]));
                                        const x_diff = x_q - input_zp;
                                        const w_diff = w_q - channel.weight_zero_point;
                                        const x_q16 = @as(i64, x_diff) * x_scale_i64;
                                        const w_q16 = @as(i64, w_diff) * weight_scale_i64;
                                        acc += (x_q16 * w_q16) >> shift;
                                    }
                                    weight_index += 1;
                                }
                            }
                        }

                        const q = quantizeAccumulator(acc, quant);
                        out_data[output_row_base + ow] = @as(InputType, @intCast(q));
                    }
                }
            }
        }
    }
}

inline fn conv1x1EmbeddedOptimized(
    comptime InputType: type,
    comptime WeightType: type,
    x_data: []const InputType,
    w_data: []const WeightType,
    out_data: []InputType,
    dims: ConvDims,
    layout: ConvLayout,
    quant: QuantParams,
    channel_params: []const ChannelParams,
) void {
    const shift = quant.scale_shift;
    const input_zp = quant.input_zero_point;
    const x_scale_i64 = quant.input_scale_i64;
    const in_height_isize = @as(isize, @intCast(dims.in_height));
    const in_width_isize = @as(isize, @intCast(dims.in_width));

    for (0..dims.batch) |n| {
        const input_batch_base = n * layout.input_batch_stride;
        const output_batch_base = n * layout.output_batch_stride;

        for (0..dims.groups) |g| {
            const in_group_base = g * dims.group_in_channels;
            const out_group_base = g * dims.group_out_channels;

            for (0..dims.group_out_channels) |oc| {
                const m = out_group_base + oc;
                const channel = channel_params[m];
                const weight_scale_i64 = @as(i64, channel.weight_scale_q16);
                const weight_base = m * layout.weight_out_stride;
                const output_channel_base = output_batch_base + m * layout.output_channel_stride;

                for (0..dims.out_height) |oh| {
                    const ih_origin = @as(isize, @intCast(oh * dims.stride_h)) - @as(isize, @intCast(dims.pad_h));
                    const output_row_base = output_channel_base + oh * layout.output_row_stride;

                    for (0..dims.out_width) |ow| {
                        const iw_origin = @as(isize, @intCast(ow * dims.stride_w)) - @as(isize, @intCast(dims.pad_w));
                        var acc: i64 = channel.bias_q16;

                        if (ih_origin >= 0 and ih_origin < in_height_isize and iw_origin >= 0 and iw_origin < in_width_isize) {
                            const ih = @as(usize, @intCast(ih_origin));
                            const iw = @as(usize, @intCast(iw_origin));

                            for (0..dims.group_in_channels) |ic| {
                                const c = in_group_base + ic;
                                const input_channel_base = input_batch_base + c * layout.input_channel_stride;
                                const input_index = input_channel_base + ih * layout.input_row_stride + iw;
                                const weight_index = weight_base + ic;

                                const x_q = @as(i32, @intCast(x_data[input_index]));
                                const w_q = @as(i32, @intCast(w_data[weight_index]));
                                const x_diff = x_q - input_zp;
                                const w_diff = w_q - channel.weight_zero_point;
                                const x_q16 = @as(i64, x_diff) * x_scale_i64;
                                const w_q16 = @as(i64, w_diff) * weight_scale_i64;
                                acc += (x_q16 * w_q16) >> shift;
                            }
                        }

                        const q = quantizeAccumulator(acc, quant);
                        out_data[output_row_base + ow] = @as(InputType, @intCast(q));
                    }
                }
            }
        }
    }
}

inline fn convGenericEmbeddedOptimized(
    comptime InputType: type,
    comptime WeightType: type,
    x_data: []const InputType,
    w_data: []const WeightType,
    out_data: []InputType,
    dims: ConvDims,
    layout: ConvLayout,
    quant: QuantParams,
    channel_params: []const ChannelParams,
) void {
    const shift = quant.scale_shift;
    const input_zp = quant.input_zero_point;
    const x_scale_i64 = quant.input_scale_i64;
    const in_height_isize = @as(isize, @intCast(dims.in_height));
    const in_width_isize = @as(isize, @intCast(dims.in_width));

    for (0..dims.batch) |n| {
        const input_batch_base = n * layout.input_batch_stride;
        const output_batch_base = n * layout.output_batch_stride;

        for (0..dims.groups) |g| {
            const in_group_base = g * dims.group_in_channels;
            const out_group_base = g * dims.group_out_channels;

            for (0..dims.group_out_channels) |oc| {
                const m = out_group_base + oc;
                const channel = channel_params[m];
                const weight_scale_i64 = @as(i64, channel.weight_scale_q16);
                const weight_base = m * layout.weight_out_stride;
                const output_channel_base = output_batch_base + m * layout.output_channel_stride;

                for (0..dims.out_height) |oh| {
                    const ih_origin = @as(isize, @intCast(oh * dims.stride_h)) - @as(isize, @intCast(dims.pad_h));
                    const output_row_base = output_channel_base + oh * layout.output_row_stride;

                    for (0..dims.out_width) |ow| {
                        const iw_origin = @as(isize, @intCast(ow * dims.stride_w)) - @as(isize, @intCast(dims.pad_w));
                        var acc: i64 = channel.bias_q16;

                        for (0..dims.group_in_channels) |ic| {
                            const c = in_group_base + ic;
                            const input_channel_base = input_batch_base + c * layout.input_channel_stride;
                            const weight_channel_base = weight_base + ic * layout.weight_channel_stride;

                            var kh: usize = 0;
                            while (kh < dims.kernel_height) : (kh += 1) {
                                const ih = ih_origin + @as(isize, @intCast(kh * dims.dilation_h));
                                if (ih < 0 or ih >= in_height_isize) continue;

                                const input_row_base = input_channel_base + @as(usize, @intCast(ih)) * layout.input_row_stride;
                                const weight_row_base = weight_channel_base + kh * layout.weight_row_stride;

                                var kw: usize = 0;
                                while (kw < dims.kernel_width) : (kw += 1) {
                                    const iw = iw_origin + @as(isize, @intCast(kw * dims.dilation_w));
                                    if (iw < 0 or iw >= in_width_isize) {
                                        continue;
                                    }

                                    const input_index = input_row_base + @as(usize, @intCast(iw));
                                    const weight_index = weight_row_base + kw;

                                    const x_q = @as(i32, @intCast(x_data[input_index]));
                                    const w_q = @as(i32, @intCast(w_data[weight_index]));
                                    const x_diff = x_q - input_zp;
                                    const w_diff = w_q - channel.weight_zero_point;
                                    const x_q16 = @as(i64, x_diff) * x_scale_i64;
                                    const w_q16 = @as(i64, w_diff) * weight_scale_i64;
                                    acc += (x_q16 * w_q16) >> shift;
                                }
                            }
                        }

                        const q = quantizeAccumulator(acc, quant);
                        out_data[output_row_base + ow] = @as(InputType, @intCast(q));
                    }
                }
            }
        }
    }
}

// Helper function to quantize multiplier for CMSIS-NN
fn quantizeMultiplier(scale: f32, multiplier: *i32, shift: *i32) void {
    if (scale == 0.0) {
        multiplier.* = 0;
        shift.* = 0;
        return;
    }

    var sig = scale;
    var exp: i32 = 0;

    // Normalize to [0.5, 1.0) range
    while (sig >= 1.0) {
        sig /= 2.0;
        exp += 1;
    }
    while (sig < 0.5) {
        sig *= 2.0;
        exp -= 1;
    }

    // Convert to fixed point representation
    const fixed_point_multiplier = @as(i32, @intFromFloat(@round(sig * (1 << 31))));

    multiplier.* = fixed_point_multiplier;
    shift.* = exp;
}

/// Direct CMSIS-NN wrapper - passes quantized data directly with minimal overhead
/// CMSIS-NN accelerated quantized convolution - direct implementation without fallback overhead
/// Compile-time dispatch function that chooses the best implementation
pub fn qlinearconv_dispatch(
    comptime InputType: anytype,
    comptime WeightType: anytype,
    comptime ScaleType: anytype,
    comptime OutputType: anytype,
    comptime BiasType: anytype,
    x: *const Tensor(InputType),
    x_scale: *const Tensor(ScaleType),
    x_zero_point: anytype,
    w: *const Tensor(WeightType),
    w_scale: *const Tensor(ScaleType),
    w_zero_point: anytype,
    output: *Tensor(InputType),
    y_scale: *const Tensor(ScaleType),
    y_zero_point: anytype,
    bias: ?*const Tensor(BiasType),
    stride: ?[]const usize,
    pads: ?[]const usize,
    dilations: ?[]const usize,
    group: ?usize,
    auto_pad: []const u8,
) !void {
    const accelerators = @import("../Accelerators/mod.zig");
    if (!accelerators.canUseCmsisHelium()) {
        // Reference build: force embedded fixed-point implementation
        return qlinearconv_embedded_lean(
            InputType,
            WeightType,
            ScaleType,
            void,
            BiasType,
            x,
            x_scale,
            x_zero_point,
            w,
            w_scale,
            w_zero_point,
            output,
            y_scale,
            y_zero_point,
            bias,
            stride,
            pads,
            dilations,
            group,
            auto_pad,
        );
    }

    // CMSIS path
    return qlinearconv_cmsis_accelerated(
        InputType,
        WeightType,
        ScaleType,
        OutputType,
        BiasType,
        x,
        x_scale,
        x_zero_point,
        w,
        w_scale,
        w_zero_point,
        output,
        y_scale,
        y_zero_point,
        bias,
        stride,
        pads,
        dilations,
        group,
        auto_pad,
    );
}

/// CMSIS-NN accelerated quantized convolution - direct implementation without fallback overhead
pub fn qlinearconv_cmsis_accelerated(
    comptime InputType: anytype,
    comptime WeightType: anytype,
    comptime ScaleType: anytype,
    comptime _: anytype,
    comptime BiasType: anytype,
    x: *const Tensor(InputType),
    _x_scale: *const Tensor(ScaleType),
    x_zero_point: anytype,
    w: *const Tensor(WeightType),
    _w_scale: *const Tensor(ScaleType),
    w_zero_point_any: anytype,
    output: *Tensor(InputType),
    _y_scale: *const Tensor(ScaleType),
    y_zero_point: anytype,
    bias: ?*const Tensor(BiasType),
    stride: ?[]const usize,
    pads: ?[]const usize,
    dilations: ?[]const usize,
    group: ?usize,
    _: []const u8,
) !void {
    // Mark CMSIS usage for testing
    const accelerators = @import("../Accelerators/mod.zig");
    accelerators.markCmsisUsed();

    // DEBUG: Print function entry
    // std.debug.print("CMSIS DEBUG: qlinearconv_cmsis_accelerated called\n", .{});

    // Suppress unused parameter warnings
    // w_zero_point_any is actually used later when packing weights (readPerChannelZP)

    // Basic validation
    if (x.shape.len != 4 or w.shape.len != 4 or output.shape.len != 4) {
        return TensorMathError.InvalidDimensions;
    }

    const group_val: usize = group orelse 1;
    const dilation_h = if (dilations) |d| d[0] else 1;
    const dilation_w = if (dilations) |d| (if (d.len > 1) d[1] else d[0]) else 1;

    const cmsis_nn = @import("../Accelerators/stm32n6/cmsis_nn.zig");

    // Extract dimensions
    const batch_size = x.shape[0];
    const in_channels = x.shape[1];
    const in_height = x.shape[2];
    const in_width = x.shape[3];
    const out_channels = w.shape[0];
    const weight_in_channels = w.shape[1];
    const kernel_height = w.shape[2];
    const kernel_width = w.shape[3];
    const out_height = output.shape[2];
    const out_width = output.shape[3];

    const stride_h = if (stride) |s| s[0] else 1;
    const stride_w = if (stride) |s| (if (s.len > 1) s[1] else s[0]) else 1;
    const pads_arr = pads orelse &[_]usize{ 0, 0, 0, 0 };
    const pad_h = pads_arr[0];
    const pad_w = pads_arr[1];

    if (group_val == 0) {
        return TensorMathError.InvalidDimensions;
    }
    if (weight_in_channels * group_val != in_channels or out_channels % group_val != 0) {
        return TensorMathError.InvalidDimensions;
    }

    const group_in_channels = weight_in_channels;
    const group_out_channels = out_channels / group_val;

    // DEBUG: Print tensor dimensions
    // std.debug.print("CMSIS DEBUG: Input dims: {}x{}x{}x{}\n", .{ batch_size, in_channels, in_height, in_width });
    // std.debug.print("CMSIS DEBUG: Weight dims: {}x{}x{}x{}\n", .{ out_channels, in_channels, kernel_height, kernel_width });
    // std.debug.print("CMSIS DEBUG: Output dims: {}x{}x{}x{}\n", .{ batch_size, out_channels, out_height, out_width });
    // std.debug.print("CMSIS DEBUG: Stride: {}x{}, Pad: {}x{}\n", .{ stride_h, stride_w, pad_h, pad_w });

    // Extract zero points
    const input_zero_point = readScalarZP(InputType, x_zero_point);
    const output_zero_point = readScalarZP(InputType, y_zero_point);

    // DEBUG: Print zero points
    // std.debug.print("CMSIS DEBUG: input_zero_point: {}, output_zero_point: {}\n", .{ input_zero_point, output_zero_point });

    // Extract quantization parameters
    const x_scale_val = toF32(ScaleType, _x_scale.data[0]);
    const y_scale_val = toF32(ScaleType, _y_scale.data[0]);

    const cache_key = CacheKey{
        .weight_ptr = if (w.data.len == 0) 0 else @intFromPtr(w.data.ptr),
        .x_scale_ptr = tensorDataPtrOrZero(ScaleType, _x_scale),
        .y_scale_ptr = tensorDataPtrOrZero(ScaleType, _y_scale),
        .w_scale_ptr = tensorDataPtrOrZero(ScaleType, _w_scale),
        .bias_ptr = if (bias) |b| tensorDataPtrOrZero(BiasType, b) else 0,
        .w_zero_point_tag = zeroPointTag(w_zero_point_any),
        .group = group_val,
        .out_channels = out_channels,
        .weight_in_channels = weight_in_channels,
        .kernel_height = kernel_height,
        .kernel_width = kernel_width,
    };

    var cache_entry = try getOrCreateCacheEntry(
        InputType,
        WeightType,
        ScaleType,
        BiasType,
        cache_key,
        w,
        _w_scale,
        w_zero_point_any,
        bias,
        x_scale_val,
        y_scale_val,
        InputType == u8,
    );

    // Setup CMSIS-NN dimensions
    var input_dims = cmsis_nn.Dims{ .n = @intCast(batch_size), .h = @intCast(in_height), .w = @intCast(in_width), .c = @intCast(in_channels) };
    var output_dims = cmsis_nn.Dims{ .n = @intCast(batch_size), .h = @intCast(out_height), .w = @intCast(out_width), .c = @intCast(out_channels) };
    var bias_dims = cmsis_nn.Dims{ .n = 1, .h = 1, .w = 1, .c = @intCast(out_channels) };
    var input_group_dims = cmsis_nn.Dims{ .n = @intCast(batch_size), .h = @intCast(in_height), .w = @intCast(in_width), .c = @intCast(group_in_channels) };
    var output_group_dims = cmsis_nn.Dims{ .n = @intCast(batch_size), .h = @intCast(out_height), .w = @intCast(out_width), .c = @intCast(group_out_channels) };
    var bias_group_dims = cmsis_nn.Dims{ .n = 1, .h = 1, .w = 1, .c = @intCast(group_out_channels) };

    const is_u8_input = InputType == u8;
    const input_zero_point_s8: i32 = if (is_u8_input)
        @as(i32, @intCast(input_zero_point)) - 128
    else
        @as(i32, @intCast(input_zero_point));
    const output_zero_point_s8: i32 = if (is_u8_input)
        @as(i32, @intCast(output_zero_point)) - 128
    else
        @as(i32, @intCast(output_zero_point));

    var conv_params = cmsis_nn.ConvParams{
        .input_offset = -input_zero_point_s8,
        .output_offset = output_zero_point_s8,
        .stride = .{ .h = @intCast(stride_h), .w = @intCast(stride_w) },
        .padding = .{ .h = @intCast(pad_h), .w = @intCast(pad_w) },
        .dilation = .{ .h = @intCast(dilation_h), .w = @intCast(dilation_w) },
        .activation = .{ .min = -128, .max = 127 },
    };

    var quant_params = cmsis_nn.PerChannelQuantParams{
        .multiplier = cache_entry.multipliers.ptr,
        .shift = cache_entry.shifts.ptr,
    };

    const bias_ptr: ?[*]const i32 = if (cache_entry.bias) |b| b.ptr else null;

    var filter_dims = cmsis_nn.Dims{ .n = @intCast(out_channels), .h = @intCast(kernel_height), .w = @intCast(kernel_width), .c = @intCast(weight_in_channels) };
    var filter_group_dims = cmsis_nn.Dims{ .n = @intCast(group_out_channels), .h = @intCast(kernel_height), .w = @intCast(kernel_width), .c = @intCast(group_in_channels) };

    var buffer_size: i32 = 0;
    if (group_val == in_channels) {
        buffer_size = cmsis_nn.conv.arm_depthwise_conv_wrapper_s8_get_buffer_size(&.{
            .input_offset = -input_zero_point_s8,
            .output_offset = output_zero_point_s8,
            .ch_mult = @intCast(group_out_channels),
            .stride = .{ .h = @intCast(stride_h), .w = @intCast(stride_w) },
            .padding = .{ .h = @intCast(pad_h), .w = @intCast(pad_w) },
            .dilation = .{ .h = @intCast(dilation_h), .w = @intCast(dilation_w) },
            .activation = .{ .min = -128, .max = 127 },
        }, &input_dims, &.{ .n = 1, .h = @intCast(kernel_height), .w = @intCast(kernel_width), .c = @intCast(out_channels) }, &output_dims);
    } else if (group_val == 1) {
        buffer_size = cmsis_nn.conv.arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    } else {
        buffer_size = cmsis_nn.conv.arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_group_dims, &filter_group_dims, &output_group_dims);
    }
    if (buffer_size < 0) buffer_size = 0;
    try cache_entry.ensureScratch(@intCast(buffer_size));
    var ctx = cmsis_nn.Context{
        .buf = if (cache_entry.scratch.len > 0) @ptrCast(cache_entry.scratch.ptr) else null,
        .size = @intCast(cache_entry.scratch.len),
    };

    const input_len = x.data.len;
    var using_cached_input = false;
    var source_input: []const i8 = undefined;

    if (x.getAcceleratorView()) |view| {
        if (view.layout == AcceleratorLayout.nhwc_s8 and
            view.buffer_kind == AcceleratorBufferKind.output and
            view.dims[0] == batch_size and
            view.dims[1] == in_height and
            view.dims[2] == in_width and
            view.dims[3] == in_channels and
            view.scale == x_scale_val and
            view.zero_point == input_zero_point and
            view.buffer.len == input_len)
        {
            if (findCacheEntryById(view.owner_id)) |producer| {
                if (producer.output_generation == view.generation and producer.output_buffer.len == view.buffer.len and producer.output_buffer.ptr == view.buffer.ptr) {
                    using_cached_input = true;
                    source_input = view.buffer;
                }
            }
        }
    }

    if (!using_cached_input) {
        var input_slice = try cache_entry.ensureInputBuffer(input_len);
        var n: usize = 0;
        while (n < batch_size) : (n += 1) {
            var h: usize = 0;
            while (h < in_height) : (h += 1) {
                var w_: usize = 0;
                while (w_ < in_width) : (w_ += 1) {
                    var c: usize = 0;
                    while (c < in_channels) : (c += 1) {
                        const src_idx = ((n * in_channels + c) * in_height + h) * in_width + w_;
                        const dst_idx = ((n * in_height + h) * in_width + w_) * in_channels + c;
                        if (is_u8_input) {
                            const q = @as(i32, @intCast(x.data[src_idx]));
                            input_slice[dst_idx] = @as(i8, @intCast(q - 128));
                        } else {
                            input_slice[dst_idx] = @as(i8, @intCast(x.data[src_idx]));
                        }
                    }
                }
            }
        }
        source_input = input_slice;
    }

    const input_ptr_s8: [*]const i8 = source_input.ptr;

    const output_len = output.data.len;
    var output_slice = try cache_entry.ensureOutputBuffer(output_len);
    const output_ptr_s8: [*]i8 = output_slice.ptr;

    if (group_val != 1 and group_val != in_channels) {
        const grouped_input_len = batch_size * in_height * in_width * group_in_channels;
        const grouped_output_len = batch_size * out_height * out_width * group_out_channels;
        try cache_entry.ensureGroupedBuffers(grouped_input_len, grouped_output_len);
    } else {
        try cache_entry.ensureGroupedBuffers(0, 0);
    }

    var status = cmsis_nn.ARM_CMSIS_NN_SUCCESS;
    if (group_val == in_channels) {
        var dw_params = cmsis_nn.DwConvParams{
            .input_offset = -input_zero_point_s8,
            .output_offset = output_zero_point_s8,
            .ch_mult = @intCast(group_out_channels),
            .stride = .{ .h = @intCast(stride_h), .w = @intCast(stride_w) },
            .padding = .{ .h = @intCast(pad_h), .w = @intCast(pad_w) },
            .dilation = .{ .h = @intCast(dilation_h), .w = @intCast(dilation_w) },
            .activation = .{ .min = -128, .max = 127 },
        };
        var dw_filter_dims = cmsis_nn.Dims{ .n = 1, .h = @intCast(kernel_height), .w = @intCast(kernel_width), .c = @intCast(out_channels) };
        status = cmsis_nn.conv.arm_depthwise_conv_wrapper_s8(
            &ctx,
            &dw_params,
            &quant_params,
            &input_dims,
            input_ptr_s8,
            &dw_filter_dims,
            cache_entry.packed_weights.ptr,
            &bias_dims,
            if (bias_ptr) |ptr| @ptrCast(ptr) else null,
            &output_dims,
            output_ptr_s8,
        );
    } else if (group_val == 1) {
        status = cmsis_nn.conv.arm_convolve_wrapper_s8(
            &ctx,
            &conv_params,
            &quant_params,
            &input_dims,
            input_ptr_s8,
            &filter_dims,
            cache_entry.packed_weights.ptr,
            &bias_dims,
            if (bias_ptr) |ptr| @ptrCast(ptr) else null,
            &output_dims,
            output_ptr_s8,
        );
    } else {
        const grouped_in_buf = cache_entry.grouped_input;
        const grouped_out_buf = cache_entry.grouped_output;
        const total_input_pixels = batch_size * in_height * in_width;
        const total_output_pixels = batch_size * out_height * out_width;
        var g: usize = 0;
        while (g < group_val) : (g += 1) {
            const channel_offset_in = g * group_in_channels;
            const channel_offset_out = g * group_out_channels;
            for (0..total_input_pixels) |idx| {
                const src_base = idx * in_channels + channel_offset_in;
                const dst_base = idx * group_in_channels;
                std.mem.copy(i8, grouped_in_buf[dst_base .. dst_base + group_in_channels], source_input[src_base .. src_base + group_in_channels]);
            }

            const group_channel_offset = g * group_out_channels;
            var group_quant_params = cmsis_nn.PerChannelQuantParams{
                .multiplier = cache_entry.multipliers[group_channel_offset .. group_channel_offset + group_out_channels].ptr,
                .shift = cache_entry.shifts[group_channel_offset .. group_channel_offset + group_out_channels].ptr,
            };
            const weights_offset = g * group_out_channels * group_in_channels * kernel_height * kernel_width;
            const group_weights_ptr = cache_entry.packed_weights.ptr + weights_offset;
            const bias_group_ptr = if (bias_ptr) |ptr| ptr + group_channel_offset else null;

            status = cmsis_nn.conv.arm_convolve_wrapper_s8(
                &ctx,
                &conv_params,
                &group_quant_params,
                &input_group_dims,
                grouped_in_buf.ptr,
                &filter_group_dims,
                group_weights_ptr,
                &bias_group_dims,
                if (bias_group_ptr) |ptr| @ptrCast(ptr) else null,
                &output_group_dims,
                grouped_out_buf.ptr,
            );
            if (status != cmsis_nn.ARM_CMSIS_NN_SUCCESS) {
                break;
            }

            for (0..total_output_pixels) |idx| {
                const src_base = idx * group_out_channels;
                const dst_base = idx * out_channels + channel_offset_out;
                std.mem.copy(i8, output_slice[dst_base .. dst_base + group_out_channels], grouped_out_buf[src_base .. src_base + group_out_channels]);
            }
        }
    }

    if (status != cmsis_nn.ARM_CMSIS_NN_SUCCESS) {
        return TensorMathError.UnexpectedError;
    }

    output.clearAcceleratorView();
    var n: usize = 0;
    while (n < batch_size) : (n += 1) {
        var h: usize = 0;
        while (h < out_height) : (h += 1) {
            var w_: usize = 0;
            while (w_ < out_width) : (w_ += 1) {
                var c: usize = 0;
                while (c < out_channels) : (c += 1) {
                    const src_idx = ((n * out_height + h) * out_width + w_) * out_channels + c;
                    const dst_idx = ((n * out_channels + c) * out_height + h) * out_width + w_;
                    if (is_u8_input) {
                        const v = @as(i32, output_slice[src_idx]) + 128;
                        output.data[dst_idx] = @as(u8, @intCast(std.math.clamp(v, 0, 255)));
                    } else {
                        output.data[dst_idx] = @as(InputType, @intCast(output_slice[src_idx]));
                    }
                }
            }
        }
    }

    output.setAcceleratorView(.{
        .layout = AcceleratorLayout.nhwc_s8,
        .buffer_kind = AcceleratorBufferKind.output,
        .dims = .{ batch_size, out_height, out_width, out_channels },
        .buffer = output_slice,
        .scale = y_scale_val,
        .zero_point = output_zero_point,
        .owner_id = cache_entry.id,
        .generation = cache_entry.output_generation,
    });
}

/// Calculate output shape for QLinearConv - same as regular Conv
pub fn get_qlinearconv_output_shape(
    comptime T: type,
    input_shape: []const usize,
    weight_shape: []const usize,
    stride: ?[]const usize,
    pads: ?[]const usize,
    dilations: ?[]const usize,
    auto_pad: ?[]const u8,
) ![]usize {
    return conv.calculateOutputShape(T, input_shape, weight_shape, stride, pads, dilations, auto_pad);
}
