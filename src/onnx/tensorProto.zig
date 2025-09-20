const std = @import("std");
const protobuf = @import("protobuf.zig");
const AttributeType = @import("onnx.zig").AttributeType;
const DataType = @import("onnx.zig").DataType;
const DataLocation = @import("onnx.zig").DataLocation;
const StringStringEntryProto = @import("stringStringEntryProto.zig").StringStringEntryProto;
const Segment = @import("segment.zig").Segment;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var printingAllocator = std.heap.ArenaAllocator.init(gpa.allocator());

const onnx_log = std.log.scoped(.tensorProto);

// onnx library reference: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L503
//TAGS:
//  - 1 : dims, repeated int64
//  - 2 : data_type, optional int32
//  - 3 : segment, optional Segment
//  - 4 : float_data, repeated float
//  - 5 : int32_data, repeated int32
//  - 6 : string_data, repeated bytes
//  - 7 : int64_data, repeated int64
//  - 8 : name, optional string
//  - 9 : raw_data, optional bytes
//  - 10: double_data, repeated double
//  - 11: uint64_data, repeated uint64
//  - 12: doc_string, optional string
//  - 13: external_data, repeated StringStringEntryProto
//  - 14: data_location, optional DataLocation
//  - 15: uint16_data, repeated uint16
//  - 16: metadata_props, repeated StringStringEntryProto
pub const TensorProto = struct {
    dims: []i64,
    data_type: DataType,
    segment: ?*Segment,
    name: ?[]const u8,
    raw_data: ?[]const u8, //DataType not necessary, it is a universal representation
    float16_data: ?[]f16, //DataType = .FLOAT16
    float_data: ?[]f32, //DataType = .FLOAT
    int32_data: ?[]i32, //DataType = .INT32
    string_data: ?[][]const u8,
    int64_data: ?[]i64, //DataType = .INT64
    double_data: ?[]f64, //DataType = .DOUBLE
    uint64_data: ?[]u64, //DataType = .UINT64
    uint16_data: ?[]u16, //DataType = .UINT16
    int8_data: ?[]i8, //DataType = .INT8
    uint8_data: ?[]u8, //DataType = .UINT8
    doc_string: ?[]const u8,
    external_data: []*StringStringEntryProto,
    data_location: ?DataLocation,
    metadata_props: []*StringStringEntryProto,

    pub fn deinit(self: *TensorProto, allocator: std.mem.Allocator) void {
        allocator.free(self.dims);
        if (self.name) |n| allocator.free(n);
        if (self.doc_string) |doc_string| allocator.free(doc_string);

        // Free allocated pointer fields if they exist
        if (self.float16_data) |data| allocator.free(data);
        if (self.float_data) |data| allocator.free(data);
        if (self.int32_data) |data| allocator.free(data);
        if (self.int64_data) |data| allocator.free(data);
        if (self.double_data) |data| allocator.free(data);
        if (self.uint64_data) |data| allocator.free(data);
        if (self.uint16_data) |data| allocator.free(data);
        if (self.int8_data) |data| allocator.free(data);
        if (self.uint8_data) |data| allocator.free(data);

        // Free string_data
        if (self.string_data) |data| {
            for (data) |str| allocator.free(str);
            allocator.free(data);
        }

        // Free pointer slices
        for (self.external_data) |ed| {
            ed.deinit(allocator);
            allocator.destroy(ed);
        }
        allocator.free(self.external_data);

        for (self.metadata_props) |mp| {
            mp.deinit(allocator);
            allocator.destroy(mp);
        }
        allocator.free(self.metadata_props);

        // Free raw_data if it wasn't converted and freed already
        if (self.raw_data) |data| allocator.free(data);

        // Null out pointers (good practice)
        self.float16_data = null;
        self.float_data = null;
        self.int32_data = null;
        self.int64_data = null;
        self.double_data = null;
        self.uint64_data = null;
        self.uint16_data = null;
        self.int8_data = null;
        self.uint8_data = null;
        self.string_data = null;
        self.segment = null;
        self.name = null;
        self.doc_string = null;
        self.external_data = &.{};
        self.metadata_props = &.{};
        self.external_data = &.{}; // Use empty slice literal
        self.metadata_props = &.{}; // Use empty slice literal
    }

    pub fn parse(reader: *protobuf.ProtoReader) !TensorProto {
        var tensor = TensorProto{
            .dims = &[_]i64{},
            .data_type = .UNDEFINED,
            .segment = null,
            .name = null,
            .raw_data = null,
            .float16_data = null,
            .float_data = null,
            .int32_data = null,
            .string_data = null,
            .int64_data = null,
            .double_data = null,
            .uint64_data = null,
            .uint16_data = null,
            .int8_data = null,
            .uint8_data = null,
            .doc_string = null,
            .external_data = undefined,
            .data_location = null,
            .metadata_props = undefined,
        };

        var dims = std.ArrayList(i64).init(reader.allocator);
        defer dims.deinit();
        var dataLen: i64 = 1; //sometimes dims: []i64 is not reported so you have to assume it from the data lenght
        var externalDataList = std.ArrayList(*StringStringEntryProto).init(reader.allocator);
        defer externalDataList.deinit();
        var metaDataList = std.ArrayList(*StringStringEntryProto).init(reader.allocator);
        defer metaDataList.deinit();

        while (reader.hasMore()) {
            const tag = try reader.readTag();
            switch (tag.field_number) {
                1 => { // dims
                    const value = try reader.readVarint();
                    try dims.append(@as(i64, @intCast(value)));
                },
                2 => { // data_type
                    const value = try reader.readVarint();
                    tensor.data_type = @enumFromInt((value));
                },
                3 => {
                    var segment_read = try reader.readLengthDelimited(); //var dim_reader
                    const seg_ptr = try reader.allocator.create(Segment);
                    seg_ptr.* = try Segment.parse(&segment_read);
                    tensor.segment = seg_ptr;
                },
                8 => { // name
                    tensor.name = try reader.readString(reader.allocator);
                },
                9 => { // raw_data
                    tensor.raw_data = try reader.readBytes(reader.allocator);
                },
                4 => { // float_data
                    var data = std.ArrayList(f32).init(reader.allocator);
                    defer data.deinit();
                    if (tag.wire_type == .LengthDelimited) {
                        var float_reader = try reader.readLengthDelimited();
                        while (float_reader.hasMore()) {
                            if (float_reader.available() < 4) break;
                            const value = try float_reader.readFixed32();
                            try data.append(@bitCast(value));
                        }
                    } else {
                        const value = try reader.readFixed32();
                        try data.append(@bitCast(value));
                    }
                    dataLen = @intCast(data.items.len);
                    tensor.float_data = try data.toOwnedSlice();
                },
                5 => { // int32_data
                    var data = std.ArrayList(i32).init(reader.allocator);
                    defer data.deinit();
                    if (tag.wire_type == .LengthDelimited) {
                        var int_reader = try reader.readLengthDelimited();
                        while (int_reader.hasMore()) {
                            const value = try int_reader.readVarint();
                            const value64 = @as(u64, @intCast(value));
                            const value32_truncated = @as(u32, @truncate(value64));
                            try data.append(@bitCast(value32_truncated));
                        }
                    } else {
                        const value = try reader.readVarint();
                        const value64 = @as(u64, @intCast(value));
                        const value32_truncated = @as(u32, @truncate(value64));
                        try data.append(@bitCast(value32_truncated));
                    }
                    dataLen = @intCast(data.items.len);
                    tensor.int32_data = try data.toOwnedSlice();
                },
                7 => { // int64_data
                    var data = std.ArrayList(i64).init(reader.allocator);
                    defer data.deinit();
                    if (tag.wire_type == .LengthDelimited) {
                        var int_reader = try reader.readLengthDelimited();
                        while (int_reader.hasMore()) {
                            const value = try int_reader.readVarint();
                            try data.append(@bitCast(value));
                        }
                    } else {
                        const value = try reader.readVarint();
                        try data.append(@bitCast(value));
                    }
                    dataLen = @intCast(data.items.len);
                    tensor.int64_data = try data.toOwnedSlice();
                },
                10 => { // double_data
                    var data = std.ArrayList(f64).init(reader.allocator);
                    defer data.deinit();
                    if (tag.wire_type == .LengthDelimited) {
                        var double_reader = try reader.readLengthDelimited();
                        while (double_reader.hasMore()) {
                            if (double_reader.available() < 8) break;
                            const value = try double_reader.readFixed64();
                            try data.append(@bitCast(value));
                        }
                    } else {
                        const value = try reader.readFixed64();
                        try data.append(@bitCast(value));
                    }
                    dataLen = @intCast(data.items.len);
                    tensor.double_data = try data.toOwnedSlice();
                },
                11 => { // uint64_data
                    var data = std.ArrayList(u64).init(reader.allocator);
                    defer data.deinit();
                    if (tag.wire_type == .LengthDelimited) {
                        var uint_reader = try reader.readLengthDelimited();
                        while (uint_reader.hasMore()) {
                            const value = try uint_reader.readVarint();
                            try data.append(value);
                        }
                    } else {
                        const value = try reader.readVarint();
                        try data.append(value);
                    }
                    dataLen = @intCast(data.items.len);
                    tensor.uint64_data = try data.toOwnedSlice();
                },
                12 => {
                    tensor.doc_string = try reader.readString(reader.allocator);
                },
                13 => {
                    var md_reader = try reader.readLengthDelimited(); //var md_reader
                    const ssep_ptr = try reader.allocator.create(StringStringEntryProto);
                    ssep_ptr.* = try StringStringEntryProto.parse(&md_reader);
                    try externalDataList.append(ssep_ptr);
                },
                14 => { //data location
                    const value = try reader.readVarint();
                    tensor.data_location = @enumFromInt((value));
                },
                16 => {
                    var md_reader = try reader.readLengthDelimited(); //var md_reader
                    const ssep_ptr = try reader.allocator.create(StringStringEntryProto);
                    ssep_ptr.* = try StringStringEntryProto.parse(&md_reader);
                    try metaDataList.append(ssep_ptr);
                },
                else => {
                    onnx_log.warn("\n\n ERROR: tag{} NOT AVAILABLE for TensorProto\n\n", .{tag});
                    try reader.skipField(tag.wire_type);
                },
            }
        }

        //from Raw data to Data Type
        if (tensor.raw_data) |_| try tensor.fromRawDataToDataType(reader.allocator);

        //check on the dims
        if (dims.items.len == 0) try dims.append(dataLen);
        tensor.dims = try dims.toOwnedSlice();
        tensor.external_data = try externalDataList.toOwnedSlice();
        tensor.metadata_props = try metaDataList.toOwnedSlice();

        return tensor;
    }
    pub fn print(self: *TensorProto, padding: ?[]const u8) void {
        const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
            return;
        };

        std.debug.print("{s}------------- TENSOR\n", .{space});

        if (self.name) |n| {
            std.debug.print("{s}Name: {s}\n", .{ space, n });
        } else {
            std.debug.print("{s}Name: (none)\n", .{space});
        }

        if (self.segment) |segment| {
            std.debug.print("{s}Segment:\n", .{space});
            segment.print(space);
        }

        std.debug.print("{s}Data Type: {any}\n", .{ space, self.data_type });

        std.debug.print("{s}Dims: [", .{space});
        for (
            self.dims,
        ) |dim| {
            std.debug.print("{}", .{dim});
        }
        std.debug.print("]\n", .{});

        if (self.raw_data) |raw| {
            std.debug.print("{s}Raw Data: {} bytes\n", .{ space, raw.len });
        }

        if (self.float_data) |data| {
            std.debug.print("{s}Float Data: [", .{space});
            for (0..if (data.len < 10) data.len else 10) |i| {
                std.debug.print("{}", .{data[i]});
            }
            if (data.len >= 10) std.debug.print(", ... ", .{});
            std.debug.print("]\n", .{});
        }

        if (self.int32_data) |data| {
            std.debug.print("{s}Int32 Data: [", .{space});
            for (0..if (data.len < 10) data.len else 10) |i| {
                std.debug.print("{}", .{data[i]});
            }
            if (data.len >= 10) std.debug.print(", ... ", .{});
            std.debug.print("]\n", .{});
        }

        if (self.int64_data) |data| {
            std.debug.print("{s}Int64 Data: [", .{space});
            for (0..if (data.len < 10) data.len else 10) |i| {
                std.debug.print("{}", .{data[i]});
            }
            if (data.len >= 10) std.debug.print(", ... ", .{});
            std.debug.print("]\n", .{});
        }

        if (self.double_data) |data| {
            std.debug.print("{s}Double Data: [", .{space});
            for (0..if (data.len < 10) data.len else 10) |i| {
                std.debug.print("{}", .{data[i]});
            }
            if (data.len >= 10) std.debug.print(", ... ", .{});
            std.debug.print("]\n", .{});
        }

        if (self.uint64_data) |data| {
            std.debug.print("{s}UInt64 Data: [", .{space});
            for (0..if (data.len < 10) data.len else 10) |i| {
                std.debug.print("{}", .{data[i]});
            }
            if (data.len >= 10) std.debug.print(", ... ", .{});
            std.debug.print("]\n", .{});
        }

        if (self.uint16_data) |data| {
            std.debug.print("{s}UInt16 Data: [", .{space});
            for (0..if (data.len < 10) data.len else 10) |i| {
                std.debug.print("{}", .{data[i]});
            }
            if (data.len >= 10) std.debug.print(", ... ", .{});
            std.debug.print("]\n", .{});
        }

        if (self.string_data) |data| {
            std.debug.print("  String Data: [", .{});
            for (data) |val| {
                std.debug.print("\"{s}\"", .{val});
            }
            std.debug.print("]\n", .{});
        }

        std.debug.print("{s}External Data (key, value) [{}]: \n", .{ space, self.external_data.len });
        for (self.external_data) |ex| {
            ex.print(space);
        }

        std.debug.print("{s}Data Location: {any}\n", .{ space, self.data_location });

        std.debug.print("{s}metadata_props (key, value) [{}]: \n", .{ space, self.metadata_props.len });
        for (self.metadata_props) |mp| {
            mp.print(space);
        }
    }

    inline fn fromRawDataToDataType(self: *TensorProto, allocator: std.mem.Allocator) !void {
        if (self.raw_data == null) return; // Nothing to do if raw_data is null

        const raw_data_slice = self.raw_data.?; // Keep the original optional pointer
        const data_type = self.data_type;
        var free_raw_data = true; // Flag to control freeing

        // Defer freeing only if the flag is set and the pointer hasn't been nulled
        // Note: The original self.raw_data points to the memory we might free.
        // It's crucial that self.raw_data isn't used after this function if conversion fails.
        // Consider documenting this behavior or returning an error for unsupported types.
        const original_raw_data_ptr = self.raw_data;
        defer if (free_raw_data and original_raw_data_ptr != null) allocator.free(original_raw_data_ptr.?);

        switch (data_type) {
            .FLOAT16 => {
                const num_elements = @divExact(raw_data_slice.len, @sizeOf(f16));
                const typed_data = try allocator.alignedAlloc(f16, @alignOf(f16), num_elements);
                errdefer allocator.free(typed_data);
                const dest_bytes = std.mem.sliceAsBytes(typed_data);
                @memcpy(dest_bytes, raw_data_slice);
                self.float16_data = typed_data;
            },
            .FLOAT => {
                const num_elements = @divExact(raw_data_slice.len, @sizeOf(f32));
                const typed_data = try allocator.alignedAlloc(f32, @alignOf(f32), num_elements);
                errdefer allocator.free(typed_data);
                const dest_bytes = std.mem.sliceAsBytes(typed_data);
                @memcpy(dest_bytes, raw_data_slice);
                self.float_data = typed_data;
            },
            .INT32 => {
                const num_elements = @divExact(raw_data_slice.len, @sizeOf(i32));
                const typed_data = try allocator.alignedAlloc(i32, @alignOf(i32), num_elements);
                errdefer allocator.free(typed_data);
                const dest_bytes = std.mem.sliceAsBytes(typed_data);
                @memcpy(dest_bytes, raw_data_slice);
                self.int32_data = typed_data;
            },
            .INT64 => {
                const num_elements = @divExact(raw_data_slice.len, @sizeOf(i64));
                const typed_data = try allocator.alignedAlloc(i64, @alignOf(i64), num_elements);
                errdefer allocator.free(typed_data);
                const dest_bytes = std.mem.sliceAsBytes(typed_data);
                @memcpy(dest_bytes, raw_data_slice);
                self.int64_data = typed_data;
            },
            .DOUBLE => {
                const num_elements = @divExact(raw_data_slice.len, @sizeOf(f64));
                const typed_data = try allocator.alignedAlloc(f64, @alignOf(f64), num_elements);
                errdefer allocator.free(typed_data);
                const dest_bytes = std.mem.sliceAsBytes(typed_data);
                @memcpy(dest_bytes, raw_data_slice);
                self.double_data = typed_data;
            },
            .UINT64 => {
                const num_elements = @divExact(raw_data_slice.len, @sizeOf(u64));
                const typed_data = try allocator.alignedAlloc(u64, @alignOf(u64), num_elements);
                errdefer allocator.free(typed_data);
                const dest_bytes = std.mem.sliceAsBytes(typed_data);
                @memcpy(dest_bytes, raw_data_slice);
                self.uint64_data = typed_data;
            },
            .UINT16 => {
                const num_elements = @divExact(raw_data_slice.len, @sizeOf(u16));
                const typed_data = try allocator.alignedAlloc(u16, @alignOf(u16), num_elements);
                errdefer allocator.free(typed_data);
                const dest_bytes = std.mem.sliceAsBytes(typed_data);
                @memcpy(dest_bytes, raw_data_slice);
                self.uint16_data = typed_data;
            },
            .INT8 => {
                const num_elements = raw_data_slice.len; // INT8 is 1 byte each
                const typed_data = try allocator.alignedAlloc(i8, @alignOf(i8), num_elements);
                errdefer allocator.free(typed_data);
                for (typed_data, 0..) |*dest, i| {
                    dest.* = @bitCast(raw_data_slice[i]);
                }
                self.int8_data = typed_data;
            },
            .UINT8 => {
                const num_elements = raw_data_slice.len; // UINT8 is 1 byte each
                const typed_data = try allocator.alignedAlloc(u8, @alignOf(u8), num_elements);
                errdefer allocator.free(typed_data);
                @memcpy(typed_data, raw_data_slice);
                self.uint8_data = typed_data;
            },
            else => {
                onnx_log.warn("\n Data type conversion not supported for {s}, keeping raw data \n", .{@tagName(data_type)});
                // If conversion is not supported, we keep the raw_data.
                // We need to prevent the deferred free.
                free_raw_data = false;
                // self.raw_data remains unchanged.
                return; // Exit
            },
        }

        // If conversion succeeded, null out raw_data pointer.
        // The deferred free will handle the original raw_data_slice memory because free_raw_data is still true.
        self.raw_data = null;
    }
};
