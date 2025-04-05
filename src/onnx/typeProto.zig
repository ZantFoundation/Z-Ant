const std = @import("std");
const protobuf = @import("protobuf.zig");
const AttributeType = @import("onnx.zig").AttributeType;
const TensorShapeProto = @import("onnx.zig").TensorShapeProto;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
var printingAllocator = std.heap.ArenaAllocator.init(gpa.allocator());

//https://github.com/onnx/onnx/blob/main/onnx/onnx.proto#L719
//TAG oneof:
//  - 1: tensor_type, type: TypeProto.Tensor
//  - 4: sequence_type, type: TypeProto.Sequence
//  - 5: map_type, type: TypeProto.Map
//  - 6: denotation, type: []const u8
//  - 8: sparse_tensor_type, type: TypeProto.SparseTensor
//  - 9: optional_type, type: TypeProto.Optional
pub const TypeProto = struct {
    //TENSOR TAG:
    //  - 1: elem_type int32
    //  - 2: shape TensorShapeProto
    pub const Tensor = struct {
        elem_type: u32,
        shape: ?*TensorShapeProto,

        pub fn deinit(self: *Tensor, allocator: std.mem.Allocator) void {
            if (self.shape) |s| {
                s.deinit(allocator);
                allocator.destroy(s);
            }
        }

        pub fn parse(reader: *protobuf.ProtoReader) !Tensor {
            var tensor = Tensor{
                .elem_type = 0,
                .shape = null,
            };

            _ = &tensor;

            while (reader.hasMore()) {
                const tag = try reader.readTag();

                switch (tag.field_number) {
                    1 => { //elem_type
                        tensor.elem_type = @intCast(try reader.readVarint());
                    },
                    2 => { //shape
                        var shape_reader = try reader.readLengthDelimited(); //var shape_reader
                        const shape_ptr = try reader.allocator.create(TensorShapeProto);
                        shape_ptr.* = try TensorShapeProto.parse(&shape_reader);
                        tensor.shape = shape_ptr;
                    },
                    else => {
                        std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE for TensorProto\n\n", .{tag});
                        try reader.skipField(tag.wire_type);
                    },
                }
            }

            return tensor;
        }

        pub fn print(self: *Tensor, padding: ?[]const u8) void {
            const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
                return;
            };
            std.debug.print("{s}------------- TENSOR_TYPE\n", .{space});

            std.debug.print("{s}Element Type: {}\n", .{ space, self.elem_type });

            if (self.shape) |s| {
                std.debug.print("{s}Shape:\n", .{space});
                s.print(space);
            } else {
                std.debug.print("{s}Shape: (none)\n", .{space});
            }
        }
    };

    //SEQUENCE TAG:
    //  - 1: elem_type TypeProto
    pub const Sequence = struct {
        elem_type: ?*TypeProto,

        pub fn deinit(self: *Sequence, allocator: std.mem.Allocator) void {
            if (self.elem_type) |e| {
                e.deinit(allocator);
                allocator.destroy(e);
            }
        }

        pub fn parse(reader: *protobuf.ProtoReader) !Sequence {
            var sequence = Sequence{
                .elem_type = null,
            };

            _ = &sequence;

            while (reader.hasMore()) {
                const tag = try reader.readTag();

                switch (tag.field_number) {
                    1 => { //elem_type
                        _ = try reader.readLengthDelimited(); //var elem_type_reader
                        // const elem_type_ptr = try reader.allocator.create(TypeProto);
                        // elem_type_ptr.* = try TypeProto.parse(&elem_type_reader);
                        // sequence.elem_type = elem_type_ptr;
                    },
                    else => {
                        std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE for ", .{tag});
                        unreachable;
                    },
                }
            }

            return sequence;
        }

        pub fn print(self: *Sequence, padding: ?[]const u8) void {
            const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
                return;
            };
            std.debug.print("{s}------------- SEQUENCE\n", .{space});

            if (self.elem_type) |t| {
                std.debug.print("{s}Element Type:\n", .{space});
                t.print(space);
            } else {
                std.debug.print("{s}Element Type: (none)\n", .{space});
            }
        }
    };

    //MAP TAG:
    //  - 1: key_type u32
    //  - 2: value_type TypeProto
    pub const Map = struct {
        key_type: u32,
        value_type: ?*TypeProto,

        pub fn deinit(self: *Map, allocator: std.mem.Allocator) void {
            if (self.value_type) |v| {
                v.deinit(allocator);
                allocator.destroy(v);
            }
        }

        pub fn parse(reader: *protobuf.ProtoReader) !Map {
            var map = Map{
                .key_type = 0,
                .value_type = null,
            };

            while (reader.hasMore()) {
                const tag = try reader.readTag();

                switch (tag.field_number) {
                    1 => { //elem_type
                        const elem_type = try reader.readVarint();
                        map.key_type = @intCast(elem_type);
                    },
                    2 => { //value_type
                        _ = try reader.readLengthDelimited(); //var value_type_reader
                        // const value_ptr = try reader.allocator.create(TypeProto);
                        // value_ptr.* = try TypeProto.parse(&value_type_reader);
                        // map.value_type = value_ptr;
                    },
                    else => {
                        std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE ", .{tag});
                        unreachable;
                    },
                }
            }

            return map;
        }

        pub fn print(self: *Map, padding: ?[]const u8) void {
            const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
                return;
            };
            std.debug.print("{s}------------- MAP\n", .{space});

            std.debug.print("{s}Key Type: {}\n", .{ space, self.key_type });

            if (self.value_type) |v| {
                std.debug.print("{s}Value Type:\n", .{space});
                v.print(space);
            } else {
                std.debug.print("{s}Value Type: (none)\n", .{space});
            }
        }
    };

    //SPARSE TENSOR
    //  - 1: elem_type int32
    //  - 2: shape TensorShapeProto
    pub const SparseTensor = struct {
        elem_type: u32,
        shape: ?*TensorShapeProto,

        pub fn deinit(self: *SparseTensor, allocator: std.mem.Allocator) void {
            if (self.shape) |s| {
                s.deinit(allocator);
                allocator.destroy(s);
            }
        }

        pub fn parse(reader: *protobuf.ProtoReader) !SparseTensor {
            var sparse_tensor = SparseTensor{
                .elem_type = 0,
                .shape = null,
            };

            _ = &sparse_tensor;

            while (reader.hasMore()) {
                const tag = try reader.readTag();

                switch (tag.field_number) {
                    1 => { //elem_type
                        const elem_type = try reader.readVarint();
                        sparse_tensor.elem_type = @intCast(elem_type);
                    },
                    2 => { //shape
                        var shape_reader = try reader.readLengthDelimited();
                        const shape_ptr = try reader.allocator.create(TensorShapeProto);
                        shape_ptr.* = try TensorShapeProto.parse(&shape_reader);
                        sparse_tensor.shape = shape_ptr;
                    },
                    else => {
                        std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE ", .{tag});
                        unreachable;
                    },
                }
            }

            return sparse_tensor;
        }

        pub fn print(self: *SparseTensor, padding: ?[]const u8) void {
            const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
                return;
            };
            std.debug.print("{s}------------- SparseTensor\n", .{space});
            std.debug.print("{s}Element Type: {}\n", .{ space, self.elem_type });

            if (self.shape) |s| {
                std.debug.print("{s}Shape:\n", .{space});
                s.print(space);
            } else {
                std.debug.print("{s}Shape: (none)\n", .{space});
            }
        }
    };

    //TAG OPTIONAL
    //  - 1: elem_type TypeProto
    pub const Optional = struct {
        elem_type: ?*TypeProto,

        pub fn deinit(self: *Optional, allocator: std.mem.Allocator) void {
            if (self.elem_type) |e| {
                e.deinit(allocator);
                allocator.destroy(e);
            }
        }

        pub fn parse(reader: *protobuf.ProtoReader) !Optional {
            var opt = Optional{
                .elem_type = null,
            };

            _ = &opt;

            while (reader.hasMore()) {
                const tag = try reader.readTag();

                switch (tag.field_number) {
                    1 => { //elem_type
                        _ = try reader.readLengthDelimited();
                        //const elm_ptr = try reader.allocator.create(TypeProto);
                        //elm_ptr.* = try TypeProto.parse(&elem_type_reader);
                        //opt.elem_type = elm_ptr;
                    },
                    else => {
                        std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE ", .{tag});
                        unreachable;
                    },
                }
            }

            return opt;
        }

        pub fn print(self: *Optional, padding: ?[]const u8) void {
            const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
                return;
            };
            std.debug.print("{s}------------- OPTIONAL\n", .{space});
            if (self.elem_type) |t| {
                std.debug.print("{s}Element Type:\n", .{space});
                t.print(space);
            } else {
                std.debug.print("{s}Element Type: (none)\n", .{space});
            }
        }
    };

    tensor_type: ?*Tensor,
    sequence_type: ?*Sequence,
    map_type: ?*Map,
    sparse_tensor_type: ?*SparseTensor,
    optional_type: ?*Optional,
    denotation: ?[]const u8,

    pub fn deinit(self: *TypeProto, allocator: std.mem.Allocator) void {
        if (self.tensor_type) |s| {
            s.deinit(allocator);
            allocator.destroy(s);
        }
        if (self.sequence_type) |st| {
            st.deinit(allocator);
            allocator.destroy(st);
        }
        if (self.map_type) |m| {
            m.deinit(allocator);
            allocator.destroy(m);
        }
        if (self.sparse_tensor_type) |stt| {
            stt.deinit(allocator);
            allocator.destroy(stt);
        }
        if (self.optional_type) |ot| {
            ot.deinit(allocator);
            allocator.destroy(ot);
        }
        if (self.denotation) |d| allocator.free(d);
    }

    pub fn parse(reader: *protobuf.ProtoReader) !TypeProto {
        var typeProto = TypeProto{
            .tensor_type = null,
            .sequence_type = null,
            .map_type = null,
            .sparse_tensor_type = null,
            .optional_type = null,
            .denotation = null,
        };

        _ = &typeProto;

        while (reader.hasMore()) {
            const tag = try reader.readTag();

            switch (tag.field_number) {
                1 => { //tensor_type
                    var tensor_type_reader = try reader.readLengthDelimited();
                    const ensor_type_ptr = try reader.allocator.create(Tensor);
                    ensor_type_ptr.* = try Tensor.parse(&tensor_type_reader);
                    typeProto.tensor_type = ensor_type_ptr;
                },
                4 => { //sequence_type
                    var sequence_reader = try reader.readLengthDelimited(); //var sequence_reader
                    const sequence_ptr = try reader.allocator.create(Sequence);
                    sequence_ptr.* = try Sequence.parse(&sequence_reader);
                    typeProto.sequence_type = sequence_ptr;
                },
                5 => { //map_type
                    var map_reader = try reader.readLengthDelimited();
                    const map_ptr = try reader.allocator.create(Map);
                    map_ptr.* = try Map.parse(&map_reader);
                    typeProto.map_type = map_ptr;
                },
                6 => { //  denotation
                    typeProto.denotation = try reader.readString(reader.allocator);
                },
                8 => { // sparse_tensor_type
                    var sparse_tensor_type_reader = try reader.readLengthDelimited();
                    const sp_tensor_ptr = try reader.allocator.create(SparseTensor);
                    sp_tensor_ptr.* = try SparseTensor.parse(&sparse_tensor_type_reader);
                    typeProto.sparse_tensor_type = sp_tensor_ptr;
                },
                9 => { // optional_type
                    var optional_reader = try reader.readLengthDelimited();
                    const opt_ptr = try reader.allocator.create(Optional);
                    opt_ptr.* = try Optional.parse(&optional_reader);
                    typeProto.optional_type = opt_ptr;
                },
                else => {
                    std.debug.print("\n\n ERROR: tag{} NOT AVAILABLE for TypeProto", .{tag});
                    try reader.skipField(tag.wire_type);
                },
            }
        }

        return typeProto;
    }

    pub fn print(self: *TypeProto, padding: ?[]const u8) void {
        const space = std.mem.concat(printingAllocator.allocator(), u8, &[_][]const u8{ if (padding) |p| p else "", "   " }) catch {
            return;
        };

        std.debug.print("{s}------------- TYPE\n", .{space});

        if (self.tensor_type) |t| {
            std.debug.print("{s}Tensor Type:\n", .{space});
            t.print(space);
        } else {
            std.debug.print("{s}Tensor Type: (none)\n", .{space});
        }

        if (self.sequence_type) |s| {
            std.debug.print("{s}Sequence Type:\n", .{space});
            s.print(space);
        } else {
            std.debug.print("{s}Sequence Type: (none)\n", .{space});
        }

        if (self.map_type) |m| {
            std.debug.print("{s}Map Type:\n", .{space});
            m.print(space);
        } else {
            std.debug.print("{s}Map Type: (none)\n", .{space});
        }

        if (self.sparse_tensor_type) |st| {
            std.debug.print("{s}Sparse Tensor Type:\n", .{space});
            st.print(space);
        } else {
            std.debug.print("{s}Sparse Tensor Type: (none)\n", .{space});
        }

        if (self.optional_type) |o| {
            std.debug.print("{s}Optional Type:\n", .{space});
            o.print(space);
        } else {
            std.debug.print("{s}Optional Type: (none)\n", .{space});
        }

        if (self.denotation) |d| {
            std.debug.print("{s}Denotation: {s}\n", .{ space, d });
        } else {
            std.debug.print("{s}Denotation: (none)\n", .{space});
        }
    }
};
