const testing = std.testing;

const std = @import("std");
const zant = @import("zant");
const IR_zant = @import("IR_zant");

const TensorZant = IR_zant.TensorZant;
const allocator = zant.utils.allocator.allocator;

const onnx = zant.onnx;
const TensorProto = onnx.TensorProto;
const StringStringEntryProto = onnx.StringStringEntryProto;

const Tensor = zant.core.tensor.Tensor;
const TensorCategory = IR_zant.TensorCategory;

const protoTensor2AnyTensor = IR_zant.utils.protoTensor2AnyTensor;

// Test for raw data not available
test "protoTensor2AnyTensor: float32 parsing" {
    std.debug.print("\n\n ------TEST: protoTensor2AnyTensor: float32 parsing", .{});

    var dims = [_]i64{ 2, 2 };
    var values = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    var proto = TensorProto{
        .dims = &dims,
        .data_type = .FLOAT,
        .segment = null,
        .name = null,
        .raw_data = null,
        .float16_data = null,
        .float_data = &values,
        .int32_data = null,
        .string_data = null,
        .int64_data = null,
        .double_data = null,
        .uint64_data = null,
        .uint16_data = null,
        .int8_data = null,
        .uint8_data = null,
        .doc_string = null,
        .external_data = &[_]*StringStringEntryProto{},
        .data_location = null,
        .metadata_props = &[_]*StringStringEntryProto{},
    };

    var anyTensor = try protoTensor2AnyTensor(&proto);
    defer anyTensor.deinit();

    // test size
    try testing.expectEqual(@as(usize, 4), anyTensor.f32.size);
    // test data
    try testing.expectEqual(1.0, anyTensor.f32.data[0]);
    try testing.expectEqual(2.0, anyTensor.f32.data[1]);
    try testing.expectEqual(3.0, anyTensor.f32.data[2]);
    try testing.expectEqual(4.0, anyTensor.f32.data[3]);
    // test shape
    try testing.expectEqual(2, anyTensor.f32.shape[0]);
    try testing.expectEqual(2, anyTensor.f32.shape[1]);
}

test "computeStride with 3D shape" {
    std.debug.print("\n\n ------TEST: computeStride with 3D shape", .{});

    var shape = [_]usize{ 2, 3, 4 };
    const expected = [_]usize{ 12, 4, 1 };

    const strides = try TensorZant.computeStride(&shape);

    try std.testing.expectEqualSlices(usize, &expected, strides);
}

test "computeStride with 2D shape" {
    std.debug.print("\n\n ------TEST: computeStride with 2D shape", .{});

    var shape = [_]usize{ 5, 10 };
    const expected = [_]usize{ 10, 1 };

    const strides = try TensorZant.computeStride(&shape);

    try std.testing.expectEqualSlices(usize, &expected, strides);
}

test "computeStride with 1D shape" {
    std.debug.print("\n\n ------TEST: computeStride with 1D shape", .{});

    var shape = [_]usize{7};
    const expected = [_]usize{1};

    const strides = try TensorZant.computeStride(&shape);

    try std.testing.expectEqualSlices(usize, &expected, strides);
}
test "get_tensorZantID generates consistent ID for same name" {
    std.debug.print("\n\n ------TEST: get_tensorZantID consistency", .{});

    var shape_data = [_]usize{ 2, 2 };
    const shape = shape_data[0..]; // equivale a `&shape_data`

    var tensor1 = try TensorZant.init(
        "my_tensor",
        null,
        null,
        shape,
        TensorCategory.INPUT,
    );

    var tensor2 = try TensorZant.init(
        "my_tensor",
        null,
        null,
        shape,
        TensorCategory.OUTPUT,
    );

    const id1 = (&tensor1).get_tensorZantID();
    const id2 = (&tensor2).get_tensorZantID();

    try std.testing.expectEqual(id1, id2);
}
