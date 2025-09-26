const std = @import("std");
const zant = @import("zant");
const IR_zant = @import("IR_zant");

const allocator = zant.utils.allocator.allocator;

const TensorZant = IR_zant.tensorZant_lib.TensorZant;
const TensorType = IR_zant.tensorZant_lib.TensorType;
const TensorCategory = IR_zant.tensorZant_lib.TensorCategory;
const NodeZant = IR_zant.NodeZant;
const Op_union = IR_zant.operators.Op_union;
const operators = IR_zant.operators;
const fused_ops = IR_zant.fused_operators;

fn initTensor(
    name: []const u8,
    ty: TensorType,
    category: TensorCategory,
    shape_values: []const usize,
) !TensorZant {
    var shape = try allocator.alloc(usize, shape_values.len);
    std.mem.copy(usize, shape, shape_values);

    return TensorZant{
        .name = name,
        .ty = ty,
        .tc = category,
        .ptr = null,
        .shape = shape,
        .stride = try TensorZant.computeStride(shape),
    };
}

fn initScalarTensor(name: []const u8, ty: TensorType, category: TensorCategory) !TensorZant {
    return initTensor(name, ty, category, &[_]usize{1});
}

fn destroyTensor(tensor: *TensorZant) void {
    allocator.free(tensor.shape);
    allocator.free(tensor.stride);
}

test "fused dequant-clip-quant output shape matches input" {
    var x = try initTensor("x", TensorType.u8, TensorCategory.LINK, &[_]usize{ 1, 16, 48, 50 });
    defer destroyTensor(&x);

    var x_scale = try initScalarTensor("x_scale", TensorType.f32, TensorCategory.INITIALIZER);
    defer destroyTensor(&x_scale);

    var x_zp = try initScalarTensor("x_zp", TensorType.u8, TensorCategory.INITIALIZER);
    defer destroyTensor(&x_zp);

    var dequant_y = try initTensor("dequant_y", TensorType.f32, TensorCategory.LINK, &[_]usize{ 1, 16, 48, 50 });
    defer destroyTensor(&dequant_y);

    var clip_y = try initTensor("clip_y", TensorType.f32, TensorCategory.LINK, &[_]usize{ 1, 16, 48, 50 });
    defer destroyTensor(&clip_y);

    var y_scale = try initScalarTensor("y_scale", TensorType.f32, TensorCategory.INITIALIZER);
    defer destroyTensor(&y_scale);

    var y_zp = try initScalarTensor("y_zp", TensorType.u8, TensorCategory.INITIALIZER);
    defer destroyTensor(&y_zp);

    var quant_y = try initTensor("quant_y", TensorType.u8, TensorCategory.LINK, &[_]usize{ 1, 16, 48, 48 });
    defer destroyTensor(&quant_y);

    var dequant = operators.DequantizeLinear{
        .x = &x,
        .x_scale = &x_scale,
        .x_zero_point = &x_zp,
        .y = &dequant_y,
        .axis = 1,
        .block_size = 0,
        .output_dtype = TensorType.f32,
    };

    var clip = operators.Clip{
        .input = &dequant_y,
        .min = null,
        .max = null,
        .output = &clip_y,
    };

    var quant = operators.QuantizeLinear{
        .x = &clip_y,
        .y_scale = &y_scale,
        .y_zero_point = &y_zp,
        .y = &quant_y,
        .axis = 1,
        .block_size = 0,
        .output_dtype = TensorType.u8,
        .precision = 0,
        .saturate = 1,
    };

    var dequant_node = try allocator.create(NodeZant);
    defer {
        dequant_node.next.deinit();
        allocator.destroy(dequant_node);
    }
    dequant_node.* = NodeZant{
        .name = "dequant",
        .op_type = "DequantizeLinear",
        .op = Op_union{ .dequantizeLinear = dequant },
        .next = std.ArrayList(*NodeZant).init(allocator),
        .is_fused = false,
        .nodeProto = null,
        .ready = false,
    };

    var clip_node = try allocator.create(NodeZant);
    defer {
        clip_node.next.deinit();
        allocator.destroy(clip_node);
    }
    clip_node.* = NodeZant{
        .name = "clip",
        .op_type = "Clip",
        .op = Op_union{ .clip = clip },
        .next = std.ArrayList(*NodeZant).init(allocator),
        .is_fused = false,
        .nodeProto = null,
        .ready = false,
    };

    var quant_node = try allocator.create(NodeZant);
    defer {
        quant_node.next.deinit();
        allocator.destroy(quant_node);
    }
    quant_node.* = NodeZant{
        .name = "quant",
        .op_type = "QuantizeLinear",
        .op = Op_union{ .quantizeLinear = quant },
        .next = std.ArrayList(*NodeZant).init(allocator),
        .is_fused = false,
        .nodeProto = null,
        .ready = false,
    };

    var fusion_nodes = std.ArrayList(*NodeZant).init(allocator);
    defer fusion_nodes.deinit();

    try fusion_nodes.append(dequant_node);
    try fusion_nodes.append(clip_node);
    try fusion_nodes.append(quant_node);

    const fused = try fused_ops.Fused_Dequant_Clip_Quant.init_fused_op(fusion_nodes);
    _ = fused;

    const updated_shape = quant_y.getShape();
    try std.testing.expectEqual(@as(usize, 50), updated_shape[3]);
    try std.testing.expectEqual(@as(usize, 48), updated_shape[2]);

    const expected_stride = try TensorZant.computeStride(updated_shape);
    defer allocator.free(expected_stride);
    try std.testing.expect(std.mem.eql(usize, expected_stride, quant_y.getStride()));
}
