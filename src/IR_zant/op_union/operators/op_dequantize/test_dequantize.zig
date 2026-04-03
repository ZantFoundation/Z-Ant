const std = @import("std");
const zant = @import("zant");

const QuantMath = zant.core.tensor.quantized_math;
const Tensor = zant.core.tensor.Tensor;
const TensorType = zant.core.tensor.TensorType;
const pkgAllocator = zant.utils.allocator.allocator;

const testing = std.testing;

test "dequantization" {
    std.debug.print("\n    test: dequantization\n", .{});

    var inputArray: [2][3]u8 = [_][3]u8{
        [_]u8{
            0,
            255,
            8,
        },
        [_]u8{
            144,
            165,
            53,
        },
    };
    var shape: [2]usize = [_]usize{ 2, 3 };
    // input quantized tensor
    var inputTensor = try Tensor(u8).fromArray(&pkgAllocator, &inputArray, &shape);
    defer inputTensor.deinit();

    std.debug.print("\n --> input tensor (quantized tensor): \n", .{});
    inputTensor.printMultidim();

    // Output dequantized tensor
    var outputTensor = try QuantMath.dequantize(u8, f32, &inputTensor);
    defer outputTensor.deinit();
    std.debug.print("\n --> dequantized tensor: \n", .{});
    outputTensor.printMultidim();

    var originalArray: [2][3]f32 = [_][3]f32{
        [_]f32{ -0.5, 0.8, -0.46 },
        [_]f32{ 0.234, 0.3435, -0.231 },
    };
    var originalTensor = try Tensor(f32).fromArray(&pkgAllocator, &originalArray, &shape);
    defer originalTensor.deinit();
    std.debug.print("\n --> original tensor: \n", .{});
    originalTensor.printMultidim();
}
