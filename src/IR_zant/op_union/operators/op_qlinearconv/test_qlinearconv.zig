const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const QuantTensMath = zant.core.tensor.quantized_math;
const Tensor = zant.core.tensor.Tensor;
const tensorType = zant.core.tensor.TensorType;
const tensorDetails = zant.core.tensor.TensorDetails;
const quantDetails = zant.core.tensor.QuantDetails;
const TensorMathError = zant.utils.error_handler.TensorMathError;

test "Convolution 4D QUANTIZED Input with 2x2x2x2 Kernel shape" {
    std.debug.print("\n     test: Convolution 4D QUANTIZED Input with 2x2x2x2 Kernel shape\n", .{});

    const allocator = pkgAllocator.allocator;

    // Input tensor
    var input_shape: [4]usize = [_]usize{ 2, 2, 3, 3 };
    var inputArray: [2][2][3][3]i8 = [_][2][3][3]i8{ //batches:2, channels:2, rows:3, cols:3
        //First Batch
        [_][3][3]i8{
            // First Channel
            [_][3]i8{
                [_]i8{ 2, 2, 3 },
                [_]i8{ 4, 5, 6 },
                [_]i8{ 7, 8, 9 },
            },
            // Second Channel
            [_][3]i8{
                [_]i8{ 8, 8, 7 },
                [_]i8{ 6, 5, 4 },
                [_]i8{ 3, 2, 1 },
            },
        },
        // Second batch
        [_][3][3]i8{
            // First channel
            [_][3]i8{
                [_]i8{ 2, 3, 4 },
                [_]i8{ 5, 6, 7 },
                [_]i8{ 8, 9, 10 },
            },
            // Second channel
            [_][3]i8{
                [_]i8{ 10, 9, 8 },
                [_]i8{ 7, 6, 5 },
                [_]i8{ 4, 3, 2 },
            },
        },
    };

    // Kernel tensor
    var kernel_shape: [4]usize = [_]usize{ 2, 2, 2, 2 };
    var kernelArray: [2][2][2][2]i8 = [_][2][2][2]i8{ //filters:2, channels:2, rows:2, cols:2
        //first filter
        [_][2][2]i8{
            //first channel
            [_][2]i8{
                [_]i8{ -1, 0 },
                [_]i8{ 0, 1 },
            },
            //second channel
            [_][2]i8{
                [_]i8{ 1, -1 },
                [_]i8{ -1, 1 },
            },
        },
        //second filter
        [_][2][2]i8{
            //first channel
            [_][2]i8{
                [_]i8{ 0, 0 },
                [_]i8{ 0, 0 },
            },
            //second channel
            [_][2]i8{
                [_]i8{ 0, 0 },
                [_]i8{ 0, 0 },
            },
        },
    };
    var kernel_tensor = try Tensor(i8).fromArray(&allocator, &kernelArray, &kernel_shape);
    defer kernel_tensor.deinit();
    kernel_tensor.details = tensorDetails{
        .quant = quantDetails{
            .tensorType = tensorType.QuantTensor,
            .scale_factor = 0.005,
            .zero_point = -2,
        },
    };

    // Bias tensor
    var inputbias: [2]i8 = [_]i8{ 1, 1 }; //batches: 2, filters:2
    var bias_shape: [1]usize = [_]usize{2};
    var bias = try Tensor(i8).fromArray(&allocator, &inputbias, &bias_shape);
    defer bias.deinit();

    // Input tensor
    var input_tensor = try Tensor(i8).fromArray(&allocator, &inputArray, &input_shape);
    defer input_tensor.deinit();
    input_tensor.details = tensorDetails{
        .quant = quantDetails{
            .tensorType = tensorType.QuantTensor,
            .scale_factor = 0.005,
            .zero_point = -2,
        },
    };

    const stride: [2]usize = [_]usize{ 1, 1 };

    var result_tensor = try QuantTensMath.convolve_tensor_with_bias(i8, i8, &input_tensor, &kernel_tensor, &bias, &stride, null, 1);
    defer result_tensor.deinit();

    // Expected results with the correct dimensions
    const expected_result: [2][2][2][2]i8 = [_][2][2][2]i8{
        // Primo batch
        [_][2][2]i8{
            [_][2]i8{
                [_]i8{ 3, 5 },
                [_]i8{ 5, 5 },
            },
            [_][2]i8{
                [_]i8{ 1, 1 },
                [_]i8{ 1, 1 },
            },
        },
        // Secondo batch
        [_][2][2]i8{
            [_][2]i8{
                [_]i8{ 5, 5 },
                [_]i8{ 5, 5 },
            },
            [_][2]i8{
                [_]i8{ 1, 1 },
                [_]i8{ 1, 1 },
            },
        },
    };

    // result_tensor.info();
    // result_tensor.print();

    const output_location = try allocator.alloc(usize, 4); //coordinates in the output space, see test below
    defer allocator.free(output_location);
    @memset(output_location, 0);

    for (0..2) |batch| {
        output_location[0] = batch;
        for (0..2) |filter| {
            output_location[1] = filter;
            for (0..2) |row| {
                output_location[2] = row;
                for (0..2) |col| {
                    output_location[3] = col;
                    //std.debug.print("\n get OUTPUT at:{any}", .{output_location});
                    try std.testing.expectEqual(expected_result[batch][filter][row][col], result_tensor.get_at(output_location));
                }
            }
        }
    }
}
