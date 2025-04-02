const std = @import("std");
const zant = @import("zant");

const pkgAllocator = zant.utils.allocator.allocator;
const TensMath = zant.core.tensor.math_standard;
const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const clustering_debug = zant.core.clustering_debug;

test "tests description" {
    std.debug.print("\n--- Running clustering tests\n", .{});
}

test "k-mean clustering test" {
    const F = f32;
    const U = u8;
    const min = -2;
    const max = 2;

    // Tensor shape
    var shape: [4]usize = [_]usize{ 1, 1, 64, 64 };

    // F tensor for weights
    var inputTensor = try Tensor(F).fromShape(&pkgAllocator, &shape);
    defer inputTensor.deinit();

    // U tensor for indexes
    var outputTensor = try Tensor(U).fromShape(&pkgAllocator, &shape);
    defer outputTensor.deinit();

    var random = std.Random.DefaultPrng.init(@intCast(std.time.nanoTimestamp()));
    var rng = random.random();

    for (0..inputTensor.size) |i| {
        inputTensor.data[i] = rng.float(F) * (max - min) - (max - min) / 2;
    }

    // F array for the lookup_table
    const numCentroids = 1 << @bitSizeOf(U);
    var lookup_table = try pkgAllocator.alloc(F, numCentroids);
    defer pkgAllocator.free(lookup_table);

    // Let's cluster some ass
    try clustering_debug.debug_kMeanClustering(F, U, &inputTensor, &outputTensor, &lookup_table);

    // Print
    std.debug.print("\n\n\nFinal clustering output:\n", .{});
    outputTensor.printMultidim();
}
