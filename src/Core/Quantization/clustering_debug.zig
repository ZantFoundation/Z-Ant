const std = @import("std");
const zant = @import("../../zant.zig");
const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const pkg_allocator = zant.utils.allocator.allocator;
const clustering = zant.core.clustering;

/// kMeanClustering with some debug prints
pub fn debug_kMeanClustering(comptime F: anytype, comptime U: anytype, input: *Tensor(F), output: *Tensor(U), lookup_table: *[]F) !void {

    // Calculate the number of centroids from the output type
    const numOfCentroids: usize = 1 << @bitSizeOf(U);

    // Initialize lookup_table with random centroids between min and max weights
    clustering.initCentroids(F, numOfCentroids, input, lookup_table);

    // Iterate until convergence: no weight changes its centroid assignment
    var period: usize = 0;
    var clustered = try Tensor(F).fromShape(&pkg_allocator, input.shape);
    defer clustered.deinit();
    while (!clustering.assignCentroids(F, U, numOfCentroids, input, output, lookup_table)) {
        debug_prints(F, U, numOfCentroids, input, output, lookup_table, &clustered, period);
        period += 1;
        try clustering.updateCentroidsValues(F, U, numOfCentroids, input, output, lookup_table);
    }
}

/// Some good prints (hopefully)
fn debug_prints(comptime F: anytype, comptime U: anytype, numOfCentroids: usize, originalWeights: *Tensor(F), indexes: *Tensor(U), lookup_table: *[]F, clustered: *Tensor(F), period: usize) void {
    // _ = indexes;

    std.debug.print("\n\n\n\n\nPERIOD {}", .{period});

    // std.debug.print("\nORIGINAL WEIGHTS:", .{});
    // originalWeights.printMultidim();

    // std.debug.print("\nINDEXES:", .{});
    // indexes.printMultidim();

    // std.debug.print("\nCLUSTERED WEIGHTS:", .{});
    for (0..clustered.size) |i| {
        clustered.data[i] = lookup_table.*[indexes.data[i]];
    }
    // clustered.printMultidim();

    std.debug.print("\nLOOKUP TABLE:", .{});
    for (0..numOfCentroids) |i| {
        std.debug.print("\n     {}) {d:.6}", .{ i, lookup_table.*[i] });
    }

    const linearError = meanDifference(F, originalWeights, clustered);
    std.debug.print("\nLinear scalar error: {d:.6}", .{linearError});

    const quadraticError = meanQuadraticDifference(F, originalWeights, clustered);
    std.debug.print("\nQuadratic scalar error: {d:.6}", .{quadraticError});

    const percentageLinearError = meanPercentageError(F, originalWeights, clustered);
    std.debug.print("\nLinear % error: {d:.2}%", .{percentageLinearError});

    const percentageQuadraticError = meanQuadraticPercentageError(F, originalWeights, clustered);
    std.debug.print("\nQuadratic % error: {d:.2}%", .{percentageQuadraticError});
}

/// Compute the average error element wise-between two tensors.
/// Returns abs(a[i] - b[i]) averaged on all elements.
/// Same size is assumed.
pub fn meanDifference(comptime F: type, a: *Tensor(F), b: *Tensor(F)) F {
    var sum: F = 0;
    for (0..a.size) |i| {
        sum += @abs(a.data[i] - b.data[i]);
    }
    return sum / @as(F, @floatFromInt(a.size));
}

/// Compute the average quadratic error element-wise between two tensors.
/// Returns (a[i] - b[i])^2 averaged on all elements.
/// Same size is assumed.
pub fn meanQuadraticDifference(comptime F: type, a: *Tensor(F), b: *Tensor(F)) F {
    var sum: F = 0;
    for (0..a.size) |i| {
        const diff: F = a.data[i] - b.data[i];
        sum += (diff * diff);
    }
    return sum / @as(F, @floatFromInt(a.size));
}

/// Compute the average percentage error element-wise between two tensors.
/// Returns abs((a[i] - b[i]) / a[i]) * 100 averaged on all elements.
/// Same size is assumed.
pub fn meanPercentageError(comptime F: type, a: *Tensor(F), b: *Tensor(F)) F {
    var sum: F = 0;
    for (0..a.size) |i| {
        var perc_err: F = 0;
        // If one of the two data is 0 the error is hardcoded
        if (a.data[i] == 0 or b.data[i] == 0) {
            if (a.data[i] == 0 and b.data[i] == 0) {
                perc_err = 0;
            } else {
                perc_err = 100;
            }
        } else {
            perc_err = @abs((a.data[i] - b.data[i]) / a.data[i]) * 100;
        }
        sum += perc_err;
    }
    return sum / @as(F, @floatFromInt(a.size));
}

/// Compute the average quadratic percentage error element-wise between two tensors.
/// Returns ((a[i] - b[i]) / a[i] * 100)^2 averaged on all elements.
/// Same size is assumed.
pub fn meanQuadraticPercentageError(comptime F: type, a: *Tensor(F), b: *Tensor(F)) F {
    var sum: F = 0;
    for (0..a.size) |i| {
        var perc_err: F = 0;
        // If one of the two data is 0 the error is hardcoded
        if (a.data[i] == 0 or b.data[i] == 0) {
            if (a.data[i] == 0 and b.data[i] == 0) {
                perc_err = 0;
            } else {
                perc_err = 100;
            }
        } else {
            perc_err = (a.data[i] - b.data[i]) / a.data[i] * 100;
        }
        sum += (perc_err * perc_err / 100);
    }
    return sum / @as(F, @floatFromInt(a.size));
}
