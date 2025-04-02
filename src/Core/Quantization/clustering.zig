const std = @import("std");
const zant = @import("../../zant.zig");
const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;
const pkg_allocator = zant.utils.allocator.allocator;

// Functions in this file:
// - initCentroids
// - updateAssignments
// - updateCentroidsValues
// - kMeanClustering

/// Initializes the centroid lookup table with random values between the minimum and maximum of the input weights.
pub fn initCentroids(comptime F: anytype, numOfCentroids: usize, weights: *Tensor(F), centroids: *[]F) void {

    // Get min and max weights
    var min: F = weights.data[0];
    var max: F = weights.data[0];
    for (weights.data) |weight| {
        if (weight < min)
            min = weight;
        if (weight > max)
            max = weight;
    }

    // Init seed for random number
    var random = std.Random.DefaultPrng.init(@intCast(std.time.nanoTimestamp()));
    var rng = random.random();

    // Assign a random value in [min, max] to each centroid
    for (0..numOfCentroids) |i| {
        const randFraction = rng.float(F);
        centroids.*[i] = min + ((max - min) * randFraction);
    }
}

/// Updates the assignment of each weight to the nearest centroid.
/// Writes the index of the chosen centroid to the weights tensor.
/// Returns true if the centroids converged, false if not.
pub fn assignCentroids(comptime F: anytype, comptime U: anytype, numOfCentroids: usize, weights: *Tensor(F), indexes: *Tensor(U), centroids: *[]F) bool {
    var converged = true;

    // Iterate over each weight
    for (0..weights.size) |i| {
        var best_index: U = 0;
        var best_distance: F = @abs(weights.data[i] - centroids.*[0]);

        // Iterate over each centroid
        for (0..numOfCentroids) |j| {
            const distance = @abs(weights.data[i] - centroids.*[j]);
            if (distance < best_distance) {
                best_distance = distance;
                best_index = @intCast(j);
            }
        }

        // Update index of nearest centroid if it changed
        if (best_index != indexes.data[i]) {
            indexes.data[i] = best_index;
            converged = false;
        }
    }

    return converged;
}

/// Recalculates the centroid positions as the mean of all input weights assigned to each centroid.
pub fn updateCentroidsValues(comptime F: anytype, comptime U: anytype, numOfCentroids: usize, weights: *Tensor(F), indexes: *Tensor(U), centroids: *[]F) !void {

    // Allocate accumulators and counters for each centroid
    var sums = try pkg_allocator.alloc(F, numOfCentroids);
    var counts = try pkg_allocator.alloc(usize, numOfCentroids);
    defer pkg_allocator.free(sums);
    defer pkg_allocator.free(counts);
    for (0..numOfCentroids) |i| {
        sums[i] = 0;
        counts[i] = 0;
    }

    // Sum weights and count assignments
    for (0..indexes.size) |i| {
        sums[indexes.data[i]] += weights.data[i];
        counts[indexes.data[i]] += 1;
    }

    // Update each centroid to the mean of assigned weights
    for (0..numOfCentroids) |i| {
        if (counts[i] > 0) {
            centroids.*[i] = sums[i] / @as(F, @floatFromInt(counts[i]));
        }
        // If no weights were assigned, leave the centroid unchanged
    }
}

/// Main function that applies k-means clustering on the weights contained in an input tensor.
/// - T: The type of the weights (e.g. f32).
/// - U: The integer type used to store the index of the centroid (u16, u8, etc.).
/// - input: Tensor containing the weights to be quantized.
/// - output: Tensor where each element is the index of the nearest centroid for the corresponding input weight.
/// - lookup_table: An array that will hold the centroid values (must be allocated by the caller).
pub fn kMeanClustering(comptime F: anytype, comptime U: anytype, input: *Tensor(F), output: *Tensor(U), lookup_table: *[]F) !void {

    // Calculate the number of centroids from the output type.
    const numOfCentroids = 1 << @bitSizeOf(U);

    // Initialize lookup_table with random centroids between min and max weights.
    initCentroids(F, numOfCentroids, input, lookup_table);

    // Iterate until convergence: no weight changes its centroid assignment.
    while (!assignCentroids(F, U, numOfCentroids, input, output, lookup_table)) {
        try updateCentroidsValues(F, U, numOfCentroids, input, output, lookup_table);
    }
}
