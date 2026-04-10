const std = @import("std");
const testing = std.testing;

const zant = @import("zant");
const gadf = zant.TensorToImage.gadf.gadf;
const lean_gadf = zant.TensorToImage.gadf.lean_gadf;
const utils = zant.TensorToImage.gaf_utils;

const NormRange = utils.NormRange;

// ---------------------------------------------------------------------------
// pyts cross-validation (reads gadf_test_vectors.json at runtime)
// ---------------------------------------------------------------------------

// Reads gadf_test_vectors.json (produced by pyts_gadf_reference.py), runs
// each series through our gadf() and compares cell-by-cell against pyts.
//
// The json file is already generated and present in the tests/TensorToImage/gadf/ directory.
// To regenerate the file (from project root):
//   python tests/TensorToImage/gadf/pyts_gadf_reference.py
test "GADF pyts cross-validation" {
    const allocator = std.testing.allocator;
    const json_path = "tests/TensorToImage/gadf/gadf_test_vectors.json";

    const file = std.fs.cwd().openFile(json_path, .{}) catch |err| {
        if (err == error.FileNotFound) {
            std.debug.print(
                "\n[SKIP] {s} not found.\n" ++
                "       Run: python tests/TensorToImage/gadf/pyts_gadf_reference.py\n",
                .{json_path},
            );
            return;
        }
        return err;
    };
    defer file.close();

    const stat = try file.stat();
    const content = try allocator.alloc(u8, stat.size);
    defer allocator.free(content);
    _ = try file.readAll(content);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();

    const vectors = parsed.value.array.items;
    //std.debug.print("\n[pyts cross-validation] {d} test vectors loaded\n", .{vectors.len});

    for (vectors) |vec| {
        const label = vec.object.get("label").?.string;
        const input_json = vec.object.get("input").?.array.items;
        const expected_json = vec.object.get("expected").?.array.items;

        // Build f32 input slice
        const input = try allocator.alloc(f32, input_json.len);
        defer allocator.free(input);
        for (input_json, input) |v, *o| o.* = jsonToF32(v);

        // Build f32 expected slice
        const expected = try allocator.alloc(f32, expected_json.len);
        defer allocator.free(expected);
        for (expected_json, expected) |v, *o| o.* = jsonToF32(v);

        // Run our implementation
        const result = try gadf(allocator, input, .MinusOneToOne);
        defer allocator.free(result);

        try testing.expectEqual(expected.len, result.len);

        const epsilon: f32 = 1e-4;
        var first_fail = true;
        var ok = true;
        for (expected, result, 0..) |exp, res, cell| {
            if (!std.math.approxEqAbs(f32, exp, res, epsilon)) {
                if (first_fail) {
                    std.debug.print("\n[FAIL] vector '{s}'\n", .{label});
                    first_fail = false;
                }
                std.debug.print(
                    "  cell {d}: pyts={d:.6}  zig={d:.6}  diff={e:.2}\n",
                    .{ cell, exp, res, @abs(exp - res) },
                );
                ok = false;
            }
        }
        try testing.expect(ok);
    }
}

/// Converts a std.json.Value number to f32 (handles both integer and float JSON tokens).
fn jsonToF32(v: std.json.Value) f32 {
    return switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        .number_string => |s| std.fmt.parseFloat(f32, s) catch unreachable,
        else => unreachable,
    };
}

// ---------------------------------------------------------------------------
// Structural / mathematical properties
// ---------------------------------------------------------------------------

test "GADF anti-symmetry: G[i][j] == -G[j][i]" {
    // GADF encodes sin(φ_i - φ_j). Since sin is odd:
    //   sin(φ_i - φ_j) = -sin(φ_j - φ_i)  =>  G[i][j] = -G[j][i]
    // This is the defining difference from GASF (which is symmetric).
    const input = [_]f32{ 1.0, 2.5, -1.0, 0.3, 0.8 };
    const n = input.len;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const result = try gadf(arena.allocator(), &input, .MinusOneToOne);

    const epsilon: f32 = 1e-6;
    for (0..n) |i| {
        for (0..n) |j| {
            const gij = result[i * n + j];
            const gji = result[j * n + i];
            if (!std.math.approxEqAbs(f32, gij, -gji, epsilon)) {
                std.debug.print("\nAnti-symmetry violation at [{d}][{d}]: G={d:.6}, -G^T={d:.6}\n", .{ i, j, gij, -gji });
            }
            try testing.expect(std.math.approxEqAbs(f32, gij, -gji, epsilon));
        }
    }
}

test "GADF zero diagonal: G[i][i] == 0 for all i" {
    // sin(φ_i - φ_i) = sin(0) = 0, so the main diagonal is always zero.
    const input = [_]f32{ 0.1, 0.5, -0.3, 0.8, 1.2, -2.0 };
    const n = input.len;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const result = try gadf(arena.allocator(), &input, .MinusOneToOne);

    const epsilon: f32 = 1e-6;
    for (0..n) |i| {
        try testing.expect(std.math.approxEqAbs(f32, result[i * n + i], 0.0, epsilon));
    }
}

test "GADF output bounds: all values in [-1.0, 1.0]" {
    // sin is bounded to [-1, 1]; clamping in normalize prevents sqrt of negatives.
    const input = [_]f32{ 1e5, 2e5, -1e5, 0.0, 5e4, -3e4 };

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const result = try gadf(arena.allocator(), &input, .MinusOneToOne);

    for (result) |val| {
        try testing.expect(val >= -1.0001 and val <= 1.0001);
    }
}

test "GADF ZeroToOne normalization: output still in [-1.0, 1.0]" {
    // With [0,1] normalization the angular range is [0, π/2] instead of [0, π],
    // so differences are in [−π/2, π/2] and sin is still in [-1, 1].
    const input = [_]f32{ 0.3, 1.2, 0.7, 2.5, 0.1 };

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const result = try gadf(arena.allocator(), &input, .ZeroToOne);

    for (result) |val| {
        try testing.expect(val >= -1.0001 and val <= 1.0001);
    }
}

test "GADF flat input: all values are zero" {
    // When all inputs are equal, normalize sets every x_tilde = 0.0.
    // => sin[i] = sqrt(1 - 0²) = 1.0 for all i
    // => G[i][j] = 1.0 * 0.0 - 0.0 * 1.0 = 0.0
    //
    // This is mathematically correct: a constant series has no angular difference.
    // Note: a zero GADF matrix is also produced by any series whose pairwise
    // angular differences cancel symmetrically — the flat case is not uniquely
    // identifiable from the matrix alone.
    const input = [_]f32{ 5.0, 5.0, 5.0, 5.0 };

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    const result = try gadf(arena.allocator(), &input, .MinusOneToOne);

    for (result) |val| {
        try testing.expectEqual(@as(f32, 0.0), val);
    }
}

// ---------------------------------------------------------------------------
// lean_gadf
// ---------------------------------------------------------------------------

test "lean_gadf: matches standard wrapper output" {
    // Verifies the zero-allocation core with explicit sines buffer produces
    // the same result as going through the full gadf() allocation path.
    const input = [_]f32{ 0.1, 0.5, -0.3, 0.8 };
    const n = input.len;

    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    // Standard path (normalizes internally)
    const standard_result = try gadf(arena.allocator(), &input, .MinusOneToOne);

    // lean path (already-normalized input; use the same normalized values)
    const normalized = [_]f32{ -0.272727, 0.454545, -1.0, 1.0 };
    var sines_buf: [n]f32 = undefined;
    var lean_out: [n * n]f32 = undefined;
    lean_gadf(&normalized, &sines_buf, &lean_out);

    const epsilon: f32 = 1e-4;
    for (standard_result, lean_out) |std_val, lean_val| {
        try testing.expect(std.math.approxEqAbs(f32, std_val, lean_val, epsilon));
    }
}

test "lean_gadf: formula correctness cell-by-cell" {
    // Spot-checks every cell against the direct algebraic definition
    // G[i][j] = sqrt(1-xi²)*xj - xi*sqrt(1-xj²)
    const normalized = [_]f32{ -1.0, -0.5, 0.0, 0.5, 1.0 };
    const n = normalized.len;

    var sines_buf: [n]f32 = undefined;
    var output: [n * n]f32 = undefined;
    lean_gadf(&normalized, &sines_buf, &output);

    const epsilon: f32 = 1e-5;
    for (0..n) |i| {
        for (0..n) |j| {
            const xi = normalized[i];
            const xj = normalized[j];
            const si = @sqrt(@max(0.0, 1.0 - xi * xi));
            const sj = @sqrt(@max(0.0, 1.0 - xj * xj));
            const expected = si * xj - xi * sj;
            try testing.expect(std.math.approxEqAbs(f32, expected, output[i * n + j], epsilon));
        }
    }
}
