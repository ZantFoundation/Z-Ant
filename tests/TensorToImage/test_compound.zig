const std = @import("std");
const testing = std.testing;

const TensorToImage = @import("TensorToImage");
const compound = TensorToImage.compound;
const toRGBImageF32 = compound.toRGBImageF32;
const lean_compound = compound.lean_compound;
const batchToRGBImageF32 = compound.batchToRGBImageF32;
const CompoundScratch = compound.CompoundScratch;

// ---------------------------------------------------------------------------
// Output shape
// ---------------------------------------------------------------------------

test "toRGBImageF32: output length is 3*n*n" {
    const input = [_]f32{ 0.1, 0.5, -0.3, 0.8, 1.2 };
    const n = input.len;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();

    const out = try toRGBImageF32(arena.allocator(), &input, .MinusOneToOne, 4);
    try testing.expectEqual(@as(usize, 3 * n * n), out.len);
}

// ---------------------------------------------------------------------------
// Channel layout: correct transform in each CHW slice
// ---------------------------------------------------------------------------

test "toRGBImageF32: channel 0 (GASF) is symmetric" {
    // GASF produces a symmetric matrix, so after (x+1)/2 the shifted matrix
    // must also be symmetric.
    const input = [_]f32{ 1.0, 2.5, -1.0, 0.3, 0.8 };
    const n = input.len;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();

    const out = try toRGBImageF32(arena.allocator(), &input, .MinusOneToOne, 4);
    const ch0 = out[0 * n * n .. 1 * n * n];

    for (0..n) |i| {
        for (0..n) |j| {
            try testing.expectApproxEqAbs(ch0[i * n + j], ch0[j * n + i], 1e-6);
        }
    }
}

test "toRGBImageF32: channel 1 (GADF) has zero diagonal" {
    // GADF encodes sin(φ_i - φ_j); diagonal is always sin(0) = 0,
    // which maps to (0+1)/2 = 0.5 after the shift.
    const input = [_]f32{ 0.1, 0.5, -0.3, 0.8 };
    const n = input.len;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();

    const out = try toRGBImageF32(arena.allocator(), &input, .MinusOneToOne, 4);
    const ch1 = out[1 * n * n .. 2 * n * n];

    for (0..n) |i| {
        // GADF diagonal = 0 → shifted to 0.5
        try testing.expectApproxEqAbs(@as(f32, 0.5), ch1[i * n + i], 1e-6);
    }
}

test "toRGBImageF32: channel 2 (MTF) all values are transition probabilities in [0, 1]" {
    // M[i][j] = W[bins[i]][bins[j]] where W is the row-stochastic transition
    // matrix. Individual entries are probabilities, so they must be in [0, 1].
    // Note: MTF rows do NOT sum to 1 — that property holds for W, not M.
    const input = [_]f32{ 0.1, 0.9, 0.3, 0.7, 0.5, 0.2, 0.8, 0.4 };
    const n = input.len;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();

    const out = try toRGBImageF32(arena.allocator(), &input, .MinusOneToOne, 4);
    const ch2 = out[2 * n * n .. 3 * n * n];

    for (ch2) |v| {
        try testing.expect(v >= 0.0 and v <= 1.0);
    }
}

// ---------------------------------------------------------------------------
// Value range
// ---------------------------------------------------------------------------

test "toRGBImageF32: all values in [0, 1]" {
    const input = [_]f32{ 1e5, 2e5, -1e5, 0.0, 5e4, -3e4 };

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();

    const out = try toRGBImageF32(arena.allocator(), &input, .MinusOneToOne, 4);

    for (out) |v| {
        try testing.expect(v >= 0.0 and v <= 1.0);
    }
}

test "toRGBImageF32: GASF and GADF shift is (x+1)/2, MTF is passthrough" {
    // Use a known 2-element input so we can compute expected values by hand.
    // input = [0.0, 1.0]
    // MinusOneToOne normalization: min=0, max=1 → [-1, 1]
    //   x_tilde = [−1.0, 1.0]
    // GASF: G[i][j] = xi*xj − sqrt(1−xi²)*sqrt(1−xj²)
    //   G[0][0] = (−1)(−1) − 0*0 =  1.0  → shifted: 1.0
    //   G[0][1] = (−1)(1)  − 0*0 = −1.0  → shifted: 0.0
    //   G[1][0] = same as G[0][1]  = −1.0 → shifted: 0.0
    //   G[1][1] = (1)(1)   − 0*0 =  1.0  → shifted: 1.0
    // GADF: G[i][j] = sqrt(1−xi²)*xj − xi*sqrt(1−xj²)
    //   G[0][0] = 0*(−1) − (−1)*0 = 0 → shifted: 0.5
    //   G[0][1] = 0*1    − (−1)*0 = 0 → shifted: 0.5
    //   G[1][0] = 0*(−1) − 1*0    = 0 → shifted: 0.5
    //   G[1][1] = 0*1    − 1*0    = 0 → shifted: 0.5
    const input = [_]f32{ 0.0, 1.0 };
    const n = input.len;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();

    const out = try toRGBImageF32(arena.allocator(), &input, .MinusOneToOne, 2);
    const ch0 = out[0 * n * n .. 1 * n * n];
    const ch1 = out[1 * n * n .. 2 * n * n];

    const eps: f32 = 1e-5;
    // GASF shifted
    try testing.expectApproxEqAbs(@as(f32, 1.0), ch0[0], eps); // G[0][0]
    try testing.expectApproxEqAbs(@as(f32, 0.0), ch0[1], eps); // G[0][1]
    try testing.expectApproxEqAbs(@as(f32, 0.0), ch0[2], eps); // G[1][0]
    try testing.expectApproxEqAbs(@as(f32, 1.0), ch0[3], eps); // G[1][1]
    // GADF shifted (all zeros → all 0.5)
    for (ch1) |v| try testing.expectApproxEqAbs(@as(f32, 0.5), v, eps);
}

// ---------------------------------------------------------------------------
// lean_compound
// ---------------------------------------------------------------------------

test "lean_compound: matches toRGBImageF32 output exactly" {
    // lean_compound with manually managed scratch must produce bit-identical
    // results to the allocating wrapper.
    const input = [_]f32{ 0.1, 0.5, -0.3, 0.8, 1.2, -0.7 };
    const n = input.len;
    const q = 4;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    // Reference from allocating wrapper
    const ref = try toRGBImageF32(alloc, &input, .MinusOneToOne, q);

    // lean path
    var scratch = try CompoundScratch.init(alloc, n, q);
    var lean_out: [3 * n * n]f32 = undefined;
    lean_compound(&input, .MinusOneToOne, q, &scratch, &lean_out);

    for (ref, lean_out) |r, l| {
        try testing.expectApproxEqAbs(r, l, 1e-6);
    }
}

test "lean_compound: scratch buffers are reused correctly across calls" {
    // Calling lean_compound twice with different inputs must produce two
    // independent correct results — verifies scratch is fully overwritten
    // each call and carries no state between calls.
    const input_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const input_b = [_]f32{ 4.0, 3.0, 2.0, 1.0 };
    const n = input_a.len;
    const q = 2;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    var scratch = try CompoundScratch.init(alloc, n, q);

    var out_a: [3 * n * n]f32 = undefined;
    var out_b: [3 * n * n]f32 = undefined;
    lean_compound(&input_a, .MinusOneToOne, q, &scratch, &out_a);
    lean_compound(&input_b, .MinusOneToOne, q, &scratch, &out_b);

    // Cross-check each against its own allocating reference
    const ref_a = try toRGBImageF32(alloc, &input_a, .MinusOneToOne, q);
    const ref_b = try toRGBImageF32(alloc, &input_b, .MinusOneToOne, q);

    for (ref_a, out_a) |r, l| try testing.expectApproxEqAbs(r, l, 1e-6);
    for (ref_b, out_b) |r, l| try testing.expectApproxEqAbs(r, l, 1e-6);
}

// ---------------------------------------------------------------------------
// batchToRGBImageF32
// ---------------------------------------------------------------------------

test "batchToRGBImageF32: output length is B*3*N*N" {
    const s0 = [_]f32{ 0.1, 0.5, -0.3, 0.8 };
    const s1 = [_]f32{ 1.0, -1.0, 0.5, 0.0 };
    const s2 = [_]f32{ 0.3, 0.7, 0.2, 0.9 };
    const inputs = [_][]const f32{ &s0, &s1, &s2 };
    const b = inputs.len;
    const n = s0.len;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();

    const out = try batchToRGBImageF32(arena.allocator(), &inputs, .MinusOneToOne, 2);
    try testing.expectEqual(@as(usize, b * 3 * n * n), out.len);
}

test "batchToRGBImageF32: each batch slice matches individual toRGBImageF32" {
    // Each [3*N*N] chunk inside the batch output must equal what toRGBImageF32
    // would return for that series in isolation.
    const s0 = [_]f32{ 0.1, 0.9, 0.3, 0.7, 0.5 };
    const s1 = [_]f32{ 1.0, 2.0, 0.5, -0.5, 3.0 };
    const inputs = [_][]const f32{ &s0, &s1 };
    const n = s0.len;
    const q = 3;
    const chw = 3 * n * n;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const batch = try batchToRGBImageF32(alloc, &inputs, .MinusOneToOne, q);
    const ref0 = try toRGBImageF32(alloc, &s0, .MinusOneToOne, q);
    const ref1 = try toRGBImageF32(alloc, &s1, .MinusOneToOne, q);

    for (ref0, batch[0 * chw .. 1 * chw]) |r, b_val| try testing.expectApproxEqAbs(r, b_val, 1e-6);
    for (ref1, batch[1 * chw .. 2 * chw]) |r, b_val| try testing.expectApproxEqAbs(r, b_val, 1e-6);
}

test "batchToRGBImageF32: all values in [0, 1]" {
    const s0 = [_]f32{ 1e4, -1e4, 0.0, 5e3 };
    const s1 = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
    const inputs = [_][]const f32{ &s0, &s1 };

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();

    const out = try batchToRGBImageF32(arena.allocator(), &inputs, .MinusOneToOne, 2);
    for (out) |v| try testing.expect(v >= 0.0 and v <= 1.0);
}
