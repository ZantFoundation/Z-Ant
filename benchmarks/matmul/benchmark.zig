const IR_zant = @import("IR_zant");
const bench_options = @import("bench_options");
const Tensor = IR_zant.core.tensor.Tensor;
const pkgAllocator = IR_zant.pkg_allocator;
const allocator = pkgAllocator.allocator;
const TensMath = IR_zant.core.math_standard;
const std = @import("std");

const benchmark_log = std.log.scoped(.benchmark);

const macs_mat_mul = @import("macs_matmul.zig").lean_mat_mul;
const simple_mat_mul = @import("simple_matmul.zig").lean_mat_mul;
const blocked_matmul = @import("blocked_matmul.zig").lean_mat_mul;

const mat_mul_fn = fn (comptime T: anytype, A: anytype, B: anytype, C: anytype) callconv(.@"inline") anyerror!void;

const Benchmark = struct {
    name: []const u8,
    mat_mul: mat_mul_fn,
};

const BenchmarkResult = struct {
    a_shape: [2]usize,
    b_shape: [2]usize,
    names: [test_bench.len + 1][]const u8,
    times: [test_bench.len + 1]u64,
};

const test_bench = [_]Benchmark{
    .{ .name = "CurrentMatMul", .mat_mul = TensMath.lean_matmul },
    .{ .name = "BlockedMatMul", .mat_mul = blocked_matmul },
};

const test_data_types = [_]type{
    //u8,
    u16,
    u32,
    f16,
    f32,
};

const test_data_n = [_]usize{5};

fn print_mat(comptime T: anytype, mat: *const Tensor(T)) void {
    benchmark_log.debug("Matrix shape: ", .{});
    for (mat.shape) |dim| benchmark_log.debug("{} ", .{dim});
    benchmark_log.debug("\n", .{});

    for (0..mat.shape[0]) |i| {
        for (0..mat.shape[1]) |j| {
            benchmark_log.debug("{d} ", .{mat.data[i * mat.shape[1] + j]});
        }
        benchmark_log.debug("\n", .{});
    }
}

inline fn mat_mul(comptime T: anytype, A: *const Tensor(T), B: *const Tensor(T), lean_mat_mul: mat_mul_fn) !Tensor(T) {
    // benchmark_log.debug("\nStarting matrix multiplication validation...\n", .{});

    const dim_num = A.shape.len;

    // Create output tensor

    var out_shape = try allocator.alloc(usize, dim_num);
    defer allocator.free(out_shape);
    errdefer allocator.free(out_shape);

    // Copy all dimensions except the last two
    for (0..(dim_num - 2)) |i| {
        out_shape[i] = A.shape[i];
    }

    // Set the last two dimensions to the dimensions of the input tensors
    out_shape[dim_num - 2] = A.shape[dim_num - 2];
    out_shape[dim_num - 1] = B.shape[dim_num - 1];

    // Create output tensor

    var Y = try Tensor(T).fromShape(&allocator, out_shape);
    errdefer Y.deinit();

    // benchmark_log.debug("Output tensor shape: ", .{});
    // for (Y.shape) |dim| benchmark_log.debug("{} ", .{dim});
    // benchmark_log.debug("\n", .{});

    @memset(Y.data, 0);

    try lean_mat_mul(T, A, B, &Y);

    return Y;
}

fn mat_mul_bench(comptime T: anytype, comptime N: u8, comptime tests_num: usize) ![]BenchmarkResult {
    const alloc = std.heap.page_allocator;
    var results: std.ArrayList(BenchmarkResult) = .empty;
    defer results.deinit(alloc);

    var seed: u64 = undefined;
    try std.posix.getrandom(std.mem.asBytes(&seed));
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();

    const twoN = std.math.pow(usize, 2, N);
    const data_type = T;

    const cache_block_size = std.atomic.cache_line / @sizeOf(data_type);

    for (0..tests_num) |_| {
        var shape_1: [2]usize = [_]usize{ twoN, twoN }; // 2^N x 2^N matrix
        var shape_2: [2]usize = [_]usize{ twoN, twoN }; // 2^N x 2^N matrix

        const rnd_a_rows: usize = rand.intRangeLessThan(u8, 1, cache_block_size);
        const rnd_a_cols: usize = rand.intRangeLessThan(u8, 1, cache_block_size);
        const rnd_b_cols: usize = rand.intRangeLessThan(u8, 1, cache_block_size);

        shape_1[0] += rnd_a_rows;
        shape_1[1] += rnd_a_cols;
        shape_2[0] += rnd_a_cols;
        shape_2[1] += rnd_b_cols;

        var input_data_size: usize = 1;
        for (shape_1) |dim| {
            input_data_size *= dim;
        }

        // Create input data array directly instead of ArrayList
        var inputArray1 = try allocator.alloc(data_type, input_data_size);
        defer allocator.free(inputArray1);

        // Fill with random values
        for (0..input_data_size) |i| {
            if (data_type == f32 or data_type == f64 or data_type == f16) {
                inputArray1[i] = @floatFromInt(rand.intRangeLessThan(u4, 0, 10));
            } else {
                inputArray1[i] = rand.intRangeLessThan(u4, 0, 10);
            }
        }

        input_data_size = 1;
        for (shape_2) |dim| {
            input_data_size *= dim;
        }

        var inputArray2 = try allocator.alloc(data_type, input_data_size);
        defer allocator.free(inputArray2);

        // Fill with random values
        for (0..input_data_size) |i| {
            if (data_type == f32 or data_type == f64 or data_type == f16) {
                inputArray2[i] = @floatFromInt(rand.intRangeLessThan(u4, 0, 10));
            } else {
                inputArray2[i] = rand.intRangeLessThan(u4, 0, 10);
            }
        }

        //Allocate input tensors
        var t1 = try Tensor(data_type).fromArray(&allocator, inputArray1, &shape_1);
        defer t1.deinit();
        var t2 = try Tensor(data_type).fromArray(&allocator, inputArray2, &shape_2);
        defer t2.deinit();

        //Allocate output tensors
        // Calculate time

        // Benchmark old implementation
        var timer = try std.time.Timer.start();
        const simple_start = timer.lap();
        var simple = try mat_mul(data_type, &t1, &t2, simple_mat_mul);
        defer simple.deinit();
        const simple_end = timer.lap();
        const simple_time_ns = simple_end - simple_start;

        var bench_times: [test_bench.len + 1]u64 = undefined;
        var bench_names: [test_bench.len + 1][]const u8 = undefined;

        bench_times[0] = simple_time_ns;
        bench_names[0] = "Naive";
        // Running the benchmarks
        inline for (0..test_bench.len, test_bench) |i, bench| {
            timer.reset();

            const start = timer.lap();
            var result = try mat_mul(data_type, &t1, &t2, bench.mat_mul);

            defer result.deinit();

            const end = timer.lap();
            const time_ns = end - start;

            bench_times[i + 1] = time_ns;
            bench_names[i + 1] = bench.name;

            // Check if the result is correct

            check_loop: for (0..simple.shape[0]) |x| {
                for (0..simple.shape[1]) |y| {
                    if (simple.data[x * simple.shape[1] + y] != result.data[x * simple.shape[1] + y]) {
                        std.debug.panic("Error: {s}: Element ({d}, {d}) mismatch: \n \t expected: {d} got {d}\n", .{ bench.name, x, y, simple.data[x * simple.shape[1] + y], result.data[x * simple.shape[1] + y] });

                        break :check_loop;
                    }
                }
            }
        }

        var benchmark_res: BenchmarkResult = .{ .a_shape = undefined, .b_shape = undefined, .names = undefined, .times = undefined };

        @memcpy(&benchmark_res.names, &bench_names);
        @memcpy(&benchmark_res.times, &bench_times);
        @memcpy(&benchmark_res.a_shape, &shape_1);
        @memcpy(&benchmark_res.b_shape, &shape_2);

        try results.append(alloc, benchmark_res);
    }

    return results.toOwnedSlice(alloc);
}

fn run_mat_mul_benchmarks(comptime T: anytype, comptime N: u8, comptime tests_num: usize) !void {
    const results = try mat_mul_bench(T, N, tests_num);

    benchmark_log.info("MatMul benchmark results for type {any} with base {d}:\n\n", .{ T, N });

    for (0..results.len) |i| {
        const result = results[i];
        benchmark_log.debug("Test #{d}\n", .{i});
        benchmark_log.debug("\tShapes\n", .{});
        benchmark_log.debug("\t\tA Shape: {any}\n \t\tB Shape: {any} \n", .{ result.a_shape, result.b_shape });
        benchmark_log.debug("\tTimes:\n", .{});
        for (0..result.names.len) |mat_mul_i| {
            benchmark_log.debug("\t\t{s} took {d} ms\n", .{ result.names[mat_mul_i], @as(f64, @floatFromInt(result.times[mat_mul_i])) / 1_000_000.0 });
        }

        benchmark_log.debug("\tSpeedups:\n", .{});
        for (2..result.names.len) |mat_mul_i| {
            const current_speedup = @as(f64, @floatFromInt(result.times[1])) / @as(f64, @floatFromInt(result.times[mat_mul_i]));
            benchmark_log.debug("\t\t{s}: {d:.2}x\n", .{ result.names[mat_mul_i], current_speedup });
        }
    }
}

pub fn run() !void {
    if (bench_options.full) {
        inline for (0..test_data_types.len) |i| {
            const T = test_data_types[i];
            inline for (0..test_data_n.len) |j| {
                const N = test_data_n[j];
                try run_mat_mul_benchmarks(T, N, 5);
                benchmark_log.debug("\n\n", .{});
            }
        }
    } else {
        const N = 10; // Change this value to test different sizes
        const data_type = u8;
        try run_mat_mul_benchmarks(data_type, N, 5);
    }
}
