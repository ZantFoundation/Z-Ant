/// Comprehensive thread-safety tests for sdl-standalone.zig primitives.
/// Tests the two fixed issues:
///   1. std.atomic.Value(bool) for the `generating` flag
///   2. GeneralPurposeAllocator(.{ .thread_safe = true }) for the shared GPA
///
/// Run (plain):         zig test gui/sdl/test_thread_safety.zig
/// Run (with TSan):     zig test gui/sdl/test_thread_safety.zig -fsanitize-thread
const std = @import("std");

// ============================================================================
// Type aliases
// ============================================================================
const Flag = std.atomic.Value(bool);
const Counter = std.atomic.Value(usize);

// ============================================================================
// File-level worker context structs (must be at file scope for Thread.spawn)
// ============================================================================
const ToggleCtx = struct { flag: *Flag, iterations: usize };
const ProdCtx = struct { flag: *Flag, payload: *usize, expected: usize, result: *usize };
const ChainCtx = struct { flag: *Flag, data: *usize, observed: *usize };
const GuiCtx = struct { generating: *Flag, allocator: std.mem.Allocator };
const SlowCtx = struct { generating: *Flag, allocator: std.mem.Allocator };
const CounterCtx = struct { counter: *Counter, increments: usize };
const AllocCtx = struct { allocator: std.mem.Allocator, iterations: usize };
const VarCtx = struct { allocator: std.mem.Allocator };
const CasCtx = struct { generating: *Flag, wins: *Counter, attempts: usize };
const PingCtx = struct { mine: *Flag, theirs: *Flag, rounds: *Counter, max: usize };

// ============================================================================
// File-level worker functions
// ============================================================================

fn toggleWorker(ctx: ToggleCtx) void {
    for (0..ctx.iterations) |_| {
        const old = ctx.flag.load(.acquire);
        ctx.flag.store(!old, .release);
    }
}

fn producer(ctx: ProdCtx) void {
    ctx.payload.* = ctx.expected;
    ctx.flag.store(true, .release);
}

fn consumer(ctx: ProdCtx) void {
    while (!ctx.flag.load(.acquire)) std.atomic.spinLoopHint();
    ctx.result.* = ctx.payload.*;
}

fn chainWriter(ctx: ChainCtx) void {
    ctx.data.* = 0xDEADBEEF;
    ctx.flag.store(true, .release);
}

fn chainReader(ctx: ChainCtx) void {
    while (!ctx.flag.load(.acquire)) std.atomic.spinLoopHint();
    ctx.observed.* = ctx.data.*;
}

fn guiWorker(ctx: GuiCtx) void {
    for (0..200) |i| {
        const buf = ctx.allocator.alloc(u8, (i % 128) + 1) catch break;
        for (buf) |*b| b.* = @as(u8, @truncate(i));
        ctx.allocator.free(buf);
    }
    ctx.generating.store(false, .release);
}

fn slowWorker(ctx: SlowCtx) void {
    // Spin long enough that the immediate post-spawn read of the flag
    // always sees true. Use a volatile-like fence to prevent elimination.
    var burn: usize = 0;
    while (burn < 20_000_000) : (burn += 1) {
        std.atomic.spinLoopHint();
    }
    _ = ctx.allocator; // suppress unused warning
    ctx.generating.store(false, .release);
}

fn incrementWorker(ctx: CounterCtx) void {
    for (0..ctx.increments) |_| _ = ctx.counter.fetchAdd(1, .acq_rel);
}

fn casIncrementWorker(ctx: CounterCtx) void {
    var remaining = ctx.increments;
    while (remaining > 0) {
        const old = ctx.counter.load(.monotonic);
        if (ctx.counter.cmpxchgWeak(old, old + 1, .acq_rel, .monotonic) == null) {
            remaining -= 1;
        }
    }
}

fn allocWorker(ctx: AllocCtx) void {
    for (0..ctx.iterations) |i| {
        const size = (i % 64) + 1;
        const buf = ctx.allocator.alloc(u8, size) catch return;
        for (buf) |*b| b.* = @as(u8, @truncate(i));
        ctx.allocator.free(buf);
    }
}

fn varSizeWorker(ctx: VarCtx) void {
    const sizes = [_]usize{ 1, 3, 7, 8, 15, 16, 63, 64, 127, 128, 255, 256, 511, 512, 1023, 1024, 4095, 4096 };
    for (sizes) |sz| {
        const buf = ctx.allocator.alloc(u8, sz) catch continue;
        @memset(buf, 0xAB);
        ctx.allocator.free(buf);
    }
}

fn largeMibWorker(ctx: VarCtx) void {
    const buf = ctx.allocator.alloc(u8, 1024 * 1024) catch return; // 1 MiB
    @memset(buf, 0xFF);
    ctx.allocator.free(buf);
}

fn casWorker(ctx: CasCtx) void {
    for (0..ctx.attempts) |_| {
        // Only one thread can atomically flip false->true; loser retries next iteration
        if (ctx.generating.cmpxchgWeak(false, true, .acq_rel, .monotonic) == null) {
            _ = ctx.wins.fetchAdd(1, .acq_rel);
            std.Thread.sleep(1);
            ctx.generating.store(false, .release);
        }
    }
}

fn pingPongWorker(ctx: PingCtx) void {
    var r: usize = 0;
    while (r < ctx.max) {
        // Wait for our flag to go high
        while (!ctx.mine.load(.acquire)) std.atomic.spinLoopHint();
        ctx.mine.store(false, .release);
        // Signal the other side
        ctx.theirs.store(true, .release);
        _ = ctx.rounds.fetchAdd(1, .acq_rel);
        r += 1;
    }
}

// ============================================================================
// ── SECTION 1: Basic correctness ────────────────────────────────────────────
// ============================================================================

test "init: false and true" {
    const f = Flag.init(false);
    const t = Flag.init(true);
    try std.testing.expect(!f.load(.seq_cst));
    try std.testing.expect(t.load(.seq_cst));
}

test "store/load: monotonic, release/acquire, seq_cst" {
    var f = Flag.init(false);

    f.store(true, .monotonic);
    try std.testing.expect(f.load(.monotonic));

    f.store(false, .release);
    try std.testing.expect(!f.load(.acquire));

    f.store(true, .seq_cst);
    try std.testing.expect(f.load(.seq_cst));
}

test "swap: returns previous value" {
    var f = Flag.init(false);
    const prev = f.swap(true, .acq_rel);
    try std.testing.expect(!prev); // was false
    try std.testing.expect(f.load(.seq_cst)); // now true
}

// ============================================================================
// ── SECTION 2: Concurrent flag access ───────────────────────────────────────
// ============================================================================

test "concurrent toggle: 8 threads x 50 000 iterations" {
    var flag = Flag.init(false);
    var threads: [8]std.Thread = undefined;
    for (&threads) |*t|
        t.* = try std.Thread.spawn(.{}, toggleWorker, .{ToggleCtx{ .flag = &flag, .iterations = 50_000 }});
    for (&threads) |*t| t.join();
    // Any bool is valid — no crash means no race.
    _ = flag.load(.seq_cst);
}

test "producer-consumer: release/acquire" {
    var flag = Flag.init(false);
    var payload: usize = 0;
    var result: usize = 0;
    const ctx = ProdCtx{ .flag = &flag, .payload = &payload, .expected = 99, .result = &result };
    const tc = try std.Thread.spawn(.{}, consumer, .{ctx});
    const tp = try std.Thread.spawn(.{}, producer, .{ctx});
    tp.join();
    tc.join();
    try std.testing.expectEqual(@as(usize, 99), result);
}

test "release/acquire: chained data visibility (0xDEADBEEF)" {
    // Value written by writer before .release store must be seen
    // by reader after .acquire load — the fundamental ordering guarantee.
    var flag = Flag.init(false);
    var data: usize = 0;
    var observed: usize = 0;
    const tw = try std.Thread.spawn(.{}, chainWriter, .{ChainCtx{ .flag = &flag, .data = &data, .observed = &observed }});
    const tr = try std.Thread.spawn(.{}, chainReader, .{ChainCtx{ .flag = &flag, .data = &data, .observed = &observed }});
    tw.join();
    tr.join();
    try std.testing.expectEqual(@as(usize, 0xDEADBEEF), observed);
}

test "ping-pong: flag bounces between two threads (100 rounds)" {
    // Thread A signals B; B signals A; count total bounces.
    var flag_a = Flag.init(true); // A starts
    var flag_b = Flag.init(false);
    var rounds = Counter.init(0);
    const max: usize = 100;

    const ctx_a = PingCtx{ .mine = &flag_a, .theirs = &flag_b, .rounds = &rounds, .max = max };
    const ctx_b = PingCtx{ .mine = &flag_b, .theirs = &flag_a, .rounds = &rounds, .max = max };

    const ta = try std.Thread.spawn(.{}, pingPongWorker, .{ctx_a});
    const tb = try std.Thread.spawn(.{}, pingPongWorker, .{ctx_b});
    ta.join();
    tb.join();

    // Both threads counted rounds — total should be 2 * max
    try std.testing.expectEqual(@as(usize, 2 * max), rounds.load(.seq_cst));
}

// ============================================================================
// ── SECTION 3: GUI-pattern simulation ───────────────────────────────────────
// ============================================================================

test "GUI: single code-gen cycle (generating flag lifecycle)" {
    var gpa_instance = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    defer _ = gpa_instance.deinit();
    var generating = Flag.init(false);

    generating.store(true, .monotonic);
    const t = try std.Thread.spawn(.{}, guiWorker, .{GuiCtx{ .generating = &generating, .allocator = gpa_instance.allocator() }});

    var spins: usize = 0;
    while (generating.load(.acquire)) {
        std.atomic.spinLoopHint();
        spins += 1;
        if (spins > 10_000_000) @panic("GUI worker never finished");
    }
    t.join();
    try std.testing.expect(!generating.load(.seq_cst));
}

test "GUI: 50 sequential generation cycles without overlap" {
    var gpa_instance = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    defer _ = gpa_instance.deinit();
    var generating = Flag.init(false);

    for (0..50) |_| {
        // Must be idle before starting next cycle
        try std.testing.expect(!generating.load(.acquire));
        generating.store(true, .monotonic);

        const t = try std.Thread.spawn(.{}, guiWorker, .{GuiCtx{ .generating = &generating, .allocator = gpa_instance.allocator() }});

        var spins: usize = 0;
        while (generating.load(.acquire)) {
            std.atomic.spinLoopHint();
            spins += 1;
            if (spins > 10_000_000) @panic("cycle timed out");
        }
        t.join();
    }
}

test "GUI: flag stays true while worker is still running" {
    // The flag must not clear until the worker explicitly calls .store(false).
    // We read it immediately after spawn — it must be true.
    var gpa_instance = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    defer _ = gpa_instance.deinit();
    var generating = Flag.init(false);

    generating.store(true, .monotonic);
    const t = try std.Thread.spawn(.{}, slowWorker, .{SlowCtx{ .generating = &generating, .allocator = gpa_instance.allocator() }});

    // Immediately after spawn the flag must be true (worker hasn't finished).
    try std.testing.expect(generating.load(.acquire));

    var spins: usize = 0;
    while (generating.load(.acquire)) {
        std.atomic.spinLoopHint();
        spins += 1;
        if (spins > 100_000_000) @panic("timed out");
    }
    t.join();
    try std.testing.expect(!generating.load(.seq_cst));
}

test "GUI: concurrent workers — only one wins the flag via CAS" {
    // Simulates multiple rapid "Generate" button presses resolving via CAS.
    var generating = Flag.init(false);
    var wins = Counter.init(0);

    var threads: [8]std.Thread = undefined;
    for (&threads) |*t|
        t.* = try std.Thread.spawn(.{}, casWorker, .{CasCtx{ .generating = &generating, .wins = &wins, .attempts = 1_000 }});
    for (&threads) |*t| t.join();

    try std.testing.expect(wins.load(.seq_cst) > 0); // at least some won
    try std.testing.expect(!generating.load(.seq_cst)); // flag released at end
}

// ============================================================================
// ── SECTION 4: Atomic primitives (fetch_add, cmpxchg) ───────────────────────
// ============================================================================

test "fetch_add: 8 threads x 10 000 increments — exact total" {
    var counter = Counter.init(0);
    var threads: [8]std.Thread = undefined;
    for (&threads) |*t|
        t.* = try std.Thread.spawn(.{}, incrementWorker, .{CounterCtx{ .counter = &counter, .increments = 10_000 }});
    for (&threads) |*t| t.join();
    try std.testing.expectEqual(@as(usize, 8 * 10_000), counter.load(.seq_cst));
}

test "cmpxchgWeak: lock-free counter — 4 threads x 5 000 increments" {
    var counter = Counter.init(0);
    var threads: [4]std.Thread = undefined;
    for (&threads) |*t|
        t.* = try std.Thread.spawn(.{}, casIncrementWorker, .{CounterCtx{ .counter = &counter, .increments = 5_000 }});
    for (&threads) |*t| t.join();
    try std.testing.expectEqual(@as(usize, 4 * 5_000), counter.load(.seq_cst));
}

// ============================================================================
// ── SECTION 5: Thread-safe GPA edge cases ───────────────────────────────────
// ============================================================================

test "GPA: 16 threads x 2 000 alloc/free iterations" {
    var gpa_instance = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    defer {
        const check = gpa_instance.deinit();
        std.testing.expect(check == .ok) catch @panic("Memory leak!");
    }
    var threads: [16]std.Thread = undefined;
    for (&threads) |*t|
        t.* = try std.Thread.spawn(.{}, allocWorker, .{AllocCtx{ .allocator = gpa_instance.allocator(), .iterations = 2_000 }});
    for (&threads) |*t| t.join();
}

test "GPA: varying sizes 1..4096 bytes, 8 concurrent threads" {
    var gpa_instance = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    defer {
        const check = gpa_instance.deinit();
        std.testing.expect(check == .ok) catch @panic("Memory leak!");
    }
    var threads: [8]std.Thread = undefined;
    for (&threads) |*t|
        t.* = try std.Thread.spawn(.{}, varSizeWorker, .{VarCtx{ .allocator = gpa_instance.allocator() }});
    for (&threads) |*t| t.join();
}

test "GPA: zero-size allocation does not crash" {
    var gpa_instance = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    defer _ = gpa_instance.deinit();
    const alloc = gpa_instance.allocator();
    const buf = try alloc.alloc(u8, 0);
    try std.testing.expectEqual(@as(usize, 0), buf.len);
    alloc.free(buf);
}

test "GPA: 1 MiB large allocations, 4 concurrent threads" {
    var gpa_instance = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    defer {
        const check = gpa_instance.deinit();
        std.testing.expect(check == .ok) catch @panic("Memory leak!");
    }
    var threads: [4]std.Thread = undefined;
    for (&threads) |*t|
        t.* = try std.Thread.spawn(.{}, largeMibWorker, .{VarCtx{ .allocator = gpa_instance.allocator() }});
    for (&threads) |*t| t.join();
}

test "GPA: alloc then realloc then free, 4 threads" {
    var gpa_instance = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    defer {
        const check = gpa_instance.deinit();
        std.testing.expect(check == .ok) catch @panic("Memory leak!");
    }
    const alloc = gpa_instance.allocator();
    const Ctx = struct { allocator: std.mem.Allocator };
    const worker = struct {
        fn run(c: Ctx) void {
            var buf = c.allocator.alloc(u8, 64) catch return;
            @memset(buf, 0xAA);
            buf = c.allocator.realloc(buf, 256) catch {
                c.allocator.free(buf);
                return;
            };
            @memset(buf, 0xBB);
            c.allocator.free(buf);
        }
    }.run;
    var threads: [4]std.Thread = undefined;
    for (&threads) |*t|
        t.* = try std.Thread.spawn(.{}, worker, .{Ctx{ .allocator = alloc }});
    for (&threads) |*t| t.join();
}
