const std = @import("std");
const zant = @import("../../../../zant.zig");
const Uops = zant.uops;
const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const Any = Uops.Any;

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;
const error_handler = zant.utils.error_handler;
const TensorMathError = error_handler.TensorMathError;
const TensorError = error_handler.TensorError;

/// • X_id   : SSA id of the data tensor
/// • in_shape / in_stride : full shape & element-strides of X
/// • axes?  : optional list of axes to reduce (may be null or empty)
/// • keepdims, noop_with_empty_axes : attributes
/// • out_dtype : element type (same as X for ONNX ReduceMean)
///
/// Returns the SSA id of the reduced tensor.
pub fn lowerReduceMean(
    b: *UOpBuilder,
    X_id: usize,
    in_shape: []const usize, // e.g. [2,3]
    in_stride: []const isize, // e.g. [3,1]
    axes_opt: ?[]const i64, // e.g. {0}
    keepdims: bool,
    noop_with_empty: bool,
    out_dtype: DType,
) usize {
    const rank = in_shape.len;

    // ── 1. Normalize & classify axes ─────────────────────────────────────
    var is_reduced = b.alloc.alloc(bool, rank) catch unreachable;
    defer b.alloc.free(is_reduced);
    @memset(is_reduced, false);

    if (axes_opt) |ax| {
        for (ax) |a| {
            const i: usize = if (a < 0) blk: {
                const r = @as(i64, @intCast(rank));
                break :blk @as(usize, @intCast(r + a));
            } else @as(usize, @intCast(a));
            is_reduced[i] = true;
        }
    }
    const axes_empty = (axes_opt orelse &.{}).len == 0;
    const bypass = axes_empty and noop_with_empty;

    // ── 2. VIEW and allocate Y ──────────────────────────────────────────
    _ = b.push(.SHAPE, .i32, &.{X_id}, null);
    const view_id = b.push(.VIEW, out_dtype, &.{X_id}, Any{ .view_meta = .{ .shape = in_shape, .strides = in_stride } });

    var out_shape_buf = std.ArrayList(usize).init(b.alloc);
    defer out_shape_buf.deinit();
    for (in_shape, 0..) |d, i| {
        if (is_reduced[i] and !bypass) {
            if (keepdims) out_shape_buf.append(1) catch unreachable;
        } else {
            out_shape_buf.append(d) catch unreachable;
        }
    }
    const out_shape = out_shape_buf.items;

    const Y_id = b.push(.DEFINE_GLOBAL, out_dtype, &.{}, Any{ .shape = b.alloc.dupe(usize, out_shape) catch unreachable });

    if (bypass) {
        const copy = b.push(.COPY, out_dtype, &.{view_id}, null);
        _ = b.push(.STORE, out_dtype, &.{ Y_id, copy }, null);
        return Y_id;
    }

    // ── 3. inv_count and helpers ────────────────────────────────────────
    var red_count: usize = 1;
    for (in_shape, 0..) |d, i| {
        if (is_reduced[i]) red_count *= d;
    }
    const invCnt = b.push(.CONST, out_dtype, &.{}, Any{ .float = 1.0 / @as(f32, @floatFromInt(red_count)) });
    const one = b.push(.CONST, out_dtype, &.{}, Any{ .float = 1.0 });
    const zero = b.push(.CONST, .i32, &.{}, Any{ .int = 0 });

    // ── 4. Outer loops for kept dims ────────────────────────────────────
    var kept_ids = std.ArrayList(usize).init(b.alloc);
    defer kept_ids.deinit();
    for (in_shape, 0..) |d, i| {
        if (!is_reduced[i]) {
            kept_ids.append(b.push(.RANGE, .i32, &.{}, Any{ .loop_bounds = .{ .start = 0, .end = d } })) catch unreachable;
        }
    }

    // ── 5. One accumulator per output element ---------------------------
    const acc = b.push(.DEFINE_ACC, out_dtype, &.{}, null);

    // ── 6. Inner loops for reduced dims ─────────────────────────────────
    var red_ids = std.ArrayList(usize).init(b.alloc);
    defer red_ids.deinit();
    for (in_shape, 0..) |d, i| {
        if (is_reduced[i]) {
            red_ids.append(b.push(.RANGE, .i32, &.{}, Any{ .loop_bounds = .{ .start = 0, .end = d } })) catch unreachable;
        }
    }

    // ── 7. Reduction body ───────────────────────────────────────────────
    var in_src = std.ArrayList(usize).init(b.alloc);
    defer in_src.deinit();
    in_src.append(view_id) catch unreachable;
    var ki: usize = 0;
    var ri: usize = 0;
    for (in_shape, 0..) |_, i| {
        if (is_reduced[i]) {
            in_src.append(red_ids.items[ri]) catch unreachable;
            ri += 1;
        } else {
            in_src.append(kept_ids.items[ki]) catch unreachable;
            ki += 1;
        }
    }
    const pX = b.push(.GEP, out_dtype, in_src.items, Any{ .mem_info = .{ .base = view_id, .offset = 0, .stride = 1 } });
    const vX = b.push(.LOAD, out_dtype, &.{pX}, null);
    _ = b.push(.MULACC, out_dtype, &.{ acc, vX, one }, null);

    for (red_ids.items) |id| _ = b.push(.ENDRANGE, .bool, &.{id}, null);

    // ── 8. Compute mean and store ───────────────────────────────────────
    const mean = b.push(.MUL, out_dtype, &.{ acc, invCnt }, null);

    var out_src = std.ArrayList(usize).init(b.alloc);
    defer out_src.deinit();
    out_src.append(Y_id) catch unreachable;
    ki = 0;
    for (in_shape, 0..) |_, i| {
        if (is_reduced[i]) {
            if (keepdims) out_src.append(zero) catch unreachable;
        } else {
            out_src.append(kept_ids.items[ki]) catch unreachable;
            ki += 1;
        }
    }
    const pY = b.push(.GEP, out_dtype, out_src.items, Any{ .mem_info = .{ .base = Y_id, .offset = 0, .stride = 1 } });
    _ = b.push(.STORE, out_dtype, &.{ pY, mean }, null);

    // ── 9. Close outer loops ────────────────────────────────────────────
    for (kept_ids.items) |id| _ = b.push(.ENDRANGE, .bool, &.{id}, null);

    return Y_id;
}
