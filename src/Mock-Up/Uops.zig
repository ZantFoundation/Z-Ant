//! ============================================================================
//!  Zant IR – *single–assignment* micro–operation layer
//! ============================================================================
//!  This file is meant to live in `src/ir.zig`.
//!  It defines **everything** a back-end needs to know about one UOp:
//!    • the tag (`UOpType`)                     – *what* the op does
//!    • the element type (`DType`)              – *with which* scalar type
//!    • its operands (`src` as indices)         – *who* produces the inputs
//!    • an optional payload (`Any`)             – *extra metadata*
//!
//!
//!  ─ SSA (Static Single Assignment) ───────────────────────────────────────
//!  •  Each `UOp` appears **once** in the slice and never mutates.
//!  •  Later ops reference earlier ones by *index* (the `src` array).
//!  •  Guarantees there is exactly **one definition** per temporary, which
//!     simplifies constant propagation, alias analysis, and code-gen.
//!
//! ============================================================================

const std = @import("std");

// ─────────────────────────────────────────────────────────────────────────────
// 1. UOpType – every primitive micro-op in Zant
//    (Comments give a one-line intuitive meaning.)
// ─────────────────────────────────────────────────────────────────────────────
pub const UOpType = enum {
    // Data movement / buffer mgmt
    DEFINE_GLOBAL, // allocate a top-level tensor/buffer (inputs, outputs)
    DEFINE_ACC, // allocate and zero a reduction accumulator
    LOAD, // read one element from memory
    STORE, // write one element to memory
    CONST, // scalar literal (f32/i32/…)

    // Pure arithmetic & logical ops (element-wise)
    ADD,
    SUB,
    MUL,
    FDIV,
    POW,
    EXP2,
    NEG,
    MAX,
    MIN,
    CLIP, // CLIP clamps to [min,max] (limits in Any)
    CMPLT, // compare <  (returns bool)
    WHERE, // ternary select (cond ? a : b)
    MULACC, // fused multiply-add into an accumulator x += y*z

    // Reductions
    REDUCE_ADD,
    REDUCE_MAX,

    // Loop / control
    RANGE, // begin counted loop   (bounds in Any.loop_bounds)
    ENDRANGE, // end   counted loop
    IF, //we already have where keep it or not ?
    ENDIF,

    // Addressing & view manipulation (no data copies)
    GEP, // Get element pointer GEP calculates the actual position inside a flat memory buffer,
    //taking into account the strides of each dimension — including broadcasting.
    VIEW, // Create a view of a tensor
    COPY, // Copy a tensor
    RESHAPE, // Reshape a tensor
    PAD, // Pad a tensor
    PERMUTE, // Permute a tensor
    EXPAND, // Expand a tensor

    // Shape & bookkeeping
    SHAPE, // Get the shape of a tensor
    CAST, // Cast a tensor to a different type

    // Scheduling hints (inserted by auto-tuner)
    TILE_M, // Tile the tensor in the M dimension
    TILE_N, // Tile the tensor in the N dimension
    VECTORIZE, // Vectorize the tensor
    UNROLL_K, // Unroll the tensor in the K dimension

    // Graph–level utilities
    FUSE, // marks a fused element-wise chain (payload = ops list)

};

// ─────────────────────────────────────────────────────────────────────────────
// 2. Any – single-slot, type-safe payload attached to `UOp.arg`
// ─────────────────────────────────────────────────────────────────────────────
pub const Any = union(enum) {
    // 2.1 Scalar immediates
    int: usize,
    float: f32,
    bool: bool,

    // 2.2 Tiny metadata blobs
    label: []const u8, // buffer/debug name
    shape: []const usize, // runtime shape vector

    // 2.3 Control-flow helpers
    loop_bounds: struct { start: usize, end: usize },

    // 2.4 Addressing info
    mem_info: struct { base: []const u8, offset: usize, stride: usize },

    // ── 2 · 6 Arithmetic Operations ────────────────────────────────────
    /// Used for arithmetic ops
    /// • `a` is the first operand
    /// • `b` is the second operand (optional)
    /// • `out` is the result of the operation
    arithmetic_op_info: struct { // • Used by arithmetic ops
        out: f32,
        a: f32,
        b: ?f32,
    },
};

// ─────────────────────────────────────────────────────────────────────────────
// 3. DType – minimalist scalar element types
// ─────────────────────────────────────────────────────────────────────────────
pub const DType = enum { f32, i32, i8, bool };

// ─────────────────────────────────────────────────────────────────────────────
// 4. UOp – ONE single-assignment micro-op stored in a linear slice
// ─────────────────────────────────────────────────────────────────────────────
pub const UOp = struct {
    id: usize, // equals position in program slice (redundant but handy)
    op: UOpType, // opcode tag
    dtype: DType, // element type of the result
    src: []const usize, // producer IDs; each ID < id  (topologically sorted)
    arg: ?Any = null, // optional payload (constants, bounds, …)

    /// Pretty-print for REPL / unit tests
    pub fn dump(self: UOp, w: anytype) !void {
        try w.print("{d:>3}  {s}", .{ self.id, @tagName(self.op) });
        if (self.src.len > 0) try w.print("  src={}", .{self.src});
        if (self.arg) |a| try w.print("  arg={any}", .{a});
        try w.print("\n", .{});
    }
};

pub const UOpBuilder = struct {
    alloc: std.mem.Allocator,
    list: std.ArrayList(UOp),

    pub fn init(a: std.mem.Allocator) UOpBuilder {
        return .{ .alloc = a, .list = .init(a) };
    }

    /// Push that dupes `src` safely.
    pub fn push(self: *UOpBuilder, op: UOpType, dt: DType, src: []const usize, arg: ?Any) usize {
        const id = self.list.len;
        const copy = if (src.len == 0)
            &[_]usize{} // empty slice → static, no alloc
        else
            self.alloc.dupe(usize, src) catch unreachable;

        self.list.append(.{ .id = id, .op = op, .dtype = dt, .src = copy, .arg = arg }) catch unreachable;
        return id;
    }

    /// Transfer ownership of the slice (caller must later free each src*)
    pub fn toOwnedSlice(self: *UOpBuilder) ![]UOp {
        return self.list.toOwnedSlice();
    }

    /// Free every `src` slice + the array buffer itself.
    pub fn deinit(self: *UOpBuilder) void {
        for (self.list.items) |uop|
            if (uop.src.len > 0) self.alloc.free(@constCast(uop.src));
        self.list.deinit();
    }
};

// Team 1 ZantSyntaxTree - From Onnx to ZantSyntaxTree
// Mirko
// Pietro
// Filippo

// ---Optimization--- Constant folding kernel fusion ---

// Team 2 IR - From High Level IR Math to Low Level IR Math (e.g.LowerRelu LowerAdd)
// Marco
// Mattia
// Alessandro/Adriano

// Team 3 Renderer -From Uops to zig code
// Burak
// Matteo
// Alessandro/Adriano
