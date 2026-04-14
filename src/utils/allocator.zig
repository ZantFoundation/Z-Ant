//! Project-wide allocator facade for Zant.
//! Exposes a single `allocator` constant whose backing implementation is
//! selected at build time (via `build_options.allocator`), defaulting to
//! `std.testing.allocator` during tests. Also provides `InfoAllocator` /
//! `ScopedInfoAllocator` wrappers that log every allocation/deallocation.
const std = @import("std");
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");
const build_options = @import("build_options");

//var info_allocator = infoAllocator(.activation, std.heap.raw_c_allocator);
const base_allocator = if (builtin.is_test) std.testing.allocator else @field(std.heap, build_options.allocator);
pub const allocator = base_allocator;

/// This allocator is used in front of another allocator and logs to `std.log`
/// on every call to the allocator.
/// For logging to a `std.io.Writer` see `std.heap.LogToWriterAllocator`
pub fn InfoAllocator(
    comptime scope: @Type(.EnumLiteral),
    comptime success_log_level: std.log.Level,
    comptime failure_log_level: std.log.Level,
) type {
    return ScopedInfoAllocator(scope, success_log_level, failure_log_level);
}

/// This allocator is used in front of another allocator and logs to `std.log`
/// with the given scope on every call to the allocator.
/// For logging to a `std.io.Writer` see `std.heap.LogToWriterAllocator`
pub fn ScopedInfoAllocator(
    comptime scope: @Type(.EnumLiteral),
    comptime success_log_level: std.log.Level,
    comptime failure_log_level: std.log.Level,
) type {
    const log = std.log.scoped(scope);

    return struct {
        parent_allocator: Allocator,
        total_memory: i64,
        last_alloc_total: i64,
        last_free_total: i64,

        const Self = @This();

        pub fn init(parent_allocator: Allocator) Self {
            return .{
                .total_memory = 0,
                .last_free_total = 0,
                .last_alloc_total = 0,
                .parent_allocator = parent_allocator,
            };
        }

        pub fn allocator(self: *Self) Allocator {
            return .{
                .ptr = self,
                .vtable = &.{
                    .alloc = alloc,
                    .resize = resize,
                    .free = free,
                },
            };
        }

        // This function is required as the `std.log.log` function is not public
        inline fn logHelper(comptime log_level: std.log.Level, comptime format: []const u8, args: anytype) void {
            switch (log_level) {
                .err => log.err(format, args),
                .warn => log.warn(format, args),
                .info => log.info(format, args),
                .debug => log.debug(format, args),
            }
        }

        fn alloc(
            ctx: *anyopaque,
            len: usize,
            log2_ptr_align: u8,
            ra: usize,
        ) ?[*]u8 {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const result = self.parent_allocator.rawAlloc(len, log2_ptr_align, ra);
            if (result != null) {
                self.total_memory += @intCast(len);
                if (self.total_memory != self.last_alloc_total) {
                    logHelper(
                        success_log_level,
                        "a t {d: >10} l {} pa {}",
                        .{ self.total_memory, len, log2_ptr_align },
                    );
                    self.last_alloc_total = self.total_memory;
                }
            } else {
                logHelper(
                    failure_log_level,
                    "{any} alloc - failure: OutOfMemory - len: {}, ptr_align: {}",
                    .{ scope, len, log2_ptr_align },
                );
            }
            return result;
        }

        fn resize(
            ctx: *anyopaque,
            buf: []u8,
            log2_buf_align: u8,
            new_len: usize,
            ra: usize,
        ) bool {
            const self: *Self = @ptrCast(@alignCast(ctx));
            if (self.parent_allocator.rawResize(buf, log2_buf_align, new_len, ra)) {
                if (new_len <= buf.len) {
                    logHelper(
                        success_log_level,
                        "shrink - success - {} to {}, buf_align: {}",
                        .{ buf.len, new_len, log2_buf_align },
                    );
                } else {
                    logHelper(
                        success_log_level,
                        "expand - success - {} to {}, buf_align: {}",
                        .{ buf.len, new_len, log2_buf_align },
                    );
                }

                return true;
            }

            std.debug.assert(new_len > buf.len);
            logHelper(
                failure_log_level,
                "expand - failure - {} to {}, buf_align: {}",
                .{ buf.len, new_len, log2_buf_align },
            );
            return false;
        }

        fn free(
            ctx: *anyopaque,
            buf: []u8,
            log2_buf_align: u8,
            ra: usize,
        ) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            self.parent_allocator.rawFree(buf, log2_buf_align, ra);
            self.total_memory -= @intCast(buf.len);
            if (self.last_free_total != self.total_memory) {
                logHelper(success_log_level, "f t {d: >10} l {}", .{ self.total_memory, buf.len });
                self.last_free_total = self.total_memory;
            }
        }
    };
}

/// This allocator is used in front of another allocator and logs to `std.log`
/// on every call to the allocator.
/// For logging to a `std.io.Writer` see `std.heap.LogToWriterAllocator`
pub fn infoAllocator(comptime scope: @Type(.EnumLiteral), parent_allocator: Allocator) InfoAllocator(scope, .debug, .err) {
    return InfoAllocator(scope, .debug, .err).init(parent_allocator);
}
