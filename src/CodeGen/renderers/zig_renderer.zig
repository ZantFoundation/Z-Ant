const std = @import("std");
const zant = @import("zant");
const UOp = zant.uops.UOp;
const UOpType = zant.uops.UOpType;

const MemoryRender = @import("memory_render.zig");
const ArithmeticRender = @import("arithmetic_render.zig");
const ReduceRender = @import("reduce_render.zig");
const ConditionalRender = @import("conditional_render.zig");
const UnaryRender = @import("unary_render.zig");
const GepRender = @import("gep_render.zig");
const ControlFlowRender = @import("controlflow_render.zig");

pub fn ZigRenderer(comptime WriterType: type) type {
    return struct {
        allocator: std.mem.Allocator,
        writer: WriterType,

        pub fn init(allocator: std.mem.Allocator, writer: WriterType) @This() {
            return .{
                .allocator = allocator,
                .writer = writer,
            };
        }

        pub fn deinit(_: @This()) void {
            // Cleanup if needed
        }

        pub fn render(self: *@This(), uops: []UOp) !void {
            for (uops) |uop| {
                switch (uop.op) {
                    .DEFINE_GLOBAL, .LOAD, .STORE => try MemoryRender.render(self.allocator, self.writer, uop),
                    .ADD, .SUB, .MUL, .FDIV, .POW => try ArithmeticRender.render(self.allocator, self.writer, uop),
                    .REDUCE_ADD, .REDUCE_MAX => try ReduceRender.render(self.allocator, self.writer, uop),
                    .MAX, .MIN => try ConditionalRender.render(self.allocator, self.writer, uop),
                    .EXP2, .NEG => try UnaryRender.render(self.allocator, self.writer, uop),
                    .GEP => try GepRender.render(self.allocator, self.writer, uop),
                    .RANGE, .ENDRANGE => try ControlFlowRender.render(self.allocator, self.writer, uop),
                    else => {
                        try std.fmt.format(self.writer, "unknown op {d}\n", .{uop.id});
                    },
                }
            }
        }
    };
}
