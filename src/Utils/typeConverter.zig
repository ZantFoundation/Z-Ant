//! This class is used to convert form a type to another.
//! The types are comptime known, so During compilation coould be triggered a @compileError("...")
const std = @import("std");

pub fn convert(comptime T_in: type, comptime T_out: type, value: T_in) T_out {
    return switch (@typeInfo(T_in)) {
        .Int => switch (@typeInfo(T_out)) {
            .Int => @intCast(value), // Integer to integer
            .Float => @floatFromInt(value), // Integer to float
            .Bool => value != 0, // Integer to bool
            else => @compileError("Unsupported conversion from integer to this type"),
        },
        .Float => switch (@typeInfo(T_out)) {
            .Int => @intFromFloat(value), // Float to integer
            .Float => @floatCast(value), // Float to float
            .Bool => value != 0.0, // Float to bool
            else => @compileError("Unsupported conversion from float to this type"),
        },
        .Bool => switch (@typeInfo(T_out)) {
            .Int => if (value) @intCast(1) else @intCast(0), // Bool to integer
            .Float => if (value) @floatCast(1.0) else @floatCast(0.0), // Bool to float
            .Bool => value, // Bool to bool (identity)
            else => @compileError("Unsupported conversion from bool to this type"),
        },
        .Pointer => @compileError("Unsupported conversion from pointer to another type"),
        .ComptimeInt => switch (@typeInfo(T_out)) {
            .Int => @intCast(value), // ComptimeInt to integer
            .Float => @floatFromInt(value), // ComptimeInt to float
            .Bool => value != 0, // ComptimeInt to bool
            else => @compileError("Unsupported conversion from comptime integer to this type"),
        },
        else => @compileError("Unsupported input type"),
    };
}
