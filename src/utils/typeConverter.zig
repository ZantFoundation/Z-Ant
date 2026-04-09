//! This class is used to convert form a type to another.
//! The types are comptime known, so During compilation coould be triggered a @compileError("...")

pub fn convert(comptime T_in: type, comptime T_out: type, value: T_in) T_out {
    return switch (@typeInfo(T_in)) {
        .int => switch (@typeInfo(T_out)) {
            .int => @intCast(value), // Integer to integer
            .float => @floatFromInt(value), // Integer to float
            .bool => value != 0, // Integer to bool
            else => @compileError("Unsupported conversion from integer to this type"),
        },
        .float => switch (@typeInfo(T_out)) {
            .int => @intFromFloat(value), // Float to integer
            .float => @floatCast(value), // Float to float
            .bool => value != 0.0, // Float to bool
            else => @compileError("Unsupported conversion from float to this type"),
        },
        .bool => switch (@typeInfo(T_out)) {
            .int => if (value) @intCast(1) else @intCast(0), // Bool to integer
            .float => if (value) @floatCast(1.0) else @floatCast(0.0), // Bool to float
            .bool => value, // Bool to bool (identity)
            else => @compileError("Unsupported conversion from bool to this type"),
        },
        .pointer => @compileError("Unsupported conversion from pointer to another type"),
        .comptime_int => switch (@typeInfo(T_out)) {
            .int => @intCast(value), // ComptimeInt to integer
            .float => @floatFromInt(value), // ComptimeInt to float
            .bool => value != 0, // ComptimeInt to bool
            else => @compileError("Unsupported conversion from comptime integer to this type"),
        },
        else => @compileError("Unsupported input type"),
    };
}
