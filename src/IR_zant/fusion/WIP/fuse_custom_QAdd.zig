//! in this file you will find everything you need to fuse two or more operation
//! into one.
//!
//!  More specifically for this file:
//!     DequantizeLinear ->
//!                         ->  Add -> QuantizeLinear
//!     DequantizeLinear ->
//! Where Add is the root node
//! is converted into:
//!     QLinearAdd

const std = @import("std");
const zant = @import("zant");
const allocator = zant.utils.allocator.allocator;

const IR_zant = @import("../IR_zant.zig");

const operators = IR_zant.operators;
const Op_union = @import("../op_union/op_union.zig").Op_union;

// // QAdd fusion pattern: two dequantized inputs added then re-quantized
// const QAddPatternResult = struct {
//     detected: bool,
//     deq_a_node: ?*NodeZant = null,
//     deq_b_node: ?*NodeZant = null,
//     add_node: ?*NodeZant = null,
//     quant_node: ?*NodeZant = null,
//     a_q: ?*TensorZant = null,
//     a_scale: ?*TensorZant = null,
//     a_zp: ?*TensorZant = null,
//     b_q: ?*TensorZant = null,
//     b_scale: ?*TensorZant = null,
//     b_zp: ?*TensorZant = null,
//     c_q: ?*TensorZant = null,
//     c_scale: ?*TensorZant = null,
//     c_zp: ?*TensorZant = null,
// };

// fn detect_qadd_pattern(nodes: []*NodeZant) !QAddPatternResult {
//     if (nodes.len < 4) return QAddPatternResult{ .detected = false };
//     const n0 = nodes[0];
//     const n1 = nodes[1];
//     const n2 = nodes[2];
//     const n3 = nodes[3];

//     if (!std.mem.eql(u8, n0.op_type, "DequantizeLinear")) return QAddPatternResult{ .detected = false };
//     if (!std.mem.eql(u8, n1.op_type, "DequantizeLinear")) return QAddPatternResult{ .detected = false };
//     if (!std.mem.eql(u8, n2.op_type, "Add")) return QAddPatternResult{ .detected = false };
//     if (!std.mem.eql(u8, n3.op_type, "QuantizeLinear")) return QAddPatternResult{ .detected = false };

//     const deq0_out = try n0.get_output_tensors();
//     const deq1_out = try n1.get_output_tensors();
//     const add_in = try n2.get_input_tensors();
//     const add_out = try n2.get_output_tensors();
//     const q_in = try n3.get_input_tensors();
//     const q_out = try n3.get_output_tensors();

//     if (deq0_out.len == 0 or deq1_out.len == 0 or add_in.len < 2 or add_out.len == 0 or q_in.len < 3 or q_out.len == 0) {
//         return QAddPatternResult{ .detected = false };
//     }

//     // Check that add inputs are the dequant outputs (order-free)
//     const d0 = deq0_out[0].name;
//     const d1 = deq1_out[0].name;
//     const a0 = add_in[0].name;
//     const a1 = add_in[1].name;
//     const add_inputs_match = (std.mem.eql(u8, d0, a0) and std.mem.eql(u8, d1, a1)) or (std.mem.eql(u8, d0, a1) and std.mem.eql(u8, d1, a0));
//     if (!add_inputs_match) return QAddPatternResult{ .detected = false };

//     // Add output feeds QuantizeLinear input x
//     if (!std.mem.eql(u8, add_out[0].name, q_in[0].name)) return QAddPatternResult{ .detected = false };

//     // Gather input quantized tensors and scales/zps
//     const deq0_in = try n0.get_input_tensors();
//     const deq1_in = try n1.get_input_tensors();
//     if (deq0_in.len < 3 or deq1_in.len < 3) return QAddPatternResult{ .detected = false };

//     return QAddPatternResult{
//         .detected = true,
//         .deq_a_node = n0,
//         .deq_b_node = n1,
//         .add_node = n2,
//         .quant_node = n3,
//         .a_q = deq0_in[0],
//         .a_scale = deq0_in[1],
//         .a_zp = deq0_in[2],
//         .b_q = deq1_in[0],
//         .b_scale = deq1_in[1],
//         .b_zp = deq1_in[2],
//         .c_q = q_out[0],
//         .c_scale = q_in[1],
//         .c_zp = q_in[2],
//     };
// }

// fn write_qadd_fused(writer: std.fs.File.Writer, pat: QAddPatternResult) !void {
//     if (!pat.detected) return;

//     // Helper to create tensor reference strings similar to other fused writers
//     const makeRef = struct {
//         fn call(t: *TensorZant) ![]u8 {
//             if (t.tc == tensorZant_lib.TensorCategory.INITIALIZER) {
//                 return try std.mem.concat(allocator, u8, &[_][]const u8{
//                     "@constCast(&param_lib.tensor_",
//                     try t.getNameSanitized(),
//                     ")",
//                 });
//             } else {
//                 return try std.mem.concat(allocator, u8, &[_][]const u8{
//                     "@constCast(&tensor_",
//                     try t.getNameSanitized(),
//                     ")",
//                 });
//             }
//         }
//     }.call;

//     const a_q_ref = try makeRef(pat.a_q.?);
//     defer allocator.free(a_q_ref);
//     const a_scale_ref = try makeRef(pat.a_scale.?);
//     defer allocator.free(a_scale_ref);
//     const a_zp_ref = try makeRef(pat.a_zp.?);
//     defer allocator.free(a_zp_ref);
//     const b_q_ref = try makeRef(pat.b_q.?);
//     defer allocator.free(b_q_ref);
//     const b_scale_ref = try makeRef(pat.b_scale.?);
//     defer allocator.free(b_scale_ref);
//     const b_zp_ref = try makeRef(pat.b_zp.?);
//     defer allocator.free(b_zp_ref);
//     const c_scale_ref = try makeRef(pat.c_scale.?);
//     defer allocator.free(c_scale_ref);
//     const c_zp_ref = try makeRef(pat.c_zp.?);
//     defer allocator.free(c_zp_ref);

//     const out_name = try pat.c_q.?.getNameSanitized();

//     if (codegen_options.comm) {
//         try writer.print(
//             \\
//             \\    // OPTIMIZED PATTERN: DequantizeLinear + DequantizeLinear + Add + QuantizeLinear
//             \\    // Replaced with direct qlinearadd_lean to save passes and allocations
//             \\
//         , .{});
//     }
//     if (codegen_options.log) {
//         try writer.print(
//             \\
//             \\    if (log_function) |log| {{
//             \\        log(@constCast(@ptrCast("Running fused QLinearAdd operation...\n")));
//             \\    }}
//             \\
//         , .{});
//     }

//     // Emit the fused call
//     try writer.print("    tensMath.qlinearadd_lean(\n", .{});
//     try writer.print("        {s},\n", .{a_q_ref});
//     try writer.print("        {s},\n", .{a_scale_ref});
//     try writer.print("        {s},\n", .{a_zp_ref});
//     try writer.print("        {s},\n", .{b_q_ref});
//     try writer.print("        {s},\n", .{b_scale_ref});
//     try writer.print("        {s},\n", .{b_zp_ref});
//     try writer.print("        &tensor_{s},\n", .{out_name});
//     try writer.print("        {s},\n", .{c_scale_ref});
//     try writer.print("        {s},\n", .{c_zp_ref});
//     try writer.print("    ) catch return -1;\n", .{});
// }
