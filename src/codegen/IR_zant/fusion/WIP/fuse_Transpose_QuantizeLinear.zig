// // Fused Transpose (NHWC f32) -> QuantizeLinear (NCHW u8) pattern
// const TransposeQuantizePatternResult = struct {
//     detected: bool,
//     transpose_node: ?*NodeZant = null,
//     quant_node: ?*NodeZant = null,
//     input_tensor: ?*TensorZant = null, // NHWC f32 input
//     scale_tensor: ?*TensorZant = null,
//     zero_point_tensor: ?*TensorZant = null,
//     output_tensor: ?*TensorZant = null, // NCHW u8 output
//     n: usize = 1,
//     h: usize = 0,
//     w: usize = 0,
//     c: usize = 0,
// };

// fn detect_transpose_quantize_pattern(nodes: []*NodeZant) !TransposeQuantizePatternResult {
//     if (nodes.len < 2) return TransposeQuantizePatternResult{ .detected = false };
//     const n1 = nodes[0];
//     const n2 = nodes[1];

//     if (!std.mem.eql(u8, n1.op_type, "Transpose")) return TransposeQuantizePatternResult{ .detected = false };
//     if (!std.mem.eql(u8, n2.op_type, "QuantizeLinear")) return TransposeQuantizePatternResult{ .detected = false };

//     const t_in = try n1.get_input_tensors();
//     const t_out = try n1.get_output_tensors();
//     const q_in = try n2.get_input_tensors();
//     const q_out = try n2.get_output_tensors();

//     if (t_in.len == 0 or t_out.len == 0 or q_in.len < 3 or q_out.len == 0) return TransposeQuantizePatternResult{ .detected = false };

//     // Connectivity: transpose output feeds quantize input x
//     if (!std.mem.eql(u8, t_out[0].name, q_in[0].name)) return TransposeQuantizePatternResult{ .detected = false };

//     // Only fuse when transpose input is external INPUT and is f32, and quantize output is u8
//     if (t_in[0].tc != tensorZant_lib.TensorCategory.INPUT) return TransposeQuantizePatternResult{ .detected = false };
//     if (t_in[0].ty != tensorZant_lib.TensorType.f32) return TransposeQuantizePatternResult{ .detected = false };
//     if (q_out[0].ty != tensorZant_lib.TensorType.u8) return TransposeQuantizePatternResult{ .detected = false };

//     // Shape check: NHWC -> NCHW
//     const in_shape = t_in[0].getShape();
//     const out_shape = q_out[0].getShape();
//     if (!is_nhwc_to_nchw(in_shape, out_shape)) return TransposeQuantizePatternResult{ .detected = false };

//     return TransposeQuantizePatternResult{
//         .detected = true,
//         .transpose_node = n1,
//         .quant_node = n2,
//         .input_tensor = t_in[0],
//         .scale_tensor = q_in[1],
//         .zero_point_tensor = q_in[2],
//         .output_tensor = q_out[0],
//         .n = in_shape[0],
//         .h = in_shape[1],
//         .w = in_shape[2],
//         .c = in_shape[3],
//     };
// }

// fn write_transpose_quantize_fused(writer: *std.Io.Writer, pat: TransposeQuantizePatternResult) !void {
//     if (!pat.detected) return;

//     const in_name = try pat.input_tensor.?.getNameSanitized();
//     const out_name = try pat.output_tensor.?.getNameSanitized();

//     // Resolve scale and zero-point expressions
//     const scale_is_param = pat.scale_tensor.?.tc == tensorZant_lib.TensorCategory.INITIALIZER;
//     const zp_is_param = pat.zero_point_tensor.?.tc == tensorZant_lib.TensorCategory.INITIALIZER;

//     const scale_name = try pat.scale_tensor.?.getNameSanitized();
//     const zp_name = try pat.zero_point_tensor.?.getNameSanitized();

//     // Comment and logging
//     if (codegen_options.comm) {
//         try writer.print(
//             \\
//             \\    // OPTIMIZED PATTERN: Transpose (NHWC f32) -> QuantizeLinear (NCHW u8)
//             \\    // Fused quantizing transpose to eliminate intermediate f32 buffer
//             \\
//         , .{});
//     }
//     if (codegen_options.log) {
//         try writer.print(
//             \\
//             \\    if (log_function) |log| {{
//             \\        log(@constCast(@ptrCast("Running fused Transpose+Quantize operation...\n")));
//             \\    }}
//             \\
//         , .{});
//     }

//     // Constants for dimensions
//     try writer.print("    const N: usize = {d};\n", .{pat.n});
//     try writer.print("    const H: usize = {d};\n", .{pat.h});
//     try writer.print("    const W: usize = {d};\n", .{pat.w});
//     try writer.print("    const C: usize = {d};\n", .{pat.c});

//     // Bind input/output data pointers
//     try writer.print("    const x = tensor_{s}.data; // NHWC f32\n", .{in_name});
//     try writer.print("    const y = tensor_{s}.data; // NCHW u8\n", .{out_name});

//     // Load scale and zero point
//     if (scale_is_param) {
//         try writer.print("    const y_scale: f32 = @constCast(&param_lib.tensor_{s}).data[0];\n", .{scale_name});
//     } else {
//         try writer.print("    const y_scale: f32 = tensor_{s}.data[0];\n", .{scale_name});
//     }
//     if (zp_is_param) {
//         try writer.print("    const y_zp: {s} = @constCast(&param_lib.tensor_{s}).data[0];\n", .{ pat.zero_point_tensor.?.ty.toString(), zp_name });
//     } else {
//         try writer.print("    const y_zp: {s} = tensor_{s}.data[0];\n", .{ pat.zero_point_tensor.?.ty.toString(), zp_name });
//     }

//     // Emit fused loops: iterate N,C,H,W and index NHWC input
//     try writer.print(
//         \\    var idx: usize = 0;
//         \\    var n: usize = 0;
//         \\    while (n < N) : (n += 1) {{
//         \\        var c: usize = 0;
//         \\        while (c < C) : (c += 1) {{
//         \\            var h: usize = 0;
//         \\            while (h < H) : (h += 1) {{
//         \\                var w: usize = 0;
//         \\                while (w < W) : (w += 1) {{
//         \\                    const nhwc_index = (((n * H) + h) * W + w) * C + c;
//         \\                    const v: f32 = x[nhwc_index];
//         \\                    const q_i32: i32 = @as(i32, @intFromFloat(@round(v / y_scale))) + @as(i32, y_zp);
//         \\                    const q_clamped: i32 = @max(0, @min(255, q_i32));
//         \\                    y[idx] = @intCast(q_clamped);
//         \\                    idx += 1;
//         \\                }}
//         \\            }}
//         \\        }}
//         \\    }}
//     , .{});
// }
