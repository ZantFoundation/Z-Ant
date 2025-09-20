const std = @import("std");
const allocator = std.heap.page_allocator;
const pkg_allocator = zant.utils.allocator.allocator;
const zant = @import("zant");
const IR_zant = @import("../../IR_zant.zig");

// --- onnx ---
const onnx = zant.onnx;
const ModelProto = onnx.ModelProto;
const GraphProto = onnx.GraphProto;
const NodeProto = onnx.NodeProto;
const TensorProto = onnx.TensorProto;

// --- zant ---
const tensorZant_lib = IR_zant.tensorZant_lib;
const TensorZant = tensorZant_lib.TensorZant;
const TensorCategory = tensorZant_lib.TensorCategory;

const tensorMath = zant.core.tensor.math_standard;

const utils = IR_zant.utils;

// --- uops ---
const cg_v2 = @import("codegen").codegen_v2;
const Uops = cg_v2.uops;
const UOpBuilder = cg_v2.builder;
const DType = Uops.DType;
const Any = Uops.Any;

// https://onnx.ai/onnx/operators/onnx__Reshape.html#l-onnx-doc-reshape
// INPUTS:
//      - data (heterogeneous) - T: An input tensor.
//      - shape (optional, heterogeneous) - tensor(int64): Specified shape for output.
//                                                         OSS!!! It is not always present! There may be an error in the ONNX docs, so we set it to optional. Usually the output shape is present as AttributeProto of a NodeProto for Reshape op.
// OUTPUTS:
//      - reshaped (heterogeneous) - T: Reshaped data.
// ATTRIBUTES:
//      - allowzero - INT (default is '0'): If '1', the shape can contain zero. TODO
//      - shape (ints): Alternative way to provide shape (used if input 'shape' is not provided). -> shape_attribute

pub const Reshape = struct {
    data: *TensorZant,
    shape: ?*TensorZant,
    reshaped: *TensorZant,
    allowzer0: bool,
    shape_attribute: ?[]const i64,

    pub fn init(nodeProto: *NodeProto) !Reshape {
        const data = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const shape = if (nodeProto.input.len < 2) null else if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.TensorShape_notFound;
        const reshaped = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var allowzer0: bool = false;
        var shape_attribute: ?[]const i64 = null;

        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "allowzero")) {
                if (attr.type == onnx.AttributeType.INT) allowzer0 = attr.i != 0;
            } else if (std.mem.eql(u8, attr.name, "shape")) {
                if (attr.type == onnx.AttributeType.INTS) shape_attribute = attr.ints;
            }
        }

        //check on the existance of the shape
        if (shape == null and shape_attribute == null) return error.shape_notFound;

        //set the output type:
        reshaped.ty = data.ty;

        return Reshape{
            .data = data,
            .shape = shape,
            .reshaped = reshaped,
            .allowzer0 = allowzer0,
            .shape_attribute = shape_attribute,
        };
    }

    pub fn get_output_shape(self: Reshape) []usize {
        return self.reshaped.getShape();
    }

    pub fn get_input_tensors(self: Reshape) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();

        try inputs.append(self.data);
        if (self.shape) |s| try inputs.append(s);

        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: Reshape) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();

        try outputs.append(self.reshaped);
        return outputs.toOwnedSlice();
    }

    pub fn write_op(self: Reshape, writer: std.fs.File.Writer) !void {
        // Input tensor string creation
        const sanitized_input_name = try utils.getSanitizedName(self.data.name);
        const input_string = try std.mem.concat(allocator, u8, &[_][]const u8{
            if (self.data.tc == TensorCategory.INITIALIZER) "param_lib." else "",
            "tensor_",
            sanitized_input_name,
        });
        defer allocator.free(input_string);

        // Shape slice generation logic
        var shape_slice_code = std.ArrayList(u8).init(allocator);
        defer shape_slice_code.deinit();
        const output_sanitized_name = try utils.getSanitizedName(self.reshaped.name);
        var shape_from_attr = false; // Track source of shape

        if (self.shape_attribute) |attr_shape| {
            shape_from_attr = true;
            // Shape from attribute
            // Generate code like: const shape_slice_<output_name> = [_]isize{ val1, val2, ... };
            try shape_slice_code.writer().print("const shape_slice_{s} = [_]isize{{", .{output_sanitized_name});
            for (attr_shape, 0..) |val, i| {
                try shape_slice_code.writer().print("{s}{}", .{ if (i > 0) ", " else "", val });
            }
            try shape_slice_code.writer().print("}};", .{});
        } else {
            // Shape from input tensor

            const shape_input_tensor = self.shape.?;
            const sanitized_shape_name = try utils.getSanitizedName(shape_input_tensor.name);
            const shape_tensor_name = try std.mem.concat(allocator, u8, &[_][]const u8{
                if (shape_input_tensor.tc == TensorCategory.INITIALIZER) "param_lib." else "",
                "tensor_",
                sanitized_shape_name,
            });
            defer allocator.free(shape_tensor_name);

            // Generate code to convert tensor data to isize slice
            try shape_slice_code.writer().print(
                \\    // Convert shape tensor data to isize slice
                \\    // Pass the local allocator to the utils function
                \\    const shape_slice_{s} = utils.sliceToIsizeSlice(allocator, {s}.data); // Removed catch return
                \\    defer allocator.free(shape_slice_{s}); // Free the runtime allocated slice
            , .{
                output_sanitized_name, // Use output name for uniqueness
                shape_tensor_name,
                output_sanitized_name,
            });
        }

        // Pre-build complex arguments for the format string
        const shape_slice_var_name = try std.fmt.allocPrint(allocator, "shape_slice_{s}", .{output_sanitized_name});
        defer allocator.free(shape_slice_var_name);
        const shape_slice_arg = try std.fmt.allocPrint(allocator, "{s}{s}", .{ if (shape_from_attr) "&" else "", shape_slice_var_name });
        defer allocator.free(shape_slice_arg);

        const output_tensor_arg = try std.fmt.allocPrint(allocator, "&tensor_{s}", .{output_sanitized_name});
        defer allocator.free(output_tensor_arg);

        // Generate the final call using pre-built arguments
        _ = try writer.print(
            \\
            \\
            \\    // Reshape Operation 
            \\    {s} // Generated shape slice code
            \\
            \\    tensMath.reshape_lean(
            \\        {s}, // Use actual input tensor type
            \\        @constCast(&{s}),
            \\        {s}, // Pre-built shape slice argument
            \\        {s}, // Format boolean correctly
            \\        {s}, // Pre-built output tensor argument
            \\    ) catch return -1;
        , .{
            shape_slice_code.items, // Arg 1 for shape code
            self.data.ty.toString(), // Arg 2 for input type
            input_string, // Arg 3 for input tensor
            shape_slice_arg, // Arg 4 for shape slice
            if (self.allowzer0) "true" else "false", // Arg 5 for allowzero
            output_tensor_arg, // Arg 6 for output tensor
        });
    }

    pub fn get_reshape_output_shape(self: Reshape) ![]usize {
        var output_shape: []usize = undefined;
        const new_shape_spec = try allocator.alloc(isize, self.shape.shape.len);
        defer allocator.free(new_shape_spec);
        for (self.shape.shape, 0..) |val, i| {
            new_shape_spec[i] = @as(isize, @intCast(val));
        }
        output_shape = try tensorMath.get_reshape_output_shape(
            self.data.shape,
            new_shape_spec,
            false,
        );
        self.reshaped.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Reshape) void {
        std.debug.print("\n Reshape:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *Reshape, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.data == old_tensor) {
            self.data = new_tensor;
            return;
        }
        if (self.shape != null and self.shape.? == old_tensor) {
            self.shape = new_tensor;
            return;
        }
        if (self.reshaped == old_tensor) {
            self.reshaped = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }

    pub fn render_lower(self: Reshape, builder: *UOpBuilder) !void {
        const X_id = self.data.get_tensorZantID();
        const out_id = self.get_output_tensor().get_tensorZantID();
        const out_shape = self.get_output_shape();
        const out_dtype = utils.tensorTypeToDtype(self.reshaped.ty);

        try lowerReshape(
            builder,
            X_id,
            out_id,
            out_shape,
            out_dtype,
        );
    }

    /// https://onnx.ai/onnx/operators/onnx__Reshape.html
    pub fn lowerReshape(
        b: *UOpBuilder,
        A_id: usize, // input-tensor SSA id
        out_id: usize,
        out_shape: []const usize,
        out_dtype: DType, // promoted element type
    ) !void { // returns id of result buffer

        // ── Set-up phase ────────────────────────────────────────────────────
        _ = b.push(.SHAPE, .i32, &.{A_id}, null); // a_shape  (dbg only)

        const id_viewA = b.push(.VIEW, out_dtype, &.{A_id}, Any{ .view_meta = .{ .shape = out_shape, .strides = &.{ 1, 1 } } });

        // ── Flat element loop ────────────────────────────────────────────────

        // For dim = -1 calculate -1 from number elemets
        // For dim = 0 get the previous dim value from the previous shape

        var nelem: usize = 1;
        for (out_shape) |dim| nelem *= dim;

        var id_ranges = std.ArrayList(usize).init(pkg_allocator);
        defer id_ranges.deinit();

        _ = b.push(.RESHAPE, out_dtype, &.{id_viewA}, Any{ .shape = out_shape });

        for (out_shape) |dim| {
            const id_range = b.push(.RANGE, .i32, &.{}, Any{ .loop_bounds = .{ .start = 0, .end = dim } });
            id_ranges.append(id_range) catch {};
        }

        var src_A = std.ArrayList(usize).init(pkg_allocator);
        defer src_A.deinit();
        try src_A.append(id_viewA);
        for (id_ranges.items) |range| {
            try src_A.append(range);
        }

        const id_gepA = b.push(.GEP, out_dtype, src_A.items, Any{ .mem_info = .{ .base = id_viewA, .offset = 0, .stride = 1 } });

        const id_loadA = b.push(.LOAD, out_dtype, &.{id_gepA}, null);

        var src_0 = std.ArrayList(usize).init(pkg_allocator);
        defer src_0.deinit();

        try src_0.append(out_id);
        for (id_ranges.items) |range| {
            try src_0.append(range);
        }

        const id_gepO = b.push(.GEP, out_dtype, src_0.items, Any{ .mem_info = .{ .base = out_id, .offset = 0, .stride = 1 } });

        _ = b.push(.STORE, out_dtype, &.{ id_gepO, id_loadA }, null);

        for (id_ranges.items) |i| {
            _ = b.push(.ENDRANGE, .bool, &.{i}, null);
        }
    }
};
