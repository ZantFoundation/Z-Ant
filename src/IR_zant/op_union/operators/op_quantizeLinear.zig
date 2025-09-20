const std = @import("std");
const allocator = std.heap.page_allocator;
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
const TensorType = tensorZant_lib.TensorType;
const TensorZant = tensorZant_lib.TensorZant;
const TensorCategory = tensorZant_lib.TensorCategory;

const tensorMath = zant.core.tensor.math_standard;

const utils = IR_zant.utils;

// https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html?utm_source=chatgpt.com
// INPUTS:
//      - x (heterogeneous) - T1:  input tensor.
//      - y_scale (heterogeneous) - T2:  Scale for doing quantization to get y.
//      - y_zero_point (optional, heterogeneous) - 3:  Zero point for doing quantization to get y.
// OUTPUTS:
//      -  y (heterogeneous) - T3:  N-D quantized output tensor. It has same shape as input x.
// ATTRIBUTES:
//      - axis - INT (Optional, default is '1'): The axis of the dequantizing dimension of the input tensor.
//      - block_size - INT (Optional, default is '0'): The size of the quantization block (number of times every scale is replicated)
//      - output_dtype - INT (Optional, default is '0'): The output data type.
//      - precision - INT (Optional, default is '0'): The precision of the division operation between x and y_scale. If not provided, it will be the same as the type of y_scale.
//      - saturate - INT (default is '1'):The parameter defines how the conversion behaves if an input value is out of range of the destination type.
pub const QuantizeLinear = struct {
    //inputs
    x: *TensorZant,
    y_scale: *TensorZant,
    y_zero_point: ?*TensorZant,
    //outputs
    y: *TensorZant,
    //attributes
    axis: i64,
    block_size: i64,
    output_dtype: TensorType,
    precision: i64,
    saturate: i64,

    pub fn init(nodeProto: *NodeProto) !QuantizeLinear {
        const x = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const y_scale = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_y_scale_notFound;
        const y_zero_point = if (nodeProto.input.len > 2) if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[2])) |ptr| ptr else return error.input_y_zero_point_notFound else null;
        const y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var axis: i64 = 1;
        var block_size: i64 = 0;
        var output_dtype: i64 = 0;
        var precision: i64 = 0;
        var saturate: i64 = 1;

        for (nodeProto.attribute) |attr| {
            if (std.mem.indexOf(u8, attr.name, "axis")) |_| {
                if (attr.type == onnx.AttributeType.INT) axis = attr.i else return error.AxisNotINT;
            } else if (std.mem.indexOf(u8, attr.name, "block_size")) |_| {
                if (attr.type == onnx.AttributeType.INT) block_size = attr.i else return error.Block_size_NotINT;
            } else if (std.mem.indexOf(u8, attr.name, "output_dtype")) |_| {
                if (attr.type == onnx.AttributeType.INT) output_dtype = attr.i else return error.Output_dtype_NotINT;
            } else if (std.mem.indexOf(u8, attr.name, "precision")) |_| {
                if (attr.type == onnx.AttributeType.INT) precision = attr.i else return error.PrecisionNotINT;
            } else if (std.mem.indexOf(u8, attr.name, "saturate")) |_| {
                if (attr.type == onnx.AttributeType.INT) saturate = attr.i else return error.SaturateNotINT;
            }
        }

        const outputType: TensorType = if (output_dtype == 0) blk: {
            // output_dtype = 0 means use the inferred type from the output tensor
            // This is the default behavior for QuantizeLinear
            break :blk if (y.ty != TensorType.undefined) y.ty else TensorType.u8; // Default to u8 if still undefined
        } else switch (output_dtype) {
            1 => TensorType.f32, // TensorProto.FLOAT
            2 => TensorType.u8, // UINT8
            3 => TensorType.i8, // INT8
            4 => TensorType.u16, // UINT16
            5 => TensorType.i16, // INT16
            6 => TensorType.i32, // INT32
            7 => TensorType.i64, // INT64
            9 => TensorType.bool, // BOOL
            10 => TensorType.f16, // FLOAT16
            11 => TensorType.f64, // DOUBLE
            12 => TensorType.u32, // UINT32
            13 => TensorType.u64, // UINT64
            // 17 => float8.E4M3FN, // FLOAT8E4M3FN
            // 18 => float8.E4M3FNUZ, // FLOAT8E4M3FNUZ
            // 19 => float8.E5M2, // FLOAT8E5M2
            // 20 => float8.E5M2FNUZ, // FLOAT8E5M2FNUZ
            21 => TensorType.u4, // UINT4
            22 => TensorType.i4, // INT4
            // 23 => float4.E2M1, // FLOAT4E2M1
            // 24 => float8.E8M0, // FLOAT8E8M0
            else => return error.outputType_notSupported,
        };

        //set the output type:
        if (y.ty == tensorZant_lib.TensorType.undefined) y.ty = outputType;

        //set the output shape if it's a placeholder:
        if (y.shape.len == 1 and y.shape[0] == 1) {
            // Use the input shape for the output
            y.shape = x.shape;
        }

        return QuantizeLinear{
            .x = x,
            .y_scale = y_scale,
            .y_zero_point = y_zero_point,
            .y = y,
            .axis = axis,
            .block_size = block_size,
            .output_dtype = outputType,
            .precision = precision,
            .saturate = saturate,
        };
    }

    pub fn get_output_shape(self: QuantizeLinear) []usize {
        return self.y.getShape();
    }

    pub fn get_input_tensors(self: QuantizeLinear) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();

        try inputs.append(self.x);
        try inputs.append(self.y_scale);
        if (self.y_zero_point) |y_zero_point| try inputs.append(y_zero_point);

        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: QuantizeLinear) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();

        try outputs.append(self.y);

        return outputs.toOwnedSlice();
    }

    pub fn write_op(self: QuantizeLinear, writer: std.fs.File.Writer) !void {
        // Create input tensor string
        var x_tensor_string: []const u8 = undefined;
        defer allocator.free(x_tensor_string);

        if (self.x.tc == TensorCategory.INITIALIZER) {
            x_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try self.x.getNameSanitized(),
                ")",
            });
        } else {
            x_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try self.x.getNameSanitized() });
        }

        // Create y_scale_tensor_string tensor string
        var y_scale_tensor_string: []const u8 = undefined;
        defer allocator.free(y_scale_tensor_string);

        if (self.y_scale.tc == TensorCategory.INITIALIZER) {
            y_scale_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try self.y_scale.getNameSanitized(),
                ")",
            });
        } else {
            y_scale_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@as(*const Tensor(f32), &tensor_", try self.y_scale.getNameSanitized(), ")" });
        }

        // Create ?y_zero_point_tensor_string tensor string
        var y_zero_point_tensor_string: []const u8 = "null";
        defer allocator.free(y_zero_point_tensor_string);

        if (self.y_zero_point != null) {
            if (self.y_zero_point.?.tc == TensorCategory.INITIALIZER) {
                y_zero_point_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                    "@constCast(&param_lib.tensor_",
                    try self.y_zero_point.?.getNameSanitized(),
                    ")",
                });
            } else {
                y_zero_point_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try self.y_zero_point.?.getNameSanitized() });
            }
        }

        // Create output tensor_string tensor string
        var y_tensor_string: []u8 = undefined;
        defer allocator.free(y_tensor_string);

        if (self.y.tc == TensorCategory.INITIALIZER) {
            y_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try self.y.getNameSanitized(),
                ")",
            });
        } else {
            y_tensor_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try self.y.getNameSanitized() });
        }

        _ = try writer.print(
            \\
            \\
            \\    tensMath.quantizeLinear_lean({s}, // InputType
            \\                                 {s}, // OutputType
            \\                                 {s}, // ZeroPointType
            \\                                 {s}, // x: input tensor
            \\                                 {s}, // y_scale
            \\                                 {s}, // y_zero_point
            \\                                 {},  // axis
            \\                                 {},  // block_size
            \\                                 &tensor_{s}, // y: output tensor
            \\    ) catch return -1;
        , .{
            self.x.ty.toString(),
            self.y.ty.toString(),
            if (self.y_zero_point) |zp| zp.ty.toString() else "i8", // Default to i8 for zero_point
            x_tensor_string,
            y_scale_tensor_string,
            y_zero_point_tensor_string,
            self.axis,
            self.block_size,
            try self.y.getNameSanitized(),
        });
    }

    pub fn compute_output_shape(self: QuantizeLinear) []usize {
        return self.x.shape;
    }

    pub fn print(self: QuantizeLinear) void {
        std.debug.print("\n QuantizeLinear:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *QuantizeLinear, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.x == old_tensor) {
            self.x = new_tensor;
            return;
        }
        if (self.y_scale == old_tensor) {
            self.y_scale = new_tensor;
            return;
        }
        if (self.y_zero_point != null and self.y_zero_point.? == old_tensor) {
            self.y_zero_point = new_tensor;
            return;
        }
        if (self.y == old_tensor) {
            self.y = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
};
