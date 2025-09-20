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
const TensorZant = tensorZant_lib.TensorZant;
const TensorCategory = tensorZant_lib.TensorCategory;

const tensorMath = zant.core.tensor.math_standard;

const utils = IR_zant.utils;

// https://onnx.ai/onnx/operators/onnx__Shape.html
// INPUTS:
//      - data (heterogeneous) - T: An input tensor.
// OUTPUTS:
//      - shape (heterogeneous) - T1: Shape of the input tensor
// ATTRIBUTES:
//      - start - INT: First dimension to take
//      - end - INT: Last dimension to take
pub const Shape = struct {
    data: *TensorZant,
    shape: *TensorZant,
    //attributes:
    start: ?i64 = null,
    end: ?i64 = null,

    pub fn init(nodeProto: *NodeProto) !Shape {
        const data = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.data_notFound;
        const shape = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.shape_notFound;

        var start: ?i64 = null;
        var end: ?i64 = null;

        for (nodeProto.attribute) |attr| {
            if (std.mem.eql(u8, attr.name, "start")) {
                if (attr.type == onnx.AttributeType.INT) start = attr.i;
            } else if (std.mem.eql(u8, attr.name, "end")) {
                if (attr.type == onnx.AttributeType.INT) end = attr.i;
            }
        }

        //set the output type:
        if (shape.ty == tensorZant_lib.TensorType.undefined) shape.ty = data.ty;

        return Shape{
            .data = data,
            .shape = shape,
            .start = start,
            .end = end,
        };
    }

    pub fn get_output_shape(self: Shape) []usize {
        return self.shape.getShape();
    }

    pub fn get_input_tensors(self: Shape) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();
        try inputs.append(self.data);
        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: Shape) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();
        try outputs.append(self.shape);
        return outputs.toOwnedSlice();
    }

    pub fn write_op(self: Shape, writer: std.fs.File.Writer) !void {
        //----create tensor_data_string
        var tensor_data_string: []u8 = undefined;
        defer allocator.free(tensor_data_string);
        if (self.data.tc == TensorCategory.INITIALIZER) {
            tensor_data_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.data.name),
                ")",
            });
        } else {
            tensor_data_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "&tensor_", try utils.getSanitizedName(self.data.name) });
        }

        //attributes
        _ = try writer.print(
            \\
            \\    tensMath.shape_onnx_lean(
            \\        {s},
            \\        i64, //output type constraint
            \\        {s}, //input data tensor
            \\        {s}, //start
            \\        {s}, //end
            \\        &tensor_{s}, //output shape tensor
            \\    ) catch return -1;
        , .{
            self.data.ty.toString(),
            tensor_data_string,
            if (self.start) |s| try std.fmt.allocPrint(allocator, "{}", .{s}) else "null",
            if (self.end) |e| try std.fmt.allocPrint(allocator, "{}", .{e}) else "null",
            try utils.getSanitizedName(self.shape.name),
        });
    }

    pub fn compute_output_shape(self: Shape) []usize {
        var output_shape: []usize = undefined;
        output_shape = try tensorMath.get_shape_output_shape(
            self.data.shape,
            self.start,
            self.end,
        );
        self.shape.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Shape) void {
        std.debug.print("\n Shape:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *Shape, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.data == old_tensor) {
            self.data = new_tensor;
            return;
        }
        if (self.shape == old_tensor) {
            self.shape = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
};
