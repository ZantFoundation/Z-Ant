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

// https://onnx.ai/onnx/operators/onnx__Pow.html

pub const Pow = struct {
    // inputs
    X: *TensorZant,
    Y: *TensorZant,

    // Z
    Z: *TensorZant,

    pub fn init(nodeProto: *NodeProto) !Pow {
        // Pow has 2-4 inputs
        if (nodeProto.input.len != 2) return error.PowInvalidInputCount;

        const X = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.X_notFound;
        const Y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.Y_notFound;
        const Z = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.Z_notFound;

        //check Z type
        if (Z.ty == tensorZant_lib.TensorType.undefined) {
            Z.ty = X.ty;
        } else if (Z.ty != X.ty) {
            return error.InvalidDataType;
        }

        return Pow{
            .X = X,
            .Y = Y,
            .Z = Z,
        };
    }

    pub fn get_output_shape(self: Pow) []usize {
        return self.Z.getShape();
    }

    //TODO use the broadcast function of pow
    pub fn compute_output_shape(self: Pow) ![]usize {
        var output_shape: []usize = undefined;
        output_shape = try utils.broadcastShapes(allocator, self.X.shape, self.Y.shape);
        self.Z.shape = output_shape;
        return output_shape;
    }

    pub fn get_input_tensors(self: Pow) ![]*TensorZant {
        var inputs: std.ArrayList(*TensorZant) = .empty;
        defer inputs.deinit(allocator);

        try inputs.append(allocator, self.X);
        try inputs.append(allocator, self.Y);

        return inputs.toOwnedSlice(allocator);
    }

    pub fn get_output_tensors(self: Pow) ![]*TensorZant {
        var outputs: std.ArrayList(*TensorZant) = .empty;
        defer outputs.deinit(allocator);

        try outputs.append(allocator, self.Z);
        return outputs.toOwnedSlice(allocator);
    }

    pub fn write_op(self: Pow, writer: *std.Io.Writer) !void {

        //----create tensor_X_string (base)
        var tensor_X_string: []u8 = undefined;
        defer allocator.free(tensor_X_string);
        if (self.X.tc == TensorCategory.INITIALIZER) {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.X.name),
                ")",
            });
        } else {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.X.name), ")" });
        }

        //----create tensor_Y_string (exponent)
        var tensor_Y_string: []u8 = undefined;
        defer allocator.free(tensor_Y_string);
        if (self.Y.tc == TensorCategory.INITIALIZER) {
            tensor_Y_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.Y.name),
                ")",
            });
        } else {
            tensor_Y_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.Y.name), ")" });
        }

        // Check if we need cast operations for mixed precision
        const target_type = self.Z.ty.toString();
        const Y_type = self.Y.ty.toString();
        //const x_type = self.X.ty.toString();
        //const need_x_cast = !std.mem.eql(u8, x_type, target_type);

        //var final_x_string: []const u8 = undefined;
        //var need_free_x = false;
        //defer if (need_free_x) allocator.free(@constCast(final_x_string));

        //if (need_x_cast) {
        // Generate cast for input X (base)
        //const x_name = try utils.getSanitizedName(self.X.name);
        //onst prefix = if (self.X.tc == TensorCategory.INITIALIZER) "param_lib." else "";
        //_ = try writer.print(
        //    \\
        //    \\    // Cast input X from {s} to {s}
        //   \\    var tensor_{s}_X_casted = Tensor({s}).fromShape(&allocator, @constCast({s}tensor_{s}.shape)) catch return -2;
        //   \\    defer tensor_{s}_X_casted.deinit();
        //    \\    tensMath.cast_lean({s}, {s}, @constCast(&{s}tensor_{s}), &tensor_{s}_X_casted, zant.onnx.DataType.FLOAT) catch return {d};
        //    \\
        //, .{
        //    x_type,
        //    target_type,
        //    x_name,
        //    target_type,
        //    prefix,
        //    x_name,
        //    x_name,
        //   x_type,
        //    target_type,
        //    prefix,
        //    x_name,
        //    x_name,
        //    utils.getMathErrorReturn(), // Error code for math errors
        //});
        //final_x_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", x_name, "_X_casted)" });
        //need_free_x = true;
        //} else {
        //    final_x_string = tensor_X_string;
        //}

        _ = try writer.print(
            \\
            \\    tensMath.pow_lean({s}, {s}, {s}, {s}, &tensor_{s}) catch return {d};
        , .{
            target_type,
            Y_type,
            tensor_X_string, //input x doesn't need casting
            tensor_Y_string, //input y doesn't need casting
            try utils.getSanitizedName(self.Z.name), // Output tensor Z
            utils.getMathErrorReturn(), // Error code for math errors
        });
    }

    pub fn print(op: *const Pow) void {
        std.debug.print("\n Pow:\n {any}", .{op});
    }

    pub fn sobstitute_tensors(self: *Pow, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.X == old_tensor) {
            self.X = new_tensor;
            return;
        }
        if (self.Y == old_tensor) {
            self.Y = new_tensor;
            return;
        }

        if (self.Z == old_tensor) {
            self.Z = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }

    //pub fn get__tensors(op: *const Pow) ![]*TensorZant {
    //   var tensors = try allocator.alloc(*TensorZant, 1);
    //    tensors[0] = op.Z;
    //    return tensors;
    //}
};
