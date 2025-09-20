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

// --- uops ---
const cg_v2 = @import("codegen").codegen_v2;
const Uops = cg_v2.uops;
const UOpBuilder = cg_v2.builder;
const DType = Uops.DType;
const Any = Uops.Any;

// https://onnx.ai/onnx/operators/onnx__Conv.html
// INPUTS:
//      - X (heterogeneous) - T: Input data tensor
//      - W (heterogeneous) - T: The weight tensor
//      - B (optional, heterogeneous) - T: Optional 1D bias to be added to the convolution, has size of M.
// OUTPUTS:
//      - Y (heterogeneous) - T: Output data tensor that contains the result of the convolution
// ATTRIBUTES:
//      - auto_pad - STRING (default is 'NOTSET'): auto_pad must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID. Where default value is NOTSET
//      - dilations - INTS : dilation value along each spatial axis of the filter. If not present, the dilation defaults is 1 along each spatial axis.
//      - group - INT (default is '1'): number of groups input channels and output channels are divided into
//      - kernel_shape - INTS : The shape of the convolution kernel. If not present, should be inferred from input W
//      - pads - INTS : Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0.
//      - strides - INTS : Stride along each spatial axis. If not present, the stride defaults is 1 along each spatial axis.

pub const Conv = struct {
    input_X: *TensorZant,
    input_W: *TensorZant,
    input_B: ?*TensorZant,
    output_Y: *TensorZant,
    //attributes:
    auto_pad: []const u8,
    dilations: ?[]i64,
    group: i64,
    kernel_shape: ?[]i64,
    pads: ?[]i64,
    strides: ?[]i64,

    pub fn init(nodeProto: *NodeProto) !Conv {
        const input_X = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const input_W = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[1])) |ptr| ptr else return error.input_W_notFound;
        const input_B = if (nodeProto.input.len > 2) if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[2])) |ptr| ptr else return error.input_B_notFound else null;
        const output_Y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;

        var auto_pad: []const u8 = "NOTSET";
        var dilations: ?[]i64 = null;
        var group: i64 = 1;
        var kernel_shape: ?[]i64 = null;
        var pads: ?[]i64 = null;
        var strides: ?[]i64 = null; //mandatory

        for (nodeProto.attribute) |attr| {
            if (std.mem.indexOf(u8, attr.name, "auto_pad")) |_| {
                if (attr.type == onnx.AttributeType.STRING) auto_pad = attr.s else return error.ConvAuto_padNotSTRING;
            } else if (std.mem.indexOf(u8, attr.name, "dilations")) |_| {
                if (attr.type == onnx.AttributeType.INTS) dilations = attr.ints else return error.ConvDilatationNoINTS;
            } else if (std.mem.indexOf(u8, attr.name, "group")) |_| {
                if (attr.type == onnx.AttributeType.INT) group = attr.i else return error.ConvGroupNotINT;
            } else if (std.mem.indexOf(u8, attr.name, "kernel_shape")) |_| {
                if (attr.type == onnx.AttributeType.INTS) kernel_shape = attr.ints else return error.ConvKernelShapeNotINTS;
            } else if (std.mem.indexOf(u8, attr.name, "pads")) |_| {
                if (attr.type == onnx.AttributeType.INTS) pads = attr.ints else return error.ConvPadsNotINTS;
            } else if (std.mem.indexOf(u8, attr.name, "strides")) |_| {
                if (attr.type == onnx.AttributeType.INTS) strides = attr.ints else return error.ConvStridesNotINTS;
            }
        }

        //set the output type:
        if (output_Y.ty == tensorZant_lib.TensorType.undefined) output_Y.ty = input_W.ty;

        return Conv{
            .input_X = input_X,
            .input_W = input_W,
            .input_B = input_B,
            .output_Y = output_Y,
            .auto_pad = auto_pad,
            .dilations = dilations,
            .group = group,
            .kernel_shape = kernel_shape,
            .pads = pads,
            .strides = strides,
        };
    }

    pub fn get_output_shape(self: Conv) []usize {
        return self.output_Y.getShape();
    }

    pub fn get_input_tensors(self: Conv) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();

        try inputs.append(self.input_X);
        try inputs.append(self.input_W);
        if (self.input_B) |bias| {
            try inputs.append(bias);
        }

        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: Conv) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();

        try outputs.append(self.output_Y);
        return outputs.toOwnedSlice();
    }

    pub fn write_op(self: Conv, writer: std.fs.File.Writer) !void {

        //----create tensor_X_string
        var tensor_X_string: []u8 = undefined;
        defer allocator.free(tensor_X_string);

        if (self.input_X.tc == TensorCategory.INITIALIZER) {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_X.name),
                ")",
            });
        } else {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_X.name), ")" });
        }

        //----create tensor_W_string
        var tensor_W_string: []u8 = undefined;
        defer allocator.free(tensor_W_string);
        if (self.input_W.tc == TensorCategory.INITIALIZER) {
            tensor_W_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_W.name),
                ")",
            });
        } else {
            tensor_W_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", try utils.getSanitizedName(self.input_W.name), ")" });
        }

        //----create ?bias string
        var bias_string: []u8 = undefined;
        // Bias Tensor B is optional! verify the presence
        if (self.input_B) |input_B| {
            const B_name = try utils.getSanitizedName(input_B.name);
            bias_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&param_lib.tensor_", B_name, ")" });
        } else {
            bias_string = try std.mem.concat(allocator, u8, &[_][]const u8{"null"});
        }

        //----create stride string (mandatory)
        // TODO: implement default stride, see docs above
        if (self.strides == null) return error.StrideNotFound;
        const stride_string: []const u8 = try utils.i64SliceToUsizeArrayString(self.strides.?);

        //----create ?pads string
        var pads_string: []const u8 = "null";
        if (self.pads != null) {
            if (self.pads.?.len > 0) { // Check if the slice is actually non-empty
                pads_string = try utils.i64SliceToUsizeArrayString(self.pads.?);
                // Assuming no allocation needed to be freed, following write_conv
            } else {
                pads_string = "&[_]usize{}"; // Use explicit empty slice literal if input slice is empty
            }
        } // else pads_string remains "null"

        //----create ?dilatations string
        var dilat_string: []const u8 = "null";
        if (self.dilations != null) {
            if (self.dilations.?.len > 0) {
                dilat_string = try utils.i64SliceToUsizeArrayString(self.dilations.?);
            } else {
                dilat_string = "&[_]usize{}";
            }
        } // else dilat_string remains "null"

        // Check if we need cast operations for mixed precision
        const target_type = self.output_Y.ty.toString();
        const need_kernel_cast = !std.mem.eql(u8, self.input_W.ty.toString(), target_type);
        const need_bias_cast = if (self.input_B) |bias| !std.mem.eql(u8, bias.ty.toString(), target_type) else false;

        var final_kernel_string: []const u8 = undefined;
        var final_bias_string: []const u8 = undefined;
        var need_free_kernel = false;
        var need_free_bias = false;
        defer if (need_free_kernel) allocator.free(@constCast(final_kernel_string));
        defer if (need_free_bias) allocator.free(@constCast(final_bias_string));

        if (need_kernel_cast) {
            // Generate cast for kernel
            const kernel_name = try utils.getSanitizedName(self.input_W.name);
            _ = try writer.print(
                \\
                \\    // Cast kernel from {s} to {s}
                \\    var tensor_{s}_casted = Tensor({s}).fromShape(&allocator, @constCast(param_lib.tensor_{s}.shape)) catch return -2;
                \\    defer tensor_{s}_casted.deinit();
                \\    tensMath.cast_lean({s}, {s}, @constCast(&param_lib.tensor_{s}), &tensor_{s}_casted, zant.onnx.DataType.FLOAT) catch return -1;
                \\
            , .{
                self.input_W.ty.toString(),
                target_type,
                kernel_name,
                target_type,
                kernel_name,
                kernel_name,
                self.input_W.ty.toString(),
                target_type,
                kernel_name,
                kernel_name,
            });
            final_kernel_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", kernel_name, "_casted)" });
            need_free_kernel = true;
        } else {
            final_kernel_string = tensor_W_string;
        }

        if (need_bias_cast and self.input_B != null) {
            // Generate cast for bias
            const bias_name = try utils.getSanitizedName(self.input_B.?.name);
            _ = try writer.print(
                \\
                \\    // Cast bias from {s} to {s}
                \\    var tensor_{s}_casted = Tensor({s}).fromShape(&allocator, @constCast(param_lib.tensor_{s}.shape)) catch return -2;
                \\    defer tensor_{s}_casted.deinit();
                \\    tensMath.cast_lean({s}, {s}, @constCast(&param_lib.tensor_{s}), &tensor_{s}_casted, zant.onnx.DataType.FLOAT) catch return -1;
                \\
            , .{
                self.input_B.?.ty.toString(),
                target_type,
                bias_name,
                target_type,
                bias_name,
                bias_name,
                self.input_B.?.ty.toString(),
                target_type,
                bias_name,
                bias_name,
            });
            final_bias_string = try std.mem.concat(allocator, u8, &[_][]const u8{ "@constCast(&tensor_", bias_name, "_casted)" });
            need_free_bias = true;
        } else {
            final_bias_string = bias_string;
        }

        // pub fn OnnxConvLean(comptime T: type, input: *Tensor(T), kernel: *Tensor(T), output: *Tensor(T), bias: ?*const Tensor(T), stride: []const usize, pads: ?[]const usize, dilations: ?[]const usize, group: ?usize, auto_pad: ?[]const u8) !void
        _ = try writer.print(
            \\    
            \\    @setEvalBranchQuota(10000);
            \\    tensMath.conv_lean(
            \\        {s}, //type
            \\        {s}, //input
            \\        {s}, //kernel
            \\        &tensor_{s}, //output
            \\        {s}, //bias
            \\        {s}, //stride
            \\        {s}, //pads
            \\        {s}, //dilatations
            \\        {}, //group
            \\        "{s}", //auto_pad
            \\    ) catch return -1;
        , .{
            target_type,
            tensor_X_string, //Input
            final_kernel_string, //Kernel (possibly casted)
            try utils.getSanitizedName(self.output_Y.name), //Output
            final_bias_string, //Bias (possibly casted)
            stride_string, //Strides
            pads_string, //Pads
            dilat_string, //Dilatations
            self.group, //Group
            self.auto_pad, //auto_pad
        });
    }

    pub fn compute_output_shape(self: Conv) ![]usize {
        var output_shape: []usize = undefined;
        const input_shape = self.input_X.getShape();
        const kernel_shape = self.input_W.getShape();
        const stride = self.strides;
        const pads = self.pads;
        const dilations = self.dilations;
        const auto_pad = self.auto_pad;
        output_shape = try tensorMath.get_convolution_output_shape(
            f32, // Type parameter
            allocator, // Allocator parameter
            input_shape,
            kernel_shape,
            try utils.i64SliceToUsizeSlice(stride.?),
            if (pads != null) try utils.i64SliceToUsizeSlice(pads.?) else null,
            try utils.i64SliceToUsizeSlice(dilations.?),
            auto_pad,
        );
        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: Conv) void { //TODO
        std.debug.print("\n CONV:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *Conv, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input_X == old_tensor) {
            self.input_X = new_tensor;
            return;
        }
        if (self.input_W == old_tensor) {
            self.input_W = new_tensor;
            return;
        }
        if (self.input_B != null and self.input_B.? == old_tensor) {
            self.input_B = new_tensor;
            return;
        }
        if (self.output_Y == old_tensor) {
            self.output_Y = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }
    pub fn render_lower(self: Conv, builder: *UOpBuilder) !void {
        const X_id = self.input_X.get_tensorZantID();
        const W_id = self.input_W.get_tensorZantID();
        const out_shape = self.get_output_shape();
        const out_id = self.output_Y.get_tensorZantID();
        const in_stride = [2]usize{ @as(usize, @intCast(self.strides.?[0])), @as(usize, @intCast(self.strides.?[1])) };
        const w_stride = [2]usize{ self.input_W.stride[0], self.input_W.stride[1] };
        const group = @as(usize, @intCast(self.group));

        const pads = if (self.pads) |p|
            [2]usize{ @as(usize, @intCast(p[0])), @as(usize, @intCast(p[1])) }
        else
            [2]usize{ 0, 0 };

        const strides_hw = [2]usize{ @as(usize, @intCast(self.strides.?[0])), @as(usize, @intCast(self.strides.?[1])) };
        const dilations = [2]usize{ @as(usize, @intCast(self.dilations.?[0])), @as(usize, @intCast(self.dilations.?[1])) };
        const kernel_shape = [2]usize{ @as(usize, @intCast(self.kernel_shape.?[0])), @as(usize, @intCast(self.kernel_shape.?[1])) };
        const C_per_grp = @as(usize, @intCast(self.kernel_shape.?[1])) / @as(usize, @intCast(self.group));
        const M_per_grp = @as(usize, @intCast(self.kernel_shape.?[0])) / @as(usize, @intCast(self.group));
        const out_dtype = utils.tensorTypeToDtype(self.output_Y.ty);

        lowerConv2d(
            builder,
            X_id,
            W_id,
            out_id,
            out_shape,
            &in_stride,
            &w_stride,
            group,
            pads,
            strides_hw,
            dilations,
            kernel_shape,
            C_per_grp,
            M_per_grp,
            out_dtype,
        );
    }

    pub fn lowerConv2d(
        b: *UOpBuilder,
        X_id: usize, // SSA id of input  X
        W_id: usize, // SSA id of weights W
        out_id: usize,
        out_shape: []const usize, // [N, M, OH, OW]
        in_stride: []const usize, // X: stride vec (len 4)
        w_stride: []const usize, // W: stride vec (len 4)
        group: usize, // number of groups G
        pads: [2]usize, // {padT, padL}
        strides_hw: [2]usize, // {strideH, strideW}
        dil_hw: [2]usize, // {dilH, dilW}
        kHW: [2]usize, // {kH, kW}
        C_per_grp: usize, // C′  in-channels per group
        M_per_grp: usize, // M′  out-channels per group
        out_dtype: DType,
    ) void {

        // ── Tiny helpers to reduce boilerplate ────────────────────────────
        const r = struct {
            fn rng(bi: *UOpBuilder, end: usize) usize { // RANGE 0..end-1
                return bi.push(.RANGE, .i32, &.{}, Any{ .loop_bounds = .{ .start = 0, .end = end } });
            }
            fn kconst(bi: *UOpBuilder, v: usize) usize { // CONST <v>
                return bi.push(.CONST, .i32, &.{}, Any{ .int = v });
            }
        };

        // ── 1. Compile-time constants (pads, strides, …)  →  CONST UOps ----
        const padT = r.kconst(b, pads[0]);
        const padL = r.kconst(b, pads[1]);
        const strH = r.kconst(b, strides_hw[0]);
        const strW = r.kconst(b, strides_hw[1]);
        const dilH = r.kconst(b, dil_hw[0]);
        const dilW = r.kconst(b, dil_hw[1]);
        const Cg = r.kconst(b, C_per_grp); // C′  (used in g*C′+ic)
        const Mg = r.kconst(b, M_per_grp); // M′  (used in g*M′+m′)

        // ── 2. Logical views for X and W  (no data copies) -----------------
        const id_viewX = b.push(.VIEW, out_dtype, &.{X_id}, Any{ .view_meta = .{ .shape = &.{ out_shape[0], C_per_grp * group, out_shape[2], out_shape[3] }, .strides = in_stride } });

        const id_viewW = b.push(.VIEW, out_dtype, &.{W_id}, Any{ .view_meta = .{ .shape = &.{ M_per_grp * group, C_per_grp, kHW[0], kHW[1] }, .strides = w_stride } });

        // ── 3. Outer loops  n · g · m′ · oh · ow  --------------------------
        const n = r.rng(b, out_shape[0]); // batch
        const g = r.rng(b, group); // group id
        const mop = r.rng(b, M_per_grp); // m′   (out-chan inside group)
        const oh = r.rng(b, out_shape[2]); // output row
        const ow = r.rng(b, out_shape[3]); // output col

        // --- fused index  oc = g*M′ + m′   (for output & W)
        const gMulMg = b.push(.MUL, .i32, &.{ g, Mg }, null);
        const oc_idx = b.push(.ADD, .i32, &.{ gMulMg, mop }, null);

        // ── 4. Accumulator register (one per output element) ---------------
        const id_acc = b.push(.DEFINE_ACC, out_dtype, &.{}, null);

        // ── 5. Reduction loops  ic · kh · kw  ------------------------------
        const ic = r.rng(b, C_per_grp); // in-chan inside group
        const kh = r.rng(b, kHW[0]); // kernel row
        const kw = r.rng(b, kHW[1]); // kernel col

        // g*C′ + ic   → all-channel idx into X
        const gMulCg = b.push(.MUL, .i32, &.{ g, Cg }, null);
        const ic_all = b.push(.ADD, .i32, &.{ gMulCg, ic }, null);

        // Input spatial coords (ih, iw) with stride, dilation, padding
        const ohMulStr = b.push(.MUL, .i32, &.{ oh, strH }, null);
        const khMulDil = b.push(.MUL, .i32, &.{ kh, dilH }, null);
        const ih_base = b.push(.ADD, .i32, &.{ ohMulStr, khMulDil }, null);
        const ih_idx = b.push(.SUB, .i32, &.{ ih_base, padT }, null);

        const owMulStr = b.push(.MUL, .i32, &.{ ow, strW }, null);
        const kwMulDil = b.push(.MUL, .i32, &.{ kw, dilW }, null);
        const iw_base = b.push(.ADD, .i32, &.{ owMulStr, kwMulDil }, null);
        const iw_idx = b.push(.SUB, .i32, &.{ iw_base, padL }, null);

        // ---- GEPs for current X and W elements ------------------------
        const id_gepX = b.push(.GEP, out_dtype, &.{ id_viewX, n, ic_all, ih_idx, iw_idx }, Any{ .mem_info = .{ .base = id_viewX, .offset = 0, .stride = 1 } });

        const id_gepW = b.push(.GEP, out_dtype, &.{ id_viewW, oc_idx, ic, kh, kw }, Any{ .mem_info = .{ .base = id_viewW, .offset = 0, .stride = 1 } });

        // ---- Multiply & accumulate  acc += x*w ------------------------
        const id_x = b.push(.LOAD, out_dtype, &.{id_gepX}, null);
        const id_w = b.push(.LOAD, out_dtype, &.{id_gepW}, null);
        _ = b.push(.MULACC, out_dtype, &.{ id_acc, id_x, id_w }, null);

        // close reduction loops                                            ↓↓↓
        _ = b.push(.ENDRANGE, .bool, &.{kw}, null);
        _ = b.push(.ENDRANGE, .bool, &.{kh}, null);
        _ = b.push(.ENDRANGE, .bool, &.{ic}, null);

        // ── 6. Write output pixel ------------------------------------------
        const id_gepY = b.push(.GEP, out_dtype, &.{ out_id, n, oc_idx, oh, ow }, Any{ .mem_info = .{ .base = out_id, .offset = 0, .stride = 1 } });

        _ = b.push(.STORE, out_dtype, &.{ id_gepY, id_acc }, null);

        // close outer loops (reverse order)                               ↓↓↓
        _ = b.push(.ENDRANGE, .bool, &.{ow}, null);
        _ = b.push(.ENDRANGE, .bool, &.{oh}, null);
        _ = b.push(.ENDRANGE, .bool, &.{mop}, null);
        _ = b.push(.ENDRANGE, .bool, &.{g}, null);
        _ = b.push(.ENDRANGE, .bool, &.{n}, null);
    }
};
