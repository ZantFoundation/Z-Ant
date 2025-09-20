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

//https://onnx.ai/onnx/operators/onnx__MaxPool.html
// INPUTS:
//      - X (heterogeneous) - T:  input tensor.
// OUTPUTS:
//      - Y (heterogeneous) - T:  output tensor.
//      - indices (optional, heterogeneous) - T:  output indices tensor.
// ATTRIBUTES:
//      - auto_pad (string) - AutoPad type. Default is NOTSET.
//      - ceil_mode (int) - Ceil mode. Default is 0.
//      - dilations (list of ints) - Dilation value. Default is 1.
//      - kernel_shape (list of ints) - Kernel shape.
//      - pads (list of ints) - Padding value. Default is 0.
//      - storage_order (int) - Storage order. Default is 0.
//      - strides (list of ints) - Stride value. Default is 1.

pub const MaxPool = struct {
    input_X: *TensorZant,
    output_Y: *TensorZant,
    output_indices: ?*TensorZant,
    //attributes:
    auto_pad: []const u8, // default = "NOTSET",
    ceil_mode: i64, // default = 0;
    dilations: ?[]i64, // default = null;
    kernel_shape: ?[]i64, // default = null; but mandatory
    pads: ?[]i64, // default = null;
    storage_order: i64, // default = 0;
    strides: ?[]i64, // default = null;

    pub fn init(nodeProto: *NodeProto) !MaxPool {
        const input_X = if (tensorZant_lib.tensorMap.getPtr(nodeProto.input[0])) |ptr| ptr else return error.input_X_notFound;
        const output_Y = if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[0])) |ptr| ptr else return error.output_Y_notFound;
        const output_indices = if (nodeProto.output.len > 1) if (tensorZant_lib.tensorMap.getPtr(nodeProto.output[1])) |ptr| ptr else return error.output_indices_notFound else null;

        var auto_pad: []const u8 = "NOTSET";
        var ceil_mode: i64 = 0;
        var dilations: ?[]i64 = null;
        var kernel_shape: ?[]i64 = null; //mandatory
        var pads: ?[]i64 = null;
        var storage_order: i64 = 0;
        var strides: ?[]i64 = null;

        for (nodeProto.attribute) |attr| {
            if (std.mem.indexOf(u8, attr.name, "auto_pad")) |_| {
                if (attr.type == onnx.AttributeType.STRING) auto_pad = attr.s else return error.MaxPoolAuto_padNotSTRING;
            } else if (std.mem.indexOf(u8, attr.name, "ceil_mode")) |_| {
                if (attr.type == onnx.AttributeType.INT) ceil_mode = attr.i else return error.MaxPoolCeil_modeNotINT;
            } else if (std.mem.indexOf(u8, attr.name, "dilations")) |_| {
                if (attr.type == onnx.AttributeType.INTS) dilations = attr.ints else return error.MaxPoolDilatationNoINTS;
            } else if (std.mem.indexOf(u8, attr.name, "kernel_shape")) |_| {
                if (attr.type == onnx.AttributeType.INTS) kernel_shape = attr.ints else return error.MaxPoolKernelShapeNotINTS;
            } else if (std.mem.indexOf(u8, attr.name, "pads")) |_| {
                if (attr.type == onnx.AttributeType.INTS) pads = attr.ints else return error.MaxPoolPadsNotINTS;
            } else if (std.mem.indexOf(u8, attr.name, "storage_order")) |_| {
                if (attr.type == onnx.AttributeType.INT) storage_order = attr.i else return error.MaxPoolStorage_orderNotINT;
            } else if (std.mem.indexOf(u8, attr.name, "strides")) |_| {
                if (attr.type == onnx.AttributeType.INTS) strides = attr.ints else return error.MaxPoolStridesNotINTS;
            }
        }

        if (dilations == null) {
            dilations = try allocator.alloc(i64, kernel_shape.?.len);
            @memset(dilations.?, 1);
        }

        //set the output type:
        if (output_Y.ty == tensorZant_lib.TensorType.undefined) output_Y.ty = input_X.ty;

        return MaxPool{
            .input_X = input_X,
            .output_Y = output_Y,
            .output_indices = output_indices,
            .auto_pad = auto_pad,
            .ceil_mode = ceil_mode,
            .dilations = dilations,
            .kernel_shape = kernel_shape,
            .pads = pads,
            .storage_order = storage_order,
            .strides = strides,
        };
    }

    pub fn get_output_shape(self: MaxPool) []usize { // TODO
        return self.output_Y.getShape();
    }

    pub fn get_input_tensors(self: MaxPool) ![]*TensorZant {
        var inputs = std.ArrayList(*TensorZant).init(allocator);
        defer inputs.deinit();
        try inputs.append(self.input_X);
        return inputs.toOwnedSlice();
    }

    pub fn get_output_tensors(self: MaxPool) ![]*TensorZant {
        var outputs = std.ArrayList(*TensorZant).init(allocator);
        defer outputs.deinit();
        try outputs.append(self.output_Y);
        if (self.output_indices) |indices| {
            try outputs.append(indices);
        }
        return outputs.toOwnedSlice();
    }

    pub fn write_op(self: MaxPool, writer: std.fs.File.Writer) !void {
        //input_X string equivalent
        var tensor_X_string: []u8 = undefined;
        defer allocator.free(tensor_X_string);

        if (self.input_X.tc == TensorCategory.INITIALIZER) {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "@constCast(&param_lib.tensor_",
                try utils.getSanitizedName(self.input_X.name),
                ")",
            });
        } else {
            tensor_X_string = try std.mem.concat(allocator, u8, &[_][]const u8{
                "&tensor_",
                try utils.getSanitizedName(self.input_X.name),
            });
        }

        // kernel_shape string equivalent
        var kernel_shape_string: []const u8 = undefined;
        if (self.kernel_shape != null) {
            kernel_shape_string = try utils.i64SliceToUsizeArrayString(self.kernel_shape.?);
        } else {
            return error.Kernel_shapeNotFound;
        }

        // strides string equivalent
        var strides_string: []const u8 = undefined;
        if (self.strides != null) {
            strides_string = try utils.i64SliceToUsizeArrayString(self.strides.?);
        } else {
            return error.StridesNotFound;
        }

        // dilations string equivalent
        var dilations_string: []const u8 = undefined;
        if (self.dilations != null) {
            dilations_string = try utils.i64SliceToUsizeArrayString(self.dilations.?);
        } else {
            dilations_string = try utils.i64SliceToUsizeArrayString(&[_]i64{ 1, 1 }); // TODO: Hardcoded in 4D, not the most elegant solution
        }

        // pads string equivalent
        var pads_string: []const u8 = undefined;
        if (self.pads != null) {
            pads_string = try utils.i64SliceToUsizeArrayString(self.pads.?);
        } else {
            return error.PadsNotFound;
        }

        _ = try writer.print(
            \\
            \\
            \\    tensMath.onnx_maxpool_lean(
            \\        {s},
            \\        {s}, //Input
            \\        &tensor_{s}, //Output
            \\        {s}, //kernel_shape
            \\        {s}, //strides
            \\        {s}, //dilations
            \\        {s}, //pads
            \\        tensMath.op_maxPool.AutoPadType.{s}, //auto_pad
            \\    ) catch return -1;
        , .{
            self.output_Y.ty.toString(),
            tensor_X_string, //Input
            try utils.getSanitizedName(self.output_Y.name), //Output
            kernel_shape_string, //kernel_shape
            strides_string, //strides
            dilations_string, //dilatations
            pads_string, //pads
            self.auto_pad, //auto_pad
        });
    }

    pub fn compute_output_shape(self: MaxPool) ![]usize {
        // Use proper ONNX MaxPool shape calculation with ceil_mode and padding support
        const kernel_shape = try utils.i64SliceToUsizeSlice(self.kernel_shape.?);
        const strides = if (self.strides) |s| try utils.i64SliceToUsizeSlice(s) else &[_]usize{1} ** kernel_shape.len;
        const dilations = if (self.dilations) |d| try utils.i64SliceToUsizeSlice(d) else &[_]usize{1} ** kernel_shape.len;
        const pads = if (self.pads) |p| try utils.i64SliceToUsizeSlice(p) else &[_]usize{0} ** (kernel_shape.len * 2);
        const ceil_mode = self.ceil_mode != 0;

        const output_shape = try tensorMath.op_maxPool.get_onnx_maxpool_output_shape(
            self.input_X.shape,
            kernel_shape,
            strides,
            dilations,
            pads,
            tensorMath.op_maxPool.AutoPadType.NOTSET,
            ceil_mode,
        );

        self.output_Y.shape = output_shape;
        return output_shape;
    }

    pub fn print(self: MaxPool) void { // TODO
        std.debug.print("\n AveragePool:\n {any}", .{self});
    }

    pub fn sobstitute_tensors(self: *MaxPool, old_tensor: *TensorZant, new_tensor: *TensorZant) !void {
        if (self.input_X == old_tensor) {
            self.input_X = new_tensor;
            return;
        }
        if (self.output_Y == old_tensor) {
            self.output_Y = new_tensor;
            return;
        }
        if (self.output_indices != null and self.output_indices.? == old_tensor) {
            self.output_indices = new_tensor;
            return;
        }
        return error.TensorNotFound;
    }

    pub fn render_lower(self: MaxPool, builder: *UOpBuilder) !void {
        const X_id = self.input_X.get_tensorZantID();
        const out_id = self.output_Y.get_tensorZantID();
        const out_shape = self.get_output_shape();
        const in_stride = self.input_X.stride;
        const pads = [2]usize{ @as(usize, @intCast(self.pads.?[0])), @as(usize, @intCast(self.pads.?[1])) };
        const strides_hw = [2]usize{ @as(usize, @intCast(self.strides.?[0])), @as(usize, @intCast(self.strides.?[1])) };

        const dil_hw = if (self.dilations) |d|
            [2]usize{ @as(usize, @intCast(d[0])), @as(usize, @intCast(d[1])) }
        else
            [2]usize{ 1, 1 }; // Default dilation is 1 for both dimensions

        const kHW = [2]usize{ @as(usize, @intCast(self.kernel_shape.?[0])), @as(usize, @intCast(self.kernel_shape.?[1])) };
        const out_dtype = utils.tensorTypeToDtype(self.output_Y.ty);
        const ceil_mode = if (self.ceil_mode != 0)
            true
        else
            false;

        lowerMaxPool2d(
            builder,
            X_id,
            out_id,
            out_shape,
            in_stride,
            pads,
            strides_hw,
            dil_hw,
            kHW,
            out_dtype,
            ceil_mode,
        );
    }

    //--------------------------------------------------------------------------
    // lowerMaxPool2d — translate an ONNX *MaxPool-2D* node into UOps.
    //
    // HOST-SIDE PRE-PASS  (everything happens **before** this function is called)
    // ──────────────────────────────────────────────────────────────────────────
    // • Shapes
    //     X : (N, C, H, W)                    kernel : (kH, kW)
    // • Attributes
    //     pads[4]        = {padT, padL, padB, padR}   (default 0)
    //     strides[2]     = {strH, strW}               (default 1)
    //     dilations[2]   = {dilH, dilW}               (default 1)
    //     ceil_mode      = 0 | 1
    //
    // • Output spatial size  OH × OW
    //     if ceil_mode==0   OH = ⌊(H+padT+padB − dilH·(kH−1) − 1)/strH⌋ + 1
    //                        OW = ⌊(W+padL+padR − dilW·(kW−1) − 1)/strW⌋ + 1
    //     else               use ⌈…⌉ instead of ⌊…⌋
    //
    // • Stride vector for X   (row-major NCHW, elements)
    //       in_stride = [C·H·W,  H·W,  W, 1]
    //
    // • DType promotion (ONNX rules) → `out_dtype`
    //
    // The host passes: `out_shape=[N,C,OH,OW]`, `in_stride`, all pads/strides/
    // dilations, kernel size, and `out_dtype`.  The IR below is then *fully
    // static*: no dynamic shape logic remains.
    //--------------------------------------------------------------------------
    pub fn lowerMaxPool2d(
        b: *UOpBuilder,
        X_id: usize, // input tensor X
        out_id: usize,
        out_shape: []const usize, // [N, C, OH, OW]
        in_stride: []const usize, // X strides (len 4)
        pads: [2]usize, // {padT, padL}
        strides_hw: [2]usize, // {strideH, strideW}
        dil_hw: [2]usize, // {dilH, dilW}
        kHW: [2]usize, // {kH, kW}
        out_dtype: DType,
        ceil_mode: bool,
    ) void {
        // ── helpers --------------------------------------------------------
        const H = struct {
            fn rng(bi: *UOpBuilder, end: usize) usize {
                return bi.push(.RANGE, .i32, &.{}, Any{ .loop_bounds = .{ .start = 0, .end = end } });
            }
            fn k(bi: *UOpBuilder, v: usize) usize {
                return bi.push(.CONST, .i32, &.{}, Any{ .int = v });
            }
            fn fneg_inf(bi: *UOpBuilder) usize { // -∞  as float32
                const negative_inf: f32 = -std.math.inf(f32);
                return bi.push(.CONST, .f32, &.{}, Any{ .float = negative_inf });
            }
        };

        // --- Derive input shape from strides ---
        if (in_stride.len != 4 or in_stride[3] != 1) {
            // Basic validation assuming dense NCHW row-major
            @panic("lowerMaxPool2d expects dense NCHW input strides (..., W, 1)");
        }
        const W_in = @as(usize, @intCast(in_stride[2]));
        const H_in = if (W_in == 0) @panic("Input W cannot be 0") else @as(usize, @intCast(in_stride[1])) / W_in;
        const C_in = if (H_in * W_in == 0) @panic("Input H*W cannot be 0") else @as(usize, @intCast(in_stride[0])) / (H_in * W_in);
        const N_in = out_shape[0]; // Assume input N matches output N
        const derived_in_shape = [_]usize{ N_in, C_in, H_in, W_in };
        // ---------------------------------------

        // --- MOVE VIEW CREATION UP ---
        // Create the VIEW of the input tensor first to ensure its ID is registered.
        _ = b.push(.SHAPE, .i32, &.{X_id}, null); // Keep for potential debug/info
        const id_viewX = b.push(.VIEW, out_dtype, &.{X_id}, Any{
            .view_meta = .{
                .shape = &derived_in_shape, // <<< USE DERIVED INPUT SHAPE
                .strides = in_stride,
            },
        });
        // --- END MOVE ---

        // ── constants ------------------------------------------------------
        // Now create constants. Their IDs will be assigned *after* the VIEW.
        const padT = H.k(b, pads[0]);
        const padL = H.k(b, pads[1]);
        const strH = H.k(b, strides_hw[0]);
        const strW = H.k(b, strides_hw[1]);
        const dilH = H.k(b, dil_hw[0]);
        const dilW = H.k(b, dil_hw[1]);
        const Hlim = H.k(b, H_in); // Use INPUT height for limit
        const Wlim = H.k(b, W_in); // Use INPUT width for limit
        const negInf = H.fneg_inf(b); // -∞ literal

        if (ceil_mode) {
            // TODO: Implement ceil mode adjustment
        }

        // ── view of X ------------------------------------------------------
        // REMOVED: _ = b.push(.SHAPE, .i32, &.{X_id}, null); // debug only -- Already done above
        // REMOVED: const id_viewX = b.push(.VIEW, out_dtype, &.{X_id}, Any{ .view_meta = .{ .shape = &.{ out_shape[0], out_shape[1], out_shape[2], out_shape[3] }, .strides = in_stride } }); -- Already done above

        // output buffer Y

        // ── outer loops n · c · oh · ow ------------------------------------
        const n = H.rng(b, out_shape[0]);
        const c = H.rng(b, out_shape[1]);
        const oh = H.rng(b, out_shape[2]);
        const ow = H.rng(b, out_shape[3]);

        // accumulator seeded to -∞
        const acc = b.push(.DEFINE_ACC, out_dtype, &.{negInf}, null);

        // ── kernel loops  kh · kw  -----------------------------------------
        const kh = H.rng(b, kHW[0]);
        const kw = H.rng(b, kHW[1]);

        // ------ spatial indices ---------------------------------------
        const ih = b.push(.SUB, .i32, &.{ b.push(.ADD, .i32, &.{ b.push(.MUL, .i32, &.{ oh, strH }, null), b.push(.MUL, .i32, &.{ kh, dilH }, null) }, null), padT }, null);

        const iw = b.push(.SUB, .i32, &.{ b.push(.ADD, .i32, &.{ b.push(.MUL, .i32, &.{ ow, strW }, null), b.push(.MUL, .i32, &.{ kw, dilW }, null) }, null), padL }, null);

        // ------ bounds check : 0 <= ih < H  &&  0 <= iw < W --------------
        // Need boolean constants for WHERE
        const bool_true = b.push(.CONST, .bool, &.{}, Any{ .bool = true });
        const bool_false = b.push(.CONST, .bool, &.{}, Any{ .bool = false });

        // Calculate: inside_h = (ih < Hlim) AND (NOT (ih < 0))
        const cond_h_lo = b.push(.CMPLT, .bool, &.{ ih, H.k(b, 0) }, null); // B = (ih < 0)
        const cond_h_hi = b.push(.CMPLT, .bool, &.{ ih, Hlim }, null); // A = (ih < Hlim)
        const not_cond_h_lo = b.push(.WHERE, .bool, &.{ cond_h_lo, bool_false, bool_true }, null); // NOT B
        const inside_h = b.push(.WHERE, .bool, &.{ cond_h_hi, not_cond_h_lo, bool_false }, null); // A AND (NOT B)

        // Calculate: inside_w = (iw < Wlim) AND (NOT (iw < 0))
        const cond_w_lo = b.push(.CMPLT, .bool, &.{ iw, H.k(b, 0) }, null); // B = (iw < 0)
        const cond_w_hi = b.push(.CMPLT, .bool, &.{ iw, Wlim }, null); // A = (iw < Wlim)
        const not_cond_w_lo = b.push(.WHERE, .bool, &.{ cond_w_lo, bool_false, bool_true }, null); // NOT B
        const inside_w = b.push(.WHERE, .bool, &.{ cond_w_hi, not_cond_w_lo, bool_false }, null); // A AND (NOT B)

        // Calculate: in_bounds = inside_h AND inside_w
        const in_bounds = b.push(.WHERE, .bool, &.{ inside_h, inside_w, bool_false }, null); // inside_h AND inside_w

        _ = b.push(.IF, .bool, &.{in_bounds}, null);

        const pX = b.push(.GEP, out_dtype, &.{ id_viewX, n, c, ih, iw }, Any{ .mem_info = .{ .base = id_viewX, .offset = 0, .stride = 1 } });

        const x = b.push(.LOAD, out_dtype, &.{pX}, null);
        _ = b.push(.MAX, out_dtype, &.{ acc, x }, null);

        _ = b.push(.ENDIF, .bool, &.{}, null);

        // end kernel loops
        _ = b.push(.ENDRANGE, .bool, &.{kw}, null);
        _ = b.push(.ENDRANGE, .bool, &.{kh}, null);

        // ── write result ----------------------------------------------------
        const pY = b.push(.GEP, out_dtype, &.{ out_id, n, c, oh, ow }, Any{ .mem_info = .{ .base = out_id, .offset = 0, .stride = 1 } });

        _ = b.push(.STORE, out_dtype, &.{ pY, acc }, null);

        // close outer loops
        _ = b.push(.ENDRANGE, .bool, &.{ow}, null);
        _ = b.push(.ENDRANGE, .bool, &.{oh}, null);
        _ = b.push(.ENDRANGE, .bool, &.{c}, null);
        _ = b.push(.ENDRANGE, .bool, &.{n}, null);
    }
};
