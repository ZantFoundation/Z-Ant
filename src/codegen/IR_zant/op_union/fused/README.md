# Fused Operators

This directory contains both the **fusion engine** (pattern detection and graph rewriting) and the **fused operator implementations** (one subfolder per fused op). Fused operators are registered as variants in `Op_union` and triggered by the `-Dfuse` build flag.

## Directory Structure

```
fused/
  fused.zig                                    # barrel file (like operators.zig)
  pattern_matcher.zig                          # graph walker: detect/fuse/substitute
  pattern_collection.zig                       # registry of active fusion patterns
  fused_conv_relu/
    fused_Conv_Relu.zig                        # operator struct + fusion protocol
    zant_conv_relu.zig                         # fused math kernel (conv+relu inline)
    conv_relu_test.zig                         # math tests
  fused_conv_clip/
    fused_Conv_Clip.zig                        # operator struct + fusion protocol
    zant_conv_clip.zig                         # fused math kernel (conv+clip inline)
    conv_clip_test.zig                         # math tests
  fused_dequant_clip_quant/
    fused_Dequant_Clip_Quant.zig               # operator struct + fusion protocol
    zant_clip_quantized.zig                    # fused math kernel (clip in quantized domain)
  fused_pad_conv/
    fused_Pad_Conv.zig                         # absorbs Pad into Conv padding attribute
  fused_quant_dequant/
    fused_Quant_Dequant.zig                    # elimination: Quant->Dequant is a no-op
  fused_dequant_quant/
    fused_Dequant_Quant.zig                    # elimination: Dequant->Quant is a no-op
  fused_2dequant_add_quant/
    fused_2Dequant_Add_Quant.zig               # delegates to QLinearAdd
  fused_dequant_pad_quant_qlinconv/
    fused_Dequant_Pad_Quant_QLinConv.zig       # delegates to QLinearConv with combined pads
  fused_conv_sigmoid_mul/
    fused_Conv_Sigmoid_Mul.zig                 # attention gate pattern (not registered)
```

## Index

| Folder | Pattern | Fuses Into | Has Math Kernel |
|--------|---------|-----------|-----------------|
| `fused_conv_relu/` | Conv -> Relu | `conv_relu_lean` kernel | Yes |
| `fused_conv_clip/` | Conv -> Clip | `conv_clip_lean` kernel | Yes |
| `fused_dequant_clip_quant/` | Dequant -> Clip -> Quant | `clip_quantized_lean` | Yes |
| `fused_pad_conv/` | Pad -> Conv | Conv (combined pads) | No |
| `fused_2dequant_add_quant/` | 2x Dequant -> Add -> Quant | QLinearAdd | No |
| `fused_dequant_pad_quant_qlinconv/` | Dequant -> Pad -> Quant -> QLinConv | QLinearConv | No |
| `fused_quant_dequant/` | Quant -> Dequant | *eliminated* | No |
| `fused_dequant_quant/` | Dequant -> Quant | *eliminated* | No |
| `fused_conv_sigmoid_mul/` | Conv -> (Sigmoid + Mul) | Fused_Conv_Sigmoid_Mul | No (not registered) |

## File Convention

Each fused operator folder follows the same pattern as `operators/`:

```
fused_<name>/
  fused_<Name>.zig        # operator struct + fusion protocol (detection/fusion/substitution)
  zant_<name>.zig          # fused math kernel (if applicable)
  <name>_test.zig          # math tests + fusion detection tests (if applicable)
```

### Operator struct interface

```zig
pub const Fused_X_Y = struct {
    // --- Fusion protocol (called by pattern_matcher.zig) ---
    pub fn init_fused_op(fusion_list: ArrayList(*NodeZant)) !Fused_X_Y;
    pub fn fn_pattern_detection(graph: *GraphZant, root_node: *NodeZant) !?ArrayList(*NodeZant);
    pub fn fn_pattern_fusion(graph: *GraphZant, node_list: ArrayList(*NodeZant)) !NodeZant;
    pub fn fn_pattern_sobstitution(graph: *GraphZant, fused_node: *NodeZant, node_list: ArrayList(*NodeZant)) !void;

    // --- Op_union interface (called by codegen) ---
    pub fn get_input_tensors(self) ![]*TensorZant;
    pub fn get_output_tensors(self) ![]*TensorZant;
    pub fn get_output_shape(self) []usize;
    pub fn compute_output_shape(self) []usize;
    pub fn write_op(self, writer) !void;
    pub fn print(self) void;
    pub fn sobstitute_tensors(self, old, new) !void;
};
```

## Registration

To activate a fused operator:

1. Export it from `fused.zig` (the barrel file in this directory).
2. Add a variant to `Op_union` in `../op_union.zig`.
3. Add a `PatternConfig` entry in `pattern_collection.zig` (in this directory).

## Known Bugs

Critical bugs have been fixed. Remaining issues:

**High** (memory leaks):
- `fused_2Dequant_Add_Quant`: `get_predecessors()` ArrayLists never freed in substitution.
- `fused_Quant_Dequant` / `fused_Dequant_Quant`: Same leak + index mismatch in substitution.

**Medium** (comments/naming):
- `pattern_collection.zig:13`: Comment says "Pad" but pattern is "Clip".
- `fused_Conv_Relu.zig:28`: Error `WrongOpAtPose2` should be `WrongOpAtPos1`.
- `fused_Conv_Clip.zig:41`: Same typo.
- `fused_Conv_Clip.zig:410`: Comments say "DequantizeLinear"/"QLinearConv" but nodes are Conv/Clip.
