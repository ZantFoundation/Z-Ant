const lib = @import("mod.zig");

const N: usize = 64;
const Q: usize = 4;

var s_norm: [N]f32 = undefined;
var s_cos: [N]f32 = undefined;
var s_sin: [N]f32 = undefined;
var s_sorted: [N]f32 = undefined;
var s_bins: [N]usize = undefined;
var s_matrix: [Q * Q]f32 = undefined;

/// Trasforma un segnale ECG di N=64 campioni nel tensore CHW f32 [3, N, N].
/// output deve puntare a un buffer caller-allocated di 3*64*64 = 12288 float32.
/// Il risultato può essere passato direttamente a predict() come input con shape {1,3,64,64}.
pub export fn zant_timeseries_to_image(
    signal: [*]const f32,
    output: [*]f32,
) void {
    var scratch = lib.compound.CompoundScratch{
        .norm_buf = &s_norm,
        .cosines_buf = &s_cos,
        .sines_buf = &s_sin,
        .sorted_buf = &s_sorted,
        .bins_buf = &s_bins,
        .matrix_buf = &s_matrix,
    };
    lib.compound.lean_compound(
        signal[0..N],
        .MinusOneToOne,
        Q,
        &scratch,
        output[0 .. 3 * N * N],
    );
    for (output[0 .. 3 * N * N]) |*v| v.* = @max(0.0, @min(1.0, v.*));
}
