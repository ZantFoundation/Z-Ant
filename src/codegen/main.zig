const std = @import("std");
const zant = @import("zant");
const onnx = zant.onnx;

const codegen = @import("codegen");
const codegen_options = codegen.codegen_options;

// called by "zig build IR_codegen" optionals:" -Dlog -Dmodel="name" -D ..." see build.zig"
pub fn main() !void {
    std.debug.print("\n\ncodegenOptions: ", .{});
    std.debug.print("\n     model:{s} ", .{codegen_options.model});
    std.debug.print("\n     model_path:{s} ", .{codegen_options.model_path});
    std.debug.print("\n     generated_path:{s} ", .{codegen_options.generated_path});
    std.debug.print("\n     user_tests:{} ", .{codegen_options.user_tests});
    std.debug.print("\n     log:{} ", .{codegen_options.log});
    std.debug.print("\n     shape:{s} ", .{codegen_options.shape});
    std.debug.print("\n     type:{s} ", .{codegen_options.type});
    std.debug.print("\n     output_type:{s} ", .{codegen_options.output_type});
    std.debug.print("\n     comm:{} ", .{codegen_options.comm});
    std.debug.print("\n     dynamic:{} ", .{codegen_options.dynamic});
    std.debug.print("\n     version:{s} ", .{codegen_options.version});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const gpa_allocator = gpa.allocator();

    const model_name = codegen_options.model;
    const model_path = try std.fmt.allocPrint(gpa_allocator, "datasets/models/{s}/{s}.onnx", .{ model_name, model_name });
    defer gpa_allocator.free(model_path);

    var model = try onnx.parseFromFile(gpa_allocator, model_path);
    defer model.deinit(gpa_allocator);

    if (std.mem.eql(u8, codegen_options.version, "v1")) {
        try codegen.codegen_v1_exe.main_v1(model);
    }

    // if (std.mem.eql(u8, codegen_options.version, "v2")) {
    //     try codegen.codegen_v2_exe.main_v2(&model);
    // }

    // model.print(); // << ---------- USEFUL FOR DEBUG
    // try onnx.printModelDetails(&model); // << ---------- USEFUL FOR DEBUG

}
