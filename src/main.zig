const std = @import("std");
const IR_zant = @import("IR_zant");
const onnx = IR_zant.onnx;

const codegen = @import("codegen");

const codegen_options = codegen.codegen_options;

const codeGen_tests = codegen.testWriter;

pub fn main() !void {
    std.debug.print("\n\ncodegenOptions: ", .{});
    std.debug.print("\n     model:{s} ", .{codegen_options.model});
    std.debug.print("\n     model_path:{s} ", .{codegen_options.model_path});
    std.debug.print("\n     generated_path:{s} ", .{codegen_options.generated_path});
    std.debug.print("\n     user_tests:{} ", .{codegen_options.user_tests});
    std.debug.print("\n     log:{} ", .{codegen_options.log});
    std.debug.print("\n     type:{s} ", .{codegen_options.type});
    std.debug.print("\n     output_type:{s} ", .{codegen_options.output_type});
    std.debug.print("\n     comm:{} ", .{codegen_options.comm});
    std.debug.print("\n     dynamic:{} ", .{codegen_options.dynamic});
    std.debug.print("\n     static_planning:{s} ", .{codegen_options.static_planning});
    std.debug.print("\n     version:{s} ", .{codegen_options.version});
    std.debug.print("\n     branch_and_bound:{d} ", .{codegen_options.branch_and_bound});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const gpa_allocator = gpa.allocator();

    var model = try onnx.parseFromFile(gpa_allocator, codegen_options.model_path);
    defer model.deinit(gpa_allocator);

    //generate the inference library
    try codegen.codeGenerateFromOnnx(codegen_options.model, codegen_options.generated_path, model);

    // Test the generated code
    try codeGen_tests.writeTestFile(codegen_options.model, codegen_options.generated_path);
}
