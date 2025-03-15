//! The aim of this file is that, given a onnx oprator (https://onnx.ai/onnx/operators/),
//! it returns the onnx.ModelProto containing only one node for the operation

const std = @import("std");
const zant = @import("zant");
const pkgAllocator = zant.utils.allocator;
const allocator = pkgAllocator.allocator;

const onnx = zant.onnx;
const codeGen = @import("codegen");

pub fn oneOpModelsCodegen() !void {
    std.debug.print("\n     test: OneOptest", .{});

    std.debug.print("\n     opening available_operations...", .{});
    const op_file = try std.fs.cwd().openFile("tests/CodeGen/Python-ONNX/available_operations.txt", .{});
    defer op_file.close();
    std.debug.print(" done", .{});

    const file_size = try op_file.getEndPos();
    const buffer = try allocator.alloc(u8, @intCast(file_size));
    //defer allocator.free(buffer);

    const bytes_read = try op_file.readAll(buffer);
    if (bytes_read != file_size) {
        return error.UnexpectedEOF;
    }

    // Read the available operations from file.
    const ops_data = buffer[0..file_size];
    defer allocator.free(ops_data);

    // Split file into lines.
    var lines_iter = std.mem.splitAny(u8, ops_data, "\n");

    var modelList = std.ArrayList(onnx.ModelProto).init(allocator);
    defer modelList.deinit();

    while (true) {
        // Get the next line from the iterator.
        const maybe_line = lines_iter.next();

        if (maybe_line) |ml| std.debug.print("maybe_line: {any}\n", .{ml}) else {
            std.debug.print("maybe_line: null -----> break\n", .{});
            break;
        }

        const raw_line = maybe_line.?;
        // Trim whitespace from the line.
        const trimmed_line = std.mem.trim(u8, raw_line, " \t\r\n");
        if (trimmed_line.len > 0) {
            std.debug.print("Operation: {s}\n", .{trimmed_line});
        }

        // Construct the model file path: "Phython-ONNX/{op}_0.onnx"
        const model_path = try std.fmt.allocPrint(allocator, "tests/CodeGen/Python-ONNX/{s}_0.onnx", .{trimmed_line});
        defer allocator.free(model_path);
        std.debug.print("model_path : {s}", .{model_path});

        // Load the model.
        var model = try onnx.parseFromFile(allocator, model_path);
        defer model.deinit(allocator);

        //Printing the model:
        //DEBUG
        model.print();

        try modelList.append(model);

        std.debug.print("\n CODEGENERATING {s} ...", .{model_path});

        // Create the generated model directory if not present
        const generated_path = try std.fmt.allocPrint(allocator, "generated/oneOpModels/{s}/", .{trimmed_line});
        //const generated_path = "src/codeGen/";
        try std.fs.cwd().makePath(generated_path);

        // ONNX model parsing
        try codeGen.globals.setGlobalAttributes(model);

        // Create the code for the model
        try codeGen.skeleton.writeZigFile(trimmed_line, generated_path, model);
    }
}

// //------------operation structs
// //Add
// pub const AddStruct = struct {
//     name: []const u8,
//     inputShape: []i64,
//     //other...
// };

// //Concat
// pub const ConcatStruct = struct {
//     name: []const u8,
//     inputShape: []i64,
//     //other...
// };

// //Conv
// pub const ConvStruct = struct {
//     name: []const u8,
//     inputShape: []i64,
//     //other...
// };

// //------------union of all the operation struct
// pub const OpStruct = union(enum) {
//     Add: AddStruct,
//     Concat: ConcatStruct,
//     Conv: ConvStruct,
//     Default: AddStruct, // delete this, just for DEBUG
// };

// pub fn oneOpModel(
//     op_struct: OpStruct,
// ) !onnx.ModelProto {

//     // ---------- GRAPH ATTRIBUTES INITIALIZATION ----------
//     var graph_nodes = std.ArrayList(*onnx.NodeProto).init(allocator);
//     defer graph_nodes.deinit();
//     var graph_initializers = std.ArrayList(*onnx.TensorProto).init(allocator);
//     defer graph_initializers.deinit();
//     var graph_inputs = std.ArrayList(*onnx.ValueInfoProto).init(allocator);
//     defer graph_inputs.deinit();
//     var graph_outputs = std.ArrayList(*onnx.ValueInfoProto).init(allocator);
//     defer graph_outputs.deinit();
//     var graph_value_infos = std.ArrayList(*onnx.ValueInfoProto).init(allocator);
//     defer graph_value_infos.deinit();

//     // ---------- NODE ATTRIBUTES INITIALIZATION ----------
//     var node_inputs = std.ArrayList([]const u8).init(allocator);
//     defer node_inputs.deinit();
//     var node_outputs = std.ArrayList([]const u8).init(allocator);
//     defer node_outputs.deinit();
//     var node_attributes = std.ArrayList(*onnx.AttributeProto).init(allocator);
//     defer node_attributes.deinit();

//     const node_ptr = try allocator.create(onnx.NodeProto);
//     node_ptr.* = onnx.NodeProto{
//         .name = null, // ?[]const u8,
//         .op_type = undefined, //[]const u8,
//         .domain = null, // ?[]const u8,
//         .input = undefined, //[][]const u8,
//         .output = undefined, // [][]const u8,
//         .attribute = undefined, // []*AttributeProto,
//     };

//     switch (op_struct) {
//         .Add => |details| {
//             node_ptr.*.name = try std.mem.concat(allocator, u8, &[_][]const u8{
//                 "node_",
//                 details.name,
//             });
//             node_ptr.*.op_type = try allocator.dupe(u8, "Add");
//             try node_inputs.append(try allocator.dupe(u8, "input_Add"));
//             try node_inputs.append(try allocator.dupe(u8, "weight_Add"));

//             try node_outputs.append(try allocator.dupe(u8, "output_Add"));

//             //attributes: TODO !! moove dims, data ecc.. into AddStruct
//             //weight matix
//             const weight_ptr = try allocator.create(onnx.TensorProto);
//             const dims: [4]i64 = [_]i64{ 1, 1, 2, 2 };
//             const data: [4]i32 = [_]i32{ 10, 20, 30, 40 };
//             weight_ptr.* = onnx.TensorProto{
//                 .dims = try allocator.dupe(i64, &dims),
//                 .data_type = onnx.DataType.INT32,
//                 .name = try allocator.dupe(u8, "weight_Add"),
//                 .raw_data = null,
//                 .float_data = null,
//                 .int32_data = try allocator.dupe(i32, &data),
//                 .string_data = null,
//                 .int64_data = null,
//                 .double_data = null,
//                 .uint64_data = null,
//             };
//             try graph_initializers.append(weight_ptr);

//             //ValueInfoProto grraph inputs initialization
//             // var input_info_ptr = try allocator.create(onnx.ValueInfoProto);
//             // input_info_ptr.* = onnx.ValueInfoProto{
//             //     .name = try allocator.dupe(u8, "input_Add"),
//             //     .type =
//             // };
//             // try graph_inputs.append()
//         },
//         .Concat => |details| {
//             node_ptr.*.name = details.name;
//             node_ptr.*.op_type = "Concat";
//             try node_inputs.append("input_Concat");
//             try node_outputs.append("output_Concat");
//         },
//         .Conv => |details| {
//             node_ptr.*.name = details.name;
//             node_ptr.*.op_type = "Conv";
//             try node_inputs.append("input_Conv");
//             try node_outputs.append("output_Conv");
//         },
//         else => return error.OpNotAvalilable,
//     }

//     //completing and append the only node of the graph
//     node_ptr.*.input = try node_inputs.toOwnedSlice();
//     node_ptr.*.output = try node_outputs.toOwnedSlice();
//     node_ptr.*.attribute = try node_attributes.toOwnedSlice();
//     try graph_nodes.append(node_ptr);

//     //graph creation
//     const graph_ptr = try allocator.create(onnx.GraphProto);
//     graph_ptr.* = onnx.GraphProto{
//         .name = try allocator.dupe(u8, "graph_Add"), //?[]const u8,
//         .nodes = try graph_nodes.toOwnedSlice(), // []*NodeProto,
//         .initializers = try graph_initializers.toOwnedSlice(), // []*TensorProto,
//         .inputs = try graph_inputs.toOwnedSlice(), // []*ValueInfoProto,
//         .outputs = try graph_outputs.toOwnedSlice(), // []*ValueInfoProto,
//         .value_info = try graph_value_infos.toOwnedSlice(), // []*ValueInfoProto,
//     };

//     return onnx.ModelProto{
//         .ir_version = onnx.Version.IR_VERSION,
//         .producer_name = null,
//         .producer_version = null,
//         .domain = null,
//         .model_version = null,
//         .doc_string = null,
//         .graph = graph_ptr,
//     };
// }
