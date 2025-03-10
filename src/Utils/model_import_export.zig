const std = @import("std");
const cwd = std.fs.cwd();
const zant = @import("../zant.zig");
const Model = zant.model.Model;
//------------layer libraries
const Layer = zant.model.layer;
const DenseLayer = Layer.DenseLayer;
const ActivationLayer = Layer.ActivationLayer;
const ConvolutionalLayer = Layer.ConvolutionalLayer;
const FlattenLayer = Layer.FlattenLayer;
const PoolingLayer = Layer.PoolingLayer;

const LayerType = Layer.LayerType;
const Tensor = zant.core.tensor.Tensor;
const ActivationType = Layer.ActivationType;
const PoolingType = Layer.poolingLayer.PoolingType;

// ----------------------------------------------------------------------------
// ---------------------------------- Export ----------------------------------
// ----------------------------------------------------------------------------
pub fn exportModel(
    comptime T: type,
    model: Model(T),
    file_path: []const u8,
) !void {
    print_exporting();
    //std.debug.print("\n ..... EXPORTING THE MODEL ......", .{});
    var file = try std.fs.cwd().createFile(file_path, .{});
    std.debug.print("\n .......... file created, path:{s}", .{file_path});
    defer file.close();

    const writer = file.writer();
    std.debug.print("\n .......... writer created ", .{});

    try writer.writeInt(usize, model.layers.items.len, std.builtin.Endian.big);
    for (model.layers.items) |*l| {
        try exportLayer(T, l.*, writer);
    }
    return;
}

pub fn exportLayer(
    comptime T: type,
    layer: Layer.Layer(T),
    writer: std.fs.File.Writer,
) !void {
    std.debug.print("\n .......... export layer :  ", .{});

    //TODO: handle Default layer and null layer
    if (layer.layer_type == LayerType.DenseLayer) { // ------ EXPORT DENSE
        _ = try writer.write("Dense....."); //see #Tags in `import_export_guide.md`
        const denseLayer: *DenseLayer(T) = @alignCast(@ptrCast(layer.layer_ptr));
        try exportLayerDense(T, denseLayer.*, writer);
    } else if (layer.layer_type == LayerType.ActivationLayer) { // ------ EXPORT ACTIVATION
        _ = try writer.write("Activation"); //see #Tags in `import_export_guide.md`
        const activationLayer: *ActivationLayer(T) = @alignCast(@ptrCast(layer.layer_ptr));
        try exportLayerActivation(T, activationLayer.*, writer);
    } else if (layer.layer_type == LayerType.ConvolutionalLayer) { // ------ EXPORT CONVOLUTIONAL
        _ = try writer.write("Convol...."); //see #Tags in `import_export_guide.md`
        const convLayer: *ConvolutionalLayer(T) = @alignCast(@ptrCast(layer.layer_ptr));
        try exportLayerConvolutional(T, convLayer.*, writer);
    } else if (layer.layer_type == LayerType.FlattenLayer) { // ------ EXPORT FLATTEN
        _ = try writer.write("Flatten..."); //see #Tags in `import_export_guide.md`
        //OSS!! There is nothing to export. We just need to know that it exists to put it in the model
        try exportLayerFlatten();
    } else if (layer.layer_type == LayerType.PoolingLayer) { // ------ EXPORT POOLING
        _ = try writer.write("Pooling..."); //see #Tags in `import_export_guide.md`
        const poolLayer: *PoolingLayer(T) = @alignCast(@ptrCast(layer.layer_ptr));
        try exportLayerPooling(T, poolLayer.*, writer);
    }
}

pub fn exportLayerDense(
    comptime T: type,
    layer: DenseLayer(T),
    writer: std.fs.File.Writer,
) !void {
    std.debug.print(" dense ", .{});

    try exportTensor(T, layer.weights, writer);
    try exportTensor(T, layer.bias, writer);
    try writer.writeInt(usize, layer.n_inputs, std.builtin.Endian.big);
    try writer.writeInt(usize, layer.n_neurons, std.builtin.Endian.big);
    //try exportTensor(T, layer.w_gradients, writer);
    //try exportTensor(T, layer.b_gradients, writer);
}

pub fn exportLayerConvolutional(
    comptime T: type,
    layer: ConvolutionalLayer(T),
    writer: std.fs.File.Writer,
) !void {
    std.debug.print(" convolutional ", .{});

    try exportTensor(T, layer.weights, writer);
    try exportTensor(T, layer.bias, writer);
    try writer.writeInt(usize, layer.input_channels, std.builtin.Endian.big);
    try writer.writeInt(usize, layer.kernel_shape[0], std.builtin.Endian.big);
    try writer.writeInt(usize, layer.kernel_shape[1], std.builtin.Endian.big);
    try writer.writeInt(usize, layer.kernel_shape[2], std.builtin.Endian.big);
    try writer.writeInt(usize, layer.kernel_shape[3], std.builtin.Endian.big);
    try writer.writeInt(usize, layer.stride[0], std.builtin.Endian.big);
    try writer.writeInt(usize, layer.stride[1], std.builtin.Endian.big);
}

pub fn exportLayerFlatten() !void {
    std.debug.print(" flatten ", .{});
}

pub fn exportLayerActivation(
    comptime T: type,
    layer: ActivationLayer(T),
    writer: std.fs.File.Writer,
) !void {
    std.debug.print(" activation ", .{});

    try writer.writeInt(usize, layer.n_inputs, std.builtin.Endian.big);
    try writer.writeInt(usize, layer.n_neurons, std.builtin.Endian.big);

    if (layer.activationFunction == ActivationType.ReLU) {
        _ = try writer.write("ReLU......");
    } else if (layer.activationFunction == ActivationType.Sigmoid) {
        _ = try writer.write("Sigmoid...");
    } else if (layer.activationFunction == ActivationType.Softmax) {
        _ = try writer.write("Softmax...");
    } else if (layer.activationFunction == ActivationType.None) {
        _ = try writer.write("None......");
    } else {
        return error.ImpossibleActivationgType;
    }
}

pub fn exportLayerPooling(
    comptime T: type,
    layer: PoolingLayer(T),
    writer: std.fs.File.Writer,
) !void {
    std.debug.print(" pooling ", .{});

    try writer.writeInt(usize, layer.kernel[0], std.builtin.Endian.big);
    try writer.writeInt(usize, layer.kernel[1], std.builtin.Endian.big);

    try writer.writeInt(usize, layer.stride[0], std.builtin.Endian.big);
    try writer.writeInt(usize, layer.stride[1], std.builtin.Endian.big);

    if (layer.poolingType == PoolingType.Max) {
        _ = try writer.write("Max");
    } else if (layer.poolingType == PoolingType.Min) {
        _ = try writer.write("Min");
    } else if (layer.poolingType == PoolingType.Avg) {
        _ = try writer.write("Avg");
    } else {
        return error.ImpossiblePoolingType;
    }
}

pub fn exportTensor(
    comptime T: type,
    tensor: Tensor(T),
    writer: std.fs.File.Writer,
) !void {
    std.debug.print("\n .......... exporting tensor", .{});

    // Write tensor size and shape
    try writer.writeInt(usize, tensor.size, std.builtin.Endian.big);
    try writer.writeInt(usize, tensor.shape.len, std.builtin.Endian.big);
    for (tensor.shape) |dim| {
        try writer.writeInt(usize, dim, std.builtin.Endian.big);
    }

    // Write tensor data
    for (tensor.data) |value| {
        try writeNumber(T, value, writer);
    }
}

pub fn writeNumber(
    comptime T: type,
    number: T,
    writer: std.fs.File.Writer,
) !void {
    const size = @sizeOf(T);
    var buffer: [size]u8 = @bitCast(number);
    try writer.writeAll(&buffer);
}

// ----------------------------------------------------------------------------
// ---------------------------------- Import ----------------------------------
// ----------------------------------------------------------------------------

pub fn importModel(
    comptime T: type,
    comptime allocator: *const std.mem.Allocator,
    file_path: []const u8,
) !Model(T) {
    //std.debug.print("\n ..... IMPORTING THE MODEL ......", .{});
    print_import();

    var file = try std.fs.cwd().openFile(file_path, .{});
    std.debug.print("\n .......... file created, path:{s}", .{file_path});

    defer file.close();
    const reader = file.reader();
    std.debug.print("\n .......... reader created ", .{});

    var model: Model(T) = Model(T){
        .layers = undefined,
        .allocator = allocator,
        .input_tensor = undefined,
    };
    try model.init();

    const n_layers = try reader.readInt(usize, std.builtin.Endian.big);
    for (0..n_layers) |_| {
        const newLayer: Layer.Layer(T) = try importLayer(T, allocator, reader);

        try model.addLayer(newLayer);
    }
    return model;
}

pub fn importLayer(
    comptime T: type,
    comptime allocator: *const std.mem.Allocator,
    reader: std.fs.File.Reader,
) !Layer.Layer(T) {
    std.debug.print("\n .......... import layer : ", .{});

    var layer_type_string: [10]u8 = undefined;
    _ = try reader.read(&layer_type_string);
    std.debug.print("{s}", .{layer_type_string});

    //TODO: handle Default layer and null layer
    if (std.mem.eql(u8, &layer_type_string, "Dense.....")) {
        const denseLayerPtr = try allocator.create(DenseLayer(T));
        denseLayerPtr.* = try importLayerDense(T, allocator, reader);
        // Transfer ownership to the Layer
        const newLayer = DenseLayer(T).create(denseLayerPtr);
        return newLayer;
    } else if (std.mem.eql(u8, &layer_type_string, "Convol....")) {
        const convactivLayerPtr = try allocator.create(ConvolutionalLayer(T));
        convactivLayerPtr.* = try importLayerConvolutional(T, allocator, reader);
        // Transfer ownership to the Layer
        const newLayer = ConvolutionalLayer(T).create(convactivLayerPtr);
        return newLayer;
    } else if (std.mem.eql(u8, &layer_type_string, "Activation")) {
        const activLayerPtr = try allocator.create(ActivationLayer(T));
        activLayerPtr.* = try importLayerActivation(T, allocator, reader);
        // Transfer ownership to the Layer
        const newLayer = ActivationLayer(T).create(activLayerPtr);
        return newLayer;
    } else if (std.mem.eql(u8, &layer_type_string, "Flatten...")) {
        const flattenLayerPtr = try allocator.create(FlattenLayer(T));
        flattenLayerPtr.* = try importLayerFlatten(T, allocator);
        // Transfer ownership to the Layer
        const newLayer = FlattenLayer(T).create(flattenLayerPtr);
        return newLayer;
    } else if (std.mem.eql(u8, &layer_type_string, "Pooling...")) {
        const poolLayerPtr = try allocator.create(PoolingLayer(T));
        poolLayerPtr.* = try importLayerPooling(T, allocator, reader);
        // Transfer ownership to the Layer
        const newLayer = PoolingLayer(T).create(poolLayerPtr);
        return newLayer;
    } else {
        return error.impossibleLayer;
    }
}

pub fn importLayerDense(
    comptime T: type,
    comptime allocator: *const std.mem.Allocator,
    reader: std.fs.File.Reader,
) !DenseLayer(T) {
    std.debug.print(" dense ", .{});

    const weights_tens: Tensor(T) = try importTensor(T, allocator, reader);
    const bias_tens: Tensor(T) = try importTensor(T, allocator, reader);
    const n_inputs = try reader.readInt(usize, std.builtin.Endian.big);
    const n_neurons = try reader.readInt(usize, std.builtin.Endian.big);
    const w_grad_tens = try Tensor(T).fromShape(allocator, weights_tens.shape);
    const b_grad_tens = try Tensor(T).fromShape(allocator, bias_tens.shape);

    return DenseLayer(T){
        .weights = weights_tens,
        .bias = bias_tens,
        .input = undefined,
        .output = undefined,
        .n_inputs = n_inputs,
        .n_neurons = n_neurons,
        .w_gradients = w_grad_tens,
        .b_gradients = b_grad_tens,
        .allocator = allocator,
    };
}

pub fn importLayerConvolutional(
    comptime T: type,
    comptime allocator: *const std.mem.Allocator,
    reader: std.fs.File.Reader,
) !ConvolutionalLayer(T) {
    std.debug.print(" convolutional ", .{});

    const weights_tens: Tensor(T) = try importTensor(T, allocator, reader);
    const bias_tens: Tensor(T) = try importTensor(T, allocator, reader);
    const input_channels = try reader.readInt(usize, std.builtin.Endian.big);

    var kernel_shape: [4]usize = .{ 0, 0, 0, 0 };
    kernel_shape[0] = try reader.readInt(usize, std.builtin.Endian.big);
    kernel_shape[1] = try reader.readInt(usize, std.builtin.Endian.big);
    kernel_shape[2] = try reader.readInt(usize, std.builtin.Endian.big);
    kernel_shape[3] = try reader.readInt(usize, std.builtin.Endian.big);

    var stride: [2]usize = .{ 0, 0 };
    stride[0] = try reader.readInt(usize, std.builtin.Endian.big);
    stride[1] = try reader.readInt(usize, std.builtin.Endian.big);

    const w_grad_tens = try Tensor(T).fromShape(allocator, &kernel_shape);
    const b_grad_tens = try Tensor(T).fromShape(allocator, bias_tens.shape);

    return ConvolutionalLayer(T){
        .weights = weights_tens,
        .bias = bias_tens,
        .input = undefined,
        .output = undefined,
        .input_channels = input_channels,
        .kernel_shape = .{ kernel_shape[0], kernel_shape[1], kernel_shape[2], kernel_shape[3] },
        .stride = .{ stride[0], stride[1] },
        .w_gradients = w_grad_tens,
        .b_gradients = b_grad_tens,
        .allocator = allocator,
    };
}

pub fn importLayerFlatten(
    comptime T: type,
    comptime allocator: *const std.mem.Allocator,
) !FlattenLayer(T) {
    std.debug.print(" flatten ", .{});

    return FlattenLayer(T){
        .input = undefined,
        .output = undefined,
        .allocator = allocator,
        .original_shape = &[_]usize{},
    };
}

pub fn importLayerActivation(
    comptime T: type,
    comptime allocator: *const std.mem.Allocator,
    reader: std.fs.File.Reader,
) !ActivationLayer(T) {
    std.debug.print(" activation ", .{});

    const n_inputs = try reader.readInt(usize, std.builtin.Endian.big);
    const n_neurons = try reader.readInt(usize, std.builtin.Endian.big);

    var activation_type_string: [10]u8 = undefined;
    _ = try reader.read(&activation_type_string);

    var layerActiv = ActivationLayer(T){
        .input = undefined,
        .output = undefined,
        .n_inputs = n_inputs,
        .n_neurons = n_neurons,
        .activationFunction = undefined,
        .allocator = allocator,
    };

    if (std.mem.eql(u8, &activation_type_string, "ReLU......")) {
        layerActiv.activationFunction = ActivationType.ReLU;
    } else if (std.mem.eql(u8, &activation_type_string, "Sigmoid...")) {
        layerActiv.activationFunction = ActivationType.Sigmoid;
    } else if (std.mem.eql(u8, &activation_type_string, "Softmax...")) {
        layerActiv.activationFunction = ActivationType.Softmax;
    } else if (std.mem.eql(u8, &activation_type_string, "None......")) {
        layerActiv.activationFunction = ActivationType.None;
    }

    return layerActiv;
}

pub fn importLayerPooling(
    comptime T: type,
    comptime allocator: *const std.mem.Allocator,
    reader: std.fs.File.Reader,
) !PoolingLayer(T) {
    std.debug.print(" pooling ", .{});

    var kernel: [4]usize = .{ 0, 0, 0, 0 };
    kernel[0] = try reader.readInt(usize, std.builtin.Endian.big);
    kernel[1] = try reader.readInt(usize, std.builtin.Endian.big);

    var stride: [2]usize = .{ 0, 0 };
    stride[0] = try reader.readInt(usize, std.builtin.Endian.big);
    stride[1] = try reader.readInt(usize, std.builtin.Endian.big);

    var pooling_type_string: [3]u8 = undefined;
    _ = try reader.read(&pooling_type_string);

    var layerPooling = PoolingLayer(T){
        .input = undefined,
        .output = undefined,
        .used_input = undefined,
        .kernel = .{ kernel[0], kernel[1] },
        .stride = .{ stride[0], stride[1] },
        .poolingType = undefined,
        .allocator = allocator,
    };

    if (std.mem.eql(u8, &pooling_type_string, "Max")) {
        layerPooling.poolingType = PoolingType.Max;
    } else if (std.mem.eql(u8, &pooling_type_string, "Min")) {
        layerPooling.poolingType = PoolingType.Min;
    } else if (std.mem.eql(u8, &pooling_type_string, "Avg")) {
        layerPooling.poolingType = PoolingType.Avg;
    } else {
        return error.ImpossiblePoolingType;
    }

    return layerPooling;
}

pub fn importTensor(
    comptime T: type,
    allocator: *const std.mem.Allocator,
    reader: std.fs.File.Reader,
) !Tensor(T) {
    std.debug.print("\n ...... importing tensor ", .{});

    // Read tensor size
    const tensor_size: usize = try reader.readInt(usize, std.builtin.Endian.big);

    // Read tensor shape lenght
    const tensor_shapeLen: usize = try reader.readInt(usize, std.builtin.Endian.big);

    // Read tensor shape
    const tensor_shape = try allocator.alloc(usize, tensor_shapeLen);
    for (0..tensor_shapeLen) |i| {
        tensor_shape[i] = try reader.readInt(usize, std.builtin.Endian.big);
    }

    const tensor_data = try allocator.alloc(T, tensor_size);
    // Read tensor data
    for (0..tensor_size) |i| {
        tensor_data[i] = try readNumber(T, reader);
    }

    return Tensor(T){
        .data = tensor_data,
        .size = tensor_size,
        .shape = tensor_shape,
        .allocator = allocator,
        .owns_memory = true,
    };
}

inline fn readNumber(
    comptime T: type,
    reader: std.fs.File.Reader,
) !T {
    const size = @sizeOf(T);
    var buffer: [size]u8 = undefined;
    _ = try reader.readAll(&buffer);
    return @bitCast(buffer);
}

fn print_import() void {
    const import_str =
        \\
        \\    ____                           __  _                   
        \\   /  _/___ ___  ____  ____  _____/ /_(_)___  ____ _       
        \\   / // __ `__ \/ __ \/ __ \/ ___/ __/ / __ \/ __ `/       
        \\ _/ // / / / / / /_/ / /_/ / /  / /_/ / / / / /_/ /  _ _ _ 
        \\/___/_/ /_/ /_/ .___/\____/_/   \__/_/_/ /_/\__, /  (_|_|_)
        \\             /_/                           /____/      
        \\
    ;

    std.debug.print("{s}", .{import_str});
}

fn print_exporting() void {
    const export_str =
        \\
        \\    ______                      __  _                   
        \\   / ____/  ______  ____  _____/ /_(_)___  ____ _       
        \\  / __/ | |/_/ __ \/ __ \/ ___/ __/ / __ \/ __ `/       
        \\ / /____>  </ /_/ / /_/ / /  / /_/ / / / / /_/ /  _ _ _ 
        \\/_____/_/|_/ .___/\____/_/   \__/_/_/ /_/\__, /  (_|_|_)
        \\          /_/                           /____/               
        \\
    ;

    std.debug.print("{s}", .{export_str});
}
