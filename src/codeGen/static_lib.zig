
 const std = @import("std");
 const zant = @import("zant");
 const Tensor = zant.core.tensor.Tensor;
 const tensMath = zant.core.tensor.math_standard;
 const pkgAllocator = zant.utils.allocator;
 const allocator = pkgAllocator.allocator;
 const codegen = @import("codegen");
 const utils = codegen.utils;
 const param_lib = @import("static_parameters.zig");

var log_function: ?*const fn ([*c]u8) callconv(.C) void = null;

pub export fn setLogFunction(func: ?*const fn ([*c]u8) callconv(.C) void) void {
    log_function = func;
}


 var buf: [4096 * 10]u8 = undefined;
 var fba_state = @import("std").heap.FixedBufferAllocator.init(&buf);
 const fba = fba_state.allocator();
 const T = f32;

 // ---------------------------------------------------
 // +         Initializing output Tensors             +
 // ---------------------------------------------------

var shape_tensor_parameter193_reshape1 : [2]usize = [_]usize{ 256, 10} ;
var array_parameter193_reshape1: [2560]T = [_]T{0} ** 2560;
var tensor_parameter193_reshape1 = Tensor(T).fromConstBuffer( &allocator, &array_parameter193_reshape1, &shape_tensor_parameter193_reshape1);

var shape_tensor_convolution28_output_0 : [4]usize = [_]usize{ 1, 8, 28, 28} ;
var array_convolution28_output_0: [6272]T = [_]T{0} ** 6272;
var tensor_convolution28_output_0 = Tensor(T).fromConstBuffer( &allocator, &array_convolution28_output_0, &shape_tensor_convolution28_output_0);

var shape_tensor_plus30_output_0 : [4]usize = [_]usize{ 1, 8, 28, 28} ;
var array_plus30_output_0: [6272]T = [_]T{0} ** 6272;
var tensor_plus30_output_0 = Tensor(T).fromConstBuffer( &allocator, &array_plus30_output_0, &shape_tensor_plus30_output_0);

var shape_tensor_relu32_output_0 : [4]usize = [_]usize{ 1, 8, 28, 28} ;
var array_relu32_output_0: [6272]T = [_]T{0} ** 6272;
var tensor_relu32_output_0 = Tensor(T).fromConstBuffer( &allocator, &array_relu32_output_0, &shape_tensor_relu32_output_0);

var shape_tensor_pooling66_output_0 : [4]usize = [_]usize{ 1, 8, 14, 14} ;
var array_pooling66_output_0: [1568]T = [_]T{0} ** 1568;
var tensor_pooling66_output_0 = Tensor(T).fromConstBuffer( &allocator, &array_pooling66_output_0, &shape_tensor_pooling66_output_0);

var shape_tensor_convolution110_output_0 : [4]usize = [_]usize{ 1, 16, 14, 14} ;
var array_convolution110_output_0: [3136]T = [_]T{0} ** 3136;
var tensor_convolution110_output_0 = Tensor(T).fromConstBuffer( &allocator, &array_convolution110_output_0, &shape_tensor_convolution110_output_0);

var shape_tensor_plus112_output_0 : [4]usize = [_]usize{ 1, 16, 14, 14} ;
var array_plus112_output_0: [3136]T = [_]T{0} ** 3136;
var tensor_plus112_output_0 = Tensor(T).fromConstBuffer( &allocator, &array_plus112_output_0, &shape_tensor_plus112_output_0);

var shape_tensor_relu114_output_0 : [4]usize = [_]usize{ 1, 16, 14, 14} ;
var array_relu114_output_0: [3136]T = [_]T{0} ** 3136;
var tensor_relu114_output_0 = Tensor(T).fromConstBuffer( &allocator, &array_relu114_output_0, &shape_tensor_relu114_output_0);

var shape_tensor_pooling160_output_0 : [4]usize = [_]usize{ 1, 16, 4, 4} ;
var array_pooling160_output_0: [256]T = [_]T{0} ** 256;
var tensor_pooling160_output_0 = Tensor(T).fromConstBuffer( &allocator, &array_pooling160_output_0, &shape_tensor_pooling160_output_0);

var shape_tensor_pooling160_output_0_reshape0 : [2]usize = [_]usize{ 1, 256} ;
var array_pooling160_output_0_reshape0: [256]T = [_]T{0} ** 256;
var tensor_pooling160_output_0_reshape0 = Tensor(T).fromConstBuffer( &allocator, &array_pooling160_output_0_reshape0, &shape_tensor_pooling160_output_0_reshape0);

var shape_tensor_times212_output_0 : [2]usize = [_]usize{ 1, 10} ;
var array_times212_output_0: [10]T = [_]T{0} ** 10;
var tensor_times212_output_0 = Tensor(T).fromConstBuffer( &allocator, &array_times212_output_0, &shape_tensor_times212_output_0);

var shape_tensor_plus214_output_0 : [2]usize = [_]usize{ 1, 10} ;
var array_plus214_output_0: [10]T = [_]T{0} ** 10;
var tensor_plus214_output_0 = Tensor(T).fromConstBuffer( &allocator, &array_plus214_output_0, &shape_tensor_plus214_output_0);

//Function to reset all output tensors to zero
fn resetOutputTensors() void {
    if (log_function) |log| {
        log(@constCast(@ptrCast("Resetting output tensors...\n")));
    }
    @memset(array_parameter193_reshape1[0..], 0);
    @memset(array_convolution28_output_0[0..], 0);
    @memset(array_plus30_output_0[0..], 0);
    @memset(array_relu32_output_0[0..], 0);
    @memset(array_pooling66_output_0[0..], 0);
    @memset(array_convolution110_output_0[0..], 0);
    @memset(array_plus112_output_0[0..], 0);
    @memset(array_relu114_output_0[0..], 0);
    @memset(array_pooling160_output_0[0..], 0);
    @memset(array_pooling160_output_0_reshape0[0..], 0);
    @memset(array_times212_output_0[0..], 0);
    @memset(array_plus214_output_0[0..], 0);
    if (log_function) |log| {
        log(@constCast(@ptrCast("Output tensors reset.\n")));
    }
}


pub export fn predict( 
    input: [*]T,
    input_shape: [*]u32,
    shape_len: u32,
    result: *[*]T,
) void {

    if (log_function) |log| {
        log(@constCast(@ptrCast("Starting prediction...\n")));
    }
    // Reset all output tensors to zero before each prediction
    resetOutputTensors();
    //checks on the input parameters
    if (shape_len == 0) return ;
    if(shape_len != 4) return ;
    if( input_shape[0] != 1) return ;
    if( input_shape[1] != 1) return ;
    if( input_shape[2] != 28) return ;
    if( input_shape[3] != 28) return ;  
    //computing the size of the input tensor
    var size: u32 = 1;
    for(0..shape_len) |dim_i| {
        size *= input_shape[dim_i];
    }
     
    //allocating space in memory for the data
    const data = allocator.alloc(T, size) catch return;
    defer allocator.free(data);
    for (0..size) |i| {
        data[i] = input[i]; // Copying input elements 
    }
    
    //converting the shape from [*]u32 to []usize
    const usized_shape: []usize = utils.u32ToUsize(input_shape, shape_len) catch return;
    var tensor_input3 = Tensor(T).fromShape(&allocator, @constCast(usized_shape)) catch return;
    defer allocator.free(usized_shape);
    defer tensor_input3.deinit();
    @memcpy(tensor_input3.data, data); 

    if (log_function) |log| {
        log(@constCast(@ptrCast("Running Reshape operation...\n")));
    }

    const newShape_tensor_parameter193_reshape1_shape: []usize = utils.sliceToUsizeSlice(param_lib.tensor_parameter193_reshape1_shape.data);
    defer allocator.free(newShape_tensor_parameter193_reshape1_shape);
    tensMath.reshape_lean(
        T, //type
        @constCast(&param_lib.tensor_parameter193), //Input tensor
        newShape_tensor_parameter193_reshape1_shape, //New shape
        false, //allowzero
        &tensor_parameter193_reshape1, //Output tensor
    ) catch return; 

    if (log_function) |log| {
        log(@constCast(@ptrCast("Running Conv operation...\n")));
    }    

    tensMath.conv_lean(
        T, //type
        &tensor_input3, //input
        @constCast(&param_lib.tensor_parameter5), //kernel
        &tensor_convolution28_output_0, //output
        null, //bias
        &[_]usize{1,1}, //stride
         null, //pads
        &[_]usize{1,1}, //dilatations
        1, //group
        "SAME_UPPER", //auto_pad
    ) catch return; 

    if (log_function) |log| {
        log(@constCast(@ptrCast("Running Add operation...\n")));
    }
    tensMath.sum_tensors_lean(T, T, &tensor_convolution28_output_0, @constCast(&param_lib.tensor_parameter6), &tensor_plus30_output_0) catch return; 

    if (log_function) |log| {
        log(@constCast(@ptrCast("Running Relu operation...\n")));
    }

    tensMath.ReLU_lean(T, &tensor_plus30_output_0, &tensor_relu32_output_0) catch return; 

    if (log_function) |log| {
        log(@constCast(@ptrCast("Running MaxPool operation...\n")));
    }

    tensMath.onnx_maxpool_lean(
        T,
        &tensor_relu32_output_0, //Input
        &tensor_pooling66_output_0, //Output
        &[_]usize{2,2}, //kernel_shape
        &[_]usize{2,2}, //strides
        &[_]usize{1,1,1,1}, //dilations
        &[_]usize{0,0,0,0}, //pads
        tensMath.AutoPadType.NOTSET, //auto_pad
    ) catch return; 

    if (log_function) |log| {
        log(@constCast(@ptrCast("Running Conv operation...\n")));
    }    

    tensMath.conv_lean(
        T, //type
        &tensor_pooling66_output_0, //input
        @constCast(&param_lib.tensor_parameter87), //kernel
        &tensor_convolution110_output_0, //output
        null, //bias
        &[_]usize{1,1}, //stride
         null, //pads
        &[_]usize{1,1}, //dilatations
        1, //group
        "SAME_UPPER", //auto_pad
    ) catch return; 

    if (log_function) |log| {
        log(@constCast(@ptrCast("Running Add operation...\n")));
    }
    tensMath.sum_tensors_lean(T, T, &tensor_convolution110_output_0, @constCast(&param_lib.tensor_parameter88), &tensor_plus112_output_0) catch return; 

    if (log_function) |log| {
        log(@constCast(@ptrCast("Running Relu operation...\n")));
    }

    tensMath.ReLU_lean(T, &tensor_plus112_output_0, &tensor_relu114_output_0) catch return; 

    if (log_function) |log| {
        log(@constCast(@ptrCast("Running MaxPool operation...\n")));
    }

    tensMath.onnx_maxpool_lean(
        T,
        &tensor_relu114_output_0, //Input
        &tensor_pooling160_output_0, //Output
        &[_]usize{3,3}, //kernel_shape
        &[_]usize{3,3}, //strides
        &[_]usize{1,1,1,1}, //dilations
        &[_]usize{0,0,0,0}, //pads
        tensMath.AutoPadType.NOTSET, //auto_pad
    ) catch return; 

    if (log_function) |log| {
        log(@constCast(@ptrCast("Running Reshape operation...\n")));
    }

    const newShape_tensor_pooling160_output_0_reshape0_shape: []usize = utils.sliceToUsizeSlice(param_lib.tensor_pooling160_output_0_reshape0_shape.data);
    defer allocator.free(newShape_tensor_pooling160_output_0_reshape0_shape);
    tensMath.reshape_lean(
        T, //type
        @constCast(&tensor_pooling160_output_0), //Input tensor
        newShape_tensor_pooling160_output_0_reshape0_shape, //New shape
        false, //allowzero
        &tensor_pooling160_output_0_reshape0, //Output tensor
    ) catch return; 

    if (log_function) |log| {
        log(@constCast(@ptrCast("Running MatMul operation...\n")));
    }
    tensMath.mat_mul_lean(T, &tensor_pooling160_output_0_reshape0, @constCast(&tensor_parameter193_reshape1), &tensor_times212_output_0) catch return; 

    if (log_function) |log| {
        log(@constCast(@ptrCast("Running Add operation...\n")));
    }
    tensMath.sum_tensors_lean(T, T, &tensor_times212_output_0, @constCast(&param_lib.tensor_parameter194), &tensor_plus214_output_0) catch return;
    result.* = tensor_plus214_output_0.data.ptr;

    if (log_function) |log| {
        log(@constCast(@ptrCast("Prediction completed.\n")));
    }
} 