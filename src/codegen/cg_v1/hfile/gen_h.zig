const std = @import("std");
const zant = @import("zant");
const IR_zant = @import("IR_zant");
const H_helper = @import("h_helper.zig").H_helper;

pub inline fn write_h_file(writer: *std.Io.Writer, h_helper: H_helper) !void {
    ///////////////////////////////////////////////
    //------------HEADERS / LIBRARIES-----------///
    ///////////////////////////////////////////////
    _ = try writer.print(
        \\#ifndef ZANT_ARDUINO_H
        \\#define ZANT_ARDUINO_H
        \\
        \\#include <Arduino.h>
        \\#include <stdint.h>
        \\#include <stdbool.h>
        \\
        \\#ifdef __cplusplus
        \\extern "C" {{
        \\#endif
        \\
    , .{});

    ///////////////////////////////////////////////
    ////------------PREDICT ELEMENTS-----------////
    ///////////////////////////////////////////////

    _ = try writer.print(
        \\typedef struct {{
        \\    {s}* data;
        \\    size_t size;
        \\}} zant_tensor_t;
        \\
        \\
        \\int predict(
        \\  {s}*    input,       
        \\  uint32_t* input_shape,                    //DA CAMBIARE? 
        \\  uint32_t  shape_len,                      //DA CAMBIARE?
        \\  {s}**   result        
        \\);
    , .{
        h_helper.input_type,
        h_helper.input_type,
        h_helper.input_type,
    });

    ////////////////////////////////////////////////
    ////------------WEIGHTS I/O LAYER-----------////
    ////////////////////////////////////////////////

    _ = try writer.print(
        \\
        \\typedef int (*zant_weight_read_callback_t)(size_t offset, uint8_t* buffer, size_t size);
        \\
        \\extern void zant_init_weights_io(void) __attribute__((weak));
        \\
        \\extern void zant_register_weight_callback(zant_weight_read_callback_t cb) __attribute__((weak));
        \\
        \\extern void zant_set_weights_base_address(const uint8_t* base) __attribute__((weak));
        \\
        \\typedef struct {{
        \\    bool            has_callback;
        \\    bool            has_base_address;
        \\    const uint8_t*  base_address;                          //DA CAMBIARE?
        \\    size_t          total_size;  
        \\}} zant_weights_io_info_t;
        \\
        \\extern zant_weights_io_info_t zant_get_weights_io_info(void) __attribute__((weak));
        \\
    , .{});

    ////////////////////////////////////////////////
    /////------------ARDUINO HELPERS-----------/////
    ////////////////////////////////////////////////

    _ = try writer.print(
        \\
        \\void zant_arduino_init(void);
        \\int  zant_predict_image(uint8_t* rgb_image, int width, int height, float* output_probabilities);      //DA CAMBIARE? Tipi dell'output?
        \\void zant_set_flash_weights_callback(void);
        \\
    , .{});

    /////////////////////////////////////////////////
    ///////------------LOG CALLBACK -----------//////
    /////////////////////////////////////////////////

    _ = try writer.print(
        \\typedef void (*zant_log_callback_t)(const char* message);
        \\extern void zant_set_log_callback(zant_log_callback_t callback) __attribute__((weak));
        \\extern void setLogFunction(void (*func)(char*)) __attribute__((weak));
        \\
        \\
    , .{});

    /////////////////////////////////////////////////
    /////////------------ Closure -----------////////
    /////////////////////////////////////////////////

    _ = try writer.print(
        \\
        \\
        \\#ifdef __cplusplus
        \\}}
        \\#endif
        \\#endif // ZANT_ARDUINO_H
    , .{});
}
