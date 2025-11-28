const std = @import("std");
const zant = @import("zant");
const IR_zant = @import("IR_zant");
pub const codegen_options = @import("codegen_options");

// --- zant IR
const GraphZant = IR_zant.GraphZant;
const TensorZant = IR_zant.TensorZant;
const NodeZant = IR_zant.NodeZant;

// --- utils
pub const utils = IR_zant.utils;
// --- onnx
const onnx = zant.onnx;
const ModelOnnx = onnx.ModelProto;
// --- allocator
const allocator = zant.utils.allocator.allocator;

pub inline fn write_ino_file(writer: *std.Io.Writer, input_shape: []const usize, input_size: usize, output_size: usize) !void {

    /////////////////////////////////////////////
    //-----------Header delcaration------------//
    /////////////////////////////////////////////
    // static u_int8_t intputData -> static float inputData
    _ = try writer.print(
        \\#include <Arduino.h> 
        \\#include <lib_zant.h> // reminder: int predict(float*, uint32_t*, uint32_t, float**)
        \\
        \\// --- Predict parameters ---
        \\#ifndef ZANT_OUTPUT_LEN
        \\#define ZANT_OUTPUT_LEN {d}
        \\#endif
        \\static const uint32_t OUT_LEN = ZANT_OUTPUT_LEN;
        \\static const uint32_t IN_N = {d};
        \\static const uint32_t IN_C = {d};
        \\static const uint32_t IN_H = {d};
        \\static const uint32_t IN_W = {d};
        \\static const uint32_t IN_SIZE = IN_N * IN_C * IN_H * IN_W;
        \\static float inputData[{d}];
        \\static uint32_t inputShape[4] = {{IN_N, IN_C, IN_H, IN_W}};
        \\
    , .{ output_size, input_shape[0], input_shape[1], input_shape[2], input_shape[3], input_size });

    /////////////////////////////////////////////
    ////-----------OUTPUT PRINTER------------////
    /////////////////////////////////////////////

    _ = try writer.print(
        \\
        \\static void printOutput(const float *out, uint32_t len)
        \\{{
        \\  if (!out || len <= 0){{
        \\      Serial.println("Output nullo");
        \\      return;
        \\      }}
        \\
        \\  Serial.println("=== Output ===");
        \\  for(int i = 0; i < len; i++){{
        \\     Serial.print("out[");
        \\     Serial.print(i);
        \\     Serial.print("] = ");
        \\     Serial.println(out[i], 6);
        \\     }}
        \\  Serial.println("==============");
        \\}}
        \\
    , .{});

    /////////////////////////////////////////////
    ////------------SETUP PRINTER------------////
    /////////////////////////////////////////////

    //
    // The hardware is Nicla Vision of dafault for now
    //
    _ = try writer.print(
        \\
        \\void setup(){{
        \\  Serial.begin(115200);
        \\  uint32_t t0 = millis();
        \\  while (!Serial && (millis() - t0) < 4000){{
        \\    delay(10);
        \\  }}
        \\  Serial.println("\n== Nicla Vision ==");
        \\
        \\  // Prepare NCHW input 
        \\  for (uint32_t c = 0; c < IN_C; c++){{
        \\      for (uint32_t h = 0; h < IN_H; h++){{
        \\          for (uint32_t w = 0; w < IN_W; w++){{
        \\              uint32_t idx = c * (IN_H * IN_W) + h * IN_W + w;
        \\              inputData[idx] = 1;
        \\          }}
        \\      }}
        \\  }}
        \\}}
        \\
    , .{});

    /////////////////////////////////////////////
    ////------------LOOP PRINTER------------/////
    /////////////////////////////////////////////
    //
    // changes: u_int8_t out -> float out
    //
    _ = try writer.print(
        \\
        \\void loop() {{
        \\  float *out = nullptr;  
        \\  Serial.println("[Predict] Calling predict()...");
        \\  int rc = -3 ;
        \\  unsigned long average_sum = 0;
        \\
        \\  for(uint32_t i = 0; i<10; i++) {{
        \\      unsigned long t_us0 = micros();
        \\      rc = predict(inputData, inputShape, 4, &out);
        \\      unsigned long t_us1 = micros();
        \\      average_sum = average_sum + t_us1 - t_us0;
        \\      if(rc!=0) break;
        \\  }}
        \\  
        \\  if (rc == 0 && out){{
        \\      printOutput(out, OUT_LEN);
        \\  }}
        \\  else{{
        \\      Serial.println("[Predict] FAIL");
        \\  }}
        \\  
        \\  Serial.print("[Predict] rc=");
        \\  Serial.println(rc);
        \\  Serial.print("[Predict] us=");
        \\  Serial.println((unsigned long)(average_sum/10));
        \\
        \\  delay(500); 
        \\}}
        \\
    , .{});
    return;
}
