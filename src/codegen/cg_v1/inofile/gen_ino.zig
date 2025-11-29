const std = @import("std");
const zant = @import("zant");
const IR_zant = @import("IR_zant");
const Ino_helper = @import("ino_helper.zig").Ino_helper;

///
///
///     THIS FILE IS TEMPORARY TAGERTED FOR ARDUINO NICLA VISION (chip STM32H7)
///
///
pub inline fn write_ino_file(writer: *std.Io.Writer, ino_helper: Ino_helper) !void {

    ///////////////////////////////////////////////
    //------------HEADERS / LIBRARIES-----------///
    ///////////////////////////////////////////////
    _ = try writer.print(
        \\#include <Arduino.h> 
        \\#include <lib_zant.h> // reminder: int predict({s}*, uint32_t*, uint32_t, {s}**)
        \\
        \\
        \\ {s}
        \\
        \\
    , .{
        ino_helper.input_type, //reminder
        ino_helper.output_type, //reminder
        ino_helper.optional_headers, //prints optionals
    });

    //////////////////////////////////////////////////////
    ////------------ZANT PREDICT PARAMETERS-----------////
    //////////////////////////////////////////////////////
    _ = try writer.print(
        \\
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
        \\static {s} inputData[{d}];
        \\static uint32_t inputShape[4] = {{IN_N, IN_C, IN_H, IN_W}};
        \\
        \\
    , .{
        ino_helper.output_size, //ZANT_OUTPUT_LEN
        ino_helper.input_shape[0], //IN_N = {d};
        ino_helper.input_shape[1], //IN_C = {d};
        ino_helper.input_shape[2], //IN_H = {d};
        ino_helper.input_shape[3], //IN_W = {d};
        ino_helper.input_type, //static {s} inputData...
        ino_helper.input_size, // inputData[{d}];
    });

    /////////////////////////////////////////////
    ////-----------OUTPUT PRINTER------------////
    /////////////////////////////////////////////
    _ = try writer.print(
        \\
        \\static void printOutput(const {s} *out, uint32_t len)
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
    , .{
        ino_helper.output_type, //printOutput(const {s} *out, uint32_t len)
    });

    /////////////////////////////////////////////
    ////------------SETUP PRINTER------------////
    /////////////////////////////////////////////
    //
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
        \\  {s} 
        \\
        \\ 
        \\  // Prepare NCHW input 
        \\  for (uint32_t c = 0; c < IN_C; c++){{
        \\      for (uint32_t h = 0; h < IN_H; h++){{
        \\          for (uint32_t w = 0; w < IN_W; w++){{
        \\              uint32_t idx = c * (IN_H * IN_W) + h * IN_W + w;
        \\              inputData[idx] = {s};
        \\          }}
        \\      }}
        \\  }}
        \\}}
        \\
    , .{ ino_helper.optional_controls, if (std.mem.eql(u8, ino_helper.input_type, "float")) "1.0" else "1" });

    /////////////////////////////////////////////
    ////------------LOOP PRINTER------------/////
    /////////////////////////////////////////////
    _ = try writer.print(
        \\
        \\void loop() {{
        \\  {s} *out = nullptr;  
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
    , .{
        ino_helper.output_type,
    });
    return;
}
