#include <Arduino.h>
#include <lib_zant.h> 
#include <stdint.h> // Necessary for int8_t

// ZANT HOOKS
extern "C" void zant_free_result(u_int8_t*) __attribute__((weak));

// ---- Predict parameters ----
#ifndef ZANT_OUTPUT_LEN
#define ZANT_OUTPUT_LEN 9216 // <<<<<<<<<<<<<<<< tune it!
#endif
static const uint32_t OUT_LEN = ZANT_OUTPUT_LEN;

// PAY ATTENTION TO THE FORMAT: NHWC (CMSIS) or NCHW (Zant) !!!!!!!
static const uint32_t IN_N = 1; // <<<<<<<<<<<<<<<< tune it!
static const uint32_t IN_H = 24; // <<<<<<<<<<<<<<<< tune it!
static const uint32_t IN_W = 24; // <<<<<<<<<<<<<<<< tune it!
static const uint32_t IN_C = 16; // <<<<<<<<<<<<<<<< tune it!
static const uint32_t IN_SIZE = IN_N * IN_C * IN_H * IN_W;
static int8_t inputData[IN_SIZE]; // for i8
// static u_int8_t inputData[IN_SIZE]; // for u8
static uint32_t inputShape[4] = {IN_N, IN_C, IN_H, IN_W}; // <<<<<<<<<<<<<<<< tune it!

int8_t *out = nullptr; // for i8
// u_int8_t *out = nullptr; // for u8
uint32_t counter;  

static void printOutput(const int8_t *out, uint32_t len)
{
    if (!out || len <= 0)
    {
        Serial.println("Output nullo");
        return;
    }
    Serial.println("=== Output ===");
    for (int i = 0; i < 10; ++i)
    {
        Serial.print("out[");
        Serial.print(i);
        Serial.print("] = ");
        Serial.println((int)out[i]); // for i8
        // Serial.println(out[i], 6); // for u8
    }
    Serial.println("==============");
}

void setup()
{
    counter = 0;
    Serial.begin(115200);
    uint32_t t0 = millis();
    while (!Serial && (millis() - t0) < 4000)
        delay(10);
    Serial.println("\n== Nucleo64 INT8 (NHWC) ==");

    // Prepare NHWC Input
    for (uint32_t h = 0; h < IN_H; ++h) {
        for (uint32_t w = 0; w < IN_W; ++w) {
            for (uint32_t c = 0; c < IN_C; ++c) {
                // Calcolo indice lineare per NHWC
                uint32_t idx = h * (IN_W * IN_C) + w * IN_C + c;
                inputData[idx] = 1; // Valore di test
            }
        }
    }

    /*
    // Prepare NCHW input 
    for (uint32_t c = 0; c < IN_C; ++c)
        for (uint32_t h = 0; h < IN_H; ++h)
            for (uint32_t w = 0; w < IN_W; ++w)
            {
                uint32_t idx = c * (IN_H * IN_W) + h * IN_W + w;
                inputData[idx] = 1;
            }
    */

}

void loop() { 
    
    Serial.println("[Predict] Calling predict()...");
    int rc = -3 ;
    unsigned long average_sum = 0;
    
    int avg = 10;

    for(uint32_t i = 0; i < avg; i++) {
        unsigned long t_us0 = micros();
        
        rc = predict((u_int8_t*)inputData, inputShape, 4, (u_int8_t**)&out); // for i8
        // rc = predict(inputData, inputShape, 4, &out); // for u8
        unsigned long t_us1 = micros();
        average_sum = average_sum + t_us1 - t_us0;
        counter++;

        // Free out mem
        if(zant_free_result) {
            zant_free_result((u_int8_t*)out); // for i8
            // zant_free_result(out); // for u8
        }
        out = nullptr;

        if(rc != 0) break;
    }

    if (rc == 0)
    {
        // printOutput(out, OUT_LEN);
        Serial.println("[Predict] WIN");
        Serial.print("COUNTER: ");
        Serial.println(counter);  
    }
    else
    {
        Serial.println("[Predict] FAIL");
        Serial.print("COUNTER: ");
        Serial.println(counter); 
    }

    Serial.print("[Predict] rc=");
    Serial.println(rc);
    Serial.print("[Predict] us=");
    Serial.println((unsigned long)(average_sum/avg));

    Serial.print("\n\n");
    delay(1500); 
}