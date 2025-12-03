
#include <Arduino.h>
#include <lib_zant.h> // int predict(float*, uint32_t*, uint32_t, float**)

// ---- Predict parameters ----
#ifndef ZANT_OUTPUT_LEN
#define ZANT_OUTPUT_LEN 30000 // <<<<<<<<<<<<<<<< ensure it is correct !!
#endif
static const uint32_t OUT_LEN = ZANT_OUTPUT_LEN;
 
// PAY ATTENTION TO THE FORMAT ( NHWC or NCWH )!!!!!!!
//
static const uint32_t IN_N = 1; // <<<<<<<<<<<<<<<< ensure it is correct !!
static const uint32_t IN_H = 100; // <<<<<<<<<<<<<<<< ensure it is correct !!
static const uint32_t IN_W = 100; // <<<<<<<<<<<<<<<< ensure it is correct !!
static const uint32_t IN_C = 2; // <<<<<<<<<<<<<<<< ensure it is correct !!
static const uint32_t IN_SIZE = IN_N * IN_C * IN_H * IN_W;
static u_int8_t inputData[IN_SIZE];
static uint32_t inputShape[4] = {IN_N, IN_H, IN_W, IN_C};

u_int8_t *out = nullptr;

uint32_t counter ;  

static void printOutput(const u_int8_t *out, uint32_t len)
{
    if (!out || len <= 0)
    {
        Serial.println("Output nullo");
        return;
    }
    Serial.println("=== Output ===");
    for (int i = 0; i < len; ++i)
    {
        Serial.print("out[");
        Serial.print(i);
        Serial.print("] = ");
        Serial.println(out[i], 6);
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
    Serial.println("\n== Nicla Vision ==");

    // Prepare NCHW input 
    for (uint32_t c = 0; c < IN_C; ++c)
        for (uint32_t h = 0; h < IN_H; ++h)
            for (uint32_t w = 0; w < IN_W; ++w)
            {
                uint32_t idx = c * (IN_H * IN_W) + h * IN_W + w;
                inputData[idx] = 1;
            }
}

void loop() { 
    
    Serial.println("[Predict] Calling predict()...");
    int rc = -3 ;
    unsigned long average_sum = 0;
    counter++;

    for(uint32_t i = 0; i<3; i++) {
        unsigned long t_us0 = micros();
        rc = predict(inputData, inputShape, 4, &out);
        unsigned long t_us1 = micros();
        average_sum = average_sum + t_us1 - t_us0;
        if(rc!=0) break;
    }

    if (rc == 0 && out)
    {
        // printOutput(out, OUT_LEN);
        Serial.println("[Predict] WIN");
        Serial.println("COUNTER: ");
        Serial.println(counter);
    }
    else
    {
        Serial.println("[Predict] FAIL");
        Serial.println("COUNTER: ");
        Serial.println(counter);
        delay(5000); 
    }

    Serial.print("[Predict] rc=");
    Serial.println(rc);
    Serial.print("[Predict] us=");
    Serial.println((unsigned long)(average_sum/3));
    
    delay(500); 
    out = nullptr;
}