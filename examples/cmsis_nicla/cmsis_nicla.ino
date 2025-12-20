#include <Arduino.h>
#include <lib_zant.h> // int predict(float*, uint32_t*, uint32_t, float**)



// ZANT HOOKS
extern "C" void zant_free_result(u_int8_t*) __attribute__((weak));



// ---- Predict parameters ----
#ifndef ZANT_OUTPUT_LEN
#define ZANT_OUTPUT_LEN 80000 // <<<<<<<<<<<<<<<< ensure it is correct !!
#endif
static const uint32_t OUT_LEN = ZANT_OUTPUT_LEN;


static const uint32_t IN_N = 1; // <<<<<<<<<<<<<<<< ensure it is correct !!
static const uint32_t IN_H = 100; // <<<<<<<<<<<<<<<< ensure it is correct !!
static const uint32_t IN_W = 100; // <<<<<<<<<<<<<<<< ensure it is correct !!
static const uint32_t IN_C = 4; // <<<<<<<<<<<<<<<< ensure it is correct !!
static const uint32_t IN_SIZE = IN_N * IN_C * IN_H * IN_W;
static u_int8_t inputData[IN_SIZE];


// PAY ATTENTION TO THE FORMAT ( NHWC or NCHW )!!!!!!!
static uint32_t inputShape[4] = {IN_N, IN_C, IN_H, IN_W};

u_int8_t *out = nullptr;

uint32_t counter;  

static void printOutput(const u_int8_t *out, uint32_t len)
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


     Serial.println("=== inputShape ===");

  Serial.print("inputShape[0] = ");
  Serial.println(inputShape[0]);

  Serial.print("inputShape[1] = ");
  Serial.println(inputShape[1]);

  Serial.print("inputShape[2] = ");
  Serial.println(inputShape[2]);

  Serial.print("inputShape[3] = ");
  Serial.println(inputShape[3]);

  Serial.println("==================");

    Serial.println("[Predict] Calling predict()...");
    int rc = -3 ;
    unsigned long average_sum = 0;
    
    int avg = 10;

    for(uint32_t i = 0; i<avg; i++) {
        unsigned long t_us0 = micros();
        rc = predict(inputData, inputShape, 4, &out);
        unsigned long t_us1 = micros();
        average_sum = average_sum + t_us1 - t_us0;
        counter++;

        // Free out mem
        if(zant_free_result) zant_free_result(out);
        out = nullptr;

        if(rc!=0) break;
    }

    //if(!out) Serial.println("OUT nullpointer");
    // Serial.print("OUT address: ");
    // Serial.println((unsigned long) out);
    // Serial.print("OUT first value pointed: ");
    // Serial.println(*out);

    if (rc == 0)
    {
        // printOutput(out, OUT_LEN);
        Serial.println("[Predict] WIN");
        Serial.println("COUNTER: ");
        Serial.println(counter);  
    }
    else
    {
        Serial.println("[Predict] FAIL--------------");
        Serial.println("COUNTER: ");
        Serial.println(counter); 
    }

    Serial.print("[Predict] rc=");
    Serial.println(rc);
    Serial.print("[Predict] us=");
    Serial.println((unsigned long)(average_sum/avg));

    Serial.print("\n\n");
    delay(1000); 
}