#ifndef ZANT_ARDUINO_H
#define ZANT_ARDUINO_H

#include <Arduino.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ===================== PREDICT =====================
typedef struct {
    float* data;
    size_t size;
} zant_tensor_t;

// Main prediction function (generated)
int predict(
    float*    input,        // e.g. [1,3,32,32] => 3072 floats
    uint32_t* input_shape,  // e.g. {1,3,32,32}
    uint32_t  shape_len,    // e.g. 4
    float**   result        // out pointer to logits/probabilities
);

// ===================== WEIGHTS I/O LAYER =====================
// Callback signature usata dal runtime per leggere "blocchi" di pesi
typedef int (*zant_weight_read_callback_t)(size_t offset, uint8_t* buffer, size_t size);

// Le seguenti API possono NON essere presenti in libzant.a se hai build XIP.
// Le dichiariamo weak per consentire il link anche quando mancano.
// Quando le usi a runtime verifica prima che il puntatore non sia NULL.

// Inizializza il layer I/O (se presente)
extern void zant_init_weights_io(void) __attribute__((weak));

// Registra la callback di lettura pesi (se presente)
extern void zant_register_weight_callback(zant_weight_read_callback_t cb) __attribute__((weak));

// (Facoltativa) Fallback diretto: imposta base address dei pesi XIP (se supportata)
extern void zant_set_weights_base_address(const uint8_t* base) __attribute__((weak));

// (Facoltative) API info/debug — dichiara weak se il runtime le espone
typedef struct {
    bool            has_callback;
    bool            has_base_address;
    const uint8_t*  base_address;  // valido solo in direct mode
    size_t          total_size;    // se disponibile
} zant_weights_io_info_t;

extern zant_weights_io_info_t zant_get_weights_io_info(void) __attribute__((weak));

// ===================== ARDUINO HELPERS =====================
// Queste le implementi nel tuo sketch o in uno .cpp di supporto
void zant_arduino_init(void);
int  zant_predict_image(uint8_t* rgb_image, int width, int height, float* output_probabilities);
void zant_set_flash_weights_callback(void);

// ===================== LOG CALLBACK =====================
// Callback per debug logging
typedef void (*zant_log_callback_t)(const char* message);
extern void zant_set_log_callback(zant_log_callback_t callback) __attribute__((weak));

// Alternative log function setter (per compatibilità con lib_face.zig)
extern void setLogFunction(void (*func)(char*)) __attribute__((weak));

#ifdef __cplusplus
}
#endif

#endif // ZANT_ARDUINO_H

