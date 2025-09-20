#ifndef MY_NICLA_LIB_PREDICT_H
#define MY_NICLA_LIB_PREDICT_H

// Include l'header standard per tipi interi a dimensione fissa come uint32_t
#include <stdint.h>

// Questo blocco assicura che il linkage C sia usato anche se l'header
// viene incluso in un file C++ (come lo sketch .ino di Arduino)
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Runs model inference based on the input data.
 *
 * @param input Pointer to the buffer containing the input data (float).
 * It must be a flat array representing the input tensor.
 * @param input_shape Pointer to an array of uint32_t describing the
 * shape (dimensions) of the input tensor (e.g., {1, 1, 28, 28}).
 * @param shape_len Number of elements in the input_shape array (number of dimensions).
 * @param result Pointer to a pointer to float (float**). The function will write
 * the memory address of the internal buffer containing the inference result
 * into the location pointed to by 'result'.
 * WARNING: Do not deallocate the memory pointed to by *result,
 * it belongs to the library. The content is valid until the
 * next call to predict().
 * 
 * Retun codes:
 *  0 : everything good
 * -1 : something when wrong in the mathematical operations
 * -2 : something when wrong in the initialization phase
 * -3 : something when wrong in the output/return phase
 */
extern "C" {
    int predict(float* input, uint32_t* input_shape, uint32_t shape_len, float** result);
}

// (Opzionale) Puoi dichiarare anche setLogFunction se prevedi di usarla
// typedef void (*log_callback_t)(const char*); // Definisci un tipo per il callback
// void setLogFunction(log_callback_t func);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MY_NICLA_LIB_PREDICT_H
