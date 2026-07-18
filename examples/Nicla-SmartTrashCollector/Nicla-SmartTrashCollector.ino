#include <Arduino.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "lib_zant.h"
#include "camera.h"
#include "gc2145.h"

// ----- LED pins (Nicla Vision) -----
#ifndef LEDR
#define LEDR 23
#define LEDG 24
#define LEDB 25
#endif

// If colors look inverted, flip this
static const bool LED_ACTIVE_HIGH = true;

// Discrete (ON/OFF) driver
static inline void setRGBDiscrete(bool r, bool g, bool b) {
  auto drive = [](int pin, bool on){
    bool level = LED_ACTIVE_HIGH ? on : !on;
    digitalWrite(pin, level ? HIGH : LOW);
  };
  drive(LEDR, r);
  drive(LEDG, g);
  drive(LEDB, b);
}

enum Cat : uint8_t { GENERAL=0, GLASS=1, ORGANIC=2, PAPER=3, PLASTIC=4 };

// Map to 5 distinct discrete colors (no PWM)
static void showCategory(uint8_t cat) {
  switch (cat) {
    case GENERAL: setRGBDiscrete(true,  true,  false); break; // yellow  (R+G)
    case GLASS:   setRGBDiscrete(false, false, true ); break; // blue    (B)
    case ORGANIC:   setRGBDiscrete(false, true,  false); break; // green   (G)
    case PAPER: setRGBDiscrete(true,  false, false); break; // red     (R) ~ brown-ish
    case PLASTIC: setRGBDiscrete(true,  true,  true ); break; // white   (R+G+B) ~ grey
    default:      setRGBDiscrete(false, false, false); break; // off
  }
}

// ================== Config ==================
#define BAUD                 921600
#define THR                  0.60f
#define RGB565_IS_MSB_FIRST  1   // 1: big-endian (MSB-first), 0: little-endian
#define STREAM_TO_PC         1   // 1: stream binario compatibile con viewer Python, 0: solo log

// ================== Modello (NCHW) ==================
static const uint32_t N=1, C=3, H=96, W=96;     // input RGB channels-first
static const uint32_t CLASSES=5;
static uint32_t inputShape[4] = {N, C, H, W};   // NCHW

// ================== Buffers ==================
// Input normalizzato 0..1 (float). ~108 KB
alignas(32) static float gInput[N*C*H*W];
// Opzionale: anteprima in GRAY8 (non normalizzata, solo preview)
static uint8_t gGray8[W*H];

// ================== Camera ==================
GC2145  sensor;
Camera  cam(sensor);
FrameBuffer fb;

// ================== ZANT hooks (deboli) ==================
extern "C" void setLogFunction(void (*logger)(char*)) __attribute__((weak));
extern "C" void zant_free_result(float*) __attribute__((weak));
extern "C" void zant_init_weights_io(void) __attribute__((weak));
extern "C" void zant_set_weights_base_address(const uint8_t*) __attribute__((weak));
extern "C" void zant_register_weight_callback(int (*cb)(size_t,uint8_t*,size_t)) __attribute__((weak));
extern "C" __attribute__((used)) const uint8_t* flash_weights_base=(const uint8_t*)0x90000000u;

// ================== QSPI / HAL (STM32H747) ==================
extern "C" {
  #ifndef STM32H747xx
  #define STM32H747xx
  #endif
  #ifndef HAL_QSPI_MODULE_ENABLED
  #define HAL_QSPI_MODULE_ENABLED
  #endif
  #include "stm32h7xx_hal.h"
  #include "stm32h7xx_hal_qspi.h"
}
static QSPI_HandleTypeDef hqspi;

// Comandi tipici NOR esterna (verifica con il tuo chip se servono diversi)
static const uint8_t CMD_RDSR1 = 0x05;
static const uint8_t CMD_RDSR2 = 0x35;
static const uint8_t CMD_WRSR  = 0x01;
static const uint8_t CMD_WREN  = 0x06;
static const uint8_t CMD_READ_QO = 0x6B;   // Quad Output Fast Read

extern "C" void HAL_QSPI_MspInit(QSPI_HandleTypeDef* h) {
  if (h->Instance != QUADSPI) return;
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOG_CLK_ENABLE();
  __HAL_RCC_QSPI_CLK_ENABLE();

  GPIO_InitTypeDef G = {0};
  // PB2: CLK
  G.Pin = GPIO_PIN_2; G.Mode = GPIO_MODE_AF_PP; G.Pull = GPIO_NOPULL;
  G.Speed = GPIO_SPEED_FREQ_VERY_HIGH; G.Alternate = GPIO_AF9_QUADSPI;
  HAL_GPIO_Init(GPIOB, &G);

  // PG6: nCS
  G.Pin = GPIO_PIN_6; G.Alternate = GPIO_AF10_QUADSPI;
  HAL_GPIO_Init(GPIOG, &G);

  // PD11..PD14: IO0..IO3
  G.Pin = GPIO_PIN_11 | GPIO_PIN_12 | GPIO_PIN_13 | GPIO_PIN_14;
  G.Alternate = GPIO_AF9_QUADSPI;
  HAL_GPIO_Init(GPIOD, &G);
}

static HAL_StatusTypeDef rd_sr(QSPI_HandleTypeDef* h, uint8_t cmd, uint8_t* v){
  QSPI_CommandTypeDef c = {0};
  c.InstructionMode = QSPI_INSTRUCTION_1_LINE;
  c.Instruction     = cmd;
  c.AddressMode     = QSPI_ADDRESS_NONE;
  c.DataMode        = QSPI_DATA_1_LINE;
  c.NbData          = 1;
  if (HAL_QSPI_Command(h, &c, HAL_MAX_DELAY) != HAL_OK) return HAL_ERROR;
  return HAL_QSPI_Receive(h, v, HAL_MAX_DELAY);
}

static HAL_StatusTypeDef wren(QSPI_HandleTypeDef* h){
  QSPI_CommandTypeDef c = {0};
  c.InstructionMode = QSPI_INSTRUCTION_1_LINE;
  c.Instruction     = CMD_WREN;
  c.AddressMode     = QSPI_ADDRESS_NONE;
  c.DataMode        = QSPI_DATA_NONE;
  return HAL_QSPI_Command(h, &c, HAL_MAX_DELAY);
}

static HAL_StatusTypeDef wait_wip_clear(QSPI_HandleTypeDef* h, uint32_t to_ms){
  uint32_t t0 = millis();
  for(;;){
    uint8_t sr1 = 0;
    if (rd_sr(h, CMD_RDSR1, &sr1) != HAL_OK) return HAL_ERROR;
    if ((sr1 & 0x01) == 0) return HAL_OK;    // WIP=0
    if ((millis() - t0) > to_ms) return HAL_TIMEOUT;
    delay(1);
  }
}

static HAL_StatusTypeDef enable_quad(QSPI_HandleTypeDef* h){
  uint8_t sr1=0, sr2=0;
  if (rd_sr(h, CMD_RDSR1, &sr1) != HAL_OK) return HAL_ERROR;
  if (rd_sr(h, CMD_RDSR2, &sr2) != HAL_OK) return HAL_ERROR;
  if (sr2 & 0x02) return HAL_OK; // QE già attivo

  if (wren(h) != HAL_OK) return HAL_ERROR;

  QSPI_CommandTypeDef c = {0};
  c.InstructionMode = QSPI_INSTRUCTION_1_LINE;
  c.Instruction     = CMD_WRSR;
  c.AddressMode     = QSPI_ADDRESS_NONE;
  c.DataMode        = QSPI_DATA_1_LINE;
  c.NbData          = 2;

  uint8_t buf[2] = { sr1, (uint8_t)(sr2 | 0x02) };
  if (HAL_QSPI_Command(h, &c, HAL_MAX_DELAY) != HAL_OK) return HAL_ERROR;
  if (HAL_QSPI_Transmit(h, buf, HAL_MAX_DELAY) != HAL_OK) return HAL_ERROR;
  if (wait_wip_clear(h, 500) != HAL_OK) return HAL_ERROR;

  if (rd_sr(h, CMD_RDSR2, &sr2) != HAL_OK) return HAL_ERROR;
  return (sr2 & 0x02) ? HAL_OK : HAL_ERROR;
}

static HAL_StatusTypeDef qspi_init_16mb(QSPI_HandleTypeDef* h){
  h->Instance               = QUADSPI;
  h->Init.ClockPrescaler    = 7;  // f_qspi = f_ahb/(Presc+1)
  h->Init.FifoThreshold     = 4;
  h->Init.SampleShifting    = QSPI_SAMPLE_SHIFTING_NONE;
  h->Init.FlashSize         = 23; // 2^23 -> 8MB; per 16MB usare 24. Nicla tipica è 16MB -> 24
  h->Init.FlashSize         = 24; // forza 16MB
  h->Init.ChipSelectHighTime= QSPI_CS_HIGH_TIME_2_CYCLE;
  h->Init.ClockMode         = QSPI_CLOCK_MODE_0;
  h->Init.FlashID           = QSPI_FLASH_ID_1;
  h->Init.DualFlash         = QSPI_DUALFLASH_DISABLE;
  return HAL_QSPI_Init(h);
}

static HAL_StatusTypeDef qspi_enter_mmap(QSPI_HandleTypeDef* h){
  QSPI_CommandTypeDef c = {0};
  c.InstructionMode  = QSPI_INSTRUCTION_1_LINE;
  c.Instruction      = CMD_READ_QO;
  c.AddressMode      = QSPI_ADDRESS_1_LINE;
  c.AddressSize      = QSPI_ADDRESS_24_BITS;
  c.Address          = 0x000000;
  c.AlternateByteMode= QSPI_ALTERNATE_BYTES_NONE;
  c.DataMode         = QSPI_DATA_4_LINES;
  c.DummyCycles      = 8; // tipico per fast read quad
  QSPI_MemoryMappedTypeDef mm = {0};
  mm.TimeOutActivation = QSPI_TIMEOUT_COUNTER_DISABLE;
  mm.TimeOutPeriod     = 0;
  return HAL_QSPI_MemoryMapped(h, &c, &mm);
}

static int qspi_xip_read(size_t off, uint8_t* buf, size_t sz) {
  if (!buf || !flash_weights_base) return -1;
  memcpy(buf, flash_weights_base + off, sz);
  return 0;
}
static void attach_weights_io() {
  if (zant_init_weights_io) zant_init_weights_io();
  if (zant_set_weights_base_address)      zant_set_weights_base_address(flash_weights_base);
  else if (zant_register_weight_callback) zant_register_weight_callback(qspi_xip_read);
}

// ================== Logger ==================
static inline bool streaming() { return STREAM_TO_PC != 0; }

static void myLogger(char* msg) {
  if (streaming()) return; // niente log quando si streamma binario
  Serial.print("[ZANT] ");
  if (msg) Serial.println(msg);
  else     Serial.println("(null)");
}
static void mySilentLogger(char*) { /* no-op quando si streamma */ }

// // ================== COCO80 labels ==================
// static const char* COCO80[CLASSES] = {
//   "general", "glass", "organic", "paper", "plastic"
// };

// ================== Helpers colore ==================
static inline uint16_t load_rgb565_BE(const uint8_t* S2, int idx) {
  return (uint16_t)((S2[2*idx] << 8) | S2[2*idx + 1]);
}
static inline uint16_t load_rgb565_LE(const uint8_t* S2, int idx) {
  return (uint16_t)((S2[2*idx + 1] << 8) | S2[2*idx]);
}
static inline void rgb565_to_rgb888_u16(uint16_t v, uint8_t &R, uint8_t &G, uint8_t &B){
  uint8_t r5=(v>>11)&0x1F, g6=(v>>5)&0x3F, b5=v&0x1F;
  R=(uint8_t)((r5<<3)|(r5>>2));
  G=(uint8_t)((g6<<2)|(g6>>4));
  B=(uint8_t)((b5<<3)|(b5>>2));
}
static inline uint8_t clamp_u8(float x){
  if (x <= 0.f)   return 0;
  if (x >= 255.f) return 255;
  return (uint8_t)lrintf(x);
}
static inline int clampi(int v, int lo, int hi){
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

// ================== Resize → NCHW + Gray ==================
// Output gInput (NCHW): float normalizzato [0,1]
static void resize_rgb565_to_96x96_rgbNCHW_and_gray_NEAREST(
    const uint8_t* src, int sw, int sh,
    float* __restrict dst_f, uint8_t* __restrict dst_gray)
{
  const float sx = (float)sw / (float)W;
  const float sy = (float)sh / (float)H;
  const float inv255 = 1.0f / 255.0f;

  const int plane = (int)(H*W);
  float* __restrict dstR = dst_f + 0*plane;
  float* __restrict dstG = dst_f + 1*plane;
  float* __restrict dstB = dst_f + 2*plane;

  for (int y = 0; y < (int)H; ++y) {
    int ys = clampi((int)floorf((y + 0.5f) * sy), 0, sh - 1);
    for (int x = 0; x < (int)W; ++x) {
      int xs = clampi((int)floorf((x + 0.5f) * sx), 0, sw - 1);
      int si = ys * sw + xs;

      uint16_t v = RGB565_IS_MSB_FIRST ? load_rgb565_BE(src, si)
                                       : load_rgb565_LE(src, si);
      uint8_t r,g,b; rgb565_to_rgb888_u16(v, r,g,b);

      const int di = y*W + x;
      dstR[di] = (float)r * inv255;
      dstG[di] = (float)g * inv255;
      dstB[di] = (float)b * inv255;

      // Gray solo per preview (0..255)
      gGray8[di] = clamp_u8(0.299f*r + 0.587f*g + 0.114f*b);
    }
  }
}

// ================== softmax + top1 ==================
// static void softmax_vec(const float* in, int n, float* out){
//   float m = -INFINITY;
//   for (int i=0;i<n;++i) if (isfinite(in[i]) && in[i]>m) m=in[i];
//   float s = 0.f;
//   for (int i=0;i<n;++i){
//     float z = isfinite(in[i]) ? (in[i]-m) : -50.f;
//     float e = expf(z);
//     out[i] = e; s += e;
//   }
//   if (s <= 0.f) {
//     float u = 1.0f / (float)n;
//     for (int i=0;i<n;++i) out[i] = u;
//   } else {
//     float inv = 1.0f / s;
//     for (int i=0;i<n;++i) out[i] *= inv;
//   }
// }
// static inline void top1(const float* p,int n,int* idx,float* val){
//   int k=0; float b=-1.f;
//   for (int i=0;i<n;++i){ if (p[i]>b){ b=p[i]; k=i; } }
//   *idx=k; *val=b;
// }

static inline uint8_t argmax5(const float* a) {
  uint8_t k=0; float m=a[0];
  for (uint8_t i=1;i<5;i++) if (a[i] > m) { m=a[i]; k=i; }
  return k;
}

static inline void softmax5(const float* z, float* p){
  float m=z[0]; for(int i=1;i<5;i++) if(z[i]>m) m=z[i];
  float s=0.f;  for(int i=0;i<5;i++){ p[i]=expf(z[i]-m); s+=p[i]; }
  float inv = s>0.f? 1.f/s : 0.2f; for(int i=0;i<5;i++) p[i]*=inv;
}

// ================== CRC32 (Arduino-like) ==================
static uint32_t crc32_arduino(const uint8_t* data, size_t len){
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i=0;i<len;++i){
    crc ^= (uint32_t)data[i];
    for (int b=0;b<8;++b){
      if (crc & 1u) crc = (crc >> 1) ^ 0xEDB88320u;
      else          crc >>= 1;
    }
  }
  return ~crc;
}

// ================== Serial frame (FRME) ==================
static const uint8_t MAGIC[4] = {'F','R','M','E'};
static uint16_t g_seq = 0;

static inline void put_le16(uint8_t* p, uint16_t v){ p[0] = (uint8_t)(v & 0xFF); p[1] = (uint8_t)(v >> 8); }
static inline void put_le32(uint8_t* p, uint32_t v){ p[0] = (uint8_t)(v & 0xFF); p[1] = (uint8_t)((v>>8)&0xFF); p[2] = (uint8_t)((v>>16)&0xFF); p[3] = (uint8_t)((v>>24)&0xFF); }

static void send_frame_gray_FRME(uint16_t w, uint16_t h, uint8_t cls, uint16_t prob_x1000, uint16_t ms_x10, const uint8_t* gray){
  const uint32_t payload_len = (uint32_t)w * (uint32_t)h;
  const uint32_t crc = crc32_arduino(gray, payload_len);

  uint8_t hdr[20];
  memcpy(hdr, MAGIC, 4);
  hdr[4] = 1; // version
  put_le16(&hdr[5], g_seq);
  put_le16(&hdr[7], w);
  put_le16(&hdr[9], h);
  hdr[11] = cls;
  put_le16(&hdr[12], prob_x1000);
  put_le16(&hdr[14], ms_x10);
  put_le32(&hdr[16], payload_len);

  Serial.write(hdr, sizeof(hdr));
  Serial.write(gray, payload_len);

  uint8_t cbuf[4];
  put_le32(cbuf, crc);
  Serial.write(cbuf, 4);

  g_seq++;
}

// ================== Setup ==================
void setup(){
  pinMode(LEDR, OUTPUT); pinMode(LEDG, OUTPUT); pinMode(LEDB, OUTPUT);
  setRGBDiscrete(false,false,false);

  Serial.begin(BAUD);
  while (Serial.available()) Serial.read();

  // Logger: silenzioso se streammo binario
  if (setLogFunction) {
    if (streaming()) setLogFunction(mySilentLogger);
    else             setLogFunction(myLogger);
  }

  // ---- QSPI in Memory-Mapped Mode (XIP) ----
  if (qspi_init_16mb(&hqspi) != HAL_OK) {
    if (!streaming()) Serial.println("[ZANT] QSPI init FAILED");
  } else if (enable_quad(&hqspi) != HAL_OK) {
    if (!streaming()) Serial.println("[ZANT] QSPI enable QUAD FAILED");
  } else if (qspi_enter_mmap(&hqspi) != HAL_OK) {
    if (!streaming()) Serial.println("[ZANT] QSPI MMAP FAILED");
  } else {
    if (!streaming()) Serial.println("[ZANT] QSPI MMAP OK @0x90000000");
  }
  attach_weights_io();

  // ---- Camera ----
  cam.begin(CAMERA_R160x120, CAMERA_RGB565, 30);

  if (!streaming()) {
    Serial.println("[ZANT] Ready (NCHW 1x3x96x96, normalized 0..1).");
  }
}

// ================== Loop ==================
void loop(){
  if (cam.grabFrame(fb, 3000) != 0){
    if (!streaming()) Serial.println("[ZANT] camera timeout");
    delay(5);
    return;
  }
  const uint8_t* buf = fb.getBuffer();

  resize_rgb565_to_96x96_rgbNCHW_and_gray_NEAREST(buf, 160, 120, gInput, gGray8);

  // Inference
  float* out_raw = nullptr;
  unsigned long t0 = micros();
  int rc = predict(gInput, inputShape, 4, &out_raw);
  unsigned long t1 = micros();

  float ms_f = (t1 - t0) / 1000.0f;
  uint16_t ms_x10 = (uint16_t) (ms_f * 10.0f + 0.5f);

  if (rc != 0 || !out_raw) {
    if (!streaming()) {
      Serial.print("[ZANT] predict() rc=");
      Serial.println(rc);
    }
    delay(5);
    return;
  }

  // --- choose class + probability (5-logit model) ---
  uint8_t cls = argmax5(out_raw);   // argmax on raw logits

  float p5[5];
  softmax5(out_raw, p5);            // full 5-way softmax
  float prob = p5[cls];
  if (prob < 0.f) prob = 0.f; else if (prob > 1.f) prob = 1.f;
  uint16_t prob_x1000 = (uint16_t)lrintf(prob * 1000.0f);

  // --- LED feedback ---
  showCategory(cls);

  // --- stream or log ---
  if (streaming()) {
    send_frame_gray_FRME(W, H, cls, prob_x1000, ms_x10, gGray8);
  } else {
    // static const char* NAMES[5] = { "Plastic","Paper","Glass","Organic","General" };
    static const char* NAMES[5] = { "General","Glass","Organic","Paper","Plastic" };

    Serial.print("[Waste Selector]: idx="); Serial.print(cls);
    Serial.print(" label="); Serial.print(NAMES[cls]);
    Serial.print(" prob=");  Serial.print(prob, 3);
    Serial.print(" time=");  Serial.print(ms_f, 1); Serial.println(" ms");
  }

  // Libera l'output SOLO tramite hook ZANT (evita double-free)
  if (zant_free_result) zant_free_result(out_raw);
  out_raw = nullptr;

  delay(5);
}

