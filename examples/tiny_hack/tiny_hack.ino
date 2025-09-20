/************ Nicla Vision QSPI XIP (Memory-Mapped) + predict ************
 * Core: arduino:mbed_nicla (4.4.1)
 * - HAL QSPI (QUADSPI)
 * - Pin: CLK=PB2, CS=PG6, IO0..3=PD11..PD14 (AF QUADSPI)
 * - QE su SR2, poi READ 0x6B 1-1-4 con 8 dummy
 *****************************************************************************/

extern "C"
{
#ifndef STM32H747xx
#define STM32H747xx
#endif
#ifndef HAL_QSPI_MODULE_ENABLED
#define HAL_QSPI_MODULE_ENABLED
#endif
#include "stm32h7xx_hal.h"
#include "stm32h7xx_hal_qspi.h"
}

// Required by the Zig library:
extern "C" __attribute__((used))
const uint8_t *flash_weights_base = (const uint8_t *)0x90000000u;

#include <Arduino.h>
#include <lib_zant.h> // int predict(float*, uint32_t*, uint32_t, float**)

static QSPI_HandleTypeDef hqspi;

static const uint8_t CMD_RDID = 0x9F, CMD_WREN = 0x06;
static const uint8_t CMD_RDSR1 = 0x05, CMD_RDSR2 = 0x35, CMD_WRSR = 0x01;
static const uint8_t CMD_READ_QO = 0x6B;

// MSP init (GPIO+clock)
extern "C" void HAL_QSPI_MspInit(QSPI_HandleTypeDef *h)
{
    if (h->Instance != QUADSPI)
        return;
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOD_CLK_ENABLE();
    __HAL_RCC_GPIOG_CLK_ENABLE();
    __HAL_RCC_QSPI_CLK_ENABLE();

    GPIO_InitTypeDef GPIO = {0};
    // CLK PB2 (AF9)
    GPIO.Pin = GPIO_PIN_2;
    GPIO.Mode = GPIO_MODE_AF_PP;
    GPIO.Pull = GPIO_NOPULL;
    GPIO.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO.Alternate = GPIO_AF9_QUADSPI;
    HAL_GPIO_Init(GPIOB, &GPIO);
    // CS PG6 (AF10)
    GPIO.Pin = GPIO_PIN_6;
    GPIO.Alternate = GPIO_AF10_QUADSPI;
    HAL_GPIO_Init(GPIOG, &GPIO);
    // IO0..IO3 PD11..PD14 (AF9)
    GPIO.Pin = GPIO_PIN_11 | GPIO_PIN_12 | GPIO_PIN_13 | GPIO_PIN_14;
    GPIO.Alternate = GPIO_AF9_QUADSPI;
    HAL_GPIO_Init(GPIOD, &GPIO);
}

static HAL_StatusTypeDef qspi_init_16mb(QSPI_HandleTypeDef *h)
{
    h->Instance = QUADSPI;
    h->Init.ClockPrescaler = 7;
    h->Init.FifoThreshold = 4;
    h->Init.SampleShifting = QSPI_SAMPLE_SHIFTING_NONE;
    h->Init.FlashSize = 23; // 2^24 = 16MB -> set 23
    h->Init.ChipSelectHighTime = QSPI_CS_HIGH_TIME_2_CYCLE;
    h->Init.ClockMode = QSPI_CLOCK_MODE_0;
    h->Init.FlashID = QSPI_FLASH_ID_1;
    h->Init.DualFlash = QSPI_DUALFLASH_DISABLE;
    return HAL_QSPI_Init(h);
}

static HAL_StatusTypeDef qspi_cmd(QSPI_HandleTypeDef *h, uint8_t inst,
                                  uint32_t addrMode, uint32_t dataMode,
                                  uint32_t addr, uint32_t dummy,
                                  uint8_t *data, size_t len, bool rx)
{
    QSPI_CommandTypeDef c = {0};
    c.InstructionMode = QSPI_INSTRUCTION_1_LINE;
    c.Instruction = inst;
    c.AddressMode = addrMode;
    c.Address = addr;
    c.AddressSize = QSPI_ADDRESS_24_BITS;
    c.DataMode = dataMode;
    c.NbData = len;
    c.DummyCycles = dummy;
    if (HAL_QSPI_Command(h, &c, HAL_MAX_DELAY) != HAL_OK)
        return HAL_ERROR;
    if (len == 0)
        return HAL_OK;
    return rx ? HAL_QSPI_Receive(h, data, HAL_MAX_DELAY)
              : HAL_QSPI_Transmit(h, data, HAL_MAX_DELAY);
}

static HAL_StatusTypeDef rd_sr(QSPI_HandleTypeDef *h, uint8_t cmd, uint8_t *val)
{
    return qspi_cmd(h, cmd, QSPI_ADDRESS_NONE, QSPI_DATA_1_LINE, 0, 0, val, 1, true);
}
static HAL_StatusTypeDef wren(QSPI_HandleTypeDef *h)
{
    return qspi_cmd(h, CMD_WREN, QSPI_ADDRESS_NONE, QSPI_DATA_NONE, 0, 0, nullptr, 0, true);
}
static HAL_StatusTypeDef wr_sr12(QSPI_HandleTypeDef *h, uint8_t sr1, uint8_t sr2)
{
    uint8_t buf[2] = {sr1, sr2};
    return qspi_cmd(h, CMD_WRSR, QSPI_ADDRESS_NONE, QSPI_DATA_1_LINE, 0, 0, buf, 2, false);
}

static HAL_StatusTypeDef wait_wip_clear(QSPI_HandleTypeDef *h, uint32_t timeout_ms)
{
    uint32_t t0 = millis();
    for (;;)
    {
        uint8_t sr1 = 0;
        if (rd_sr(h, CMD_RDSR1, &sr1) != HAL_OK)
            return HAL_ERROR;
        if ((sr1 & 0x01) == 0)
            return HAL_OK;
        if ((millis() - t0) > timeout_ms)
            return HAL_TIMEOUT;
        delay(1);
    }
}
static HAL_StatusTypeDef enable_quad(QSPI_HandleTypeDef *h)
{
    uint8_t sr1 = 0, sr2 = 0;
    if (rd_sr(h, CMD_RDSR1, &sr1) != HAL_OK)
        return HAL_ERROR;
    if (rd_sr(h, CMD_RDSR2, &sr2) != HAL_OK)
        return HAL_ERROR;
    if (sr2 & 0x02)
        return HAL_OK; // QE already 1
    if (wren(h) != HAL_OK)
        return HAL_ERROR;
    sr2 |= 0x02;
    if (wr_sr12(h, sr1, sr2) != HAL_OK)
        return HAL_ERROR;
    if (wait_wip_clear(h, 500) != HAL_OK)
        return HAL_ERROR;
    if (rd_sr(h, CMD_RDSR2, &sr2) != HAL_OK)
        return HAL_ERROR;
    return (sr2 & 0x02) ? HAL_OK : HAL_ERROR;
}

static HAL_StatusTypeDef qspi_enter_mmap(QSPI_HandleTypeDef *h)
{
    QSPI_CommandTypeDef c = {0};
    c.InstructionMode = QSPI_INSTRUCTION_1_LINE;
    c.Instruction = CMD_READ_QO; // 0x6B
    c.AddressMode = QSPI_ADDRESS_1_LINE;
    c.AddressSize = QSPI_ADDRESS_24_BITS;
    c.Address = 0x000000;
    c.AlternateByteMode = QSPI_ALTERNATE_BYTES_NONE;
    c.DataMode = QSPI_DATA_4_LINES;
    c.DummyCycles = 8;
#ifdef QSPI_DDR_MODE_DISABLE
    c.DdrMode = QSPI_DDR_MODE_DISABLE;
    c.DdrHoldHalfCycle = QSPI_DDR_HHC_ANALOG_DELAY;
#endif
#ifdef QSPI_SIOO_INST_EVERY_CMD
    c.SIOOMode = QSPI_SIOO_INST_EVERY_CMD;
#endif
    QSPI_MemoryMappedTypeDef mm = {0};
    mm.TimeOutActivation = QSPI_TIMEOUT_COUNTER_DISABLE;
    mm.TimeOutPeriod = 0;
    return HAL_QSPI_MemoryMapped(h, &c, &mm);
}

// ---- Predict demo ----
#ifndef ZANT_OUTPUT_LEN
#define ZANT_OUTPUT_LEN 4
#endif
static const int OUT_LEN = ZANT_OUTPUT_LEN;
static const uint32_t IN_N = 1, IN_C = 3, IN_H = 32, IN_W = 32;
static const uint32_t IN_SIZE = IN_N * IN_C * IN_H * IN_W;
static float inputData[IN_SIZE];
static uint32_t inputShape[4] = {IN_N, IN_C, IN_H, IN_W};

static void printOutput(const float *out, int len)
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
    Serial.begin(115200);
    uint32_t t0 = millis();
    while (!Serial && (millis() - t0) < 4000)
        delay(10);
    Serial.println("\n== Nicla Vision QSPI XIP (HAL) + predict ==");

    if (qspi_init_16mb(&hqspi) != HAL_OK)
    {
        Serial.println("QSPI init FAIL");
        for (;;)
        {
        }
    }
    if (enable_quad(&hqspi) != HAL_OK)
    {
        Serial.println("Enable QE FAIL");
        for (;;)
        {
        }
    }
    if (qspi_enter_mmap(&hqspi) != HAL_OK)
    {
        Serial.println("XIP FAIL");
        for (;;)
        {
        }
    }

    // Prepare NCHW input (simple constant pattern per channel)
    for (uint32_t c = 0; c < IN_C; ++c)
        for (uint32_t h = 0; h < IN_H; ++h)
            for (uint32_t w = 0; w < IN_W; ++w)
            {
                uint32_t idx = c * (IN_H * IN_W) + h * IN_W + w;
                inputData[idx] = (c == 0) ? 0.8f : (c == 1 ? 0.5f : 0.2f);
            }

    float *out = nullptr;
    Serial.println("[Predict] Calling predict()...");
    unsigned long t_us0 = micros();
    int rc = predict(inputData, inputShape, 4, &out);
    unsigned long t_us1 = micros();

    Serial.print("[Predict] rc=");
    Serial.println(rc);
    Serial.print("[Predict] us=");
    Serial.println((unsigned long)(t_us1 - t_us0));
    if (rc == 0 && out)
    {
        printOutput(out, OUT_LEN);
    }
    else
    {
        Serial.println("[Predict] FAIL");
    }
}

void loop() { delay(500); }