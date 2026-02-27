const std = @import("std");
const zant = @import("zant");
const IR_zant = @import("IR_zant");
pub const codegen_options = @import("codegen_options");

// --- zant IR
const GraphZant = IR_zant.GraphZant;
const TensorZant = IR_zant.TensorZant;
const tensorZant_lib = IR_zant.tensorZant_lib;
const TensorType = tensorZant_lib.TensorType;
const NodeZant = IR_zant.NodeZant;
const IR_utils = IR_zant.utils;

// --- allocator
const allocator = zant.utils.allocator.allocator;

pub const Ino_helper = struct {
    //optional headers and controlls to add into the codegen
    optional_headers: []u8,
    optional_controls: []u8,

    input_shape: []usize,
    input_type: []const u8,
    input_size: usize,

    output_shape: []usize,
    output_type: []const u8,
    output_size: usize,

    pub fn init() !Ino_helper {
        const input: TensorZant = try extract_input();
        const output: TensorZant = try extract_output();
        const optional_headers: []u8 = try print_optional_headers();
        const optional_controls: []u8 = try print_optional_controls();

        return Ino_helper{
            .optional_headers = optional_headers,
            .optional_controls = optional_controls,
            .input_shape = input.shape,
            .input_type = try from_TensorType_to_C_string_type(input.ty),
            .input_size = from_shape_to_size(input.shape),
            .output_shape = output.shape,
            .output_type = try from_TensorType_to_C_string_type(output.ty),
            .output_size = from_shape_to_size(output.shape),
        };
    }

    pub fn deinit(self: Ino_helper) void {
        if (self.optional_headers.len > 0) allocator.free(self.optional_headers);
        if (self.optional_controls.len > 0) allocator.free(self.optional_controls);
    }
};

///////////////////////////////////////////
//-----------HELPER FUNCTIONS------------//
///////////////////////////////////////////

fn extract_input() !TensorZant {
    const inputs: []TensorZant = try IR_utils.getInputs(&tensorZant_lib.tensorMap);

    if (inputs.len == 0) return error.noInputAvailable;

    // Finding the first non initializer input
    //TODO if there are no inputs but only initializers?
    var primary_index: usize = std.math.maxInt(usize);
    for (inputs, 0..) |*tz, idx| {
        if (tz.tc != tensorZant_lib.TensorCategory.INITIALIZER) {
            primary_index = idx;
            break;
        }
    }

    // only initializer???
    // return inputs[0].getShape(); ??
    return inputs[primary_index];
}

fn extract_output() !TensorZant {
    const outputs: []TensorZant = try IR_utils.getOutputs(&tensorZant_lib.tensorMap);
    return outputs[0];
}

fn from_TensorType_to_C_string_type(tensorType: TensorType) ![]const u8 {
    return switch (tensorType) {
        .f32 => "float",
        .f64 => "double",
        .i8 => "int8_t",
        .i16 => "int16_t",
        .i32 => "int32_t",
        .i64 => "int64_t",
        .u8 => "uint8_t",
        .u16 => "uint16_t",
        .u32 => "uint32_t",
        .u64 => "uint64_t",
        else => error.typeNotSupported,
    };
}

fn from_shape_to_size(input_shape: []usize) usize {
    var size: usize = 1;
    for (input_shape) |dim| {
        size *= dim;
    }
    return size;
}

//
// funcitions that will print in the codegen header to import and check to implement ES. (for xip enabled)
//
fn print_optional_headers() ![]u8 {
    var headers = std.ArrayList(u8){};
    if (codegen_options.xip) {
        try headers.appendSlice(allocator,
            \\////////////////////////////////////////////////
            \\///-----------PERIPHERALS SETTINGS-----------///
            \\////////////////////////////////////////////////
            \\
            \\
            \\extern "C"
            \\{
            \\#ifndef STM32H747xx
            \\#define STM32H747xx
            \\#endif
            \\#ifndef HAL_QSPI_MODULE_ENABLED
            \\#define HAL_QSPI_MODULE_ENABLED
            \\#endif
            \\#include "stm32h7xx_hal.h"
            \\#include "stm32h7xx_hal_qspi.h"
            \\}
            \\// Required by the Zig library:
            \\extern "C" __attribute__((used))
            \\const uint8_t *flash_weights_base = (const uint8_t *)0x90000000u;
            \\
            \\
            \\static QSPI_HandleTypeDef hqspi;
            \\
            \\static const uint8_t CMD_RDID = 0x9F, CMD_WREN = 0x06;
            \\static const uint8_t CMD_RDSR1 = 0x05, CMD_RDSR2 = 0x35, CMD_WRSR = 0x01;
            \\static const uint8_t CMD_READ_QO = 0x6B;
            \\// MSP init (GPIO+clock)
            \\extern "C" void HAL_QSPI_MspInit(QSPI_HandleTypeDef *h)
            \\{
            \\    if (h->Instance != QUADSPI)
            \\        return;
            \\    __HAL_RCC_GPIOB_CLK_ENABLE();
            \\    __HAL_RCC_GPIOD_CLK_ENABLE();
            \\    __HAL_RCC_GPIOG_CLK_ENABLE();
            \\    __HAL_RCC_QSPI_CLK_ENABLE();
            \\
            \\    GPIO_InitTypeDef GPIO = {0};
            \\    // CLK PB2 (AF9)
            \\    GPIO.Pin = GPIO_PIN_2;
            \\    GPIO.Mode = GPIO_MODE_AF_PP;
            \\    GPIO.Pull = GPIO_NOPULL;
            \\    GPIO.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
            \\    GPIO.Alternate = GPIO_AF9_QUADSPI;
            \\    HAL_GPIO_Init(GPIOB, &GPIO);
            \\    // CS PG6 (AF10)
            \\    GPIO.Pin = GPIO_PIN_6;
            \\    GPIO.Alternate = GPIO_AF10_QUADSPI;
            \\    HAL_GPIO_Init(GPIOG, &GPIO);
            \\    // IO0..IO3 PD11..PD14 (AF9)
            \\    GPIO.Pin = GPIO_PIN_11 | GPIO_PIN_12 | GPIO_PIN_13 | GPIO_PIN_14;
            \\    GPIO.Alternate = GPIO_AF9_QUADSPI;
            \\    HAL_GPIO_Init(GPIOD, &GPIO);
            \\}
            \\
            \\static HAL_StatusTypeDef qspi_init_16mb(QSPI_HandleTypeDef *h)
            \\{
            \\    h->Instance = QUADSPI;
            \\    h->Init.ClockPrescaler = 7;
            \\    h->Init.FifoThreshold = 4;
            \\    h->Init.SampleShifting = QSPI_SAMPLE_SHIFTING_NONE;
            \\    h->Init.FlashSize = 23; // 2^24 = 16MB -> set 23
            \\    h->Init.ChipSelectHighTime = QSPI_CS_HIGH_TIME_2_CYCLE;
            \\    h->Init.ClockMode = QSPI_CLOCK_MODE_0;
            \\    h->Init.FlashID = QSPI_FLASH_ID_1;
            \\    h->Init.DualFlash = QSPI_DUALFLASH_DISABLE;
            \\    return HAL_QSPI_Init(h);
            \\}
            \\
            \\static HAL_StatusTypeDef qspi_cmd(QSPI_HandleTypeDef *h, uint8_t inst,
            \\                                  uint32_t addrMode, uint32_t dataMode,
            \\                                  uint32_t addr, uint32_t dummy,
            \\                                  uint8_t *data, size_t len, bool rx)
            \\{
            \\    QSPI_CommandTypeDef c = {0};
            \\    c.InstructionMode = QSPI_INSTRUCTION_1_LINE;
            \\    c.Instruction = inst;
            \\    c.AddressMode = addrMode;
            \\    c.Address = addr;
            \\    c.AddressSize = QSPI_ADDRESS_24_BITS;
            \\    c.DataMode = dataMode;
            \\    c.NbData = len;
            \\    c.DummyCycles = dummy;
            \\    if (HAL_QSPI_Command(h, &c, HAL_MAX_DELAY) != HAL_OK)
            \\        return HAL_ERROR;
            \\    if (len == 0)
            \\        return HAL_OK;
            \\    return rx ? HAL_QSPI_Receive(h, data, HAL_MAX_DELAY)
            \\              : HAL_QSPI_Transmit(h, data, HAL_MAX_DELAY);
            \\}
            \\
            \\static HAL_StatusTypeDef rd_sr(QSPI_HandleTypeDef *h, uint8_t cmd, uint8_t *val)
            \\{
            \\    return qspi_cmd(h, cmd, QSPI_ADDRESS_NONE, QSPI_DATA_1_LINE, 0, 0, val, 1, true);
            \\}
            \\static HAL_StatusTypeDef wren(QSPI_HandleTypeDef *h)
            \\{
            \\    return qspi_cmd(h, CMD_WREN, QSPI_ADDRESS_NONE, QSPI_DATA_NONE, 0, 0, nullptr, 0, true);
            \\}
            \\static HAL_StatusTypeDef wr_sr12(QSPI_HandleTypeDef *h, uint8_t sr1, uint8_t sr2)
            \\{
            \\    uint8_t buf[2] = {sr1, sr2};
            \\    return qspi_cmd(h, CMD_WRSR, QSPI_ADDRESS_NONE, QSPI_DATA_1_LINE, 0, 0, buf, 2, false);
            \\}
            \\
            \\static HAL_StatusTypeDef wait_wip_clear(QSPI_HandleTypeDef *h, uint32_t timeout_ms)
            \\{
            \\    uint32_t t0 = millis();
            \\    for (;;)
            \\    {
            \\        uint8_t sr1 = 0;
            \\        if (rd_sr(h, CMD_RDSR1, &sr1) != HAL_OK)
            \\            return HAL_ERROR;
            \\        if ((sr1 & 0x01) == 0)
            \\            return HAL_OK;
            \\        if ((millis() - t0) > timeout_ms)
            \\            return HAL_TIMEOUT;
            \\        delay(1);
            \\    }
            \\}
            \\static HAL_StatusTypeDef enable_quad(QSPI_HandleTypeDef *h)
            \\{
            \\    uint8_t sr1 = 0, sr2 = 0;
            \\    if (rd_sr(h, CMD_RDSR1, &sr1) != HAL_OK)
            \\        return HAL_ERROR;
            \\    if (rd_sr(h, CMD_RDSR2, &sr2) != HAL_OK)
            \\        return HAL_ERROR;
            \\    if (sr2 & 0x02)
            \\        return HAL_OK; // QE already 1
            \\    if (wren(h) != HAL_OK)
            \\        return HAL_ERROR;
            \\    sr2 |= 0x02;
            \\    if (wr_sr12(h, sr1, sr2) != HAL_OK)
            \\        return HAL_ERROR;
            \\    if (wait_wip_clear(h, 500) != HAL_OK)
            \\        return HAL_ERROR;
            \\    if (rd_sr(h, CMD_RDSR2, &sr2) != HAL_OK)
            \\        return HAL_ERROR;
            \\    return (sr2 & 0x02) ? HAL_OK : HAL_ERROR;
            \\}
            \\
            \\static HAL_StatusTypeDef qspi_enter_mmap(QSPI_HandleTypeDef *h)
            \\{
            \\    QSPI_CommandTypeDef c = {0};
            \\    c.InstructionMode = QSPI_INSTRUCTION_1_LINE;
            \\    c.Instruction = CMD_READ_QO; // 0x6B
            \\    c.AddressMode = QSPI_ADDRESS_1_LINE;
            \\    c.AddressSize = QSPI_ADDRESS_24_BITS;
            \\    c.Address = 0x000000;
            \\    c.AlternateByteMode = QSPI_ALTERNATE_BYTES_NONE;
            \\    c.DataMode = QSPI_DATA_4_LINES;
            \\    c.DummyCycles = 8;
            \\#ifdef QSPI_DDR_MODE_DISABLE
            \\    c.DdrMode = QSPI_DDR_MODE_DISABLE;
            \\    c.DdrHoldHalfCycle = QSPI_DDR_HHC_ANALOG_DELAY;
            \\#endif
            \\#ifdef QSPI_SIOO_INST_EVERY_CMD
            \\    c.SIOOMode = QSPI_SIOO_INST_EVERY_CMD;
            \\#endif
            \\    QSPI_MemoryMappedTypeDef mm = {0};
            \\    mm.TimeOutActivation = QSPI_TIMEOUT_COUNTER_DISABLE;
            \\    mm.TimeOutPeriod = 0;
            \\    return HAL_QSPI_MemoryMapped(h, &c, &mm);
            \\}
        );
    }
    return headers.toOwnedSlice(allocator);
}

fn print_optional_controls() ![]u8 {
    var controls = std.ArrayList(u8){};
    if (codegen_options.xip) {
        try controls.appendSlice(allocator,
            \\
            \\  //-----------PERIPHERALS CONTROLS-----------//
            \\  
            \\  
            \\  if (qspi_init_16mb(&hqspi) != HAL_OK)
            \\  {
            \\      Serial.println("QSPI init FAIL");
            \\      for (;;)
            \\      {
            \\      }
            \\  }
            \\  if (enable_quad(&hqspi) != HAL_OK)
            \\  {
            \\      Serial.println("Enable QE FAIL");
            \\      for (;;)
            \\      {
            \\      }
            \\  }
            \\  if (qspi_enter_mmap(&hqspi) != HAL_OK)
            \\  {
            \\      Serial.println("XIP FAIL");
            \\      for (;;)
            \\      {
            \\      }
            \\  }
        );
    }
    return controls.toOwnedSlice(allocator);
}
