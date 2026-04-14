//! Shared ONNX type definitions used across proto files.
//! See the ONNX proto spec: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto

pub const Version = enum(i64) {
    IR_VERSION_2017_10_10 = 0x0000000000000001,
    IR_VERSION_2017_10_30 = 0x0000000000000002,
    IR_VERSION_2017_11_3 = 0x0000000000000003,
    IR_VERSION_2019_1_22 = 0x0000000000000004,
    IR_VERSION_2019_3_18 = 0x0000000000000005,
    IR_VERSION_2019_9_19 = 0x0000000000000006,
    IR_VERSION_2020_5_8 = 0x0000000000000007,
    IR_VERSION_2021_7_30 = 0x0000000000000008,
    IR_VERSION_2023_5_5 = 0x0000000000000009,
    IR_VERSION_2024_3_25 = 0x000000000000000A,
    IR_VERSION_2025_05_12 = 0x000000000000000B,
    IR_VERSION = 0x000000000000000C,
};

pub const DataType = enum(i32) {
    UNDEFINED = 0,
    FLOAT = 1,
    UINT8 = 2,
    INT8 = 3,
    UINT16 = 4,
    INT16 = 5,
    INT32 = 6,
    INT64 = 7,
    STRING = 8,
    BOOL = 9,
    FLOAT16 = 10,
    DOUBLE = 11,
    UINT32 = 12,
    UINT64 = 13,
    COMPLEX64 = 14,
    COMPLEX128 = 15,
    BFLOAT16 = 16,
    FLOAT8E4M3FN = 17,
    FLOAT8E4M3FNUZ = 18,
    FLOAT8E5M2 = 19,
    FLOAT8E5M2FNUZ = 20,
    UINT4 = 21,
    INT4 = 22,
    FLOAT4E2M1 = 23,
};

pub const DataLocation = enum(i32) {
    DEFAULT = 0,
    EXTERNAL = 1,
};

pub const AttributeType = enum {
    UNDEFINED,
    FLOAT,
    INT,
    STRING,
    TENSOR,
    GRAPH,
    SPARSE_TENSOR,
    FLOATS,
    INTS,
    STRINGS,
    TENSORS,
    GRAPHS,
    SPARSE_TENSORS,
};
