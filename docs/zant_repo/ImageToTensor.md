# ImageToTensor Module

## Main Functions

### `imageToRGB`

Converts an image to an RGB tensor.

```zig
pub fn imageToRGB(
    allocator: *const std.mem.Allocator,
    image_path: []const u8,
    norm_type: usize,
    comptime T: anytype,
) !Tensor(T)
```

**Parameters:**
- `allocator`: Memory allocator
- `image_path`: Path to the image
- `norm_type`: Normalization type (0: 0-1, 1: -1-1)
- `T`: Tensor data type (e.g., f32, f64)

**Returns:**
- A tensor with shape `[3, height, width]` where channels are R, G, B

### `imageToYCbCr`

Converts an image to a YCbCr tensor.

```zig
pub fn imageToYCbCr(
    allocator: *const std.mem.Allocator,
    image_path: []const u8,
    norm_type: usize,
    comptime T: anytype,
) !Tensor(T)
```

**Parameters:** (same as above)

**Returns:**
- A tensor with shape `[3, height, width]` where channels are Y, Cb, Cr

### `imageToGray`

Converts an image to a grayscale tensor.

```zig
pub fn imageToGray(
    allocator: *const std.mem.Allocator,
    image_path: []const u8,
    norm_type: usize,
    comptime T: anytype,
) !Tensor(T)
```

**Parameters:** (same as above)

**Returns:**
- A tensor with shape `[1, height, width]` containing the grayscale image

## Normalization

The output tensors are normalized as follows:

1. **Standard normalization (0-1)**: Pixel values are normalized to the range [0, 1]
   - Formula: `x_normalized = x/255`

2. **Signed normalization (-1-1)**: Pixel values are normalized to the range [-1, 1]
   - Formula: `x_normalized = (x/127.5) - 1`

Where `x` is the original pixel value in the range [0, 255].

## Output Tensor Structure

Generated tensors have the following structure:
- Dimension 0: Channels (3 for RGB/YCbCr, 1 for grayscale)
- Dimension 1: Image height (rows)
- Dimension 2: Image width (columns)

## Supported Formats

Currently, the library supports:
- JPEG (baseline)
- Chroma subsampling of type: 4:4:4, 4:2:2, 4:4:0, 4:2:0
- Progressive JPEG images are not supported, but can be easily implemented


## Debug Functions

For debugging purposes, functions are available to render the decoded images as BMP files:

```zig
// Save the decoded image as RGB BMP
try ImageToTensor.jpeg.debug_jpegToRGB(&allocator, "path/to/image.jpg");

// Save the decoded image as YCbCr BMP
try ImageToTensor.jpeg.debug_jpegToYCbCr(&allocator, "path/to/image.jpg");

// Save the decoded image as grayscale BMP
try ImageToTensor.jpeg.debug_jpegToGrayscale(&allocator, "path/to/image.jpg");
```

## Submodules:
The ImageToTensor module contains the following key files:

- `imageToTensor.zig`: Main module that defines the functions to convert images to tensors, handles file I/O, and coordinates the decoding process.
- `formatVerifier.zig`: Detects image format by analyzing file headers/signatures
- `utils.zig`: Common utilities and data structures used across the module such as SegmentReader, BitReader and normalization funzionctions. 
- `writerBMP.zig`: Debug functionality to write decoded images as BMP files.

JPEG-specific files in the `jpeg/` subdirectory:
- `jpegDecoder.zig`: Coordinates the jpeg decoding process using parser and algorithms.
- `jpegParser.zig`: Parses JPEG file structure and extracts metadata/segments.
- `jpegAlgorithms.zig`: Core JPEG decoding algorithms (Huffman, IDCT, Dequantizzation, Upsampling, Colorspace conversion).

## Jpeg decoding process:
The JPEG decoding process follows these key steps:

1. **Segment Parsing**: The JPEG file is parsed into segments (markers and data):
   - SOI (Start of Image)
   - DQT (Define Quantization Tables) 
   - DHT (Define Huffman Tables)
   - RST (Restart Interval)
   - SOF (Start of Frame - image dimensions and components)
   - SOS (Start of Scan - compressed image data)

2. Segmet Reading: Each segment is analized storing the usefull information in a JpegData struct 

3. **Entropy Decoding**: The compressed data is decoded using:
   - Huffman decoding to convert bit sequences into DCT coefficients
   - Run-length decoding to expand zero runs
   - Differential decoding for DC coefficients

4. **Dequantization**: DCT coefficients are multiplied by quantization tables to restore magnitude

5. **Inverse DCT**: 8x8 blocks of frequency coefficients are converted back to spatial domain

6. **Color Processing**:
   - Blocks are reassembled into complete color components
   - If subsampled (e.g. 4:2:0), chroma components are upsampled
   - YCbCr is converted to RGB if requested

7. **Output Preparation**:
   - Pixel values are stored in ColorChannels struct with values ranging from 0 to 255
