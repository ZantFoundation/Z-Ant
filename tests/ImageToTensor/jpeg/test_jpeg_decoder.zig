const std = @import("std");
const zant = @import("zant");
const jpeg = zant.ImageToTensor.jpeg;
const pgkAllocator = zant.utils.allocator;
const debug_jpegToRGB = jpeg.debug_jpegToRGB;
const debug_jpegToYCbCr = jpeg.debug_jpegToYCbCr;
const debug_jpegToGrayscale = jpeg.debug_jpegToGrayscale;

//these are visive tests, can be used to check if an image is decoded correctly

test "jpeg baseline standard" {
    const allocator = pgkAllocator.allocator;
    const image_path = "tests/ImageToTensor/jpeg/baseline.jpg";
    try debug_jpegToRGB(allocator, image_path);

    try debug_jpegToYCbCr(allocator, image_path);

    try debug_jpegToGrayscale(allocator, image_path);
}

test "jpeg restart interval" {
    const allocator = pgkAllocator.allocator;
    const image_path = "tests/ImageToTensor/jpeg/restart_intervals.jpg";
    try debug_jpegToRGB(allocator, image_path);

    // try debug_jpegToYCbCr(&allocator, image_path);

    // try debug_jpegToGrayscale(&allocator, image_path);
}

test "jpeg subsampling" {
    const allocator = pgkAllocator.allocator;
    const image_path = "tests/ImageToTensor/jpeg/subsampling.jpg";
    try debug_jpegToRGB(allocator, image_path);

    // try debug_jpegToYCbCr(&allocator, image_path);  <<<<<<<<<<<<<<<<<<<<< BUGGED

    // try debug_jpegToGrayscale(&allocator, image_path);  <<<<<<<<<<<<<<<<<<<<< BUGGED
}

test "grayscale" {
    const allocator = pgkAllocator.allocator;
    const image_path = "tests/ImageToTensor/jpeg/grayscale.jpg";
    try debug_jpegToRGB(allocator, image_path);

    // try debug_jpegToYCbCr(&allocator, image_path); <<<<<<<<<<<<<<<<<<<<< BUGGED

    // try debug_jpegToGrayscale(&allocator, image_path);  <<<<<<<<<<<<<<<<<<<<< BUGGED
}
