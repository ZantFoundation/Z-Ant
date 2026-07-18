# List of known discrepancies of math functions from their onnx version

Discrepancies of math or lean_math functions from the onnx standard that are not subject to changes in the short term should be listed here so that they are taken into account when using those functions for example for code gen.

Since these are not definitive, remove a note when it is no longer true due to code changes.

## gather()

### Negative indices

According to onnx standard, values in indices tensor can be negative and if so they are converted to non negative values by adding the size of the axis pointed dimension of the data tensor. For performance and code clarity reasons (check + double casting) we support only non negative indices instead.

## mat_mul()

### multibatch/multichannel matrix (affects gemm())

The implemented version doesn't work for stacks of matrices (tensor in the form of {batch, channel, rows, cols} with batch or channel > 1) but only with mono matrix tensors ({1, 1, rows, cols}).

### batch/channel broadcast

Since multibatch/multichannel is not supported neither batch/channel broadcast is.

## gemm()

### multibatch/multichannel matrix

Since multibatch/multichannel is not supported by mat_mul neither gemm does.