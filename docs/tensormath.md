# TensorMath - File Explanations

## lib_activation_function_math
- **op_lasky_relU.zig**: Implementation of the **Leaky ReLU** activation function. Leaky ReLU is a variant of ReLU that allows a small, non-zero gradient when the unit is not active, helping to mitigate the "dying ReLU" problem.
- **op_relU.zig**: Implementation of the **ReLU (Rectified Linear Unit)** activation function. ReLU returns the input directly if it is positive; otherwise, it returns zero. It is widely used in neural networks for its simplicity and effectiveness.
- **op_sigmoid.zig**: Implementation of the **Sigmoid** activation function. Sigmoid maps input values to a range between 0 and 1, making it useful for binary classification and probabilistic outputs.
- **op_softmax.zig**: Implementation of the **Softmax** activation function. Softmax converts a vector of values into probabilities by exponentiating and normalizing the inputs. It is commonly used in the output layer of classification models.

## lib_elementWise_math
- **op_addrion.zig**: Element-wise **addition** operation. 
- **op_cell.zig**: Element-wise **comparison** operation. 
- **op_division.zig**: Element-wise **division** operation. 
- **op_multiplication.zig**: Element-wise **multiplication** operation. 
- **op_subtraction.zig**: Element-wise **subtraction** operation.
- **op_tanh.zig**: Implementation of the **Tanh** activation function.

## lib_shape_math
- **op_concatenate.zig**: Operation to **concatenate** tensors along a specified axis. Combines tensors by joining them along a dimension. For example, concatenating two tensors of shape `(3, 4)` along axis 0 results in a tensor of shape `(6, 4)`.
- **op_gather.zig**: Operation to **gather** elements from a tensor based on indices. Extracts specific elements or slices from a tensor. For example, gathering elements at indices `[1, 3, 5]` from a tensor.
- **op_identity.zig**: **Identity** operation that returns the same tensor. Often used as a placeholder or for testing. It ensures the input tensor is returned unchanged.
- **op_reg.zig**: **Regularization** operation. Adds a penalty term to the loss function to prevent overfitting in machine learning models. Common regularization techniques include L1 and L2 regularization.
- **op_padding.zig**: Operation to add **padding** to a tensor. Adds extra values (often zeros) around the edges of a tensor. For example, padding a `(3, 3)` tensor with 1 layer of zeros results in a `(5, 5)` tensor.
- **op_reinape.zig**: Operation to **reshape** a tensor to a new shape. Changes the dimensions of the tensor without altering its data. For example, reshaping a tensor of shape `(6)` to `(2, 3)`.
- **op_restax.zig**: Operation to **restore** the original shape of a tensor. Reverts a tensor to its previous shape after transformations. Useful for undoing reshape operations.
- **op_shape.zig**: Operation to retrieve the **shape** of a tensor. Returns the dimensions of the tensor. For example, a tensor of shape `(2, 3, 4)` returns `[2, 3, 4]`.
- **op_slice.zig**: Operation to **slice** a tensor. Extracts a portion of a tensor based on specified indices. For example, slicing a tensor along the first dimension to get a subset of rows.
- **op_spit.zig**: Operation to **split** a tensor into multiple tensors. Divides a tensor into smaller tensors along a specified axis. For example, splitting a tensor of shape `(6, 4)` into two tensors of shape `(3, 4)`.
- **op_transpose.zig**: Operation to **transpose** a tensor. Swaps the dimensions of a tensor, often used in matrix operations. For example, transposing a `(2, 3)` tensor results in a `(3, 2)` tensor.
- **op_unequeeze.zig**: Operation to **unsqueeze** a tensor. Adds a new dimension to the tensor, often used to match shapes for broadcasting. For example, unsqueezing a tensor of shape `(3)` along axis 0 results in a tensor of shape `(1, 3)`.

## Other Files
- **lib_logical_matrix.zig**: Logical operations on matrices, such as **AND**, **OR**, and **NOT**. Performs element-wise logical comparisons. For example, applying `A AND B` to two boolean tensors.
- **lib_reduction_matrix.zig**: **Reduction** operations on matrices, such as summing or averaging elements. Reduces the tensor along specified dimensions. For example, summing all elements of a tensor or averaging along a specific axis.
- **op_convolution.zig**: **Convolution** operation. Applies a filter to a tensor, commonly used in image processing and convolutional neural networks (CNNs). It slides a kernel over the input tensor to produce a feature map.
- **op_gamma.zig**: Operation to compute the **Gamma function**. A generalization of the factorial function, used in probability and statistics. It is defined as `Γ(n) = (n-1)!` for positive integers.
- **op_max_mul.zig**: Operation to perform **maximum multiplication**. Often used in optimization problems to find the maximum product of elements. For example, finding the maximum value in a tensor after multiplying elements.
- **op_pooling.zig**: **Pooling** operation. Reduces the spatial dimensions of data by applying a function (e.g., max or average) to subregions of the tensor. Commonly used in CNNs to downsample feature maps.
