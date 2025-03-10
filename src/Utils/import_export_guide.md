
## Tensor  
Given a tensor of type T, where T is a zig type, its values will be reported in the following order and types:

- **usize** : `tensor.size`
- **usize** : shapeLenght, representing tensor.shape length
- **usize** : `tensor.shape[i]`for shapeLenght times
- **T** : `tensor.data[i]` for tensor.size times

## Layer  
- **[10]u8** :  a `string` tag representing the type of layer  
Depending on the type of layer see the relative format. See [Layer tags](#Layer-tags)

### Activation Layer   
- **usize** : `n_inputs`
- **usize** : `n_neurons`
- **[10]u8** : `activationFunction`, see [Activation Function tags](#Activation-Function-tags)

### Dense Layer  
- **Tensor** : `weights` tensor
- **Tensor** : `bias` tensor
- **usize** : `n_inputs`
- **usize** : `n_neurons`

### Convolutional Layer  
- **Tensor** : `weights` tensor  
- **Tensor** : `bias` tensor  
- **usize** : `input_channels`  
- **[4]usize** : `kernel_shape`  
- **[2]usize** : `stride`  

### Flatten Layer  
Nothing relevant to export.

### Pooling Layer  
- **[2]usize** : `kernel`  
- **[2]usize** : `stride`  
- **[3]u8** : `poolingType`, see [Pooling Type tags](#Pooling-Type-tags)

## Model  
- **usize** : NoLayer, representing the number of layers in the model
- **Layer** : representing a layer. See Layer above.

### Tags
#### Activation Function tags
len = 10  
- "ReLU......"
- "Sigmoid..."
- "Softmax..."
- "None......"
#### Layer tags
len = 10  
- "Dense....."
- "Activation"
- "Convol...."
- "Flatten..."  
- "Pooling..."  
#### Pooling Type tags
len = 3 
- "Max"  
- "Min"  
- "Avg"  




