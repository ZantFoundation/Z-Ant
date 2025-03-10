# Zant Code Generation

The purpose of Zant code generation is to create a zig file containing some methods that can be than exported into a static library.  

The Starting file is `main.zig`, where the onnx model is parsed and passed to the code generator.

## Structures
### ReadyNode
- nodeProto: pointer to an onnx node. I decide to keep it so tho have a reference to the onnx correspondant.  
- inputs: list of pointers to ReadyTensors representing the inputs of the node, all the ReadyTensors are inside the tensorHashMap.
- outputs: list of pointers to ReadyTensors representing the outputs of the node, all the ReadyTensors are inside the tensorHashMap.
- ready: this flag is set to true if the node has all the inputs and all the outputs are set as ready.  

### ReadyTensor
- name: name of the tensor as presented in the onnx file (I sanitize the name only when writing it on the generated file, it is the last step).  
- ready: is true only if the Tensopor is a constant, an initializer or if it has been previously computed.  
- shape: shape of the tensor. The shape is computed during the creation of the ReadyGraph.

### readyGraph
ArrayList containing all the nodes of the graph. Global var.

### tensorHashMap
StringHashMap containing all the Tensors used in the NN. Here is contained the only copy of the ReadyTensor.  
    **key**: ReadyTensor.name  
    **value**: ReadyTensor  
Use a pointer to ReadyTensor when working on it. Pay attention `hasMap.get("key")` returns a copy of the value, call `hashMap.getPtr("key")` for the pointer to the value, otherwise any modification is not valid.

## The Algorithm
The crucial part of the code generation is inside [predict.zig](predict.zig).

First, in main, I populate tensorHashMap by calling `try createReadyTensorHashMap(model)`. In this step, if the shapes are not already present, they are initialized to {1, 1, 1, 1}.

Next, I create readyGraph with `try createReadyGraph(model)`. During this phase, the output shapes of each node are also computed.

Then, the initializers are written to the generated file, along with the `predict() `function.

Inside predict(), a sequence of instructions must be written to propagate the input through the network. A node is considered "Computable" if all its input tensors are "ready" and the node itself is not yet "ready". A tensor is considered "ready" if it is a constant (such as layer weights) or if it has been returned as the output of a node that has already been computed. When a node has been computed (i.e., its operation has been written in the generated code), it is marked as "ready".

The algorithm consists of writing the operations of all "computable" nodes until all nodes in the graph are ready.
  
1. get computable_nodes.
2. write-append the mathmatical operation of each computable node in the predict() and set the node and its outputs tensor to ready.
3. get new computable nodes.
4. if (new_computable_nodes.len > 0) go to step 2 else exit.



