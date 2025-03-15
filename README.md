# Z-Ant
![](https://github.com/ZIGTinyBook/Z-Ant/actions/workflows/zig-tests.yml/badge.svg)
![](https://github.com/ZIGTinyBook/Z-Ant/actions/workflows/zig-heavy-tests.yml/badge.svg)


![image](https://github.com/user-attachments/assets/6a5346e5-58ec-4069-8143-c3b7b03586f3)
## Project Overview

**Zant** (Zig-Ant) is an open-source SDK designed to simplify deploying Neural Networks (NN) on microcontrollers. Written in Zig, Zant prioritizes cross-compatibility and efficiency, providing tools to import, optimize, and deploy NNs seamlessly, tailored to specific hardware.

### Why Zant?

1. Many microcontrollers (e.g., ATMEGA, TI Sitara) lack robust deep learning libraries.
2. No open-source solution exists for end-to-end NN optimization and deployment.
3. Inspired by cutting-edge research (e.g., MIT Han Lab), we leverage state-of-the-art optimization techniques.
4. Collaborating with institutions like Politecnico di Milano to advance NN deployment on constrained devices.
5. Built for flexibility to adapt to new hardware without codebase changes.

### Key Features

- **Optimized Performance:** Supports quantization, pruning, and hardware acceleration (SIMD, GPU offloading).
- **Efficient Memory Usage:** Incorporates memory pooling, static allocation, and buffer optimization.
- **Cross-Platform Support:** Works on ARM Cortex-M, RISC-V, and more.
- **Ease of Integration:** Modular design with clear APIs, examples, and documentation.

### Use Cases

- **Real-Time Applications:** Object detection, anomaly detection, and predictive maintenance on edge devices.
- **IoT and Autonomous Systems:** Enable AI in IoT, drones, robots, and vehicles with constrained resources.

---

## Roadmap & Timeline

We are actively working towards achieving key milestones for Zant's development and optimization:

### **Short-Term Goals (Q1 2025)**

- **March 5, 2025:** Run MNIST inference on a Raspberry Pi Pico 2, importing models from ONNX.
- **April 30, 2025:** Get YOLO running efficiently on Raspberry Pi Pico 2.

### **Mid-Term Goals (Q2-Q3 2025)**

- Transition to a **Shape Tracker**, optimizing tensor operations.
- Develop a **frontend interface** for easier interaction with the library.
- Implement **im2tensor**, converting JPEG/PNG images into our optimized tensor format.
- Optimize **code generation**, allowing users to choose between **flash storage** or **RAM** for execution.
- Expand **ONNX operation support**, ensuring compatibility with more layers and architectures.

### **Long-Term Goals (Q3 2025)**

- Begin work on **pruning and quantization** to improve network efficiency.
- Support additional **microcontrollers and architectures**.
- Develop **benchmarking tools** for profiling model execution on constrained devices.
- Improve the **compiler backend**, integrating more advanced optimization passes.
- Enhance support for **real-time inference**, focusing on low-latency applications.

---

## Getting Started

### Prerequisites

1. Install the [latest Zig compiler](https://ziglang.org/learn/getting-started/).
2. Brush up your Zig skills with [Ziglings exercises](https://codeberg.org/ziglings/exercises).

### Running the Project

Navigate to the project folder and execute:

```sh
zig build run
```

### Testing

1. Add new test files to `build.zig/test_list` if not already listed.
2. Run:
   ```sh
   zig build
   zig build test --summary all
   ```
   *(Ignore stderr warnings.)*

To run all tests, including computationally heavy ones, use:

```sh
zig build test -Dheavy --summary all
```

### Documentation

Generated using [Zig's standard documentation format](https://ziglang.org/documentation/master/#Doc-Comments).

---

## Using the Library

To use Zant effectively, you can auto generate all the necessary code for running your neural network using a structured approach. Below is an example of how to define and configure a simple model.

### **1. Setup model directory**

Grab your model onnx file and place it in the `models/model_name` directory.
Remember that the onnx file name must match the model name folder.

### **2. Generate code**

Run the following command to generate the code for your model:

```bash
zig build codegen -Dmodel=model_name 
[ -Dlog -Duser_tests=path/to/user_tests.json ]
```
You can optionally add your own tests by providing a JSON file with the following structure:

```json
[
    {
        "name": "test_name",
        "input": [0.0, 1.0, 2.0],
        "output": [0.0, 1.0, 2.0]
    }
]
```

Input and output must match the model's input and output shapes.

After running the codegen command, you will find the generated code in the `generated/model_name` directory.
Inside this directory, you will find the following files:
- `lib_{model_name}.zig`: Contains the model definition and inference function.
- `test_{model_name}.zig`: Contains the model tests.
- `user_tests.json`: Contains the user-defined tests.
- `model_options.zig`: Contains the model configuration options.
- `static_parameters.zig`: Contains the model's static parameters like weights and biases.

### **3. Run model tests**

To run the tests, execute the following command:

```bash
zig build test-generated-lib -Dmodel=model_name
```

### **4. Integrate the model into your existing project**

To integrate the model into your project, you can build the static library binary by running:

```bash
zig build lib -Dmodel=model_name -Dtarget={target_architecture} -Dcpu={specific_cpu} -Doptimize=ReleaseFast
```

After that you will find the static library in the `zig-out/{model_name}/libzant.a` directory.

Finally you can link the library to your project.

For instance, if you are using a Raspberry Pi Pico, you can link the library by adding the following line to your `CMakeList.txt` file:

```cmake
target_link_libraries(your_project_name PUBLIC path/to/libzant.a)
```

And add the following lines in your C code:

```c
extern void setLogFunction(void (*log_function)(uint8_t *string)); // Mandatory only if you codegen with -Dlog flag
extern void predict(float *input, uint32_t *input_shape, uint32_t shape_len, float **result);

// Example of how to use the model
// ....
predict(input, input_shape, shape_len, &result);
// ....
```

---

## Docker

Follow the [Docker Guide](/docs/How_TO_DOCKER_101.md) for containerized usage.

## Join Us!

Contribute to Zant on [GitHub](#). Let’s make NN deployment on microcontrollers efficient, accessible, and open!

## Contributors

[View all contributors](https://github.com/ZIGTinyBook/Z-Ant/graphs/contributors)

