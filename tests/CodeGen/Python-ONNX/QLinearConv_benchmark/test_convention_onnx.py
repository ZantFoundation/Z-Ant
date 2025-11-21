import onnx
from onnx import numpy_helper

m_nchw = onnx.load("Models/QLinearConv_0.onnx")
m_nhwc = onnx.load("Models/QLinearConv_0_NHWC.onnx")

w_name = "weight_5506"  # from metadata["param_names"]["w"]

w_nchw = next(numpy_helper.to_array(i) for i in m_nchw.graph.initializer if i.name == w_name)
w_nhwc = next(numpy_helper.to_array(i) for i in m_nhwc.graph.initializer if i.name == w_name)

print("NCHW weights shape:", w_nchw.shape)  # OIHW
print("NHWC weights shape:", w_nhwc.shape)  # OHWI

print("\nZANT weights in onnx:\n")
print(w_nchw)

print("\nCMSIS weights in onnx:\n")
print(w_nhwc)

