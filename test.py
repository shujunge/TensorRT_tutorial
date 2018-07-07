
from tensorflow.examples.tutorials.mnist import input_data
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from time import time #import system tools

import uff


G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)
engine = trt.utils.load_engine(G_LOGGER, "./tf_mnist.engine")

MNIST_DATASETS = input_data.read_data_sets('datasets/')
# MNIST_DATASETS=tf.data.Dataset.
img, label = MNIST_DATASETS.test.next_batch(100)

Start=time()
#convert input data to Float32
img = img.astype(np.float32)
print("img shape:",img.shape)
runtime = trt.infer.create_infer_runtime(G_LOGGER)
context = engine.create_execution_context()
output = np.empty((100,10), dtype = np.float32)
#alocate device memory
d_input = cuda.mem_alloc(100* img[0].size * img[0].dtype.itemsize)
d_output = cuda.mem_alloc(100 * output[0].size * output[0].dtype.itemsize)
bindings = [int(d_input), int(d_output)]

stream = cuda.Stream()
#transfer input data to device
cuda.memcpy_htod_async(d_input, img, stream)
#execute model
context.enqueue(100, bindings, stream.handle, None)
#transfer predictions back
cuda.memcpy_dtoh_async(output, d_output, stream)
#syncronize threads
stream.synchronize()
print("output shape:",output.shape)
print("output:",output)
print("Test Case: " + str(label))
print ("Prediction: " + str(np.argmax(output,axis=1)))
print("tensorrt time:",time()-Start)

context.destroy()
engine.destroy()
runtime.destroy()
