from keras import backend as K

from layers.attention_layer import Attention
import numpy as np

layer = Attention((64, 3), input_shape=(64, 3))
layer.build((64, 3))


x = np.ones((64, 3), dtype="float32")

output = K.eval(layer.call(x))

print(output)
print("Shape: ")
print(output.shape)