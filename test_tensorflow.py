import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print("TensorFlow version:", tf.__version__)
print("TensorFlow is available:", tf.test.is_built_with_cuda()) 