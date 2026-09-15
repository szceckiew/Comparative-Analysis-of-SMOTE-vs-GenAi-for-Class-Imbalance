import torch

print(torch.cuda.is_available())
print(torch.version.cuda)

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))