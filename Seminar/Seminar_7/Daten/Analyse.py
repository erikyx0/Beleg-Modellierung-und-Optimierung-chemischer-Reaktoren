import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers, models

# set path to file
os.chdir(os.path.dirname(__file__))

# Data Path
data_path = "S7/Daten/"
np.random.seed(42)
tf.random.set_seed(42)

# Load Files
train_normal_path = os.path.join(data_path, "Train_Normal.npy")
val_normal_path = os.path.join(data_path, "Val_Normal.npy")

X_train = np.load(train_normal_path)
X_val = np.load(val_normal_path)

print(f"Loaded Train_Normal: shape = {X_train.shape}")
print(f"Loaded Val_Normal: shape = {X_val.shape}")