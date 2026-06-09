import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers, models
from pathlib import Path

# ==============================
# 1. Data Path
# ==============================
data_path = Path("Daten")
# Set random Seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ==============================
# 2. Load only necassary files
# ==============================

# Load train and validation normal datasets
train_normal_path = os.path.join(data_path, "Train_Normal.npy")
val_normal_path = os.path.join(data_path, "Val_Normal.npy")

# Load the Data
X_train = np.load(train_normal_path)
X_val = np.load(val_normal_path)
print(X_train.shape)
print(X_val.shape)

# ==============================
# 3. Preparing Data
# ==============================

# Normalize images to range 0,1
X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0

# Add channel dimension (grayscale images have 1 Channel)
X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)

print(X_train.shape)
print(X_val.shape)

# ==============================
# 4. create Autoencoder Model
# ==============================

# Get Image dimensions from training data
input_shape = X_train.shape[1:]
print(f"Input shape: {input_shape}")

# Build Encoder
encoder_input = layers.Input(shape=input_shape, name = 'encoder_input')

# Encoder
conv1 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(encoder_input)
pool1 = layers.MaxPooling2D(pool_size=(2,2))(conv1)

conv2 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(pool1)
pool2 = layers.MaxPooling2D(pool_size=(2,2))(conv2)

conv3 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(pool2)
encoded = layers.MaxPooling2D(pool_size=(2,2))(conv3)

# Decoder
conv4 = layers.Conv2D(8,(3,3), activation='relu', padding='same')(encoded)
up1 = layers.UpSampling2D(size=(2,2))(conv4)

conv5 = layers.Conv2D(8,(3,3), activation='relu', padding='same')(up1)
up2 = layers.UpSampling2D(size=(2,2))(conv5)

conv6 = layers.Conv2D(8,(3,3), activation='relu', padding='same')(up2)
up3 = layers.UpSampling2D(size=(2,2))(conv6)

decoder_output = layers.Conv2D(1, (3,3), activation='sigmoid', padding='same', name='decoder_output')(up3)

autoencoder = models.Model(encoder_input, decoder_output, name='autoencoder')
autoencoder.compile(optimizer='adam', loss='mse')

print(autoencoder.summary())
autoencoder.summary()

# ==============================
# Train on Normal Data Only
# ==============================

print(f"Training on Train_Normal: {X_train.shape[0]} normal images ")
print(f"Training on Val_Normal: {X_val.shape[0]} normal images ")

# Train Autoencoder
history = autoencoder.fit(
    X_train, X_train,
    validation_data=(X_val, X_val), #Autoencoder: input = output (reconstruction)
    epochs=2500,
    batch_size=32,
    verbose=1)

# ==============================
# 6. Visualize Training History
# ==============================

plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==============================
# 7. Save the Model
# ==============================

model_path = os.path.join(data_path, "anomaly_detection_model.h5")
autoencoder.save(model_path)
print(f"Model saved to {model_path}")