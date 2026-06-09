import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
from pathlib import Path

# ==============================
# Paths
# ==============================

data_path = Path("model")
model_path = data_path / "anomaly_detection_model.h5"

# ==============================
# Load model
# ==============================

print("Loading trained autoencoder model...")

if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")

autoencoder = load_model(model_path, compile=False)

print("Model loaded successfully!")

# ============================================
# 2. load testing data
# ============================================
# Load test datasets
test_normal_path = os.path.join(data_path, "Test_Normal.npy")
test_abnormal_path = os.path.join(data_path, "train_Abnormal.npy")

test_normal = np.load(test_normal_path)
test_abnormal = np.load(test_abnormal_path)

print(f"\nLoaded Test_Normal: shape = {test_normal.shape}")
print(f"Loaded Test_Abnormal: shape = {test_abnormal.shape}")

# ============================================
# 3. Preprocess the data to be the same like the steps used in the training part
# ============================================
# Normalize and add channel dimension (same as training)
test_normal = test_normal.astype('float32') / 255.0
test_abnormal = test_abnormal.astype('float32') / 255.0

test_normal = np.expand_dims(test_normal, axis=-1)
test_abnormal = np.expand_dims(test_abnormal, axis=-1)

# ============================================
# 4. Reconstruct images
# ============================================
print("\nReconstructing images...")

# Get reconstructions for both datasets
normal_reconstructions = autoencoder.predict(test_normal, verbose=0)
abnormal_reconstructions = autoencoder.predict(test_abnormal, verbose=0)

print("Reconstructions completed!")

# ============================================
# 5. Calculate reconstruction errors
# ============================================
# Calculate MSE for each image
def calculate_mse(original, reconstructed):
    # Mean Squared Error per image
    return np.mean((original - reconstructed) ** 2, axis=(1, 2, 3))

normal_errors = calculate_mse(test_normal, normal_reconstructions)
abnormal_errors = calculate_mse(test_abnormal, abnormal_reconstructions)

print(f"\nReconstruction Errors:")
print(f"Normal Test: Mean = {normal_errors.mean():.6f}, Std = {normal_errors.std():.6f}")
print(f"Abnormal Test: Mean = {abnormal_errors.mean():.6f}, Std = {abnormal_errors.std():.6f}")

# ============================================
# 6. SIMPLE ANOMALY DETECTION
# ============================================
print("\n" + "="*50)
print("SIMPLE ANOMALY DETECTION ANALYSIS")
print("="*50)

# Calculate threshold
threshold = normal_errors.mean() + 2 * normal_errors.std()
print(f"\nSuggested Threshold (Normal Mean + 2*Std): {threshold:.6f}")

# Classify based on threshold
normal_predictions = normal_errors > threshold
abnormal_predictions = abnormal_errors > threshold

print(f"\nNormal Test Images:")
print(f"  - Classified as Normal: {(~normal_predictions).sum()}/{len(normal_predictions)}")
print(f"  - Classified as Abnormal: {normal_predictions.sum()}/{len(normal_predictions)}")

print(f"\nAbnormal Test Images:")
print(f"  - Classified as Normal: {(~abnormal_predictions).sum()}/{len(abnormal_predictions)}")
print(f"  - Classified as Abnormal: {abnormal_predictions.sum()}/{len(abnormal_predictions)}")

# Calculate accuracy metrics
true_normal = (~normal_predictions).sum()
false_abnormal = normal_predictions.sum()
true_abnormal = abnormal_predictions.sum()
false_normal = (~abnormal_predictions).sum()

accuracy = (true_normal + true_abnormal) / (len(normal_predictions) + len(abnormal_predictions))
precision = true_abnormal / (true_abnormal + false_abnormal) if (true_abnormal + false_abnormal) > 0 else 0
recall = true_abnormal / (true_abnormal + false_normal) if (true_abnormal + false_normal) > 0 else 0

print(f"\nPerformance Metrics:")
print(f"  Accuracy: {accuracy:.2%}")
print(f"  Precision: {precision:.2%}")
print(f"  Recall: {recall:.2%}")

# ============================================
# 7. VISUALIZATION: Scatter plot
# ============================================
plt.figure(figsize=(14, 7))

# Combine all errors for scatter plot
all_errors = np.concatenate([normal_errors, abnormal_errors])
all_indices = np.arange(len(all_errors))

# Create colors array
colors = ['blue'] * len(normal_errors) + ['red'] * len(abnormal_errors)

# Create scatter plot
plt.scatter(all_indices, all_errors, c=colors, alpha=0.6, s=30)

# Add threshold line
plt.axhline(y=threshold, color='green', linestyle='--', linewidth=2,
            label=f'Threshold: {threshold:.6f}')

# Add separation line
plt.axvline(x=len(normal_errors)-0.5, color='black', linestyle=':',
            linewidth=1.5, alpha=0.5)

plt.title('Reconstruction Error Scatter Plot', fontsize=16, fontweight='bold')
plt.xlabel('Image Index', fontsize=12)
plt.ylabel('Reconstruction Error (MSE)', fontsize=12)

# Create custom legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='blue', alpha=0.7, label=f'Normal ({len(normal_errors)} images)'),
                   Patch(facecolor='red', alpha=0.7, label=f'Abnormal ({len(abnormal_errors)} images)'),
                   plt.Line2D([0], [0], color='green', linestyle='--',
                             label=f'Threshold: {threshold:.6f}')]

plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Plot_Results.png", dpi=500)

