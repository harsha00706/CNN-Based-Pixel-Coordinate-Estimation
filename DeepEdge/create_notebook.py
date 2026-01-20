import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """# CNN-Based Pixel Coordinate Prediction

## Problem Statement
Predict the coordinates (x, y) of a pixel which has a value of 255 for 1 pixel in a given 50x50 pixel grayscale image and all other pixels are 0.

## Why This Is a Regression Problem
This task predicts continuous spatial coordinates (x, y), making it a regression problem. Treating it as a classification task would require 2500 classes (one for each pixel), which is inefficient and ignores the spatial relationship between adjacent pixels. Regression with Mean Squared Error (MSE) naturally handles the proximity of predictions to the target.

## Dataset and Approach
1. **Dataset Generation**: We generate 3000 synthetic images of size 50x50.
   - **Why Synthetic?**: The problem is deterministic and mathematically defined, so synthetic data perfectly represents the task without noise.
   - **Size Justification**: 3000 samples provide >1x coverage of the 2500 possible pixel locations. Uniform random sampling ensures all regions are adequately represented.
   - **Coordinate Convention**: (x, y) corresponds to (column index, row index), aligning with standard image processing frameworks (e.g., OpenCV, Matplotlib).

2. **Preprocessing**: Normalize pixel values to [0, 1] and coordinates to [0, 1].
3. **Model**: A simple Convolutional Neural Network (CNN) with 2 Convolutional layers, Max Pooling, and 2 Dense layers.
4. **Training**: Train for 10 epochs using MSE loss.
5. **Evaluation**: MAE (Mean Absolute Error) and **Euclidean Pixel Error**.
   - The validation set acts as a proxy test set given the synthetic nature of the data.
"""

code_imports = """import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
"""

code_dataset = """def generate_dataset(num_samples=3000, img_size=50):
    '''
    Generates a dataset of 50x50 grayscale images with one pixel set to 255.
    
    Args:
        num_samples: Number of images to generate.
        img_size: Size of the image (square).
        
    Returns:
        X: Array of images (num_samples, img_size, img_size, 1) normalized to [0, 1].
        y: Array of coordinates (num_samples, 2) normalized to [0, 1].
    '''
    X = np.zeros((num_samples, img_size, img_size, 1), dtype=np.float32)
    y = np.zeros((num_samples, 2), dtype=np.float32)
    
    for i in range(num_samples):
        # Random coordinate (row, col)
        row = np.random.randint(0, img_size)
        col = np.random.randint(0, img_size)
        
        X[i, row, col, 0] = 1.0 # Normalized 255 -> 1.0
        
        # Normalize coordinates to [0, 1]
        y[i, 0] = col / (img_size - 1) # x
        y[i, 1] = row / (img_size - 1) # y
        
    return X, y

IMG_SIZE = 50
NUM_SAMPLES = 3000

print("Generating dataset...")
X, y = generate_dataset(NUM_SAMPLES, IMG_SIZE)

# Split into train and validation
split_idx = int(NUM_SAMPLES * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
print(f"Val shapes: X={X_val.shape}, y={y_val.shape}")
"""

code_model = """def create_model(img_size=50):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2, activation='linear') # Output x, y (normalized)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = create_model(IMG_SIZE)
model.summary()
"""

code_train = """print("Training model...")
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)
"""

code_plot = """# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.show()
"""

code_eval = """# Evaluate
print("Evaluating...")
val_loss, val_mae = model.evaluate(X_val, y_val)
print(f"Validation MAE (normalized): {val_mae}")
print(f"Validation MAE (pixels): {val_mae * (IMG_SIZE - 1)}")

# Euclidean Pixel Error
# De-normalize coordinates
true_coords = y_val * (IMG_SIZE - 1)
pred_coords = model.predict(X_val) * (IMG_SIZE - 1)

# Calculate Euclidean distance for each sample
euclidean_errors = np.sqrt(np.sum((true_coords - pred_coords)**2, axis=1))

print(f"Mean Euclidean Error: {np.mean(euclidean_errors):.4f} pixels")
print(f"Max Euclidean Error: {np.max(euclidean_errors):.4f} pixels")

# Histogram of errors
plt.figure(figsize=(10, 4))
plt.hist(euclidean_errors, bins=30, edgecolor='k')
plt.title('Distribution of Euclidean Pixel Errors')
plt.xlabel('Error (pixels)')
plt.ylabel('Count')
plt.show()
"""

code_viz = """# Predictions and Visualization
y_pred = model.predict(X_val)

# Scatter plot of Ground Truth vs Predicted
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_val[:, 0] * (IMG_SIZE-1), y_pred[:, 0] * (IMG_SIZE-1), alpha=0.5)
plt.plot([0, IMG_SIZE], [0, IMG_SIZE], 'r--')
plt.xlabel('True X')
plt.ylabel('Predicted X')
plt.title('X Coordinate Prediction')

plt.subplot(1, 2, 2)
plt.scatter(y_val[:, 1] * (IMG_SIZE-1), y_pred[:, 1] * (IMG_SIZE-1), alpha=0.5)
plt.plot([0, IMG_SIZE], [0, IMG_SIZE], 'r--')
plt.xlabel('True Y')
plt.ylabel('Predicted Y')
plt.title('Y Coordinate Prediction')
plt.show()

# Visualize some images with predicted points
plt.figure(figsize=(15, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    # Reshape for display
    img = X_val[i].reshape(IMG_SIZE, IMG_SIZE)
    plt.imshow(img, cmap='gray')
    
    # Ground Truth
    true_x = y_val[i, 0] * (IMG_SIZE - 1)
    true_y = y_val[i, 1] * (IMG_SIZE - 1)
    
    # Prediction
    pred_x = y_pred[i, 0] * (IMG_SIZE - 1)
    pred_y = y_pred[i, 1] * (IMG_SIZE - 1)
    
    plt.scatter(true_x, true_y, c='green', marker='o', label='True')
    plt.scatter(pred_x, pred_y, c='red', marker='x', label='Pred')
    plt.title(f"T:({true_x:.1f},{true_y:.1f})\nP:({pred_x:.1f},{pred_y:.1f})")
    if i == 0:
        plt.legend()
plt.show()
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_code_cell(code_dataset),
    nbf.v4.new_code_cell(code_model),
    nbf.v4.new_code_cell(code_train),
    nbf.v4.new_code_cell(code_plot),
    nbf.v4.new_code_cell(code_eval),
    nbf.v4.new_code_cell(code_viz)
]

with open('solution.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Created solution.ipynb")

