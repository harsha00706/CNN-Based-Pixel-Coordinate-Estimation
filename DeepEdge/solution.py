import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def generate_dataset(num_samples=3000, img_size=50):
    X = np.zeros((num_samples, img_size, img_size, 1), dtype=np.float32)
    y = np.zeros((num_samples, 2), dtype=np.float32)
    
    for i in range(num_samples):
        # Random coordinate (row, col) ie (y, x)
        # Note: problem statement says "predict coordinates (x,y)". conventional image coordinates: x=col, y=row.
        # Let's stick to (x, y) = (col, row) order for output.
        row = np.random.randint(0, img_size)
        col = np.random.randint(0, img_size)
        
        X[i, row, col, 0] = 1.0 # Normalized 255 -> 1.0
        
        # Normalize coordinates to [0, 1] for better model stability
        y[i, 0] = col / (img_size - 1) # x
        y[i, 1] = row / (img_size - 1) # y
        
    return X, y

def create_model(img_size=50):
    """
    Creates a simple CNN model.
    """
    model = models.Sequential([
        # Convolutional layers to detect features (the bright pixel)
        # A simple filter should easily pick up the non-zero value
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

def main():
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
    
    print("Creating model...")
    model = create_model(IMG_SIZE)
    model.summary()
    
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Evaluate
    print("Evaluating...")
    val_loss, val_mae = model.evaluate(X_val, y_val)
    print(f"Validation MAE (normalized): {val_mae}")
    print(f"Validation MAE (pixels): {val_mae * (IMG_SIZE - 1)}")
    
    # Predictions
    y_pred = model.predict(X_val[:5])
    print("\nSample Predictions (x, y pixels):")
    print("True vs Pred")
    for true, pred in zip(y_val[:5], y_pred):
        true_px = true * (IMG_SIZE - 1)
        pred_px = pred * (IMG_SIZE - 1)
        print(f"{true_px.astype(int)} vs {pred_px}")

if __name__ == "__main__":
    main()
