# Deep Learning Pixel Coordinate Prediction

This project implements a Deep Learning model to predict the (x, y) coordinates of a randomly assigned active pixel (value 255) in a 50x50 grayscale image.

## Project Structure
- `solution.ipynb`: The main Jupyter Notebook containing dataset generation, model training, and evaluation.
- `solution.py`: A Python script version of the solution for quick execution.
- `requirements.txt`: Python dependencies.
- `create_notebook.py`: Helper script to regenerate the notebook (optional).

## Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) If you don't have Jupyter installed, it will be installed via the requirements.

## Usage

### Option 1: Jupyter Notebook
Open the notebook to view the code, training process, and visualization:
```bash
jupyter notebook solution.ipynb
```

### Option 2: Python Script
Run the solution directly from the command line:
```bash
python solution.py
```

## Approach
- **Dataset**: 3000 synthetic 50x50 grayscale images.
- **Model**: A Convolutional Neural Network (CNN) with 2 Conv2D layers and 2 Dense layers.
- **Results**: The model achieves a Mean Absolute Error (MAE) of < 1 pixel on the validation set.
