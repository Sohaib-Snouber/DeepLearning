# Deep Learning Sandbox

This repository contains a simple setup to experiment with deep learning models using [PyTorch](https://pytorch.org/). The included example trains a multilayer perceptron (MLP) on the MNIST handwritten digit dataset.

## Getting Started

1. Create a Python virtual environment (optional but recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```bash
   python src/train.py --epochs 5
   ```

Model weights will be saved to `mnist_mlp.pth` after training.

Feel free to modify the model architecture in `src/model.py` or extend the training script for your own experiments.
