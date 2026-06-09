# Neural Network from Scratch

Simple neural network implementation for MNIST digit classification, built from scratch using NumPy.

Based on the implementation by Fajrul Fx, inspired by Michael Nielsen's book *Neural Networks and Deep Learning*. This project includes several adjustments and modifications for learning purposes and experimentation.

## Original References

- YouTube: https://youtu.be/WLmY9icEOQk?si=ni6BE5CuF5W9nwrp
- GitHub: https://github.com/fajrulfx/neural_network

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Network

#### Option A: Jupyter Notebook

```bash
jupyter notebook train.ipynb
```

Run all cells to train the neural network and save the trained model.

#### Option B: Python Directly

```python
import network
import mnist_loader

net = network.Network([784, 16, 16, 10])  # Adjust according to your architecture

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net.train(
    training_data,
    epochs=30,
    mini_batch_size=10,
    eta=3.0,
    test_data=test_data
)

net.save("trained_network.pkl")
```

### Test with Web App

```bash
streamlit run app.py
```

Draw digits and view prediction results directly in your browser.

## Files

| File | Description |
|--------|-------------|
| `network.py` | Neural network implementation |
| `mnist_loader.py` | MNIST dataset loader |
| `train.ipynb` | Training notebook |
| `app.py` | Streamlit web interface |
| `data/mnist.pkl.gz` | MNIST dataset |

## Network Architecture Example

- Input Layer: 784 neurons (28×28 pixels)
- Hidden Layer 1: 16 neurons
- Hidden Layer 2: 16 neurons
- Output Layer: 10 neurons (digits 0–9)
- Activation Function: Sigmoid

## Notes

This project is based on the educational implementation created by **Fajrul Fx**, which itself adapts concepts from Michael Nielsen's *Neural Networks and Deep Learning*. Several modifications and adjustments have been made in this version to better suit my learning objectives, experimentation, and project requirements.