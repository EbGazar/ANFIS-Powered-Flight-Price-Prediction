# ANFIS Flight Price Predictor
### *Price Forecasting using Adaptive Neuro-Fuzzy Inference Systems*

<div align="center">

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](docs/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

## Table of Contents
- [Overview](#overview)
- [Technical Architecture](#technical-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Training](#model-training)
- [Performance Optimization](#performance-optimization)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Overview

The ANFIS Flight Price Predictor is a sophisticated implementation of Adaptive Neuro-Fuzzy Inference Systems for accurate flight price forecasting. Built on TensorFlow 2.0+, it combines the interpretability of fuzzy logic with the learning capabilities of neural networks.

### Key Features

- **Custom ANFIS Architecture**: Five-layer neuro-fuzzy system with:
  - Gaussian membership functions
  - Product t-norm rule implementation
  - Normalized weighted average defuzzification
  - Hybrid learning algorithm

- **Advanced Data Processing**:
  ```python
  processor = DataProcessor(
      categorical_features=['airline', 'source', 'destination'],
      numerical_features=['duration', 'days_left', 'stops'],
      target='price'
  )
  ```

- **Production-Ready Model**:
  ```python
  model = ANFISModel(
      input_dim=X_train.shape[1],
      n_membership_functions=3,
      learning_rate=0.001
  )
  ```

## Technical Architecture

### Layer Implementation

1. **Fuzzy Layer**
```python
class FuzzyLayer(keras.layers.Layer):
    """Fuzzification using Gaussian membership functions"""
    def __init__(self, n_input, n_memb, memb_func='gaussian'):
        super().__init__()
        self.n = n_input
        self.m = n_memb
        self.memb_func = memb_func
```

2. **Rule Layer**
```python
class RuleLayer(keras.layers.Layer):
    """Rule generation using product t-norm"""
    def __init__(self, n_input, n_memb):
        super().__init__()
        self.n = n_input
        self.m = n_memb
```

3. **Normalization Layer**
```python
class NormLayer(keras.layers.Layer):
    """Rule strength normalization"""
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
```

4. **Defuzzification Layer**
```python
class DefuzzLayer(keras.layers.Layer):
    """Defuzzification with adaptive consequent parameters"""
    def __init__(self, n_input, n_memb):
        super().__init__()
        self.n = n_input
        self.m = n_memb
```

## Installation

### Prerequisites
- Python 3.9+
- TensorFlow 2.0+
- CUDA-capable GPU (recommended)

### Setup

1. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify Installation**
```bash
python -c "import anfis; print(anfis.__version__)"
```

## Quick Start

### Basic Usage

```python
from anfis import ANFISModel
from data import DataProcessor

# Prepare data
processor = DataProcessor()
X_train, X_test, y_train, y_test = processor.prepare_data('flight_data.csv')

# Create and train model
model = ANFISModel(input_dim=X_train.shape[1])
model.fit(X_train, y_train, epochs=100)

# Make predictions
predictions = model.predict(X_test)
```

### Advanced Configuration

```python
# Custom model configuration
model = ANFISModel(
    input_dim=X_train.shape[1],
    n_membership_functions=3,
    learning_rate=0.001,
    membership_function='gaussian',
    optimizer='adam',
    loss='huber'
)

# Training configuration
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10),
        tf.keras.callbacks.ModelCheckpoint('best_model.h5')
    ]
)
```

## Model Training

### Data Preprocessing

```python
def prepare_data(df):
    """
    Prepare data for ANFIS model
    
    Args:
        df (pd.DataFrame): Raw flight data
        
    Returns:
        tuple: Processed features and target
    """
    # Feature engineering
    processor = DataProcessor(
        categorical_features=['airline', 'source', 'destination'],
        numerical_features=['duration', 'days_left', 'stops'],
        target='price'
    )
    
    return processor.transform(df)
```

### Training Process

```python
# Initialize model with optimal parameters
model = ANFISModel(
    input_dim=X_train.shape[1],
    n_membership_functions=3,
    learning_rate=0.001
)

# Train with validation
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[
        EarlyStopping(patience=10),
        ModelCheckpoint('best_model.h5'),
        ReduceLROnPlateau(factor=0.1, patience=5)
    ]
)

# Evaluate performance
metrics = model.evaluate(X_test, y_test)
print(f"Test RMSE: {metrics['rmse']:.2f}")
```

### Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| R² Score | 0.89 | Coefficient of determination |
| MAPE | 12.3% | Mean Absolute Percentage Error |
| RMSE | $43.21 | Root Mean Square Error |
| MAE | $32.54 | Mean Absolute Error |

## Performance Optimization

### Hyperparameter Tuning

```python
from anfis.tuning import ANFISOptimizer

# Define parameter search space
param_space = {
    'n_membership_functions': [2, 3, 4],
    'learning_rate': [0.1, 0.01, 0.001],
    'batch_size': [16, 32, 64],
    'optimizer': ['adam', 'rmsprop'],
    'loss': ['mse', 'huber']
}

# Initialize optimizer
optimizer = ANFISOptimizer(
    model_class=ANFISModel,
    param_space=param_space,
    n_trials=50
)

# Find best parameters
best_params = optimizer.optimize(X_train, y_train)
print(f"Best parameters: {best_params}")
```

### Memory Optimization

```python
# Enable memory growth for GPU
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Use mixed precision training
tf.keras.mixed_precision.set_global_policy('mixed_float16')
```

### Batch Processing for Large Datasets

```python
def batch_generator(X, y, batch_size=32):
    """Generate batches for large datasets"""
    idx = 0
    while True:
        if idx + batch_size > len(X):
            idx = 0
        yield X[idx:idx + batch_size], y[idx:idx + batch_size]
        idx += batch_size

# Train using generator
model.fit_generator(
    generator=batch_generator(X_train, y_train),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=100
)
```

## Model Visualization

### Membership Functions

```python
def plot_membership_functions(model, feature_names):
    """
    Visualize membership functions for each input feature
    
    Args:
        model: Trained ANFIS model
        feature_names: List of feature names
    """
    fuzzy_layer = model.layers[0]
    mus = fuzzy_layer.mu.numpy()
    sigmas = fuzzy_layer.sigma.numpy()
    
    plt.figure(figsize=(15, 5*len(feature_names)))
    for i, feature in enumerate(feature_names):
        plt.subplot(len(feature_names), 1, i+1)
        x = np.linspace(-3, 3, 1000)
        for j in range(model.n_memb):
            y = np.exp(-0.5 * ((x - mus[j,i])/sigmas[j,i])**2)
            plt.plot(x, y, label=f'MF {j+1}')
        plt.title(f'Membership Functions: {feature}')
        plt.legend()
    plt.tight_layout()
    plt.show()
```

### Training History

```python
def plot_training_history(history):
    """Visualize training metrics"""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.legend()
    
    # Plot metrics
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='MAE')
    plt.plot(history.history['mse'], label='MSE')
    plt.title('Model Metrics')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
```

## API Reference

### ANFISModel

```python
class ANFISModel:
    """
    Main ANFIS model implementation
    
    Args:
        input_dim (int): Number of input features
        n_membership_functions (int): Number of membership functions per feature
        learning_rate (float): Learning rate for optimization
        membership_function (str): Type of membership function
        optimizer (str): Optimizer name
        loss (str): Loss function name
        
    Methods:
        fit: Train the model
        predict: Make predictions
        evaluate: Evaluate model performance
        save: Save model weights
        load: Load model weights
    """
```

### DataProcessor

```python
class DataProcessor:
    """
    Data preprocessing utility
    
    Args:
        categorical_features (list): Names of categorical features
        numerical_features (list): Names of numerical features
        target (str): Name of target variable
        
    Methods:
        prepare_data: Process raw data
        transform: Transform new data
        inverse_transform: Reverse transformations
    """
```

## Troubleshooting Guide

### Common Issues

1. **Memory Errors**
```python
# Solution: Enable memory growth
tf.config.experimental.set_memory_growth(gpu, True)
```

2. **Convergence Issues**
```python
# Solution: Adjust learning rate and batch size
model = ANFISModel(
    learning_rate=0.001,  # Try smaller learning rate
    batch_size=32        # Adjust batch size
)
```

3. **Overfitting**
```python
# Solution: Add regularization
model.add_regularization(l2=0.01)
```

### Error Messages

| Error | Solution |
|-------|----------|
| `OutOfMemoryError` | Reduce batch size or enable memory growth |
| `ValueError: NaN loss` | Reduce learning rate or clip gradients |
| `ResourceExhaustedError` | Free GPU memory or use CPU training |

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/username/anfis-predictor.git
cd anfis-predictor

# Create development environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

### Code Style

We use [Black](https://github.com/psf/black) for code formatting:

```bash
# Format code
black src/

# Check style
flake8 src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{anfis_predictor,
    title = {ANFIS Flight Price Predictor},
    author = {Ebrahim Elgazar},
    year = {2024},
    url = {https://github.com/EbGazar/ANFIS-Powered-Flight-Price-Prediction}
}
```

## Acknowledgments

- TensorFlow team for the deep learning framework
- The fuzzy logic and neural network research community

---

<div align="center">
Made with ❤️
</div>
