# -*- coding: utf-8 -*-
"""
ANFIS (Adaptive Neuro-Fuzzy Inference System) Implementation
This implementation combines fuzzy logic with neural networks for accurate price prediction.
Key components:
1. Fuzzy Layer: Creates membership functions
2. Rule Layer: Generates fuzzy rules
3. Normalization Layer: Normalizes rule strengths
4. Defuzzification Layer: Converts fuzzy to crisp outputs
5. Summation Layer: Produces final prediction
"""

# Import required libraries
import logging
logging.getLogger('tensorflow').disabled = True
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # remove WARNING Messages

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy import stats
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# Parameter class for FIS parameters
class fis_parameters():
    def __init__(self, n_input: int = 3, n_memb: int = 3, batch_size: int = 16, 
                 n_epochs: int = 100, memb_func: str = 'gaussian', 
                 optimizer: str = 'adam', loss: str = 'mse',
                 learning_rate: float = 0.001):
        self.n_input = n_input      # Number of input features
        self.n_memb = n_memb        # Number of membership functions
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.memb_func = memb_func  # 'gaussian'
        self.optimizer = optimizer
        self.loss = loss
        self.learning_rate = learning_rate

# Custom weight initializer
def equally_spaced_initializer(shape, minval=-1.5, maxval=1.5, dtype=tf.float32):
    """Initialize weights with equally spaced values"""
    linspace = tf.reshape(tf.linspace(minval, maxval, shape[0]), (-1, 1))
    weights = tf.tile(linspace, (1, shape[1]))
    return tf.Variable(weights, dtype=dtype)

# Layer 1: Fuzzy Layer
class FuzzyLayer(keras.layers.Layer):
    def __init__(self, n_input, n_memb, memb_func='gaussian', **kwargs):
        super(FuzzyLayer, self).__init__(**kwargs)
        self.n = n_input
        self.m = n_memb
        self.memb_func = memb_func

    def build(self, batch_input_shape):
        if self.memb_func == 'gaussian':
            self.mu = self.add_weight(
                name='mu',
                shape=(self.m, self.n),
                initializer=equally_spaced_initializer,
                trainable=True
            )
            self.sigma = self.add_weight(
                name='sigma',
                shape=(self.m, self.n),
                initializer=keras.initializers.RandomUniform(
                    minval=0.7, maxval=1.3, seed=42
                ),
                trainable=True,
                constraint=keras.constraints.NonNeg()
            )
        super(FuzzyLayer, self).build(batch_input_shape)

    def call(self, x_inputs):
        if self.memb_func == 'gaussian':
            return tf.exp(-1 * tf.square(tf.subtract(
                tf.reshape(tf.tile(x_inputs, (1, self.m)), 
                (-1, self.m, self.n)), self.mu)) / (2 * tf.square(self.sigma)))

# Layer 2: Rule Layer
class RuleLayer(keras.layers.Layer):
    def __init__(self, n_input, n_memb, **kwargs):
        super(RuleLayer, self).__init__(**kwargs)
        self.n = n_input
        self.m = n_memb

    def call(self, input_):
        input_reshaped = tf.reshape(input_, [-1, self.m, self.n])
        return tf.reduce_prod(input_reshaped, axis=2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.m)

# Layer 3: Normalization Layer
class NormLayer(keras.layers.Layer):
    def __init__(self, epsilon=1e-7, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, w):
        w_sum = tf.reduce_sum(w, axis=1, keepdims=True)
        return w / (w_sum + self.epsilon)

# Layer 4: Enhanced Defuzzification Layer
class DefuzzLayer(keras.layers.Layer):
    def __init__(self, n_input, n_memb, **kwargs):
        super().__init__(**kwargs)
        self.n = n_input
        self.m = n_memb

        self.CP_bias = self.add_weight(
            name='Consequence_bias',
            shape=(1, self.m),
            initializer=keras.initializers.RandomUniform(minval=-2, maxval=2),
            trainable=True
        )
        self.CP_weight = self.add_weight(
            name='Consequence_weight',
            shape=(self.n, self.m),
            initializer=keras.initializers.RandomUniform(minval=-2, maxval=2),
            trainable=True
        )

    def call(self, inputs):
        w_norm, input_ = inputs  # Unpack the input list
        expanded_input = tf.expand_dims(input_, 2)
        expanded_weight = tf.expand_dims(self.CP_weight, 0)
        prod = tf.reduce_sum(expanded_input * expanded_weight, axis=1)
        return w_norm * (prod + self.CP_bias)

# Layer 5: Summation Layer
class SummationLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input_):
        return tf.reduce_sum(input_, axis=1, keepdims=True)

# Main ANFIS Class
class ANFIS:
    def __init__(self, n_input: int, n_memb: int, batch_size: int = 16, 
                 memb_func: str = 'gaussian', name: str = 'MyAnfis'):
        self.n = n_input
        self.m = n_memb
        self.batch_size = batch_size
        self.memb_func = memb_func
        
        # Build enhanced ANFIS model
        inputs = keras.layers.Input(shape=(n_input,), name='inputLayer')
        normalized = keras.layers.BatchNormalization()(inputs)
        
        L1 = FuzzyLayer(n_input, n_memb, memb_func, name='fuzzyLayer')(normalized)
        L2 = RuleLayer(n_input, n_memb, name='ruleLayer')(L1)
        L3 = NormLayer(name='normLayer')(L2)
        L4 = DefuzzLayer(n_input, n_memb, name='defuzzLayer')(L3, normalized)
        L5 = SummationLayer(name='sumLayer')(L4)
        
        output = keras.layers.Dropout(0.1)(L5)
        
        self.model = keras.Model(inputs=[inputs], outputs=[output], name=name)
        self.update_weights()

    def __call__(self, X):
        return self.model.predict(X, batch_size=self.batch_size)

    def update_weights(self):
        if self.memb_func == 'gaussian':
            self.mus, self.sigmas = self.model.get_layer('fuzzyLayer').get_weights()
        self.bias, self.weights = self.model.get_layer('defuzzLayer').get_weights()

    def plotmfs(self, feature_names=None):
        """Plot membership functions"""
        n_input = self.n
        n_memb = self.m

        mus, sigmas = np.around(self.model.get_layer('fuzzyLayer').get_weights(), 2)
        mus = mus.reshape((n_memb, n_input, 1))
        sigmas = sigmas.reshape(n_memb, n_input, 1)

        xn = np.linspace(np.min(mus) - 2 * np.max(abs(sigmas)),
                       np.max(mus) + 2 * np.max(abs(sigmas)), 100).reshape((1, 1, -1))
        xn = np.tile(xn, (n_memb, n_input, 1))
        memb_curves = np.exp(-np.square((xn - mus)) / (2 * np.square(sigmas)))

        fig, axs = plt.subplots(nrows=n_input, ncols=1, figsize=(10, 3*n_input))
        fig.suptitle('Membership Functions for Each Input Feature', size=16)
        
        if n_input == 1:
            axs = [axs]
        
        colors = plt.cm.viridis(np.linspace(0, 1, n_memb))
            
        for n in range(self.n):
            axs[n].grid(True)
            title = f'Input {n+1}' if feature_names is None else feature_names[n]
            axs[n].set_title(title)
            for m in range(self.m):
                axs[n].plot(xn[m, n, :], memb_curves[m, n, :], 
                           color=colors[m], label=f'MF {m+1}')
            axs[n].legend()
        
        plt.tight_layout()
        plt.show()

    def evaluate_performance(self, X_test, y_test, scaler_y=None, feature_names=None):
        """Evaluate model performance"""
        y_pred = self(X_test)
        
        if scaler_y is not None:
            y_test_orig = scaler_y.inverse_transform(y_test)
            y_pred_orig = scaler_y.inverse_transform(y_pred)
        else:
            y_test_orig = y_test
            y_pred_orig = y_pred

        metrics = {
            'R2 Score': r2_score(y_test_orig, y_pred_orig),
            'MSE': mean_squared_error(y_test_orig, y_pred_orig),
            'RMSE': np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
            'MAE': mean_absolute_error(y_test_orig, y_pred_orig),
            'MAPE': np.mean(np.abs((y_test_orig - y_pred_orig) / y_test_orig)) * 100,
            'Explained Variance': np.var(y_pred_orig) / np.var(y_test_orig)
        }

        # Plot results
        plt.figure(figsize=(20, 15))
        
        # Scatter plot
        plt.subplot(321)
        plt.scatter(y_test_orig, y_pred_orig, alpha=0.5)
        plt.plot([y_test_orig.min(), y_test_orig.max()], 
                [y_test_orig.min(), y_test_orig.max()], 
                'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Prediction vs Actual')
        
        # Residual plot
        plt.subplot(322)
        residuals = y_test_orig - y_pred_orig
        plt.scatter(y_pred_orig, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Analysis')
        
        # Residual distribution
        plt.subplot(323)
        sns.histplot(residuals, kde=True)
        plt.title('Residual Distribution')
        
        # Q-Q plot
        plt.subplot(324)
        stats.probplot(residuals.flatten(), dist="norm", plot=plt)
        plt.title('Q-Q Plot')
        
        # Error distribution
        plt.subplot(325)
        percentage_errors = (residuals / y_test_orig) * 100
        sns.histplot(percentage_errors, kde=True)
        plt.title('Percentage Error Distribution')
        
        # Prediction intervals
        plt.subplot(326)
        sorted_idx = np.argsort(y_test_orig.flatten())
        y_test_sorted = y_test_orig[sorted_idx]
        y_pred_sorted = y_pred_orig[sorted_idx]
        
        plt.plot(y_test_sorted, label='Actual')
        plt.plot(y_pred_sorted, label='Predicted')
        plt.fill_between(range(len(y_test_sorted)), 
                        y_pred_sorted.flatten() - 2*np.std(residuals),
                        y_pred_sorted.flatten() + 2*np.std(residuals),
                        alpha=0.2)
        plt.title('Prediction Intervals')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return metrics

def prepare_data(df, target_col='price'):
    """
    data preparation
    """
    # Remove outliers using a more sophisticated method
    z_scores = stats.zscore(df[target_col])
    df = df[abs(z_scores) < 3]
    
    # Enhanced feature engineering
    df['time_of_day'] = pd.Categorical(df['departure_time']).codes
    df['arrival_time_code'] = pd.Categorical(df['arrival_time']).codes
    df['route'] = df['source_city'] + '_' + df['destination_city']
    df['route'] = pd.Categorical(df['route']).codes
    
    # Create interaction features
    df['total_stops'] = pd.Categorical(df['stops']).codes
    df['airline_code'] = pd.Categorical(df['airline']).codes
    df['class_code'] = pd.Categorical(df['class']).codes
    df['duration_minutes'] = df['duration'] * 60
    
    # Add nonlinear features
    df['duration_squared'] = df['duration_minutes'] ** 2
    df['days_left_squared'] = df['days_left'] ** 2
    df['stop_duration_interaction'] = df['total_stops'] * df['duration_minutes']
    
    # Select features based on importance
    feature_cols = [
        'duration_minutes', 'duration_squared',
        'days_left', 'days_left_squared',
        'time_of_day', 'arrival_time_code',
        'route', 'total_stops',
        'airline_code', 'class_code',
        'stop_duration_interaction'
    ]
    
    X = df[feature_cols].copy()
    y = df[target_col].values.reshape(-1, 1)
    
    # Robust scaling for better handling of outliers
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    return X_scaled, y_scaled, scaler_X, scaler_y, feature_cols

def create_model(params):
    
    tf.keras.mixed_precision.set_global_policy('float32')
    
    inputs = keras.Input(shape=(params.n_input,))
    
    # Add multiple batch normalization layers
    normalized = keras.layers.BatchNormalization()(inputs)
    
    # Fuzzy layer with optimized initialization
    L1 = FuzzyLayer(params.n_input, params.n_memb, params.memb_func)(normalized)
    L1 = keras.layers.BatchNormalization()(L1)
    
    # Enhanced rule layer
    L2 = RuleLayer(params.n_input, params.n_memb)(L1)
    L2 = keras.layers.Dropout(0.15)(L2)  # Slight increase in dropout
    L2 = keras.layers.BatchNormalization()(L2)
    
    # Norm layer with epsilon
    L3 = NormLayer(epsilon=1e-6)(L2)
    
    # Enhanced defuzz layer
    L4 = DefuzzLayer(params.n_input, params.n_memb)([L3, normalized])
    L4 = keras.layers.BatchNormalization()(L4)
    
    # Final summation
    output = SummationLayer()(L4)
    
    model = keras.Model(inputs=inputs, outputs=output, name="ANFIS")
    
    # Optimizer with warmup
    initial_learning_rate = params.learning_rate
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100,
        decay_rate=0.9,
        staircase=True
    )
    
    optimizer = keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        clipnorm=0.5  # Reduced clipnorm for better gradients
    )
    
    model.compile(
        optimizer=optimizer,
        loss='huber',  # Changed to huber loss for better handling of outliers
        metrics=['mae', 'mse']
    )
    
    return model

def plot_membership_functions(model, feature_names):
    """Plot membership functions for each input feature"""
    fuzzy_layer = model.layers[2]  # Get the fuzzy layer
    mus = fuzzy_layer.mu.numpy()
    sigmas = fuzzy_layer.sigma.numpy()
    
    n_features = len(feature_names)
    n_memb = mus.shape[0]
    
    fig, axes = plt.subplots(n_features, 1, figsize=(12, 4*n_features))
    if n_features == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        x = np.linspace(-3, 3, 1000)
        for j in range(n_memb):
            y = np.exp(-((x - mus[j,i])**2)/(2*sigmas[j,i]**2))
            ax.plot(x, y, label=f'MF {j+1}')
        
        ax.set_title(f'Membership Functions for {feature_names[i]}')
        ax.set_xlabel('Normalized Input')
        ax.set_ylabel('Membership Degree')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('membership_functions.png')
    plt.show()

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot metrics
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def plot_prediction_analysis(y_test, y_pred):
    """Plot prediction analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Scatter plot
    ax1.scatter(y_test, y_pred, alpha=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_title('Predicted vs Actual Values')
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    ax1.grid(True)
    
    # Residuals plot
    residuals = y_pred - y_test
    ax2.scatter(y_pred, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_title('Residuals vs Predicted Values')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.grid(True)
    
    # Histogram of residuals
    ax3.hist(residuals, bins=50, edgecolor='black')
    ax3.set_title('Distribution of Residuals')
    ax3.set_xlabel('Residual Value')
    ax3.set_ylabel('Frequency')
    ax3.grid(True)
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot of Residuals')
    
    plt.tight_layout()
    plt.savefig('prediction_analysis.png')
    plt.show()

def plot_anfis_structure(model, feature_names):
    """
    Visualizes the ANFIS network structure
    Shows the interconnections between different layers
    """
    plt.figure(figsize=(15, 10))
    
    # Plot layers
    layers = ['Input', 'Fuzzification', 'Rules', 'Normalization', 'Defuzzification', 'Output']
    colors = ['#FFA07A', '#98FB98', '#87CEFA', '#DDA0DD', '#F0E68C', '#FFA07A']
    
    for i, (layer, color) in enumerate(zip(layers, colors)):
        plt.plot([i, i], [0, 1], 'k-', linewidth=2)
        plt.fill_between([i-0.2, i+0.2], [0, 0], [1, 1], color=color, alpha=0.3)
        plt.text(i, 1.1, layer, ha='center')
    
    plt.title('ANFIS Network Structure')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('anfis_structure.png')
    plt.show()

def plot_fuzzy_rules_visualization(model, feature_names):
    """
    Visualizes the fuzzy rules and their strengths
    Shows how different inputs activate different rules
    """
    fuzzy_layer = model.layers[2]
    rule_weights = fuzzy_layer.get_weights()[0]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(rule_weights, 
                xticklabels=feature_names,
                yticklabels=[f'Rule {i+1}' for i in range(rule_weights.shape[0])],
                cmap='RdYlBu',
                center=0)
    plt.title('Fuzzy Rules Strength Visualization')
    plt.tight_layout()
    plt.savefig('fuzzy_rules.png')
    plt.show()

def train_model(X_train, y_train, X_test, y_test, params, feature_names):
    model = create_model(params)
    
    # Enhanced callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,  # Reduced for 50 epochs
            restore_best_weights=True,
            min_delta=1e-4
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    try:
        # Train with optimized parameters
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,  # Reduced epochs
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Generate visualizations and evaluate
        plot_anfis_structure(model, feature_names)
        plot_membership_functions(model, feature_names)
        plot_fuzzy_rules_visualization(model, feature_names)
        plot_training_history(history)
        
        y_pred = model.predict(X_test, batch_size=32)
        
        # Reshape for evaluation
        y_test_eval = y_test.reshape(-1)
        y_pred = y_pred.reshape(-1)
        
        # Calculate metrics
        metrics = {
            'R2': r2_score(y_test_eval, y_pred),
            'MAPE': mean_absolute_percentage_error(y_test_eval, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test_eval, y_pred)),
            'MAE': mean_absolute_error(y_test_eval, y_pred)
        }
        
        print("\nModel Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        plot_prediction_analysis(y_test_eval, y_pred)
        
        return model, history
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None

def main():
    # Enable GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    print("Loading and preparing data...")
    train_df = pd.read_csv('Clean_Dataset.csv')
    
    if 'Unnamed: 0' in train_df.columns:
        train_df = train_df.drop('Unnamed: 0', axis=1)
    
    X_scaled, y_scaled, scaler_X, scaler_y, features = prepare_data(train_df)
    
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=pd.qcut(y_scaled.flatten(), q=5, labels=False)
    )
    
    # Optimized parameters
    params = fis_parameters(
        n_input=len(features),
        n_memb=6,  # Adjusted membership functions
        batch_size=32,
        n_epochs=50,
        memb_func='gaussian',
        optimizer='adam',
        loss='huber',
        learning_rate=0.001
    )
    
    print("\nTraining optimized ANFIS model...")
    model, history = train_model(X_train, y_train, X_test, y_test, params, features)

if __name__ == "__main__":
    main()