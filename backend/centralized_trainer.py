"""
Centralized Training Module for baseline comparison.
"""

import os
import warnings

# Fix OpenBLAS and TensorFlow compatibility issues
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

import tensorflow as tf
import numpy as np
import time
from typing import Tuple, Dict, Optional, Callable
import logging

logger = logging.getLogger(__name__)


def train_centralized_model(
    train_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray],
    model: tf.keras.Model,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    optimizer_name: str = 'adam',
    callbacks: list = None,
    verbose: int = 1
) -> Tuple[Dict, tf.keras.Model]:
    """
    Train model in centralized manner for comparison.

    Args:
        train_data: Tuple of (x_train, y_train)
        test_data: Tuple of (x_test, y_test)
        model: Keras model to train
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        optimizer_name: Optimizer type ('adam', 'sgd')
        callbacks: Optional list of Keras callbacks
        verbose: Verbosity level

    Returns:
        Tuple of (training_history_dict, trained_model)
    """
    x_train, y_train = train_data
    x_test, y_test = test_data

    # Create optimizer
    if optimizer_name.lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Default callbacks
    if callbacks is None:
        callbacks = []

    # Train
    start_time = time.time()

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=verbose
    )

    training_time = time.time() - start_time

    # Extract history
    history_dict = {
        'train_loss': history.history['loss'],
        'train_accuracy': history.history['accuracy'],
        'test_loss': history.history['val_loss'],
        'test_accuracy': history.history['val_accuracy'],
        'epochs': list(range(1, epochs + 1)),
        'training_time': training_time,
        'final_train_accuracy': history.history['accuracy'][-1],
        'final_test_accuracy': history.history['val_accuracy'][-1],
        'final_train_loss': history.history['loss'][-1],
        'final_test_loss': history.history['val_loss'][-1]
    }

    logger.info(f"Centralized training complete in {training_time:.2f}s")
    logger.info(f"Final accuracy: {history_dict['final_test_accuracy']:.4f}")

    return history_dict, model


def run_centralized_training(
    dataset_name: str = 'mnist',
    epochs: int = 10,
    batch_size: int = 32,
    model_fn: Callable = None,
    save_model: bool = True,
    results_dir: str = 'results'
) -> Tuple[Dict, tf.keras.Model]:
    """
    Complete centralized training pipeline.

    Args:
        dataset_name: Name of dataset to use
        epochs: Training epochs
        batch_size: Batch size
        model_fn: Optional model creation function
        save_model: Whether to save the trained model
        results_dir: Directory to save results

    Returns:
        Tuple of (history_dict, trained_model)
    """
    from backend.data_loader import load_and_preprocess_data
    from backend.models import create_model_for_dataset

    print(f"Starting Centralized Training on {dataset_name}...")

    # Load data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data(dataset_name)

    # Create model
    if model_fn is not None:
        model = model_fn()
    else:
        model = create_model_for_dataset(dataset_name)

    # Train
    history_dict, model = train_centralized_model(
        train_data=(x_train, y_train),
        test_data=(x_test, y_test),
        model=model,
        epochs=epochs,
        batch_size=batch_size
    )

    # Save model
    if save_model:
        os.makedirs(results_dir, exist_ok=True)
        model_path = os.path.join(results_dir, 'centralized_model.h5')
        model.save(model_path)
        print(f"Model saved to {model_path}")

    print(f"Centralized training complete.")
    print(f"Final Test Accuracy: {history_dict['final_test_accuracy']:.4f}")

    return history_dict, model


if __name__ == '__main__':
    # Test centralized training
    history, model = run_centralized_training(
        dataset_name='mnist',
        epochs=5,
        batch_size=32
    )
    print(f"Training complete. Final accuracy: {history['final_test_accuracy']:.4f}")
