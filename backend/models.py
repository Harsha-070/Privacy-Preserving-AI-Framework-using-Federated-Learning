"""
Enhanced Model Architecture Module for Federated Learning.
Supports CNN, MLP, ResNet variants with TFF compatibility.
"""

import tensorflow as tf
from typing import Tuple, Optional, List, Callable
import logging

logger = logging.getLogger(__name__)


def create_keras_model(
    input_shape: Tuple[int, ...] = (28, 28, 1),
    num_classes: int = 10,
    architecture: str = 'cnn',
    dropout_rate: float = 0.5,
    use_batch_norm: bool = False,
    hidden_units: List[int] = None
) -> tf.keras.Model:
    """
    Create a Keras model with specified architecture.

    Args:
        input_shape: Shape of input data
        num_classes: Number of output classes
        architecture: Model architecture ('cnn', 'mlp', 'resnet_small')
        dropout_rate: Dropout rate for regularization
        use_batch_norm: Whether to use batch normalization
        hidden_units: Hidden layer units for MLP

    Returns:
        Compiled Keras model
    """
    if hidden_units is None:
        hidden_units = [128, 64]

    if architecture == 'cnn':
        return _create_cnn_model(input_shape, num_classes, dropout_rate, use_batch_norm)
    elif architecture == 'mlp':
        return _create_mlp_model(input_shape, num_classes, dropout_rate, hidden_units)
    elif architecture == 'resnet_small':
        return _create_resnet_small(input_shape, num_classes, use_batch_norm)
    elif architecture == 'cnn_large':
        return _create_cnn_large_model(input_shape, num_classes, dropout_rate, use_batch_norm)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def _create_cnn_model(
    input_shape: Tuple[int, ...],
    num_classes: int,
    dropout_rate: float,
    use_batch_norm: bool
) -> tf.keras.Model:
    """Create a simple CNN model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    ])
    if use_batch_norm:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    if use_batch_norm:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    return model


def _create_cnn_large_model(
    input_shape: Tuple[int, ...],
    num_classes: int,
    dropout_rate: float,
    use_batch_norm: bool
) -> tf.keras.Model:
    """Create a larger CNN model for complex datasets like CIFAR."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    ])
    if use_batch_norm:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    if use_batch_norm:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    return model


def _create_mlp_model(
    input_shape: Tuple[int, ...],
    num_classes: int,
    dropout_rate: float,
    hidden_units: List[int]
) -> tf.keras.Model:
    """Create a Multi-Layer Perceptron model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Flatten(),
    ])
    for units in hidden_units:
        model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    return model


def _residual_block(x, filters, kernel_size=3, stride=1, use_batch_norm=True):
    """Create a residual block."""
    shortcut = x

    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        if use_batch_norm:
            shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x


def _create_resnet_small(
    input_shape: Tuple[int, ...],
    num_classes: int,
    use_batch_norm: bool
) -> tf.keras.Model:
    """Create a small ResNet model suitable for federated learning."""
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, 3, padding='same')(inputs)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = _residual_block(x, 32, use_batch_norm=use_batch_norm)
    x = _residual_block(x, 64, stride=2, use_batch_norm=use_batch_norm)
    x = _residual_block(x, 64, use_batch_norm=use_batch_norm)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)


def model_fn(input_spec, input_shape=(28, 28, 1), num_classes=10, architecture='cnn'):
    """Create a model function for TFF compatibility."""
    try:
        import tensorflow_federated as tff

        keras_model = create_keras_model(
            input_shape=input_shape,
            num_classes=num_classes,
            architecture=architecture
        )

        return tff.learning.models.from_keras_model(
            keras_model,
            input_spec=input_spec,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )
    except ImportError:
        logger.warning("TensorFlow Federated not installed. Using standalone Keras model.")
        return create_keras_model(input_shape, num_classes, architecture)


def create_model_for_dataset(dataset_name: str, architecture: str = 'cnn') -> tf.keras.Model:
    """Create model appropriate for the specified dataset."""
    dataset_configs = {
        'mnist': {'input_shape': (28, 28, 1), 'num_classes': 10},
        'fashion_mnist': {'input_shape': (28, 28, 1), 'num_classes': 10},
        'cifar10': {'input_shape': (32, 32, 3), 'num_classes': 10},
        'cifar100': {'input_shape': (32, 32, 3), 'num_classes': 100},
    }

    if dataset_name not in dataset_configs:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    config = dataset_configs[dataset_name]

    if dataset_name in ['cifar10', 'cifar100'] and architecture == 'cnn':
        architecture = 'cnn_large'

    return create_keras_model(
        input_shape=config['input_shape'],
        num_classes=config['num_classes'],
        architecture=architecture
    )


def get_model_size(model: tf.keras.Model) -> dict:
    """Calculate model size statistics."""
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_params = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)

    return {
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'total_params': trainable_params + non_trainable_params,
        'model_size_mb': (trainable_params + non_trainable_params) * 4 / (1024 * 1024)
    }


if __name__ == '__main__':
    print("Testing model architectures...")

    for arch in ['cnn', 'mlp', 'resnet_small']:
        print(f"\nArchitecture: {arch}")
        model = create_keras_model(architecture=arch)
        size_info = get_model_size(model)
        print(f"  Total params: {size_info['total_params']:,}")
        print(f"  Model size: {size_info['model_size_mb']:.2f} MB")

    print("\nModel tests complete!")
