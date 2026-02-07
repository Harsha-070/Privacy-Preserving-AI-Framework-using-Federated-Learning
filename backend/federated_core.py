"""
Federated Learning Core Module.
Implements FedAvg algorithm with client update and server aggregation.
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
import collections
from typing import List, Tuple, Dict, Optional, Callable
import time
import logging

logger = logging.getLogger(__name__)

# Named tuple for client output
ClientOutput = collections.namedtuple(
    'ClientOutput',
    ['weights_delta', 'client_weight', 'loss', 'accuracy', 'num_samples', 'training_time']
)

# Named tuple for server state
ServerState = collections.namedtuple(
    'ServerState',
    ['model_weights', 'round_num', 'optimizer_state']
)


def client_update(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    global_weights: List[np.ndarray],
    epochs: int = 1,
    learning_rate: float = 0.01,
    optimizer_name: str = 'sgd',
    fedprox_mu: float = 0.0
) -> ClientOutput:
    """
    Train model locally on client data and return weight updates.

    Args:
        model: Keras model instance
        dataset: Client's local dataset (batched)
        global_weights: Current global model weights
        epochs: Number of local epochs
        learning_rate: Local learning rate
        optimizer_name: Optimizer type ('sgd' or 'adam')
        fedprox_mu: FedProx proximal term coefficient (0 for standard FedAvg)

    Returns:
        ClientOutput with weight deltas and metrics
    """
    start_time = time.time()

    # Set model weights to global weights
    model.set_weights(global_weights)
    initial_weights = [w.copy() for w in global_weights]

    # Create optimizer
    if optimizer_name.lower() == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    total_loss = 0.0
    num_batches = 0
    num_samples = 0

    # Training loop
    for epoch in range(epochs):
        for x_batch, y_batch in dataset:
            with tf.GradientTape() as tape:
                predictions = model(x_batch, training=True)
                loss = loss_fn(y_batch, predictions)

                # FedProx proximal term
                if fedprox_mu > 0:
                    proximal_term = 0.0
                    for w, w_init in zip(model.trainable_weights, initial_weights[:len(model.trainable_weights)]):
                        proximal_term += tf.reduce_sum(tf.square(w - w_init))
                    loss += (fedprox_mu / 2) * proximal_term

            gradients = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))

            acc_metric.update_state(y_batch, predictions)
            total_loss += float(loss)
            num_batches += 1
            num_samples += x_batch.shape[0]

    # Calculate metrics
    avg_loss = total_loss / max(num_batches, 1)
    avg_acc = float(acc_metric.result())

    # Calculate weight deltas
    new_weights = model.get_weights()
    weights_delta = [new - old for new, old in zip(new_weights, global_weights)]

    training_time = time.time() - start_time

    return ClientOutput(
        weights_delta=weights_delta,
        client_weight=num_samples,
        loss=avg_loss,
        accuracy=avg_acc,
        num_samples=num_samples,
        training_time=training_time
    )


def server_aggregate(
    global_weights: List[np.ndarray],
    client_outputs: List[ClientOutput],
    server_learning_rate: float = 1.0
) -> Tuple[List[np.ndarray], float, float]:
    """
    Aggregate client updates using Federated Averaging (FedAvg).

    Args:
        global_weights: Current global model weights
        client_outputs: List of ClientOutput from participating clients
        server_learning_rate: Server-side learning rate for update scaling

    Returns:
        Tuple of (new_global_weights, aggregated_loss, aggregated_accuracy)
    """
    if not client_outputs:
        return global_weights, 0.0, 0.0

    # Calculate total weight (sum of client samples)
    total_weight = sum(c.client_weight for c in client_outputs)

    if total_weight == 0:
        return global_weights, 0.0, 0.0

    # Initialize aggregated delta with zeros
    aggregated_delta = [np.zeros_like(w) for w in global_weights]

    # Weighted sum of deltas
    for client_out in client_outputs:
        weight_factor = client_out.client_weight / total_weight
        for i, delta in enumerate(client_out.weights_delta):
            aggregated_delta[i] += delta * weight_factor

    # Apply server learning rate and update global weights
    new_global_weights = [
        g + server_learning_rate * d
        for g, d in zip(global_weights, aggregated_delta)
    ]

    # Aggregate metrics (weighted average)
    avg_loss = sum(c.loss * c.client_weight for c in client_outputs) / total_weight
    avg_acc = sum(c.accuracy * c.client_weight for c in client_outputs) / total_weight

    return new_global_weights, avg_loss, avg_acc


def select_clients(
    num_clients: int,
    clients_per_round: int,
    selection_strategy: str = 'random'
) -> List[int]:
    """
    Select clients for the current training round.

    Args:
        num_clients: Total number of clients
        clients_per_round: Number of clients to select
        selection_strategy: Selection strategy ('random', 'sequential')

    Returns:
        List of selected client indices
    """
    clients_per_round = min(clients_per_round, num_clients)

    if selection_strategy == 'random':
        return list(np.random.choice(num_clients, clients_per_round, replace=False))
    elif selection_strategy == 'sequential':
        # Round-robin selection
        return list(range(clients_per_round))
    else:
        return list(np.random.choice(num_clients, clients_per_round, replace=False))


def evaluate_model(
    model: tf.keras.Model,
    test_data: Tuple[np.ndarray, np.ndarray],
    weights: Optional[List[np.ndarray]] = None
) -> Dict[str, float]:
    """
    Evaluate model on test data.

    Args:
        model: Keras model
        test_data: Tuple of (x_test, y_test)
        weights: Optional weights to set before evaluation

    Returns:
        Dictionary with evaluation metrics
    """
    if weights is not None:
        model.set_weights(weights)

    x_test, y_test = test_data

    # Always compile the model for evaluation
    model.compile(
        optimizer='sgd',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    return {
        'loss': float(loss),
        'accuracy': float(accuracy)
    }


def calculate_model_update_norm(weights_delta: List[np.ndarray]) -> float:
    """Calculate L2 norm of weight updates."""
    total_norm = 0.0
    for delta in weights_delta:
        total_norm += np.sum(np.square(delta))
    return np.sqrt(total_norm)


def calculate_communication_cost(
    weights: List[np.ndarray],
    precision_bytes: int = 4
) -> Dict[str, float]:
    """
    Calculate communication cost for transmitting model weights.

    Args:
        weights: Model weights
        precision_bytes: Bytes per parameter (4 for float32, 2 for float16)

    Returns:
        Dictionary with communication statistics
    """
    total_params = sum(w.size for w in weights)
    total_bytes = total_params * precision_bytes

    return {
        'total_parameters': total_params,
        'bytes': total_bytes,
        'kilobytes': total_bytes / 1024,
        'megabytes': total_bytes / (1024 * 1024)
    }


class FederatedTrainer:
    """
    Main class for running federated learning experiments.
    """

    def __init__(
        self,
        model_fn: Callable,
        num_clients: int,
        clients_per_round: int = 10,
        num_rounds: int = 100,
        local_epochs: int = 5,
        batch_size: int = 32,
        client_learning_rate: float = 0.01,
        server_learning_rate: float = 1.0,
        client_optimizer: str = 'sgd',
        fedprox_mu: float = 0.0
    ):
        """
        Initialize federated trainer.

        Args:
            model_fn: Function that returns a Keras model
            num_clients: Total number of clients
            clients_per_round: Clients participating per round
            num_rounds: Total communication rounds
            local_epochs: Local training epochs per round
            batch_size: Batch size for local training
            client_learning_rate: Client-side learning rate
            server_learning_rate: Server-side learning rate
            client_optimizer: Client optimizer type
            fedprox_mu: FedProx proximal coefficient
        """
        self.model_fn = model_fn
        self.num_clients = num_clients
        self.clients_per_round = clients_per_round
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.client_learning_rate = client_learning_rate
        self.server_learning_rate = server_learning_rate
        self.client_optimizer = client_optimizer
        self.fedprox_mu = fedprox_mu

        # Initialize global model
        self.global_model = model_fn()
        self.global_weights = self.global_model.get_weights()

        # Training history
        self.history = {
            'round': [],
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'communication_rounds': [],
            'bytes_communicated': [],
            'round_time': [],
            'selected_clients': []
        }

    def train(
        self,
        client_datasets: List[tf.data.Dataset],
        test_data: Tuple[np.ndarray, np.ndarray],
        verbose: int = 1
    ) -> Dict:
        """
        Run federated training.

        Args:
            client_datasets: List of client datasets
            test_data: Test data tuple (x_test, y_test)
            verbose: Verbosity level

        Returns:
            Training history dictionary
        """
        # Batch client datasets
        batched_datasets = []
        for ds in client_datasets:
            # Always batch the dataset fresh to ensure correct batch dimension
            batched_ds = ds.batch(self.batch_size)
            batched_datasets.append(batched_ds)

        # Create client model for training
        client_model = self.model_fn()

        total_bytes = 0

        # Per-round timeout: 5 minutes max per round as safety net
        max_round_time = 300

        for round_num in range(1, self.num_rounds + 1):
            round_start = time.time()

            # Select clients
            selected_clients = select_clients(
                self.num_clients,
                self.clients_per_round
            )

            # Client training
            client_outputs = []
            for client_id in selected_clients:
                client_start = time.time()
                output = client_update(
                    model=client_model,
                    dataset=batched_datasets[client_id],
                    global_weights=self.global_weights,
                    epochs=self.local_epochs,
                    learning_rate=self.client_learning_rate,
                    optimizer_name=self.client_optimizer,
                    fedprox_mu=self.fedprox_mu
                )
                client_outputs.append(output)

                # Check if round is taking too long
                if time.time() - round_start > max_round_time:
                    logger.warning(f"Round {round_num} exceeded {max_round_time}s timeout, using partial results")
                    break

            # Server aggregation
            self.global_weights, train_loss, train_acc = server_aggregate(
                self.global_weights,
                client_outputs,
                self.server_learning_rate
            )

            # Evaluate on test data
            test_metrics = evaluate_model(
                self.global_model,
                test_data,
                self.global_weights
            )

            # Calculate communication cost
            comm_cost = calculate_communication_cost(self.global_weights)
            round_bytes = comm_cost['bytes'] * (len(selected_clients) + 1)  # Upload + download
            total_bytes += round_bytes

            round_time = time.time() - round_start

            # Record history
            self.history['round'].append(round_num)
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['test_loss'].append(test_metrics['loss'])
            self.history['test_accuracy'].append(test_metrics['accuracy'])
            self.history['communication_rounds'].append(round_num)
            self.history['bytes_communicated'].append(total_bytes)
            self.history['round_time'].append(round_time)
            self.history['selected_clients'].append(selected_clients)

            if verbose >= 1:
                print(f"Round {round_num}/{self.num_rounds} - "
                      f"Loss: {test_metrics['loss']:.4f} - "
                      f"Accuracy: {test_metrics['accuracy']:.4f} - "
                      f"Time: {round_time:.2f}s", flush=True)

        # Set final weights to global model
        self.global_model.set_weights(self.global_weights)

        return self.history

    def get_global_model(self) -> tf.keras.Model:
        """Return the trained global model."""
        return self.global_model

    def get_history(self) -> Dict:
        """Return training history."""
        return self.history


if __name__ == '__main__':
    print("Federated Core module loaded successfully.")
    print("Available functions: client_update, server_aggregate, FederatedTrainer")
