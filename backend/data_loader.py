"""
Data Loading and Preprocessing Module for Federated Learning.
"""

import os
# Fix OpenBLAS memory allocation error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Tuple, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def load_and_preprocess_data(dataset_name='mnist', data_path=None, normalize=True):
    """Load dataset and apply preprocessing."""
    if dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
    elif dataset_name == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
    elif dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train = y_train.flatten()
        y_test = y_test.flatten()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    if normalize:
        x_train = x_train / 255.0
        x_test = x_test / 255.0
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    return (x_train, y_train), (x_test, y_test)


def create_federated_data(data, num_clients=10, distribution='non-iid', shards_per_client=2, seed=None):
    """Split dataset into client datasets."""
    if seed is not None:
        np.random.seed(seed)

    x_data, y_data = data
    y_data = y_data.flatten()
    n_samples = len(x_data)

    if distribution == 'iid':
        indices = np.random.permutation(n_samples)
        client_shards = np.array_split(indices, num_clients)
        client_indices = {i: shard for i, shard in enumerate(client_shards)}
    elif distribution == 'non-iid':
        sorted_indices = np.argsort(y_data)
        total_shards = num_clients * shards_per_client
        shard_size = n_samples // total_shards
        shard_indices = np.random.permutation(total_shards)
        client_indices = defaultdict(list)
        for client_id in range(num_clients):
            client_shards = shard_indices[client_id * shards_per_client:(client_id + 1) * shards_per_client]
            for shard_id in client_shards:
                start = shard_id * shard_size
                end = start + shard_size if shard_id < total_shards - 1 else n_samples
                client_indices[client_id].extend(sorted_indices[start:end])
        client_indices = {k: np.array(v) for k, v in client_indices.items()}
    elif distribution == 'dirichlet':
        num_classes = len(np.unique(y_data))
        class_indices = {c: np.where(y_data == c)[0] for c in range(num_classes)}
        client_indices = defaultdict(list)
        for c in range(num_classes):
            indices = class_indices[c]
            np.random.shuffle(indices)
            proportions = np.random.dirichlet([0.5] * num_clients)
            counts = (proportions * len(indices)).astype(int)
            remainder = len(indices) - counts.sum()
            for i in range(remainder):
                counts[i % num_clients] += 1
            idx_start = 0
            for client_id in range(num_clients):
                idx_end = idx_start + counts[client_id]
                client_indices[client_id].extend(indices[idx_start:idx_end])
                idx_start = idx_end
        client_indices = {k: np.array(v) for k, v in client_indices.items()}
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    client_datasets = []
    for i in range(num_clients):
        indices = client_indices[i]
        client_x = x_data[indices]
        client_y = y_data[indices]
        shuffle_idx = np.random.permutation(len(client_x))
        ds = tf.data.Dataset.from_tensor_slices((client_x[shuffle_idx], client_y[shuffle_idx]))
        client_datasets.append(ds)
    return client_datasets


def get_client_data_stats(client_datasets, num_classes=10):
    """Calculate statistics about client data distribution."""
    stats = {
        'num_clients': len(client_datasets),
        'samples_per_client': [],
        'class_distribution': [],
        'label_counts': {}
    }
    for i, ds in enumerate(client_datasets):
        labels = [y.numpy() for _, y in ds]
        stats['samples_per_client'].append(len(labels))
        unique, counts = np.unique(labels, return_counts=True)
        class_dist = np.zeros(num_classes)
        for u, c in zip(unique, counts):
            if u < num_classes:
                class_dist[u] = c
        stats['class_distribution'].append(class_dist)
        stats['label_counts'][i] = dict(zip(unique.tolist(), counts.tolist()))

    stats['total_samples'] = sum(stats['samples_per_client'])
    stats['mean_samples'] = np.mean(stats['samples_per_client'])
    stats['std_samples'] = np.std(stats['samples_per_client'])
    stats['min_samples'] = min(stats['samples_per_client'])
    stats['max_samples'] = max(stats['samples_per_client'])
    return stats


def visualize_client_data_distribution(client_datasets, num_classes=10, save_path='client_data_distribution.png'):
    """Visualize the distribution of labels across clients."""
    stats = get_client_data_stats(client_datasets, num_classes)
    counts_matrix = np.array(stats['class_distribution'])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    bottom = np.zeros(len(client_datasets))
    client_ids = range(len(client_datasets))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    for cls in range(num_classes):
        ax1.bar(client_ids, counts_matrix[:, cls], bottom=bottom, label=f'Class {cls}', color=colors[cls])
        bottom += counts_matrix[:, cls]
    ax1.set_xlabel('Client ID')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Label Distribution per Client')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    ax2 = axes[1]
    sns.heatmap(counts_matrix.T, ax=ax2, cmap='YlOrRd',
                xticklabels=range(len(client_datasets)), yticklabels=range(num_classes))
    ax2.set_xlabel('Client ID')
    ax2.set_ylabel('Class')
    ax2.set_title('Class Distribution Heatmap')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved data distribution plot to '{save_path}'")
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_data, test_data = load_and_preprocess_data('mnist')
    fed_data = create_federated_data(train_data, num_clients=10, distribution='non-iid', seed=42)
    print(f"Created {len(fed_data)} client datasets.")
    os.makedirs('results', exist_ok=True)
    visualize_client_data_distribution(fed_data, save_path='results/client_data_distribution.png')
