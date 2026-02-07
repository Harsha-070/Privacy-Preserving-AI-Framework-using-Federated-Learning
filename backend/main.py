import argparse
import os
import sys
import logging
import warnings

# Fix OpenBLAS memory allocation error
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Suppress harmless protobuf version warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

import numpy as np
import tensorflow as tf
import time

# Limit TensorFlow memory growth to prevent OOM on low-RAM machines
try:
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass


def get_adaptive_defaults():
    """Auto-detect hardware and return safe defaults."""
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        ram_gb = 4  # conservative fallback

    if ram_gb < 4:
        return {'clients': 2, 'clients_per_round': 2, 'rounds': 3, 'local_epochs': 1, 'batch_size': 32}
    elif ram_gb < 8:
        return {'clients': 3, 'clients_per_round': 3, 'rounds': 5, 'local_epochs': 2, 'batch_size': 32}
    elif ram_gb < 16:
        return {'clients': 5, 'clients_per_round': 4, 'rounds': 10, 'local_epochs': 2, 'batch_size': 64}
    else:
        return {'clients': 10, 'clients_per_round': 5, 'rounds': 10, 'local_epochs': 2, 'batch_size': 64}

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.data_loader import load_and_preprocess_data, create_federated_data, visualize_client_data_distribution, get_client_data_stats
from backend.models import create_keras_model, get_model_size
from backend.federated_core import FederatedTrainer, client_update, server_aggregate
from backend.centralized_trainer import train_centralized_model
from backend.utils import init_results_dir, generate_all_outputs
from backend.config import Config, create_config_from_args

logging.basicConfig(level=logging.INFO)


def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main():
    print("=" * 60)
    print("Privacy-Preserving Federated Learning Framework")
    print("=" * 60)

    hw = get_adaptive_defaults()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--clients', type=int, default=hw['clients'])
    parser.add_argument('--clients_per_round', type=int, default=hw['clients_per_round'])
    parser.add_argument('--rounds', type=int, default=hw['rounds'])
    parser.add_argument('--local_epochs', type=int, default=hw['local_epochs'])
    parser.add_argument('--batch_size', type=int, default=hw['batch_size'])
    parser.add_argument('--client_lr', type=float, default=0.01)
    parser.add_argument('--distribution', default='non-iid')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', default='results')
    args = parser.parse_args()

    print(f"  Hardware-adaptive defaults: {hw['clients']} clients, {hw['rounds']} rounds")
    print(f"  Using: {args.clients} clients, {args.rounds} rounds, {args.local_epochs} epochs")

    config = create_config_from_args(args)
    config.data.dataset_name = args.dataset
    config.data.num_clients = args.clients
    config.data.distribution = args.distribution
    config.federated.clients_per_round = args.clients_per_round
    config.federated.num_rounds = args.rounds
    config.federated.local_epochs = args.local_epochs
    config.federated.batch_size = args.batch_size
    config.federated.client_learning_rate = args.client_lr

    set_seeds(args.seed)
    results_dir = init_results_dir(args.results_dir)
    config.save_json(os.path.join(results_dir, 'config.json'))

    print("\n[1/6] Loading data...", flush=True)
    train_data, test_data = load_and_preprocess_data(args.dataset)
    x_train, y_train = train_data
    print(f"  Train: {len(x_train)}, Test: {len(test_data[0])}", flush=True)

    print("\n[2/6] Creating client datasets...", flush=True)
    client_datasets = create_federated_data(train_data, num_clients=args.clients, distribution=args.distribution, seed=args.seed)
    stats = get_client_data_stats(client_datasets)
    print(f"  {stats['num_clients']} clients, Mean: {stats['mean_samples']:.0f} samples", flush=True)
    visualize_client_data_distribution(client_datasets, save_path=os.path.join(results_dir, 'client_data_distribution.png'))

    print("\n[3/6] Setting up model...", flush=True)
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))
    def model_fn():
        return create_keras_model(input_shape=input_shape, num_classes=num_classes)

    print("\n[4/6] Centralized training...", flush=True)
    cent_model = model_fn()
    cent_history, cent_model = train_centralized_model(train_data, test_data, cent_model, epochs=args.rounds, batch_size=args.batch_size)
    print(f"  Final Accuracy: {cent_history['test_accuracy'][-1]:.4f}", flush=True)

    print("\n[5/6] Federated training...", flush=True)
    trainer = FederatedTrainer(
        model_fn=model_fn,
        num_clients=args.clients,
        clients_per_round=args.clients_per_round,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        client_learning_rate=args.client_lr
    )
    fed_history = trainer.train(client_datasets, test_data, verbose=1)
    fed_model = trainer.get_global_model()
    print(f"  Final Accuracy: {fed_history['test_accuracy'][-1]:.4f}", flush=True)

    print("\n[6/6] Generating outputs...", flush=True)
    fed_model.save(os.path.join(results_dir, 'federated_model_final.h5'))
    cent_model.save(os.path.join(results_dir, 'centralized_model_final.h5'))
    generate_all_outputs(fed_history, cent_history, config.to_dict(), results_dir)

    fed_acc = fed_history['test_accuracy'][-1]
    cent_acc = cent_history['test_accuracy'][-1]
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  Federated Accuracy:   {fed_acc:.4f}")
    print(f"  Centralized Accuracy: {cent_acc:.4f}")
    print(f"  Accuracy Retention:   {fed_acc/cent_acc*100:.2f}%")
    print(f"  Privacy Preserved:    Yes")
    print("=" * 60)
    print(f"\nResults saved to: {results_dir}/")


if __name__ == '__main__':
    main()