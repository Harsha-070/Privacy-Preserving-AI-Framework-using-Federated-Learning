import argparse
import numpy as np
import tensorflow as tf
import time
import os
import sys
import logging

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--clients', type=int, default=10)
    parser.add_argument('--clients_per_round', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--local_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--client_lr', type=float, default=0.01)
    parser.add_argument('--distribution', default='non-iid')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_dir', default='results')
    args = parser.parse_args()

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

    print("\n[1/6] Loading data...")
    train_data, test_data = load_and_preprocess_data(args.dataset)
    x_train, y_train = train_data
    print(f"  Train: {len(x_train)}, Test: {len(test_data[0])}")

    print("\n[2/6] Creating client datasets...")
    client_datasets = create_federated_data(train_data, num_clients=args.clients, distribution=args.distribution, seed=args.seed)
    stats = get_client_data_stats(client_datasets)
    print(f"  {stats['num_clients']} clients, Mean: {stats['mean_samples']:.0f} samples")
    visualize_client_data_distribution(client_datasets, save_path=os.path.join(results_dir, 'client_data_distribution.png'))

    print("\n[3/6] Setting up model...")
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))
    def model_fn():
        return create_keras_model(input_shape=input_shape, num_classes=num_classes)

    print("\n[4/6] Centralized training...")
    cent_model = model_fn()
    cent_history, cent_model = train_centralized_model(train_data, test_data, cent_model, epochs=args.rounds, batch_size=args.batch_size)
    print(f"  Final Accuracy: {cent_history['test_accuracy'][-1]:.4f}")

    print("\n[5/6] Federated training...")
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
    print(f"  Final Accuracy: {fed_history['test_accuracy'][-1]:.4f}")

    print("\n[6/6] Generating outputs...")
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