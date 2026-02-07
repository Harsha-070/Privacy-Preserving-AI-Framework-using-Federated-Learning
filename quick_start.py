"""
Quick Start Script - Lightweight Federated Learning Demo
Runs fast with reduced parameters for testing purposes
"""

import os
import sys
import warnings

# Fix OpenBLAS memory errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logs

# Suppress protobuf warnings (harmless)
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

import argparse
import numpy as np
import tensorflow as tf
import time
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.data_loader import load_and_preprocess_data, create_federated_data
from backend.models import create_keras_model
from backend.federated_core import FederatedTrainer
from backend.centralized_trainer import train_centralized_model
from backend.utils import init_results_dir

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def main():
    print("\n" + "="*70)
    print("  QUICK START - Privacy-Preserving Federated Learning")
    print("  Lightweight configuration for fast testing")
    print("="*70 + "\n")

    # LIGHTWEIGHT PARAMETERS
    dataset_name = 'mnist'
    num_clients = 3
    clients_per_round = 2
    num_rounds = 3
    local_epochs = 1
    batch_size = 64

    logger.info(f"Configuration: {num_clients} clients, {num_rounds} rounds, {local_epochs} local epochs")

    set_seeds(42)
    results_dir = init_results_dir('results')

    # Step 1: Load data
    logger.info("[1/5] Loading MNIST dataset...")
    start_time = time.time()
    train_data, test_data = load_and_preprocess_data(dataset_name)
    x_train, y_train = train_data
    x_test, y_test = test_data
    logger.info(f"  [OK] Loaded {len(x_train)} training samples in {time.time()-start_time:.1f}s")

    # Step 2: Create federated datasets
    logger.info(f"[2/5] Creating {num_clients} client datasets...")
    start_time = time.time()
    client_datasets = create_federated_data(
        train_data,
        num_clients=num_clients,
        distribution='iid',  # IID is faster
        seed=42
    )
    logger.info(f"  [OK] Created client datasets in {time.time()-start_time:.1f}s")

    # Step 3: Setup model
    logger.info("[3/5] Creating neural network model...")
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))

    def model_fn():
        model = create_keras_model(
            input_shape=input_shape,
            num_classes=num_classes,
            architecture='cnn'
        )
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        return model

    logger.info(f"  [OK] Model architecture: CNN for {input_shape} -> {num_classes} classes")

    # Step 4: Centralized training (quick baseline)
    logger.info("[4/5] Training centralized baseline...")
    start_time = time.time()
    cent_model = model_fn()

    # Very quick centralized training
    history = cent_model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=num_rounds,
        batch_size=batch_size,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: logger.info(
                    f"  Epoch {epoch+1}/{num_rounds}: acc={logs['accuracy']:.4f}, val_acc={logs['val_accuracy']:.4f}"
                )
            )
        ]
    )

    cent_loss, cent_acc = cent_model.evaluate(x_test, y_test, verbose=0)
    cent_time = time.time() - start_time
    logger.info(f"  [OK] Centralized training completed in {cent_time:.1f}s")
    logger.info(f"  [OK] Final accuracy: {cent_acc:.4f}")

    # Step 5: Federated training
    logger.info(f"[5/5] Starting federated learning ({num_rounds} rounds)...")
    start_time = time.time()

    trainer = FederatedTrainer(
        model_fn=model_fn,
        num_clients=num_clients,
        clients_per_round=clients_per_round,
        num_rounds=num_rounds,
        local_epochs=local_epochs,
        batch_size=batch_size,
        client_learning_rate=0.01
    )

    fed_history = trainer.train(client_datasets, test_data, verbose=1)
    fed_time = time.time() - start_time

    fed_acc = fed_history['test_accuracy'][-1]
    fed_loss = fed_history['test_loss'][-1]

    logger.info(f"  [OK] Federated training completed in {fed_time:.1f}s")
    logger.info(f"  [OK] Final accuracy: {fed_acc:.4f}")

    # Save models
    fed_model = trainer.get_global_model()
    fed_model.save(os.path.join(results_dir, 'federated_model_final.h5'))
    cent_model.save(os.path.join(results_dir, 'centralized_model_final.h5'))

    # Results
    print("\n" + "="*70)
    print("  RESULTS SUMMARY")
    print("="*70)
    print(f"  Federated Accuracy:   {fed_acc*100:.2f}%  (trained in {fed_time:.1f}s)")
    print(f"  Centralized Accuracy: {cent_acc*100:.2f}%  (trained in {cent_time:.1f}s)")
    print(f"  Accuracy Retention:   {(fed_acc/cent_acc)*100:.1f}%")
    print(f"  Privacy Preserved:    [OK] Yes (data stayed on clients)")
    print("="*70)
    print(f"\n[OK] Models saved to: {results_dir}/\n")

    # Create simple report
    report = {
        'federated_accuracy': float(fed_acc),
        'centralized_accuracy': float(cent_acc),
        'retention': float((fed_acc/cent_acc)*100),
        'federated_time': float(fed_time),
        'centralized_time': float(cent_time),
        'config': {
            'dataset': dataset_name,
            'clients': num_clients,
            'rounds': num_rounds,
            'local_epochs': local_epochs
        }
    }

    import json
    with open(os.path.join(results_dir, 'quick_start_report.json'), 'w') as f:
        json.dump(report, indent=2, fp=f)

    print("[OK] Success! To run full version with more parameters:")
    print("  python backend/main.py --clients 5 --rounds 10\n")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n[ERROR] Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
