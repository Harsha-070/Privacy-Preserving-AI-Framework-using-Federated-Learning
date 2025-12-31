"""
Utility functions for visualization, logging, and reporting.
Generates comparison plots and performance reports.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json
import csv
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def init_results_dir(dir_name: str = 'results') -> str:
    """Initialize results directory."""
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def plot_accuracy_comparison(
    federated_history: Dict,
    centralized_history: Dict,
    save_path: str = 'results/accuracy_comparison.png',
    title: str = 'Federated vs Centralized: Accuracy Comparison'
) -> None:
    """
    Plot accuracy comparison between federated and centralized training.

    Args:
        federated_history: Federated training history
        centralized_history: Centralized training history
        save_path: Path to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Federated accuracy
    fed_rounds = federated_history.get('round', range(len(federated_history['test_accuracy'])))
    fed_acc = federated_history['test_accuracy']
    ax.plot(fed_rounds, fed_acc, 'b-o', label='Federated Learning', linewidth=2, markersize=4)

    # Centralized accuracy
    cent_epochs = centralized_history.get('epochs', range(len(centralized_history['test_accuracy'])))
    cent_acc = centralized_history['test_accuracy']
    ax.plot(cent_epochs, cent_acc, 'r--s', label='Centralized Training', linewidth=2, markersize=4)

    # Final accuracy lines
    ax.axhline(y=fed_acc[-1], color='blue', linestyle=':', alpha=0.5, label=f'Fed Final: {fed_acc[-1]:.4f}')
    ax.axhline(y=cent_acc[-1], color='red', linestyle=':', alpha=0.5, label=f'Cent Final: {cent_acc[-1]:.4f}')

    ax.set_xlabel('Round / Epoch', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved accuracy comparison plot to '{save_path}'")
    plt.close()


def plot_loss_comparison(
    federated_history: Dict,
    centralized_history: Dict,
    save_path: str = 'results/loss_comparison.png',
    title: str = 'Federated vs Centralized: Loss Comparison'
) -> None:
    """
    Plot loss comparison between federated and centralized training.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Federated loss
    fed_rounds = federated_history.get('round', range(len(federated_history['test_loss'])))
    fed_loss = federated_history['test_loss']
    ax.plot(fed_rounds, fed_loss, 'b-o', label='Federated Learning', linewidth=2, markersize=4)

    # Centralized loss
    cent_epochs = centralized_history.get('epochs', range(len(centralized_history['test_loss'])))
    cent_loss = centralized_history['test_loss']
    ax.plot(cent_epochs, cent_loss, 'r--s', label='Centralized Training', linewidth=2, markersize=4)

    ax.set_xlabel('Round / Epoch', fontsize=12)
    ax.set_ylabel('Test Loss', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved loss comparison plot to '{save_path}'")
    plt.close()


def plot_communication_overhead(
    federated_history: Dict,
    save_path: str = 'results/communication_overhead.png'
) -> None:
    """
    Plot communication overhead during federated training.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    rounds = federated_history.get('round', range(len(federated_history.get('bytes_communicated', []))))

    if 'bytes_communicated' in federated_history:
        bytes_data = np.array(federated_history['bytes_communicated'])
        mb_data = bytes_data / (1024 * 1024)

        # Cumulative communication
        ax1 = axes[0]
        ax1.plot(rounds, mb_data, 'g-o', linewidth=2, markersize=4)
        ax1.fill_between(rounds, 0, mb_data, alpha=0.3, color='green')
        ax1.set_xlabel('Communication Round', fontsize=12)
        ax1.set_ylabel('Cumulative Data (MB)', fontsize=12)
        ax1.set_title('Cumulative Communication Cost', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Per-round communication
        ax2 = axes[1]
        per_round = np.diff(mb_data, prepend=0)
        ax2.bar(rounds, per_round, color='teal', alpha=0.7)
        ax2.set_xlabel('Communication Round', fontsize=12)
        ax2.set_ylabel('Data per Round (MB)', fontsize=12)
        ax2.set_title('Per-Round Communication Cost', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved communication overhead plot to '{save_path}'")
    plt.close()


def plot_training_curves(
    federated_history: Dict,
    centralized_history: Dict,
    save_path: str = 'results/training_curves.png'
) -> None:
    """
    Generate comprehensive training curves subplot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Test Accuracy
    ax1 = axes[0, 0]
    fed_acc = federated_history['test_accuracy']
    cent_acc = centralized_history['test_accuracy']
    ax1.plot(fed_acc, 'b-', label='Federated', linewidth=2)
    ax1.plot(cent_acc, 'r--', label='Centralized', linewidth=2)
    ax1.set_title('Test Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Round/Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Test Loss
    ax2 = axes[0, 1]
    fed_loss = federated_history['test_loss']
    cent_loss = centralized_history['test_loss']
    ax2.plot(fed_loss, 'b-', label='Federated', linewidth=2)
    ax2.plot(cent_loss, 'r--', label='Centralized', linewidth=2)
    ax2.set_title('Test Loss', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Round/Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Train Accuracy
    ax3 = axes[1, 0]
    if 'train_accuracy' in federated_history:
        ax3.plot(federated_history['train_accuracy'], 'b-', label='Federated', linewidth=2)
    if 'train_accuracy' in centralized_history:
        ax3.plot(centralized_history['train_accuracy'], 'r--', label='Centralized', linewidth=2)
    ax3.set_title('Training Accuracy', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Round/Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Train Loss
    ax4 = axes[1, 1]
    if 'train_loss' in federated_history:
        ax4.plot(federated_history['train_loss'], 'b-', label='Federated', linewidth=2)
    if 'train_loss' in centralized_history:
        ax4.plot(centralized_history['train_loss'], 'r--', label='Centralized', linewidth=2)
    ax4.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Round/Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Training Comparison: Federated vs Centralized', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to '{save_path}'")
    plt.close()


def generate_comparison_report(
    federated_history: Dict,
    centralized_history: Dict,
    config: Dict = None,
    save_path: str = 'results/performance_report.json'
) -> Dict:
    """
    Generate comprehensive performance comparison report.

    Args:
        federated_history: Federated training history
        centralized_history: Centralized training history
        config: Training configuration
        save_path: Path to save JSON report

    Returns:
        Report dictionary
    """
    fed_final_acc = federated_history['test_accuracy'][-1]
    cent_final_acc = centralized_history['test_accuracy'][-1]
    accuracy_retention = (fed_final_acc / cent_final_acc) * 100 if cent_final_acc > 0 else 0

    fed_time = sum(federated_history.get('round_time', [0]))
    cent_time = centralized_history.get('training_time', 0)

    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'federated_accuracy': round(fed_final_acc, 4),
            'centralized_accuracy': round(cent_final_acc, 4),
            'accuracy_retention': round(accuracy_retention, 2),
            'accuracy_gap': round(cent_final_acc - fed_final_acc, 4),
            'privacy_preserved': True
        },
        'federated_model': {
            'final_accuracy': round(fed_final_acc, 4),
            'final_loss': round(federated_history['test_loss'][-1], 4),
            'total_rounds': len(federated_history['test_accuracy']),
            'training_time_seconds': round(fed_time, 2),
            'best_accuracy': round(max(federated_history['test_accuracy']), 4),
            'convergence_round': int(np.argmax(federated_history['test_accuracy']) + 1),
            'privacy': 'High (Data stays local)'
        },
        'centralized_model': {
            'final_accuracy': round(cent_final_acc, 4),
            'final_loss': round(centralized_history['test_loss'][-1], 4),
            'total_epochs': len(centralized_history['test_accuracy']),
            'training_time_seconds': round(cent_time, 2),
            'best_accuracy': round(max(centralized_history['test_accuracy']), 4),
            'privacy': 'None (Data aggregated centrally)'
        },
        'communication': {
            'total_bytes': federated_history.get('bytes_communicated', [0])[-1],
            'total_megabytes': round(federated_history.get('bytes_communicated', [0])[-1] / (1024*1024), 2)
        }
    }

    if config:
        report['configuration'] = config

    # Save JSON report
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Saved performance report to '{save_path}'")

    # Also save as CSV for easy viewing
    csv_path = save_path.replace('.json', '.csv')
    summary_df = pd.DataFrame([report['summary']])
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved summary CSV to '{csv_path}'")

    return report


def log_communication(
    round_num: int,
    client_id: int,
    metrics: Dict,
    log_path: str = 'results/communication_logs.csv'
) -> None:
    """Log communication data per round."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    header = not os.path.exists(log_path)
    data = {'round': round_num, 'client_id': client_id, **metrics}

    df = pd.DataFrame([data])
    df.to_csv(log_path, mode='a', header=header, index=False)


def save_training_log(
    federated_history: Dict,
    centralized_history: Dict,
    save_path: str = 'results/training_log.txt'
) -> None:
    """Save detailed training log to text file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("FEDERATED LEARNING TRAINING LOG\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("-" * 40 + "\n")
        f.write("FEDERATED LEARNING RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Rounds: {len(federated_history['test_accuracy'])}\n")
        f.write(f"Final Accuracy: {federated_history['test_accuracy'][-1]:.4f}\n")
        f.write(f"Final Loss: {federated_history['test_loss'][-1]:.4f}\n")
        f.write(f"Best Accuracy: {max(federated_history['test_accuracy']):.4f}\n\n")

        f.write("-" * 40 + "\n")
        f.write("CENTRALIZED TRAINING RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Epochs: {len(centralized_history['test_accuracy'])}\n")
        f.write(f"Final Accuracy: {centralized_history['test_accuracy'][-1]:.4f}\n")
        f.write(f"Final Loss: {centralized_history['test_loss'][-1]:.4f}\n")
        f.write(f"Best Accuracy: {max(centralized_history['test_accuracy']):.4f}\n\n")

        f.write("-" * 40 + "\n")
        f.write("ROUND-BY-ROUND FEDERATED METRICS\n")
        f.write("-" * 40 + "\n")
        for i in range(len(federated_history['test_accuracy'])):
            f.write(f"Round {i+1}: Acc={federated_history['test_accuracy'][i]:.4f}, "
                   f"Loss={federated_history['test_loss'][i]:.4f}\n")

    print(f"Saved training log to '{save_path}'")


def generate_all_outputs(
    federated_history: Dict,
    centralized_history: Dict,
    config: Dict = None,
    results_dir: str = 'results'
) -> None:
    """Generate all output files and visualizations."""
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 50)
    print("GENERATING ALL OUTPUT FILES")
    print("=" * 50)

    # Generate plots
    plot_accuracy_comparison(
        federated_history, centralized_history,
        save_path=os.path.join(results_dir, 'accuracy_comparison.png')
    )

    plot_loss_comparison(
        federated_history, centralized_history,
        save_path=os.path.join(results_dir, 'loss_comparison.png')
    )

    plot_communication_overhead(
        federated_history,
        save_path=os.path.join(results_dir, 'communication_overhead.png')
    )

    plot_training_curves(
        federated_history, centralized_history,
        save_path=os.path.join(results_dir, 'training_curves.png')
    )

    # Generate reports
    report = generate_comparison_report(
        federated_history, centralized_history, config,
        save_path=os.path.join(results_dir, 'performance_report.json')
    )

    save_training_log(
        federated_history, centralized_history,
        save_path=os.path.join(results_dir, 'training_log.txt')
    )

    print("\n" + "=" * 50)
    print("ALL OUTPUTS GENERATED SUCCESSFULLY")
    print("=" * 50)
    print(f"\nResults saved to: {results_dir}/")
    print(f"  - accuracy_comparison.png")
    print(f"  - loss_comparison.png")
    print(f"  - communication_overhead.png")
    print(f"  - training_curves.png")
    print(f"  - performance_report.json")
    print(f"  - performance_report.csv")
    print(f"  - training_log.txt")

    return report


if __name__ == '__main__':
    # Test with dummy data
    fed_history = {
        'test_accuracy': [0.7, 0.8, 0.85, 0.9, 0.92],
        'test_loss': [1.0, 0.6, 0.4, 0.3, 0.25],
        'train_accuracy': [0.65, 0.75, 0.82, 0.88, 0.91],
        'train_loss': [1.2, 0.7, 0.5, 0.35, 0.28],
        'bytes_communicated': [1e6, 2e6, 3e6, 4e6, 5e6],
        'round_time': [10, 10, 10, 10, 10]
    }

    cent_history = {
        'test_accuracy': [0.72, 0.82, 0.87, 0.91, 0.93],
        'test_loss': [0.9, 0.5, 0.35, 0.25, 0.2],
        'train_accuracy': [0.68, 0.78, 0.84, 0.89, 0.92],
        'train_loss': [1.1, 0.6, 0.45, 0.3, 0.22],
        'training_time': 50
    }

    generate_all_outputs(fed_history, cent_history)
