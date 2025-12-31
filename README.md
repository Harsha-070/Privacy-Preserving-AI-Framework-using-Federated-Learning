# Privacy Preserving AI Framework using Federated Learning

A comprehensive **Federated Learning (FL)** framework implementing the FedAvg algorithm with TensorFlow. Train deep learning models across distributed clients while keeping data private - raw data never leaves the client.

## Features

- **Privacy-First**: Raw data never leaves clients - only model weight updates are shared
- **Federated Averaging (FedAvg)**: Core FL aggregation algorithm with weighted averaging
- **Multiple Datasets**: Supports MNIST, Fashion-MNIST, and CIFAR-10
- **Data Distribution**: IID, Non-IID (shard-based), and Dirichlet distribution options
- **FedProx Support**: Optional proximal term for handling heterogeneous data
- **Interactive Dashboard**: Streamlit-based UI to configure, run, and visualize simulations
- **Comprehensive Metrics**: Accuracy/loss curves, communication overhead, client data distribution
- **Baseline Comparison**: Centralized training baseline to verify FL performance

## Project Structure

```
.
├── backend/                    # Core Federated Learning Logic
│   ├── config.py               # Configuration management
│   ├── data_loader.py          # Dataset loading and partitioning
│   ├── models.py               # CNN model definitions
│   ├── federated_core.py       # FedAvg client/server implementation
│   ├── centralized_trainer.py  # Baseline centralized training
│   ├── utils.py                # Visualization and output generation
│   └── main.py                 # CLI orchestrator script
├── frontend/                   # User Interface
│   └── app.py                  # Streamlit Web Dashboard
├── results/                    # Generated outputs
│   ├── accuracy_comparison.png # Federated vs Centralized accuracy
│   ├── loss_comparison.png     # Training loss curves
│   ├── communication_overhead.png # Data transfer visualization
│   ├── client_data_distribution.png # Non-IID data distribution
│   ├── performance_report.json # Detailed metrics (JSON)
│   ├── performance_report.csv  # Summary metrics (CSV)
│   └── training_log.txt        # Training session log
├── requirements.txt            # Python dependencies
├── config_example.yaml         # Example configuration file
└── README.md                   # This file
```

## Getting Started

### Prerequisites

Python 3.8+ required. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Project

#### Option A: Interactive Web Dashboard (Recommended)

Launch the Streamlit interface for easy configuration and visualization:

```bash
streamlit run frontend/app.py
```

Open the URL shown in terminal (usually http://localhost:8501).

#### Option B: Command Line Interface

Run simulations directly with custom parameters:

```bash
python backend/main.py --dataset mnist --clients 5 --rounds 10 --local_epochs 2 --distribution non-iid
```

**CLI Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | mnist | Dataset: mnist, fashion_mnist, cifar10 |
| `--clients` | 10 | Total number of clients |
| `--clients_per_round` | 5 | Clients selected per round |
| `--rounds` | 10 | Communication rounds |
| `--local_epochs` | 2 | Local training epochs per round |
| `--batch_size` | 32 | Training batch size |
| `--client_lr` | 0.01 | Client learning rate |
| `--distribution` | non-iid | Data distribution: iid, non-iid, dirichlet |
| `--seed` | 42 | Random seed for reproducibility |
| `--results_dir` | results | Output directory |

## How It Works

1. **Data Partitioning**: Dataset is split across N clients using IID or Non-IID distribution
2. **Global Model Init**: Server initializes a CNN model (architecture varies by dataset)
3. **Training Rounds**:
   - Server broadcasts global model weights to selected clients
   - Clients train locally on their private data for specified epochs
   - Clients compute weight deltas and send back to server
   - Server aggregates updates using weighted FedAvg
4. **Evaluation**: Global model is evaluated on test set after each round
5. **Comparison**: Results compared against centralized training baseline

## Generated Outputs

After training completes, the `results/` directory contains:

- **Visual Graphs** (PNG):
  - `accuracy_comparison.png` - Federated vs Centralized accuracy over rounds
  - `loss_comparison.png` - Training loss comparison
  - `communication_overhead.png` - Cumulative data transfer
  - `client_data_distribution.png` - Label distribution heatmap across clients

- **Performance Reports**:
  - `performance_report.json` - Complete metrics in JSON format
  - `performance_report.csv` - Summary for easy analysis

- **Trained Models**:
  - `federated_model_final.h5` - Final federated model
  - `centralized_model_final.h5` - Baseline centralized model

## Sample Results

With default settings (MNIST, 5 clients, 10 rounds, Non-IID):

| Metric | Federated | Centralized |
|--------|-----------|-------------|
| Final Accuracy | ~76% | ~99% |
| Privacy | Preserved | Not Preserved |
| Data Sharing | Weights Only | Full Dataset |

Accuracy retention varies based on:
- Data heterogeneity (IID vs Non-IID)
- Number of clients and participation rate
- Local epochs and learning rate

## Technical Notes

- **TensorFlow Optimization**: If you encounter MKL errors, set environment variable:
  ```bash
  set TF_ENABLE_ONEDNN_OPTS=0  # Windows
  export TF_ENABLE_ONEDNN_OPTS=0  # Linux/Mac
  ```

- **GPU Support**: TensorFlow will automatically use GPU if available

- **Memory**: CIFAR-10 requires more memory than MNIST/Fashion-MNIST

## References

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) - Original FedAvg paper
- [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127) - FedProx paper

---
*Privacy Preserving AI Framework - Federated Learning Implementation*
