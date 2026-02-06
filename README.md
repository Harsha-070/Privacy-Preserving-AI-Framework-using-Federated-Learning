# Privacy Preserving AI Framework using Federated Learning

A comprehensive **Federated Learning (FL)** framework implementing the FedAvg algorithm with TensorFlow. Train deep learning models across distributed clients while keeping data private - raw data never leaves the client.

## What is Federated Learning?

Federated Learning is a machine learning approach where AI models are trained across multiple devices (like phones or computers) without collecting data in one central location. Instead of sending your data to a server, the AI model comes to your device, learns from your data locally, and only sends back what it learned (model weights) - never your actual data.

**Simple Analogy**: Imagine 5 families want to create the best pizza recipe together, but no one wants to share their secret recipe. So instead, each family tries a basic recipe, makes improvements based on their secret knowledge, and shares only the improvement tips (not the recipe). A chef combines all tips to make a better recipe. That's Federated Learning!

## Features

- **Privacy-First**: Raw data never leaves clients - only model weight updates are shared
- **Federated Averaging (FedAvg)**: Core FL aggregation algorithm with weighted averaging
- **Multiple Datasets**: Supports MNIST, Fashion-MNIST, and CIFAR-10
- **Data Distribution**: IID, Non-IID (shard-based), and Dirichlet distribution options
- **FedProx Support**: Optional proximal term for handling heterogeneous data
- **Interactive Dashboard**: Beautiful Streamlit-based UI with Plotly interactive charts
- **Comprehensive Metrics**: Accuracy/loss curves, communication overhead, privacy analysis
- **Baseline Comparison**: Centralized training baseline to verify FL performance

## Technology Stack

| Technology | Purpose | Why We Use It |
|------------|---------|---------------|
| **Python 3.8+** | Programming Language | Easy to read, most popular for AI/ML |
| **TensorFlow 2.x** | Deep Learning Framework | Google's powerful AI library for neural networks |
| **Keras** | Neural Network API | Simplifies building and training models |
| **NumPy** | Numerical Computing | Fast mathematical operations on arrays |
| **Streamlit** | Web Dashboard | Turns Python scripts into interactive web apps |
| **Plotly** | Interactive Charts | Beautiful, zoomable, hoverable visualizations |
| **Matplotlib** | Static Plots | Generates comparison graphs and heatmaps |
| **python-docx** | Documentation | Generates Word documentation |

## Real-World Applications

Federated Learning is used by major companies to improve AI while respecting user privacy:

| Company | Application | How FL Helps |
|---------|-------------|--------------|
| **Google** | Gboard Keyboard | Improves next-word prediction without reading your messages |
| **Apple** | Siri & QuickType | Enhances voice recognition without listening to conversations |
| **Hospitals** | Medical AI | Trains diagnostic models across hospitals without sharing patient records |
| **Banks** | Fraud Detection | Detects fraud patterns across banks without exposing transaction data |
| **Automotive** | Self-Driving Cars | Improves driving AI using data from multiple vehicles |
| **Healthcare** | Drug Discovery | Enables pharmaceutical companies to collaborate without sharing proprietary data |
| **Smartphones** | Face Recognition | Improves facial recognition without uploading your photos |

### Why This Matters

- **GDPR Compliance**: Meets European data protection regulations
- **HIPAA Compliance**: Suitable for healthcare applications
- **User Trust**: Users keep control of their personal data
- **Reduced Risk**: No central data store = no single point of breach
- **Cost Savings**: No need to store and manage massive central datasets

## Project Structure

```
.
├── backend/                    # Core Federated Learning Logic
│   ├── config.py               # Configuration management (YAML/JSON)
│   ├── data_loader.py          # Dataset loading and Non-IID partitioning
│   ├── models.py               # CNN, MLP, ResNet model architectures
│   ├── federated_core.py       # FedAvg client update & server aggregation
│   ├── centralized_trainer.py  # Baseline centralized training
│   ├── utils.py                # Visualization and report generation
│   └── main.py                 # CLI orchestrator script
├── frontend/                   # User Interface
│   └── app.py                  # Streamlit Web Dashboard with Plotly charts
├── results/                    # Generated outputs
│   ├── accuracy_comparison.png # Federated vs Centralized accuracy
│   ├── loss_comparison.png     # Training loss curves
│   ├── training_curves.png     # Combined 4-panel metrics view
│   ├── communication_overhead.png # Data transfer visualization
│   ├── client_data_distribution.png # Non-IID data heatmap
│   ├── performance_report.json # Detailed metrics (JSON)
│   ├── performance_report.csv  # Summary metrics (CSV)
│   ├── training_log.txt        # Training session log
│   ├── federated_model_final.h5  # Trained federated model
│   └── centralized_model_final.h5 # Trained centralized model
├── requirements.txt            # Python dependencies
├── config_example.yaml         # Example configuration file
├── Project_Documentation_Final.docx # Complete project documentation
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
python -m streamlit run frontend/app.py
```

Open the URL shown in terminal (usually http://localhost:8501).

> **Note**: If `streamlit run` doesn't work, always use `python -m streamlit run` instead.

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

### The Federated Learning Process

```
┌─────────────────────────────────────────────────────────────────┐
│                         SERVER                                   │
│  1. Initialize global model                                      │
│  2. Send model to selected clients                              │
│  5. Aggregate updates using FedAvg                              │
│  6. Repeat until convergence                                    │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  CLIENT 1   │      │  CLIENT 2   │      │  CLIENT 3   │
│ 3. Train on │      │ 3. Train on │      │ 3. Train on │
│ local data  │      │ local data  │      │ local data  │
│ 4. Send     │      │ 4. Send     │      │ 4. Send     │
│ weights only│      │ weights only│      │ weights only│
│ (NOT data!) │      │ (NOT data!) │      │ (NOT data!) │
└─────────────┘      └─────────────┘      └─────────────┘
```

### Step-by-Step Process

1. **Data Partitioning**: Dataset is split across N clients using IID or Non-IID distribution
2. **Global Model Init**: Server initializes a CNN model (architecture varies by dataset)
3. **Training Rounds**:
   - Server broadcasts global model weights to selected clients
   - Clients train locally on their private data for specified epochs
   - Clients compute weight deltas and send back to server (NOT the data!)
   - Server aggregates updates using weighted FedAvg
4. **Evaluation**: Global model is evaluated on test set after each round
5. **Comparison**: Results compared against centralized training baseline

## Generated Outputs

After training completes, the `results/` directory contains:

### Visual Graphs (PNG)
- `accuracy_comparison.png` - Federated vs Centralized accuracy over rounds
- `loss_comparison.png` - Training loss comparison
- `training_curves.png` - Combined 4-panel view of all metrics
- `communication_overhead.png` - Cumulative data transfer
- `client_data_distribution.png` - Label distribution heatmap across clients

### Performance Reports
- `performance_report.json` - Complete metrics in JSON format
- `performance_report.csv` - Summary for easy analysis
- `training_log.txt` - Detailed training session log

### Trained Models
- `federated_model_final.h5` - Final federated model
- `centralized_model_final.h5` - Baseline centralized model

## Sample Results

With optimized settings (MNIST, 5 clients, 10 rounds, Non-IID):

| Metric | Federated | Centralized |
|--------|-----------|-------------|
| Final Accuracy | **89.30%** | 99.22% |
| Privacy | **100% Preserved** | Not Preserved |
| Data Sharing | Weights Only | Full Dataset |
| Accuracy Retention | **90%** | Baseline |

### Key Insights
- We achieve **90% of centralized performance** while maintaining **100% privacy**
- Trade-off: ~10% accuracy loss for complete data protection
- This is an excellent trade-off for privacy-sensitive applications!

## Dashboard Features

Our Streamlit dashboard includes:

- **5 Key Metrics**: Federated Accuracy, Centralized Accuracy, Accuracy Retention, Privacy Status, Communication Cost
- **Interactive Plotly Charts**: Hover for exact values, zoom, pan
- **5 Tabs**: Performance, Training Analysis, Privacy & Security, Configuration, Run Simulation
- **Privacy Radar Chart**: Visual comparison of privacy vs utility trade-offs
- **Export Options**: Download reports as JSON, CSV, or TXT

## Technical Notes

- **TensorFlow Optimization**: If you encounter MKL errors, set environment variable:
  ```bash
  set TF_ENABLE_ONEDNN_OPTS=0  # Windows
  export TF_ENABLE_ONEDNN_OPTS=0  # Linux/Mac
  ```

- **GPU Support**: TensorFlow will automatically use GPU if available

- **Memory**: CIFAR-10 requires more memory than MNIST/Fashion-MNIST

## Key Algorithms

### FedAvg (Federated Averaging)
The core aggregation algorithm that combines client updates:
```
w_global = Σ (n_k / n_total) * w_k
```
Where `n_k` is the number of samples on client k, and `w_k` is the client's model weights.

### Non-IID Data Distribution
Realistic data partitioning where each client has different data distributions (e.g., one client has mostly 0s and 1s, another has 8s and 9s).

## Documentation

Complete project documentation is available in `Project_Documentation_Final.docx` which includes:
- Simple explanations suitable for beginners
- Visual diagrams and flowcharts
- Technical details for advanced users
- FAQ section for common questions

## References

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) - Original FedAvg paper (McMahan et al., 2017)
- [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127) - FedProx paper
- [Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977) - Comprehensive FL survey

## License

This project is for educational purposes.

---

**Privacy Preserving AI Framework** - *Because AI should be smart without being nosy.*

*Built with TensorFlow, Streamlit & Python*
