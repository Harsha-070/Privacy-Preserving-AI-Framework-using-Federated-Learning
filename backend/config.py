"""
Configuration module for Federated Learning System.
Provides centralized configuration management with YAML/JSON support.
"""

import os
import json
import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


@dataclass
class DataConfig:
    """Data loading and partitioning configuration."""
    dataset_name: str = 'mnist'
    data_path: Optional[str] = None
    num_clients: int = 10
    distribution: str = 'non-iid'  # 'iid', 'non-iid', 'dirichlet'
    shards_per_client: int = 2
    dirichlet_alpha: float = 0.5  # For Dirichlet distribution
    min_samples_per_client: int = 10
    max_samples_per_client: Optional[int] = None
    validation_split: float = 0.1
    normalize: bool = True
    augmentation: bool = False


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    architecture: str = 'cnn'  # 'cnn', 'mlp', 'resnet', 'custom'
    input_shape: tuple = (28, 28, 1)
    num_classes: int = 10
    dropout_rate: float = 0.5
    use_batch_norm: bool = False
    hidden_units: List[int] = field(default_factory=lambda: [128, 64])


@dataclass
class FederatedConfig:
    """Federated learning configuration."""
    num_rounds: int = 100
    clients_per_round: int = 10
    local_epochs: int = 5
    batch_size: int = 32
    client_learning_rate: float = 0.01
    server_learning_rate: float = 1.0
    client_optimizer: str = 'sgd'  # 'sgd', 'adam'
    server_optimizer: str = 'sgd'
    use_tff: bool = False  # Use TensorFlow Federated or custom implementation
    aggregation_method: str = 'fedavg'  # 'fedavg', 'fedprox', 'scaffold'
    fedprox_mu: float = 0.01  # FedProx proximal term coefficient


@dataclass
class PrivacyConfig:
    """Privacy and security configuration."""
    enable_dp: bool = False  # Differential Privacy
    dp_noise_multiplier: float = 1.0
    dp_l2_norm_clip: float = 1.0
    dp_delta: float = 1e-5
    secure_aggregation: bool = False


@dataclass
class TrainingConfig:
    """Training execution configuration."""
    seed: int = 42
    checkpoint_interval: int = 10
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.001
    verbose: int = 1


@dataclass
class OutputConfig:
    """Output and logging configuration."""
    results_dir: str = 'results'
    save_model: bool = True
    save_checkpoints: bool = True
    save_logs: bool = True
    save_plots: bool = True
    log_level: str = 'INFO'


@dataclass
class Config:
    """Main configuration class combining all sub-configurations."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'federated': asdict(self.federated),
            'privacy': asdict(self.privacy),
            'training': asdict(self.training),
            'output': asdict(self.output)
        }

    def save_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def save_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def from_json(cls, path: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary."""
        return cls(
            data=DataConfig(**data.get('data', {})),
            model=ModelConfig(**data.get('model', {})),
            federated=FederatedConfig(**data.get('federated', {})),
            privacy=PrivacyConfig(**data.get('privacy', {})),
            training=TrainingConfig(**data.get('training', {})),
            output=OutputConfig(**data.get('output', {}))
        )


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def create_config_from_args(args) -> Config:
    """Create configuration from argparse arguments."""
    config = Config()

    # Map command line arguments to config
    if hasattr(args, 'dataset'):
        config.data.dataset_name = args.dataset
    if hasattr(args, 'num_clients') or hasattr(args, 'clients'):
        config.data.num_clients = getattr(args, 'num_clients', getattr(args, 'clients', 10))
    if hasattr(args, 'distribution'):
        config.data.distribution = args.distribution
    if hasattr(args, 'rounds'):
        config.federated.num_rounds = args.rounds
    if hasattr(args, 'local_epochs'):
        config.federated.local_epochs = args.local_epochs
    if hasattr(args, 'batch_size'):
        config.federated.batch_size = args.batch_size
    if hasattr(args, 'client_lr'):
        config.federated.client_learning_rate = args.client_lr
    if hasattr(args, 'server_lr'):
        config.federated.server_learning_rate = args.server_lr
    if hasattr(args, 'seed'):
        config.training.seed = args.seed
    if hasattr(args, 'enable_dp'):
        config.privacy.enable_dp = args.enable_dp
    if hasattr(args, 'use_tff'):
        config.federated.use_tff = args.use_tff

    return config


# Dataset-specific configurations
DATASET_CONFIGS = {
    'mnist': {
        'input_shape': (28, 28, 1),
        'num_classes': 10,
        'normalize_mean': 0.1307,
        'normalize_std': 0.3081
    },
    'fashion_mnist': {
        'input_shape': (28, 28, 1),
        'num_classes': 10,
        'normalize_mean': 0.2860,
        'normalize_std': 0.3530
    },
    'cifar10': {
        'input_shape': (32, 32, 3),
        'num_classes': 10,
        'normalize_mean': (0.4914, 0.4822, 0.4465),
        'normalize_std': (0.2470, 0.2435, 0.2616)
    },
    'cifar100': {
        'input_shape': (32, 32, 3),
        'num_classes': 100,
        'normalize_mean': (0.5071, 0.4867, 0.4408),
        'normalize_std': (0.2675, 0.2565, 0.2761)
    }
}


if __name__ == '__main__':
    # Test configuration
    config = get_default_config()
    print("Default Configuration:")
    print(json.dumps(config.to_dict(), indent=2))

    # Save example configs
    config.save_yaml('config_example.yaml')
    config.save_json('config_example.json')
    print("\nSaved example configurations.")
