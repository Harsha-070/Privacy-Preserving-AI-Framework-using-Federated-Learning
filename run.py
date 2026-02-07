"""
Universal Launcher - Works on All Machines
Auto-detects system capabilities and adjusts parameters accordingly
"""

import os
import sys
import platform
import subprocess
import json
import warnings

# Critical: Set these BEFORE any other imports
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow warnings
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Suppress harmless protobuf version warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

def get_system_info():
    """Detect system capabilities"""
    import psutil

    info = {
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'cpu_count': os.cpu_count() or 2,
        'ram_gb': round(psutil.virtual_memory().total / (1024**3), 1)
    }
    return info

def check_dependencies():
    """Check and install missing dependencies"""
    required = {
        'tensorflow': 'tensorflow',
        'numpy': 'numpy',
        'streamlit': 'streamlit',
        'plotly': 'plotly',
        'psutil': 'psutil',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }

    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    return missing

def install_dependencies(packages):
    """Install missing packages"""
    print(f"📦 Installing missing packages: {', '.join(packages)}")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '--quiet', '--upgrade'
        ] + packages)
        print("✓ Dependencies installed successfully\n")
        return True
    except Exception as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("Please run manually: pip install -r requirements.txt\n")
        return False

def get_recommended_config(ram_gb):
    """Get recommended config based on system RAM"""
    if ram_gb < 4:
        return {
            'dataset': 'mnist',
            'clients': 2,
            'rounds': 3,
            'local_epochs': 1,
            'batch_size': 32,
            'description': 'Low RAM (< 4GB) - Minimal config'
        }
    elif ram_gb < 8:
        return {
            'dataset': 'mnist',
            'clients': 3,
            'rounds': 5,
            'local_epochs': 2,
            'batch_size': 32,
            'description': 'Medium RAM (4-8GB) - Standard config'
        }
    else:
        return {
            'dataset': 'mnist',
            'clients': 5,
            'rounds': 10,
            'local_epochs': 2,
            'batch_size': 64,
            'description': 'High RAM (> 8GB) - Full config'
        }

def run_quick_test():
    """Run a minimal test to verify setup"""
    print("🧪 Running quick system test...")
    try:
        import tensorflow as tf
        import numpy as np

        # Test TensorFlow
        a = tf.constant([[1, 2], [3, 4]])
        b = tf.constant([[5, 6], [7, 8]])
        c = tf.matmul(a, b)

        print(f"  ✓ TensorFlow {tf.__version__} working")
        print(f"  ✓ NumPy {np.__version__} working")
        print(f"  ✓ GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
        print()
        return True
    except Exception as e:
        print(f"  ❌ Test failed: {e}\n")
        return False

def run_training(config):
    """Run federated learning training"""
    print(f"🚀 Starting Federated Learning Training")
    print(f"   Config: {config['description']}")
    print(f"   Clients: {config['clients']}, Rounds: {config['rounds']}\n")

    cmd = [
        sys.executable,
        'backend/main.py',
        '--dataset', config['dataset'],
        '--clients', str(config['clients']),
        '--clients_per_round', str(max(2, config['clients'] - 1)),
        '--rounds', str(config['rounds']),
        '--local_epochs', str(config['local_epochs']),
        '--batch_size', str(config['batch_size'])
    ]

    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed!")
        print(f"Error output:\n{e.stderr}")
        return False
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        return False

def run_dashboard():
    """Launch Streamlit dashboard"""
    print("🌐 Launching Web Dashboard...")
    print("   This will open in your default browser\n")

    cmd = [sys.executable, '-m', 'streamlit', 'run', 'frontend/app.py']

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n⚠ Dashboard closed by user")

def main():
    print("\n" + "="*70)
    print("  Privacy-Preserving Federated Learning Framework")
    print("  Universal Launcher - Auto-configures for your system")
    print("="*70 + "\n")

    # Check Python version
    py_version = tuple(map(int, platform.python_version_tuple()[:2]))
    if py_version < (3, 8):
        print(f"❌ Python 3.8+ required. You have {platform.python_version()}")
        sys.exit(1)
    print(f"✓ Python {platform.python_version()}")

    # Check dependencies
    print("🔍 Checking dependencies...")
    missing = check_dependencies()
    if missing:
        print(f"⚠ Missing packages: {', '.join(missing)}")
        response = input("Install automatically? (y/n): ").strip().lower()
        if response == 'y':
            if not install_dependencies(missing):
                sys.exit(1)
        else:
            print("Please install manually: pip install -r requirements.txt")
            sys.exit(1)
    else:
        print("✓ All dependencies installed\n")

    # Get system info
    print("💻 Detecting system capabilities...")
    sys_info = get_system_info()
    print(f"   Platform: {sys_info['platform']}")
    print(f"   RAM: {sys_info['ram_gb']} GB")
    print(f"   CPU Cores: {sys_info['cpu_count']}\n")

    # Get recommended config
    config = get_recommended_config(sys_info['ram_gb'])

    # Run test
    if not run_quick_test():
        print("⚠ System test failed. Trying to continue anyway...\n")

    # Show menu
    while True:
        print("\n" + "="*70)
        print("  What would you like to do?")
        print("="*70)
        print("  1. Quick Test (3 minutes, recommended for first run)")
        print("  2. Run Full Training (with your system's optimal settings)")
        print("  3. Launch Dashboard (view previous results)")
        print("  4. Exit")
        print("="*70)

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == '1':
            # Quick test config
            test_config = {
                'dataset': 'mnist',
                'clients': 2,
                'rounds': 2,
                'local_epochs': 1,
                'batch_size': 64,
                'description': 'Quick Test (2-3 minutes)'
            }
            if run_training(test_config):
                print("\n✓ Quick test completed successfully!")
                print("  You can now run option 2 for full training")
            else:
                print("\n❌ Quick test failed. Check TROUBLESHOOTING.md")

        elif choice == '2':
            if run_training(config):
                print("\n✓ Training completed successfully!")
                print("  Run option 3 to view results in dashboard")
            else:
                print("\n❌ Training failed. Try option 1 (Quick Test) first")

        elif choice == '3':
            run_dashboard()

        elif choice == '4':
            print("\n👋 Goodbye!\n")
            sys.exit(0)

        else:
            print("❌ Invalid choice. Please enter 1-4")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\nPlease check TROUBLESHOOTING.md for help")
        import traceback
        traceback.print_exc()
        sys.exit(1)
