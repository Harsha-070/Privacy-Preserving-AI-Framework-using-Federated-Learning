# Troubleshooting Guide: OpenBLAS Memory Allocation Error

## Error Message
```
OpenBLAS error: memory allocation still failed after 10 retries, giving up
```

---

## Solutions (Try in Order)

### ✅ Solution 1: Use the Fixed Version (Already Applied)

The code has been updated to automatically fix this issue. The following environment variables are now set:

```python
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
```

**Simply re-run the application** - the error should be resolved.

---

### ✅ Solution 2: Set Environment Variables Before Running (Alternative)

If you still encounter issues, set these environment variables **before** running the code:

#### **Windows (Command Prompt)**
```cmd
set TF_ENABLE_ONEDNN_OPTS=0
set OMP_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
python -m streamlit run frontend/app.py
```

#### **Windows (PowerShell)**
```powershell
$env:TF_ENABLE_ONEDNN_OPTS="0"
$env:OMP_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"
python -m streamlit run frontend/app.py
```

#### **Linux/Mac**
```bash
export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -m streamlit run frontend/app.py
```

---

### ✅ Solution 3: Reduce Model Size/Batch Size

If the error persists, your system may have limited RAM. Modify these parameters:

**In Streamlit Dashboard:**
- Use **MNIST** instead of CIFAR-10 (smaller images)
- Reduce **Number of Clients** to 3-5
- Reduce **Communication Rounds** to 5-10
- Use **IID** distribution (faster)

**In Command Line:**
```bash
python backend/main.py --dataset mnist --clients 3 --rounds 5 --batch_size 16
```

---

### ✅ Solution 4: Reinstall NumPy with Correct BLAS

Sometimes the issue is with the NumPy installation using the wrong BLAS library.

```bash
pip uninstall numpy -y
pip install numpy
pip install --upgrade tensorflow
```

Or install a specific NumPy build:
```bash
pip install numpy==1.23.5
```

---

### ✅ Solution 5: Increase System Memory Limit

If running in a virtual environment or container, increase memory allocation:

**Docker:**
```bash
docker run --memory="4g" ...
```

**WSL (Windows Subsystem for Linux):**
Edit `.wslconfig`:
```ini
[wsl2]
memory=4GB
```

---

### ✅ Solution 6: Use CPU-Only TensorFlow

Uninstall GPU version and use CPU-only TensorFlow:
```bash
pip uninstall tensorflow tensorflow-gpu
pip install tensorflow-cpu
```

---

## System Requirements

**Minimum:**
- RAM: 4 GB
- Python: 3.8+
- Disk Space: 2 GB

**Recommended:**
- RAM: 8 GB
- Python: 3.9 or 3.10
- Disk Space: 5 GB

---

## Testing After Fix

Run this simple test to verify the fix:

```python
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np

print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)

# Simple test
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(28,)),
    tf.keras.layers.Dense(10, activation='softmax')
])
print("Model created successfully!")
```

If this runs without errors, the fix is working.

---

## Additional Help

1. **Check Python Version:**
   ```bash
   python --version
   ```
   Should be 3.8 or higher.

2. **Check Installed Packages:**
   ```bash
   pip list | grep -E "tensorflow|numpy"
   ```

3. **Monitor Memory Usage:**
   - **Windows:** Task Manager → Performance
   - **Linux:** `htop` or `free -h`
   - **Mac:** Activity Monitor

4. **Run with Reduced Settings:**
   ```bash
   python backend/main.py --dataset mnist --clients 3 --rounds 3 --local_epochs 1
   ```

---

## Still Having Issues?

If none of these solutions work, please provide:

1. Operating System (Windows/Linux/Mac)
2. Python version (`python --version`)
3. TensorFlow version (`pip show tensorflow`)
4. NumPy version (`pip show numpy`)
5. Total RAM available
6. Full error traceback

---

**Note:** The code has already been patched with Solution 1. Just re-run the application!
