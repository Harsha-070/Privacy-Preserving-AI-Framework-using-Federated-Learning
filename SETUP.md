# Universal Setup Guide - Works on All Machines

This guide ensures the Federated Learning framework works on **any machine** (Windows, Linux, Mac) regardless of specifications.

---

## 🚀 Quick Start (3 Steps)

### Step 1: Install Python
**Minimum Required: Python 3.8+**

Download from: https://www.python.org/downloads/

**During installation:**
- ✅ Check "Add Python to PATH"
- ✅ Check "Install pip"

**Verify installation:**
```bash
python --version
pip --version
```

---

### Step 2: Install Dependencies

Open terminal/command prompt in project folder:

```bash
pip install -r requirements.txt
```

**If you get errors, try:**
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

**For slow internet / behind firewall:**
```bash
pip install -r requirements.txt --timeout 120
```

---

### Step 3: Run the Application

**Option A: Universal Launcher (Recommended)**
```bash
python run.py
```
This auto-detects your system and configures everything!

**Option B: Quick Test (2-3 minutes)**
```bash
python quick_start.py
```

**Option C: Web Dashboard**
```bash
python -m streamlit run frontend/app.py
```

---

## 💻 System Requirements

### Minimum (Will Auto-Configure)
- **RAM:** 2 GB (script will use lightweight settings)
- **Disk:** 1 GB free space
- **CPU:** Any modern processor
- **OS:** Windows 7+, Ubuntu 18.04+, macOS 10.14+

### Recommended
- **RAM:** 4-8 GB
- **Disk:** 5 GB free space
- **CPU:** 2+ cores

**Note:** The `run.py` script automatically adjusts parameters based on your system's capabilities!

---

## 🔧 Platform-Specific Instructions

### Windows

**Option 1: Command Prompt**
```cmd
cd path\to\project
python run.py
```

**Option 2: PowerShell**
```powershell
cd path\to\project
python run.py
```

**Option 3: Double-click**
- Right-click `run.py` → "Open with" → Python

---

### Linux / Ubuntu

```bash
cd /path/to/project
python3 run.py
```

**If `python3` not found:**
```bash
sudo apt update
sudo apt install python3 python3-pip
```

---

### macOS

```bash
cd /path/to/project
python3 run.py
```

**If Python not installed:**
```bash
# Install Homebrew first
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python3
```

---

## ⚡ Fast Setup (One Command)

### Windows (PowerShell)
```powershell
git clone <repo-url> ; cd <project> ; pip install -r requirements.txt ; python run.py
```

### Linux/Mac
```bash
git clone <repo-url> && cd <project> && pip install -r requirements.txt && python run.py
```

---

## 🐛 Common Issues & Fixes

### Issue 1: "Python not recognized"
**Fix:** Python not in PATH. Reinstall Python with "Add to PATH" checked.

**Quick Fix (Windows):**
```cmd
set PATH=%PATH%;C:\Python39
```

---

### Issue 2: "No module named 'tensorflow'"
**Fix:** Dependencies not installed.
```bash
pip install tensorflow numpy streamlit matplotlib seaborn plotly psutil
```

---

### Issue 3: "OpenBLAS memory allocation failed"
**Fix:** Already handled in code! Just rerun:
```bash
python run.py
```

If still failing:
```bash
set TF_ENABLE_ONEDNN_OPTS=0
set OMP_NUM_THREADS=1
python run.py
```

---

### Issue 4: "Application hangs/freezes for 2+ hours"
**Fix:** System too slow or RAM too low. Use quick test:

```bash
python quick_start.py
```

This completes in **2-5 minutes** with minimal resources.

---

### Issue 5: "Streamlit not found"
**Fix:**
```bash
pip install streamlit
python -m streamlit run frontend/app.py
```

---

### Issue 6: Slow dataset download
**Fix:** First run downloads MNIST (~11MB). This is normal.

**Speed up:**
```python
# Downloads happen automatically, but you can pre-download:
python -c "import tensorflow as tf; tf.keras.datasets.mnist.load_data()"
```

---

## 📊 Running Your First Experiment

### For Low-End Machines (< 4GB RAM)
```bash
python backend/main.py --dataset mnist --clients 2 --rounds 3 --local_epochs 1
```
**Time:** 3-5 minutes

### For Mid-Range Machines (4-8GB RAM)
```bash
python backend/main.py --dataset mnist --clients 5 --rounds 5 --local_epochs 2
```
**Time:** 5-10 minutes

### For High-End Machines (> 8GB RAM)
```bash
python backend/main.py --dataset mnist --clients 10 --rounds 10 --local_epochs 2
```
**Time:** 10-20 minutes

---

## 🌐 Using the Web Dashboard

After training completes:

```bash
python -m streamlit run frontend/app.py
```

**Features:**
- ✅ Interactive charts (Plotly)
- ✅ Real-time metrics
- ✅ Run new experiments
- ✅ Export reports

**Default URL:** http://localhost:8501

---

## 🔥 Testing Installation

Run this quick test to verify everything works:

```bash
python -c "import tensorflow as tf; import numpy as np; import streamlit; print('✓ All imports successful!')"
```

If no errors appear, you're ready!

---

## 📦 Offline Installation

If you don't have internet on target machine:

### On Internet-Connected Machine:
```bash
pip download -r requirements.txt -d packages/
```

### On Offline Machine:
```bash
pip install --no-index --find-links packages/ -r requirements.txt
```

---

## 🆘 Still Having Issues?

### Check System Info:
```bash
python -c "import platform; print(f'OS: {platform.system()}\nPython: {platform.python_version()}')"
```

### Full Diagnostic:
```bash
python run.py
# Select option 1 (Quick Test)
```

This will run diagnostics and show exactly what's wrong.

---

## 📝 Files Explained

| File | Purpose | When to Use |
|------|---------|-------------|
| `run.py` | Universal launcher with auto-config | **Start here!** |
| `quick_start.py` | Fast 3-minute test | Low RAM / Testing |
| `backend/main.py` | Full training script | Custom parameters |
| `frontend/app.py` | Web dashboard | View results |
| `TROUBLESHOOTING.md` | Detailed error fixes | When errors occur |

---

## ✅ Verification Checklist

Before sharing with client, verify:

- [ ] `python run.py` works
- [ ] `python quick_start.py` completes in < 5 minutes
- [ ] Dashboard opens: `python -m streamlit run frontend/app.py`
- [ ] No OpenBLAS errors
- [ ] Results saved in `results/` folder
- [ ] Models saved as `.h5` files

---

## 🎯 Client Instructions (Simple)

**For non-technical users, give them this:**

1. **Install Python 3.8+** from python.org
2. **Open terminal** in project folder
3. **Run:** `pip install -r requirements.txt`
4. **Run:** `python run.py`
5. **Choose option 1** (Quick Test)

That's it! The script handles everything else.

---

## 📞 Support

If you encounter any issues:

1. Check `TROUBLESHOOTING.md`
2. Run `python run.py` → Option 1 (Quick Test)
3. Check system requirements (min 2GB RAM)
4. Try `python quick_start.py` for minimal version

---

**Note:** All files have been updated with OpenBLAS fixes and memory optimizations. Just use the latest version!
