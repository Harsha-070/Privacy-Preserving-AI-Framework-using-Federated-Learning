# Privacy-Preserving AI Framework using Federated Learning

Train AI models without sharing private data. Your data stays on your device - only the learning is shared.

---

## Quick Start (3 Easy Steps)

### Step 1: Install Python
Download Python 3.8+ from [python.org](https://www.python.org/downloads/)

**Important:** Check "Add Python to PATH" during installation!

### Step 2: Install Requirements
Open terminal/command prompt in this folder and run:
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application

**Easiest Method (Recommended):**
```
Windows: Double-click START.bat
Mac/Linux: Run bash start.sh
```

**Or use command line:**
```bash
python run.py
```

---

## What Does This Do?

This framework trains AI models using **Federated Learning** - a privacy-preserving technique where:

1. **Your data stays private** - Never leaves your device
2. **Only learning is shared** - Model improvements, not raw data
3. **Results are comparable** - Achieves ~90% of centralized accuracy

### Simple Example
Imagine 5 hospitals want to build an AI to detect diseases, but can't share patient data due to privacy laws. With Federated Learning:
- Each hospital trains locally on their own patient data
- They share only what the AI learned (model weights)
- A central server combines the learnings
- Result: Better AI for everyone, zero data sharing!

---

## How to Use

### Option 1: Quick Test (2-5 minutes)
```bash
python quick_start.py
```
Best for first-time users and testing.

### Option 2: Full Application
```bash
python run.py
```
Then choose from the menu:
- **Option 1**: Quick test
- **Option 2**: Full training (10-30 min)
- **Option 3**: View results in dashboard

### Option 3: Web Dashboard
```bash
python -m streamlit run frontend/app.py
```
Opens a beautiful interactive dashboard at http://localhost:8501

---

## What You'll See

After training, check the `results/` folder:

| File | What It Shows |
|------|---------------|
| `accuracy_comparison.png` | Graph comparing Federated vs Centralized accuracy |
| `loss_comparison.png` | Training loss over time |
| `training_curves.png` | All metrics in one view |
| `client_data_distribution.png` | How data was split across clients |
| `performance_report.json` | Detailed metrics |
| `federated_model_final.h5` | The trained AI model |

**Note:** All outputs are **dynamic** - they change based on your training parameters!

---

## Expected Results

| Metric | Value |
|--------|-------|
| Federated Accuracy | ~89-92% |
| Centralized Accuracy | ~98-99% |
| Accuracy Retention | ~90% |
| Privacy | 100% Preserved |

You get **90% of the performance** while keeping **100% of the privacy**!

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Python not found" | Reinstall Python with "Add to PATH" checked |
| "Module not found" | Run `pip install -r requirements.txt` |
| App freezes/hangs | Use `python quick_start.py` instead |
| OpenBLAS error | Already fixed - just run again |

For more help, see:
- `CLIENT_INSTRUCTIONS.md` - Step-by-step guide
- `TROUBLESHOOTING.md` - Detailed solutions
- `SETUP.md` - Platform-specific setup

---

## Complete Documentation

For detailed technical documentation, diagrams, and explanations, see:

**📄 Project_Documentation_Final.docx**

This Word document contains:
- Complete project overview
- Technical architecture
- Algorithm explanations
- Visual diagrams
- FAQ section

---

## Files Included

| File | Purpose |
|------|---------|
| `run.py` | Main launcher (recommended) |
| `quick_start.py` | Fast 2-5 minute test |
| `START.bat` | Windows one-click launcher |
| `start.sh` | Mac/Linux launcher |
| `backend/` | Core FL algorithms |
| `frontend/` | Web dashboard |
| `results/` | Generated outputs |
| `Project_Documentation_Final.docx` | Full documentation |

---

## System Requirements

- **Python:** 3.8 or higher
- **RAM:** 2 GB minimum (4+ GB recommended)
- **OS:** Windows, Mac, or Linux
- **Internet:** Required for first run (downloads dataset)

---

## Quick Reference

```bash
# Install
pip install -r requirements.txt

# Run (choose one)
python run.py              # Recommended
python quick_start.py      # Fast test
START.bat                  # Windows one-click

# View dashboard
python -m streamlit run frontend/app.py
```

---

## About Federated Learning

Federated Learning is used by:
- **Google** - Keyboard predictions (Gboard)
- **Apple** - Siri improvements
- **Hospitals** - Medical AI without sharing patient data
- **Banks** - Fraud detection across institutions

It's the future of privacy-preserving AI!

---

**Privacy-Preserving AI Framework** - *AI that learns without seeing your data.*

For complete documentation, open **Project_Documentation_Final.docx**
