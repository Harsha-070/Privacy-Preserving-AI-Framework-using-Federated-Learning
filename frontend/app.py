import streamlit as st
import subprocess
import os
import json

st.set_page_config(page_title="Federated Learning Dashboard", layout="wide")

st.title("Privacy Preserving AI Framework")
st.markdown("### Federated Learning with TensorFlow")

# Sidebar for Configuration
st.sidebar.header("Simulation Parameters")
dataset = st.sidebar.selectbox("Dataset", ["mnist", "fashion_mnist", "cifar10"])
num_clients = st.sidebar.slider("Number of Clients", min_value=2, max_value=10, value=5)
clients_per_round = st.sidebar.slider("Clients per Round", min_value=1, max_value=num_clients, value=min(3, num_clients))
training_rounds = st.sidebar.slider("Communication Rounds", min_value=1, max_value=20, value=5)
local_epochs = st.sidebar.slider("Local Epochs per Round", min_value=1, max_value=5, value=1)
distribution = st.sidebar.selectbox("Data Distribution", ["non-iid", "iid", "dirichlet"])

if st.sidebar.button("Run Simulation"):
    with st.spinner("Running Federated Learning Simulation... This may take a while."):
        env = os.environ.copy()
        env["TF_ENABLE_ONEDNN_OPTS"] = "0"

        cmd = [
            "python", "backend/main.py",
            "--dataset", dataset,
            "--clients", str(num_clients),
            "--clients_per_round", str(clients_per_round),
            "--rounds", str(training_rounds),
            "--local_epochs", str(local_epochs),
            "--distribution", distribution
        ]

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
        )
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            st.success("Simulation Completed Successfully!")
            st.text_area("Output", stdout[-2000:] if len(stdout) > 2000 else stdout, height=200)
        else:
            st.error("Simulation Failed!")
            st.code(stderr[-1000:] if len(stderr) > 1000 else stderr)

st.markdown("---")

results_dir = "results"

# Performance Report
report_file = os.path.join(results_dir, "performance_report.json")
if os.path.exists(report_file):
    with open(report_file, "r") as f:
        report = json.load(f)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Federated Accuracy", f"{report.get('federated_accuracy', 0):.4f}")
    with col2:
        st.metric("Centralized Accuracy", f"{report.get('centralized_accuracy', 0):.4f}")
    with col3:
        retention = report.get('accuracy_retention', 0)
        st.metric("Accuracy Retention", f"{retention:.2f}%")

st.markdown("---")

# Display Plots
col1, col2 = st.columns(2)

with col1:
    st.header("Accuracy Comparison")
    accuracy_plot = os.path.join(results_dir, "accuracy_comparison.png")
    if os.path.exists(accuracy_plot):
        st.image(accuracy_plot)
    else:
        st.info("Run simulation to see accuracy plot.")

with col2:
    st.header("Loss Comparison")
    loss_plot = os.path.join(results_dir, "loss_comparison.png")
    if os.path.exists(loss_plot):
        st.image(loss_plot)
    else:
        st.info("Run simulation to see loss plot.")

col3, col4 = st.columns(2)

with col3:
    st.header("Communication Overhead")
    comm_plot = os.path.join(results_dir, "communication_overhead.png")
    if os.path.exists(comm_plot):
        st.image(comm_plot)
    else:
        st.info("Run simulation to see communication plot.")

with col4:
    st.header("Client Data Distribution")
    dist_plot = os.path.join(results_dir, "client_data_distribution.png")
    if os.path.exists(dist_plot):
        st.image(dist_plot)
    else:
        st.info("Run simulation to see distribution plot.")

# Training Log
st.markdown("---")
st.header("Training Log")
log_file = os.path.join(results_dir, "training_log.txt")
if os.path.exists(log_file):
    with open(log_file, "r") as f:
        st.text_area("Log", f.read(), height=300)
else:
    st.info("Run simulation to see logs.")

st.sidebar.markdown("---")
st.sidebar.info(
    "**How to use:**\n"
    "1. Select dataset and parameters.\n"
    "2. Click 'Run Simulation'.\n"
    "3. View results and visualizations."
)
