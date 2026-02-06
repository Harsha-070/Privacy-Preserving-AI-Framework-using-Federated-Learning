import streamlit as st
import subprocess
import os
import json
import re
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime

# ─── Page Config ───
st.set_page_config(
    page_title="FL Privacy Framework",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Global */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1117 50%, #0a0a1a 100%);
    }

    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 40px 50px;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 60%);
        border-radius: 50%;
    }
    .hero-container::after {
        content: '';
        position: absolute;
        bottom: -30%;
        left: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(255,255,255,0.05) 0%, transparent 60%);
        border-radius: 50%;
    }
    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        color: white;
        margin: 0 0 8px 0;
        letter-spacing: -0.5px;
        position: relative;
        z-index: 1;
    }
    .hero-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 400;
        color: rgba(255,255,255,0.85);
        margin: 0;
        position: relative;
        z-index: 1;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 16px;
        padding: 28px 24px;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.4);
    }
    .metric-card .icon {
        font-size: 2rem;
        margin-bottom: 10px;
    }
    .metric-card .label {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #8892b0;
        margin-bottom: 8px;
    }
    .metric-card .value {
        font-family: 'Inter', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .metric-card .subtext {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: #5a6785;
    }

    /* Green variant */
    .metric-card.green .value {
        background: linear-gradient(135deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card.green:hover {
        box-shadow: 0 12px 40px rgba(0, 210, 255, 0.15);
        border-color: rgba(0, 210, 255, 0.3);
    }

    /* Amber variant */
    .metric-card.amber .value {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card.amber:hover {
        box-shadow: 0 12px 40px rgba(245, 87, 108, 0.15);
        border-color: rgba(245, 87, 108, 0.3);
    }

    /* Gold variant */
    .metric-card.gold .value {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card.gold:hover {
        box-shadow: 0 12px 40px rgba(255, 210, 0, 0.15);
        border-color: rgba(255, 210, 0, 0.3);
    }

    /* Emerald variant */
    .metric-card.emerald .value {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card.emerald:hover {
        box-shadow: 0 12px 40px rgba(56, 239, 125, 0.15);
        border-color: rgba(56, 239, 125, 0.3);
    }

    /* Section headers */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 40px 0 20px 0;
        padding-bottom: 12px;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
        letter-spacing: -0.3px;
    }

    /* Glass panel */
    .glass-panel {
        background: rgba(22, 33, 62, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(102, 126, 234, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }

    /* Privacy badge */
    .privacy-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, rgba(56, 239, 125, 0.1), rgba(17, 153, 142, 0.1));
        border: 1px solid rgba(56, 239, 125, 0.3);
        border-radius: 50px;
        padding: 8px 20px;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        font-weight: 600;
        color: #38ef7d;
    }

    /* Comparison table */
    .comparison-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-family: 'Inter', sans-serif;
    }
    .comparison-table th {
        background: rgba(102, 126, 234, 0.15);
        color: #b8c1ec;
        font-weight: 600;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 14px 20px;
        text-align: left;
    }
    .comparison-table th:first-child { border-radius: 12px 0 0 0; }
    .comparison-table th:last-child { border-radius: 0 12px 0 0; }
    .comparison-table td {
        padding: 14px 20px;
        color: #e2e8f0;
        font-size: 0.95rem;
        border-bottom: 1px solid rgba(102, 126, 234, 0.08);
    }
    .comparison-table tr:last-child td:first-child { border-radius: 0 0 0 12px; }
    .comparison-table tr:last-child td:last-child { border-radius: 0 0 12px 0; }
    .comparison-table tr:hover td { background: rgba(102, 126, 234, 0.05); }
    .fed-value { color: #667eea; font-weight: 600; }
    .cent-value { color: #f5576c; font-weight: 600; }
    .winner { color: #38ef7d !important; }

    /* Config grid */
    .config-item {
        background: rgba(22, 33, 62, 0.4);
        border: 1px solid rgba(102, 126, 234, 0.1);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .config-key {
        font-family: 'Inter', sans-serif;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8892b0;
    }
    .config-val {
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-top: 2px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(22, 33, 62, 0.4);
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #8892b0;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.15);
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 30px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 30px 0 20px 0;
        color: #5a6785;
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        border-top: 1px solid rgba(102, 126, 234, 0.1);
        margin-top: 50px;
    }
    .footer a { color: #667eea; text-decoration: none; }

    /* Plotly chart container */
    .chart-container {
        background: rgba(22, 33, 62, 0.4);
        border: 1px solid rgba(102, 126, 234, 0.12);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: #b8c1ec !important;
        background: rgba(22, 33, 62, 0.4) !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Plotly Theme ───
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(13,17,23,0.8)',
    font=dict(family='Inter, sans-serif', color='#b8c1ec', size=13),
    margin=dict(l=50, r=30, t=50, b=50),
    legend=dict(
        bgcolor='rgba(22,33,62,0.6)',
        bordercolor='rgba(102,126,234,0.2)',
        borderwidth=1,
        font=dict(size=12)
    ),
    hoverlabel=dict(bgcolor='#1a1a2e', font_size=13, font_family='Inter'),
)

GRID_STYLE = dict(gridcolor='rgba(102,126,234,0.08)', zerolinecolor='rgba(102,126,234,0.15)')

FED_COLOR = '#667eea'
CENT_COLOR = '#f5576c'
ACCENT_1 = '#38ef7d'
ACCENT_2 = '#f7971e'
ACCENT_3 = '#00d2ff'

results_dir = "results"


def load_report():
    report_file = os.path.join(results_dir, "performance_report.json")
    if os.path.exists(report_file):
        with open(report_file, "r") as f:
            return json.load(f)
    return None


def load_training_log():
    log_file = os.path.join(results_dir, "training_log.txt")
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            return f.read()
    return None


def parse_round_data(log_text):
    """Parse round-by-round data from training log."""
    rounds, accs, losses = [], [], []
    for line in log_text.split('\n'):
        m = re.match(r'Round (\d+): Acc=([\d.]+), Loss=([\d.]+)', line)
        if m:
            rounds.append(int(m.group(1)))
            accs.append(float(m.group(2)))
            losses.append(float(m.group(3)))
    return rounds, accs, losses


# ─── Hero Section ───
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Privacy-Preserving AI Framework</div>
    <div class="hero-subtitle">Federated Learning with Differential Privacy &bull; Secure Model Training Without Data Sharing</div>
</div>
""", unsafe_allow_html=True)

# Load data
report = load_report()
log_text = load_training_log()

# ─── Key Metrics Row ───
if report:
    summary = report.get('summary', {})
    fed_model = report.get('federated_model', {})
    cent_model = report.get('centralized_model', {})
    comm = report.get('communication', {})
    config = report.get('configuration', {})

    fed_acc = summary.get('federated_accuracy', 0)
    cent_acc = summary.get('centralized_accuracy', 0)
    retention = summary.get('accuracy_retention', 0)
    gap = summary.get('accuracy_gap', 0)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="icon">🤖</div>
            <div class="label">Federated Accuracy</div>
            <div class="value">{fed_acc*100:.2f}%</div>
            <div class="subtext">FedAvg | {fed_model.get('total_rounds', 0)} rounds</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card green">
            <div class="icon">🎯</div>
            <div class="label">Centralized Accuracy</div>
            <div class="value">{cent_acc*100:.2f}%</div>
            <div class="subtext">Baseline | {cent_model.get('total_epochs', 0)} epochs</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card gold">
            <div class="icon">📊</div>
            <div class="label">Accuracy Retention</div>
            <div class="value">{retention:.1f}%</div>
            <div class="subtext">Gap: {gap:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card emerald">
            <div class="icon">🛡️</div>
            <div class="label">Privacy Status</div>
            <div class="value">Secure</div>
            <div class="subtext">Data never leaves clients</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="metric-card amber">
            <div class="icon">📡</div>
            <div class="label">Communication Cost</div>
            <div class="value">{comm.get('total_megabytes', 0):.1f}</div>
            <div class="subtext">MB total transferred</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Tabs ───
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "  Performance  ",
        "  Training Analysis  ",
        "  Privacy & Security  ",
        "  Configuration  ",
        "  Run Simulation  "
    ])

    # ═══════════════════════════════════════════
    # TAB 1: PERFORMANCE
    # ═══════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-header">Model Performance Comparison</div>', unsafe_allow_html=True)

        if log_text:
            rounds, fed_accs, fed_losses = parse_round_data(log_text)

            if rounds:
                col_a, col_b = st.columns(2)

                # Accuracy Chart
                with col_a:
                    fig_acc = go.Figure()
                    fig_acc.add_trace(go.Scatter(
                        x=rounds, y=fed_accs,
                        name='Federated Learning',
                        line=dict(color=FED_COLOR, width=3),
                        mode='lines+markers',
                        marker=dict(size=8, symbol='circle', line=dict(width=2, color='white')),
                        fill='tozeroy',
                        fillcolor='rgba(102,126,234,0.08)',
                        hovertemplate='Round %{x}<br>Accuracy: %{y:.4f}<extra>Federated</extra>'
                    ))
                    # Centralized baseline
                    fig_acc.add_trace(go.Scatter(
                        x=rounds, y=[cent_acc]*len(rounds),
                        name=f'Centralized ({cent_acc:.4f})',
                        line=dict(color=CENT_COLOR, width=2, dash='dash'),
                        hovertemplate='Centralized: %{y:.4f}<extra></extra>'
                    ))
                    fig_acc.update_layout(
                        title=dict(text='Accuracy Over Rounds', font=dict(size=16, color='#e2e8f0')),
                        xaxis_title='Communication Round',
                        yaxis_title='Test Accuracy',
                        yaxis=dict(range=[0, 1.05], **GRID_STYLE),
                        xaxis=dict(**GRID_STYLE),
                        height=420,
                        **PLOTLY_LAYOUT
                    )
                    st.plotly_chart(fig_acc, use_container_width=True)

                # Loss Chart
                with col_b:
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(
                        x=rounds, y=fed_losses,
                        name='Federated Learning',
                        line=dict(color=FED_COLOR, width=3),
                        mode='lines+markers',
                        marker=dict(size=8, symbol='circle', line=dict(width=2, color='white')),
                        fill='tozeroy',
                        fillcolor='rgba(102,126,234,0.08)',
                        hovertemplate='Round %{x}<br>Loss: %{y:.4f}<extra>Federated</extra>'
                    ))
                    cent_loss = cent_model.get('final_loss', 0)
                    fig_loss.add_trace(go.Scatter(
                        x=rounds, y=[cent_loss]*len(rounds),
                        name=f'Centralized ({cent_loss:.4f})',
                        line=dict(color=CENT_COLOR, width=2, dash='dash'),
                        hovertemplate='Centralized: %{y:.4f}<extra></extra>'
                    ))
                    fig_loss.update_layout(
                        title=dict(text='Loss Over Rounds', font=dict(size=16, color='#e2e8f0')),
                        xaxis_title='Communication Round',
                        yaxis_title='Test Loss',
                        xaxis=dict(**GRID_STYLE),
                        yaxis=dict(**GRID_STYLE),
                        height=420,
                        **PLOTLY_LAYOUT
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)

                # Convergence speed gauge
                st.markdown('<div class="section-header">Convergence Analysis</div>', unsafe_allow_html=True)

                gc1, gc2, gc3 = st.columns(3)

                with gc1:
                    improvement = fed_accs[-1] - fed_accs[0] if len(fed_accs) > 1 else 0
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=fed_accs[-1] * 100,
                        title={'text': "Final Federated Accuracy", 'font': {'size': 14, 'color': '#b8c1ec'}},
                        delta={'reference': fed_accs[0] * 100, 'increasing': {'color': ACCENT_1}, 'suffix': '%'},
                        number={'suffix': '%', 'font': {'size': 36, 'color': '#e2e8f0'}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickcolor': '#5a6785'},
                            'bar': {'color': FED_COLOR},
                            'bgcolor': 'rgba(22,33,62,0.6)',
                            'bordercolor': 'rgba(102,126,234,0.2)',
                            'steps': [
                                {'range': [0, 50], 'color': 'rgba(245,87,108,0.1)'},
                                {'range': [50, 80], 'color': 'rgba(247,151,30,0.1)'},
                                {'range': [80, 100], 'color': 'rgba(56,239,125,0.1)'}
                            ],
                            'threshold': {
                                'line': {'color': ACCENT_1, 'width': 3},
                                'thickness': 0.8,
                                'value': cent_acc * 100
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=280, **PLOTLY_LAYOUT)
                    st.plotly_chart(fig_gauge, use_container_width=True)

                with gc2:
                    # Per-round improvement
                    improvements = [fed_accs[i] - fed_accs[i-1] for i in range(1, len(fed_accs))]
                    fig_imp = go.Figure()
                    colors = [ACCENT_1 if v > 0 else CENT_COLOR for v in improvements]
                    fig_imp.add_trace(go.Bar(
                        x=[f'R{i}→R{i+1}' for i in range(1, len(fed_accs))],
                        y=[v*100 for v in improvements],
                        marker=dict(
                            color=colors,
                            cornerradius=6,
                            line=dict(width=0)
                        ),
                        hovertemplate='%{x}<br>Improvement: +%{y:.2f}%<extra></extra>'
                    ))
                    fig_imp.update_layout(
                        title=dict(text='Per-Round Accuracy Gain', font=dict(size=14, color='#e2e8f0')),
                        yaxis_title='Accuracy Gain (%)',
                        height=280,
                        showlegend=False,
                        **PLOTLY_LAYOUT
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)

                with gc3:
                    # Accuracy gap closing
                    gaps = [(cent_acc - a) * 100 for a in fed_accs]
                    fig_gap = go.Figure()
                    fig_gap.add_trace(go.Scatter(
                        x=rounds, y=gaps,
                        fill='tozeroy',
                        fillcolor='rgba(245,87,108,0.15)',
                        line=dict(color=CENT_COLOR, width=2),
                        mode='lines+markers',
                        marker=dict(size=6),
                        hovertemplate='Round %{x}<br>Gap: %{y:.2f}%<extra></extra>'
                    ))
                    fig_gap.update_layout(
                        title=dict(text='Accuracy Gap vs Centralized', font=dict(size=14, color='#e2e8f0')),
                        yaxis_title='Gap (%)',
                        xaxis_title='Round',
                        height=280,
                        showlegend=False,
                        **PLOTLY_LAYOUT
                    )
                    st.plotly_chart(fig_gap, use_container_width=True)

        # Model Comparison Table
        st.markdown('<div class="section-header">Detailed Model Comparison</div>', unsafe_allow_html=True)

        fed_time = fed_model.get('training_time_seconds', 0)
        cent_time = cent_model.get('training_time_seconds', 0)

        best_acc = "winner" if fed_acc > cent_acc else ""
        best_acc_c = "winner" if cent_acc > fed_acc else ""
        best_loss = "winner" if fed_model.get('final_loss', 1) < cent_model.get('final_loss', 1) else ""
        best_loss_c = "winner" if cent_model.get('final_loss', 1) < fed_model.get('final_loss', 1) else ""

        st.markdown(f"""
        <div class="glass-panel">
        <table class="comparison-table">
            <tr>
                <th>Metric</th>
                <th>Federated Learning</th>
                <th>Centralized Training</th>
                <th>Analysis</th>
            </tr>
            <tr>
                <td>Final Accuracy</td>
                <td class="fed-value {best_acc}">{fed_acc:.4f} ({fed_acc*100:.2f}%)</td>
                <td class="cent-value {best_acc_c}">{cent_acc:.4f} ({cent_acc*100:.2f}%)</td>
                <td>{retention:.1f}% retention rate</td>
            </tr>
            <tr>
                <td>Final Loss</td>
                <td class="fed-value {best_loss}">{fed_model.get('final_loss', 0):.4f}</td>
                <td class="cent-value {best_loss_c}">{cent_model.get('final_loss', 0):.4f}</td>
                <td>{'Converging well' if fed_model.get('final_loss', 1) < 0.5 else 'Room to improve'}</td>
            </tr>
            <tr>
                <td>Best Accuracy</td>
                <td class="fed-value">{fed_model.get('best_accuracy', 0):.4f}</td>
                <td class="cent-value">{cent_model.get('best_accuracy', 0):.4f}</td>
                <td>Peak performance achieved</td>
            </tr>
            <tr>
                <td>Training Time</td>
                <td class="fed-value">{fed_time/60:.1f} min</td>
                <td class="cent-value">{cent_time/60:.1f} min</td>
                <td>{fed_time/cent_time:.1f}x overhead for privacy</td>
            </tr>
            <tr>
                <td>Privacy Level</td>
                <td class="fed-value" style="color: #38ef7d !important;">{fed_model.get('privacy', 'High')}</td>
                <td class="cent-value" style="color: #f5576c !important;">{cent_model.get('privacy', 'None')}</td>
                <td>Federated keeps data local</td>
            </tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    # TAB 2: TRAINING ANALYSIS
    # ═══════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-header">Communication Overhead Analysis</div>', unsafe_allow_html=True)

        if log_text:
            rounds_data, _, _ = parse_round_data(log_text)

            if rounds_data:
                total_mb = comm.get('total_megabytes', 0)
                per_round_mb = total_mb / len(rounds_data) if rounds_data else 0
                cumulative = [per_round_mb * (i+1) for i in range(len(rounds_data))]

                co1, co2 = st.columns(2)

                with co1:
                    fig_comm = go.Figure()
                    fig_comm.add_trace(go.Scatter(
                        x=rounds_data, y=cumulative,
                        fill='tozeroy',
                        fillcolor='rgba(0,210,255,0.1)',
                        line=dict(color=ACCENT_3, width=3),
                        mode='lines+markers',
                        marker=dict(size=8, symbol='diamond', line=dict(width=2, color='white')),
                        name='Cumulative',
                        hovertemplate='Round %{x}<br>Total: %{y:.1f} MB<extra></extra>'
                    ))
                    fig_comm.update_layout(
                        title=dict(text='Cumulative Communication Cost', font=dict(size=16, color='#e2e8f0')),
                        xaxis_title='Communication Round',
                        yaxis_title='Data Transferred (MB)',
                        height=400,
                        **PLOTLY_LAYOUT
                    )
                    st.plotly_chart(fig_comm, use_container_width=True)

                with co2:
                    fig_bar = go.Figure()
                    fig_bar.add_trace(go.Bar(
                        x=rounds_data,
                        y=[per_round_mb] * len(rounds_data),
                        marker=dict(
                            color=[f'rgba(0,210,255,{0.4 + 0.06*i})' for i in range(len(rounds_data))],
                            cornerradius=8,
                            line=dict(width=1, color='rgba(0,210,255,0.3)')
                        ),
                        name='Per Round',
                        hovertemplate='Round %{x}<br>Cost: %{y:.1f} MB<extra></extra>'
                    ))
                    fig_bar.update_layout(
                        title=dict(text='Per-Round Communication Cost', font=dict(size=16, color='#e2e8f0')),
                        xaxis_title='Communication Round',
                        yaxis_title='Data Transferred (MB)',
                        height=400,
                        showlegend=False,
                        **PLOTLY_LAYOUT
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

        # Client data distribution
        st.markdown('<div class="section-header">Client Data Distribution</div>', unsafe_allow_html=True)

        dist_plot = os.path.join(results_dir, "client_data_distribution.png")
        if os.path.exists(dist_plot):
            st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
            st.image(dist_plot, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            data_config = config.get('data', {})
            st.markdown(f"""
            <div class="glass-panel" style="margin-top:12px;">
                <div style="display: flex; gap: 30px; flex-wrap: wrap;">
                    <div>
                        <span class="config-key">Distribution Type</span>
                        <div class="config-val">{data_config.get('distribution', 'N/A').upper()}</div>
                    </div>
                    <div>
                        <span class="config-key">Number of Clients</span>
                        <div class="config-val">{data_config.get('num_clients', 'N/A')}</div>
                    </div>
                    <div>
                        <span class="config-key">Shards per Client</span>
                        <div class="config-val">{data_config.get('shards_per_client', 'N/A')}</div>
                    </div>
                    <div>
                        <span class="config-key">Dataset</span>
                        <div class="config-val">{data_config.get('dataset_name', 'N/A').upper()}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Run simulation to see client data distribution.")

        # Training Log
        st.markdown('<div class="section-header">Training Log</div>', unsafe_allow_html=True)
        if log_text:
            with st.expander("View Full Training Log", expanded=False):
                st.code(log_text, language="text")

    # ═══════════════════════════════════════════
    # TAB 3: PRIVACY & SECURITY
    # ═══════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-header">Privacy & Security Analysis</div>', unsafe_allow_html=True)

        privacy_config = config.get('privacy', {})
        fed_config = config.get('federated', {})

        # Privacy status overview
        st.markdown(f"""
        <div class="glass-panel">
            <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 20px;">
                <span class="privacy-badge">Data Never Leaves Client Devices</span>
            </div>
            <p style="color: #b8c1ec; font-family: 'Inter', sans-serif; font-size: 0.95rem; line-height: 1.7; margin: 0;">
                This framework implements <strong style="color: #667eea;">Federated Averaging (FedAvg)</strong> where each client trains
                a local model on its private data and only shares <strong style="color: #38ef7d;">model weight updates</strong> with the
                central server. Raw training data is <strong style="color: #f5576c;">never transmitted</strong>, ensuring data privacy
                by design. The server aggregates updates using weighted averaging to produce an improved global model.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Privacy mechanisms
        pc1, pc2, pc3 = st.columns(3)

        with pc1:
            dp_status = "Enabled" if privacy_config.get('enable_dp', False) else "Available"
            dp_color = "#38ef7d" if privacy_config.get('enable_dp', False) else "#f7971e"
            st.markdown(f"""
            <div class="metric-card">
                <div class="icon">🔐</div>
                <div class="label">Differential Privacy</div>
                <div class="value" style="font-size: 1.6rem; background: linear-gradient(135deg, {dp_color}, {dp_color}); -webkit-background-clip: text;">{dp_status}</div>
                <div class="subtext">Noise: {privacy_config.get('dp_noise_multiplier', 'N/A')} | Clip: {privacy_config.get('dp_l2_norm_clip', 'N/A')}</div>
            </div>
            """, unsafe_allow_html=True)

        with pc2:
            sa_status = "Enabled" if privacy_config.get('secure_aggregation', False) else "Available"
            sa_color = "#38ef7d" if privacy_config.get('secure_aggregation', False) else "#f7971e"
            st.markdown(f"""
            <div class="metric-card">
                <div class="icon">🔒</div>
                <div class="label">Secure Aggregation</div>
                <div class="value" style="font-size: 1.6rem; background: linear-gradient(135deg, {sa_color}, {sa_color}); -webkit-background-clip: text;">{sa_status}</div>
                <div class="subtext">Encrypted model updates</div>
            </div>
            """, unsafe_allow_html=True)

        with pc3:
            st.markdown(f"""
            <div class="metric-card emerald">
                <div class="icon">✅</div>
                <div class="label">Data Locality</div>
                <div class="value" style="font-size: 1.6rem;">Enforced</div>
                <div class="subtext">100% local data retention</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Privacy-utility tradeoff
        st.markdown('<div class="section-header">Privacy-Utility Tradeoff</div>', unsafe_allow_html=True)

        fig_tradeoff = go.Figure()

        # Radar chart
        categories = ['Accuracy', 'Privacy', 'Speed', 'Scalability', 'Communication']

        fig_tradeoff.add_trace(go.Scatterpolar(
            r=[fed_acc * 100, 95, 30, 85, 60],
            theta=categories,
            fill='toself',
            name='Federated',
            fillcolor='rgba(102,126,234,0.15)',
            line=dict(color=FED_COLOR, width=2),
            marker=dict(size=6)
        ))
        fig_tradeoff.add_trace(go.Scatterpolar(
            r=[cent_acc * 100, 10, 90, 40, 95],
            theta=categories,
            fill='toself',
            name='Centralized',
            fillcolor='rgba(245,87,108,0.1)',
            line=dict(color=CENT_COLOR, width=2),
            marker=dict(size=6)
        ))

        fig_tradeoff.update_layout(
            polar=dict(
                bgcolor='rgba(13,17,23,0.8)',
                radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(102,126,234,0.15)', tickfont=dict(size=10, color='#5a6785')),
                angularaxis=dict(gridcolor='rgba(102,126,234,0.15)', tickfont=dict(size=12, color='#b8c1ec'))
            ),
            showlegend=True,
            height=450,
            **PLOTLY_LAYOUT
        )
        st.plotly_chart(fig_tradeoff, use_container_width=True)

        # Privacy details
        st.markdown('<div class="section-header">Privacy Configuration Details</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-panel">
        <table class="comparison-table">
            <tr><th>Parameter</th><th>Value</th><th>Description</th></tr>
            <tr>
                <td>Aggregation Method</td>
                <td class="fed-value">{fed_config.get('aggregation_method', 'fedavg').upper()}</td>
                <td>Algorithm used for combining client updates</td>
            </tr>
            <tr>
                <td>Differential Privacy</td>
                <td class="{'fed-value' if privacy_config.get('enable_dp') else 'cent-value'}">{'Enabled' if privacy_config.get('enable_dp') else 'Disabled'}</td>
                <td>Adds calibrated noise for formal privacy guarantees</td>
            </tr>
            <tr>
                <td>DP Noise Multiplier</td>
                <td class="fed-value">{privacy_config.get('dp_noise_multiplier', 'N/A')}</td>
                <td>Controls noise level (higher = more private, less accurate)</td>
            </tr>
            <tr>
                <td>L2 Norm Clip</td>
                <td class="fed-value">{privacy_config.get('dp_l2_norm_clip', 'N/A')}</td>
                <td>Bounds individual client contribution</td>
            </tr>
            <tr>
                <td>Delta (DP)</td>
                <td class="fed-value">{privacy_config.get('dp_delta', 'N/A')}</td>
                <td>Privacy budget parameter</td>
            </tr>
            <tr>
                <td>Secure Aggregation</td>
                <td class="{'fed-value' if privacy_config.get('secure_aggregation') else 'cent-value'}">{'Enabled' if privacy_config.get('secure_aggregation') else 'Disabled'}</td>
                <td>Cryptographic aggregation of model updates</td>
            </tr>
            <tr>
                <td>FedProx Mu</td>
                <td class="fed-value">{fed_config.get('fedprox_mu', 'N/A')}</td>
                <td>Proximal term for handling heterogeneity</td>
            </tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    # TAB 4: CONFIGURATION
    # ═══════════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-header">Training Configuration</div>', unsafe_allow_html=True)

        data_cfg = config.get('data', {})
        model_cfg = config.get('model', {})
        fed_cfg = config.get('federated', {})
        train_cfg = config.get('training', {})

        cfg1, cfg2 = st.columns(2)

        with cfg1:
            st.markdown("""<div class="glass-panel"><h3 style="color: #e2e8f0; font-family: 'Inter', sans-serif; margin-top:0;">📦 Data Configuration</h3>""", unsafe_allow_html=True)
            for key, val in data_cfg.items():
                display_val = str(val).upper() if isinstance(val, str) else str(val)
                st.markdown(f"""<div class="config-item"><div class="config-key">{key.replace('_', ' ')}</div><div class="config-val">{display_val}</div></div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("""<div class="glass-panel"><h3 style="color: #e2e8f0; font-family: 'Inter', sans-serif; margin-top:0;">🧠 Model Architecture</h3>""", unsafe_allow_html=True)
            for key, val in model_cfg.items():
                display_val = str(val).upper() if isinstance(val, str) else str(val)
                st.markdown(f"""<div class="config-item"><div class="config-key">{key.replace('_', ' ')}</div><div class="config-val">{display_val}</div></div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with cfg2:
            st.markdown("""<div class="glass-panel"><h3 style="color: #e2e8f0; font-family: 'Inter', sans-serif; margin-top:0;">🔄 Federated Learning</h3>""", unsafe_allow_html=True)
            for key, val in fed_cfg.items():
                display_val = str(val).upper() if isinstance(val, str) else str(val)
                st.markdown(f"""<div class="config-item"><div class="config-key">{key.replace('_', ' ')}</div><div class="config-val">{display_val}</div></div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("""<div class="glass-panel"><h3 style="color: #e2e8f0; font-family: 'Inter', sans-serif; margin-top:0;">⚙️ Training Settings</h3>""", unsafe_allow_html=True)
            for key, val in train_cfg.items():
                display_val = str(val)
                st.markdown(f"""<div class="config-item"><div class="config-key">{key.replace('_', ' ')}</div><div class="config-val">{display_val}</div></div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Download section
        st.markdown('<div class="section-header">Export Results</div>', unsafe_allow_html=True)

        dl1, dl2, dl3 = st.columns(3)

        with dl1:
            report_json = json.dumps(report, indent=2)
            st.download_button(
                label="Download Full Report (JSON)",
                data=report_json,
                file_name="federated_learning_report.json",
                mime="application/json"
            )

        with dl2:
            csv_file = os.path.join(results_dir, "performance_report.csv")
            if os.path.exists(csv_file):
                with open(csv_file, 'r') as f:
                    csv_data = f.read()
                st.download_button(
                    label="Download Summary (CSV)",
                    data=csv_data,
                    file_name="performance_summary.csv",
                    mime="text/csv"
                )

        with dl3:
            if log_text:
                st.download_button(
                    label="Download Training Log",
                    data=log_text,
                    file_name="training_log.txt",
                    mime="text/plain"
                )

    # ═══════════════════════════════════════════
    # TAB 5: RUN SIMULATION
    # ═══════════════════════════════════════════
    with tab5:
        st.markdown('<div class="section-header">Run New Simulation</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-panel">
            <p style="color: #b8c1ec; font-family: 'Inter', sans-serif; margin: 0; line-height: 1.7;">
                Configure your federated learning experiment using the sidebar parameters, then click the button below
                to start training. The simulation will train both a <strong style="color: #667eea;">federated model</strong>
                and a <strong style="color: #f5576c;">centralized baseline</strong> for comparison.
                Results will automatically update across all dashboard tabs.
            </p>
        </div>
        """, unsafe_allow_html=True)

        sim_c1, sim_c2 = st.columns([2, 1])

        with sim_c1:
            st.markdown(f"""
            <div class="glass-panel">
                <h4 style="color: #e2e8f0; font-family: 'Inter', sans-serif; margin-top:0;">Current Parameters</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                    <div class="config-item"><div class="config-key">Dataset</div><div class="config-val">{st.session_state.get('sim_dataset', 'MNIST').upper()}</div></div>
                    <div class="config-item"><div class="config-key">Clients</div><div class="config-val">{st.session_state.get('sim_clients', 5)}</div></div>
                    <div class="config-item"><div class="config-key">Rounds</div><div class="config-val">{st.session_state.get('sim_rounds', 5)}</div></div>
                    <div class="config-item"><div class="config-key">Distribution</div><div class="config-val">{st.session_state.get('sim_dist', 'NON-IID').upper()}</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with sim_c2:
            st.markdown("""
            <div class="glass-panel" style="text-align: center; padding: 30px;">
                <div style="font-size: 3rem; margin-bottom: 10px;">🚀</div>
                <p style="color: #b8c1ec; font-family: 'Inter', sans-serif; font-size: 0.85rem;">
                    Adjust parameters in the sidebar then click Run below.
                </p>
            </div>
            """, unsafe_allow_html=True)

else:
    # No results yet
    st.markdown("""
    <div class="glass-panel" style="text-align: center; padding: 60px 40px;">
        <div style="font-size: 4rem; margin-bottom: 20px;">🚀</div>
        <h2 style="color: #e2e8f0; font-family: 'Inter', sans-serif; margin-bottom: 10px;">No Results Yet</h2>
        <p style="color: #8892b0; font-family: 'Inter', sans-serif; font-size: 1.05rem; max-width: 500px; margin: 0 auto;">
            Configure your experiment parameters in the sidebar and click <strong style="color: #667eea;">Run Simulation</strong>
            to train your first federated learning model.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─── Sidebar ───
st.sidebar.markdown("""
<div style="text-align: center; padding: 20px 0 10px 0;">
    <div style="font-size: 1.8rem; font-weight: 800; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-family: 'Inter', sans-serif;">FL Framework</div>
    <div style="font-size: 0.75rem; color: #5a6785; font-family: 'Inter', sans-serif; letter-spacing: 2px; text-transform: uppercase; margin-top: 4px;">Privacy-Preserving AI</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Simulation Parameters")

dataset = st.sidebar.selectbox("Dataset", ["mnist", "fashion_mnist", "cifar10"], key="sim_dataset")
num_clients = st.sidebar.slider("Number of Clients", min_value=2, max_value=10, value=5, key="sim_clients")
clients_per_round = st.sidebar.slider("Clients per Round", min_value=1, max_value=num_clients, value=min(5, num_clients))
training_rounds = st.sidebar.slider("Communication Rounds", min_value=1, max_value=20, value=10, key="sim_rounds")
local_epochs = st.sidebar.slider("Local Epochs per Round", min_value=1, max_value=5, value=2)
distribution = st.sidebar.selectbox("Data Distribution", ["non-iid", "iid", "dirichlet"], key="sim_dist")

st.sidebar.markdown("---")

if st.sidebar.button("Run Simulation", type="primary"):
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
            st.success("Simulation Completed Successfully! Refresh the page to see updated results.")
            st.text_area("Output", stdout[-3000:] if len(stdout) > 3000 else stdout, height=250)
        else:
            st.error("Simulation Failed!")
            st.code(stderr[-2000:] if len(stderr) > 2000 else stderr)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="background: rgba(102,126,234,0.08); border: 1px solid rgba(102,126,234,0.2); border-radius: 10px; padding: 16px; font-family: 'Inter', sans-serif;">
    <div style="font-weight: 600; color: #e2e8f0; margin-bottom: 8px; font-size: 0.85rem;">How to use</div>
    <div style="color: #8892b0; font-size: 0.8rem; line-height: 1.6;">
        1. Select dataset and parameters<br>
        2. Click <strong style="color: #667eea;">Run Simulation</strong><br>
        3. View results across all tabs<br>
        4. Export reports for documentation
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Footer ───
st.markdown("""
<div class="footer">
    Privacy-Preserving AI Framework &bull; Built with TensorFlow &bull; Federated Learning
    <br>
    <span style="font-size: 0.7rem;">Secure Model Training Without Compromising Data Privacy</span>
</div>
""", unsafe_allow_html=True)
