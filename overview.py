import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model_utils import load_model, generate_synthetic_data

def show():
    st.markdown("""
    <div class='hero'>
        <h1>Customer Churn Intelligence</h1>
        <p>Predict which customers are at risk of leaving — before they do. 
        Powered by XGBoost with real-time analysis and actionable insights.</p>
    </div>
    """, unsafe_allow_html=True)

    # Load model and metrics
    with st.spinner("Loading model..."):
        model, metrics = load_model()

    # KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("Accuracy", f"{metrics['accuracy']*100:.1f}%"),
        ("F1 Score", f"{metrics['f1']:.3f}"),
        ("Precision", f"{metrics['precision']:.3f}"),
        ("Recall", f"{metrics['recall']:.3f}"),
        ("ROC AUC", f"{metrics['roc_auc']:.3f}"),
    ]
    for col, (label, value) in zip([c1, c2, c3, c4, c5], kpis):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='value'>{value}</div>
            <div class='label'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Churn Distribution in Dataset")
        df = generate_synthetic_data()
        churn_counts = df["Churn"].value_counts()
        fig = go.Figure(go.Pie(
            labels=["Retained", "Churned"],
            values=[churn_counts[0], churn_counts[1]],
            hole=0.55,
            marker=dict(colors=["#4fd1c5", "#f87171"]),
            textfont=dict(family="DM Sans", size=13, color="white"),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#8b9ab8"),
            legend=dict(font=dict(color="#e8eaf0")),
            margin=dict(t=20, b=20, l=20, r=20),
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### ROC Curve")
        fpr, tpr, _ = metrics["roc_curve"]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"AUC = {metrics['roc_auc']:.3f}",
            line=dict(color="#4fd1c5", width=2.5)
        ))
        fig2.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random",
            line=dict(color="#374151", width=1.5, dash="dash")
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,22,40,0.6)",
            font=dict(color="#8b9ab8"),
            xaxis=dict(title="False Positive Rate", gridcolor="#1e2d4a", color="#8b9ab8"),
            yaxis=dict(title="True Positive Rate", gridcolor="#1e2d4a", color="#8b9ab8"),
            legend=dict(font=dict(color="#e8eaf0"), bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=20, b=20, l=20, r=20),
            height=280,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Feature Importance
    st.markdown("#### Top Feature Importances")
    fi = metrics["feature_importance"]
    fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
    colors = ["#4fd1c5" if i < 3 else "#6366f1" if i < 7 else "#374151"
              for i in range(len(fi_sorted))]

    fig3 = go.Figure(go.Bar(
        y=list(fi_sorted.keys()),
        x=list(fi_sorted.values()),
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.3f}" for v in fi_sorted.values()],
        textposition="outside",
        textfont=dict(color="#8b9ab8", size=11),
    ))
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,22,40,0.6)",
        font=dict(color="#8b9ab8"),
        xaxis=dict(title="Importance Score", gridcolor="#1e2d4a", color="#8b9ab8"),
        yaxis=dict(autorange="reversed", color="#e8eaf0"),
        margin=dict(t=10, b=20, l=20, r=60),
        height=420,
    )
    st.plotly_chart(fig3, use_container_width=True)
