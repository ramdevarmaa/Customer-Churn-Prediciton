import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model_utils import load_model, generate_synthetic_data


def show():
    st.markdown("##  Model Insights")
    st.markdown("<p style='color:#8b9ab8'>Deep dive into model performance, confusion matrix, and feature analysis.</p>", unsafe_allow_html=True)

    with st.spinner("Loading model..."):
        model, metrics = load_model()

    # Confusion matrix
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Confusion Matrix")
        cm = np.array(metrics["cm"])
        labels = ["Retained", "Churned"]
        fig = go.Figure(go.Heatmap(
            z=cm,
            x=labels, y=labels,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 20, "color": "white", "family": "Syne"},
            colorscale=[[0, "#0d1425"], [1, "#4fd1c5"]],
            showscale=False,
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Predicted", color="#8b9ab8", tickfont=dict(color="#e8eaf0")),
            yaxis=dict(title="Actual", color="#8b9ab8", tickfont=dict(color="#e8eaf0")),
            margin=dict(t=20, b=20, l=20, r=20),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Metrics Summary")
        metric_names = ["Accuracy", "F1 Score", "Precision", "Recall", "ROC AUC"]
        metric_vals = [
            metrics["accuracy"], metrics["f1"], metrics["precision"],
            metrics["recall"], metrics["roc_auc"]
        ]
        colors = ["#4fd1c5", "#818cf8", "#f472b6", "#fb923c", "#a3e635"]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=metric_names, y=metric_vals,
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.3f}" for v in metric_vals],
            textposition="outside",
            textfont=dict(color="#e8eaf0", size=13),
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,22,40,0.6)",
            yaxis=dict(range=[0, 1.15], gridcolor="#1e2d4a", color="#8b9ab8"),
            xaxis=dict(color="#8b9ab8"),
            margin=dict(t=30, b=20, l=20, r=20),
            height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Distribution of predicted probabilities
    st.markdown("#### Predicted Probability Distribution")
    y_prob = metrics["y_prob"]
    y_test = metrics["y_test"]

    prob_retained = y_prob[y_test == 0]
    prob_churned = y_prob[y_test == 1]

    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(
        x=prob_retained, name="Retained",
        marker_color="#4fd1c5", opacity=0.7,
        xbins=dict(size=0.05),
    ))
    fig3.add_trace(go.Histogram(
        x=prob_churned, name="Churned",
        marker_color="#f87171", opacity=0.7,
        xbins=dict(size=0.05),
    ))
    fig3.update_layout(
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,22,40,0.6)",
        xaxis=dict(title="Predicted Churn Probability", gridcolor="#1e2d4a", color="#8b9ab8"),
        yaxis=dict(title="Count", gridcolor="#1e2d4a", color="#8b9ab8"),
        legend=dict(font=dict(color="#e8eaf0"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=20, l=20, r=20),
        height=300,
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Feature analysis
    st.markdown("#### Churn vs Retained — Feature Comparison")
    df = generate_synthetic_data()
    num_features = ["Tenure", "CashbackAmount", "DaySinceLastOrder", "SatisfactionScore", "OrderCount"]

    selected = st.selectbox("Select Feature", num_features,
                            format_func=lambda x: x.replace("_", " "))

    fig4 = go.Figure()
    for label, color, name in [(0, "#4fd1c5", "Retained"), (1, "#f87171", "Churned")]:
        fig4.add_trace(go.Violin(
            y=df[df["Churn"] == label][selected],
            name=name,
            box_visible=True,
            meanline_visible=True,
            fillcolor=color,
            opacity=0.6,
            line_color=color,
        ))
    fig4.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,22,40,0.6)",
        yaxis=dict(gridcolor="#1e2d4a", color="#8b9ab8", title=selected),
        xaxis=dict(color="#8b9ab8"),
        legend=dict(font=dict(color="#e8eaf0"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=10, b=20, l=20, r=20),
        height=320,
    )
    st.plotly_chart(fig4, use_container_width=True)
