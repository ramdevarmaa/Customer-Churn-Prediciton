import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import io
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model_utils import load_model, predict_batch, generate_synthetic_data, FEATURES


def show():
    st.markdown("##  Batch Analysis")
    st.markdown("<p style='color:#8b9ab8'>Upload a CSV with customer data to predict churn for multiple customers at once.</p>", unsafe_allow_html=True)

    with st.spinner("Loading model..."):
        model, _ = load_model()

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("#### Upload Customer Data")
        st.markdown(f"""
        <div style='background:#111827; border:1px solid #1e3a5f; border-radius:10px; padding:16px; margin-bottom:16px;'>
            <div style='color:#4fd1c5; font-weight:600; margin-bottom:8px; font-family:Syne,sans-serif;'>Required CSV Columns</div>
            <div style='color:#8b9ab8; font-size:0.82rem; line-height:1.8;'>
                {' · '.join(FEATURES)}
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("Drop your CSV here", type=["csv"])

        st.markdown("**Or use sample data:**")
        if st.button("📋 Generate & Analyze Sample Data"):
            df_sample = generate_synthetic_data(n=200)
            df_sample = df_sample.drop(columns=["Churn"])
            st.session_state["batch_df"] = df_sample

        if uploaded is not None:
            df_up = pd.read_csv(uploaded)
            st.session_state["batch_df"] = df_up

    with col2:
        st.markdown("#### Quick Stats")

        if "batch_result" in st.session_state:
            result = st.session_state["batch_result"]
            total = len(result)
            churned = result["ChurnPrediction"].sum()
            retained = total - churned
            churn_rate = churned / total * 100

            st.markdown(f"""
            <div class='metric-card' style='margin-bottom:12px;'>
                <div class='value'>{total}</div>
                <div class='label'>Total Customers</div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            c1.markdown(f"""
            <div class='metric-card'>
                <div class='value' style='color:#f87171;'>{churned}</div>
                <div class='label'>At Risk</div>
            </div>""", unsafe_allow_html=True)
            c2.markdown(f"""
            <div class='metric-card'>
                <div class='value'>{retained}</div>
                <div class='label'>Retained</div>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class='metric-card' style='margin-top:12px;'>
                <div class='value' style='color:#fb923c;'>{churn_rate:.1f}%</div>
                <div class='label'>Predicted Churn Rate</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#111827; border:1px dashed #1e3a5f; border-radius:10px;
                        padding: 48px 20px; text-align:center; color:#374151;'>
                <div style='font-size:2rem; margin-bottom:8px;'>📊</div>
                <div style='color:#4b5563; font-size:0.85rem;'>Run an analysis to see stats</div>
            </div>
            """, unsafe_allow_html=True)

    # Process batch
    if "batch_df" in st.session_state and "batch_result" not in st.session_state:
        df = st.session_state["batch_df"]
        with st.spinner("Predicting churn for all customers..."):
            result, error = predict_batch(model, df)
        if error:
            st.error(f"Error: {error}")
        else:
            st.session_state["batch_result"] = result
            st.rerun()

    if "batch_result" in st.session_state:
        result = st.session_state["batch_result"]

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Risk Distribution")
            risk_counts = result["Risk"].value_counts()
            colors = {"High": "#f87171", "Medium": "#fb923c", "Low": "#4fd1c5"}
            fig = go.Figure(go.Bar(
                x=risk_counts.index.tolist(),
                y=risk_counts.values.tolist(),
                marker=dict(color=[colors.get(k, "#6366f1") for k in risk_counts.index]),
                text=risk_counts.values.tolist(),
                textposition="outside",
                textfont=dict(color="#e8eaf0"),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,22,40,0.6)",
                yaxis=dict(gridcolor="#1e2d4a", color="#8b9ab8"),
                xaxis=dict(color="#8b9ab8"),
                margin=dict(t=20, b=20, l=20, r=20),
                height=260,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown("#### Probability Histogram")
            fig2 = go.Figure(go.Histogram(
                x=result["ChurnProbability"],
                xbins=dict(size=0.05),
                marker=dict(color="#6366f1", line=dict(color="#818cf8", width=0.5)),
            ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,22,40,0.6)",
                xaxis=dict(title="Churn Probability", gridcolor="#1e2d4a", color="#8b9ab8"),
                yaxis=dict(gridcolor="#1e2d4a", color="#8b9ab8"),
                margin=dict(t=20, b=20, l=20, r=20),
                height=260,
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Data Table
        st.markdown("#### 📋 Results Table")
        display_cols = ["ChurnProbability", "ChurnPrediction", "Risk"] + [c for c in FEATURES if c in result.columns]
        st.dataframe(
            result[display_cols].style
            .background_gradient(subset=["ChurnProbability"], cmap="RdYlGn_r")
            .format({"ChurnProbability": "{:.1%}"}),
            use_container_width=True, height=350
        )

        # Download
        csv_buffer = io.StringIO()
        result.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇️ Download Results CSV",
            data=csv_buffer.getvalue(),
            file_name="churn_predictions.csv",
            mime="text/csv"
        )

        if st.button("🔄 Clear Results"):
            del st.session_state["batch_result"]
            del st.session_state["batch_df"]
            st.rerun()
