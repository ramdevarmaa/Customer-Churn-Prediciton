import streamlit as st
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model_utils import load_model, predict_single


def show():
    st.markdown("##  Predict Customer Churn")
    st.markdown("<p style='color:#8b9ab8'>Fill in customer details below to get an instant churn prediction.</p>", unsafe_allow_html=True)

    with st.spinner("Loading model..."):
        model, _ = load_model()

    col_left, col_right = st.columns([1.6, 1])

    with col_left:
        st.markdown("#### Customer Profile")

        c1, c2 = st.columns(2)
        with c1:
            tenure = st.slider("Tenure (months)", 0, 61, 12)
            satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
            order_count = st.slider("Order Count", 1, 16, 3)
            hour_app = st.slider("Hours on App / Day", 1, 5, 2)
            num_devices = st.slider("Devices Registered", 1, 6, 3)
            complain = st.selectbox("Has Filed Complaint?", [0, 1], format_func=lambda x: "Yes" if x else "No")

        with c2:
            cashback = st.slider("Cashback Amount ($)", 50, 400, 160)
            city_tier = st.selectbox("City Tier", [1, 2, 3])
            warehouse_dist = st.slider("Warehouse to Home (km)", 5, 60, 15)
            days_last = st.slider("Days Since Last Order", 0, 46, 5)
            order_hike = st.slider("Order Hike vs Last Year (%)", 5, 40, 15)
            coupons = st.slider("Coupons Used", 0, 10, 2)

        c3, c4 = st.columns(2)
        with c3:
            gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            num_addr = st.slider("Number of Addresses", 1, 10, 3)
        with c4:
            marital = st.selectbox("Marital Status", [0, 1, 2],
                                   format_func=lambda x: ["Single", "Married", "Divorced"][x])

        predict_clicked = st.button(" Predict Churn Risk")

    with col_right:
        st.markdown("#### Prediction Result")

        if predict_clicked or "last_prediction" in st.session_state:
            input_data = {
                "Tenure": tenure,
                "CashbackAmount": cashback,
                "CityTier": city_tier,
                "WarehouseToHome": warehouse_dist,
                "OrderAmountHikeFromlastYear": order_hike,
                "DaySinceLastOrder": days_last,
                "SatisfactionScore": satisfaction,
                "NumberOfAddress": num_addr,
                "NumberOfDeviceRegistered": num_devices,
                "Complain": complain,
                "OrderCount": order_count,
                "HourSpendOnApp": hour_app,
                "MaritalStatus": marital,
                "CouponUsed": coupons,
                "Gender": gender,
            }

            label, prob = predict_single(model, input_data)
            st.session_state["last_prediction"] = (label, prob)

            pct = prob * 100
            color = "#f87171" if label == 1 else "#4fd1c5"
            css_class = "result-churn" if label == 1 else "result-safe"
            icon = "wrong.png" if label == 1 else "like.png"
            # Emoji shown next to the verdict (must be defined)
            emoji = "😟" if label == 1 else "🙂"
            st.image(icon, width=80)
            verdict = "High Churn Risk" if label == 1 else "Likely to Stay"

            st.markdown(f"""
            <div class='{css_class}'>
                <div style='font-size: 2.5rem;'>{emoji}</div>
                <div class='result-label' style='color:{color}'>{verdict}</div>
                <div style='color:#8b9ab8; margin-top: 8px; font-size: 0.85rem;'>
                    Churn Probability
                </div>
                <div style='font-family:Syne,sans-serif; font-size: 3rem; font-weight:800; color:{color}'>
                    {pct:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number={"suffix": "%", "font": {"color": color, "size": 28, "family": "Syne"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8b9ab8",
                             "tickfont": {"color": "#8b9ab8"}},
                    "bar": {"color": color},
                    "bgcolor": "#0d1425",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30], "color": "#0f2d1e"},
                        {"range": [30, 60], "color": "#1e2d10"},
                        {"range": [60, 100], "color": "#2d1010"},
                    ],
                    "threshold": {
                        "line": {"color": "#ffffff", "width": 2},
                        "thickness": 0.75,
                        "value": 50
                    }
                }
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8b9ab8"),
                margin=dict(t=20, b=20, l=30, r=30),
                height=220,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Recommendation
            st.markdown("#### 💡 Recommended Action")
            if pct >= 70:
                st.error("**Immediate retention action needed.** Offer a loyalty discount or personal outreach. Risk of losing this customer is very high.")
            elif pct >= 45:
                st.warning("**Monitor closely.** Consider sending a personalized re-engagement campaign or cashback offer.")
            else:
                st.success("**Customer appears healthy.** Continue standard engagement. Consider loyalty rewards to maintain satisfaction.")
        else:
            st.markdown("""
            <div style='background:#111827; border:1px dashed #1e3a5f; border-radius:12px;
                        padding: 48px 24px; text-align:center; color:#374151;'>
                <div style='font-size:2.5rem; margin-bottom:12px;'><img src="decision-making.png" width="60"></div>
                <div style='color:#4b5563'>Fill in customer details and click<br><strong style='color:#4fd1c5'>Predict Churn Risk</strong></div>
            </div>
            """, unsafe_allow_html=True)
