import streamlit as st

st.set_page_config(
    page_title="ChurnSight | Customer Churn Prediction",
    page_icon="financial-profit.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #0a0f1e;
        color: #e8eaf0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0d1425 !important;
        border-right: 1px solid #1e2d4a;
    }

    /* Headings */
    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
        color: #ffffff;
    }

    /* Sidebar Buttons */
    .stButton > button {
        background: transparent !important;
        color: #8b9ab8 !important;
        border: 1px solid #1e3a5f !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        padding: 0.6rem 1rem !important;
        margin-bottom: 6px !important;
        text-align: left !important;
    }

    .stButton > button:hover {
        background: #1e3a5f !important;
        color: #4fd1c5 !important;
    }

    /* Active page highlight */
    .active-btn {
        background: #1e3a5f !important;
        color: #4fd1c5 !important;
        border: 1px solid #4fd1c5 !important;
    }

    /* Hide default Streamlit elements */
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 2rem; }

</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "Overview"

with st.sidebar:
    st.markdown("""
    <div style='padding: 20px 0 32px 0;'>
        <div style='font-family: Syne, sans-serif; font-size: 1.5rem; font-weight: 800;
                    background: linear-gradient(90deg, #4fd1c5, #818cf8);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            ChurnSight
        </div>
        <div style='color: #8b9ab8; font-size: 0.78rem; margin-top: 4px;'>
            Customer Churn Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    def nav_button(label, page_name):
        if st.button(label, use_container_width=True):
            st.session_state.page = page_name

    nav_button(" Overview", "Overview")
    nav_button(" Predict Churn", "Predict")
    nav_button(" Model Insights", "Insights")
    nav_button(" Batch Analysis", "Batch")

    st.markdown("---")
    st.markdown(
        "<div style='color:#8b9ab8; font-size:0.75rem;'>Powered by XGBoost</div>",
        unsafe_allow_html=True
    )

page = st.session_state.page

if page == "Overview":
    from overview import show
    show()

elif page == "Predict":
    from predict import show
    show()

elif page == "Insights":
    from insights import show
    show()

elif page == "Batch":
    from batch import show
    show()
