import streamlit as st
import pandas as pd

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Online Shoppers | Home",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# DATA
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("online_shoppers_intention.csv")


df = load_data()

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 3rem;
    font-weight: 800;
    color: #14213D;
    margin-bottom: 0.2rem;
}

.subtitle {
    font-size: 1.1rem;
    color: #64748B;
    margin-bottom: 2rem;
}

.kpi-card {
    background: white;
    border-radius: 16px;
    padding: 22px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 5px 15px rgba(15, 23, 42, 0.06);
    text-align: center;
    transition: 0.2s;
}

.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 25px rgba(15, 23, 42, 0.10);
}

.kpi-title {
    color: #64748B;
    font-size: 0.9rem;
    font-weight: 600;
}

.kpi-value {
    color: #14213D;
    font-size: 2rem;
    font-weight: 800;
    margin-top: 5px;
}

.section {
    background: #F8FAFC;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HERO
# =========================================================

st.markdown(
    '<div class="main-title">🛍️ Online Shoppers Purchase Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    An interactive Machine Learning application for predicting whether
    an online shopping session will result in a purchase.
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# =========================================================
# DATA OVERVIEW
# =========================================================

st.subheader("📊 Data Overview")

total_records = len(df)
total_features = df.shape[1] - 1
purchase_rate = df["Revenue"].mean() * 100
no_purchase_rate = 100 - purchase_rate

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">TOTAL RECORDS</div>
            <div class="kpi-value">{total_records:,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">FEATURES</div>
            <div class="kpi-value">{total_features}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">PURCHASE RATE</div>
            <div class="kpi-value">{purchase_rate:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">NO PURCHASE</div>
            <div class="kpi-value">{no_purchase_rate:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# PROJECT SUMMARY
# =========================================================

st.markdown("### 🎯 Project Objective")

st.info(
    """
    The objective is to predict whether an online shopping session
    will generate revenue based on the visitor's browsing behavior,
    engagement metrics, traffic source, visitor type, and session context.
    """
)

# =========================================================
# COLUMN UNDERSTANDING
# =========================================================

st.markdown("### 📚 Understand the Columns")

if st.button(
    "🔎 Understand the Columns",
    type="primary",
    use_container_width=True
):

    column_descriptions = {
        "Administrative":
            "Number of administrative pages visited.",

        "Administrative_Duration":
            "Time spent on administrative pages.",

        "Informational":
            "Number of informational pages visited.",

        "Informational_Duration":
            "Time spent on informational pages.",

        "ProductRelated":
            "Number of product-related pages visited.",

        "ProductRelated_Duration":
            "Time spent on product-related pages.",

        "BounceRates":
            "Bounce rate of the session.",

        "ExitRates":
            "Exit rate of the session.",

        "PageValues":
            "Average page value before a transaction.",

        "SpecialDay":
            "Closeness to a special day or holiday.",

        "Month":
            "Month of the session.",

        "OperatingSystems":
            "Operating system identifier.",

        "Browser":
            "Browser identifier.",

        "Region":
            "Geographic region identifier.",

        "TrafficType":
            "Traffic source identifier.",

        "VisitorType":
            "Visitor category.",

        "Weekend":
            "Whether the session occurred on a weekend.",

        "Revenue":
            "Target: whether the session generated revenue."
    }

    description_df = pd.DataFrame({
        "Column": list(column_descriptions.keys()),
        "Description": list(column_descriptions.values())
    })

    st.dataframe(
        description_df,
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Online Shoppers Purchase Prediction • Machine Learning Deployment"
)