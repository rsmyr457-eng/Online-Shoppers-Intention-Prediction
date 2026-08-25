import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Live Purchase Prediction",
    page_icon="🔮",
    layout="wide"
)

# =========================================================
# DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "online_shoppers_intention.csv"
    )

    # Same feature engineering as notebook

    df["page_value_category"] = pd.cut(
        df["PageValues"],
        bins=[-1, 0, 10, 50, 400],
        labels=[
            "Zero (0)",
            "Low (1-10)",
            "Medium (10-50)",
            "High (>50)"
        ]
    )

    df["product_minutes"] = (
        df["ProductRelated_Duration"] / 60
    )

    df["product_category"] = pd.cut(
        df["product_minutes"],
        bins=[-1, 5, 20, 60, 2000],
        labels=[
            "Very Low (<5m)",
            "Low (5-20m)",
            "Medium (20-60m)",
            "High (>60m)"
        ]
    )

    top_channels = (
        df["TrafficType"]
        .value_counts()
        .head(6)
        .index
    )

    df["Traffic_Grouped"] = df["TrafficType"].apply(
        lambda x: str(x)
        if x in top_channels
        else "Other"
    )

    return df


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():

    search = joblib.load(
        "best_model.pkl"
    )

    # best_model.pkl contains RandomizedSearchCV
    return search.best_estimator_


df = load_data()

try:

    model = load_model()

except Exception as e:

    st.error(
        f"""
        Model loading failed.

        Make sure your environment uses the same
        scikit-learn version used to save best_model.pkl.

        Error:
        {e}
        """
    )

    st.stop()

# =========================================================
# HERO
# =========================================================

st.markdown("""
<style>

.hero {
    background: linear-gradient(
        135deg,
        #14213D,
        #243B64
    );

    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    margin: 0;
    font-size: 2.6rem;
}

.hero p {
    color: #D9E2F2;
    margin-top: 8px;
}

.result-box {
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    color: white;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">

<h1>🔮 Live Purchase Predictor</h1>

<p>
Enter the browsing-session information and let the trained
XGBoost model predict the purchase probability.
</p>

</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🔮 Live Predictor")

    st.info(
        """
        The inputs are transformed using the same
        feature engineering used during model training.
        """
    )

    st.markdown("### Model")

    st.write("XGBoost Classifier")

    st.write("Best Parameters:")

    st.code(
        """
max_depth = 5
n_estimators = 50
        """
    )

# =========================================================
# FORM
# =========================================================

with st.form("prediction_form"):

    # ------------------------------------------------------
    # PAGE ENGAGEMENT
    # ------------------------------------------------------

    st.markdown("### 📄 Page Engagement")

    c1, c2, c3 = st.columns(3)

    with c1:

        administrative = st.number_input(
            "Administrative Pages",
            min_value=0,
            max_value=100,
            value=2
        )

        admin_duration = st.number_input(
            "Administrative Duration (sec)",
            min_value=0.0,
            value=30.0
        )

    with c2:

        informational = st.number_input(
            "Informational Pages",
            min_value=0,
            max_value=100,
            value=0
        )

        info_duration = st.number_input(
            "Informational Duration (sec)",
            min_value=0.0,
            value=0.0
        )

    with c3:

        product_related = st.number_input(
            "Product Related Pages",
            min_value=0,
            max_value=1000,
            value=20
        )

        product_duration = st.number_input(
            "Product Related Duration (sec)",
            min_value=0.0,
            value=600.0
        )

    # ------------------------------------------------------
    # ENGAGEMENT QUALITY
    # ------------------------------------------------------

    st.markdown("### 📊 Engagement Quality")

    c4, c5, c6 = st.columns(3)

    with c4:

        bounce_rate = st.slider(
            "Bounce Rate",
            0.0,
            0.2,
            0.02,
            0.001
        )

    with c5:

        exit_rate = st.slider(
            "Exit Rate",
            0.0,
            0.2,
            0.04,
            0.001
        )

    with c6:

        page_values = st.number_input(
            "Page Values",
            min_value=0.0,
            value=0.0
        )

    # ------------------------------------------------------
    # SESSION CONTEXT
    # ------------------------------------------------------

    st.markdown("### 🗓️ Session Context")

    c7, c8, c9 = st.columns(3)

    with c7:

        month = st.selectbox(
            "Month",
            sorted(df["Month"].unique())
        )

        special_day = st.slider(
            "Special Day",
            0.0,
            1.0,
            0.0,
            0.1
        )

    with c8:

        visitor_type = st.selectbox(
            "Visitor Type",
            sorted(df["VisitorType"].unique())
        )

        weekend = st.selectbox(
            "Weekend",
            [False, True]
        )

    with c9:

        operating_system = st.selectbox(
            "Operating System",
            sorted(df["OperatingSystems"].unique())
        )

        browser = st.selectbox(
            "Browser",
            sorted(df["Browser"].unique())
        )

    # ------------------------------------------------------
    # LOCATION + TRAFFIC
    # ------------------------------------------------------

    st.markdown("### 🚦 Traffic & Location")

    c10, c11, c12 = st.columns(3)

    with c10:

        region = st.selectbox(
            "Region",
            sorted(df["Region"].unique())
        )

    with c11:

        traffic_type = st.selectbox(
            "Traffic Type",
            sorted(df["TrafficType"].unique())
        )

    with c12:

        top_channels = (
            df["TrafficType"]
            .value_counts()
            .head(6)
            .index
        )

        if traffic_type in top_channels:

            st.success(
                "Top traffic channel"
            )

        else:

            st.info(
                "Will be grouped as Other"
            )

    # ------------------------------------------------------
    # BUTTON
    # ------------------------------------------------------

    submitted = st.form_submit_button(
        "🔮 Predict Purchase",
        type="primary",
        use_container_width=True
    )

# =========================================================
# PREDICTION
# =========================================================

if submitted:

    # ------------------------------------------------------
    # ENGINEERED FEATURES
    # ------------------------------------------------------

    page_value_category = pd.cut(
        pd.Series([page_values]),
        bins=[-1, 0, 10, 50, 400],
        labels=[
            "Zero (0)",
            "Low (1-10)",
            "Medium (10-50)",
            "High (>50)"
        ]
    )[0]

    product_minutes = (
        product_duration / 60
    )

    product_category = pd.cut(
        pd.Series([product_minutes]),
        bins=[-1, 5, 20, 60, 2000],
        labels=[
            "Very Low (<5m)",
            "Low (5-20m)",
            "Medium (20-60m)",
            "High (>60m)"
        ]
    )[0]

    top_channels = (
        df["TrafficType"]
        .value_counts()
        .head(6)
        .index
    )

    traffic_grouped = (
        str(traffic_type)
        if traffic_type in top_channels
        else "Other"
    )

    # ------------------------------------------------------
    # BUILD EXACT INPUT
    # ------------------------------------------------------

    input_df = pd.DataFrame({

        "Administrative": [administrative],

        "Administrative_Duration": [
            admin_duration
        ],

        "Informational": [
            informational
        ],

        "Informational_Duration": [
            info_duration
        ],

        "ProductRelated": [
            product_related
        ],

        "ProductRelated_Duration": [
            product_duration
        ],

        "BounceRates": [
            bounce_rate
        ],

        "ExitRates": [
            exit_rate
        ],

        "PageValues": [
            page_values
        ],

        "SpecialDay": [
            special_day
        ],

        "Month": [
            month
        ],

        "OperatingSystems": [
            operating_system
        ],

        "Browser": [
            browser
        ],

        "Region": [
            region
        ],

        "TrafficType": [
            traffic_type
        ],

        "VisitorType": [
            visitor_type
        ],

        "Weekend": [
            weekend
        ],

        "page_value_category": [
            page_value_category
        ],

        "product_minutes": [
            product_minutes
        ],

        "product_category": [
            product_category
        ],

        "Traffic_Grouped": [
            traffic_grouped
        ]
    })

    # IMPORTANT:
    # Same dtypes as notebook

    for col in [
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType"
    ]:

        input_df[col] = input_df[col].astype(
            "object"
        )

    input_df["page_value_category"] = (
        pd.Categorical(
            input_df["page_value_category"],
            categories=[
                "Zero (0)",
                "Low (1-10)",
                "Medium (10-50)",
                "High (>50)"
            ]
        )
    )

    input_df["product_category"] = (
        pd.Categorical(
            input_df["product_category"],
            categories=[
                "Very Low (<5m)",
                "Low (5-20m)",
                "Medium (20-60m)",
                "High (>60m)"
            ]
        )
    )

    # ------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------

    try:

        prediction = model.predict(
            input_df
        )[0]

        probability = model.predict_proba(
            input_df
        )[0]

        purchase_probability = probability[1]

        # --------------------------------------------------
        # RESULT
        # --------------------------------------------------

        st.divider()

        r1, r2 = st.columns(2)

        with r1:

            if prediction == 1:

                st.markdown("""
                <div class="result-box"
                     style="background: #059669;">

                <h2>🛍️ Purchase Expected</h2>

                <p>
                The model predicts that this session
                is likely to result in a purchase.
                </p>

                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div class="result-box"
                     style="background: #DC2626;">

                <h2>🚪 No Purchase Expected</h2>

                <p>
                The model predicts that this session
                is unlikely to result in a purchase.
                </p>

                </div>
                """, unsafe_allow_html=True)

            st.metric(
                "Purchase Probability",
                f"{purchase_probability:.2%}"
            )

            st.progress(
                float(purchase_probability)
            )

        # --------------------------------------------------
        # GAUGE
        # --------------------------------------------------

        with r2:

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=purchase_probability * 100,
                    number={
                        "suffix": "%"
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100]
                        },
                        "threshold": {
                            "line": {
                                "width": 4
                            },
                            "value": 50
                        }
                    }
                )
            )

            fig.update_layout(
                height=300,
                margin=dict(
                    l=20,
                    r=20,
                    t=20,
                    b=20
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # --------------------------------------------------
        # DETAILS BUTTON
        # --------------------------------------------------

        if st.button(
            "🧾 Show Prediction Details",
            use_container_width=True
        ):

            st.markdown(
                "### Prediction Details"
            )

            st.write(
                {
                    "Predicted Class":
                        int(prediction),

                    "No Purchase Probability":
                        f"{probability[0]:.4f}",

                    "Purchase Probability":
                        f"{probability[1]:.4f}"
                }
            )

            st.markdown(
                "### ⚙️ Engineered Features"
            )

            st.dataframe(
                input_df[
                    [
                        "page_value_category",
                        "product_minutes",
                        "product_category",
                        "Traffic_Grouped"
                    ]
                ],
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )