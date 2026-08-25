import streamlit as st
import pandas as pd
import plotly.express as px

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Online Shoppers | EDA",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("online_shoppers_intention.csv")

    # Same feature engineering from notebook

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


df = load_data()

# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.eda-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #14213D;
}

.eda-subtitle {
    color: #64748B;
    font-size: 1rem;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="eda-title">📊 Exploratory Data Analysis</div>',
    unsafe_allow_html=True
)


st.divider()

# =========================================================
# SIDEBAR FILTERS
# =========================================================

st.sidebar.header("🎛️ Interactive Filters")

visitor_options = sorted(
    df["VisitorType"].dropna().unique()
)

selected_visitors = st.sidebar.multiselect(
    "Visitor Type",
    visitor_options,
    default=visitor_options
)

revenue_filter = st.sidebar.multiselect(
    "Revenue",
    [False, True],
    default=[False, True]
)

filtered_df = df[
    df["VisitorType"].isin(selected_visitors)
    &
    df["Revenue"].isin(revenue_filter)
]

st.sidebar.metric(
    "Filtered Sessions",
    f"{len(filtered_df):,}"
)

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "🎯 Purchase Overview",
    "📄 Engagement Analysis",
    "🚦 Traffic & Behavior"
])

# =========================================================
# TAB 1
# Revenue + Visitor Type
# =========================================================

with tab1:

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # Revenue Distribution
    # -----------------------------------------------------

    with col1:

        revenue_df = (
            filtered_df["Revenue"]
            .value_counts()
            .reset_index()
        )

        revenue_df.columns = [
            "Revenue",
            "Count"
        ]

        revenue_df["Revenue"] = revenue_df[
            "Revenue"
        ].map({
            False: "No Purchase",
            True: "Purchase"
        })

        fig1 = px.pie(
            revenue_df,
            names="Revenue",
            values="Count",
            title="Revenue Distribution"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    # -----------------------------------------------------
    # Visitor Type
    # -----------------------------------------------------

    with col2:

        vist_df = (
            filtered_df
            .groupby("VisitorType")["Revenue"]
            .value_counts(
                normalize=True
            )
            .reset_index(
                name="Percentage"
            )
        )

        vist_df["Percentage"] *= 100

        fig2 = px.bar(
            vist_df,
            x="VisitorType",
            y="Percentage",
            color="Revenue",
            barmode="group",
            title="Revenue Distribution by Visitor Type"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

# =========================================================
# TAB 2
# Page Values + Product Duration
# =========================================================

with tab2:

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # Page Value Category
    # -----------------------------------------------------

    with col1:

        df_page = (
            filtered_df
            .groupby(
                "page_value_category",
                observed=True
            )["Revenue"]
            .value_counts(
                normalize=True
            )
            .reset_index(
                name="Percentage"
            )
        )

        df_page["Percentage"] *= 100

        fig3 = px.bar(
            df_page,
            x="page_value_category",
            y="Percentage",
            color="Revenue",
            barmode="group",
            title="Revenue Distribution by Page Value Category",
            text_auto=True
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    # -----------------------------------------------------
    # Product Duration Category
    # -----------------------------------------------------

    with col2:

        df_min = (
            filtered_df
            .groupby(
                "product_category",
                observed=True
            )["Revenue"]
            .value_counts(
                normalize=True
            )
            .reset_index(
                name="Percentage"
            )
        )

        df_min["Percentage"] *= 100

        fig4 = px.bar(
            df_min,
            x="product_category",
            y="Percentage",
            color="Revenue",
            barmode="group",
            title=(
                "Revenue Distribution by "
                "Product Related Duration Category"
            ),
            text_auto=True
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )

# =========================================================
# TAB 3
# Traffic + Bounce/Exit
# =========================================================

with tab3:

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # Traffic Group
    # -----------------------------------------------------

# -----------------------------------------------------
# Conversion Rate Across Traffic Types
# -----------------------------------------------------

with col1:

    traffic_df = (
        filtered_df
        .groupby("Traffic_Grouped")["Revenue"]
        .value_counts(normalize=True)
        .reset_index(name="Percentage")
    )

    traffic_df["Percentage"] = (
        traffic_df["Percentage"] * 100
    ).round(2)

    # ترتيب Traffic Groups
    traffic_order = sorted(
        traffic_df["Traffic_Grouped"].unique(),
        key=lambda x: int(x) if str(x).isdigit() else 999
    )

    fig5 = px.bar(
        traffic_df,
        x="Traffic_Grouped",
        y="Percentage",
        color="Revenue",
        barmode="group",
        text="Percentage",
        title="Conversion Rate Across Traffic Types",
        category_orders={
            "Traffic_Grouped": traffic_order
        }
    )

    # كتابة النسبة فوق كل عمود
    fig5.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        width=0.35
    )

    # أهم جزء: إظهار كل Traffic Group تحت الأعمدة
    fig5.update_xaxes(
        type="category",
        title="Traffic Group",
        tickmode="array",
        tickvals=traffic_order,
        ticktext=traffic_order,
        tickangle=0
    )

    fig5.update_yaxes(
        title="Conversion Rate (%)",
        range=[0, 105]
    )

    fig5.update_layout(
        height=550,
        margin=dict(
            l=50,
            r=30,
            t=80,
            b=80
        ),
        legend_title="Revenue"
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )
    # -----------------------------------------------------
    # Bounce vs Exit
    # -----------------------------------------------------

    with col2:

        fig6 = px.scatter(
            filtered_df,
            x="BounceRates",
            y="ExitRates",
            color="Revenue",
            size="PageValues",
            title=(
                "Bounce Rates vs Exit Rates "
                "with Page Values and Revenue Status"
            ),
            opacity=0.7
        )

        st.plotly_chart(
            fig6,
            use_container_width=True
        )

st.divider()

