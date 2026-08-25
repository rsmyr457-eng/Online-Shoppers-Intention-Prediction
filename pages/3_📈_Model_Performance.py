
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Model Performance")

st.write(
    "Comparison of classification models before tuning "
    "and performance of the final XGBoost model."
)

# =========================================================
# MODEL COMPARISON BEFORE TUNING
# =========================================================

st.subheader("🔹 Before Hyperparameter Tuning")

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Gaussian NB",
        "Decision Tree",
        "Random Forest",
        "XGBoost"
    ],

    "Train Precision": [
        51.90, 22.15, 100.00, 100.00, 91.90
    ],

    "Test Precision": [
        50.82, 21.94, 56.89, 75.98, 62.79
    ],

    "Train Recall": [
        77.59, 91.08, 100.00, 99.97, 99.97
    ],

    "Test Recall": [
        76.51, 90.62, 55.19, 50.59, 70.40
    ]
})

st.dataframe(
    comparison.style.format({
        "Train Precision": "{:.2f}%",
        "Test Precision": "{:.2f}%",
        "Train Recall": "{:.2f}%",
        "Test Recall": "{:.2f}%"
    }),
    use_container_width=True
)

# =========================================================
# PRECISION VS RECALL
# =========================================================

st.subheader("🎯 Test Precision vs Recall")

plot_df = comparison[
    ["Model", "Test Precision", "Test Recall"]
].melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score"
)

fig = px.bar(
    plot_df,
    x="Model",
    y="Score",
    color="Metric",
    barmode="group",
    text="Score"
)

fig.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)

fig.update_layout(
    yaxis_title="Score (%)",
    xaxis_title="Model",
    yaxis_range=[0, 100]
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# RANDOMIZED SEARCH CV
# =========================================================

st.divider()

st.subheader("⚙️ Hyperparameter Tuning")

st.write(
    "XGBoost was optimized using RandomizedSearchCV "
    "with Accuracy as the scoring metric."
)

mean_train_scores = np.array([
    1.00000000,
    0.99803185,
    0.98622023,
    1.00000000,
    0.96177813,
    0.99393039
])

mean_test_scores = np.array([
    0.65744823,
    0.72570966,
    0.76441544,
    0.68828947,
    0.78474978,
    0.74014236
])

best_index = mean_test_scores.argmax()

best_cv_score = mean_test_scores[best_index]
best_train_score = mean_train_scores[best_index]

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Best CV Accuracy",
        f"{best_cv_score:.2%}"
    )

with col2:
    st.metric(
        "Train Accuracy",
        f"{best_train_score:.2%}"
    )

st.caption(
    "RandomizedSearchCV scoring metric: Accuracy"
)

# =========================================================
# FINAL MODEL
# =========================================================

st.divider()

st.subheader("🏆 Final Model")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "XGBoost"
    )

with col2:
    st.metric(
        "Test Precision",
        "62.79%"
    )

with col3:
    st.metric(
        "Test Recall",
        "70.40%"
    )

# =========================================================
# MODEL INSIGHT
# =========================================================

st.subheader("💡 Why XGBoost?")

st.info(
    """
    XGBoost was selected because it provided a good balance
    between Precision and Recall compared with the other
    tested models.

    Before tuning:
    • Test Precision = 62.79%
    • Test Recall = 70.40%

    After tuning:
    • Cross-Validation Accuracy = 78.47%

    Note: 78.47% is Accuracy, while 62.79% and 70.40%
    are Precision and Recall.
    """
)

