import streamlit as st
import pandas as pd
import joblib
import os
from model_comparison import compare_models


# ==========================
# PAGE CONFIGURATION
# ==========================

st.set_page_config(
    page_title="House Price Prediction Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# LOAD MODEL
# ==========================

model = joblib.load("model/house_price_model.pkl")
city_encoder = joblib.load("model/city_encoder.pkl")
locality_encoder = joblib.load("model/locality_encoder.pkl")

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("house_price_dataset_10000 (1).csv")

# ==========================
# CUSTOM CSS
# ==========================

st.markdown("""
<style>

.main{
    background:#f7f9fc;
}

.title{
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:#1f2937;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}

.block-container{
    padding-top:2rem;
}

div[data-testid="stMetric"]{
    background:white;
    border-radius:12px;
    padding:15px;
    box-shadow:0px 0px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ==========================
# HEADER
# ==========================

st.markdown("<div class='title'>🏠 House Price Prediction Dashboard</div>", unsafe_allow_html=True)

st.markdown("<div class='subtitle'>Machine Learning Based House Price Prediction System</div>", unsafe_allow_html=True)

st.write("")

# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("🏠 Dashboard")

st.sidebar.success("House Price Prediction")

st.sidebar.markdown("---")

st.sidebar.write("### Model")

st.sidebar.info("Random Forest Regressor")

st.sidebar.write("### Dataset Size")

st.sidebar.info(len(df))

st.sidebar.write("### Cities")

st.sidebar.info(df["city"].nunique())

st.sidebar.write("### Features")

st.sidebar.info(11)

st.sidebar.markdown("---")

st.sidebar.success("Developed by")

st.sidebar.write("Deepak Prajapati")

# ==========================
# DASHBOARD CARDS
# ==========================

c1,c2,c3,c4=st.columns(4)

with c1:
    st.metric("Dataset",len(df))

with c2:
    st.metric("Cities",df["city"].nunique())

with c3:
    st.metric("Average price",f"₹ {df['price'].mean():,.0f}")

with c4:
    st.metric("Maximum price",f"₹ {df['price'].max():,.0f}")

st.markdown("---")

# ==========================
# TABS
# ==========================

tab1,tab2,tab3=st.tabs([
    "🏠 Prediction",
    "📊 Visualizations",
    "📈 Model Comparison"
])
# ======================================================
# TAB 1 : HOUSE PRICE PREDICTION
# ======================================================

with tab1:

    st.subheader("🏠 Predict House Price")

    st.write("Fill all the details below.")

    left,right = st.columns(2)

    with left:

        area = st.number_input(
            "Area (sqft)",
            min_value=100,
            max_value=10000,
            value=1500
        )

        bedrooms = st.number_input(
            "Bedrooms",
            min_value=1,
            max_value=10,
            value=3
        )

        bathrooms = st.number_input(
            "Bathrooms",
            min_value=1,
            max_value=10,
            value=2
        )

        floors = st.number_input(
            "Floors",
            min_value=1,
            max_value=5,
            value=2
        )

        parking = st.number_input(
            "Parking Spaces",
            min_value=0,
            max_value=10,
            value=1
        )

        age = st.number_input(
            "House Age (Years)",
            min_value=0,
            max_value=100,
            value=5
        )

    with right:

        city = st.selectbox(
            "City",
            city_encoder.classes_
        )

        locality = st.selectbox(
            "Locality",
            locality_encoder.classes_
        )

        school = st.number_input(
            "School Distance (km)",
            min_value=0.0,
            value=1.0
        )

        hospital = st.number_input(
            "Hospital Distance (km)",
            min_value=0.0,
            value=2.0
        )

        metro = st.number_input(
            "Metro Distance (km)",
            min_value=0.0,
            value=1.0
        )

    st.write("")

    if st.button("🔍 Predict Price", use_container_width=True):

        city_encoded = city_encoder.transform([city])[0]
        locality_encoded = locality_encoder.transform([locality])[0]

        sample = pd.DataFrame({

            "area_sqft":[area],

            "bedrooms":[bedrooms],

            "bathrooms":[bathrooms],

            "floors":[floors],

            "parking":[parking],

            "age_years":[age],

            "city":[city_encoded],

            "locality":[locality_encoded],

            "school_distance_km":[school],

            "hospital_distance_km":[hospital],

            "metro_distance_km":[metro]

        })

        prediction = model.predict(sample)[0]

        st.success(
            f"🏠 Estimated House Price : ₹ {prediction:,.2f}"
        )

        st.markdown("### 📋 Prediction Summary")

        summary = pd.DataFrame({

            "Feature":[
                "Area",
                "Bedrooms",
                "Bathrooms",
                "Floors",
                "Parking",
                "Age",
                "City",
                "Locality",
                "School Distance",
                "Hospital Distance",
                "Metro Distance"
            ],

            "Value":[
                area,
                bedrooms,
                bathrooms,
                floors,
                parking,
                age,
                city,
                locality,
                school,
                hospital,
                metro
            ]

        })

        st.dataframe(
            summary,
            use_container_width=True
        )

        report = pd.DataFrame({

            "Area":[area],

            "Bedrooms":[bedrooms],

            "Bathrooms":[bathrooms],

            "Floors":[floors],

            "Parking":[parking],

            "Age":[age],

            "City":[city],

            "Locality":[locality],

            "School Distance":[school],

            "Hospital Distance":[hospital],

            "Metro Distance":[metro],

            "Predicted Price":[prediction]

        })

        st.download_button(

            "📄 Download Prediction Report",

            report.to_csv(index=False),

            "prediction_report.csv",

            "text/csv"

        )
        # ======================================================
# TAB 2 : DATA VISUALIZATION
# ======================================================

with tab2:

    st.subheader("📊 Data Visualization Dashboard")

    st.write("Explore your dataset through different charts.")

    # ----------------------------------
    # Row 1
    # ----------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### House Price Distribution")

        if os.path.exists("graphs/price_distribution.png"):

            st.image(
                "graphs/price_distribution.png",
                use_container_width=True
            )

        else:

            st.warning("price_distribution.png not found.")

    with col2:

        st.markdown("#### Price vs Area")

        if os.path.exists("graphs/price_vs_area.png"):

            st.image(
                "graphs/price_vs_area.png",
                use_container_width=True
            )

        else:

            st.warning("price_vs_area.png not found.")

    st.markdown("---")

    # ----------------------------------
    # Row 2
    # ----------------------------------

    col3, col4 = st.columns(2)

    with col3:

        st.markdown("#### Correlation Heatmap")

        if os.path.exists("graphs/correlation_heatmap.png"):

            st.image(
                "graphs/correlation_heatmap.png",
                use_container_width=True
            )

        else:

            st.warning("correlation_heatmap.png not found.")

    with col4:

        st.markdown("#### Average Price by City")

        if os.path.exists("graphs/city_average_price.png"):

            st.image(
                "graphs/city_average_price.png",
                use_container_width=True
            )

        else:

            st.warning("city_average_price.png not found.")

    st.markdown("---")

    # ----------------------------------
    # Feature Importance
    # ----------------------------------

    st.markdown("### ⭐ Feature Importance")

    if os.path.exists("graphs/feature_importance.png"):

        st.image(
            "graphs/feature_importance.png",
            use_container_width=True
        )

    else:

        st.warning("feature_importance.png not found.")

    st.markdown("---")

    # ----------------------------------
    # Dataset Preview
    # ----------------------------------

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.markdown("---")

    # ----------------------------------
    # Dataset Statistics
    # ----------------------------------

    st.subheader("📈 Dataset Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.markdown("---")

    # ----------------------------------
    # Missing Values
    # ----------------------------------

    st.subheader("❗ Missing Values")

    missing = pd.DataFrame(
        df.isnull().sum(),
        columns=["Missing Values"]
    )

    st.dataframe(
        missing,
        use_container_width=True
    )
    comparison = compare_models()

st.dataframe(comparison, use_container_width=True)

best = comparison.sort_values(
    "R2 Score",
    ascending=False
).iloc[0]

st.success(
    f"🏆 Best Model: {best['Model']}\n\n"
    f"R² Score: {best['R2 Score']}"
)

st.bar_chart(
    comparison.set_index("Model")["R2 Score"]
)

st.bar_chart(
    comparison.set_index("Model")["RMSE"]
)

st.bar_chart(
    comparison.set_index("Model")["MAE"]
)