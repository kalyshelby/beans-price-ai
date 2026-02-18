import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tanzania Pulse Price Predictor", layout="centered")

st.title("🌾 Tanzania Pulse (Beans) Price Prediction System")
st.markdown("### AI-based Agricultural Decision Support Tool")

st.write("""
This system predicts the market price of beans in Tanzania using Machine Learning.
It helps:
- Farmers decide when to sell
- Traders plan stock
- Consumers understand price trends
""")

st.divider()
st.subheader("Enter Market Conditions")

month = st.slider("Month of the Year", 1, 12, 6)

rainfall = st.number_input(
    "Rainfall Amount (mm)",
    min_value=200.0,
    max_value=1200.0,
    value=800.0
)

fuel_price = st.number_input(
    "Transport Fuel Price (TZS/Litre)",
    min_value=2200.0,
    max_value=4000.0,
    value=3000.0
)

supply = st.number_input(
    "Market Supply (tons)",
    min_value=50.0,
    max_value=300.0,
    value=120.0
)

demand = st.number_input(
    "Demand Index",
    min_value=60.0,
    max_value=200.0,
    value=100.0
)

st.divider()

# ---------- PREDICTION ----------
if st.button("Predict Beans Price"):

    # Reload model
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    # Prepare input
    input_data = pd.DataFrame([{
        "month": month,
        "rainfall": rainfall,
        "fuel_price": fuel_price,
        "supply": supply,
        "demand": demand
    }])

    # Predict
    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Beans Market Price: {prediction:.2f} TZS per Kg")

    # Model info
    st.info("Model trained using 2000 simulated Tanzanian market records.")
    st.info("Algorithm used: Decision Tree Regression (Supervised Machine Learning)")

    # Interpretation
    if prediction > 4500:
        st.warning("⚠ High price expected — good time for farmers to sell.")
    elif prediction < 2500:
        st.info("📉 Low price expected — traders should stock beans now.")
    else:
        st.write("📊 Normal market price range.")

    # Seasonal explanation
    if month in [5,6,7,8]:
        st.write("🌧 Harvest season detected: supply is high → prices usually fall.")
    elif month in [11,12,1,2]:
        st.write("☀ Scarcity season detected: supply is low → prices usually increase.")

    # Visualization
    st.subheader("Market Insight")

    factors = ['Rainfall','Fuel','Supply','Demand']
    values = [rainfall, fuel_price/10, supply*5, demand*5]

    fig, ax = plt.subplots()
    ax.bar(factors, values)
    ax.set_ylabel("Relative Influence")
    ax.set_title("Market Factors Affecting Beans Price")

    st.pyplot(fig)

st.divider()
st.caption("Developed by Data Science Students - Tanzania")
