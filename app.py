import streamlit as st
import pickle
import pandas as pd

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Page configuration
st.set_page_config(page_title="Tanzania Pulse Price Predictor", layout="centered")

# Title
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

# ---------------- INPUT SECTION ----------------
st.subheader("Enter Market Conditions")

month = st.slider("Month of the Year", 1, 12, 6)
rainfall = st.number_input("Rainfall Amount (mm)", min_value=0.0, value=800.0)
fuel_price = st.number_input("Transport Fuel Price (TZS/Litre)", min_value=0.0, value=5000.0)
supply = st.number_input("Market Supply (tons)", min_value=0.0, value=120.0)
demand = st.number_input("Demand Index", min_value=0.0, value=100.0)

st.divider()

# ---------------- PREDICTION ----------------
if st.button("Predict Beans Price"):

    # Use EXACT column names from training dataset (lowercase!)
    input_data = pd.DataFrame({
        'month': [month],
        'rainfall': [rainfall],
        'fuel_price': [fuel_price],
        'supply': [supply],
        'demand': [demand]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Beans Market Price: {prediction:.2f} TZS per Kg")

    if prediction > 3500:
        st.warning("⚠ High price expected — good time for farmers to sell.")
    elif prediction < 2000:
        st.info("📉 Low price expected — traders may consider buying and storing.")
    else:
        st.info("📊 Normal market conditions expected.")

st.divider()
st.caption("Developed by shelby - Tanzania")
