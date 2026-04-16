import streamlit as st
import joblib 
import pandas as pd 

# Load model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title
st.title('🔮 Customer Churn Prediction System')
st.write('Enter customer details to predict churn probability')

# Input fields
st.subheader('Customer Details')

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider('Tenure (months)', 0, 72, 12)
    monthly_charges = st.number_input('Monthly Charges', 0.0, 150.0, 50.0)
    total_charges = st.number_input('Total Charges', 0.0, 9000.0, 500.0)

with col2:
    contract = st.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
    internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    payment_method = st.selectbox(
        'Payment Method',
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
    )
    gender = st.selectbox('Gender', ['Male', 'Female'])

with col1:
    senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.selectbox('Partner', ['No', 'Yes'])
    dependents = st.selectbox('Dependents', ['No', 'Yes'])

with col2:
    phone_service = st.selectbox('Phone Service', ['No', 'Yes'])
    online_security = st.selectbox('Online Security', ['No', 'Yes'])
    tech_support = st.selectbox('Tech Support', ['No', 'Yes'])

with col1:
    multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes'])
    online_backup = st.selectbox('Online Backup', ['No', 'Yes'])
    device_protection = st.selectbox('Device Protection', ['No', 'Yes'])

with col2:
    streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes'])
    streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes'])
    paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])

# Prediction button
if st.button('🔮 Predict Churn'):

    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,

        'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
        'Partner': 1 if partner == 'Yes' else 0,
        'Dependents': 1 if dependents == 'Yes' else 0,
        'PhoneService': 1 if phone_service == 'Yes' else 0,
        'OnlineSecurity': 1 if online_security == 'Yes' else 0,
        'TechSupport': 1 if tech_support == 'Yes' else 0,

        # ✅ Missing features FIXED
        'gender': 1 if gender == 'Male' else 0,
        'MultipleLines': 1 if multiple_lines == 'Yes' else 0,
        'OnlineBackup': 1 if online_backup == 'Yes' else 0,
        'DeviceProtection': 1 if device_protection == 'Yes' else 0,
        'StreamingTV': 1 if streaming_tv == 'Yes' else 0,
        'StreamingMovies': 1 if streaming_movies == 'Yes' else 0,
        'PaperlessBilling': 1 if paperless_billing == 'Yes' else 0,

        # Contract OHE
        'Contract_One year': 1 if contract == 'One year' else 0,
        'Contract_Two year': 1 if contract == 'Two year' else 0,
        'Contract_Month-to-month': 1 if contract == 'Month-to-month' else 0,

        # Internet Service OHE
        'InternetService_DSL': 1 if internet_service == 'DSL' else 0,
        'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
        'InternetService_No': 1 if internet_service == 'No' else 0,

        # Payment Method OHE
        'PaymentMethod_Electronic check': 1 if payment_method == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == 'Mailed check' else 0,
        'PaymentMethod_Bank transfer (automatic)': 1 if payment_method == 'Bank transfer (automatic)' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == 'Credit card (automatic)' else 0,
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Match training column order
    input_df = input_df[scaler.feature_names_in_]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    # Output
    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is not likely to churn")

    st.info(f"Churn Probability: {probability:.2f}")