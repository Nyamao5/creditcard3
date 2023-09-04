import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split  # Add this import
from sklearn.metrics import classification_report
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Credit Card Fraud Detection")
st.caption("Visualization of the data below:-")

# Load your dataset
data = pd.read_csv('creditcard.csv', nrows=1000)

st.write(data)  # Display the data as text
st.dataframe(data)  # Display the data as a DataFrame

st.header("Line graph ")
st.line_chart(data)  # Display a line chart (this might not be suitable for displaying the data)

# Create a histogram for a specific column (e.g., 'Amount')
selected_column = 'Amount'

# Create a histogram using Matplotlib
plt.hist(data[selected_column], bins=20)
plt.xlabel(selected_column)
plt.ylabel('Frequency')
plt.title(f'Histogram of {selected_column}')
st.pyplot()  # Display the Matplotlib figure using Streamlit

st.set_option('deprecation.showPyplotGlobalUse', False)

# Create a sidebar for adjusting test size
st.sidebar.title("Random Forest Classifier App")

# Data preprocessing
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Instantiate and fit the model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Display classification report
st.sidebar.subheader("Classification Report")
classification_rep = classification_report(y_test, y_pred)
st.sidebar.text(classification_rep)

# Allow users to input values for prediction
st.sidebar.header("Fraud Prediction")

# Create input fields for relevant features
time = st.sidebar.number_input("Time", min_value=0, max_value=1000)
v1 = st.sidebar.number_input("V1", min_value=-50, max_value=50)
v2 = st.sidebar.number_input("V2", min_value=-50, max_value=50)
v3 = st.sidebar.number_input("V3", min_value=-50, max_value=50)
v4 = st.sidebar.number_input("V4", min_value=-50, max_value=50)
v5 = st.sidebar.number_input("V5", min_value=-50, max_value=50)
v6 = st.sidebar.number_input("V6", min_value=-50, max_value=50)
v7 = st.sidebar.number_input("V7", min_value=-50, max_value=50)
v8 = st.sidebar.number_input("V8", min_value=-50, max_value=50)
v9 = st.sidebar.number_input("V9", min_value=-50, max_value=50)
v10 = st.sidebar.number_input("V10", min_value=-50, max_value=50)
v11 = st.sidebar.number_input("V11", min_value=-50, max_value=50)
v12 = st.sidebar.number_input("V12", min_value=-50, max_value=50)
v13 = st.sidebar.number_input("V13", min_value=-50, max_value=50)
v14 = st.sidebar.number_input("V14", min_value=-50, max_value=50)
v15 = st.sidebar.number_input("V15", min_value=-50, max_value=50)
v16 = st.sidebar.number_input("V16", min_value=-50, max_value=50)
v17 = st.sidebar.number_input("V17", min_value=-50, max_value=50)
v18 = st.sidebar.number_input("V18", min_value=-50, max_value=50)
v19 = st.sidebar.number_input("V19", min_value=-50, max_value=50)
v20 = st.sidebar.number_input("V20", min_value=-50, max_value=50)
v21 = st.sidebar.number_input("V21", min_value=-50, max_value=50)
v22 = st.sidebar.number_input("V22", min_value=-50, max_value=50)
v23 = st.sidebar.number_input("V23", min_value=-50, max_value=50)
v24 = st.sidebar.number_input("V24", min_value=-50, max_value=50)
v25 = st.sidebar.number_input("V25", min_value=-50, max_value=50)
v26 = st.sidebar.number_input("V26", min_value=-50, max_value=50)
v27 = st.sidebar.number_input("V27", min_value=-50, max_value=50)
v28 = st.sidebar.number_input("V28", min_value=-50, max_value=50)
amount = st.sidebar.number_input("Amount", min_value=0.0, max_value=10000.0)



 
# Create a button to make predictions
if st.sidebar.button("Predict Fraud"):
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'Time': [time],
        'V1': [v1],
        'V2': [v2],
        'V3': [v3],
        'V4': [v4],
        'V5': [v5],
        'V6': [v6],
        'V7': [v7],
        'V8': [v8],
        'V9': [v9],
        'V10': [v10],
        'V11': [v11],
        'V12': [v12],
        'V13': [v13],
        'V14': [v14],
        'V15': [v15],
        'V16': [v16],
        'V17': [v17],
        'V18': [v18],
        'V19': [v19],
        'V20': [v20],
        'V21': [v21],
        'V22': [v22],
        'V23': [v23],
        'V24': [v24],
        'V25': [v25],
        'V26': [v26],
        'V27': [v27],
        'V28': [v28],
        'Amount': [amount]
        
        
    })

     # Make a prediction using your model
    prediction = rf.predict(input_data)
    # Display the result as "Fraudulent" or "Not Fraudulent"
    if prediction[0] == 1:
        st.sidebar.subheader("Prediction Result")
        st.sidebar.text("Fraudulent")
    else:
        st.sidebar.subheader("Prediction Result")
        st.sidebar.text("Not Fraudulent")
