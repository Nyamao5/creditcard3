import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
 
st.title("Credit Card Fraud Detection")
st.caption("visualisation of the data below:-")

import pandas as pd
import streamlit as st

def load_data(nrows):
    data = pd.read_csv('creditcard.csv', nrows=nrows)
    return data

data = load_data(1000)  # Load only 1000 rows

st.write(data)  # Display the data as text
st.dataframe(data)# Display the data as a DataFrame
st.header("Line graph ")
st.line_chart(data)  # Display a line chart (this might not be suitable for displaying the data)

st.set_option('deprecation.showPyplotGlobalUse', False)
# Display a histogram for a specific column (e.g., 'Amount')
selected_column = 'Amount'

# Create a histogram using Matplotlib
plt.hist(data[selected_column], bins=20)
plt.xlabel(selected_column)
plt.ylabel('Frequency')
plt.title(f'Histogram of {selected_column}')
st.pyplot()  # Display the Matplotlib figure using Streamlit

# create a base classification model for credit card fraud detection

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

st.title("Random Forest Classifier App")


# Data preprocessing
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create a sidebar for adjusting test size
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.25, 0.05)

# Instantiate and fit the model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Display classification report
st.subheader("Classification Report")
classification_rep = classification_report(y_test, y_pred)
st.text(classification_rep)




 
