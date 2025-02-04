"""A simple Union app using Streamlit to serve an sklearn model with streamlit."""

import streamlit as st
import joblib
from union_runtime import get_input

# Load the model artifact downloaded by Union.
model_path = get_input("sklearn_model")
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Title and description
st.title("Iris Classifier")
st.write("Enter the iris features to predict the species.")

# Input fields for the four iris features.
# You can label them as appropriate (e.g., sepal length, sepal width, etc.)
sepal_length = st.number_input("Sepal Length", value=5.0, format="%.2f")
sepal_width  = st.number_input("Sepal Width",  value=3.0, format="%.2f")
petal_length = st.number_input("Petal Length", value=1.5, format="%.2f")
petal_width  = st.number_input("Petal Width",  value=0.2, format="%.2f")

flower_types = ['setosa','versicolor','virginica']


if st.button("Predict"):
    try:
        # Prepare the features as a 2D array.
        features = [[sepal_length, sepal_width, petal_length, petal_width]]
        # The model should return a prediction (e.g., a class label or integer)
        prediction = model.predict(features)
        st.success(f"Predicted species: {flower_types[prediction[0]]}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# union deploy apps app.py simple-streamlit-iris