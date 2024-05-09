import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
@st.cache
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/auto-mpg.data"
    columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']
    return pd.read_csv(url, sep='\s+', names=columns)

data = load_data()

# Sidebar
st.sidebar.header('User Input')

def user_input_features():
    cylinders = st.sidebar.slider('Cylinders', 3, 8, 4)
    displacement = st.sidebar.slider('Displacement', 70, 500, 200)
    horsepower = st.sidebar.slider('Horsepower', 40, 300, 120)
    weight = st.sidebar.slider('Weight', 1000, 5000, 2500)
    acceleration = st.sidebar.slider('Acceleration', 8, 25, 15)
    model_year = st.sidebar.slider('Model Year', 70, 82, 76)
    origin = st.sidebar.selectbox('Origin', ('USA', 'Europe', 'Japan'))
    origin = 1 if origin == 'USA' else 2 if origin == 'Europe' else 3
    return cylinders, displacement, horsepower, weight, acceleration, model_year, origin

cylinders, displacement, horsepower, weight, acceleration, model_year, origin = user_input_features()

# Preprocessing
X = data.drop(['mpg', 'car name'], axis=1)
y = data['mpg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Make predictions
prediction = model.predict([[cylinders, displacement, horsepower, weight, acceleration, model_year, origin]])

st.header('Predicted Fuel Consumption')
st.write('The predicted fuel consumption (MPG) of the car is:', prediction[0])
