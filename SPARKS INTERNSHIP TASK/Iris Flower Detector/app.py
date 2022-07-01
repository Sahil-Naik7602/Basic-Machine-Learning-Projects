import joblib
import streamlit as st
import pandas as pd

data = pd.read_csv('Iris.csv')
print(data.head(5))


st.set_page_config(page_title = 'Iris Species Predictor',page_icon = '🌺' )
st.title('Machine Learning Model Deployment')
st.write('### Iris Flower Classifier Using *Decision Tree*')
sepal_lt = st.slider('Sepal Length(in cm)',min(data['SepalLengthCm']),max(data['SepalLengthCm'])) 
sepal_wd = st.slider('Sepal Width(in cm)',min(data['SepalWidthCm']),max(data['SepalWidthCm']))
petal_lt = st.slider('Petal Length(in cm)',min(data['PetalLengthCm']),max(data['PetalLengthCm']))
petal_wd = st.slider('Petal Width(in cm)',min(data['PetalWidthCm']),max(data['PetalWidthCm']))



mymodel = joblib.load('Iris_Predictor.pkl')
prediction = mymodel.predict([[sepal_lt,sepal_wd,petal_lt,petal_wd]])
st.write(f'### The Flower is: *{prediction[0]}*')
