from cProfile import label
import joblib
import streamlit as st

st.set_page_config(page_title = 'My Marks Prediction',page_icon = '👨‍🎓' )
st.title('Machine Learning Model Deployment')
st.write('### *Marks* Vs Number of Hours')
mymodel = joblib.load('Marks_Predictior.pkl')
hours = st.slider('Study Time (in hrs.)', 0.0, 10.0)
st.write(f"##### I study {hours} hours in a day.")

marks = mymodel.predict([[hours]])
if marks[0]>=100:
    marks[0] = 100

st.write(f"### The marks obtained by the student will be approximately: {round(marks[0])}/100")
