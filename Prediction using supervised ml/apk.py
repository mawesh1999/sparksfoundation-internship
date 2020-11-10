import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.ensemble import RandomForestRegressor

st.title('Student Score Prediction Using Study Hours Per Day')

st.sidebar.title('User Input Parameters')

model = st.sidebar.selectbox('Model to be used',['Random Forest', 'Linear Regression'])

hours = st.sidebar.number_input('Enter study Hours per day', min_value = 0.0, max_value=24.0)

if model=='Random Forest':
     model = pickle.load(open('forest_model.pkl', 'rb'))
else:
     model = pickle.load(open('reg_model.pkl', 'rb'))

prediction = model.predict(np.array(hours).reshape((1, 1)))

if prediction>=50:
    st.success('Estimated Score  \t    {} '.format(prediction))
else:
    st.warning('Estimated Score {}'.format(prediction))


