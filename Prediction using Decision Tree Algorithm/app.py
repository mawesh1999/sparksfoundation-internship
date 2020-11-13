import streamlit as st
import pickle
import pandas as pd
import sklearn
import numpy as np

st.sidebar.title('User Input Parameters')
st.header('Predicting Iris Flower Species')


def mapper(x):
    dict = {0:'Iris-Setosa',1:'Iris-Versicolour',2:'Iris-Verginica'}
    if x in dict.keys():
        return dict[x]



def user_params():
    length_s = st.sidebar.slider('Sepal Length', 0.0, 10.0, 3.0)
    width_s = st.sidebar.slider('Sepal Width', 0.0, 5.0, 3.0)
    length_p =  st.sidebar.slider('Petal Length',0.0,10.0,3.0)
    width_p = st.sidebar.slider('Petal Width',0.0,5.0,2.0)

    data = pd.DataFrame({'Sepal Length':length_s,'Sepal width':width_s,'Petal length':length_p,'Sepal length':width_p},index=[0])
    return data


user_data = user_params()
model = pickle.load( open('tree_model.pkl','rb'))
prediction = model.predict(user_data.values)
species = mapper(prediction[0])
st.subheader('Input Data')
st.write(user_data)
st.subheader('Predicted Species:-')
st.success(species)






