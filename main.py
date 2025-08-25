import streamlit as st
import pandas as pd
import numpy as np
from os import path
import pickle #it is used to read pickle file

st.title("Iris dataset")
df_iris = pd.read_csv(path.join("Data", "iris.csv"))
st.write(df_iris)

st.title("Flower species Predictor")

petal_length = st.number_input("please choose a petal length",placeholder="please enter a value between 1 and 6.9",
                                    min_value=1.0, max_value=6.9,value=None)
petal_width = st.number_input("please choose a petal_width",placeholder="please enter a value between 0.1 and 2.5",min_value=0.100000, max_value=2.500000,value=None)
sepal_length = st.number_input("please choose a sepal_length",placeholder="please enter a value between 4.3 and 7.9",min_value=4.300000, max_value=7.900000,value=None)
sepal_width = st.number_input("please choose a sepal_width",placeholder="please enter a value between 2 and 4.4",min_value=2.000000, max_value=4.400000,value=None)


#prepare the dataframe for preparation
df_user_input = pd.DataFrame([[sepal_length, sepal_width,petal_length,petal_width]],
                          columns=['sepal_length','sepal_width','petal_length', 'petal_width'])


#using the .pkl file, creating an ML model named 'iris_predictor'
model_path = path.join("Model", "iris_model.pkl")
with open (model_path,'rb') as file:
    iris_predictor = pickle.load(file)


species = {0:'setosa', 1:'versicolor', 2:'virginica'}


if st.button("predict species"):
    if((petal_length==None) or (petal_width==None) or (sepal_length==None) or (sepal_width==None)):
        st.write("Please fill all values") #willl be executed when any values is not entered properly
    else:

     predicted_species = iris_predictor.predict(df_user_input)
     st.write("the Species is ",species[predicted_species[0]])