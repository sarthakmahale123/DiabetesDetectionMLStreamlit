# Importing the dependencies

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import streamlit as st

diabetes_dataset = pd.read_csv('diabetes.csv')

diabetes_dataset.head()

#Getting statistical measures of dataset
diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()

diabetes_dataset.groupby('Outcome').mean()

#Separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome',axis = 1)
Y = diabetes_dataset['Outcome']

scaler = StandardScaler()

scaler.fit(X)

standardized_data = scaler.transform(X)

X = standardized_data

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,stratify=Y,random_state=2)

classifier = svm.SVC(kernel = 'linear')

classifier.fit(X_train,Y_train)

#accuracy score on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)

print("The accuracy of training data is",training_data_accuracy)

#accuracy score on testing data
X_test_prediction = classifier.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction,Y_test)

print("The accuracy of testing data is",testing_data_accuracy)

# Making a Predictive System

def diabetesprediction(input_data):

    #changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    std_data = scaler.transform(input_data_reshaped)


    prediction = classifier.predict(std_data)
    if(prediction[0] == 0):
        return "The person is not diabetic"
    else:
        return "The person is diabetic"

def main():
    st.title('Diabetes Prediction Web App')


    Pregnancies = st.text_input("Enter number of pregnancies:")
    Glucose = st.text_input("Enter Glucose:")
    BloodPressure = st.text_input("Enter Blood Pressure:")
    SkinThickness = st.text_input("Enter Skin thickness:")
    Insulin = st.text_input("Enter Skin Thickness:")
    BMI = st.text_input("Enter BMI:")
    DiabetesPedigreeFunction = st.text_input("Enter Diabetes Pedigree Function:")
    Age = st.text_input("Enter Age:")


    diagnosis = ''

    if st.button('Diabetes Test Result'):
        diagnosis = diabetesprediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])


    st.success(diagnosis)

if __name__ == '__main__':
   main()
