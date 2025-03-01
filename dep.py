import pandas as pd 
import pickle
import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder
import openpyxl
def load_model():
  with open('model.pkl', 'rb') as file:
      model = pickle.load(file)
  return model

def preprocess(input_data):
  encoder = LabelEncoder()
  for column in input_data.columns:
    if input_data[column].dtype == object:
      input_data[column] = encoder.fit_transform(input_data[column])
  return input_data

def predict_data(input_data, model):
  prediction = model.predict(input_data)
  return prediction

def count_outliers(data):
  outlier_counts = {}
  for column in data.columns:
    if data[column].dtype != 'int64' and data[column].dtype != 'float64':
      continue
    feature_data = data[column]
    Q1 = np.percentile(feature_data, 10)
    Q3 = np.percentile(feature_data, 90)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    outliers = feature_data[(feature_data < lower_bound) | (feature_data > upper_bound)]
    outlier_counts[column] = len(outliers)
  return outlier_counts

def main():
  st.title('Date Fruit Class Prediction')
  st.write('The model used here is an XGB Classifier')
  About_work, Prediction = st.tabs(["About the Work", "Prediction"])
  with About_work:
    st.title('Data Information')
    st.write('The dataset should contain the following columns:')
    data = pd.read_excel('raw_data.xlsx')
    columns = data.columns.tolist()
    st.write(columns)
    st.write('The target variable is "Class" that contains these Classes:')
    st.write(data['Class'].unique().tolist())
    st.markdown('### **About Data Quality :**')
    
    overview, missValues, outliers, imbalancedData = st.tabs(["Overview", "Missing Values", "Outliers", "Imbalanced Data"])
    with overview:
      st.write('The dataset contains the following columns:')
      st.write(data.columns)
      st.write('The dataset contains the following number of rows and columns:')
      st.write(data.shape)
      st.write('The dataset contains the following data types:')
      st.write(data.dtypes)
    with missValues:
      st.write('The dataset contains the following missing values:')
      st.write(data.isnull().sum())
    with outliers:
      st.write('The dataset contains the following outliers:')
      st.write(count_outliers(data))
    with imbalancedData:
      st.write('The dataset contains the following imbalanced data:')
      st.write(data['Class'].value_counts())
    st.markdown('### **Data Pre-processing :**')
    st.write('since the data is in good shape, we will only encode the categorical data and scale the input features')
    st.write('the data will be preprocessed using the following steps:')
    st.write('1. Encoding the categorical data')
    st.write('2. Scaling the input data')
    st.write('the input data should be in the following format: ')
    st.write(preprocess(data.drop('Class', axis=1).head()))
  with Prediction:
    st.title('Prediction')
    st.write('To make a prediction, please upload a file containing the data to be predicted.')
    st.markdown('### The submitted data should be in the following format:')
    st.write(data.drop('Class', axis=1).head()) 
    uploaded_file = st.file_uploader("Upload a file (CSV or Excel)", type=["csv", "xlsx"])
    if uploaded_file is not None:
      # Read the file into a DataFrame
      if uploaded_file.name.endswith('.csv'):
          df = pd.read_csv(uploaded_file)
      elif uploaded_file.name.endswith('.xlsx'):
          df = pd.read_excel(uploaded_file)

      st.success('The dataset was submitted successfully')
      st.write('Preview of the dataset:')
      st.dataframe(df.head())
      try:
        processed_data = preprocess(df)
        st.success('Data preprocessing completed successfully.')

        # Load the model
        model = load_model()
        st.success('Model loaded successfully.')

        # Making predictions
        prediction = predict_data(processed_data, model)
        st.write('Predictions:')
        st.write(prediction)

      except Exception as e:
          st.error(f"An error occurred during processing: {e}")
    else:
        st.error('Error: No file was uploaded.')
        st.warning('Please upload a file to make a prediction.')

if __name__ == '__main__':
    main()
