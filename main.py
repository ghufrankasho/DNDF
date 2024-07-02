# import streamlit as st
# import pandas as pd
# import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import  accuracy_score
#
# # Load your trained model
#
# import keras
# # Load your trained model
# model = keras.models.load_model('EVAallvar.keras')
# # Load your dataset
# def proccess_data(df):
#     df_copy=df
#     df_copy['Y'].replace('a', 1, inplace=True)
#     df_copy['Y'].replace('b', 0, inplace=True)
#
#
#     X = df_copy.iloc[:88, 0:12]
#     y=df_copy.iloc[:88,12:13]
#     scaler = MinMaxScaler()
#     X_train_scaled = scaler.fit_transform(X)
#     return X_train_scaled,df,y
#
# data = pd.read_csv('old_Dataset.csv')
# X,data,y=proccess_data(data)
#
# # Define the function to classify a row
# def classify_row(row):
#
#     row = row.reshape(1, -1)
#
#     prediction = model.predict(row)
#
#     return 'Good' if prediction[0] >= 0.99 else 'Bad'
#
#
# # Streamlit app
# st.title('Bank Classification App')
#
# # Let user select a row
# row_index = st.number_input('Select row number (0 to 87):', min_value=0, max_value=len(data)-1, value=0)
#
# # Display the selected row
# selected_row = data.iloc[row_index]
# st.write('Selected Row Data:')
# st.write(selected_row)
#
# # Classify the selected row
# if st.button('Classify'):
#     processed_row = X[row_index]  # Get the corresponding processed row from X
#     result = classify_row(
#         processed_row
#     )
#     st.write(f'The model classifies this bank as: {result} ')
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
# Load your trained model

import keras

# Load your trained model
model = keras.models.load_model('EVAallvar.keras')

# Load your dataset
def process_data(df):
    df_copy = df.copy()
    df_copy['Y'].replace('a', 1, inplace=True)
    df_copy['Y'].replace('b', 0, inplace=True)
    X = df_copy.iloc[:88, 0:12]
    y = df_copy.iloc[:88, 12:13]
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X)
    return X_train_scaled, df, y

data = pd.read_csv('old_Dataset.csv')
X, data, y = process_data(data)

# Define the function to classify a row
def classify_row(row):
    row = row.reshape(1, -1)
    prediction = model.predict(row)
    return 'Good' if prediction[0] >= 0.99 else 'Bad'

# Streamlit app
st.title('Bank Classification App')

# Let user select a row
row_index = st.number_input('Select row number (0 to 87):', min_value=0, max_value=len(data)-1, value=0)

# Display the selected row
selected_row = data.iloc[row_index]
st.write('Selected Row Data:')
st.write(selected_row)

# Classify the selected row
if st.button('Classify Selected Row'):
    processed_row = X[row_index]  # Get the corresponding processed row from X
    result = classify_row(processed_row)
    st.write(f'The model classifies this bank as: {result}')

st.write('---')

# User input for custom data
st.header('Classify Custom Data')
user_input = []

# Assuming there are 12 features

user_input.append(st.number_input(f'X1', value=0.0))
user_input.append(st.number_input(f'X2', value=0.0))
user_input.append(st.number_input(f'X3', value=0.0))
user_input.append(st.number_input(f'X4', value=0.0))
user_input.append(st.number_input(f'X5', value=0.0))
user_input.append(st.number_input(f'X6', value=0.0))
user_input.append(st.number_input(f'X7', value=0.0))
user_input.append(st.number_input(f'X8', value=0.0))
user_input.append(st.number_input(f'X9', value=0.0))
user_input.append(st.number_input(f'NOPAT', value=0.0))
user_input.append(st.number_input(f'WACC', value=0.0))
user_input.append(st.number_input(f'IC', value=0.0))


if st.button('Classify Custom Data'):
    user_input = np.array(user_input).reshape(1, -1)
    scaler = MinMaxScaler()
    user_input_scaled = scaler.fit_transform(user_input)
    result = classify_row(user_input_scaled[0])
    st.write(f'The model classifies this custom data as: {result}')
