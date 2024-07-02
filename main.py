import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import  accuracy_score

# Load your trained model

import keras
# Load your trained model
model = keras.models.load_model('EVAallvar.keras')
# Load your dataset
def proccess_data(df):
    df_copy=df
    df_copy['Y'].replace('a', 1, inplace=True)
    df_copy['Y'].replace('b', 0, inplace=True)


    X = df_copy.iloc[:88, 0:12]
    y=df_copy.iloc[:88,12:13]

    # # Apply SMOTE
    # smote = SMOTE(random_state=100)
    # X_resampled, y_resampled = smote.fit_resample(X, y)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X)
    return X_train_scaled,df,y

data = pd.read_csv('old_Dataset.csv')  # Make sure to replace 'your_dataset.csv' with your actual dataset file
X,data,y=proccess_data(data)
# Define the function to classify a row
def classify_row(row):
    # Assuming your model expects the input as a numpy array
    row = row.reshape(1, -1)

    prediction = model.predict(row)
    # return  prediction
    return 'Good' if prediction[0] >= 0.99 else 'Bad'


# Streamlit app
st.title('Bank Classification App')
y_pred_proba = model.predict(X)
y_pred = (y_pred_proba >= 0.99).astype(int)
accuracy = accuracy_score(y, y_pred)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
# Let user select a row
row_index = st.number_input('Select row number (0 to 87):', min_value=0, max_value=len(data)-1, value=0)

# Display the selected row
selected_row = data.iloc[row_index]
st.write('Selected Row Data:')
st.write(selected_row)

# Classify the selected row
if st.button('Classify'):
    processed_row = X[row_index]  # Get the corresponding processed row from X
    result = classify_row(
        processed_row
    )
    st.write(f'The model classifies this bank as: {result} ')

# Run the app using the command:
# streamlit run your_script_name.py
import streamlit as st
from sklearn.metrics import  accuracy_score
import pandas as pd

from imblearn.over_sampling import SMOTE
import keras
# def proccess_data(df):
#     df_copy=df
#     df_copy['Y'].replace('a', 1, inplace=True)
#     df_copy['Y'].replace('b', 0, inplace=True)
#
#
#     X = df_copy.iloc[:88, 0:12]
#     y=df_copy.iloc[:88,12:13]
#
#     # # Apply SMOTE
#     # smote = SMOTE(random_state=100)
#     # X_resampled, y_resampled = smote.fit_resample(X, y)
#     scaler = MinMaxScaler()
#     X_train_scaled = scaler.fit_transform(X)
#     return X_train_scaled,y
#
# model = keras.models.load_model('EVAallvar.keras')
#
# # Load your dataset
# data = pd.read_csv('old_Dataset.csv')  # Make sure to replace 'old_Dataset.csv' with your actual dataset file
# X, y= proccess_data(data)
# predictions = pd.read_csv('predictions.csv')
#
#
# # Streamlit app
# st.title('Bank Classification App')
#
# y_pred=[]
# # Calculate accuracy
# for index,row in predictions.iterrows():
#     y_pred.append(row[1])
# accuracy = accuracy_score(y, y_pred) * 100
#
# # Display model accuracy
# st.write(f'**Model Accuracy: {accuracy:.2f}%**')
#
#
# # Display the whole dataset with original performance and model predictions
# st.write('Dataset with Model Predictions:')
#
# # Prepare the results dataframe
# results = []
#
#
# for index,row in y.iterrows():
#
#
#     results.append({
#         'Original Performance': row[0],
#         'Model Prediction': y_pred[index],
#         'Row Data': X[index]
#     })
#
# results_df = pd.DataFrame(results)
#
# # Display the results
# st.write(results_df[['Row Data','Original Performance', 'Model Prediction' ]])
#
# # Run the app using the command:
# # streamlit run your_script_name.py
