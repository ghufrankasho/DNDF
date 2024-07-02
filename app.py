from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load your trained model


model = keras.models.load_model('EVAallvar.keras')

# Load and process your dataset
def process_data(df):
    df_copy = df.copy()
    df_copy['Y'].replace('a', 1, inplace=True)
    df_copy['Y'].replace('b', 0, inplace=True)
    X = df_copy.iloc[:88, 0:12]
    y = df_copy.iloc[:88, 12:13]
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X)
    return X_train_scaled, df, y, scaler

data = pd.read_csv('old_Dataset.csv')
X, data, y, scaler = process_data(data)

# Define the function to classify a row
def classify_row(row):
    row = row.reshape(1, -1)
    prediction = model.predict(row)
    return 'Good' if prediction[0] >= 0.99 else 'Bad'

@app.route('/')
def index():
    return render_template('index.html', row_data=data.to_dict(orient='records'))

@app.route('/classify', methods=['POST'])
def classify():
    row_index = int(request.form['row_index'])
    selected_row = data.iloc[row_index]
    processed_row = X[row_index]
    result = classify_row(processed_row)
    return render_template('index.html', row_data=data.to_dict(orient='records'), selected_row=selected_row.to_dict(), classification_result=result)

@app.route('/classify_custom', methods=['POST'])
def classify_custom():
    user_input = [float(request.form[f'feature_{i}']) for i in range(12)]
    user_input = np.array(user_input).reshape(1, -1)
    user_input_scaled = scaler.transform(user_input)
    result = classify_row(user_input_scaled[0])
    return render_template('index.html', row_data=data.to_dict(orient='records'), custom_classification_result=result)

if __name__ == '__main__':
    app.run(debug=True)
