from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import io
import base64


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and columns
model = pickle.load(open('model/churn_model.pkl', 'rb'))
columns = json.load(open('model/columns.json'))


@app.route('/predict', methods=['POST'])
def predict():
    # Check if file is uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Invalid file type. Please upload a CSV file'}), 400

    # Read the CSV file
    try:
        data = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'Error reading file: {str(e)}'}), 400

    # Validate required columns
    required_columns = columns['columns']
    if not set(required_columns).issubset(data.columns):
        return jsonify({'error': f'Missing required columns: {set(required_columns) - set(data.columns)}'}), 400

    # Process the data
    input_data = data[required_columns]
    predictions = model.predict(input_data)

    # Add predictions and reasons
    results = []
    for i, pred in enumerate(predictions):
        reason = ", ".join([f"{col}: {data.iloc[i][col]}" for col in required_columns])
        if pred == 1:
            results.append({
                "prediction": "Churn",
                "reason": f"It is churn because of the following factors: {reason}"
            })
        else:
            results.append({
                "prediction": "No Churn",
                "reason": f"It is no churn because of the following factors: {reason}"
            })

    # Add churn predictions to data for visualization
    data['Churn Prediction'] = ['Churn' if pred == 1 else 'No Churn' for pred in predictions]

    # Create a bar graph with percentages
    churn_counts = data['Churn Prediction'].value_counts(normalize=True) * 100
    colors = ['red' if label == 'Churn' else 'blue' for label in churn_counts.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    churn_counts.plot(kind='bar', color=colors, ax=ax)
    ax.set_title('Churn Predictions (Percentage)')
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Prediction')
    for i, v in enumerate(churn_counts):
        ax.text(i, v + 1, f"{v:.2f}%", ha='center', fontsize=12)

    # Convert bar plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    bar_graph = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    # Create a line graph (cumulative churn predictions)
    data['Cumulative Churn'] = np.cumsum(data['Churn Prediction'] == 'Churn')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data.index, data['Cumulative Churn'], marker='o', color='purple')
    ax.set_title('Cumulative Churn Predictions')
    ax.set_ylabel('Cumulative Count')
    ax.set_xlabel('Index')

    # Convert line plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    line_graph = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    return jsonify({
        'results': results,
        'bar_graph': f"data:image/png;base64,{bar_graph}",
        'line_graph': f"data:image/png;base64,{line_graph}"
    })



if __name__ == '__main__':
    app.run(debug=True)
