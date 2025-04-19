#please install - pip install Flask joblib scikit-learn neattext matplotlib

from flask import Flask, request, jsonify, render_template, send_file
import joblib
import os
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)

# Load the trained model
model = joblib.load("text_emotion.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    prediction = model.predict([text])[0]

    # Generate and save the graph
    fig, ax = plt.subplots()
    emotion_counts = model.predict([text])  # Replace with actual counts if available
    ax.bar(['Emotion'], [1], color='blue')
    ax.set_title('Emotion Prediction')
    ax.set_ylabel('Count')

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_url = '/static/emotion_graph.png'
    img_path = os.path.join('static', 'emotion_graph.png')
    with open(img_path, 'wb') as f:
        f.write(img.getbuffer())
    plt.close(fig)

    return jsonify({'emotion': prediction, 'img_url': img_url})

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_file(os.path.join('static', filename))

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
