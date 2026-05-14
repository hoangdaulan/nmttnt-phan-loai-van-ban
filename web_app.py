import json
import math
import re
import os

from flask import Flask, render_template, request, jsonify
from underthesea import word_tokenize

app = Flask(__name__, template_folder='templates', static_folder='static')

# ===== Reuse logic from main.py =====

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = word_tokenize(text, format="text")
    return text.split()


def load_model(file_path):
    """Đọc file JSON và trả về 3 biến mô hình"""
    with open(file_path, 'r', encoding='utf-8') as f:
        model_data = json.load(f)

    vocab = set(model_data['vocab'])
    log_priors = model_data['log_priors']
    log_likelihoods = model_data['log_likelihoods']

    print(f"📂 Đã tải mô hình thành công từ: {file_path}")
    return vocab, log_priors, log_likelihoods


def predict_naive_bayes(text_tokens, log_priors, log_likelihoods, vocab):
    scores = {}
    for class_label in log_priors:
        score = log_priors[class_label]
        for word in text_tokens:
            if word in vocab:
                score += log_likelihoods[class_label][word]
        scores[class_label] = score
    return max(scores, key=scores.get), scores


# ===== Load model on startup =====
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spam_model.json')
vocab, log_priors, log_likelihoods = load_model(MODEL_PATH)


# ===== Routes =====

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'Vui lòng nhập nội dung email'}), 400

    tokens = clean_text(text)
    prediction, scores = predict_naive_bayes(tokens, log_priors, log_likelihoods, vocab)

    # Convert log scores to probabilities using softmax
    max_score = max(scores.values())
    exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
    total = sum(exp_scores.values())
    probabilities = {k: round(v / total * 100, 2) for k, v in exp_scores.items()}

    return jsonify({
        'prediction': prediction,
        'probabilities': probabilities,
        'tokens_count': len(tokens),
        'vocab_matches': sum(1 for t in tokens if t in vocab),
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
