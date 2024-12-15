from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Načteme model a tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    
    # Tokenizace textu
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    # Predikce
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    
    # Mapování na labely
    sentiment = ["Negative", "Neutral", "Positive"]
    return jsonify({'sentiment': sentiment[prediction]})

if __name__ == '__main__':
    app.run(debug=True)
