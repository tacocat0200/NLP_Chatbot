import json
import logging
from flask import Flask, request, jsonify
from transformers import AutoModelForTokenClassification, AutoTokenizer
from sentiment_analysis import analyze_sentiment
from topic_modeling import get_topics

# Load configuration
with open('config.json') as config_file:
    config = json.load(config_file)

# Set up logging
logging.basicConfig(level=config['logging']['level'], filename=config['logging']['log_file'])
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer
model_name = config['model']['name']
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    logger.info(f'User input: {user_input}')

    # Process the input
    tokens = tokenizer(user_input, return_tensors='pt')
    outputs = model(**tokens)
    
    # Extract predictions (this will depend on your specific model)
    predictions = outputs.logits.argmax(dim=1).tolist()
    
    # Analyze sentiment
    sentiment = analyze_sentiment(user_input)
    response_message = generate_response(user_input, predictions, sentiment)

    return jsonify({'response': response_message, 'sentiment': sentiment})

def generate_response(user_input, predictions, sentiment):
    # Basic logic for generating a response based on the input and predictions
    if sentiment['label'] == 'negative':
        return "I'm sorry to hear that. How can I assist you further?"
    elif sentiment['label'] == 'positive':
        return "I'm glad to hear that! How can I help you?"
    else:
        return "I understand. What else would you like to discuss?"

@app.route('/topics', methods=['GET'])
def topics():
    user_input = request.args.get('input')
    topics = get_topics(user_input)
    return jsonify({'topics': topics})

if __name__ == '__main__':
    app.run(host=config['server']['host'], port=config['server']['port'])
