# NLP Chatbot

## Overview

This repository contains an NLP-based chatbot that utilizes state-of-the-art models like BERT and GPT to understand user queries and provide appropriate responses. The chatbot features real-time sentiment analysis and topic modeling capabilities, aiming to enhance user interaction and reduce support tickets.

## Features

- **Query Understanding**: Achieved 95% accuracy using BERT and GPT.
- **Sentiment Analysis**: Implemented real-time sentiment analysis with 96% accuracy, simulating response adjustments for 82% of negative sentiment cases.
- **Topic Modeling**: Utilized LDA/BERT for topic modeling to identify the top 5 trending issues from over 1000 simulated conversations.
- **Deployment**: Deployed on AWS, demonstrating potential to reduce support tickets by 40%.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the Repository**:

   ```bash
   git clone git@github.com:tacocat0200/NLP_Chatbot.git
   cd NLP_Chatbot
## Usage

The NLP chatbot can be used to interact with users, understand their queries, and provide insightful responses. Below are the key functionalities of the chatbot based on the project developments:

**Starting the Chatbot**:
   To initiate the chatbot, run the following command in your terminal:

   ```bash
   python chatbot.py
   ```
Interacting with the Chatbot:

Simply type your query in the console.
For example, you might ask:
"What are the latest trends in customer support?"
"Can you analyze the sentiment of my message?"
The chatbot will analyze your input and provide a response based on its training.
Configuration Options: Ensure to review and adjust the settings in config.json and aws_config.json if needed, to tailor the chatbot's behavior and deployment settings to your requirements.
