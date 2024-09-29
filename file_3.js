// Function to send a message to the chatbot
async function sendMessage() {
    const userInput = document.getElementById('user-input').value;
    if (!userInput) return;

    // Display the user's message
    displayMessage(userInput, 'user');

    // Clear the input field
    document.getElementById('user-input').value = '';

    // Send the message to the server (API endpoint)
    const response = await fetch('https://api.yourchatbot.com/message', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userInput }),
    });

    const data = await response.json();

    // Display the bot's response
    displayMessage(data.reply, 'bot');
}

// Function to display messages in the chat
function displayMessage(message, sender) {
    const chatWindow = document.getElementById('chat-window');
    const messageElement = document.createElement('div');
    messageElement.className = 'message ' + (sender === 'user' ? 'user-message' : 'bot-message');
    messageElement.textContent = message;
    chatWindow.appendChild(messageElement);
}

// Add event listener to the send button
document.getElementById('send-button').addEventListener('click', sendMessage);
