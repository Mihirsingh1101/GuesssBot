document.addEventListener("DOMContentLoaded", () => {
    const chatIcon = document.getElementById('chat-icon');
    const chatWindow = document.getElementById('chat-window');
    const closeChat = document.getElementById('close-chat');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const chatBody = document.getElementById('chat-body');

    // Toggle chat window
    chatIcon.addEventListener('click', () => {
        chatWindow.classList.toggle('open');
    });

    closeChat.addEventListener('click', () => {
        chatWindow.classList.remove('open');
    });

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    function sendMessage() {
        const question = userInput.value.trim();
        if (question === "") return;

        addMessage(question, 'user');
        userInput.value = '';

        // ... inside your sendMessage function

        // --- Send question to the Python backend ---
        // Verify this URL is EXACTLY where your Python server is running.
        const apiUrl = 'http://127.0.0.1:8765/ask';

        fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question }),
        })
            .then(response => {
                if (!response.ok) {
                    // Throw an error if the server responded with a status like 400 or 500
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.answer) {
                    addMessage(data.answer, 'bot');
                } else if (data.error) {
                    // Display errors from the Flask app directly
                    addMessage(`Server Error: ${data.error}`, 'bot');
                } else {
                    addMessage('Sorry, I received an unexpected response.', 'bot');
                }
            })
            .catch(error => {
                // This block catches network errors and the error thrown above.
                console.error('Fetch Error:', error);
                addMessage(`Error: Could not connect to the chatbot. Please check the console (F12) for more details. Details: ${error.message}`, 'bot');
            });
    }

    function addMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', sender);
        messageDiv.textContent = text;
        chatBody.appendChild(messageDiv);
        // Scroll to the bottom
        chatBody.scrollTop = chatBody.scrollHeight;
    }
});