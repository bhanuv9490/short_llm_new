document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const chatMessages = document.getElementById('chatMessages');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const settingsButton = document.getElementById('settingsButton');
    const closeSettings = document.getElementById('closeSettings');
    const settingsPanel = document.getElementById('settingsPanel');
    const modelOptions = document.querySelectorAll('.model-option');
    
    // Settings elements
    const maxLengthSlider = document.getElementById('maxLength');
    const temperatureSlider = document.getElementById('temperature');
    const topPSlider = document.getElementById('topP');
    const maxLengthValue = document.getElementById('maxLengthValue');
    const temperatureValue = document.getElementById('temperatureValue');
    const topPValue = document.getElementById('topPValue');

    // State
    let currentModel = 'phi2';
    let isGenerating = false;

    // Initialize event listeners
    function init() {
        // Send message on button click or Enter key (with Shift+Enter for new line)
        sendButton.addEventListener('click', handleSendMessage);
        userInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage();
            }
        });

        // Settings panel
        settingsButton.addEventListener('click', () => settingsPanel.classList.add('open'));
        closeSettings.addEventListener('click', () => settingsPanel.classList.remove('open'));

        // Model selection
        modelOptions.forEach(option => {
            option.addEventListener('click', () => {
                modelOptions.forEach(opt => opt.classList.remove('active'));
                option.classList.add('active');
                currentModel = option.dataset.model;
                addSystemMessage(`Switched to ${currentModel === 'phi2' ? 'Microsoft Phi-2' : 'GPT-2'}`);
            });
        });

        // Update slider value displays
        maxLengthSlider.addEventListener('input', (e) => {
            maxLengthValue.textContent = e.target.value;
        });

        temperatureSlider.addEventListener('input', (e) => {
            temperatureValue.textContent = e.target.value;
        });

        topPSlider.addEventListener('input', (e) => {
            topPValue.textContent = e.target.value;
        });
    }


    // Handle sending a message
    async function handleSendMessage() {
        const message = userInput.value.trim();
        if (!message || isGenerating) return;

        // Add user message to chat
        addMessage('user', message);
        userInput.value = '';
        autoResize(userInput);

        // Show typing indicator
        const typingId = showTypingIndicator();

        try {
            isGenerating = true;
            sendButton.disabled = true;
            
            // Get generation parameters
            const params = {
                prompt: message,
                model: currentModel,
                max_length: parseInt(maxLengthSlider.value),
                temperature: parseFloat(temperatureSlider.value),
                top_p: parseFloat(topPSlider.value)
            };

            // Call the API
            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(params)
            });

            if (!response.ok) {
                throw new Error('Failed to generate response');
            }

            const data = await response.json();
            
            // Remove typing indicator and add assistant's response
            removeTypingIndicator(typingId);
            addMessage('assistant', data.response);
            
        } catch (error) {
            console.error('Error:', error);
            removeTypingIndicator(typingId);
            addMessage('assistant', 'Sorry, I encountered an error. Please try again.');
        } finally {
            isGenerating = false;
            sendButton.disabled = false;
        }
    }

    // Add a message to the chat
    function addMessage(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        messageDiv.textContent = content;
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    // Add a system message
    function addSystemMessage(content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message system';
        messageDiv.textContent = content;
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    // Show typing indicator
    function showTypingIndicator() {
        const id = 'typing-' + Date.now();
        const typingDiv = document.createElement('div');
        typingDiv.id = id;
        typingDiv.className = 'message assistant typing';
        typingDiv.innerHTML = `
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        chatMessages.appendChild(typingDiv);
        scrollToBottom();
        return id;
    }

    // Remove typing indicator
    function removeTypingIndicator(id) {
        const typingElement = document.getElementById(id);
        if (typingElement) {
            typingElement.remove();
        }
    }

    // Auto-resize textarea
    function autoResize(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = textarea.scrollHeight + 'px';
    }

    // Scroll to bottom of chat
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Initialize the chat
    init();
});
