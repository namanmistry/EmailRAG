
document.addEventListener('DOMContentLoaded', function() {
    // Initialize markdown-it with options
    const md = window.markdownit({
        html: false,
        xhtmlOut: false,
        breaks: true,
        langPrefix: 'language-',
        linkify: true,
        typographer: true
    });
    const sendButton = document.getElementById('send-button');
    const userInput = document.getElementById('user-input');
    const chatBox = document.getElementById('chat-box');
    const chatList = document.getElementById('chat-list');

    let currentChatId = null;

    marked.setOptions({
        breaks: true,
        gfm: true,
        headerIds: false,
        mangle: false,
        sanitize: false,
        smartypants: true,
        xhtml: true
    });

    // Initialize by creating a new chat
    startNewChat();

    sendButton.addEventListener('click', function() {
        sendMessage();
    });

    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // Load all chats and show in sidebar
    function loadChatHistoryList() {
        fetch('/api/history/' + currentChatId)
            .then(response => response.json())
            .then(data => {
                chatList.innerHTML = '';  // Clear existing list
                const button = document.createElement('button');
                button.textContent = `Chat ${currentChatId.substring(0, 8)}`;
                button.addEventListener('click', () => loadChatById(currentChatId));
                chatList.appendChild(button);
            });
    }

    // Start a new chat session
    function startNewChat() {
        fetch('/api/new_chat')
            .then(response => response.json())
            .then(data => {
                currentChatId = data.chat_id;
                loadChatHistoryList();
                clearChatBox();
            });
    }

    // Load an existing chat by ID
    function loadChatById(chatId) {
        currentChatId = chatId;
        fetch('/api/history/' + chatId)
            .then(response => response.json())
            .then(messages => {
                clearChatBox();
                messages.forEach(msg => {
                    if (msg.role === 'user') {
                        appendUserMessage(msg.message);
                    } else if (msg.role === 'assistant') {
                        appendBotMessage(msg.message);
                    }
                });
            });
    }

    function sendMessage() {
        const question = userInput.value.trim();
        if (question === '') return;

        appendUserMessage(question);
        userInput.value = '';

        const botMessageDiv = document.createElement('div');
        botMessageDiv.className = 'bot-message';

        const markdownDiv = document.createElement('div');
        markdownDiv.className = 'markdown-content';
        botMessageDiv.appendChild(markdownDiv);

        const loadingSpinner = document.createElement('span');
        loadingSpinner.className = 'spinner';
        botMessageDiv.appendChild(loadingSpinner);

        chatBox.appendChild(botMessageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;

        let fullMessage = '';
        let responseStarted = false;
        const eventSource = new EventSource(`/api/query/${currentChatId}?question=${encodeURIComponent(question)}`);


eventSource.onmessage = function(event) {
    if (event.data === "[DONE]") {
        eventSource.close();
        if (loadingSpinner.parentNode) {
            botMessageDiv.removeChild(loadingSpinner);
        }
        renderMarkdownSafely(markdownDiv, fullMessage);  // Final render with complete message
        return;
    }

    // Accumulate the message
    fullMessage += event.data;
    
    // Render incrementally so user sees formatted content as it arrives
    renderMarkdownSafely(markdownDiv, fullMessage);
};


        eventSource.onerror = function() {
            console.error("Error with EventSource connection");
            if (loadingSpinner.parentNode) {
                botMessageDiv.removeChild(loadingSpinner);
            }
            renderMarkdownSafely(markdownDiv, '❌ An error occurred.');
            eventSource.close();
        };
    }

function renderMarkdownSafely(element, markdown) {
        try {
            const processedMarkdown = preprocessMarkdown(markdown);
            
            // Use markdown-it instead of marked
            element.innerHTML = md.render(processedMarkdown);
            
            // Apply code highlighting
            element.querySelectorAll('pre code').forEach((block) => {
                if (window.hljs) {
                    window.hljs.highlightBlock(block);
                }
            });
        } catch (error) {
            console.error("Error rendering markdown:", error);
            element.innerHTML = markdown.replace(/\n/g, '<br>');
        }
        chatBox.scrollTop = chatBox.scrollHeight;
    }


    function appendUserMessage(text) {
        const userMessageDiv = document.createElement('div');
        userMessageDiv.className = 'user-message';
        userMessageDiv.textContent = text;
        chatBox.appendChild(userMessageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function appendBotMessage(markdown) {
        const botMessageDiv = document.createElement('div');
        botMessageDiv.className = 'bot-message';
        const markdownDiv = document.createElement('div');
        markdownDiv.className = 'markdown-content';
        botMessageDiv.appendChild(markdownDiv);
        chatBox.appendChild(botMessageDiv);
        renderMarkdownSafely(markdownDiv, markdown);
    }
    const newChatButton = document.getElementById('new-chat-button');
    newChatButton.addEventListener('click', () => {
        startNewChat();
    });

    function clearChatBox() {
        chatBox.innerHTML = '';
    }

function preprocessMarkdown(text) {
    if (!text) return '';
    
    // More comprehensive preprocessing for better markdown formatting
    return text
        // Fix numbered lists - ensure they have proper spacing before them
        .replace(/(\n)(\d+\.\s)/g, '$1\n$2')
        
        // Fix bullet lists - ensure they have proper spacing before them
        .replace(/(\n)([*+-]\s)/g, '$1\n$2')
        
        // Ensure headers have proper spacing before them
        .replace(/(\n)(#{1,6}\s)/g, '$1\n$2')
        
        // Fix code blocks to ensure proper formatting
        .replace(/```([a-z]*)\n/gi, '\n```$1\n')
        .replace(/\n```/g, '\n\n```')
        
        // Ensure paragraphs have proper line spacing 
        .replace(/\.\s+([A-Z])/g, '.\n\n$1')
        
        // Make sure list items are properly separated
        .replace(/(\d+\.\s.+)(\n)(\d+\.\s)/g, '$1\n$3')
        .replace(/([*+-]\s.+)(\n)([*+-]\s)/g, '$1\n$3');
}
});
