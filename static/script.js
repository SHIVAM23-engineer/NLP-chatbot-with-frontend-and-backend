function toggleChat() {
    var chatContainer = document.getElementById('chatContainer');
    var showChatBtn = document.querySelector('.show-chat-btn');

    if (chatContainer.style.display === 'none' || chatContainer.style.display === '') {
        chatContainer.style.display = 'block';
        showChatBtn.style.display = 'none';
    } else {
        chatContainer.style.display = 'none';
        showChatBtn.style.display = 'block';
    }
}
