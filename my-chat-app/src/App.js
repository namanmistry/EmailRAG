import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const dummySummaries = [
  { subject: 'Security Alert', summary: 'New sign-in detected from Chrome on Windows.' },
  { subject: 'Subscription Update', summary: 'Your subscription to X Service has been renewed.' },
  { subject: 'Event Invitation', summary: 'You are invited to the annual tech meetup.' },
];

function stripHtml(html) {
  return html;
}

function App() {
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const [input, setInput] = useState('');
  const [sourcesMap, setSourcesMap] = useState({});
  const [viewMode, setViewMode] = useState('chat');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [apiKey, setApiKey] = useState('');
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, sourcesMap]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = { text: input, sender: 'user' };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    try {
      const res = await fetch(`/api/query?question=${encodeURIComponent(input)}`);
      const data = await res.json();
      const reply = data.error ? `❌ ${data.error}` : data.answer;

      const newAssistantMessage = { text: reply, sender: 'assistant', id: Date.now() };
      setMessages((prev) => [...prev, newAssistantMessage]);

      if (data.sources?.length) {
        setSourcesMap((prev) => ({
          ...prev,
          [newAssistantMessage.id]: { visible: false, sources: data.sources },
        }));
      }
    } catch (error) {
      setMessages((prev) => [...prev, { text: '❌ Failed to fetch response.', sender: 'assistant' }]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleIndexing = () => {
    console.log('Starting indexing with', { email, password, apiKey });
    // Replace with your indexing logic
  };

  const toggleSources = (id) => {
    setSourcesMap((prev) => ({
      ...prev,
      [id]: { ...prev[id], visible: !prev[id].visible },
    }));
  };

  return (
    <div style={styles.page}>
      {/* Sidebar */}
      <div style={styles.sidebar}>
        {/* Input Fields */}
        <div style={styles.sectionTitle}>Setup Email</div>
        <input
          type="text"
          placeholder="Email ID"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          style={styles.inputField}
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          style={styles.inputField}
        />
        <input
          type="text"
          placeholder="NVIDIA NIMs API Key"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          style={styles.inputField}
        />
        <button style={styles.navButton} onClick={handleIndexing}>Start Indexing</button>

        {/* New Chat Button */}
        <button style={styles.navButton}>+ New Chat</button>

        {/* Footer Toggle for Email Summaries */}
        <div style={{ marginTop: 'auto', paddingTop: '10px', borderTop: '1px solid #2d2f32' }}>
          {viewMode === 'chat' ? (
            <button style={styles.footerButton} onClick={() => setViewMode('summary')}>
              📝 Email Summaries
            </button>
          ) : (
            <button style={styles.footerButton} onClick={() => setViewMode('chat')}>
              💬 Back to Assistant
            </button>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div style={styles.mainContent}>
        {viewMode === 'chat' ? (
          <>
            <h1 style={styles.title}>📧 Email Query Assistant</h1>
            <div style={styles.chatContainer}>
              {messages.map((msg, idx) => (
                <div key={idx} style={{
                  ...styles.messageRow,
                  justifyContent: msg.sender === 'user' ? 'flex-end' : 'flex-start'
                }}>
                  <div style={msg.sender === 'user' ? styles.userBubble : styles.assistantBubble}>
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {stripHtml(msg.text)}
                    </ReactMarkdown>
                    {msg.sender === 'assistant' && sourcesMap[msg.id] && (
                      <div style={{ marginTop: '10px' }}>
                        <button style={styles.collapseButton} onClick={() => toggleSources(msg.id)}>
                          {sourcesMap[msg.id].visible ? 'Hide Sources' : 'View Sources'}
                        </button>
                        {sourcesMap[msg.id].visible && (
                          <div style={styles.sourcesBox}>
                            {sourcesMap[msg.id].sources.map((src, srcIdx) => (
                              <div key={srcIdx} style={styles.sourceItem}>
                                <div><strong>Subject:</strong> {src.subject}</div>
                                <div><strong>From:</strong> {src.from_email_id}</div>
                                <div><strong>Sent Date:</strong> {src.sent_date}</div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {isTyping && (
                <div style={{ ...styles.messageRow, justifyContent: 'flex-start' }}>
                  <div style={styles.typingIndicator}>Assistant is typing...</div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>
            <div style={styles.inputContainer}>
              <textarea
                style={styles.inputBox}
                placeholder="Type your message..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                rows={1}
              />
              <button style={styles.sendButton} onClick={handleSend}>➤</button>
            </div>
          </>
        ) : (
          <div style={styles.summaryContainer}>
            <h1 style={styles.title}>📝 Email Summaries</h1>
            {dummySummaries.map((item, idx) => (
              <div key={idx} style={styles.summaryBox}>
                <strong>{item.subject}</strong>: {item.summary}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

const styles = {
  page: { display: 'flex', height: '100vh', backgroundColor: '#202123', color: '#d1d5db' },
  sidebar: { width: '200px', backgroundColor: '#343541', display: 'flex', flexDirection: 'column', padding: '10px', borderRight: '1px solid #2d2f32' },
  inputField: { marginBottom: '8px', padding: '8px', borderRadius: '4px', border: '1px solid #2d2f32', backgroundColor: '#202123', color: '#d1d5db' },
  navButton: { padding: '10px', marginBottom: '10px', borderRadius: '5px', border: 'none', backgroundColor: '#0b93f6', color: '#fff', cursor: 'pointer' },
  footerButton: { width: '100%', textAlign: 'left', backgroundColor: 'transparent', color: '#d1d5db', border: 'none', padding: '10px 0', cursor: 'pointer', fontSize: '14px', opacity: 0.8, transition: 'opacity 0.2s', borderRadius: '4px', paddingLeft: '10px' },
  mainContent: { flex: 1, display: 'flex', flexDirection: 'column' },
  title: { color: 'white', textAlign: 'center', padding: '20px 0', borderBottom: '1px solid #2d2f32' },
  chatContainer: { flex: 1, overflowY: 'auto', padding: '20px', display: 'flex', flexDirection: 'column' },
  messageRow: { display: 'flex', marginBottom: '10px' },
  userBubble: { backgroundColor: '#343541', color: '#e0e0e0', borderRadius: '8px', padding: '12px', maxWidth: '75%' },
  assistantBubble: { backgroundColor: '#444654', color: '#e0e0e0', borderRadius: '8px', padding: '12px', maxWidth: '75%' },
  typingIndicator: { backgroundColor: '#444654', color: '#a1a1aa', fontStyle: 'italic', borderRadius: '8px', padding: '12px', maxWidth: '75%' },
  inputContainer: { display: 'flex', borderTop: '1px solid #2d2f32', padding: '10px', backgroundColor: '#343541' },
  inputBox: { flex: 1, resize: 'none', backgroundColor: '#202123', color: '#fff', border: 'none', borderRadius: '5px', padding: '10px', outline: 'none' },
  sendButton: { marginLeft: '10px', padding: '10px 15px', borderRadius: '5px', border: 'none', backgroundColor: '#0b93f6', color: '#fff', cursor: 'pointer' },
  collapseButton: { marginTop: '10px', padding: '5px 10px', borderRadius: '5px', border: 'none', backgroundColor: '#0b93f6', color: '#fff', cursor: 'pointer' },
  sourcesBox: { marginTop: '10px', backgroundColor: '#2d2f32', borderRadius: '8px', padding: '10px', color: '#e0e0e0' },
  sourceItem: { padding: '8px', borderBottom: '1px solid #3a3b3c', textAlign: 'left' },
  summaryContainer: { flex: 1, padding: '20px', overflowY: 'auto' },
  summaryBox: { backgroundColor: '#343541', borderRadius: '8px', padding: '12px', marginBottom: '10px', color: '#e0e0e0' },
  sectionTitle: {
  color: '#d1d5db',
  fontWeight: 'bold',
  fontSize: '14px',
  marginBottom: '5px',
  marginTop: '5px',
},

};

export default App;
