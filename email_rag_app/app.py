# from flask import Flask, request, jsonify, render_template, Response, stream_with_context
# import sqlite3
# import traceback
# import sys
# import uuid
# from datetime import datetime

# app = Flask(__name__)
# print("Starting Flask app...")
# from models import emailrag_engine


# # ----------------- Database Setup -----------------
# DB_FILE = 'chat_history.db'

# def init_db():
#     with sqlite3.connect(DB_FILE) as conn:
#         cursor = conn.cursor()
#         cursor.execute('''
#             CREATE TABLE IF NOT EXISTS chats (
#                 chat_id TEXT,
#                 role TEXT,
#                 message TEXT,
#                 timestamp TEXT
#             )
#         ''')
#         conn.commit()

# init_db()

# def save_message(chat_id, role, message):
#     with sqlite3.connect(DB_FILE) as conn:
#         cursor = conn.cursor()
#         cursor.execute('INSERT INTO chats (chat_id, role, message, timestamp) VALUES (?, ?, ?, ?)',
#                        (chat_id, role, message, datetime.utcnow().isoformat()))
#         conn.commit()

# def get_chat_history(chat_id):
#     with sqlite3.connect(DB_FILE) as conn:
#         cursor = conn.cursor()
#         cursor.execute('SELECT role, message FROM chats WHERE chat_id = ? ORDER BY timestamp', (chat_id,))
#         return cursor.fetchall()

# # ----------------- Flask Routes -----------------

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/api/history/<chat_id>')
# def chat_history(chat_id):
#     messages = get_chat_history(chat_id)
#     return jsonify([{'role': role, 'message': message} for role, message in messages])

# @app.route('/api/new_chat')
# def new_chat():
#     new_chat_id = str(uuid.uuid4())
#     return jsonify({'chat_id': new_chat_id})

# @app.route('/api/query/<chat_id>')
# def query(chat_id):
#     question = request.args.get('question', '')
#     if not question:
#         return jsonify({'error': 'No question provided.'}), 400

#     # Save user message
#     save_message(chat_id, 'user', question)

#     ai_response_collector = []

#     def generate():
#         for token in emailrag_engine.process_query(question):
#             ai_response_collector.append(token)
#             yield f"data: {token}\n\n"
#         # Save the complete AI response after streaming is done
#         ai_response = ''.join(ai_response_collector)
#         save_message(chat_id, 'assistant', ai_response)
#         yield "data: [DONE]\n\n"

#     response = Response(stream_with_context(generate()), mimetype='text/event-stream')
#     response.headers['Cache-Control'] = 'no-cache'
#     response.headers['X-Accel-Buffering'] = 'no'
#     response.headers['Connection'] = 'keep-alive'
#     return response

# if __name__ == '__main__':
#     print("Flask app starting at http://127.0.0.1:5000")
#     app.run(debug=True, threaded=True)
from flask import Flask, request, Response, stream_with_context
from models import emailrag_engine  # Your AI engine
from flask import request, jsonify
app = Flask(__name__)
print("Starting Flask app...")

@app.route('/api/query')
def query():
    question = request.args.get('question', '')
    if not question:
        return jsonify({"error": "No question provided."}), 400

    # Call your engine and return the result as plain text or JSON
    answer, sources = emailrag_engine.process_query(question)
    return jsonify({"answer": answer, "sources": sources})

if __name__ == '__main__':
    print("Flask app starting at http://127.0.0.1:5000")
    app.run(debug=True, threaded=True)
