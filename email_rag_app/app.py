from flask import Flask, request, Response, stream_with_context
from models import emailrag_engine  # Your AI engine
from flask import request, jsonify
from models.indexing import Indexing
from langchain_community.embeddings import HuggingFaceEmbeddings
from flask_cors import CORS
from models.email_summarizer import EmailSummarizer
app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})
nvidia_api_key = ""
print("Starting Flask app...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
@app.route('/api/query')
def query():
    question = request.args.get('question', '')
    if not question:
        return jsonify({"error": "No question provided."}), 400

    # Call your engine and return the result as plain text or JSON
    email_rag_engine = emailrag_engine.EmailRAGEngine(embedding_model=embedding_model)
    answer, sources = email_rag_engine.process_query(question, api_key=nvidia_api_key)
    return jsonify({"answer": answer, "sources": sources})


@app.route('/api/index', methods=['GET'])
def index_emails():
    global nvidia_api_key
    email_id = request.args.get('email')
    password = request.args.get('password')
    nvidia_api_key = request.args.get('nvidia_api_key')
    print("route called")
    def generate():
        try:
            yield 'data: collecting\n\n'
            indexer = Indexing()
            indexer.login_email(email_id, password)
            emails = indexer.fetch_emails()

            yield 'data: storing\n\n'
            chunked = indexer.chunk_email(emails)

            yield 'data: indexing\n\n'
            indexer.create_vector_store(chunked, embedding_model)

            yield 'data: done\n\n'
        except Exception as e:
            yield f'data: error: {str(e)}\n\n'

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/api/summarize', methods=['POST'])
def summarize_emails():
    data = request.get_json()
    email_id = data.get('email')
    password = data.get('password')

    if not email_id or not password:
        return jsonify({"error": "Missing email or password."}), 400

    try:
        summarizer = EmailSummarizer(email_id, password)
        results = summarizer.process_uncategorized_emails(nvidia_api_key=nvidia_api_key)
        print("Summarization results:", results)
        return jsonify({"status": "success", "processed": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    print("Flask app starting at http://127.0.0.1:5000")
    app.run(debug=True, threaded=True)
