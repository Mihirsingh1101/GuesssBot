from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
from waitress import serve
import RAG_Solution_FIXED as rag # Import your existing RAG code

app = Flask(__name__)
CORS(app)

# --- Load your RAG model components ---
print("--- RAG Model Initializing ---")

# Define the data directory
data_dir = Path("./data")

# 1. Load text files
print("[1/4] Loading text files...")
texts = rag.load_text_files(data_dir)
print("      ...Done.")

# 2. Build chunks
print("[2/4] Building text chunks...")
docs = rag.build_chunks(texts)
print("      ...Done.")

# 3. Build vector store (This can be slow)
print("[3/4] Building vector store with embeddings. This may take a moment...")
vs = rag.build_vectorstore(docs)
print("      ...Done.")

# 4. Load the generative model (This is the slowest part)
print("[4/4] Loading generative model (e.g., Flan-T5). This can be very slow...")
generator = rag.make_generator()
print("      ...Done.")

print("--- Model loading complete. Server is now starting. ---")
# ------------------------------------

@app.route('/ask', methods=['POST'])
def ask_question():
    """ This is the API endpoint that will receive questions from the frontend. """
    # ... (rest of your function is the same)
    if not request.json or 'question' not in request.json:
        return jsonify({'error': 'Missing question in request body'}), 400
    question = request.json['question']
    try:
        answer, _, _ = rag.answer_question(vs, generator, question)
        return jsonify({'answer': answer.strip()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # This line will only run AFTER all the models above are loaded
    serve(app, host="127.0.0.1", port=8765)
    print("Server has started on http://127.0.0.1:3000 and is ready for requests.")