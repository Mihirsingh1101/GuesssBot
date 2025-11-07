import argparse
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GEN_MODEL = "google/flan-t5-base"

def load_text_files(data_dir: Path, files: List[Path] = None) -> List[str]:
    texts = []
    if files:
        paths = files
    else:
        paths = sorted(Path(data_dir).glob("*.txt"))
    for p in paths:
        if p.exists() and p.is_file():
            texts.append(p.read_text(encoding="utf-8", errors="ignore"))
    if not texts:
        raise FileNotFoundError(f"No text files found in {Path(data_dir).resolve()} or provided via --files")
    return texts

def build_chunks(texts: List[str], chunk_size: int = 800, chunk_overlap: int = 120):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = splitter.create_documents(texts)
    return docs

def build_vectorstore(docs):
    embeddings = SentenceTransformerEmbeddings(model_name=DEFAULT_EMBED_MODEL)
    vs = FAISS.from_documents(docs, embeddings)
    return vs

def make_generator(model_name: str = DEFAULT_GEN_MODEL, device: int = -1):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    gen = pipeline("text2text-generation", model=mdl, tokenizer=tok, device=device)
    return gen

def format_prompt(question: str, contexts):
    context_block = "\n\n".join([d.page_content for d in contexts])
    return (
        "You are an expert assistant. Use ONLY the context to answer.\n"
        "If the answer can't be found in the context, say you don't know.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

def answer_question(vs, generator, question: str, k: int = 4, max_new_tokens: int = 256):
    contexts = vs.similarity_search(question, k=k)
    prompt = format_prompt(question, contexts)
    out = generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    text = out[0]["generated_text"]
    sources = [d.metadata.get("source", "") for d in contexts]
    return text, contexts, sources

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", required=True, help="Question to ask the RAG system.")
    ap.add_argument("--data_dir", default="./data", help="Folder with .txt files.")
    ap.add_argument("--files", nargs="*", help="Specific files to use (overrides data_dir)")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--chunk_size", type=int, default=800)
    ap.add_argument("--chunk_overlap", type=int, default=120)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    file_paths = [Path(f) for f in args.files] if args.files else None

    texts = load_text_files(data_dir, file_paths)
    docs = build_chunks(texts, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    vs = build_vectorstore(docs)
    generator = make_generator()

    answer, contexts, sources = answer_question(
        vs, generator, args.question, k=args.k, max_new_tokens=args.max_new_tokens
    )
    print("\n=== Answer ===\n", answer.strip())
    print("\n=== Top Sources (chunk previews) ===")
    for i, d in enumerate(contexts, 1):
        preview = d.page_content[:200].replace("\n", " ")
        print(f"[{i}] {preview}...")

if __name__ == "__main__":
    main()
