import requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# import chromadb
DB_DIR = "./chroma_db"
# chromadb.Settings.anonymized_telemetry = False

def load_and_index_docs(filepath="chat/knowledge.txt"):
    loader = TextLoader(filepath)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=DB_DIR)
    db.persist()

def get_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

def query_deepseek(prompt, model="deepseek-r1:7b"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        }
    )
    print(f"Response status code: {response.status_code}")
    if response.status_code != 200:
        return f"Error: Received status code {response.status_code} from DeepSeek API"
    return response.json()["response"]

def ask_question(question):
    db = get_db()
    docs = db.similarity_search(question, k=1)
    context = "\n\n".join([doc.page_content for doc in docs])
    print(f"Context retrieved: {context[:1000]}...")  # Print first 1000 characters for brevity
    print(f"Question asked: {question}")
    # breakpoint()

    prompt = f"""You are an expert AI assistant. Use the context below to answer the user's question.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"""

    return query_deepseek(prompt)
