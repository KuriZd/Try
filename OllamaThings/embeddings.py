
import chromadb
from chromadb.config import Settings

chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))
collection = chroma_client.get_or_create_collection(name="bioetica")

# Simulación del embedding con Ollama
def fake_embed(text):  # Aquí deberías integrar con Ollama si tienes soporte de embeddings
    import hashlib
    return [float(int(hashlib.md5(text.encode()).hexdigest(), 16) % 1000) / 1000.0 for _ in range(384)]

def index_chunks(chunks):
    for i, chunk in enumerate(chunks):
        collection.add(documents=[chunk], ids=[f"doc_{i}"], embeddings=[fake_embed(chunk)])
