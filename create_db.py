from tqdm import tqdm
from chromadb.utils import embedding_functions
import os
import chromadb

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="ai-forever/sbert_large_nlu_ru")

path = 'e:\\RAG\\'

files = os.listdir(path)

documents = []
ids = []

for id, filename in tqdm(enumerate(files)):
    with open(os.path.join(path,filename), 'r', encoding='UTF-8') as f:
        text = f.read()
        documents.append(text)
        ids.append(filename)

client = chromadb.PersistentClient(path='db/')

collection = client.create_collection(name="vectordb", embedding_function=sentence_transformer_ef, metadata={"hnsw:space": "cosine"})
collection.add(documents=documents, ids=ids)

client = chromadb.PersistentClient(path='db/')