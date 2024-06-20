from os import getenv
from chromadb.utils import embedding_functions

import chromadb
import aiohttp
import asyncio
import json

url = "https://api.awanllm.com/v1/chat/completions"
model_name = "Awanllm-Llama-3-8B-Dolfin"

api_token = getenv("API_TOKEN")

headers = {
    'Content-Type': 'application/json',
    'Authorization': f"Bearer {api_token}"
}

# Загрузка БД
client = chromadb.PersistentClient(path='db/')
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="ai-forever/sbert_large_nlu_ru")
collection = client.get_or_create_collection(name="vectordb", embedding_function=sentence_transformer_ef, metadata={"hnsw:space": 'cosine'})

def find_closest_files(query: str, n_results: int = 5):
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )

        return results

async def create_promt(user_message: str, messages):
    docs = find_closest_files(user_message)
    info = ''.join(*docs['documents'])

    messages.append({'role': 'system', "content" : "Используй эту информацию при ответе на вопрос." + info})
    messages.append({'role': 'user', "content" : user_message})

    return messages

async def create_payload(messages):
    payload = json.dumps({
        "model": f"{model_name}",
        "messages": messages
    })

    return payload

async def send_promt(url, headers, payload):
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=payload) as response:
                res = await response.json()
        return res['choices'][0]['message']['content']


async def get_asnwer(message: str):   
    messages = [
    {
        "role": "system", 
        "content": "Тебя зовут HistoryAI. Ты разговариваешь с людьми и помогаешь им разобраться с историческими вопросами. Тебе будет дана дополнительная информация. Можешь использовать её. Не выдумывай ничего нового. Отвечай на русском языке."
    }]
    
    messages = await create_promt(message, messages)
    payload = await create_payload(messages)
    answer = await send_promt(url, headers, payload)
    
    return answer