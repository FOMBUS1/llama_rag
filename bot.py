from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message
from chromadb.utils import embedding_functions
import chromadb


# from llm_local import LLM_Local
from os import getenv

import asyncio
import aiohttp
import env
import json


TOKEN = getenv("BOT_TOKEN")
token = getenv("API_TOKEN")

headers = {
    'Content-Type': 'application/json',
    'Authorization': f"Bearer {token}"
}

url = "https://api.awanllm.com/v1/chat/completions"
model_name = "Awanllm-Llama-3-8B-Dolfin"
client = chromadb.PersistentClient(path='db/')
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="ai-forever/sbert_large_nlu_ru")
collection = client.get_or_create_collection(name="vectordb", embedding_function=sentence_transformer_ef, metadata={"hnsw:space": 'cosine'})

dp = Dispatcher()
# model = LLM_Local()

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer(f"Привет! Я виртуальный помощник! Меня зовут HistoryAI. Чем я могу тебе помочь?")

@dp.message()
async def echo_handler(message: Message) -> None:
    print("Message is received!")
    messages = [
    {
        "role": "system", 
        "content": "Тебя зовут HistoryAI. Ты разговариваешь с людьми и помогаешь им разобраться с историческими вопросами. Тебе будет дана дополнительная информация. Можешь использовать её. Не выдумывай ничего нового."
    }]
    messages = await create_promt(message.text, messages)
    payload = await create_payload(messages)
    answer = await send_promt(url, headers, payload)
    await message.answer(answer)

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

async def main() -> None:
    # Initialize Bot instance with default bot properties which will be passed to all API calls
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    # And the run events dispatching
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())