from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from chromadb.utils import embedding_functions

import torch
import warnings
import chromadb

warnings.filterwarnings('ignore')

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="ai-forever/sbert_large_nlu_ru")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model directly
model_checkpoint = "IlyaGusev/saiga_llama3_8b"
#Загрузка токенайзера для перевода предложения в чилса
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(model_checkpoint, quantization_config=bnb_config, attn_implementation="eager", device_map='auto')

generation_config = GenerationConfig(
    do_sample=True,
    temperature=0.3,
    repeatition_penalty=1.2,
    max_length=8192,
    max_new_tokens=512,
    min_new_tokens=2,
    pad_token_id=tokenizer.eos_token_id,
)

#Loading DB
client = chromadb.PersistentClient(path='db/')

collection = client.get_or_create_collection(name="vectordb", embedding_function=sentence_transformer_ef, metadata={"hnsw:space": "cosine"})

messages = [
            {"role": "system", "content": "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им разобраться с историческими вопросами. Тебе будет дана дополнительная информация. Можешь использовать её. Не выдумывай ничего нового."},
        ]


while True:
    message = input("Введите свой вопрос: ")

    results = collection.query(
        query_texts=[message],
        n_results=5
    )

    docs = ''.join(*results['documents'])

    messages.append({'role': 'system', "content" : "Используй эту информацию при ответе на вопрос." + docs})
    messages.append({'role': 'user', "content" : message})

    data = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            **data,
            generation_config=generation_config
        )[0]

    output_ids = output_ids[len(data["input_ids"][0]):]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(output.strip())

