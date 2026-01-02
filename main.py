import os
import requests 
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq 
from ddgs import DDGS

load_dotenv()

app = FastAPI(title="Groq RAG Backend")

client= Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME= "llama-3.1-8b-instant"

LATEST_KEYWORDS=  [
    "latest", "today", "current", "now", "news",
    "2024", "2025", "recent", "update", "launch"
]

def needs_live_data(query:str)-> bool:
    q= query.lower()
    return any(word in q for word in LATEST_KEYWORDS)

def is_gold_price_query(query: str) -> bool:
    q = query.lower()
    return "gold" in q and ("price" in q or "rate" in q or "today" in q)

def fetch_live_context(query: str)-> str:
    snippets= []

    if "news" in query.lower():
        query = f"{query} India today breaking news"

    with DDGS() as ddgs:
        results= ddgs.text(query, max_results=5)
        for r in results:
            title= r.get("title","")
            body= r.get("body","")
            snippets.append(f"{title}:{body}")

    return "\n".join(snippets)

def fetch_gold_price() -> str:
    try:
        data = requests.get(
            "https://api.metals.live/v1/spot/gold",
            timeout=5
        ).json()
        price = data[0][1]
        return f"Gold price today: ${price} per ounce (USD). Source: metals.live"
    except Exception:
        return ""

def build_messages(user_query: str):
    messages= []
    if is_gold_price_query(user_query):
        gold_info= fetch_gold_price()
        messages.append({
            "role": "system",
            "content": (
                "You are an assistant. Use the LIVE PRICE information below "
                "to answer accurately.\n\n"
                f"LIVE INFORMATION:\n{gold_info}"
            )
        })


    elif needs_live_data(user_query):
        live_context = fetch_live_context(user_query)
        messages.append({
            "role": "system",
            "content": (
                "You are an assistant. Use ONLY the LIVE INFORMATION below "
                "to answer accurately. If insufficient, say you do not know.\n\n"
                f"LIVE INFORMATION:\n{live_context}"
            )
        })

    else:
        messages.append({
            "role": "system",
            "content": "You are a helpful assistant."
        })

    messages.append({"role": "user", "content": user_query})
    return messages   

class ChatRequest(BaseModel):
    message: str

class CharResponse(BaseModel):
    reply: str
    used_live_data: bool

@app.post('/chat',response_model=CharResponse)
def chat(req: ChatRequest):
    messages= build_messages(req.message)

    completion= client.chat.completions.create(
        model= MODEL_NAME,
        messages= messages,
        temperature= 0.2,
        max_tokens= 500
    )

    return {
        "reply": completion.choices[0].message.content,
        "used_live_data": len(messages) > 1
    }
