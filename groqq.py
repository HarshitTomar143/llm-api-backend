import os
from dotenv import load_dotenv
from groq import Groq
from ddgs import DDGS

load_dotenv()

client= Groq(api_key=os.getenv("GROQ_API_KEY"))

LATEST_KEYWORDS = [
    "latest", "today", "current", "now", "news", "2024", "2025", "2026", "recent", "update", "price",
    "score", "launch", "last"
]

def needsLiveData(query:str)-> bool:
    q= query.lower()
    return any(word in q for word in LATEST_KEYWORDS)

def fetchLiveContent(query: str)-> str:
    snippets= []
    with DDGS() as ddgs:
        results= ddgs.text(query, max_results=5)
        for r in results: 
            title= r.get("title","")
            body= r.get("body","")
            snippets.append(f"{title}:{body}")
        return "\n".join(snippets)

while True:
    inp = input("\nAsk your question (or type exit): ")
    if inp.lower()=='exit':
        break

    messages= []

    if needsLiveData(inp):
        liveContext= fetchLiveContent(inp)

        messages.append({
            "role": "system",
            "content": (
                "You are a helpful assistant. Use ONLY the LIVE INFORMATION "
                "below to answer accurately. If the information is insufficient, "
                "say you do not know.\n\n"
                f"LIVE INFORMATION:\n{liveContext}")
        })   

    else:
         messages.append({
            "role": "system",
            "content": "You are a helpful assistant."
        })         
         
    messages.append({"role": "user", "content": inp})

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.2,
        max_tokens=1000
    )


    print("\nAnswer:\n")
    print(completion.choices[0].message.content)     