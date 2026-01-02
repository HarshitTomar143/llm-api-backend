import os 
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client= Groq(api_key=os.getenv("GROQ_API_KEY"))

inp= input("Ask your question: ")

completion = client.chat.completions.create(
    model= "llama-3.1-8b-instant",
    messages= [
        {
            "role": "user", "content": inp
        },{
            "role": "system", "content": "You are a very helpful assistant. You have to answer each and every question even if you dont know it"
        }
    ],
    temperature= 0.1,
    max_tokens=500
)

print(completion.choices[0].message.content)