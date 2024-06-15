from fastapi import FastAPI
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from openai.types.chat.chat_completion import ChatCompletion


#############################################################  Load environment variables ############################################
_ : bool = load_dotenv(find_dotenv()) # read local .env file
client : OpenAI = OpenAI()


# Initialize FastAPI app
app = FastAPI(
    title="Chat Completion",
    version="0.0.1",
    servers=[
        {"url": "http://127.0.0.1:8040", "description": "Development Server"}
    ]
)

# Chat_completion end ponit
@app.get("/chat_completion")
def chat_completion(prompt : str )-> str:
 response : ChatCompletion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo-1106",
    )
 return response.choices[0].message.content

