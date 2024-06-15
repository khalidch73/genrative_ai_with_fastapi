from fastapi import FastAPI
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from openai.types.chat.chat_completion import ChatCompletion


#############################################################  Load environment variables ############################################
_ : bool = load_dotenv(find_dotenv()) # read local .env file
client : OpenAI = OpenAI()



# Initialize FastAPI app
app = FastAPI(
    title="Chta Compeletion Multi Role",
    version="0.0.1",
    servers=[
        {"url": "http://127.0.0.1:8041", "description": "Development Server"}
    ]
)

# Chat_completion end ponit
@app.get("/chat_completion")
def Multi_role_chat_completion(prompt : str)->str:
  completion : ChatCompletion = client.chat.completions.create(
    model  = "gpt-3.5-turbo-1106",
    messages= [
      {"role": "system", "content": "You are a medical specilist doctor."},
      {"role": "user", "content": prompt}
    ]
  )
  return completion.choices[0].message.content
