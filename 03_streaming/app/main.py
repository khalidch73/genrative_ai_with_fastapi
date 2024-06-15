from fastapi import FastAPI, Query
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from fastapi.responses import StreamingResponse



#############################################################  Load environment variables ############################################
_ : bool = load_dotenv(find_dotenv()) # read local .env file
client : OpenAI = OpenAI()


# Initialize FastAPI app
app = FastAPI(
    title="Chta Compeletion streaming",
    version="0.0.1",
    servers=[
        {"url": "http://127.0.0.1:8042", "description": "Development Server"}
    ]
)


# streaming for single role
@app.get("/streaming_fastapi_single_role")
async def stream_openai_get_single_role(
    user_input: str = Query(..., description="User input for the singe role streaming")
):
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[{"role": "user", "content": user_input}],
        stream=True,
    )

    def event_stream():
        for part in stream:
            content = part.choices[0].delta.content or ""
            if content:
                yield content

    return StreamingResponse(event_stream(), media_type="text/plain")

# streaming for multi role
@app.get("/streaming_fastapi_multi_role")
async def stream_openai_get_multi_role(user_input: str = Query(..., description="User input for the multi role streaming")):
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[{"role": "system", "content": "You are a medical specilist doctor."},
                  {"role": "user", "content": user_input}],
        stream=True,
    )

    def event_stream():
        for part in stream:
            content = part.choices[0].delta.content or ""
            if content:
                yield content

    return StreamingResponse(event_stream(), media_type="text/plain")