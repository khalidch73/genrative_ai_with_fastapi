from fastapi import FastAPI, HTTPException
from openai import OpenAI, APIError
from dotenv import load_dotenv, find_dotenv
import asyncio
import json


# Load environment variables
load_dotenv(find_dotenv())
client = OpenAI()

# Initialize FastAPI app
app = FastAPI(
    title="Assistants Function Calling",
    version="0.0.1",
    servers=[
        {"url": "http://127.0.0.1:8046", "description": "Development Server"}
    ]
)

# Define Functions
def getCurrentWeather(location: str, unit: str = "fahrenheit") -> dict:
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return {"location": "Tokyo", "temperature": "10", "unit": "celsius"}
    elif "los angeles" in location.lower():
        return {"location": "Los Angeles", "temperature": "72", "unit": "fahrenheit"}
    elif "paris" in location.lower():
        return {"location": "Paris", "temperature": "22", "unit": "celsius"}
    else:
        return {"location": location, "temperature": "unknown"}

def getNickname(location: str) -> str:
    """Get the nickname of a city"""
    if "tokyo" in location.lower():
        return "tk"
    elif "los angeles" in location.lower():
        return "la"
    elif "paris" in location.lower():
        return "py"
    else:
        return location

# Create an Assistant
assistant = client.beta.assistants.create(
    name="Weather Bot",
    instructions="You are a weather bot. Use the provided functions to answer questions.",
    model="gpt-3.5-turbo-1106",
    tools=[{
        "type": "function",
        "function": {
            "name": "getCurrentWeather",
            "description": "Get the weather in location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state e.g. San Francisco, CA"},
                    "unit": {"type": "string", "enum": ["c", "f"]}
                },
                "required": ["location"]
            }
        }
    }, {
        "type": "function",
        "function": {
            "name": "getNickname",
            "description": "Get the nickname of a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state e.g. San Francisco, CA"},
                },
                "required": ["location"]
            }
        }
    }]
)

# Create a thread for conversation
thread = client.beta.threads.create()

available_functions = {
    "getCurrentWeather": getCurrentWeather,
    "getNickname": getNickname
}

@app.post("/get_weather")
async def get_weather(location: str, unit: str = "fahrenheit"):
    try:
        # Create a user message
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"What is the weather in {location} in {unit}?"
        )

        # Run the assistant
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )

        # Loop until the run completes or requires action
        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if run_status.status == "requires_action":
                if run_status.required_action.submit_tool_outputs and run_status.required_action.submit_tool_outputs.tool_calls:
                    tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
                    tool_outputs = []
                    for tool_call in tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        if function_name in available_functions:
                            function_to_call = available_functions[function_name]
                            if function_to_call.__name__ == "getCurrentWeather":
                                response = function_to_call(
                                    location=function_args.get("location"),
                                    unit=function_args.get("unit")
                                )
                            elif function_to_call.__name__ == "getNickname":
                                response = function_to_call(
                                    location=function_args.get("location")
                                )
                            tool_outputs.append({
                                "tool_call_id": tool_call.id,
                                "output": json.dumps(response)  # Convert the response to a JSON string
                            })
                    client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread.id,
                        run_id=run.id,
                        tool_outputs=tool_outputs
                    )
            elif run_status.status == "completed":
                messages = client.beta.threads.messages.list(thread_id=thread.id)
                for message in messages.data:
                    if message.role == "assistant":
                        return {"weather": message.content[0].text.value}
                break
            elif run_status.status == "failed":
                raise HTTPException(status_code=500, detail="Run failed.")
            elif run_status.status in ["in_progress", "queued"]:
                await asyncio.sleep(1)
            else:
                raise HTTPException(status_code=500, detail=f"Unexpected status: {run_status.status}")
    except APIError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_nickname")
async def get_nickname(location: str):
    try:
        # Create a user message
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"What is the nickname of {location}?"
        )

        # Run the assistant
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )

        # Loop until the run completes or requires action
        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if run_status.status == "requires_action":
                if run_status.required_action.submit_tool_outputs and run_status.required_action.submit_tool_outputs.tool_calls:
                    tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
                    tool_outputs = []
                    for tool_call in tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        if function_name in available_functions:
                            function_to_call = available_functions[function_name]
                            if function_to_call.__name__ == "getCurrentWeather":
                                response = function_to_call(
                                    location=function_args.get("location"),
                                    unit=function_args.get("unit")
                                )
                            elif function_to_call.__name__ == "getNickname":
                                response = function_to_call(
                                    location=function_args.get("location")
                                )
                            tool_outputs.append({
                                "tool_call_id": tool_call.id,
                                "output": json.dumps(response)  # Convert the response to a JSON string
                            })
                    client.beta.threads.runs.submit_tool_outputs(
                        thread_id=thread.id,
                        run_id=run.id,
                        tool_outputs=tool_outputs
                    )
            elif run_status.status == "completed":
                messages = client.beta.threads.messages.list(thread_id=thread.id)
                for message in messages.data:
                    if message.role == "assistant":
                        return {"nickname": message.content[0].text.value}
                break
            elif run_status.status == "failed":
                raise HTTPException(status_code=500, detail="Run failed.")
            elif run_status.status in ["in_progress", "queued"]:
                await asyncio.sleep(1)
            else:
                raise HTTPException(status_code=500, detail=f"Unexpected status: {run_status.status}")
    except APIError as e:
        raise HTTPException(status_code=500, detail=str(e))
