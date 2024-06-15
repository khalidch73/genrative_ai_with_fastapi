import json
from fastapi import FastAPI
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from openai.types.chat.chat_completion import ChatCompletionMessage, ChatCompletion


#############################################################  Load environment variables ############################################
_ : bool = load_dotenv(find_dotenv()) # read local .env file
client : OpenAI = OpenAI()


# Initialize FastAPI app
app = FastAPI(
    title="Function Calling",
    version="0.0.1",
    servers=[
        {"url": "http://127.0.0.1:8043", "description": "Development Server"}
    ]
)




def get_current_weather(location:str, unit:str="fahrenheit")->str:
    """Get the current weather in a given location"""
    if "karachi" in location.lower():
        return json.dumps({"location": "Karachi", "temperature": "10", "unit": "celsius"})
    elif "islamabad" in location.lower():
        return json.dumps({"location": "Islamabad", "temperature": "72", "unit": "fahrenheit"})
    elif "lahore" in location.lower():
        return json.dumps({"location": "Lahore", "temperature": "22", "unit": "celsius"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})
    
    
# single function calling 
@app.get("/single_function_calling")
def single_function_calling(prompt: str)->str:
    # Step 1: send the conversation and available functions to the model
    messages = [{"role": "user", "content": prompt}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    # First Request
    response: ChatCompletion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message: ChatCompletionMessage = response.choices[0].message
  
    tool_calls = response_message.tool_calls

    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_current_weather": get_current_weather,
        }  
        messages.append(response_message)  # extend conversation with assistant's reply
        
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                location=function_args.get("location"),
                unit=function_args.get("unit"),
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        second_response: ChatCompletion = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
        )  # get a new response from the model where it can see the function response
        return second_response.choices[0].message.content



        