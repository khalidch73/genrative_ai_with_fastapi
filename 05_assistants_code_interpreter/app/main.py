from fastapi import FastAPI
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from openai.types.beta import Assistant
from openai.types.beta.thread import Thread
from openai.types.beta.threads.run import Run
from typing_extensions import override
from openai import AssistantEventHandler

# Load environment variables
load_dotenv(find_dotenv())
client = OpenAI()

# Initialize FastAPI app
app = FastAPI(
    title="Assistants code Interpreter",
    version="0.0.1",
    servers=[
        {"url": "http://127.0.0.1:8044", "description": "Development Server"}
    ]
)

# Function to create an assistant
def create_assistant() -> Assistant:
    assistant = client.beta.assistants.create(
        name="Math Tutor",
        instructions="You are a personal math tutor. Write and run code to answer math questions.",
        tools=[{"type": "code_interpreter"}],
        model="gpt-3.5-turbo-1106"
    )
    return assistant

# Function to create a thread
def create_thread() -> Thread:
    thread = client.beta.threads.create()
    return thread

# Function to add a message to a thread
def add_message_to_thread(thread_id: str, content: str):
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=content
    )
    return message

# Function to run the assistant and poll for the result
def run_assistant(thread_id: str, assistant_id: str) -> Run:
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread_id,
        assistant_id=assistant_id,
        instructions="Please address the user as Jane Doe. The user has a premium account."
    )
    return run

# Function to list messages in a thread
def list_messages(thread_id: str):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    return messages

# Custom event handler
class EventHandler(AssistantEventHandler):
    @override
    def on_text_created(self, text) -> None:
        print(f"\nassistant > ", end="", flush=True)
    
    @override
    def on_text_delta(self, delta, snapshot):
        print(delta.value, end="", flush=True)
    
    @override
    def on_tool_call_created(self, tool_call):
        print(f"\nassistant > {tool_call.type}\n", flush=True)
    
    @override
    def on_tool_call_delta(self, delta, snapshot):
        if delta.type == 'code_interpreter':
            if delta.code_interpreter.input:
                print(delta.code_interpreter.input, end="", flush=True)
            if delta.code_interpreter.outputs:
                print(f"\n\noutput >", flush=True)
                for output in delta.code_interpreter.outputs:
                    if output.type == "logs":
                        print(f"\n{output.logs}", flush=True)

# Function to stream the assistant's responses
def stream_assistant_responses(thread_id: str, assistant_id: str):
    with client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        instructions="Please address the user as Jane Doe. The user has a premium account.",
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()

# Endpoint to perform all steps
@app.get("/solve_equation")
def solve_equation(prompt: str):
    # Step 1: Create an Assistant
    assistant = create_assistant()
    
    # Step 2: Create a Thread
    thread = create_thread()
    
    # Step 3: Add a Message to a Thread
    message = add_message_to_thread(thread.id, prompt)
    
    # Step 4: Stream the Assistant's Responses
    stream_assistant_responses(thread.id, assistant.id)
    
    # Step 5: List Messages in the Thread
    messages = list_messages(thread.id)
    
    # Collect and format messages for the response
    formatted_messages = [{"role": m.role, "content": m.content[0].text.value} for m in reversed(messages.data)]
    
    # Return the assistant, thread, message, run, and messages details
    response = {
        # "assistant": assistant,
        # "thread": thread,
        # "message": message,
        "messages": formatted_messages
    }
    return response
