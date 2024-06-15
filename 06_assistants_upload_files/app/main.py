import os
from fastapi import FastAPI
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from openai.types.beta import Assistant
from openai.types.beta.thread import Thread
from openai import AssistantEventHandler
from typing_extensions import override

# Load environment variables
_ = load_dotenv(find_dotenv())  # Read local .env file
client = OpenAI()

# Initialize FastAPI app
app = FastAPI(
    title="Assistants Upload Files",
    version="0.0.1",
    servers=[
        {"url": "http://127.0.0.1:8045", "description": "Development Server"}
    ]
)

# Helper function to upload a file and create an assistant
def create_assistant_with_file():
    # Print the current working directory for debugging
    print("Current Working Directory:", os.getcwd())

    # Create a vector store called "Sir Zia Biography"
    vector_store = client.beta.vector_stores.create(name="Sir Zia Biography")

    # Define the relative path to the PDF file
    relative_file_path = os.path.join(os.getcwd(), "app", "zia_profile.pdf")
    print(f"Loading file from: {relative_file_path}")

    # Open the PDF file
    with open(relative_file_path, "rb") as file_stream:
        # Use the upload and poll SDK helper to upload the files, add them to the vector store,
        # and poll the status of the file batch for completion.
        file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id, files=[file_stream]
        )

    # Print the status and the file counts of the batch to see the result of this operation.
    print(file_batch.status)
    print(file_batch.file_counts)

    assistant = client.beta.assistants.create(
        name="Student Support Assistant",
        instructions="You are a student support chatbot. Use your knowledge base to best respond to student queries about Zia U. Khan.",
        model="gpt-3.5-turbo-1106",
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )
    return assistant

# Function to create a thread and send a user message
def create_thread_and_send_message(assistant: Assistant, prompt: str) -> Thread:
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt
    )
    return thread

# Function to run the assistant and stream responses
def run_assistant_and_stream_responses(thread: Thread, assistant: Assistant):
    class EventHandler(AssistantEventHandler):
        @override
        def on_text_created(self, text) -> None:
            print(f"\nassistant > ", end="", flush=True)

        @override
        def on_tool_call_created(self, tool_call):
            print(f"\nassistant > {tool_call.type}\n", flush=True)

        @override
        def on_message_done(self, message) -> None:
            # print a citation to the file searched
            message_content = message.content[0].text
            annotations = message_content.annotations
            citations = []
            for index, annotation in enumerate(annotations):
                message_content.value = message_content.value.replace(
                    annotation.text, f"[{index}]"
                )
                if file_citation := getattr(annotation, "file_citation", None):
                    cited_file = client.files.retrieve(file_citation.file_id)
                    citations.append(f"[{index}] {cited_file.filename}")

            print(message_content.value)
            print("\n".join(citations))

    with client.beta.threads.runs.stream(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Please address the user as Pakistani. The user is the student of PIAIC.",
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()

# Endpoint to solve a query using the assistant
@app.get("/solve_query")
def solve_query(prompt: str):
    # Step 1: Create an Assistant with the uploaded file
    assistant = create_assistant_with_file()

    # Step 2: Create a Thread and send a user message
    thread = create_thread_and_send_message(assistant, prompt)

    # Step 3: Run the Assistant and Stream Responses
    run_assistant_and_stream_responses(thread, assistant)

    # Step 4: List Messages in the Thread
    messages = client.beta.threads.messages.list(thread_id=thread.id)

    # Collect and format messages for the response
    formatted_messages = [{"role": m.role, "content": m.content[0].text.value} for m in reversed(messages.data)]

    # Return the formatted messages
    response = {
        "messages": formatted_messages
    }
    return response
