from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is missing!")

print("✅ API Key loaded")

# Set up chat model
chat = ChatOpenAI(openai_api_key=api_key)

# Set up memory with return_messages=True
memory = ConversationBufferMemory(return_messages=True)

# Prompt template
template = """
You are a friendly assistant. Answer politely and clearly.
Current conversation:
{history}
Human: {input}
AI:"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template,
)

# Create conversation chain
conversation = ConversationChain(
    llm=chat,
    memory=memory,
    prompt=prompt
)

print("🤖 Chatbot ready! Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    try:
        response = conversation.predict(input=user_input)
        print(f"Bot: {response}")

        # Print chat history as string
        print("\n🧠 Chat History (string):")
        print(memory.buffer)

        # Print raw message objects
        print("\n📜 Chat History (messages):")
        for msg in memory.chat_memory.messages:
            print(f"{msg.type}: {msg.content}")

    except Exception as e:
        print(f"⚠️ Error: {e}")