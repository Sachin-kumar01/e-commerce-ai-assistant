import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError(
        "OPENROUTER_API_KEY is missing from .env"
    )

# OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    timeout=120.0,
    max_retries=2,
    default_headers={
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "E-Commerce AI Assistant"
    }
)

# Test texts
texts = [
    "This is a mobile phone product.",
    "This is a laptop computer product."
]

print("Testing OpenRouter Embedding API...")
print("Model: openai/text-embedding-3-small")
print("Sending request...\n")

try:

    response = client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=texts
    )

    print("======================================")
    print("Embedding API is working!")
    print("======================================")

    print(
        "Number of embeddings:",
        len(response.data)
    )

    print(
        "Embedding dimension:",
        len(response.data[0].embedding)
    )

    print("\nFirst embedding generated successfully!")

except Exception as e:

    print("======================================")
    print("Embedding API request failed")
    print("======================================")

    print(type(e).__name__)
    print(str(e))