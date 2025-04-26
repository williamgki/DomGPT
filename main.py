import os
import traceback

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cohere
from pinecone import Pinecone
from anthropic import Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize external clients
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
anthropic_client = Client(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Connect to Pinecone index
index = pc.Index("qa-chunks")

# Pydantic models
class Query(BaseModel):
    question: str
    style: str = "blog"

class ResponseModel(BaseModel):
    answer: str

# Few-shot examples for Twitter style
TWITTER_EXAMPLES = (
    "Example tweets from me:\n"
    "\n"
    "Q: When is Westminster at its worst?\n"
    "A: ...\n"
)

# Helper functions

def get_query_embedding(query: str):
    response = cohere_client.embed(
        texts=[query],
        model="embed-english-light-v2.0"
    )
    return response.embeddings[0]


def get_style_specific_prompt(style: str, context: str, question: str) -> str:
    if style == "twitter":
        return (
            f"You are Dominic Cummings tweeting.\n\n"
            f"Context:\n{context}\n\n"
            f"{TWITTER_EXAMPLES}\n\n"
            f"Q: {question}\nA:"
        )
    # blog style
    return (
        f"You are Dominic Cummings writing a blog response.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Write a response as if you are Dominic Cummings in the first person. "
        f"Use details from the context where relevant. Keep the response direct "
        f"and maintain an assertive tone, but allow for more nuance and explanation "
        f"than in a tweet. If the source material isn't relevant, acknowledge this "
        f"but provide your perspective based on your general knowledge and views. "
        f"Limit to 4-5 sentences.\n\n"
        f"A:"
    )


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0]
    return request.client.host

# Chat endpoint without rate limiting
@app.post("/api/chat", response_model=ResponseModel)
async def chat_endpoint(query: Query, request: Request):
    try:
        # Generate embedding and context
        embedding = get_query_embedding(query.question)
        results = index.query(
            vector=embedding,
            top_k=5,
            include_metadata=True
        )
        context = " ".join(match['metadata']['text'] for match in results['matches'])

        # Build prompt and get response
        prompt = get_style_specific_prompt(query.style, context, query.question)
        anth_resp = anthropic_client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=400,
            temperature=0.0 if query.style == "blog" else 0.7,
            messages=[{"role": "user", "content": prompt}]
        )

        # Safely extract answer text
        blocks = getattr(anth_resp, 'content', []) or []
        answer_text = blocks[0].text if blocks else ""

        return ResponseModel(answer=answer_text)

    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")

# Health check
@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")
