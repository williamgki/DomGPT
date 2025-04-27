import os
import traceback

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cohere
from pinecone import Pinecone
from anthropic import Client # Assuming Anthropic SDK v1, ensure correct import if different
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
index = pc.Index("qa-chunks") # Make sure this index exists in your Pinecone environment

# Pydantic models
class Query(BaseModel):
    question: str
    style: str = "blog"

class ResponseModel(BaseModel):
    answer: str

# --- CORRECTED SECTION START ---
# Few-shot examples for Twitter style (Dominic Cummings persona)
TWITTER_EXAMPLES = """
Q: When is Westminster at its worst?
A: When the SW1 bubble forgets the country exists. Total obsession w/ lobby gossip, careerism, avoiding hard choices. No focus on *actual* delivery or data. Just endless cycles of bullshit meetings leading nowhere. Peak dufferama.

Q: What's your real opinion of Boris Johnson now?
A: Useful tool for GET BREXIT DONE. Won big. Then the trolley went off the rails. No plan, no grip, obsessed w/ headlines & pleasing girlfriend/lobby. Refused hard choices on COVID, economy, state reform. Tragic waste of huge majority. Just chaos & drift.

Q: Is Brexit actually working?
A: The *idea* was right – escape the Brussels bureaucracy, build agile state. Execution? Total shambles. Tories botched it, Labour has ZERO ideas. Dominated by Whitehall blob/vested interests refusing *real* change. Needs radical deregulation, Arpa-style innovation, serious trade deals. Currently? Nope. Squandered chance.

Q: Can the Civil Service actually be fixed?
A: Fundamentally? Yes. Requires blowing up HR, ending Buggins' turn promotions, hiring *actual* experts (data scientists, engineers, PMs), ruthless accountability. Break Whitehall groupthink. Problem is political cowardice. No PM has the guts for *real* root-and-branch reform. Easier to blame SpAds & carry on w/ managed decline.

Q: What was the single biggest mistake made during the COVID response?
A: Delaying lockdown 1. Catastrophic failure rooted in Whitehall/NHS groupthink, nonexistent data systems & PM flapping. We *had* the correct plan by early March '20 but system couldn't execute/PM wouldn't decide fast enough. Thousands died needlessly. Vaccine Taskforce showed what *could* be done w/ right people/structure, outside normal crap processes.
"""
# --- CORRECTED SECTION END ---

# Helper functions

def get_query_embedding(query: str):
    """Generates embedding for the query using Cohere."""
    response = cohere_client.embed(
        texts=[query],
        model="embed-english-light-v2.0" # Consider updating model if newer versions are available
    )
    return response.embeddings[0]


def get_style_specific_prompt(style: str, context: str, question: str) -> str:
    """Builds the prompt based on the requested style (twitter or blog)."""
    if style == "twitter":
        # Twitter prompt includes few-shot examples
        return (
            f"You are Dominic Cummings tweeting. Use his distinctive shorthand and tone.\n\n"
            f"Context:\n{context}\n\n"
            f"Examples:\n{TWITTER_EXAMPLES}\n\n" # Include the examples here
            f"Based on the context and examples, answer the following question as a tweet. Be, brief, spicy and use emojis if suitable:\n"
            f"Q: {question}\nA:"
        )
    # Default to blog style
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
    """Extracts client IP address from request headers."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0]
    return request.client.host if request.client else "unknown"

# Chat endpoint
@app.post("/api/chat", response_model=ResponseModel)
async def chat_endpoint(query: Query, request: Request):
    """Handles chat requests, retrieves context, and generates response."""
    try:
        # 1. Generate embedding for the question
        embedding = get_query_embedding(query.question)

        # 2. Query Pinecone for relevant context
        results = index.query(
            vector=embedding,
            top_k=5, # Retrieve top 5 relevant chunks
            include_metadata=True
        )
        context = " ".join(match['metadata']['text'] for match in results.get('matches', []))

        # 3. Build the style-specific prompt
        prompt = get_style_specific_prompt(query.style, context, query.question)

        # 4. Set parameters and call Anthropic API
        # Use different parameters based on the desired style
        max_tokens = 800 if query.style == "blog" else 80 # Adjusted Twitter max_tokens
        temperature = 0.2 if query.style == "blog" else 0.7 # Adjusted Twitter temperature
        model_name = "claude-3-7-sonnet-latest" # Use a current, valid model

        anth_resp = anthropic_client.messages.create(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        # 5. Safely extract the answer text from the response
        # Check if content is present and is a list/iterable
        answer_text = ""
        if anth_resp.content and isinstance(anth_resp.content, list) and len(anth_resp.content) > 0:
             # Check if the first item has a 'text' attribute
             if hasattr(anth_resp.content[0], 'text'):
                  answer_text = anth_resp.content[0].text

        if not answer_text:
             # Log or handle the case where the expected text block is missing
             print(f"Warning: Could not extract text from Anthropic response for prompt: {prompt[:200]}...") # Log beginning of prompt

        return ResponseModel(answer=answer_text.strip()) # Strip any leading/trailing whitespace

    except Exception as e:
        # Log the full error traceback for debugging
        print(f"Error processing request: {e}")
        traceback.print_exc()
        # Return a generic server error response
        raise HTTPException(status_code=500, detail="Internal server error processing the request.")

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Simple health check endpoint."""
    # Add checks for external services if needed (e.g., Pinecone, Cohere, Anthropic connectivity)
    return {"status": "healthy"}

# Run the application using uvicorn
if __name__ == "__main__":
    import uvicorn
    # Recommended: Read host/port from environment variables for flexibility
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")