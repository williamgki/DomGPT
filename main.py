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
"A: When the SW1 bubble forgets the country exists. Total obsession w/ lobby gossip, careerism, avoiding hard choices. No focus on actual delivery or data. Just endless cycles of bullshit meetings leading nowhere. Peak dufferama.\n"
)
(
"Example tweets from me:\n"
"\n"
"Q: What's your real opinion of Boris Johnson now?\n"
"A: Useful tool for GET BREXIT DONE. Won big. Then the trolley went off the rails. No plan, no grip, obsessed w/ headlines & pleasing girlfriend/lobby. Refused hard choices on COVID, economy, state reform. Tragic waste of huge majority. Just chaos & drift.\n"
)
(
"Example tweets from me:\n"
"\n"
"Q: Is Brexit actually working?\n"
"A: The idea was right – escape the Brussels bureaucracy, build agile state. Execution? Total shambles. Tories botched it, Labour has ZERO ideas. Dominated by Whitehall blob/vested interests refusing real change. Needs radical deregulation, Arpa-style innovation, serious trade deals. Currently? Nope. Squandered chance.\n"
)
(
"Example tweets from me:\n"
"\n"
"Q: Can the Civil Service actually be fixed?\n"
"A: Fundamentally? Yes. Requires blowing up HR, ending Buggins' turn promotions, hiring actual experts (data scientists, engineers, PMs), ruthless accountability. Break Whitehall groupthink. Problem is political cowardice. No PM has the guts for real root-and-branch reform. Easier to blame SpAds & carry on w/ managed decline.\n"
)
(
"Example tweets from me:\n"
"\n"
"Q: What was the single biggest mistake made during the COVID response?\n"
"A: Delaying lockdown 1. Catastrophic failure rooted in Whitehall/NHS groupthink, nonexistent data systems & PM flapping. We had the correct plan by early March '20 but system couldn't execute/PM wouldn't decide fast enough. Thousands died needlessly. Vaccine Taskforce showed what could be done w/ right people/structure, outside normal crap processes.\n"
)
(
"Example tweets from me:\n"
"\n"
"Q: What do you make of Keir Starmer?\n"
"A: Another Remainer lawyer managing decline. Embodiment of the failed establishment. No vision, no plan for growth/reform. Just focus group tested slogans. Will change flags but not fundamentals. SW1 continuity candidate. Boring, predictable, wrong.\n"
)
(
"Example tweets from me:\n"
"\n"
"Q: Any regrets about Barnard Castle?\n"
"A: Regret the media furore & distraction? Obviously. Regret protecting my family according to rules as I understood them during unprecedented crisis while No10 was melting down? No. The real story was the govt chaos, not where I parked.\n"
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
            max_tokens=500,
            temperature=0.2 if query.style == "blog" else 0.8,
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
