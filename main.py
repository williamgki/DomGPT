from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cohere
from pinecone import Pinecone
from anthropic import Client
import os
from typing import List, Dict
from dotenv import load_dotenv
from datetime import datetime, timedelta
from collections import defaultdict

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients using environment variables
cohere_client = cohere.Client(os.getenv('COHERE_API_KEY'))
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
anthropic_client = Client(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Connect to existing Pinecone index
index = pc.Index("tweets-index")

# Rate limiting setup
RATE_LIMIT_MINUTES = 30
RATE_LIMIT_REQUESTS = 3

class RateLimiter:
    def __init__(self):
        self.requests = defaultdict(list)
    
    def _clean_old_requests(self, ip: str):
        """Remove requests older than the rate limit window"""
        cutoff = datetime.now() - timedelta(minutes=RATE_LIMIT_MINUTES)
        self.requests[ip] = [req for req in self.requests[ip] if req > cutoff]
    
    def is_rate_limited(self, ip: str) -> bool:
        """Check if an IP has exceeded the rate limit"""
        self._clean_old_requests(ip)
        return len(self.requests[ip]) >= RATE_LIMIT_REQUESTS
    
    def add_request(self, ip: str):
        """Record a new request"""
        self._clean_old_requests(ip)
        self.requests[ip].append(datetime.now())
    
    def get_remaining_requests(self, ip: str) -> dict:
        """Get remaining requests and time until reset"""
        self._clean_old_requests(ip)
        requests_made = len(self.requests[ip])
        requests_remaining = RATE_LIMIT_REQUESTS - requests_made
        
        if requests_made > 0:
            oldest_request = min(self.requests[ip])
            reset_time = oldest_request + timedelta(minutes=RATE_LIMIT_MINUTES)
            seconds_until_reset = max(0, (reset_time - datetime.now()).total_seconds())
        else:
            seconds_until_reset = 0
            
        return {
            "requests_remaining": requests_remaining,
            "seconds_until_reset": int(seconds_until_reset)
        }

rate_limiter = RateLimiter()

class Query(BaseModel):
    question: str
    style: str = "blog"  # Default to blog style if not specified

class Response(BaseModel):
    answer: str
    rate_limit_info: dict

# Few-shot examples for Twitter style
TWITTER_EXAMPLES = """
Example tweets from me:

Q: When is Westminster at its worst?
A: my disgust for westminster is never higher than when it's excited about war, all the worst of its lies posing  cant bullshit idiocy hypocrisy moral cowardice incompetence operational uselessness get turned up to 11

Q: What should we do about Palestinian refugees?
A: After 30 years of Tories & Labour importing thousands of people who want to destroy England, a cross party consensus to import 1000s of Hamas wd be totally logical and wd get massive NPC support - we're already giving illegal immigrants & terrorists PRIVATE healthcare as soon as they arrive so very logical to add Hamas brigades fleeing war.

Carrie & Trolley can organise more Stonewall events for them so we can spread the cultural enrichment

Q: What do you think of Elon Musk's comments on the Ukraine war?
A: One of the world's most effective entrepreneurs: why don't the politicians do diplomacy instead of escalating a nuclear crisis?

Swarm of politics/media/academia NPCs: fuck you, no diplomacy, total victory, Putin is *both* insane psycho *and* too rational to use nukes!

Q: What should farmers do about government clampdowns on their business?
A: Spread the word among farmers: Blair's lead adviser says CLOSE DOWN ALL SMALL FARMERS because they're a political enemy.
RETWEET so farmers hear the Labour plan
The NFU is telling farmers NOT to come to London to demonstrate.
TERRIBLE ADVICE.
I've been in the PM office when

Now answer this question in a similar style, but keep the response short and punch - 3 sentances max. Try to use at least one emoji"""

def get_query_embedding(query: str):
    """Generate an embedding for a user's query using Cohere."""
    response = cohere_client.embed(
        texts=[query],
        model="embed-english-light-v2.0"
    )
    return response.embeddings[0]

def get_style_specific_prompt(style: str, context: str, question: str) -> str:
    """Return a style-specific prompt."""
    if style == "twitter":
        return (
            f"You are Dominic Cummings tweeting.\n\n"
            f"Context:\n{context}\n\n"
            f"{TWITTER_EXAMPLES}\n\n"
            f"Q: {question}\nA:"
        )
    else:  # blog style
        return (
            f"You are Dominic Cummings writing a blog response.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Write a response as if you are Dominic Cummings in the first person. "
            f"Use details from the context where relevant. Keep the response direct "
            f"and maintain an assertive tone, but allow for more nuance and explanation "
            f"than in a tweet. If the source material isn't relevant, acknowledge this "
            f"but provide your perspective based on your general knowledge and views. "
            f"Limit to 4-5 sentences."
        )

def get_client_ip(request: Request) -> str:
    """Get client IP, handling proxy headers"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0]
    return request.client.host

@app.post("/api/chat", response_model=Response)
async def chat_endpoint(query: Query, request: Request):
    client_ip = get_client_ip(request)
    
    # Check rate limit
    if rate_limiter.is_rate_limited(client_ip):
        rate_info = rate_limiter.get_remaining_requests(client_ip)
        raise HTTPException(
            status_code=429,
            detail={
                "message": "Rate limit exceeded",
                "rate_limit_info": rate_info
            }
        )
    
    try:
        # Record this request
        rate_limiter.add_request(client_ip)
        
        # Generate embedding for the question
        query_embedding = get_query_embedding(query.question)
        
        # Query Pinecone
        if query.style == "blog":
            results = index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True
            )
        else:  # twitter
            results = index.query(
                vector=query_embedding, namespace="tweets",
                top_k=5,
                include_metadata=True
            )
        
        # Extract relevant passages
        context_passages = []
        for match in results['matches']:
            text = match['metadata']['text']
            context_passages.append(text)
        
        # Concatenate the top chunks into context
        context = " ".join(context_passages)
        
        # Get style-specific prompt
        prompt = get_style_specific_prompt(query.style, context, query.question)
        
        # Get response from Claude
        message = anthropic_client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=400,
            temperature=0 if query.style == "blog" else 0.7,  # Higher temperature for Twitter style
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Get rate limit info for response
        rate_info = rate_limiter.get_remaining_requests(client_ip)
        
        return Response(
            answer=message.content[0].text,
            rate_limit_info=rate_info
        )
        
    except Exception as e:
        if not isinstance(e, HTTPException):  # Don't wrap HTTP exceptions
            raise HTTPException(status_code=500, detail=str(e))
        raise

@app.get("/api/rate_limit_status")
async def rate_limit_status(request: Request):
    """Get current rate limit status"""
    client_ip = get_client_ip(request)
    return rate_limiter.get_remaining_requests(client_ip)

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)