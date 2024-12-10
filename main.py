from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cohere
from pinecone import Pinecone
from anthropic import Client
import os
from typing import List, Dict
from dotenv import load_dotenv

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
index = pc.Index("qa-chunks")

class Query(BaseModel):
    question: str
    style: str = "blog"  # Default to blog style if not specified

class Response(BaseModel):
    answer: str

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

Now answer this question in a similar style:"""

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

@app.post("/api/chat", response_model=Response)
async def chat_endpoint(query: Query):
    try:
        # Generate embedding for the question
        query_embedding = get_query_embedding(query.question)
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
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
        
        return Response(
            answer=message.content[0].text,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)