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
index = pc.Index("qa-chunks")  # Your existing index name

class Query(BaseModel):
    question: str

class Response(BaseModel):
    answer: str
    sources: List[Dict]

def get_query_embedding(query: str):
    """Generate an embedding for a user's query using Cohere."""
    response = cohere_client.embed(
        texts=[query],
        model="embed-english-light-v2.0"
    )
    return response.embeddings[0]

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
        
        # Extract relevant passages and metadata
        context_passages = []
        sources = []
        for match in results['matches']:
            text = match['metadata']['text']
            context_passages.append(text)
            sources.append({
                'text': text[:200] + "...",  # Preview
                'source': 'Document',
                'score': match['score']
            })
        
        # Concatenate the top chunks into context
        context = " ".join(context_passages)
        
        # Create prompt for Claude
        prompt = (
            f"You are Dominic Cummings.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query.question}\n\n"
            f"Answer as if you are Dominic Cummings in the first person, providing some details from the context. Keep the response short (2 sentances max), direct and aggressive in tone. Clear up any grammatical errors in the source material"
        )
        
        # Get response from Claude
        message = anthropic_client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=400,
            temperature=0,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return Response(
            answer=message.content[0].text,
            sources=sources
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)