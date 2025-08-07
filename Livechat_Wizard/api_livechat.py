from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import uvicorn
from livechat import LiveChatController

# Global variable to hold the VtuberExllamav2 instance
live_chat_controller = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    global live_chat_controller
    
    # Startup
    print("Starting up...")
    try:
        # Initialize the LiveChatController instance
        live_chat_controller = LiveChatController.create()
        print("Live chat controller initialized successfully")
    except Exception as e:
        print(f"Error initializing live chat controller: {e}")
        live_chat_controller = None
    
    yield  # Application runs here
    
    # Shutdown
    print("Shutting down...")
    if live_chat_controller:
        live_chat_controller.cleanup()
        print("live_chat_controller cleanup completed")


# <<< 5. DEFINE A RESPONSE MODEL FOR THE NEW ENDPOINT
class ChatMessageResponse(BaseModel):
    picked_message: Optional[str]
    remaining_messages: Optional[List[str]]


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Live chat fetcher API",
    description="API endpoint for fetching chat messages from live stream chat",
    version="1.0.0",
    lifespan=lifespan
)
# --- NEW ENDPOINT ---
@app.get("/fetch_message", response_model=ChatMessageResponse)
async def fetch_message_endpoint():
    """
    Fetches a single chat message from the live feed.
    
    Returns a JSON object containing the picked message and a list of
    other messages fetched in the same batch.
    """
    if live_chat_controller is None:
        raise HTTPException(status_code=503, detail="Chat retriever not initialized")

    try:
        message, remaining = await live_chat_controller.fetch_chat_message()
        return ChatMessageResponse(
            picked_message=message,
            remaining_messages=remaining
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chat message: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "live_chat_controller_initialized": live_chat_controller is not None}

if __name__ == "__main__":
    # Run the server

    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001,
        # reload=True
    )