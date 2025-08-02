# kick_client.py
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple, List

# CORRECTED: AsyncSession is imported from curl_cffi.requests
from curl_cffi.requests import AsyncSession

# It's good practice to set up logging for a library or worker component.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

@dataclass(frozen=True)
class ChatUser:
    """Represents a user in a Kick chat."""
    username: str
    color: Tuple[int, int, int]

@dataclass(frozen=True)
class ChatMessage:
    """Represents a single, structured message in a Kick chat."""
    id: str
    user: ChatUser
    content: str
    timestamp: datetime

def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Converts a hex color string to an (R, G, B) tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (255, 255, 255) # Default to white for invalid formats
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

class KickApiError(Exception):
    """Custom exception for Kick API-related errors."""
    pass

class KickClient:
    """
    An async, on-demand client to fetch live chat messages from a Kick.com channel.
    Designed to be called by a worker system when needed.
    """
    BASE_URL = "https://kick.com/api/v2"

    def __init__(self, username: str, session: AsyncSession):
        if not username:
            raise ValueError("Username cannot be empty.")
        if not session:
            raise ValueError("An AsyncSession must be provided.")
            
        self.username = username
        self._session = session
        self._channel_id: Optional[str] = None

    async def _get_channel_id(self) -> str:
        """Fetches and caches the internal channel ID."""
        if self._channel_id:
            return self._channel_id

        url = f"{self.BASE_URL}/channels/{self.username}"
        try:
            res = await self._session.get(url, impersonate="chrome110")
            res.raise_for_status()
            data = res.json()
            channel_id = data.get("id")
            if not channel_id:
                raise KickApiError(f"Could not find 'id' in channel data for '{self.username}'.")
            self._channel_id = str(channel_id)
            return self._channel_id
        except Exception as e:
            logging.error(f"Failed to get channel ID for '{self.username}': {e}")
            raise KickApiError(f"Could not retrieve channel ID for '{self.username}'.") from e

    async def fetch_new_messages(self, last_seen_timestamp: datetime) -> List[ChatMessage]:
        """
        Fetches the latest batch of messages and filters out any that are older
        than the provided timestamp.

        Args:
            last_seen_timestamp: The timestamp of the last message the worker processed.
                                 The function will return messages newer than this.

        Returns:
            A list of new ChatMessage objects, sorted chronologically.
        """
        try:
            channel_id = await self._get_channel_id()
            url = f"{self.BASE_URL}/channels/{channel_id}/messages"
            
            res = await self._session.get(url, impersonate="chrome110")
            res.raise_for_status()
            
            data = res.json().get("data", {})
            raw_messages = data.get("messages", [])
            if not raw_messages:
                return []
            
            # Parse all messages from the API response
            parsed_messages = [
                ChatMessage(
                    id=msg["id"],
                    user=ChatUser(
                        username=msg["sender"]["username"],
                        color=_hex_to_rgb(msg["sender"]["identity"]["color"]),
                    ),
                    content=msg["content"],
                    timestamp=datetime.fromisoformat(msg["created_at"].replace("Z", "+00:00")),
                )
                for msg in raw_messages
            ]
            
            # Filter for new messages and sort them chronologically (oldest to newest)
            new_messages = sorted(
                [msg for msg in parsed_messages if msg.timestamp > last_seen_timestamp],
                key=lambda m: m.timestamp
            )
            
            return new_messages
        
        except Exception as e:
            logging.error(f"Failed to fetch messages for channel '{self.username}': {e}")
            # Return an empty list to make the caller resilient to transient API errors.
            return []

### How to Use the New On-Demand Client

async def check_for_new_kick_messages(client: KickClient, state: dict):
    """
    An example of a single, on-demand check for new messages.
    This function would be called periodically by your worker's scheduler.
    """
    username = client.username
    last_timestamp = state.get(username, datetime.now(timezone.utc))
    
    print(f"\nChecking for new messages for '{username}' since {last_timestamp.isoformat()}")
    
    # Fetch only new messages using the client
    new_messages = await client.fetch_new_messages(last_seen_timestamp=last_timestamp)

    if not new_messages:
        print("No new messages found.")
        return

    for msg in new_messages:
        # Your worker decides what to do with the message
        time_str = msg.timestamp.strftime("%H:%M:%S")
        print(f"  -> [{time_str}] {msg.user.username}: {msg.content}")

    # The worker updates its state with the newest timestamp for the next call
    state[username] = new_messages[-1].timestamp
    print(f"State updated. New last_seen_timestamp is {state[username].isoformat()}")


async def main():
    """
    Example of managing multiple channels and calling the fetcher on demand.
    """
    import dotenv
    from general_utils import get_env_var

    dotenv.load_dotenv()

    channel_username = get_env_var("KI_CHANNEL")

    
    # In a real worker, the session would be managed globally.
    # The `async with` block ensures it's closed properly.
    async with AsyncSession() as session:
        
        # --- State Management ---
        # Your worker system is responsible for state.
        # This could be a simple dict, a database, or a Redis cache.
        # We initialize it here for demonstration.
        chat_states = {
            "scikous": datetime.now(timezone.utc),
        }
        
        # Create clients for the channels you want to monitor
        client = KickClient(username=channel_username, session=session)

        try:
            # --- On-Demand Polling ---
            # Imagine your worker's event loop triggers this every 15 seconds.
            print("--- First Poll ---")
            await check_for_new_kick_messages(client, chat_states)
            
            print("\n...worker is doing other things for 15 seconds...")
            await asyncio.sleep(15)

            print("\n--- Second Poll ---")
            await check_for_new_kick_messages(client, chat_states)
            
        except KickApiError as e:
            print(f"🚨 A critical API error occurred: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting.")