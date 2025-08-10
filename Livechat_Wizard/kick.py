# kick.py
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple, List

# We will use this model when we integrate with LiveChatController
# but the client itself doesn't need to know about it.
# from data_models import UnifiedMessage

# Use the async-native version of curl_cffi
from curl_cffi.requests import AsyncSession, RequestsError

# The parent application is responsible for configuring the root logger.
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class KickChatUser:
    """Represents a user in a Kick chat. (Internal to this module)"""
    username: str
    color: Tuple[int, int, int]

@dataclass(frozen=True)
class KickChatMessage:
    """Represents a single, structured message from Kick. (Internal to this module)"""
    id: str
    user: KickChatUser
    content: str
    timestamp: datetime

def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Converts a hex color string to an (R, G, B) tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (255, 255, 255)  # Default to white for invalid formats
    try:
        return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    except ValueError:
        return (255, 255, 255) # Default to white on parsing error

class KickApiError(Exception):
    """Custom exception for Kick API-related errors."""
    pass

class KickClient:
    """
    An async, on-demand client to fetch live chat messages from a Kick.com channel.
    This client is stateless and designed to be called by an orchestrator.
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
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        }

    async def _get_channel_id(self) -> str:
        """Fetches and caches the internal channel ID for the given username."""
        if self._channel_id:
            return self._channel_id

        url = f"{self.BASE_URL}/channels/{self.username}"
        try:
            # Use impersonate to mimic a real browser's TLS fingerprint
            res = await self._session.get(url, headers=self._headers, impersonate="chrome110")
            res.raise_for_status()
            data = res.json()
            
            channel_id = data.get("id")
            if not channel_id:
                raise KickApiError(f"Could not find 'id' in channel data for '{self.username}'. Response: {data}")
            
            self._channel_id = str(channel_id)
            logger.info(f"Successfully fetched Kick channel ID for '{self.username}': {self._channel_id}")
            return self._channel_id
        except RequestsError as e:
            logger.error(f"Network error while getting channel ID for '{self.username}': {e}")
            raise KickApiError(f"Could not retrieve channel ID for '{self.username}'.") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred while getting channel ID for '{self.username}': {e}")
            raise KickApiError(f"Could not retrieve channel ID for '{self.username}'.") from e

    async def fetch_new_messages(self, last_seen_timestamp: datetime) -> List[KickChatMessage]:
        """
        Fetches the latest batch of messages and filters for new ones.

        Args:
            last_seen_timestamp: The timestamp of the last message processed.
                                 The function will return messages newer than this.

        Returns:
            A list of new KickChatMessage objects, sorted chronologically.
        """
        try:
            channel_id = await self._get_channel_id()
            url = f"{self.BASE_URL}/channels/{channel_id}/messages"
            
            res = await self._session.get(url, headers=self._headers, impersonate="chrome110")
            res.raise_for_status()
            
            data = res.json().get("data", {})
            raw_messages = data.get("messages", [])
            if not raw_messages:
                return []
            
            parsed_messages = [
                KickChatMessage(
                    id=msg["id"],
                    user=KickChatUser(
                        username=msg["sender"]["username"],
                        color=_hex_to_rgb(msg["sender"]["identity"]["color"]),
                    ),
                    content=msg["content"],
                    timestamp=datetime.fromisoformat(msg["created_at"].replace("Z", "+00:00")),
                )
                for msg in raw_messages
            ]
            
            new_messages = sorted(
                [msg for msg in parsed_messages if msg.timestamp > last_seen_timestamp],
                key=lambda m: m.timestamp
            )
            
            return new_messages
        
        except KickApiError:
            # This was already logged in _get_channel_id, re-raise to notify the controller
            raise
        except RequestsError as e:
            logger.error(f"Network error fetching messages for '{self.username}': {e}")
            # Return an empty list to make the caller resilient to transient API errors.
            return []
        except Exception as e:
            logger.error(f"Failed to parse messages for channel '{self.username}': {e}")
            return []

### The section below is for standalone testing and can be commented out later.
### It will not be used when integrated into LiveChatController.

async def _test_kick_client():
    """
    An example of a single, on-demand check for new messages.
    This function would be called periodically by your worker's scheduler.
    """
    # This is an example of how to use the client.
    # In our final app, the LiveChatController will manage the session and state.
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    channel_username = os.getenv("KI_CHANNEL")
    if not channel_username:
        print("KI_CHANNEL environment variable not set. Skipping test.")
        return

    # State is managed by the caller. For this test, it's a simple dict.
    # Initialize with the current time to only fetch messages from this point forward.
    test_state = {"last_timestamp": datetime.now(timezone.utc)}
    
    print(f"\nChecking for new messages for '{channel_username}' since {test_state['last_timestamp'].isoformat()}")
    
    async with AsyncSession() as session:
        client = KickClient(username=channel_username, session=session)
        
        try:
            # Fetch only new messages using the client
            new_messages = await client.fetch_new_messages(last_seen_timestamp=test_state["last_timestamp"])

            if not new_messages:
                print("No new messages found.")
                return

            for msg in new_messages:
                time_str = msg.timestamp.strftime("%H:%M:%S")
                print(f"  -> [{time_str}] {msg.user.username}: {msg.content}")

            # The caller updates its state with the newest timestamp for the next call
            test_state["last_timestamp"] = new_messages[-1].timestamp
            print(f"State updated. New last_seen_timestamp is {test_state['last_timestamp'].isoformat()}")

        except KickApiError as e:
            print(f"🚨 A critical API error occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during testing: {e}")


if __name__ == "__main__":
    print("--- Running KickClient Standalone Test ---")
    try:
        asyncio.run(_test_kick_client())
    except KeyboardInterrupt:
        print("\nExiting test.")