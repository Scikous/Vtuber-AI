# models.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

@dataclass(frozen=True)
class UnifiedMessage:
    """
    A standardized, immutable representation of a chat message from any platform.

    This structure decouples the main application logic from the specific
    data formats of different chat service APIs.

    Attributes:
        platform (str): The source platform, e.g., 'Twitch', 'YouTube', 'Kick'.
        username (str): The display name of the user who sent the message.
        content (str): The actual text content of the message.
        timestamp (datetime): A timezone-aware datetime object indicating when
                              the message was created.
        color (Optional[Tuple[int, int, int]]): An optional (R, G, B) tuple
                                                 representing the user's chat color.
    """
    platform: str
    username: str
    content: str
    timestamp: datetime
    color: Optional[Tuple[int, int, int]] = None