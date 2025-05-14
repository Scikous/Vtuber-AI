"""
File Operations Utility for Vtuber-AI
Provides functions for reading and writing files, e.g., CSV logs.
"""
import csv
from collections import deque
import os

async def write_messages_csv(file_path, message_data):
    """
    Intended for writing tuples of chat message_data -> ('<user name>: <message>', '<LLM output>')
    Can technically be any format.
    Appends to the CSV file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode='a+', newline='\n', encoding='utf-8') as file:
        csv_writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        csv_writer.writerow(message_data)

async def read_messages_csv(file_path, num_messages=10):
    """
    Intended for returning tuples of chat messages -> (<user name>, <message>)
    Can technically be any format.
    By default returns the latest num_messages chat messages as a list of tuples.
    Returns an empty list if the file doesn't exist.
    """
    if not os.path.exists(file_path):
        return []
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        messages = deque(csv_reader, maxlen=num_messages)
    return [tuple(row) for row in messages]