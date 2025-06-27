import re
import json
from base64 import b64decode
from uuid import uuid4
from typing import List, Dict
from pymongo import MongoClient
from dotenv import load_dotenv
import os
load_dotenv()
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["hackathon"]
files_col = db["files"]
messages_col = db["messages"]


def load_members_info(file_id: str = None) -> Dict[str, Dict]:
    """
    Loads members_info.json from MongoDB.
    """
    try:
        if not file_id:
            file_doc = files_col.find_one({"file_name": "members_info.json"})
            if not file_doc:
                print("⚠️ Members file not found in MongoDB. Please ensure 'members_info.json' is uploaded.")
                return {}
            file_id = file_doc["_id"]
        
        file_doc = files_col.find_one({"_id": file_id})
        if not file_doc:
            print(f"⚠️ File with ID {file_id} not found in MongoDB.")
            return {}
        
        content = b64decode(file_doc["content"]).decode("utf-8")
        data = json.loads(content)
        return {item["phone_number"]: item for item in data.get("police_department_personnel", [])}
    except Exception as e:
        print(f"❌ Error loading members info: {type(e).__name__} - {e}")
        return {}

class WhatsAppChatExtractor:
    def __init__(self):
        # Updated regex to handle various WhatsApp formats (e.g., dd/mm/yy or mm/dd/yy, with/without seconds, AM/PM)
        self.pattern = r'\[?(\d{1,2}/\d{1,2}/\d{2,4},? \d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\]?)\s*(?:~ )?([+\d\s]+?|[^:]+?)(?::\s*|\s*-\s*)(.+?)(?=(?:\n\[|\Z|\n\d{1,2}/\d{1,2}/\d{2,4}))'
        self.members_info = load_members_info()
    
    def extract_messages(self, file_id: str) -> List[Dict]:
        """
        Parses WhatsApp chat log from MongoDB and stores messages.

        Parameters:
        - file_id (str): MongoDB file ID.

        Returns:
        - List[Dict]: List of message dictionaries.
        """
        try:
            file_doc = files_col.find_one({"_id": file_id})
            if not file_doc:
                raise ValueError(f"File not found: {file_id}")

            content = b64decode(file_doc["content"]).decode("utf-8")
            group_id = file_doc["group_id"]
            
            # Debug: Print first 500 characters of content
            print("DEBUG: Chat content preview:", content[:500])
            
            matches = re.findall(self.pattern, content, re.DOTALL | re.MULTILINE)
            if not matches:
                print("⚠️ No messages matched with regex. Check WhatsApp chat format in file.")
            
            messages = []
            for match in matches:
                timestamp, sender, message = match
                # Debug: Print each matched message
                print(f"DEBUG: Matched - Timestamp: {timestamp}, Sender: {sender}, Message: {message[:50]}")
                
                emojis = [char for char in message if char in "🚨🕵️‍♂️⏱️💪🎥🚗🧐📄🚔✍️🚪🕸️🎯🔍🕐📊🖼️🏃‍♂️🙏📹🚀🔬📂🌃🤔🤫💡👣🤯💻👏✨👍📩🔗🔒🚓🗺️📢⬆️🔑👮‍♂️📞🌟"]
                
                sender_info = self.members_info.get(sender.strip(), {})
                sender_name = sender_info.get("name", "Unknown")
                sender_role = sender_info.get("role", "Unknown")
                
                message_id = str(uuid4())
                message_doc = {
                    "timestamp": timestamp.strip(),
                    "phone_number": sender.strip(),
                    "message": message.strip(),
                    "emojis": emojis,
                    "sender_name": sender_name,
                    "sender_role": sender_role,
                    "group_id": group_id,
                    "file_id": file_id
                }
                
                messages_col.insert_one({"_id": message_id, **message_doc})
                messages.append({"_id": message_id, **message_doc})
            
            print(f"DEBUG: Extracted {len(messages)} messages from file_id: {file_id}")
            return messages
        
        except Exception as e:
            print(f"❌ Error extracting messages: {type(e).__name__} - {e}")
            return []