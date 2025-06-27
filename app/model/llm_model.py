import boto3
import json
from pymongo import MongoClient
from datetime import datetime
from typing import Optional
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import os

load_dotenv()
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["hackathon"]
results_col = db["results"]

def run_claude_task(prompt: str, task_type: str = "summarization", file_id: Optional[str] = None, message_id: Optional[str] = None) -> str:
    """
    Runs a Claude task and stores results in MongoDB.

    Parameters:
    - prompt (str): Input prompt for Claude.
    - task_type (str): Type of task ('summarization', 'task_extraction', 'warning_detection').
    - file_id (str, optional): MongoDB file ID.
    - message_id (str, optional): MongoDB message ID.

    Returns:
    - str: Generated text output.
    """
    try:
        client = boto3.client("bedrock-runtime", region_name="us-east-1")
        model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

        native_request = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "temperature": 0.5,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
        }

        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(native_request),
            contentType="application/json",
            accept="application/json"
        )

        model_response = json.loads(response["body"].read())
        output = model_response["content"][0]["text"]

        # Store result in MongoDB
        if file_id or message_id:
            result_doc = {
                "file_id": file_id,
                "message_id": message_id,
                "task_type": task_type,
                "output": output,
                "created_at": datetime.now().isoformat()
            }
            results_col.insert_one(result_doc)

        return output

    except ClientError as e:
        print(f"❌ Bedrock error running Claude task: {e.response['Error']['Message']}")
        return ""
    except Exception as e:
        print(f"❌ Error running Claude task: {type(e).__name__} - {e}")
        return ""