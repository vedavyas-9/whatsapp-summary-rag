import boto3
import json
from typing import List
from botocore.exceptions import ClientError

def get_embedding(text: str) -> List[float]:
    """
    Generates embeddings for text input using Amazon Titan.

    Parameters:
    - text (str): Input text to embed (e.g., WhatsApp message content).

    Returns:
    - List[float]: Embedding vector.
    """
    try:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string")

        # Debug: Confirm text type and preview
        print("TEXT TYPE:", type(text))
        print("TEXT PREVIEW:", text[:100])

        bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
        payload = {"inputText": text}
        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v1",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload)
        )
        result = json.loads(response["body"].read())
        return result["embedding"]

    except ClientError as e:
        print(f"❌ Bedrock error generating embedding: {e.response['Error']['Message']}")
        return []
    except Exception as e:
        print(f"❌ Error generating embedding: {type(e).__name__} - {e}")
        return []