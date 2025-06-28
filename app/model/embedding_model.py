# app/embedding/titan_embedder.py
import boto3
import json
def get_embedding(text: str) -> list[float]:
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")  # Change region if needed

    payload = {
        "inputText": text
    }

    try:
        response = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v1",  # Use the correct model version
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload)
        )
        result = json.loads(response["body"].read())
        return result["embedding"]
    except Exception as e:
        print("❌ ERROR during Titan embedding:", e)
        raise
