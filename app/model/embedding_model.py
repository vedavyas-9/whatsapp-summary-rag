# app/embedding/titan_embedder.py
import boto3
import json

def get_embedding(text: str) -> list[float]:
    bedrock = boto3.client("bedrock-runtime")

    # Debug: Confirm text type before API call
    print("TEXT TYPE:", type(text))         # should be <class 'str'>
    print("TEXT PREVIEW:", text[:100])      # first 100 chars

    payload = {
        "inputText": text  # Titan requires a plain string
    }

    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v1",
        contentType="application/json",
        accept="application/json",
        body=json.dumps(payload)
    )

    result = json.loads(response["body"].read())
    return result["embedding"]
