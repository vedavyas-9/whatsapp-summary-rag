import boto3
import json

def run_claude_task(prompt: str) -> str:
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
    return model_response["content"][0]["text"]
