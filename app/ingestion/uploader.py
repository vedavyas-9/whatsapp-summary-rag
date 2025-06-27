import boto3
from uuid import uuid4

s3 = boto3.client("s3")
BUCKET = "ap-hackathon-us4"

def upload_file_to_s3(file, folder="uploads"):
    key = f"{folder}/{uuid4()}_{file.name}"
    s3.upload_fileobj(file, BUCKET, key)
    return f"s3://{BUCKET}/{key}"
