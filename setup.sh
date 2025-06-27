#!/bin/bash
sudo apt update -y && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git unzip

# Python virtual env
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install streamlit boto3 pymongo pypdf python-docx pandas chromadb \
    neo4j langchain openai requests pdfplumber \
    amazon-textract amazon-textract-response-parser \
    botocore langchain-aws bedrock-python-sdk

# Clone your repo or create a new directory
mkdir -p ~/hackathon && cd ~/hackathon

echo "✅ Environment setup complete!"
