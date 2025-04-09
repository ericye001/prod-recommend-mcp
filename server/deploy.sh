#!/bin/bash

# Install AWS CLI if not already installed
if ! command -v aws &> /dev/null; then
    echo "Installing AWS CLI..."
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
fi

# Configure AWS credentials
aws configure

# Install EB CLI
pip install awsebcli

# Initialize EB application
eb init -p python-3.9 gofanco-mcp-server

# Create EB environment
eb create gofanco-mcp-env

# Deploy application
eb deploy 