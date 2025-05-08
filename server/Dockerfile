# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED True
ENV APP_HOME /app
ENV PORT 8000

# Create and set the working directory
WORKDIR $APP_HOME

# Install system dependencies if any (e.g., for libraries that need compilation)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Install Poetry (or pip if you prefer)
# RUN pip install poetry
# COPY poetry.lock pyproject.toml ./
# RUN poetry config virtualenvs.create false && poetry install --no-dev --no-interaction --no-ansi

RUN python -m venv venv
RUN . venv/bin/activate

# Using pip and requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY ./src $APP_HOME/src

# Command to run the application using Gunicorn
# Gunicorn is recommended for production deployment of FastAPI apps
# Cloud Run automatically provides the PORT environment variable
CMD exec gunicorn --bind :$PORT --workers 1 --timeout 120 --worker-class uvicorn.workers.UvicornWorker --threads 8 src.mcp_server:app 