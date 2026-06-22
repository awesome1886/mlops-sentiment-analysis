# TODO (Task 4)
# Write a Dockerfile for the FastAPI service using python:3.12-slim as the base image.
# Documentation: https://docs.docker.com/reference/dockerfile/
# Use python:3.12-slim as the base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /workspace

# Install project Python dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install -y build-essential
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app directory into the image
COPY app/ app/

# Expose port 8000
EXPOSE 8000

# Define a CMD that starts the FastAPI service
CMD ["python", "app/main.py"]