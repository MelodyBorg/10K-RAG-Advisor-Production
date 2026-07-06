#!/bin/sh

# Start Ollama in the background
ollama serve &

# Wait for Ollama to start (adjust sleep if needed)
echo "Waiting for Ollama to start..."
sleep 10

# Pull your models
echo "Pulling models..."
ollama pull qwen3:8b
ollama pull mistral

# Start your Streamlit app in the foreground
echo "Starting Streamlit..."
streamlit run app.py
