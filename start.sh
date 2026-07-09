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

# ... (Keep your existing Ollama start and pull commands above this)

echo "Starting Streamlit..."
streamlit run app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
