# Use a slim Python 3.12 image
FROM python:3.12-slim

# Install system dependencies (curl and zstd for Ollama)
RUN apt-get update && apt-get install -y curl zstd && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else
COPY . .

# Ollama must run in background, then we pull models, then run app
# We pull both models during build so they are ready at runtime
RUN ollama serve & sleep 5 && ollama pull qwen3:8b && ollama pull mistral && pkill ollama

EXPOSE 8501

# Replace the old CMD with this
# We use 'exec' to ensure the app runs as the main process
CMD ["sh", "-c", "ollama serve & sleep 10 && ollama pull qwen3:8b && ollama pull mistral && streamlit run app.py"]
