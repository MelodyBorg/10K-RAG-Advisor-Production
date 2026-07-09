FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl zstd && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files, including start.sh
COPY . .

# Make the script executable
RUN chmod +x start.sh

EXPOSE 8501

# Run the script instead of a complex command line
CMD ["./start.sh"]
