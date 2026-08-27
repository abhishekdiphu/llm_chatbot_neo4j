FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing .pyc files and buffering logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file first for better Docker caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Streamlit default port
EXPOSE 8501

# Streamlit Docker-friendly settings
CMD ["streamlit", "run", "chatbot.py", "--server.address=0.0.0.0", "--server.port=8501"]
