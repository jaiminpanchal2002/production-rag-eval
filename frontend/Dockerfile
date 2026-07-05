# Use Python runtime as parent image
FROM python:3.10-slim

# Set environmental variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY app.py .

# We need libraries for streamlit app, let's copy dependencies or run them
RUN pip install --no-cache-dir streamlit langchain langchain-community langchain-chroma langchain-ollama pypdf sentence-transformers

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
