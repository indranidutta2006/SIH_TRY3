# ==============================================================================
# SIH26138 - Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization
# Multi-stage / Lean Dockerfile for containerized deployment fallback on Render
# ==============================================================================

FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8501

WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Install Python runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy repository source code and assets
COPY . .

# Ensure build script is executable and run artifact generation pipeline
RUN chmod +x scripts/build_for_deploy.sh && \
    bash scripts/build_for_deploy.sh

# Expose Streamlit default port
EXPOSE 8501

# Run executive Streamlit dashboard
CMD ["sh", "-c", "streamlit run app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false"]
