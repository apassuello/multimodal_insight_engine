# Multi-stage Dockerfile for Constitutional AI Interactive Demo
# Optimized for production deployment with small image size

# ============================================================================
# Stage 1: Base Python Environment
# ============================================================================
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# Stage 2: Dependencies Builder
# ============================================================================
FROM base as builder

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY requirements.txt demo/requirements.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -r demo/requirements.txt

# ============================================================================
# Stage 3: Application
# ============================================================================
FROM base as application

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY demo/ ./demo/
COPY DEMO_ARCHITECTURE.md ./
COPY README.md ./

# Create necessary directories
RUN mkdir -p demo/checkpoints demo/logs demo/exports && \
    chmod -R 777 demo/checkpoints demo/logs demo/exports

# Expose Gradio port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the application (configuration via environment variables)
CMD ["python", "-m", "demo.main"]
