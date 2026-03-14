FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY prompts/ ./prompts/

WORKDIR /app/backend

# Cloud Run sets PORT env var
ENV PORT=8080

EXPOSE 8080

CMD ["python", "main.py"]
