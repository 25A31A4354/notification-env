FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]