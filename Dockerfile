FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port for Gradio / FastAPI
EXPOSE 7860

# Start the FastAPI server (OpenEnv grader calls /reset and /step)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]