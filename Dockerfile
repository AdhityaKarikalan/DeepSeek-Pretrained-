# Use an official lightweight Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (needed for many Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies (crucial for caching and path fixes)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your pre-trained model file
COPY your_model_file.pth /app/your_model_file.pth

# Copy the rest of your code (including handler.py)
COPY . .

# Set the command to run your handler
CMD ["python", "handler.py"]
