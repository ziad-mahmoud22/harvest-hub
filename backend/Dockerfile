# Use Python 3.9 slim image for better performance
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Upgrade pip and install wheel for better package compilation
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies with verbose output for debugging
RUN pip install --no-cache-dir -r requirements.txt --verbose

# Copy the entire backend directory
COPY . .

# Create necessary directories
RUN mkdir -p uploads static

# Expose port 5000
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py

# Run the application
CMD ["python", "app.py"]