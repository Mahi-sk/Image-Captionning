# Use a lightweight, official Python runtime image optimized for deep learning deployments
FROM python:3.9-slim

# Set system environment path variables so Python modules can find each other inside the container
ENV PYTHONPATH=/app/Image-Captionning
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container's storage system
WORKDIR /app

# Install minimal OS dependencies required for processing images with Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements.txt first to take advantage of Docker's caching layers
COPY Image-Captionning/requirements.txt ./Image-Captionning/

# Upgrade pip and install the production-grade deep learning and server dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r Image-Captionning/requirements.txt

# Copy all your modular source code, app logic, and dummy weight structures into the image space
COPY Image-Captionning/ ./Image-Captionning/

# Expose port 8000 so web routers can cleanly pass API traffic to Uvicorn
EXPOSE 8000

# Tell Docker to execute your web engine server when launching on a cloud node
CMD ["python", "-m", "uvicorn", "Image-Captionning.app.main:app", "--host", "0.0.0.0", "--port", "8000"]