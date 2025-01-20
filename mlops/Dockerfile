# Use a lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy application files
COPY main.py requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose logs folder
VOLUME /app/logs

# Default command
CMD ["python", "main.py"]
