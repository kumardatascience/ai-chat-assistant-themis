# Use the official Python 3.12 slim image as our base
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first (Docker layer caching: deps only re-install if requirements change)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY src/ ./src/
COPY data/ ./data/
COPY pytest.ini .

# Chainlit needs to know where to find chainlit.md 
WORKDIR /app/src/app

# Expose the port Chainlit runs on
EXPOSE 8000

# Run Chainlit when the container starts
# --host 0.0.0.0 makes it accessible from outside the container
CMD ["chainlit", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]