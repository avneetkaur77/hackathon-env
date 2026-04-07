FROM python:3.10-slim-bullseye

WORKDIR /app

# Prevent python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system deps (important for some builds)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Upgrade pip first (important)
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r server/requirements.txt

# Expose port (HF uses 7860)
EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]